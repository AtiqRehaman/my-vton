"""
warping/train.py — TPS warping module training loop

Training config (from spec):
  Optimizer  : Adam, lr=1e-4
  Epochs     : 50 (500-pair subset validation) → 200 (full dataset)
  Batch size : 8 (fits T4 at 512×384)
  Loss       : L1 + 0.1×VGG_perceptual + 0.01×second_order_grid
  LR schedule: StepLR, decay by 0.5 every 20 epochs
  Checkpoint : save best (lowest val loss) + every 10 epochs

VRAM budget on T4 (15 GB):
  Model (B=8): ~3.2 GB
  VGG perceptual loss: ~0.4 GB
  Optimizer states:    ~0.8 GB
  Total:               ~4.4 GB  ← well within T4 limits

For RTX 2050 (4 GB):
  Reduce batch_size to 2, use gradient_checkpointing=True in FeatureExtractor
"""

import os
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR

import numpy as np
import cv2
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────
# Default training configuration
# ─────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Data
    'manifest':    'preprocessed_dataset/train_manifest.json',
    'output_dir':  'warping/checkpoints',
    'cloth_type':  'upper',
    'target_h':    512,
    'target_w':    384,

    # Training
    'batch_size':  8,
    'num_epochs':  50,
    'lr':          1e-4,
    'val_split':   0.1,          # 10% of data for validation
    'num_workers': 4,
    'seed':        42,

    # Loss weights
    'w_perceptual': 0.1,
    'w_smooth':     0.01,

    # Schedule
    'lr_step_size': 20,          # StepLR: decay every N epochs
    'lr_gamma':     0.5,         # multiply LR by this factor

    # TPS
    'n_control_points': 25,

    # Checkpointing
    'save_every': 10,            # save checkpoint every N epochs
    'vis_every':  200,           # save visualization every N steps
    'vis_n':      4,             # how many samples to visualize

    # Hardware
    'fp16':        True,         # use fp16 on T4
    'device':      'auto',
}


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────
# Visualization helper
# ─────────────────────────────────────────────────────────────────

def save_warp_visualization(
    cloth:         torch.Tensor,   # (N, 3, H, W)
    warped_cloth:  torch.Tensor,   # (N, 3, H, W)
    gt_worn:       torch.Tensor,   # (N, 3, H, W)
    agnostic:      torch.Tensor,   # (N, 3, H, W)
    save_path:     str,
    n_samples:     int = 4,
):
    """
    Save side-by-side visualization: cloth | warped | gt_worn | agnostic
    All tensors in [-1, 1].
    """
    def to_np(t):
        # [-1,1] → [0,255] uint8 RGB → BGR for cv2
        t = t.detach().cpu().float().clamp(-1, 1)
        t = ((t + 1) / 2 * 255).byte()
        imgs = []
        for i in range(min(n_samples, t.shape[0])):
            img = t[i].permute(1, 2, 0).numpy()  # CHW → HWC RGB
            imgs.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return imgs

    cloths    = to_np(cloth)
    warpeds   = to_np(warped_cloth)
    gts       = to_np(gt_worn)
    agnostics = to_np(agnostic)

    rows = []
    for c, w, g, a in zip(cloths, warpeds, gts, agnostics):
        # Add column dividers (2px white line)
        divider = np.ones((c.shape[0], 2, 3), dtype=np.uint8) * 200
        row = np.concatenate([c, divider, w, divider, g, divider, a], axis=1)
        rows.append(row)

    # Stack all rows vertically
    header_h = 30
    grid = np.concatenate(rows, axis=0)
    canvas = np.ones((grid.shape[0] + header_h, grid.shape[1], 3), dtype=np.uint8) * 240
    canvas[header_h:] = grid

    # Add column labels
    col_w = cloths[0].shape[1]
    labels = ['Cloth (input)', 'Warped cloth', 'GT worn region', 'Agnostic body']
    for i, label in enumerate(labels):
        x = i * (col_w + 2) + col_w // 4
        cv2.putText(canvas, label, (x, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (40, 40, 40), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    cv2.imwrite(save_path, canvas)


# ─────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────

class WarpingTrainer:

    def __init__(self, config: dict):
        self.cfg = {**DEFAULT_CONFIG, **config}
        set_seed(self.cfg['seed'])

        # Device
        if self.cfg['device'] == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.cfg['device'])
        print(f"Device: {self.device}")

        # FP16 scaler (T4 doesn't support bf16)
        self.use_fp16 = self.cfg['fp16'] and self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_fp16)
        print(f"FP16: {self.use_fp16}")

        # Output dirs
        self.out_dir = Path(self.cfg['output_dir'])
        self.vis_dir = self.out_dir / 'vis'
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        self._build_datasets()
        self._build_model()
        self._build_optimizer()

        self.best_val_loss = float('inf')
        self.global_step = 0
        self.history = {'train': [], 'val': []}

    # ── Setup ──────────────────────────────────────────────────────

    def _build_datasets(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from warping.dataset import WarpingDataset

        full_dataset = WarpingDataset(
            manifest_path = self.cfg['manifest'],
            target_h      = self.cfg['target_h'],
            target_w      = self.cfg['target_w'],
            cloth_type    = self.cfg['cloth_type'],
            augment       = True,
        )

        # Train / val split
        n_val   = max(1, int(len(full_dataset) * self.cfg['val_split']))
        n_train = len(full_dataset) - n_val
        train_ds, val_ds = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(self.cfg['seed'])
        )
        # Disable augmentation for val
        val_ds.dataset.augment = False

        self.train_loader = DataLoader(
            train_ds,
            batch_size  = self.cfg['batch_size'],
            shuffle     = True,
            num_workers = self.cfg['num_workers'],
            pin_memory  = True,
            drop_last   = True,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size  = self.cfg['batch_size'],
            shuffle     = False,
            num_workers = self.cfg['num_workers'],
            pin_memory  = True,
        )
        print(f"Train: {n_train} | Val: {n_val} | "
              f"Steps/epoch: {len(self.train_loader)}")

    def _build_model(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from warping.model  import TPSWarpingNet
        from warping.losses import WarpingLoss

        self.model = TPSWarpingNet(
            height           = self.cfg['target_h'],
            width            = self.cfg['target_w'],
            n_control_points = self.cfg['n_control_points'],
        ).to(self.device)

        self.criterion = WarpingLoss(
            w_perceptual = self.cfg['w_perceptual'],
            w_smooth     = self.cfg['w_smooth'],
        ).to(self.device)

        from warping.model import count_parameters
        print(count_parameters(self.model))

    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = self.cfg['lr'],
        )
        self.scheduler = StepLR(
            self.optimizer,
            step_size = self.cfg['lr_step_size'],
            gamma     = self.cfg['lr_gamma'],
        )

    # ── Train / Val steps ─────────────────────────────────────────

    def _run_batch(self, batch: dict, train: bool) -> dict:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from warping.losses import extract_gt_worn_region

        cloth     = batch['cloth'].to(self.device)
        agnostic  = batch['agnostic'].to(self.device)
        pose      = batch['pose'].to(self.device)
        densepose = batch['densepose'].to(self.device)
        worn      = batch['worn_person'].to(self.device)
        cloth_mask = batch['cloth_mask'].to(self.device)
        parse_map  = batch['parse_map'].to(self.device)

        # Extract ground truth worn region
        gt_worn, _ = extract_gt_worn_region(worn, parse_map, self.cfg['cloth_type'])

        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            warped_cloth, grid, offsets = self.model(cloth, agnostic, pose, densepose)
            losses = self.criterion(warped_cloth, gt_worn, grid, cloth_mask)

        if train:
            self.optimizer.zero_grad()
            self.scaler.scale(losses['total']).backward()
            # Gradient clipping — prevents exploding gradients in correlation layer
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return {k: v.item() for k, v in losses.items()}, warped_cloth, gt_worn

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        totals = {'total': 0, 'l1': 0, 'perceptual': 0, 'smooth': 0}
        n = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:3d} [train]", leave=False)
        for batch in pbar:
            losses, warped, gt_worn = self._run_batch(batch, train=True)
            self.global_step += 1

            for k in totals:
                totals[k] += losses[k]
            n += 1

            pbar.set_postfix({
                'loss': f"{losses['total']:.4f}",
                'l1':   f"{losses['l1']:.4f}",
                'lr':   f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

            # Save visualization every N steps
            if self.global_step % self.cfg['vis_every'] == 0:
                save_warp_visualization(
                    batch['cloth'][:self.cfg['vis_n']],
                    warped[:self.cfg['vis_n']],
                    gt_worn[:self.cfg['vis_n']],
                    batch['agnostic'][:self.cfg['vis_n']],
                    save_path=str(self.vis_dir / f'step_{self.global_step:06d}.jpg'),
                    n_samples=self.cfg['vis_n'],
                )

        return {k: v / n for k, v in totals.items()}

    def _val_epoch(self, epoch: int) -> dict:
        self.model.eval()
        totals = {'total': 0, 'l1': 0, 'perceptual': 0, 'smooth': 0}
        n = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {epoch:3d} [val]  ", leave=False):
                losses, _, _ = self._run_batch(batch, train=False)
                for k in totals:
                    totals[k] += losses[k]
                n += 1

        return {k: v / n for k, v in totals.items()}

    # ── Checkpointing ─────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, tag: str = ''):
        ckpt = {
            'epoch':       epoch,
            'global_step': self.global_step,
            'model':       self.model.state_dict(),
            'optimizer':   self.optimizer.state_dict(),
            'scheduler':   self.scheduler.state_dict(),
            'scaler':      self.scaler.state_dict(),
            'config':      self.cfg,
            'best_val':    self.best_val_loss,
        }
        fname = f'epoch_{epoch:03d}{tag}.pth'
        torch.save(ckpt, self.out_dir / fname)
        print(f"  Saved checkpoint: {fname}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.scaler.load_state_dict(ckpt['scaler'])
        self.best_val_loss = ckpt.get('best_val', float('inf'))
        self.global_step = ckpt.get('global_step', 0)
        print(f"Resumed from {path} | epoch={ckpt['epoch']} | "
              f"best_val={self.best_val_loss:.4f}")
        return ckpt['epoch']

    # ── Main run ───────────────────────────────────────────────────

    def run(self, resume_from: str = None):
        start_epoch = 1
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1

        print(f"\n{'='*60}")
        print(f"Starting warping training | {self.cfg['num_epochs']} epochs")
        print(f"{'='*60}\n")

        for epoch in range(start_epoch, self.cfg['num_epochs'] + 1):
            t0 = time.time()

            train_losses = self._train_epoch(epoch)
            val_losses   = self._val_epoch(epoch)
            self.scheduler.step()

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]['lr']

            # Log
            print(
                f"Epoch {epoch:3d}/{self.cfg['num_epochs']} | "
                f"train_loss={train_losses['total']:.4f} "
                f"(l1={train_losses['l1']:.4f} "
                f"perc={train_losses['perceptual']:.4f} "
                f"smooth={train_losses['smooth']:.5f}) | "
                f"val_loss={val_losses['total']:.4f} | "
                f"lr={lr:.2e} | {elapsed:.0f}s"
            )

            self.history['train'].append(train_losses)
            self.history['val'].append(val_losses)

            # Save best
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint(epoch, tag='_best')
                print(f"  ★ New best val loss: {self.best_val_loss:.4f}")

            # Periodic save
            if epoch % self.cfg['save_every'] == 0:
                self.save_checkpoint(epoch)

            # Save loss history
            with open(self.out_dir / 'history.json', 'w') as f:
                json.dump(self.history, f, indent=2)

        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")
        print(f"Best checkpoint: {self.out_dir}/*_best.pth")

        # Final checkpoint
        self.save_checkpoint(self.cfg['num_epochs'], tag='_final')
        return self.history


# ─────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TPS Warping Module')
    parser.add_argument('--manifest',    default='preprocessed_dataset/train_manifest.json')
    parser.add_argument('--output_dir',  default='warping/checkpoints')
    parser.add_argument('--cloth_type',  default='upper', choices=['upper','lower','overall'])
    parser.add_argument('--batch_size',  type=int,   default=8)
    parser.add_argument('--num_epochs',  type=int,   default=50)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--fp16',        action='store_true', default=True)
    parser.add_argument('--resume',      default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    config = vars(args)
    trainer = WarpingTrainer(config)
    trainer.run(resume_from=config.pop('resume', None))