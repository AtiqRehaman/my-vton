"""
generation/train.py — Diffusion UNet training loop

─────────────────────────────────────────────────────────────────
TRAINING STRATEGY
─────────────────────────────────────────────────────────────────
Loss: standard diffusion noise prediction loss (MSE)
  ε_θ(z_t, t, c) should predict the noise ε added to z_0

  For each training step:
    1. Encode GT person → z_0 (scaled latent)
    2. Sample random timestep t ~ U[0, T]
    3. Sample noise ε ~ N(0, I)
    4. Compute noisy latent: z_t = sqrt(ᾱ_t)*z_0 + sqrt(1-ᾱ_t)*ε
    5. Predict noise: ε_pred = UNet(z_t, t, all_conditioning)
    6. Loss = MSE(ε_pred, ε)

Optimizer: AdamW, lr=1e-5, weight_decay=0.01
  Why 1e-5 (not 1e-4 from warping)?
  The UNet was pretrained — too high LR destroys pretrained weights.
  1e-5 is the standard fine-tuning LR for SD 1.5.

Gradient checkpointing: MANDATORY on T4
  Reduces VRAM from ~13 GB to ~8 GB at cost of ~20% slower step.
  Without it, B=1 at 512×384 barely fits; B=2 causes OOM.

EMA: decay=0.9999, applied after every optimizer step
  Used for inference — gives smoother, more realistic outputs.

Text conditioning: 10% null-text dropout (classifier-free guidance)
  10% of training steps use empty string prompt.
  This enables CFG at inference time (guidance_scale=2.0).

─────────────────────────────────────────────────────────────────
VRAM BUDGET (T4, 15 GB, FP16)
─────────────────────────────────────────────────────────────────
  UNet (B=2, grad ckpt ON):   ~7.5 GB
  VAE (frozen, encode only):  ~1.0 GB
  CLIP (frozen):              ~0.6 GB
  VGG perceptual (eval only): ~0.3 GB
  Optimizer states (AdamW):   ~1.5 GB
  EMA weights:                ~0.7 GB
  Activations + buffers:      ~1.5 GB
  Total:                     ~13.1 GB ← fits T4 at B=2
─────────────────────────────────────────────────────────────────
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm


DEFAULT_CONFIG = {
    # Data
    'manifest':     'preprocessed_dataset/train_manifest.json',
    'output_dir':   'generation/checkpoints',
    'cloth_type':   'upper',
    'target_h':     512,
    'target_w':     384,

    # Base model
    'model_id':     'sd-legacy/stable-diffusion-v1-5',

    # Training
    'batch_size':   2,           # T4 at 512×384 with grad ckpt
    'num_steps':    50_000,      # total optimizer steps
    'lr':           1e-5,
    'weight_decay': 0.01,
    'val_every':    5_000,       # run val every N steps
    'save_every':   5_000,       # save checkpoint every N steps
    'vis_every':    500,         # save sample images every N steps
    'val_split':    0.1,
    'num_workers':  2,           # Colab: keep low
    'seed':         42,

    # Diffusion
    'num_train_timesteps': 1000,
    'null_text_prob':      0.1,  # 10% null text for CFG training

    # Loss
    'loss_type':    'mse',       # 'mse' or 'huber'

    # Hardware
    'fp16':         True,
    'grad_ckpt':    True,        # gradient checkpointing (MANDATORY on T4)
    'xformers':     False,       # set True if xformers is installed
    'ema_decay':    0.9999,
    'grad_clip':    1.0,

    # Inference
    'ddim_steps':   50,
    'guidance_scale': 2.0,
}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────

class MetricsCalculator:
    """
    Computes FID, SSIM, LPIPS on val set.
    Only instantiated during validation (saves VRAM during training).

    FID  (Fréchet Inception Distance): lower is better, target < 20
    SSIM (Structural Similarity):      higher is better, target > 0.75
    LPIPS(Learned Perceptual):         lower is better, target < 0.15
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self._fid   = None
        self._lpips = None

    def _init_models(self):
        if self._fid is None:
            try:
                from torchmetrics.image.fid import FrechetInceptionDistance
                self._fid = FrechetInceptionDistance(normalize=True).to(self.device)
            except ImportError:
                print("torchmetrics not installed — FID will be skipped")

        if self._lpips is None:
            try:
                import lpips
                self._lpips = lpips.LPIPS(net='alex').to(self.device)
                for p in self._lpips.parameters():
                    p.requires_grad = False
            except ImportError:
                print("lpips not installed — LPIPS will be skipped")

    def compute(
        self,
        preds:   torch.Tensor,   # (N, 3, H, W) [-1,1]
        targets: torch.Tensor,   # (N, 3, H, W) [-1,1]
    ) -> dict:
        self._init_models()
        metrics = {}

        # ── SSIM (from torchmetrics or skimage) ──
        try:
            from torchmetrics.functional import structural_similarity_index_measure as ssim
            ssim_val = ssim(
                ((preds + 1) / 2).clamp(0, 1),
                ((targets + 1) / 2).clamp(0, 1),
                data_range=1.0,
            )
            metrics['ssim'] = float(ssim_val)
        except Exception:
            metrics['ssim'] = 0.0

        # ── LPIPS ──
        if self._lpips is not None:
            try:
                with torch.no_grad():
                    lpips_val = self._lpips(
                        preds.to(self.device),
                        targets.to(self.device)
                    ).mean()
                metrics['lpips'] = float(lpips_val)
            except Exception:
                metrics['lpips'] = 1.0

        # ── FID (needs real+fake updates, run once per val set) ──
        if self._fid is not None:
            try:
                imgs_real = ((targets + 1) / 2).clamp(0, 1).to(self.device)
                imgs_fake = ((preds   + 1) / 2).clamp(0, 1).to(self.device)
                self._fid.reset()
                self._fid.update(imgs_real, real=True)
                self._fid.update(imgs_fake, real=False)
                metrics['fid'] = float(self._fid.compute())
            except Exception:
                metrics['fid'] = 999.0

        return metrics


# ─────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────

class GenerationTrainer:

    def __init__(self, config: dict):
        self.cfg = {**DEFAULT_CONFIG, **config}
        set_seed(self.cfg['seed'])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype  = torch.float16 if (self.cfg['fp16'] and self.device.type == 'cuda') \
                      else torch.float32
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == torch.float16))

        self.out_dir = Path(self.cfg['output_dir'])
        self.vis_dir = self.out_dir / 'vis'
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        print(f"Device: {self.device} | dtype: {self.dtype}")

        self._load_models()
        self._build_datasets()
        self._build_optimizer()

        self.global_step  = 0
        self.best_ssim    = 0.0
        self.history      = {'train_loss': [], 'val': []}
        self.metrics_calc = MetricsCalculator(str(self.device))

    # ── Model loading ────────────────────────────────────────────

    def _load_models(self):
        from diffusers import (
            AutoencoderKL, DDIMScheduler, DDPMScheduler
        )
        from transformers import CLIPTextModel, CLIPTokenizer
        from generation.unet_modified import VTONUNet
        from generation.pipeline import LatentEncoder, EMAModel
        from fit.measurement_encoder import MeasurementEncoder
    
        mid = self.cfg['model_id']
        print(f"Loading base model: {mid}")
    
        # ── Frozen components: safe to load in self.dtype (fp16 on GPU) ──
        # These are NEVER touched by the optimizer or GradScaler, so their
        # parameters being fp16 is fine — only forward-pass inference runs
        # through them.
        self.vae = AutoencoderKL.from_pretrained(
            mid, subfolder='vae', torch_dtype=self.dtype
        ).to(self.device)
        for p in self.vae.parameters():
            p.requires_grad = False
    
        self.text_encoder = CLIPTextModel.from_pretrained(
            mid, subfolder='text_encoder', torch_dtype=self.dtype
        ).to(self.device)
        for p in self.text_encoder.parameters():
            p.requires_grad = False
    
        self.tokenizer = CLIPTokenizer.from_pretrained(mid, subfolder='tokenizer')
    
        self.noise_scheduler = DDPMScheduler.from_pretrained(mid, subfolder='scheduler')
        self.ddim_scheduler  = DDIMScheduler.from_pretrained(mid, subfolder='scheduler')
    
        # ── Trainable components: MUST stay fp32 ──
        # GradScaler.unscale_() only operates on fp32 gradients — this is
        # the standard "fp32 master weights + autocast fp16 forward pass"
        # AMP pattern. Loading these in fp16 causes:
        #   ValueError: Attempting to unscale FP16 gradients
        # autocast() (already used in _train_step via
        # torch.cuda.amp.autocast(enabled=self.use_fp16)) handles running
        # the forward pass in fp16 internally — the weights don't need to
        # BE fp16 for activations to get fp16 memory/speed benefits.
        self.unet = VTONUNet.from_pretrained(
            mid, torch_dtype=torch.float32, device=str(self.device)
        )
        if self.cfg['grad_ckpt']:
            self.unet.enable_gradient_checkpointing()
            print("  Gradient checkpointing: ON")
        if self.cfg['xformers']:
            self.unet.enable_xformers_memory_efficient_attention()
    
        self.fit_encoder = MeasurementEncoder(
            target_h=self.cfg['target_h'],
            target_w=self.cfg['target_w'],
        ).to(device=self.device, dtype=torch.float32)
    
        # EMA — built AFTER unet/fit_encoder are fp32, so shadow buffers
        # are fp32 too (matches what GradScaler/optimizer expect)
        self.ema = EMAModel(
            list(self.unet.parameters()) + list(self.fit_encoder.parameters()),
            decay=self.cfg['ema_decay'],
        )
    
        self.lat_enc = LatentEncoder(self.vae)
        print("All models loaded. Trainable params: fp32 | Frozen VAE/CLIP: "
            f"{self.dtype}")

    # ── Dataset & DataLoader ─────────────────────────────────────

    def _build_datasets(self):
        from generation.dataset import GenerationDataset

        full_ds = GenerationDataset(
            manifest_path = self.cfg['manifest'],
            target_h      = self.cfg['target_h'],
            target_w      = self.cfg['target_w'],
            cloth_type    = self.cfg['cloth_type'],
            augment       = True,
        )
        n_val   = max(1, int(len(full_ds) * self.cfg['val_split']))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(self.cfg['seed'])
        )
        val_ds.dataset.augment = False

        self.train_loader = DataLoader(
            train_ds,
            batch_size  = self.cfg['batch_size'],
            shuffle     = True,
            num_workers = self.cfg['num_workers'],
            pin_memory  = (self.device.type == 'cuda'),
            drop_last   = True,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size  = self.cfg['batch_size'],
            shuffle     = False,
            num_workers = self.cfg['num_workers'],
            pin_memory  = (self.device.type == 'cuda'),
        )
        # Steps per epoch — for logging
        self.steps_per_epoch = len(self.train_loader)
        print(f"Train: {n_train} | Val: {n_val} | "
              f"Steps/epoch: {self.steps_per_epoch}")

    # ── Optimizer ────────────────────────────────────────────────

    def _build_optimizer(self):
        trainable = (
            list(self.unet.parameters()) +
            list(self.fit_encoder.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr           = self.cfg['lr'],
            weight_decay = self.cfg['weight_decay'],
        )
        # Cosine annealing — smoothly decays LR to 0 over num_steps
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max  = self.cfg['num_steps'],
            eta_min = self.cfg['lr'] * 0.01,
        )

    # ── Text encoding ────────────────────────────────────────────

    @torch.no_grad()
    def _encode_text(self, prompts: list) -> torch.Tensor:
        """Encode list of prompts to CLIP embeddings. (B, 77, 768)"""
        tokens = self.tokenizer(
            prompts,
            padding    = 'max_length',
            max_length = 77,
            truncation = True,
            return_tensors = 'pt',
        ).input_ids.to(self.device)
        return self.text_encoder(tokens)[0].to(self.dtype)

    # ── Single training step ─────────────────────────────────────

    def _train_step(self, batch: dict) -> float:
        """Run one forward+backward step. Returns scalar loss."""
        B = batch['person_img'].shape[0]

        def to(t):
            return t.to(device=self.device, dtype=self.dtype)

        person_img   = to(batch['person_img'])
        agnostic_img = to(batch['agnostic_img'])
        agnostic_msk = to(batch['agnostic_mask'])
        warped_cloth = to(batch['warped_cloth'])
        cloth_clean  = to(batch['cloth_clean'])
        pose_img     = to(batch['pose_img'])
        densepose    = to(batch['densepose_img'])
        fit_feats    = to(batch['fit_features'])

        with torch.cuda.amp.autocast(enabled=(self.dtype == torch.float16)):

            # ── Encode GT person to latent (training target z_0) ──
            z0 = self.lat_enc.encode(person_img, sample=True)

            # ── Sample random timesteps ──
            t = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=self.device
            ).long()

            # ── Add noise to latent (forward diffusion) ──
            noise = torch.randn_like(z0)
            z_t   = self.noise_scheduler.add_noise(z0, noise, t)

            # ── Prepare all conditioning channels ──
            lH = self.cfg['target_h'] // 8
            lW = self.cfg['target_w'] // 8

            agnostic_lat = self.lat_enc.encode(agnostic_img, sample=False)
            warped_lat   = self.lat_enc.encode(warped_cloth,  sample=False)
            cloth_lat    = self.lat_enc.encode(cloth_clean,   sample=False)

            mask_lat     = F.interpolate(agnostic_msk, (lH, lW), mode='nearest')
            pose_lat     = F.interpolate(pose_img,     (lH, lW), mode='bilinear',
                                          align_corners=False)
            dense_lat    = F.interpolate(densepose,    (lH, lW), mode='nearest')
            fit_emb      = self.fit_encoder(fit_feats)

            # ── Text conditioning with 10% null dropout ──
            prompts = batch['prompt']
            if np.random.random() < self.cfg['null_text_prob']:
                prompts = [''] * B
            text_emb = self._encode_text(prompts)

            # ── UNet forward → predict noise ──
            noise_pred = self.unet(
                noisy_latent          = z_t,
                timestep              = t,
                encoder_hidden_states = text_emb,
                agnostic_latent       = agnostic_lat,
                agnostic_mask         = mask_lat,
                warped_cloth_latent   = warped_lat,
                cloth_latent          = cloth_lat,
                pose_img              = pose_lat,
                densepose_img         = dense_lat,
                fit_embedding         = fit_emb,
            )

            # ── Loss ──
            target = noise  # epsilon-prediction (standard SD training)
            if self.cfg['loss_type'] == 'huber':
                loss = F.huber_loss(noise_pred, target, delta=0.1)
            else:
                loss = F.mse_loss(noise_pred, target)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(self.unet.parameters()) + list(self.fit_encoder.parameters()),
            self.cfg['grad_clip']
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        # EMA update
        self.ema.step(
            list(self.unet.parameters()) + list(self.fit_encoder.parameters())
        )

        return loss.item()

    # ── Validation ───────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self) -> dict:
        """
        Run DDIM inference on val set, compute SSIM/LPIPS/FID.
        Uses EMA weights for inference.
        """
        from generation.pipeline import VTONPipeline

        print(f"\n  Running validation (step {self.global_step})...")

        # Temporarily apply EMA weights
        all_params = list(self.unet.parameters()) + list(self.fit_encoder.parameters())
        self.ema.store(all_params)
        self.ema.copy_to(all_params)

        self.unet.eval()
        all_preds, all_targets = [], []

        self.ddim_scheduler.set_timesteps(
            self.cfg['ddim_steps'], device=self.device
        )

        for batch in tqdm(self.val_loader, desc='  val', leave=False):
            B = batch['person_img'].shape[0]
            def to(t): return t.to(device=self.device, dtype=self.dtype)

            # Prepare conditioning
            agnostic_lat = self.lat_enc.encode(to(batch['agnostic_img']),  sample=False)
            warped_lat   = self.lat_enc.encode(to(batch['warped_cloth']),   sample=False)
            cloth_lat    = self.lat_enc.encode(to(batch['cloth_clean']),    sample=False)
            fit_emb      = self.fit_encoder(to(batch['fit_features']))
            lH = self.cfg['target_h'] // 8
            lW = self.cfg['target_w'] // 8
            mask_lat  = F.interpolate(to(batch['agnostic_mask']), (lH,lW), mode='nearest')
            pose_lat  = F.interpolate(to(batch['pose_img']),      (lH,lW), mode='bilinear', align_corners=False)
            dense_lat = F.interpolate(to(batch['densepose_img']), (lH,lW), mode='nearest')
            text_emb  = self._encode_text(batch['prompt'])
            neg_emb   = self._encode_text([''] * B)

            # DDIM loop
            latents = torch.randn(B, 4, lH, lW, device=self.device, dtype=self.dtype)
            latents = latents * self.ddim_scheduler.init_noise_sigma

            for t in self.ddim_scheduler.timesteps:
                lat_in = self.ddim_scheduler.scale_model_input(
                    torch.cat([latents]*2), t
                )
                ts = torch.tensor([t]*(2*B), device=self.device)
                def dup(x): return torch.cat([x]*2)

                np_ = self.unet(
                    noisy_latent=lat_in, timestep=ts,
                    encoder_hidden_states=torch.cat([neg_emb, text_emb]),
                    agnostic_latent=dup(agnostic_lat),
                    agnostic_mask=dup(mask_lat),
                    warped_cloth_latent=dup(warped_lat),
                    cloth_latent=dup(cloth_lat),
                    pose_img=dup(pose_lat),
                    densepose_img=dup(dense_lat),
                    fit_embedding=dup(fit_emb),
                )
                uncond, cond = np_.chunk(2)
                np_ = uncond + self.cfg['guidance_scale'] * (cond - uncond)
                latents = self.ddim_scheduler.step(np_, t, latents).prev_sample

            preds = self.lat_enc.decode(latents).float().cpu()
            all_preds.append(preds)
            all_targets.append(batch['person_img'].float())

        # Restore training weights
        self.ema.restore(all_params)
        self.unet.train()

        all_preds   = torch.cat(all_preds,   dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        metrics = self.metrics_calc.compute(all_preds, all_targets)
        return metrics

    # ── Save visualization ────────────────────────────────────────

    @torch.no_grad()
    def _save_vis(self, batch: dict, step: int):
        """Save a quick 1-sample inference visualization."""
        import cv2

        B = 1
        def to(t): return t[:B].to(device=self.device, dtype=self.dtype)

        lH, lW = self.cfg['target_h']//8, self.cfg['target_w']//8

        agnostic_lat = self.lat_enc.encode(to(batch['agnostic_img']),  sample=False)
        warped_lat   = self.lat_enc.encode(to(batch['warped_cloth']),   sample=False)
        cloth_lat    = self.lat_enc.encode(to(batch['cloth_clean']),    sample=False)
        fit_emb      = self.fit_encoder(to(batch['fit_features']))
        mask_lat  = F.interpolate(to(batch['agnostic_mask']), (lH,lW), mode='nearest')
        pose_lat  = F.interpolate(to(batch['pose_img']),      (lH,lW), mode='bilinear', align_corners=False)
        dense_lat = F.interpolate(to(batch['densepose_img']), (lH,lW), mode='nearest')
        text_emb  = self._encode_text(batch['prompt'][:B])
        neg_emb   = self._encode_text([''])

        self.ddim_scheduler.set_timesteps(20, device=self.device)  # fast 20-step vis
        latents = torch.randn(B, 4, lH, lW, device=self.device, dtype=self.dtype)
        latents = latents * self.ddim_scheduler.init_noise_sigma

        for t in self.ddim_scheduler.timesteps:
            lat_in = self.ddim_scheduler.scale_model_input(torch.cat([latents]*2), t)
            ts = torch.tensor([t]*2, device=self.device)
            def dup(x): return torch.cat([x]*2)
            np_ = self.unet(
                noisy_latent=lat_in, timestep=ts,
                encoder_hidden_states=torch.cat([neg_emb, text_emb]),
                agnostic_latent=dup(agnostic_lat), agnostic_mask=dup(mask_lat),
                warped_cloth_latent=dup(warped_lat), cloth_latent=dup(cloth_lat),
                pose_img=dup(pose_lat), densepose_img=dup(dense_lat),
                fit_embedding=dup(fit_emb),
            )
            uncond, cond = np_.chunk(2)
            np_ = uncond + self.cfg['guidance_scale'] * (cond - uncond)
            latents = self.ddim_scheduler.step(np_, t, latents).prev_sample

        pred = self.lat_enc.decode(latents)

        def t2bgr(t):
            t = t[0].float().clamp(-1,1)
            t = ((t+1)/2*255).byte().permute(1,2,0).cpu().numpy()
            return cv2.cvtColor(t, cv2.COLOR_RGB2BGR)

        divider = np.ones((self.cfg['target_h'], 3, 3), dtype=np.uint8) * 200
        row = np.concatenate([
            t2bgr(batch['cloth_clean'][:1]),     divider,
            t2bgr(batch['agnostic_img'][:1]),    divider,
            t2bgr(pred),                          divider,
            t2bgr(batch['person_img'][:1]),
        ], axis=1)

        # Add labels
        import cv2 as _cv2
        for i, lbl in enumerate(['Cloth', 'Agnostic', 'Generated', 'GT']):
            x = i * (self.cfg['target_w'] + 3) + 5
            _cv2.putText(row, lbl, (x, 20), _cv2.FONT_HERSHEY_SIMPLEX,
                         0.6, (255,255,255), 1, _cv2.LINE_AA)

        path = str(self.vis_dir / f'step_{step:07d}.jpg')
        cv2.imwrite(path, row, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # ── Checkpoint ───────────────────────────────────────────────

    def save_checkpoint(self, tag: str = ''):
        ckpt = {
            'step':         self.global_step,
            'model':        self.unet.state_dict(),
            'fit_encoder':  self.fit_encoder.state_dict(),
            'optimizer':    self.optimizer.state_dict(),
            'scheduler':    self.scheduler.state_dict(),
            'scaler':       self.scaler.state_dict(),
            'ema':          self.ema.state_dict(),
            'config':       self.cfg,
            'best_ssim':    self.best_ssim,
        }
        fname = f'step_{self.global_step:07d}{tag}.pth'
        torch.save(ckpt, self.out_dir / fname)
        print(f"  Saved: {fname}")

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.unet.load_state_dict(ckpt['model'], strict=False)
        self.fit_encoder.load_state_dict(ckpt['fit_encoder'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.scaler.load_state_dict(ckpt['scaler'])
        self.ema.load_state_dict(ckpt['ema'])
        self.best_ssim   = ckpt.get('best_ssim', 0.0)
        self.global_step = ckpt.get('step', 0)
        print(f"Resumed from {path} | step={self.global_step}")
        return self.global_step

    # ── Main training loop ────────────────────────────────────────

    def run(self, resume_from: Optional[str] = None):
        if resume_from:
            self.load_checkpoint(resume_from)

        print(f"\n{'='*60}")
        print(f"Generation model training | {self.cfg['num_steps']} steps")
        print(f"{'='*60}\n")

        self.unet.train()
        self.fit_encoder.train()

        train_iter  = iter(self.train_loader)
        running_loss = 0.0
        t0 = time.time()
        log_interval = 50

        vis_batch = None   # cache one batch for visualization

        while self.global_step < self.cfg['num_steps']:
            # Cycle dataloader
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            if vis_batch is None:
                vis_batch = batch

            loss = self._train_step(batch)
            running_loss += loss
            self.global_step += 1

            # ── Periodic logging ──
            if self.global_step % log_interval == 0:
                avg_loss = running_loss / log_interval
                elapsed  = time.time() - t0
                steps_remaining = self.cfg['num_steps'] - self.global_step
                eta_hrs = steps_remaining * (elapsed / self.global_step) / 3600
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Step {self.global_step:6d}/{self.cfg['num_steps']} | "
                      f"loss={avg_loss:.4f} | lr={lr:.2e} | "
                      f"ETA: {eta_hrs:.1f}h")
                self.history['train_loss'].append(
                    {'step': self.global_step, 'loss': avg_loss}
                )
                running_loss = 0.0

            # ── Visualization ──
            if self.global_step % self.cfg['vis_every'] == 0:
                self.unet.eval()
                self._save_vis(vis_batch, self.global_step)
                self.unet.train()

            # ── Validation ──
            if self.global_step % self.cfg['val_every'] == 0:
                metrics = self._validate()
                ssim_val = metrics.get('ssim', 0.0)
                lpips_val = metrics.get('lpips', 1.0)
                fid_val   = metrics.get('fid', 999.0)
                print(f"\n  ★ Val step {self.global_step} | "
                      f"SSIM={ssim_val:.4f} | "
                      f"LPIPS={lpips_val:.4f} | "
                      f"FID={fid_val:.1f}")
                self.history['val'].append({
                    'step': self.global_step, **metrics
                })

                if ssim_val > self.best_ssim:
                    self.best_ssim = ssim_val
                    self.save_checkpoint(tag='_best')
                    print(f"  ★★ New best SSIM: {self.best_ssim:.4f}")

                with open(self.out_dir / 'history.json', 'w') as f:
                    json.dump(self.history, f, indent=2)

            # ── Periodic save ──
            if self.global_step % self.cfg['save_every'] == 0:
                self.save_checkpoint()

        print(f"\nTraining complete at step {self.global_step}")
        print(f"Best SSIM: {self.best_ssim:.4f}")
        self.save_checkpoint(tag='_final')
        return self.history


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',    default='preprocessed_dataset/train_manifest.json')
    parser.add_argument('--output_dir',  default='generation/checkpoints')
    parser.add_argument('--model_id',    default='sd-legacy/stable-diffusion-v1-5')
    parser.add_argument('--batch_size',  type=int,   default=2)
    parser.add_argument('--num_steps',   type=int,   default=50_000)
    parser.add_argument('--lr',          type=float, default=1e-5)
    parser.add_argument('--grad_ckpt',   action='store_true', default=True)
    parser.add_argument('--fp16',        action='store_true', default=True)
    parser.add_argument('--resume',      default=None)
    args = parser.parse_args()

    import argparse
    config = vars(args)
    trainer = GenerationTrainer(config)
    trainer.run(resume_from=config.pop('resume', None))
