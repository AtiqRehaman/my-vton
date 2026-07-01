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

Optimizer: AdamW8bit, lr=1e-5, weight_decay=0.01
  Why 1e-5? The UNet was pretrained — too high LR destroys
  pretrained weights. 1e-5 is the standard fine-tuning LR for SD 1.5.

  Why AdamW8bit (bitsandbytes), not torch.optim.AdamW?
  Fine-tuning the FULL ~860M-parameter SD UNet at fp32 needs roughly:
      weights (fp32):            3.44 GB
      gradients (fp32):           3.44 GB
      AdamW fp32 exp_avg:         3.44 GB
      AdamW fp32 exp_avg_sq:      3.44 GB
      ─────────────────────────────────
      Subtotal:                  13.76 GB  ← exceeds a T4's 14.56 GB
                                              before a single activation
                                              tensor is allocated.
  AdamW8bit keeps exp_avg/exp_avg_sq in 8-bit instead of fp32 — cuts
  that to ~1.7 GB combined, which is what makes full UNet fine-tuning
  fit on a T4 (or Kaggle's P100/T4) at all. Falls back to standard
  AdamW with a loud warning if bitsandbytes isn't installed.

Mixed precision — IMPORTANT, this is the one configuration detail
most likely to be gotten wrong:
  Trainable models (unet, fit_encoder) are ALWAYS loaded/kept in
  fp32, regardless of the 'fp16' config flag. GradScaler.unscale_()
  only operates on fp32 gradients — loading trainable weights in fp16
  causes "ValueError: Attempting to unscale FP16 gradients" the first
  time backward() runs.
  The 'fp16' flag instead controls:
    (a) dtype of the FROZEN components (VAE, CLIP text encoder) —
        these are never touched by the optimizer/scaler, so fp16 is
        safe and saves real memory for them.
    (b) whether torch.cuda.amp.autocast() runs the forward pass in
        fp16 — this is what gives you fp16-speed/memory benefits on
        ACTIVATIONS without the weights themselves needing to be fp16.
  Leave 'fp16': True. Setting it False does not protect against any
  remaining bug — it just doubles frozen-component memory and disables
  autocast's activation savings, which has caused OOM in its own right
  before.

Gradient checkpointing: MANDATORY on T4-class GPUs (including
  Kaggle's T4 and P100). Reduces activation VRAM substantially at the
  cost of ~20% slower steps — without it, even B=1 struggles to fit
  alongside the fp32 trainable weights + AdamW8bit state above.

EMA: decay=0.9999, applied after every optimizer step. EMA shadow
  buffers are fp32 (same dtype as the trainable params they track) —
  this falls out automatically since EMAModel is constructed AFTER
  unet/fit_encoder are already fp32, no separate handling needed.

Text conditioning: 10% null-text dropout (classifier-free guidance).
  10% of training steps use empty string prompt, enabling CFG at
  inference time (guidance_scale=2.0).

Checkpoint save/load — both deliberately avoid materializing a full
  extra GPU- or CPU-resident copy of the checkpoint all at once:
    - save_checkpoint() moves each state_dict to CPU incrementally
      (one component at a time, with gc.collect() between), then
      writes to a temp file and atomically renames — so a crash
      mid-write never leaves a corrupted file at the real checkpoint
      filename.
    - load_checkpoint() loads with map_location='cpu' (never touches
      GPU during deserialization) and lets each load_state_dict()
      transfer its own tensors to GPU as it overwrites the live ones,
      rather than torch.load materializing the whole checkpoint on
      GPU simultaneously alongside the already-live model.
  Both of these were root-caused from real OOM/RAM-exhaustion crashes
  during long Colab training runs — see project history for the
  exact tracebacks that motivated each one.

─────────────────────────────────────────────────────────────────
VRAM BUDGET (T4-class, ~15 GB, fp32 trainable + fp16 frozen + AdamW8bit)
─────────────────────────────────────────────────────────────────
  UNet + fit_encoder weights (fp32):    ~3.44 GB
  Gradients (fp32):                      ~3.44 GB
  AdamW8bit state (8-bit):                ~1.72 GB
  VAE + CLIP (frozen, fp16):              ~0.41 GB
  EMA shadow (fp32):                      ~3.44 GB
  ─────────────────────────────────────────────
  Static subtotal:                       ~12.45 GB
  Activations (B=2, grad ckpt ON):        ~1.5-2 GB
  ─────────────────────────────────────────────
  Total:                                 ~14-14.5 GB ← tight but fits
                                                          a 14.56GB T4
  If this still OOMs: reduce batch_size to 1 first (cheapest lever).
─────────────────────────────────────────────────────────────────
"""

import io
import os
import gc
import json
import time
import argparse
from pathlib import Path
from typing import Optional
import tempfile  
import shutil  

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from huggingface_hub import HfApi, upload_file, list_repo_files, hf_hub_download


# Kaggle defaults — /kaggle/working is the only writable directory;
# /kaggle/input/... is READ-ONLY (mounted dataset). Adjust the input
# path to match your actual dataset slug, e.g.:
#   /kaggle/input/preprocessed-dataset/preprocessed_dataset/train_manifest.json
# Override these via the config dict passed to GenerationTrainer —
# they're just sane starting points, not hardcoded requirements.
DEFAULT_CONFIG = {
    # Data
    'manifest':     '/kaggle/input/preprocessed-dataset/preprocessed_dataset/train_manifest.json',
    'output_dir':   '/kaggle/working/my-vton/generation/checkpoints',
    'cloth_type':   'upper',
    'target_h':     512,
    'target_w':     384,

    # Base model
    'model_id':     'sd-legacy/stable-diffusion-v1-5',

    # Training
    'batch_size':   2,           # T4/P100 at 512×384 with grad ckpt + AdamW8bit
    'num_steps':    50_000,      # total optimizer steps
    'lr':           1e-5,
    'weight_decay': 0.01,
    'val_every':    5_000,       # run val every N steps
    'save_every':   5_000,       # save checkpoint every N steps
    'vis_every':    500,         # save sample images every N steps
    'val_split':    0.1,
    'num_workers':  2,           # Kaggle: keep low, same caution as Colab
    'seed':         42,

    # Diffusion
    'num_train_timesteps': 1000,
    'null_text_prob':      0.1,  # 10% null text for CFG training

    # Loss
    'loss_type':    'mse',       # 'mse' or 'huber'

    # Hardware
    'fp16':         True,        # controls FROZEN component dtype + autocast.
                                  # Trainable weights are ALWAYS fp32 regardless
                                  # — see module docstring. Do not set False as
                                  # a workaround; it doesn't fix anything anymore
                                  # and wastes memory.
    'grad_ckpt':    True,        # gradient checkpointing (MANDATORY on T4/P100)
    'xformers':     False,       # set True if xformers is installed
    'use_8bit_adam': True,       # MANDATORY in practice — see VRAM budget above
    'frozen_device': None,       # e.g. 'cuda:1' on a Kaggle T4x2 runtime to put
                                  # the FROZEN VAE/CLIP on the second GPU,
                                  # freeing their memory from the main training
                                  # device. None = same device as everything
                                  # else (self.device). Only meaningful with 2+
                                  # GPUs visible; harmless no-op otherwise (falls
                                  # back to self.device with a warning).
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

        # ── SSIM (from torchmetrics) ──
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
        # NOTE: self.dtype controls FROZEN components (VAE, CLIP) and
        # autocast — it does NOT control trainable weight dtype.
        # Trainable models are forced to fp32 explicitly in
        # _load_models() regardless of this value. See module docstring.
        self.dtype  = torch.float16 if (self.cfg['fp16'] and self.device.type == 'cuda') \
                      else torch.float32
        self.use_fp16 = (self.dtype == torch.float16)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_fp16)

        self.out_dir = Path(self.cfg['output_dir'])
        self.vis_dir = self.out_dir / 'vis'
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        print(f"Device: {self.device} | frozen/autocast dtype: {self.dtype} "
              f"| trainable dtype: float32 (always)")

        self._load_models()
        self._build_datasets()
        self._build_optimizer()

        self.global_step  = 0
        self.best_ssim    = 0.0
        self.history      = {'train_loss': [], 'val': []}
        self.metrics_calc = MetricsCalculator(str(self.device))
        
        # ── Hugging Face Hub setup (HTTP API only - NO CLONING) ──
        self.hf_repo_id = self.cfg.get('hf_repo_id')
        self.hf_token = self.cfg.get('hf_token')
        self.hf_api = None
        
        if self.hf_repo_id and self.hf_token:
            try:
                from huggingface_hub import HfApi
                self.hf_api = HfApi()
                print(f"✅ Connected to HF Hub (HTTP API): {self.hf_repo_id}")
                
                # Try to load local history if exists
                history_path = self.out_dir / 'history.json'
                if history_path.exists():
                    with open(history_path) as f:
                        self.history = json.load(f)
                        print(f"  Loaded existing local history")
                        
                        # Find best SSIM from history
                        if self.history.get('val'):
                            self.best_ssim = max([v.get('ssim', 0) for v in self.history['val']])
                            print(f"  Best SSIM from history: {self.best_ssim:.4f}")
                    
            except Exception as e:
                print(f"⚠️ HF Hub setup failed: {e}")
                self.hf_api = None
        else:
            print("ℹ️ HF Hub not configured - saving locally only")

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

        # ── Resolve frozen_device: where VAE/CLIP actually live ──
        # On a Kaggle T4x2 runtime, setting frozen_device='cuda:1' moves
        # the frozen components off the main training GPU entirely,
        # freeing their ~0.4GB (fp16) from cuda:0. This is a SMALL win
        # for this model specifically (frozen components are tiny next
        # to the ~860M-param UNet) but it's free headroom if a second
        # GPU is sitting idle anyway. Falls back to self.device with a
        # warning if the requested device doesn't actually exist.
        requested = self.cfg.get('frozen_device')
        if requested:
            try:
                idx = int(str(requested).split(':')[-1])
                if idx < torch.cuda.device_count():
                    self.frozen_device = torch.device(requested)
                else:
                    print(f"  ⚠️  frozen_device={requested} requested but only "
                          f"{torch.cuda.device_count()} GPU(s) visible — "
                          f"falling back to {self.device}")
                    self.frozen_device = self.device
            except (ValueError, IndexError):
                print(f"  ⚠️  Could not parse frozen_device={requested!r} — "
                      f"falling back to {self.device}")
                self.frozen_device = self.device
        else:
            self.frozen_device = self.device

        self.cross_device = (self.frozen_device != self.device)
        if self.cross_device:
            print(f"  Frozen components (VAE/CLIP) → {self.frozen_device} "
                  f"| Trainable (UNet/optimizer) → {self.device}")

        # ── Frozen components: safe to load in self.dtype (fp16) ──
        # Never touched by the optimizer/GradScaler — only forward-pass
        # inference runs through them, so fp16 here is pure savings with
        # no GradScaler interaction to worry about.
        self.vae = AutoencoderKL.from_pretrained(
            mid, subfolder='vae', torch_dtype=self.dtype
        ).to(self.frozen_device)
        for p in self.vae.parameters():
            p.requires_grad = False

        self.text_encoder = CLIPTextModel.from_pretrained(
            mid, subfolder='text_encoder', torch_dtype=self.dtype
        ).to(self.frozen_device)
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        self.tokenizer = CLIPTokenizer.from_pretrained(mid, subfolder='tokenizer')

        self.noise_scheduler = DDPMScheduler.from_pretrained(mid, subfolder='scheduler')
        self.ddim_scheduler  = DDIMScheduler.from_pretrained(mid, subfolder='scheduler')

        # ── Trainable components: MUST stay fp32, regardless of self.dtype ──
        # GradScaler.unscale_() only operates on fp32 gradients. Loading
        # these in fp16 causes "ValueError: Attempting to unscale FP16
        # gradients" the first time backward() runs. autocast() (used in
        # _train_step) already gives fp16 forward-pass benefits on
        # activations without the weights themselves needing to be fp16.
        # ALWAYS on self.device (the main training GPU), regardless of
        # frozen_device — only the frozen components can be offloaded.
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

        # EMA — constructed AFTER unet/fit_encoder are confirmed fp32,
        # so shadow buffers are fp32 too (dtype-consistent with the
        # live params EMAModel.step()/.copy_to() operate on).
        self.ema = EMAModel(
            list(self.unet.parameters()) + list(self.fit_encoder.parameters()),
            decay=self.cfg['ema_decay'],
        )

        self.lat_enc = LatentEncoder(self.vae)
        print(f"All models loaded. Trainable: fp32 | Frozen VAE/CLIP: {self.dtype}")

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
        self.steps_per_epoch = len(self.train_loader)
        print(f"Train: {n_train} | Val: {n_val} | "
              f"Steps/epoch: {self.steps_per_epoch}")

    # ── Optimizer ────────────────────────────────────────────────

    def _build_optimizer(self):
        trainable = (
            list(self.unet.parameters()) +
            list(self.fit_encoder.parameters())
        )

        # 8-bit AdamW: keeps exp_avg/exp_avg_sq momentum buffers in
        # 8-bit instead of fp32 — cuts optimizer state from ~6.9GB to
        # ~1.7GB for this model size. This is what makes fine-tuning
        # the full ~860M-param UNet fit on a T4/P100 at all; with
        # standard fp32 AdamW, static weights+gradients+optimizer
        # state alone exceed a 14.56GB T4 before a single activation
        # tensor is allocated. See module docstring for the full
        # VRAM breakdown.
        use_8bit = self.cfg.get('use_8bit_adam', True)
        if use_8bit:
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(
                    trainable,
                    lr           = self.cfg['lr'],
                    weight_decay = self.cfg['weight_decay'],
                )
                print("  Optimizer: AdamW8bit (bitsandbytes) — required to "
                      "fit the full UNet fine-tune on a 14-16GB GPU")
            except ImportError:
                print("  ⚠️  bitsandbytes not installed — falling back to "
                      "standard AdamW.\n"
                      "      This WILL likely OOM at fp32 on a T4/P100. "
                      "Run: pip install bitsandbytes")
                self.optimizer = torch.optim.AdamW(
                    trainable,
                    lr           = self.cfg['lr'],
                    weight_decay = self.cfg['weight_decay'],
                )
        else:
            self.optimizer = torch.optim.AdamW(
                trainable,
                lr           = self.cfg['lr'],
                weight_decay = self.cfg['weight_decay'],
            )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max  = self.cfg['num_steps'],
            eta_min = self.cfg['lr'] * 0.01,
        )

    # ── Text encoding ────────────────────────────────────────────

    @torch.no_grad()
    def _encode_text(self, prompts: list) -> torch.Tensor:
        """
        Encode list of prompts to CLIP embeddings. (B, 77, 768)

        Routes tokens to wherever self.text_encoder actually lives
        (self.frozen_device — which equals self.device unless a
        separate frozen_device was configured for a dual-GPU setup),
        then brings the result back to self.device so every call site
        elsewhere in this file can keep using the output as if
        everything were on one GPU.
        """
        tokens = self.tokenizer(
            prompts,
            padding    = 'max_length',
            max_length = 77,
            truncation = True,
            return_tensors = 'pt',
        ).input_ids.to(self.frozen_device)
        text_emb = self.text_encoder(tokens)[0]
        return text_emb.to(device=self.device, dtype=self.dtype)

    # ── Single training step ─────────────────────────────────────

    def _train_step(self, batch: dict) -> float:
        """Run one forward+backward step. Returns scalar loss."""
        B = batch['person_img'].shape[0]

        def to(t):
            return t.to(device=self.device, dtype=self.dtype)

        # fit_encoder is a TRAINABLE component, kept deliberately fp32
        # always (see _load_models() — required for GradScaler to work
        # on its gradients). Its input must therefore also be fp32,
        # never self.dtype. Earlier reasoning assumed autocast() would
        # transparently bridge a fp16 input into fp32-weighted layers
        # inside the context below — that assumption was WRONG and
        # caused a real crash ("expected mat1 and mat2 to have the
        # same dtype, but got: c10::Half != float") after the same fix
        # had already been applied to _validate()/_save_vis() but not
        # here. Do not move fit_feats back to to() — autocast does not
        # reliably rescue a fp32-weight/fp16-input mismatch for
        # nn.Linear in all cases.
        def to_fp32(t):
            return t.to(device=self.device, dtype=torch.float32)

        person_img   = to(batch['person_img'])
        agnostic_img = to(batch['agnostic_img'])
        agnostic_msk = to(batch['agnostic_mask'])
        warped_cloth = to(batch['warped_cloth'])
        cloth_clean  = to(batch['cloth_clean'])
        pose_img     = to(batch['pose_img'])
        densepose    = to(batch['densepose_img'])
        fit_feats    = to_fp32(batch['fit_features'])

        with torch.cuda.amp.autocast(enabled=self.use_fp16):

            # ── Encode GT person to latent (training target z_0) ──
            z0 = self.lat_enc.encode(person_img, sample=True, output_device=self.device)

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

            agnostic_lat = self.lat_enc.encode(agnostic_img, sample=False, output_device=self.device)
            warped_lat   = self.lat_enc.encode(warped_cloth,  sample=False, output_device=self.device)
            cloth_lat    = self.lat_enc.encode(cloth_clean,   sample=False, output_device=self.device)

            mask_lat     = F.interpolate(agnostic_msk, (lH, lW), mode='nearest')
            pose_lat     = F.interpolate(pose_img,     (lH, lW), mode='bilinear',
                                          align_corners=False)
            dense_lat    = F.interpolate(densepose,    (lH, lW), mode='nearest')
            # fit_encoder's weights are fp32 (trainable component), so
            # its output is fp32 regardless of autocast. But the UNet
            # concatenates all 8 channel groups together (see
            # unet_modified.py forward()), and torch.cat silently
            # upcasts the WHOLE result to fp32 if even one input is
            # fp32 — that doesn't crash, but it defeats fp16 autocast's
            # memory/speed benefit for the entire UNet forward pass
            # without any visible error. Cast fit_emb to match
            # z_t's dtype (whatever autocast actually produced for the
            # other tensors) right before the concat, not earlier.
            fit_emb = self.fit_encoder(fit_feats).to(z_t.dtype)

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
            # fit_encoder is deliberately kept fp32 (it's a TRAINABLE
            # component — see _load_models()), so its input must also
            # be fp32, not self.dtype (which is fp16 when fp16=True).
            # Using to() here would feed a fp16 tensor into fp32
            # nn.Linear layers and raise "mat1 and mat2 must have the
            # same dtype" — confirmed by a real crash during _save_vis,
            # which has the identical pattern (see below).
            def to_fp32(t): return t.to(device=self.device, dtype=torch.float32)

            agnostic_lat = self.lat_enc.encode(to(batch['agnostic_img']),  sample=False, output_device=self.device)
            warped_lat   = self.lat_enc.encode(to(batch['warped_cloth']),   sample=False, output_device=self.device)
            cloth_lat    = self.lat_enc.encode(to(batch['cloth_clean']),    sample=False, output_device=self.device)
            # fit_encoder's weights are fp32, so its raw output is
            # fp32 regardless of self.dtype. The UNet concatenates all
            # 8 channel groups (unet_modified.py forward()), and
            # torch.cat silently upcasts the WHOLE result to fp32 if
            # even one input is fp32 — no error, but it defeats fp16
            # for the entire UNet forward pass here. Cast to self.dtype
            # to match every other tensor in this loop.
            fit_emb      = self.fit_encoder(to_fp32(batch['fit_features'])).to(self.dtype)
            lH = self.cfg['target_h'] // 8
            lW = self.cfg['target_w'] // 8
            mask_lat  = F.interpolate(to(batch['agnostic_mask']), (lH,lW), mode='nearest')
            pose_lat  = F.interpolate(to(batch['pose_img']),      (lH,lW), mode='bilinear', align_corners=False)
            dense_lat = F.interpolate(to(batch['densepose_img']), (lH,lW), mode='nearest')
            text_emb  = self._encode_text(batch['prompt'])
            neg_emb   = self._encode_text([''] * B)

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

            preds = self.lat_enc.decode(latents, output_device=self.device).float().cpu()
            all_preds.append(preds)
            all_targets.append(batch['person_img'].float())

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
        # Same fp32 requirement as _validate() — fit_encoder is a
        # TRAINABLE component kept deliberately fp32, so its input
        # can't go through the fp16 to() helper used for everything
        # else here. This is the exact line that crashed:
        # "RuntimeError: mat1 and mat2 must have the same dtype, but
        # got Half and Float" at step 250 via vis_every.
        def to_fp32(t): return t[:B].to(device=self.device, dtype=torch.float32)

        lH, lW = self.cfg['target_h']//8, self.cfg['target_w']//8

        agnostic_lat = self.lat_enc.encode(to(batch['agnostic_img']),  sample=False, output_device=self.device)
        warped_lat   = self.lat_enc.encode(to(batch['warped_cloth']),   sample=False, output_device=self.device)
        cloth_lat    = self.lat_enc.encode(to(batch['cloth_clean']),    sample=False, output_device=self.device)
        # Same upcast risk as _train_step/_validate — cast to self.dtype
        # to match the other tensors feeding into the unet concat.
        fit_emb      = self.fit_encoder(to_fp32(batch['fit_features'])).to(self.dtype)
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

        pred = self.lat_enc.decode(latents, output_device=self.device)

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

        for i, lbl in enumerate(['Cloth', 'Agnostic', 'Generated', 'GT']):
            x = i * (self.cfg['target_w'] + 3) + 5
            cv2.putText(row, lbl, (x, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255,255,255), 1, cv2.LINE_AA)

        path = str(self.vis_dir / f'step_{step:07d}.jpg')
        cv2.imwrite(path, row, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # ── Checkpoint save (CPU-incremental + atomic write) ──────────

    def _state_dict_to_cpu(self, module_or_optimizer) -> dict:
        """
        Move a state_dict to CPU explicitly, tensor by tensor, rather
        than letting torch.save() pull the whole thing to CPU
        internally in one shot. This doesn't reduce the TOTAL bytes
        copied, but it lets us free each big piece (gc.collect())
        before building the next one, so peak CPU RAM usage during a
        save is closer to "one component's worth" rather than
        "unet + EMA + optimizer all alive on CPU simultaneously" —
        the latter was confirmed to exhaust system RAM and crash the
        runtime during a real training run.

        Handles both plain module state_dicts ({name: tensor, ...})
        and optimizer state_dicts, which have the shape:
            {'state': {param_idx: {'step': tensor, 'exp_avg': tensor,
                                    'exp_avg_sq': tensor}, ...},
             'param_groups': [{'lr': ..., 'params': [...]}, ...]}
        i.e. exactly two levels of nesting for 'state', and a
        tensor-free 'param_groups' list — not arbitrary depth.
        """
        sd = module_or_optimizer.state_dict()

        if 'state' in sd and 'param_groups' in sd:
            cpu_state = {
                param_idx: {
                    k: (v.detach().cpu() if torch.is_tensor(v) else v)
                    for k, v in per_param_state.items()
                }
                for param_idx, per_param_state in sd['state'].items()
            }
            return {'state': cpu_state, 'param_groups': sd['param_groups']}

        return {
            k: (v.detach().cpu() if torch.is_tensor(v) else v)
            for k, v in sd.items()
        }

    def save_checkpoint(self, tag: str = ''):
        fname = f'step_{self.global_step:07d}{tag}.pth'
        final_path = self.out_dir / fname
        tmp_path = self.out_dir / f'{fname}.tmp'

        # Build the checkpoint dict (same as before)
        ckpt = {'step': self.global_step, 'config': self.cfg, 'best_ssim': self.best_ssim}
        ckpt['model'] = self._state_dict_to_cpu(self.unet)
        gc.collect()
        ckpt['fit_encoder'] = self._state_dict_to_cpu(self.fit_encoder)
        gc.collect()
        ckpt['optimizer'] = self._state_dict_to_cpu(self.optimizer)
        gc.collect()
        ckpt['scheduler'] = self.scheduler.state_dict()
        ckpt['scaler'] = self.scaler.state_dict()
        ema_state = self.ema.state_dict()
        ckpt['ema'] = {
            'decay': ema_state['decay'],
            'shadow': [s.detach().cpu() for s in ema_state['shadow']],
        }
        gc.collect()

        try:
            # Save locally first
            torch.save(ckpt, tmp_path)
            os.replace(tmp_path, final_path)
            print(f"  ✅ Saved locally: {fname} ({final_path.stat().st_size / 1e9:.2f} GB)")

            # ── Upload to HF Hub via HTTP API ──
            if self.hf_api is not None and self.hf_repo_id:
                try:
                    # Upload checkpoint file
                    hf_path = f"checkpoints/{fname}"
                    print(f"  📤 Uploading to HF Hub: {hf_path} ...")
                    
                    upload_file(
                        path_or_fileobj=str(final_path),
                        path_in_repo=hf_path,
                        repo_id=self.hf_repo_id,
                        token=self.hf_token,
                    )
                    print(f"  ✅ Uploaded to HF Hub: {hf_path}")
                    
                    # Upload latest.txt (to track latest checkpoint)
                    latest_content = fname.encode('utf-8')
                    latest_bytes = io.BytesIO(latest_content)
                    upload_file(
                        path_or_fileobj=latest_bytes,
                        path_in_repo="checkpoints/latest.txt",
                        repo_id=self.hf_repo_id,
                        token=self.hf_token,
                    )
                    
                    # Upload history.json
                    history_json = json.dumps(self.history, indent=2).encode('utf-8')
                    history_bytes = io.BytesIO(history_json)
                    upload_file(
                        path_or_fileobj=history_bytes,
                        path_in_repo="history.json",
                        repo_id=self.hf_repo_id,
                        token=self.hf_token,
                    )
                    
                    print(f"  ✅ Pushed all files to HF Hub: {self.hf_repo_id}")
                    
                    # ── Clean up: Delete old local checkpoints, keep only latest ──
                    self._keep_only_latest_checkpoint()
                    
                except Exception as e:
                    print(f"  ⚠️ Failed to upload to HF Hub: {e}")
                    # Keep the local checkpoint if upload failed
                    
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            print(f"  ❌ Checkpoint save FAILED at step {self.global_step}: {e}")
            raise
        finally:
            del ckpt
            gc.collect()
            
    
    def download_latest_from_hf(self) -> Optional[str]:
        """
        Download the latest checkpoint from HF Hub to local directory.
        Returns the path to the downloaded checkpoint.
        """
        if self.hf_api is None or not self.hf_repo_id:
            print("⚠️ HF Hub not configured")
            return None
        
        try:
            from huggingface_hub import hf_hub_download
            import shutil
            
            print("📥 Fetching latest checkpoint info from HF...")
            
            # Download latest.txt to a temp file (not in a nested folder)
            latest_path = hf_hub_download(
                repo_id=self.hf_repo_id,
                filename="checkpoints/latest.txt",
                token=self.hf_token,
                local_dir=self.out_dir,  # ← Download directly to out_dir
                local_files_only=False,
            )
            
            with open(latest_path, 'r') as f:
                fname = f.read().strip()
            
            print(f"📥 Downloading checkpoint: {fname}")
            
            # Download the checkpoint directly to out_dir
            ckpt_path = hf_hub_download(
                repo_id=self.hf_repo_id,
                filename=f"checkpoints/{fname}",
                token=self.hf_token,
                local_dir=self.out_dir,  # ← Download directly to out_dir
                local_files_only=False,
            )
            
            # The file will be at: self.out_dir / 'checkpoints' / fname
            # Move it to the main directory
            downloaded_path = self.out_dir / 'checkpoints' / fname
            final_path = self.out_dir / fname
            
            if downloaded_path.exists():
                shutil.move(str(downloaded_path), str(final_path))
                print(f"✅ Downloaded latest checkpoint: {final_path}")
                
                # Clean up the checkpoints folder if empty
                try:
                    (self.out_dir / 'checkpoints').rmdir()
                except:
                    pass
                
                # Update latest.txt locally
                with open(self.out_dir / 'latest.txt', 'w') as f:
                    f.write(fname)
                
                return str(final_path)
            else:
                print(f"⚠️ Downloaded file not found at {downloaded_path}")
                return None
            
        except Exception as e:
            print(f"⚠️ Failed to download from HF: {e}")
            return None
            
    def _keep_only_latest_checkpoint(self):
        """Delete all old local checkpoints, keep only the latest."""
        try:
            ckpt_files = sorted(
                [p for p in self.out_dir.glob('step_*.pth') if not p.is_symlink()],
                key=lambda p: int(p.stem.split('_')[1])
            )
            
            # Keep only the latest checkpoint
            if len(ckpt_files) > 1:
                for old_ckpt in ckpt_files[:-1]:  # Delete all except the newest
                    try:
                        old_ckpt.unlink()
                        print(f"  🗑️ Deleted old local: {old_ckpt.name}")
                    except Exception as e:
                        print(f"  ⚠️ Could not delete {old_ckpt.name}: {e}")
            
            # Update local latest.txt
            if ckpt_files:
                with open(self.out_dir / 'latest.txt', 'w') as f:
                    f.write(ckpt_files[-1].name)
                print(f"  📝 Local latest: {ckpt_files[-1].name}")
                
        except Exception as e:
            print(f"  ⚠️ Could not clean up old checkpoints: {e}")
                    
    # ── Checkpoint load (CPU-staged, no GPU double-allocation) ────

    def load_checkpoint(self, path: str) -> int:
        # Load to CPU, not GPU. torch.load with map_location='cpu'
        # never touches the GPU during deserialization — the whole
        # checkpoint sits in CPU RAM first. Each load_state_dict()
        # call below then moves ONE tensor to GPU at a time as it
        # overwrites the live tensor, instead of torch.load
        # materializing the ENTIRE checkpoint dict on GPU
        # simultaneously alongside the already-live model/optimizer/
        # EMA state — that simultaneous double-allocation was
        # confirmed to OOM on resume once optimizer state grew past
        # trivial size.
        ckpt = torch.load(path, map_location='cpu')

        self.unet.load_state_dict(ckpt['model'], strict=False)
        self.fit_encoder.load_state_dict(ckpt['fit_encoder'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.scaler.load_state_dict(ckpt['scaler'])
        self.ema.load_state_dict(ckpt['ema'])

        self.best_ssim   = ckpt.get('best_ssim', 0.0)
        self.global_step = ckpt.get('step', 0)

        del ckpt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Resumed from {path} | step={self.global_step}")
        return self.global_step
    
    def load_checkpoint_from_hf(self, step: Optional[int] = None):
        """
        Load a specific checkpoint from HF Hub without downloading everything.
        """
        if self.hf_api is None or not self.hf_repo_id:
            raise ValueError("HF Hub not configured")
        
        try:
            # Get the latest checkpoint name from HF
            if step is None:
                # Download the latest.txt file from HF
                latest_path = hf_hub_download(
                    repo_id=self.hf_repo_id,
                    filename="checkpoints/latest.txt",
                    token=self.hf_token,
                    local_dir=self.out_dir,  # ← Download directly
                )
                with open(latest_path, 'r') as f:
                    fname = f.read().strip()
                step = int(fname.split('_')[1])
            
            # Download the specific checkpoint
            ckpt_name = f'step_{step:07d}.pth'
            print(f"📥 Downloading checkpoint from HF: {ckpt_name}")
            
            ckpt_path = hf_hub_download(
                repo_id=self.hf_repo_id,
                filename=f"checkpoints/{ckpt_name}",
                token=self.hf_token,
                local_dir=self.out_dir,  # ← Download directly
            )
            
            # The file will be at: self.out_dir / 'checkpoints' / ckpt_name
            # Move it to the main directory if needed
            downloaded_path = self.out_dir / 'checkpoints' / ckpt_name
            final_path = self.out_dir / ckpt_name
            
            if downloaded_path.exists() and downloaded_path != final_path:
                import shutil
                shutil.move(str(downloaded_path), str(final_path))
                # Clean up empty folder
                try:
                    (self.out_dir / 'checkpoints').rmdir()
                except:
                    pass
            
            # Load the checkpoint
            return self.load_checkpoint(str(final_path))
            
        except Exception as e:
            print(f"⚠️ Failed to load from HF: {e}")
            raise
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the latest local checkpoint."""
        latest_file = self.out_dir / 'latest.txt'
        if latest_file.exists():
            with open(latest_file, 'r') as f:
                fname = f.read().strip()
            path = self.out_dir / fname
            if path.exists():
                return str(path)
        return None
    
    
    def resume_or_start(self):
        """
        Automatically resume from the latest checkpoint (local or HF).
        """
        # First check if we have a local checkpoint
        latest_local = self.out_dir / 'latest.txt'
        if latest_local.exists():
            with open(latest_local, 'r') as f:
                fname = f.read().strip()
            local_path = self.out_dir / fname
            if local_path.exists():
                print(f"✅ Found local checkpoint: {fname}")
                return self.run(resume_from=str(local_path))
        
        # If no local checkpoint, try HF Hub
        if self.hf_api is not None and self.hf_repo_id:
            print("🔄 No local checkpoint found, trying HF Hub...")
            try:
                # Check if there's a checkpoint on HF
                from huggingface_hub import list_repo_files
                files = list_repo_files(self.hf_repo_id, token=self.hf_token)
                checkpoints = [f for f in files if f.startswith('checkpoints/step_') and f.endswith('.pth')]
                
                if checkpoints:
                    print(f"📥 Found {len(checkpoints)} checkpoints on HF")
                    ckpt_path = self.download_latest_from_hf()
                    if ckpt_path:
                        return self.run(resume_from=ckpt_path)
            except Exception as e:
                print(f"⚠️ Could not check HF: {e}")
        
        # No checkpoint found anywhere, start fresh
        print("🆕 No checkpoint found - Starting fresh training")
        return self.run()

    # ── Main training loop ────────────────────────────────────────

    def run(self, resume_from: Optional[str] = None):
        if resume_from:
            # Clear any cached-but-unused GPU memory before resuming —
            # combined with the CPU-staged load above, this minimizes
            # peak GPU usage during the resume transition.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.load_checkpoint(resume_from)
            
        # elif resume_from_hf and self.hf_repo is not None:
        #     if torch.cuda.is_available():
        #         torch.cuda.empty_cache()
        #     self.load_checkpoint_from_hf()

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

            if self.global_step % self.cfg['vis_every'] == 0:
                self.unet.eval()
                self._save_vis(vis_batch, self.global_step)
                self.unet.train()

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
    print("Welcome to new trainer")
    cli = argparse.ArgumentParser()
    cli.add_argument('--manifest',    default=DEFAULT_CONFIG['manifest'])
    cli.add_argument('--output_dir',  default=DEFAULT_CONFIG['output_dir'])
    cli.add_argument('--model_id',    default=DEFAULT_CONFIG['model_id'])
    cli.add_argument('--batch_size',  type=int,   default=2)
    cli.add_argument('--num_steps',   type=int,   default=50_000)
    cli.add_argument('--lr',          type=float, default=1e-5)
    cli.add_argument('--grad_ckpt',   action='store_true', default=True)
    cli.add_argument('--fp16',        action='store_true', default=True)
    cli.add_argument('--use_8bit_adam', action='store_true', default=True)
    cli.add_argument('--resume',      default=None)
    cli.add_argument('--hf_repo_id', default=None, help='HF Hub repo ID (e.g., username/repo)')
    cli.add_argument('--hf_token', default=None, help='HF Hub token')
    cli.add_argument('--resume_from_hf', action='store_true', help='Resume from HF Hub latest checkpoint')
    args = cli.parse_args()

    config = vars(args)
    resume = config.pop('resume', None)
    auto_resume = config.pop('auto_resume', False)
    trainer = GenerationTrainer(config)
    if auto_resume:
        history = trainer.resume_or_start()
    elif resume:
        history = trainer.run(resume_from=resume)
    else:
        history = trainer.run()