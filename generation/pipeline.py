"""
generation/pipeline.py — VAE encoding/decoding + full inference pipeline

─────────────────────────────────────────────────────────────────
WHAT THIS MODULE DOES
─────────────────────────────────────────────────────────────────
Three responsibilities:

1. LatentEncoder — wraps the frozen SD 1.5 VAE encoder.
   Converts pixel-space images [−1,1] → latent space [~N(0,1)].
   The VAE is ALWAYS frozen — we never train it.
   Scaling factor: 0.18215 (SD 1.5 standard)

2. ConditioningPreparer — prepares all 8 channel groups at
   latent resolution (H/8, W/8) ready for UNet concatenation.
   Handles encoding, downsampling, and fit embedding injection.

3. VTONPipeline — end-to-end inference pipeline.
   Input: person photo + cloth photo + measurements
   Output: person wearing the cloth
   Runs DDIM sampling (50 steps by default) for fast inference.

─────────────────────────────────────────────────────────────────
VAE LATENT SCALING
─────────────────────────────────────────────────────────────────
The VAE outputs raw latents z. Before feeding to UNet:
  z_scaled = z * 0.18215

This scale factor is hardcoded in SD 1.5 and must be applied
consistently during both training and inference. Forgetting it
produces completely wrong outputs (common bug).

During decode: z_decoded = vae.decode(z_scaled / 0.18215)
─────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np


VAE_SCALE_FACTOR = 0.18215


# ─────────────────────────────────────────────────────────────────
# Frozen VAE wrapper
# ─────────────────────────────────────────────────────────────────

class LatentEncoder(nn.Module):
    """
    Wraps the frozen SD 1.5 VAE for encoding images to latents.

    The VAE is NEVER trained — freeze it immediately after loading.
    Fine-tuning the VAE causes color/saturation drift and destroys
    the latent space structure the UNet was pretrained on.
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

        # Freeze all VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False

        print(f"VAE frozen | dtype={next(vae.parameters()).dtype}")

    @torch.no_grad()
    def encode(
        self,
        images: torch.Tensor,
        sample: bool = True,
        output_device=None,
    ) -> torch.Tensor:
        """
        Encode pixel images to scaled latents.

        Args:
            images: (B, 3, H, W) in [-1, 1] — moved to self.vae's device
                    automatically if it isn't already there, so callers
                    on a different device (e.g. a cross-device training
                    setup with frozen VAE on cuda:1) don't need to
                    pre-route the input themselves.
            sample: if True, sample from posterior; if False, use mode
                    During training: True (adds diversity)
                    During inference: False (deterministic) or True (slight variation)
            output_device: if set, the returned latents are moved to
                    this device before returning. None (default) keeps
                    the original behavior — latents stay on whatever
                    device self.vae produced them on. Pass the
                    TRAINING device here when self.vae lives on a
                    separate frozen_device, so the rest of the training
                    step doesn't need its own device-juggling.

        Returns:
            latents: (B, 4, H/8, W/8) scaled by VAE_SCALE_FACTOR
        """
        vae_device = next(self.vae.parameters()).device
        if images.device != vae_device:
            images = images.to(vae_device)

        posterior = self.vae.encode(images).latent_dist
        if sample:
            latents = posterior.sample()
        else:
            latents = posterior.mode()

        latents = latents * VAE_SCALE_FACTOR
        if output_device is not None and latents.device != torch.device(output_device):
            latents = latents.to(output_device)
        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, output_device=None) -> torch.Tensor:
        """
        Decode latents back to pixel images.

        Args:
            latents: (B, 4, H/8, W/8) scaled latents — moved to
                    self.vae's device automatically if needed.
            output_device: if set, the decoded image is moved to this
                    device before returning. None keeps prior behavior.

        Returns:
            images: (B, 3, H, W) in [-1, 1]
        """
        vae_device = next(self.vae.parameters()).device
        if latents.device != vae_device:
            latents = latents.to(vae_device)

        latents = latents / VAE_SCALE_FACTOR
        images = self.vae.decode(latents).sample
        images = images.clamp(-1, 1)
        if output_device is not None and images.device != torch.device(output_device):
            images = images.to(output_device)
        return images


# ─────────────────────────────────────────────────────────────────
# Conditioning preparer
# ─────────────────────────────────────────────────────────────────

class ConditioningPreparer:
    """
    Prepares all 8 UNet input channel groups from raw inputs.

    Call prepare() to get a dict of all tensors at latent resolution,
    ready to be concatenated by VTONUNet.forward().

    Spatial inputs that DON'T go through VAE (pose, densepose, mask):
      → bilinear/nearest downsample to (H/8, W/8)
      → normalize to [-1, 1] if not already

    Spatial inputs that DO go through VAE (agnostic, cloth, warped_cloth):
      → encode with LatentEncoder
    """

    def __init__(self, latent_encoder: LatentEncoder, fit_encoder):
        """
        fit_encoder: MeasurementEncoder instance (from fit/measurement_encoder.py)
        """
        self.latent_enc = latent_encoder
        self.fit_enc    = fit_encoder

    def prepare(
        self,
        agnostic_img:    torch.Tensor,   # (B, 3, H, W) [-1,1]
        agnostic_mask:   torch.Tensor,   # (B, 1, H, W) [0,1]
        warped_cloth:    torch.Tensor,   # (B, 3, H, W) [-1,1]
        cloth_clean:     torch.Tensor,   # (B, 3, H, W) [-1,1]
        pose_img:        torch.Tensor,   # (B, 3, H, W) [-1,1]
        densepose_img:   torch.Tensor,   # (B, 3, H, W) [-1,1]
        fit_features:    torch.Tensor,   # (B, 17) normalized measurements
        training:        bool = False,
    ) -> dict:
        """
        Returns dict with all channel groups at latent resolution.
        """
        B = agnostic_img.shape[0]
        H, W = agnostic_img.shape[2], agnostic_img.shape[3]
        lH, lW = H // 8, W // 8   # latent spatial size

        # ── Groups encoded through VAE ──
        agnostic_latent     = self.latent_enc.encode(agnostic_img,  sample=training)
        warped_cloth_latent = self.latent_enc.encode(warped_cloth,  sample=training)
        cloth_latent        = self.latent_enc.encode(cloth_clean,   sample=training)

        # ── Mask: nearest downsample (binary — no interpolation artifacts) ──
        mask_latent = F.interpolate(
            agnostic_mask, size=(lH, lW), mode='nearest'
        )  # (B, 1, lH, lW)

        # ── Pose: bilinear downsample (already in [-1,1] from skeleton renderer) ──
        pose_latent = F.interpolate(
            pose_img, size=(lH, lW), mode='bilinear', align_corners=False
        )  # (B, 3, lH, lW)

        # ── DensePose: nearest downsample (categorical IUV — avoid blending) ──
        densepose_latent = F.interpolate(
            densepose_img, size=(lH, lW), mode='nearest'
        )  # (B, 3, lH, lW)
        # Normalize IUV from [0,255] uint8 (loaded as float) to [-1,1]
        # densepose_img from WarpingDataset is already [-1,1] via _to_tensor()

        # ── Fit embedding: MeasurementEncoder ──
        fit_embedding = self.fit_enc(fit_features)   # (B, 4, lH, lW)

        return {
            'agnostic_latent':     agnostic_latent,      # (B, 4, lH, lW)
            'agnostic_mask':       mask_latent,           # (B, 1, lH, lW)
            'warped_cloth_latent': warped_cloth_latent,  # (B, 4, lH, lW)
            'cloth_latent':        cloth_latent,          # (B, 4, lH, lW)
            'pose_img':            pose_latent,           # (B, 3, lH, lW)
            'densepose_img':       densepose_latent,      # (B, 3, lH, lW)
            'fit_embedding':       fit_embedding,         # (B, 4, lH, lW)
        }


# ─────────────────────────────────────────────────────────────────
# Full inference pipeline
# ─────────────────────────────────────────────────────────────────

class VTONPipeline:
    """
    End-to-end VTON inference pipeline.

    Input:
        - person_img:      (1, 3, H, W) [-1,1]  person photo
        - cloth_img:       (1, 3, H, W) [-1,1]  flat garment
        - warped_cloth:    (1, 3, H, W) [-1,1]  pre-warped by Phase 2 model
        - agnostic_img:    (1, 3, H, W) [-1,1]  from Phase 1
        - agnostic_mask:   (1, 1, H, W) [0,1]   from Phase 1
        - pose_img:        (1, 3, H, W) [-1,1]  from Phase 1
        - densepose_img:   (1, 3, H, W) [-1,1]  from Phase 1
        - fit_features:    (1, 17)               normalized measurements
        - prompt:          str                   e.g. "a photo of a person wearing ..."

    Output:
        - result_img: (1, 3, H, W) [-1,1] — person wearing the garment

    DDIM sampling:
        - 50 steps (quality/speed tradeoff — adjustable)
        - guidance_scale=2.0 (low — we have strong spatial conditioning,
          don't need much CFG to avoid overfitting to text)
        - Both conditional AND unconditional pass needed for CFG
    """

    def __init__(
        self,
        vton_unet,            # VTONUNet
        vae,                  # AutoencoderKL (frozen)
        text_encoder,         # CLIPTextModel (frozen)
        tokenizer,            # CLIPTokenizer
        scheduler,            # DDIMScheduler
        fit_encoder,          # MeasurementEncoder
        device: str = 'cuda',
        dtype: torch.dtype = torch.float16,
    ):
        self.unet         = vton_unet.to(device)
        self.vae          = vae.to(device)
        self.text_encoder = text_encoder.to(device)
        self.tokenizer    = tokenizer
        self.scheduler    = scheduler
        self.fit_encoder  = fit_encoder.to(device)
        self.device       = device
        self.dtype        = dtype

        self.lat_enc   = LatentEncoder(vae)
        self.cond_prep = ConditioningPreparer(self.lat_enc, fit_encoder)

        # Freeze text encoder and VAE
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False

    @classmethod
    def from_pretrained(
        cls,
        unet_checkpoint: str,
        fit_checkpoint:  str,
        model_id: str = 'sd-legacy/stable-diffusion-v1-5',
        device: str = 'cuda',
    ) -> 'VTONPipeline':
        """Load all components from pretrained sources."""
        from diffusers import (
            AutoencoderKL, DDIMScheduler
        )
        from transformers import CLIPTextModel, CLIPTokenizer
        from generation.unet_modified import VTONUNet
        from fit.measurement_encoder import MeasurementEncoder

        dtype = torch.float16 if device == 'cuda' else torch.float32

        vae          = AutoencoderKL.from_pretrained(model_id, subfolder='vae',
                                                      torch_dtype=dtype)
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder='text_encoder',
                                                      torch_dtype=dtype)
        tokenizer    = CLIPTokenizer.from_pretrained(model_id, subfolder='tokenizer')
        scheduler    = DDIMScheduler.from_pretrained(model_id, subfolder='scheduler')

        unet         = VTONUNet.from_checkpoint(unet_checkpoint, model_id, device)
        fit_encoder  = MeasurementEncoder(target_h=512, target_w=384)
        fit_encoder.load_state_dict(
            torch.load(fit_checkpoint, map_location=device)
        )

        return cls(unet, vae, text_encoder, tokenizer, scheduler,
                   fit_encoder, device, dtype)

    @torch.no_grad()
    def encode_prompt(self, prompt: str, negative_prompt: str = '') -> tuple:
        """CLIP text encoding with classifier-free guidance."""
        def tokenize_and_encode(text):
            tokens = self.tokenizer(
                text, padding='max_length', max_length=77,
                truncation=True, return_tensors='pt'
            ).input_ids.to(self.device)
            return self.text_encoder(tokens)[0]   # (1, 77, 768)

        cond_embeds   = tokenize_and_encode(prompt)
        uncond_embeds = tokenize_and_encode(negative_prompt)
        return cond_embeds, uncond_embeds

    @torch.no_grad()
    def __call__(
        self,
        person_img:     torch.Tensor,
        cloth_img:      torch.Tensor,
        warped_cloth:   torch.Tensor,
        agnostic_img:   torch.Tensor,
        agnostic_mask:  torch.Tensor,
        pose_img:       torch.Tensor,
        densepose_img:  torch.Tensor,
        fit_features:   torch.Tensor,
        prompt:         str = 'a photo of a person wearing clothes',
        negative_prompt:str = 'worst quality, low quality, artifacts',
        num_steps:      int = 50,
        guidance_scale: float = 2.0,
        seed:           Optional[int] = None,
    ) -> torch.Tensor:
        """Run full DDIM sampling. Returns (1, 3, H, W) in [-1, 1]."""
        B = person_img.shape[0]
        H, W = person_img.shape[2], person_img.shape[3]
        lH, lW = H // 8, W // 8

        # Cast all inputs to pipeline dtype
        def cast(t): return t.to(device=self.device, dtype=self.dtype)
        agnostic_img  = cast(agnostic_img)
        agnostic_mask = cast(agnostic_mask)
        warped_cloth  = cast(warped_cloth)
        cloth_img     = cast(cloth_img)
        pose_img      = cast(pose_img)
        densepose_img = cast(densepose_img)
        fit_features  = cast(fit_features)

        # ── Prepare conditioning channels (computed once, reused every step) ──
        cond = self.cond_prep.prepare(
            agnostic_img, agnostic_mask, warped_cloth, cloth_img,
            pose_img, densepose_img, fit_features, training=False
        )

        # ── Text conditioning ──
        prompt_embeds, negative_embeds = self.encode_prompt(prompt, negative_prompt)
        # For CFG: cat [uncond, cond] → single batch pass
        text_embeds_cfg = torch.cat([negative_embeds, prompt_embeds], dim=0)

        # ── Initialize latent noise ──
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        latents = torch.randn(
            (B, 4, lH, lW), generator=generator,
            device=self.device, dtype=self.dtype
        )
        latents = latents * self.scheduler.init_noise_sigma

        # ── DDIM sampling loop ──
        self.scheduler.set_timesteps(num_steps, device=self.device)
        self.unet.eval()

        for t in self.scheduler.timesteps:
            # Duplicate latents for CFG: [uncond_latents, cond_latents]
            latent_input = torch.cat([latents] * 2)
            latent_input = self.scheduler.scale_model_input(latent_input, t)
            timestep     = torch.tensor([t] * (2 * B), device=self.device)

            # Duplicate all conditioning for CFG batch
            def dup(x): return torch.cat([x] * 2)

            noise_pred = self.unet(
                noisy_latent          = latent_input,
                timestep              = timestep,
                encoder_hidden_states = text_embeds_cfg,
                agnostic_latent       = dup(cond['agnostic_latent']),
                agnostic_mask         = dup(cond['agnostic_mask']),
                warped_cloth_latent   = dup(cond['warped_cloth_latent']),
                cloth_latent          = dup(cond['cloth_latent']),
                pose_img              = dup(cond['pose_img']),
                densepose_img         = dup(cond['densepose_img']),
                fit_embedding         = dup(cond['fit_embedding']),
            )

            # CFG: noise_pred = uncond + scale * (cond - uncond)
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            # DDIM step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # ── Decode to pixel space ──
        result = self.lat_enc.decode(latents)   # (B, 3, H, W) [-1,1]
        return result


# ─────────────────────────────────────────────────────────────────
# EMA (Exponential Moving Average) — for stable inference weights
# ─────────────────────────────────────────────────────────────────

class EMAModel:
    """
    Maintains an exponential moving average of model parameters.

    EMA provides smoother, more stable weights for inference
    without affecting the training dynamics of the base model.

    decay=0.9999 means EMA updates very slowly — 10k+ steps to converge.
    Typical usage:
        ema = EMAModel(model.parameters(), decay=0.9999)
        for batch in dataloader:
            loss.backward()
            optimizer.step()
            ema.step(model.parameters())  ← after every optimizer step

        # For inference: temporarily apply EMA weights
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        output = model(...)
        ema.restore(model.parameters())  ← restore training weights
    """

    def __init__(self, parameters, decay: float = 0.9999):
        self.decay = decay
        self.shadow = [p.clone().detach() for p in parameters]
        self._stored = None

    def step(self, parameters):
        with torch.no_grad():
            for s, p in zip(self.shadow, parameters):
                s.sub_((1 - self.decay) * (s - p.data))

    def copy_to(self, parameters):
        for s, p in zip(self.shadow, parameters):
            p.data.copy_(s)

    def store(self, parameters):
        self._stored = [p.clone() for p in parameters]

    def restore(self, parameters):
        assert self._stored is not None, "Must call store() before restore()"
        for s, p in zip(self._stored, parameters):
            p.data.copy_(s)
        self._stored = None

    def state_dict(self) -> dict:
        return {'shadow': [s.cpu() for s in self.shadow], 'decay': self.decay}

    def load_state_dict(self, state: dict):
        self.decay  = state['decay']
        self.shadow = [s.to(self.shadow[0].device) for s in state['shadow']]