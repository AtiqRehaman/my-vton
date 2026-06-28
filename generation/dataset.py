"""
generation/dataset.py — Dataset for Phase 4 generation model training

Reads the manifest produced by Phase 1 preprocessing (and updated by
Phase 2 with warped_cloth paths), and yields the full set of tensors
GenerationTrainer needs per sample: the ground-truth person image
(training target), all conditioning inputs (agnostic image/mask,
warped cloth, flat cloth, pose, densepose), and the 17-dim fit
features vector consumed by MeasurementEncoder.

This mirrors warping/dataset.py's loading conventions ([-1,1] RGB
tensors, [0,1] mask tensors) so the same manifest schema works
across both training stages without transformation differences
creeping in between phases.
"""

import json
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class GenerationDataset(Dataset):
    """
    Dataset for the Phase 4 diffusion UNet fine-tuning.

    Expects each manifest record to have:
        person          — ground truth, the training target
        agnostic_image  — person with clothing region greyed out
        agnostic_mask   — binary mask of the greyed region
        warped_cloth    — TPS-warped garment (Phase 2 output);
                          falls back to cloth_clean if Phase 2 hasn't
                          been run yet on a given record, so Phase 4
                          can still be smoke-tested end-to-end before
                          warping is fully trained
        cloth_clean     — flat garment, prepared on white background
        pose_img        — skeleton visualization
        densepose_iuv   — IUV body surface map
        fit_features    — OPTIONAL: precomputed 17-dim normalized
                          measurement vector. If absent, a neutral
                          default (Comfortable fit, average body) is
                          used — see fit.measurement_encoder.default_measurements
        prompt          — OPTIONAL: text prompt for CLIP conditioning.
                          If absent, a generic prompt is built from
                          cloth_type.
    """

    DEFAULT_PROMPT_TEMPLATE = "a photo of a person wearing {cloth_type} clothing"

    def __init__(
        self,
        manifest_path: str,
        target_h:   int = 512,
        target_w:   int = 384,
        cloth_type: str = 'upper',
        augment:    bool = False,
    ):
        with open(manifest_path) as f:
            self.records = json.load(f)

        self.H = target_h
        self.W = target_w
        self.cloth_type = cloth_type
        self.augment    = augment

        n_warped = sum(1 for r in self.records if r.get('warped_cloth'))
        print(f"[GenerationDataset] {len(self.records)} samples | "
              f"{n_warped} with real warped_cloth "
              f"({len(self.records)-n_warped} fall back to cloth_clean) | "
              f"{target_w}×{target_h} | augment={augment}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        wh = (self.W, self.H)

        person    = self._load_rgb(rec['person'],         wh, cv2.INTER_LINEAR)
        agnostic  = self._load_rgb(rec['agnostic_image'],  wh, cv2.INTER_LINEAR)
        cloth     = self._load_rgb(rec['cloth_clean'],     wh, cv2.INTER_LINEAR)
        pose      = self._load_rgb(rec['pose_img'],         wh, cv2.INTER_LINEAR)
        densepose = self._load_rgb(rec['densepose_iuv'],   wh, cv2.INTER_NEAREST)

        # Warped cloth — fall back to cloth_clean if Phase 2 hasn't
        # produced it yet for this record (lets Phase 4 be smoke-
        # tested without blocking on warping training completing).
        warped_path = rec.get('warped_cloth') or rec['cloth_clean']
        warped = self._load_rgb(warped_path, wh, cv2.INTER_LINEAR)

        mask = self._load_gray(rec['agnostic_mask'], wh, cv2.INTER_NEAREST)

        if self.augment and np.random.random() > 0.5:
            person, agnostic, cloth, pose, densepose, warped = [
                cv2.flip(x, 1) for x in [person, agnostic, cloth, pose, densepose, warped]
            ]
            mask = cv2.flip(mask, 1)

        fit_features = self._get_fit_features(rec)
        prompt       = rec.get('prompt') or self.DEFAULT_PROMPT_TEMPLATE.format(
            cloth_type=self.cloth_type
        )

        return {
            'person_img':    self._to_tensor(person),       # (3,H,W) [-1,1] — training target
            'agnostic_img':  self._to_tensor(agnostic),     # (3,H,W) [-1,1]
            'cloth_clean':   self._to_tensor(cloth),        # (3,H,W) [-1,1]
            'pose_img':      self._to_tensor(pose),         # (3,H,W) [-1,1]
            'densepose_img': self._to_tensor(densepose),    # (3,H,W) [-1,1]
            'warped_cloth':  self._to_tensor(warped),       # (3,H,W) [-1,1]
            'agnostic_mask': self._to_mask_tensor(mask),    # (1,H,W) [0,1]
            'fit_features':  fit_features,                   # (17,)
            'prompt':        prompt,
            'person_id':     rec.get('person_id', str(idx)),
            'cloth_id':      rec.get('cloth_id',  str(idx)),
        }

    def _get_fit_features(self, rec: dict) -> torch.Tensor:
        """
        Use precomputed fit_features if the manifest has them
        (e.g. from a future Phase that ties real body measurements
        per-sample); otherwise fall back to a neutral default so
        Phase 4 can train without requiring per-sample measurements
        to exist yet — the MeasurementEncoder is trained jointly and
        will still learn meaningful structure from whatever
        measurement diversity IS present in the manifest.
        """
        from fit.measurement_encoder import normalize_measurements, default_measurements

        if 'person_measurements' in rec and 'garment_measurements' in rec:
            return normalize_measurements(
                rec['person_measurements'], rec['garment_measurements'],
                rec.get('garment_type', self.cloth_type)
            )
        return default_measurements(rec.get('garment_type', self.cloth_type))

    # ── Helpers ──

    @staticmethod
    def _load_rgb(path: str, wh: tuple, interp: int) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Missing image: {path}")
        return cv2.cvtColor(cv2.resize(img, wh, interpolation=interp), cv2.COLOR_BGR2RGB)

    @staticmethod
    def _load_gray(path: str, wh: tuple, interp: int) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Missing: {path}")
        return cv2.resize(img, wh, interpolation=interp)

    @staticmethod
    def _to_tensor(rgb: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(rgb.astype(np.float32) / 255.0)
        return (t * 2.0 - 1.0).permute(2, 0, 1)   # [-1,1] CHW

    @staticmethod
    def _to_mask_tensor(gray: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(
            (gray > 127).astype(np.float32)
        ).unsqueeze(0)   # (1, H, W)