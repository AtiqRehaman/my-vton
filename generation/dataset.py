"""
generation/dataset.py — Dataset for generation model training

Reads the manifest updated by Phase 2 (which added warped_cloth paths)
and yields all inputs needed for the diffusion UNet training.

Key difference from WarpingDataset:
  - Loads warped_cloth (Phase 2 output) as an additional input
  - Also needs fit_features (17-dim measurement vector)
  - Since VITON-HD doesn't have real measurements, we synthesize them
    from garment size labels using standard size chart lookup
  - Returns worn_person as the TRAINING TARGET (the full try-on GT)
"""

import json
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# Standard size chart: maps garment size → approximate measurements (cm)
# Source: ISO 8559, ASTM D5585
SIZE_CHART = {
    'XS': {'garment_chest': 84, 'garment_waist': 66,  'garment_hip': 90},
    'S':  {'garment_chest': 88, 'garment_waist': 70,  'garment_hip': 94},
    'M':  {'garment_chest': 92, 'garment_waist': 74,  'garment_hip': 98},
    'L':  {'garment_chest': 96, 'garment_waist': 80,  'garment_hip': 102},
    'XL': {'garment_chest': 104,'garment_waist': 88,  'garment_hip': 110},
    '2XL':{'garment_chest': 112,'garment_waist': 96,  'garment_hip': 118},
}

# Default person measurements (average body — used when real measurements unavailable)
DEFAULT_PERSON = {
    'chest': 90, 'waist': 74, 'hip': 96,
    'shoulder_width': 40, 'height': 165, 'weight': 62,
}


class GenerationDataset(Dataset):
    """
    Dataset for generation model (Phase 4) training.

    Requires the manifest to have been updated by Phase 2 notebook
    Cell 8 (which adds the 'warped_cloth' key to each record).

    If warped_cloth is missing from a record, falls back to cloth_clean
    (degraded but won't crash — useful for quick debug runs).

    Outputs per sample:
        person_img      (3, H, W) [-1,1]  ← GT target for diffusion loss
        agnostic_img    (3, H, W) [-1,1]  ← person with clothes erased
        agnostic_mask   (1, H, W) [0,1]   ← clothing region mask
        warped_cloth    (3, H, W) [-1,1]  ← Phase 2 output
        cloth_clean     (3, H, W) [-1,1]  ← flat garment (additional cond)
        pose_img        (3, H, W) [-1,1]  ← skeleton visualization
        densepose_img   (3, H, W) [-1,1]  ← IUV surface map
        fit_features    (17,)     [0,1]   ← normalized measurements
        prompt          str               ← text description for CLIP
    """

    PROMPT_TEMPLATES = [
        "a photo of a person wearing {cloth_type} clothing",
        "front view of a person in {cloth_type} attire, high quality",
        "fashion photo, person wearing {cloth_type}, studio lighting",
        "realistic photo of someone in {cloth_type} clothes",
    ]

    def __init__(
        self,
        manifest_path: str,
        target_h:    int = 512,
        target_w:    int = 384,
        cloth_type:  str = 'upper',
        augment:     bool = False,
        garment_size: str = 'M',   # default size for synthetic measurements
    ):
        with open(manifest_path) as f:
            self.records = json.load(f)

        self.H = target_h
        self.W = target_w
        self.cloth_type   = cloth_type
        self.augment      = augment
        self.garment_size = garment_size

        # Check how many records have warped_cloth
        n_warped = sum(1 for r in self.records if r.get('warped_cloth'))
        print(f"[GenerationDataset] {len(self.records)} samples | "
              f"{n_warped} with warped_cloth | {target_w}×{target_h}")
        if n_warped < len(self.records):
            print(f"  WARNING: {len(self.records)-n_warped} samples missing "
                  f"warped_cloth — run Phase 2 Cell 8 first")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        wh = (self.W, self.H)

        # ── Load images ──
        person   = self._load_rgb(rec['person'],          wh, cv2.INTER_LINEAR)
        agnostic = self._load_rgb(rec['agnostic_image'],  wh, cv2.INTER_LINEAR)
        cloth    = self._load_rgb(rec['cloth_clean'],     wh, cv2.INTER_LINEAR)
        pose     = self._load_rgb(rec['pose_img'],         wh, cv2.INTER_LINEAR)
        dense    = self._load_rgb(rec['densepose_iuv'],   wh, cv2.INTER_NEAREST)

        # Warped cloth — fallback to cloth_clean if Phase 2 not run yet
        warped_path = rec.get('warped_cloth') or rec['cloth_clean']
        warped   = self._load_rgb(warped_path, wh, cv2.INTER_LINEAR)

        # Mask (single channel)
        mask = self._load_gray(rec['agnostic_mask'], wh, cv2.INTER_NEAREST)

        # ── Augmentation: horizontal flip ──
        if self.augment and np.random.random() > 0.5:
            person, agnostic, cloth, pose, dense, warped = [
                cv2.flip(x, 1) for x in [person, agnostic, cloth, pose, dense, warped]
            ]
            mask = cv2.flip(mask, 1)

        # ── Build fit features (17-dim) ──
        fit_features = self._build_fit_features(rec)

        # ── Build text prompt ──
        prompt = self._build_prompt(idx)

        return {
            'person_img':    self._to_tensor(person),            # (3,H,W) [-1,1]
            'agnostic_img':  self._to_tensor(agnostic),          # (3,H,W) [-1,1]
            'cloth_clean':   self._to_tensor(cloth),             # (3,H,W) [-1,1]
            'pose_img':      self._to_tensor(pose),              # (3,H,W) [-1,1]
            'densepose_img': self._to_tensor(dense),             # (3,H,W) [-1,1]
            'warped_cloth':  self._to_tensor(warped),            # (3,H,W) [-1,1]
            'agnostic_mask': self._to_mask_tensor(mask),         # (1,H,W) [0,1]
            'fit_features':  fit_features,                       # (17,) [0,1]
            'prompt':        prompt,
            'person_id':     rec.get('person_id', str(idx)),
            'cloth_id':      rec.get('cloth_id',  str(idx)),
        }

    def _build_fit_features(self, rec: dict) -> torch.Tensor:
        """
        Build normalized 17-dim measurement vector.

        In real deployment this comes from the user's body measurements.
        For VITON-HD training we synthesize from size chart — the model
        learns to use fit conditioning but isn't calibrated to specific bodies.
        This is fine for Phase 4 — the fit conditioning is validated in Phase 3.
        """
        from fit.measurement_encoder import normalize_measurements

        garment_meas = SIZE_CHART.get(self.garment_size, SIZE_CHART['M']).copy()
        garment_meas['garment_length']   = 65
        garment_meas['garment_shoulder'] = 40
        garment_meas['garment_type']     = self.cloth_type

        return normalize_measurements(
            DEFAULT_PERSON, garment_meas, self.cloth_type
        )

    def _build_prompt(self, idx: int) -> str:
        """Cycle through prompt templates for diversity."""
        template = self.PROMPT_TEMPLATES[idx % len(self.PROMPT_TEMPLATES)]
        return template.format(cloth_type=self.cloth_type)

    # ── Helpers ──

    @staticmethod
    def _load_rgb(path: str, wh: tuple, interp: int) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Missing: {path}")
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
