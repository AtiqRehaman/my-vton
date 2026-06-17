"""
warping/dataset.py — Dataset loader for the TPS warping training stage

Reads the manifest produced by preprocessing/preprocess_dataset.py and
yields the 5 tensors needed for warping training:

  cloth_clean   (3, H, W)  — flat garment (model input A)
  agnostic_img  (3, H, W)  — person with clothes erased (model input B part 1)
  pose_img      (3, H, W)  — skeleton (model input B part 2)
  densepose_iuv (3, H, W)  — IUV surface (model input B part 3)
  cloth_mask    (1, H, W)  — garment binary mask (for masked L1 loss)
  worn_person   (3, H, W)  — GROUND TRUTH: full person photo
  parse_map     (1, H, W)  — label map (to extract gt_worn_region in loss)
"""

import json
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class WarpingDataset(Dataset):
    """
    Dataset for TPS warping training.

    Expects the manifest JSON produced by preprocess_dataset.py.
    Each record must have keys:
      person, cloth_clean, cloth_mask, agnostic_image,
      pose_img, densepose_iuv, parse_map

    The worn_person (ground truth) IS the original person image —
    because VITON-HD is paired: person is already wearing cloth.
    We extract the clothing region from worn_person using parse_map
    in the loss function.
    """

    def __init__(
        self,
        manifest_path: str,
        target_h: int = 512,
        target_w: int = 384,
        cloth_type: str = 'upper',
        augment: bool = False,
    ):
        with open(manifest_path) as f:
            self.records = json.load(f)

        self.H = target_h
        self.W = target_w
        self.cloth_type = cloth_type
        self.augment = augment
        print(f"[WarpingDataset] {len(self.records)} samples | "
              f"{target_w}×{target_h} | augment={augment}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        wh = (self.W, self.H)

        # ── Load images ──
        cloth     = self._load_rgb(rec['cloth_clean'],    wh, interp=cv2.INTER_LINEAR)
        agnostic  = self._load_rgb(rec['agnostic_image'], wh, interp=cv2.INTER_LINEAR)
        pose      = self._load_rgb(rec['pose_img'],        wh, interp=cv2.INTER_LINEAR)
        densepose = self._load_rgb(rec['densepose_iuv'],  wh, interp=cv2.INTER_NEAREST)
        worn      = self._load_rgb(rec['person'],          wh, interp=cv2.INTER_LINEAR)
        cloth_mask = self._load_gray(rec['cloth_mask'],   wh, interp=cv2.INTER_NEAREST)
        parse_map  = self._load_gray(rec['parse_map'],    wh, interp=cv2.INTER_NEAREST)

        # ── Augmentation: horizontal flip only ──
        # Note: DensePose IUV left/right body part indices are NOT swapped
        # on flip — this is a simplification. For exact correctness you
        # would remap the I channel. For warping training it's acceptable.
        if self.augment and np.random.random() > 0.5:
            cloth      = cv2.flip(cloth,      1)
            agnostic   = cv2.flip(agnostic,   1)
            pose       = cv2.flip(pose,       1)
            densepose  = cv2.flip(densepose,  1)
            worn       = cv2.flip(worn,       1)
            cloth_mask = cv2.flip(cloth_mask, 1)
            parse_map  = cv2.flip(parse_map,  1)

        return {
            'cloth':          self._to_tensor(cloth),           # (3, H, W) [-1,1]
            'agnostic':       self._to_tensor(agnostic),        # (3, H, W) [-1,1]
            'pose':           self._to_tensor(pose),            # (3, H, W) [-1,1]
            'densepose':      self._to_tensor(densepose),       # (3, H, W) [-1,1]
            'worn_person':    self._to_tensor(worn),            # (3, H, W) [-1,1]
            'cloth_mask':     self._to_mask_tensor(cloth_mask), # (1, H, W) [0,1]
            'parse_map':      torch.from_numpy(parse_map.astype(np.int64)).unsqueeze(0),  # (1,H,W)
            'cloth_type':     self.cloth_type,
            'person_id':      rec.get('person_id', str(idx)),
            'cloth_id':       rec.get('cloth_id',  str(idx)),
        }

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _load_rgb(path: str, wh: tuple, interp: int) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Missing image: {path}")
        img = cv2.resize(img, wh, interpolation=interp)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _load_gray(path: str, wh: tuple, interp: int) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Missing image: {path}")
        return cv2.resize(img, wh, interpolation=interp)

    @staticmethod
    def _to_tensor(rgb: np.ndarray) -> torch.Tensor:
        """uint8 RGB [0,255] → float32 [-1,1] CHW tensor."""
        t = torch.from_numpy(rgb.astype(np.float32) / 255.0)   # HWC [0,1]
        t = t * 2.0 - 1.0                                       # [0,1] → [-1,1]
        return t.permute(2, 0, 1)                               # HWC → CHW

    @staticmethod
    def _to_mask_tensor(gray: np.ndarray) -> torch.Tensor:
        """uint8 mask [0,255] → float32 [0,1] 1HW tensor."""
        t = torch.from_numpy((gray > 127).astype(np.float32))
        return t.unsqueeze(0)   # (1, H, W)