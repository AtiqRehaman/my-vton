"""
fit/measurement_encoder.py — Converts body/garment measurements into
a spatial fit-conditioning embedding for the Phase 4 generation UNet.

This is the bridge between Phase 3 (fit classification) and Phase 4
(generation): the same 17-dim measurement feature vector format used
to train the RandomForest fit classifier is reused here, but instead
of predicting a discrete fit label, a small MLP projects it into a
spatial tensor that gets concatenated as one of the UNet's 27 input
channel groups (the "fit_embedding" group — 4 channels at latent
resolution), so the diffusion model can condition its generation on
how loose/tight the garment should appear.

─────────────────────────────────────────────────────────────────
17-DIM FEATURE VECTOR LAYOUT (must match across all consumers)
─────────────────────────────────────────────────────────────────
  [0]  person chest    (normalized 0-1, range 60-150 cm)
  [1]  person waist     (normalized 0-1, range 50-140 cm)
  [2]  person hip        (normalized 0-1, range 60-155 cm)
  [3]  person shoulder_width (normalized 0-1, range 30-60 cm)
  [4]  person height       (normalized 0-1, range 140-210 cm)
  [5]  person weight        (normalized 0-1, range 40-160 kg)
  [6]  garment_chest          (normalized 0-1, range 60-160 cm)
  [7]  garment_waist           (normalized 0-1, range 50-150 cm)
  [8]  garment_hip               (normalized 0-1, range 60-165 cm)
  [9]  garment_length              (normalized 0-1, range 40-130 cm)
  [10] garment_shoulder               (normalized 0-1, range 30-60 cm)
  [11] ease_chest = garment_chest - person_chest (normalized, range -10..25)
  [12] ease_waist  = garment_waist - person_waist  (normalized, range -10..22)
  [13] ease_hip      = garment_hip - person_hip      (normalized, range -10..26)
  [14] garment_type == 'upper'   (one-hot)
  [15] garment_type == 'lower'   (one-hot)
  [16] garment_type == 'overall' (one-hot)
"""

import torch
import torch.nn as nn


# Normalization ranges — same convention as fit/ease_calculator.py's
# real-world cm/kg scales, kept local here since measurement_encoder
# must work even if ease_calculator isn't imported (e.g. at inference
# time in a stripped-down deployment image).
RANGES = {
    'chest':            (60,  150),
    'waist':            (50,  140),
    'hip':              (60,  155),
    'shoulder_width':   (30,   60),
    'height':           (140, 210),
    'weight':           (40,  160),
    'garment_chest':    (60,  160),
    'garment_waist':    (50,  150),
    'garment_hip':      (60,  165),
    'garment_length':   (40,  130),
    'garment_shoulder': (30,   60),
    'ease_chest':       (-10,  25),
    'ease_waist':       (-10,  22),
    'ease_hip':         (-10,  26),
}

GARMENT_TYPES = ['upper', 'lower', 'overall']

# Neutral default measurements — average adult body, garment with
# moderate positive ease (lands in "Comfortable" per
# ease_calculator.py's tolerance tables). Used whenever real
# per-sample measurements aren't available, so Phase 4 training can
# proceed without requiring every manifest record to carry real
# measurement data.
DEFAULT_PERSON = {
    'chest': 90, 'waist': 74, 'hip': 96,
    'shoulder_width': 40, 'height': 165, 'weight': 62,
}
DEFAULT_GARMENT = {
    'garment_chest': 94, 'garment_waist': 78, 'garment_hip': 100,
    'garment_length': 65, 'garment_shoulder': 40,
}


def _norm(value: float, key: str) -> float:
    lo, hi = RANGES[key]
    return float(max(0.0, min(1.0, (value - lo) / (hi - lo))))


def normalize_measurements(
    person_measurements: dict,
    garment_measurements: dict,
    garment_type: str = 'upper',
) -> torch.Tensor:
    """
    Convert raw person/garment measurement dicts into the normalized
    17-dim feature tensor described in the module docstring.

    person_measurements keys:  chest, waist, hip, shoulder_width,
                                height, weight
    garment_measurements keys: garment_chest, garment_waist,
                                garment_hip, garment_length,
                                garment_shoulder
    garment_type: 'upper', 'lower', or 'overall'
    """
    p = {**DEFAULT_PERSON,  **person_measurements}
    g = {**DEFAULT_GARMENT, **garment_measurements}

    ease_chest = g['garment_chest'] - p['chest']
    ease_waist = g['garment_waist'] - p['waist']
    ease_hip   = g['garment_hip']   - p['hip']

    features = [
        _norm(p['chest'],            'chest'),
        _norm(p['waist'],            'waist'),
        _norm(p['hip'],              'hip'),
        _norm(p['shoulder_width'],   'shoulder_width'),
        _norm(p['height'],           'height'),
        _norm(p['weight'],           'weight'),
        _norm(g['garment_chest'],    'garment_chest'),
        _norm(g['garment_waist'],    'garment_waist'),
        _norm(g['garment_hip'],      'garment_hip'),
        _norm(g['garment_length'],   'garment_length'),
        _norm(g['garment_shoulder'], 'garment_shoulder'),
        _norm(ease_chest, 'ease_chest'),
        _norm(ease_waist, 'ease_waist'),
        _norm(ease_hip,   'ease_hip'),
        1.0 if garment_type == 'upper'   else 0.0,
        1.0 if garment_type == 'lower'   else 0.0,
        1.0 if garment_type == 'overall' else 0.0,
    ]
    return torch.tensor(features, dtype=torch.float32)


def default_measurements(garment_type: str = 'upper') -> torch.Tensor:
    """
    Returns the neutral fit embedding's input features (Comfortable
    fit, average adult body) — used whenever per-sample measurements
    aren't available in the manifest.
    """
    return normalize_measurements(DEFAULT_PERSON, DEFAULT_GARMENT, garment_type)


class MeasurementEncoder(nn.Module):
    """
    Small MLP that projects the 17-dim measurement feature vector
    into a spatial (4, latent_H, latent_W) tensor for concatenation
    into the Phase 4 UNet's input channels.

    Architecture: 17 → 64 → 256 → reshape to (4, 8, 8) → upsample to
    (4, latent_H, latent_W). Kept deliberately small — this is a
    conditioning signal, not a heavy feature extractor — so it adds
    negligible parameter/VRAM cost on top of the UNet fine-tune.
    """

    def __init__(self, target_h: int = 512, target_w: int = 384):
        super().__init__()
        self.latent_h = target_h // 8
        self.latent_w = target_w // 8

        self.net = nn.Sequential(
            nn.Linear(17, 64),
            nn.SiLU(),
            nn.Linear(64, 256),
            nn.SiLU(),
        )
        # 256 = 4 channels * 8 * 8 spatial seed, upsampled to latent res
        self.seed_h, self.seed_w = 8, 8
        self.seed_channels = 4

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (B, 17) normalized measurement vector
        returns:  (B, 4, latent_h, latent_w)
        """
        B = features.shape[0]
        x = self.net(features)                                    # (B, 256)
        x = x.view(B, self.seed_channels, self.seed_h, self.seed_w)  # (B,4,8,8)
        x = nn.functional.interpolate(
            x, size=(self.latent_h, self.latent_w),
            mode='bilinear', align_corners=False,
        )
        return x