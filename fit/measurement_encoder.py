"""
fit/measurement_encoder.py — MLP that encodes body/garment measurements
                              into a spatial fit embedding tensor for the UNet

─────────────────────────────────────────────────────────────────
WHY THIS MODULE?
─────────────────────────────────────────────────────────────────
The generation model (Group C) needs to know whether to generate:
  • Tight wrinkles and visible body contours  (negative ease)
  • Clean flat fabric                         (zero ease)
  • Drape, folds, extra width                 (positive ease)
  • Billowing fabric pooling at shoulders     (large positive ease)

We encode this as a spatial tensor [B, 4, H/8, W/8] that gets
concatenated into the UNet as "channel group 8" (see spec Group C).

Architecture (from spec):
  Linear(17 → 64) → GELU → Linear(64 → 128) → GELU → Linear(128 → 256)
  Reshape(256 → 4 × 8 × 8) → Upsample to (4, latent_H, latent_W)

The 17 input features are:
  Person: chest, waist, hip, shoulder_width, height, weight   (6)
  Garment: garment_chest, garment_waist, garment_hip,
           garment_length, garment_shoulder                   (5)
  Ease: ease_chest, ease_waist, ease_hip                      (3)
  Garment type (one-hot): [upper, lower, overall]             (3)
  Total: 17

All scalar inputs are normalized to [0, 1] range before feeding in.
The normalization ranges are set based on typical human measurements.

─────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# Feature normalization ranges (for [0,1] scaling)
# ─────────────────────────────────────────────────────────────────

MEASUREMENT_RANGES = {
    # feature_name: (min_cm, max_cm)
    'chest':            (60,  150),
    'waist':            (50,  140),
    'hip':              (60,  155),
    'shoulder_width':   (30,   60),
    'height':           (140, 210),
    'weight':           (40,  160),    # kg
    'garment_chest':    (60,  160),
    'garment_waist':    (50,  150),
    'garment_hip':      (60,  165),
    'garment_length':   (40,  130),
    'garment_shoulder': (30,   60),
    'ease_chest':       (-10,  25),    # ease range: far outside → very oversized
    'ease_waist':       (-10,  22),
    'ease_hip':         (-10,  26),
}

GARMENT_TYPES = ['upper', 'lower', 'overall']


def normalize_measurements(
    person_measurements: dict,
    garment_measurements: dict,
    garment_type: str = 'upper',
) -> torch.Tensor:
    """
    Convert raw measurements to a normalized 17-dim float tensor.

    Args:
        person_measurements:  dict with keys matching MEASUREMENT_RANGES person keys
        garment_measurements: dict with keys matching MEASUREMENT_RANGES garment keys
        garment_type: 'upper', 'lower', or 'overall'

    Returns:
        features: (17,) float32 tensor, values in [0, 1]
    """

    def norm(val, key):
        lo, hi = MEASUREMENT_RANGES[key]
        return float(max(0.0, min(1.0, (val - lo) / (hi - lo))))

    # Compute ease values
    ease_chest = (garment_measurements.get('garment_chest', 94) -
                  person_measurements.get('chest', 90))
    ease_waist = (garment_measurements.get('garment_waist', 77) -
                  person_measurements.get('waist', 74))
    ease_hip   = (garment_measurements.get('garment_hip', 100) -
                  person_measurements.get('hip', 96))

    # Build feature vector (order must match model's expected input)
    features = [
        # Person measurements (6)
        norm(person_measurements.get('chest',          90),  'chest'),
        norm(person_measurements.get('waist',          74),  'waist'),
        norm(person_measurements.get('hip',            96),  'hip'),
        norm(person_measurements.get('shoulder_width', 40),  'shoulder_width'),
        norm(person_measurements.get('height',        165),  'height'),
        norm(person_measurements.get('weight',         60),  'weight'),
        # Garment measurements (5)
        norm(garment_measurements.get('garment_chest',    94),  'garment_chest'),
        norm(garment_measurements.get('garment_waist',    77),  'garment_waist'),
        norm(garment_measurements.get('garment_hip',     100),  'garment_hip'),
        norm(garment_measurements.get('garment_length',   65),  'garment_length'),
        norm(garment_measurements.get('garment_shoulder', 40),  'garment_shoulder'),
        # Ease (3)
        norm(ease_chest, 'ease_chest'),
        norm(ease_waist, 'ease_waist'),
        norm(ease_hip,   'ease_hip'),
        # Garment type one-hot (3)
        1.0 if garment_type == 'upper'   else 0.0,
        1.0 if garment_type == 'lower'   else 0.0,
        1.0 if garment_type == 'overall' else 0.0,
    ]

    return torch.tensor(features, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────
# Measurement Encoder MLP
# ─────────────────────────────────────────────────────────────────

class MeasurementEncoder(nn.Module):
    """
    MLP that encodes 17 measurement features → spatial fit embedding.

    Architecture (from spec):
        Linear(17 → 64) → GELU → Dropout(0.1)
        Linear(64 → 128) → GELU → Dropout(0.1)
        Linear(128 → 256)
        Reshape: 256 → (4, 8, 8)
        Upsample: (4, 8, 8) → (4, latent_H, latent_W)

    Where latent_H = target_H / 8, latent_W = target_W / 8
    For 512×384: latent = 64×48

    The output (B, 4, latent_H, latent_W) is concatenated into the
    UNet input as channel group 8.

    Training: jointly with the generation model (Group C).
    Parameters: very small (~21k) — won't significantly slow training.
    """

    INPUT_DIM  = 17
    LATENT_DIM = 256   # 4 × 8 × 8
    OUT_CH     = 4     # matches latent space channels

    def __init__(
        self,
        target_h: int = 512,
        target_w: int = 384,
        dropout:  float = 0.1,
    ):
        super().__init__()
        # Latent spatial size (UNet operates at 1/8 of image resolution)
        self.latent_h = target_h // 8   # 64 for 512
        self.latent_w = target_w // 8   # 48 for 384

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.INPUT_DIM, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.LATENT_DIM),
        )

        # Initialize weights: small random → model starts with near-zero
        # fit influence, learning to use it progressively during training
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, 17) normalized measurement vector
        Returns:
            fit_embedding: (B, 4, latent_H, latent_W)
        """
        B = features.shape[0]

        # MLP: (B, 17) → (B, 256)
        x = self.mlp(features)

        # Reshape to (B, 4, 8, 8) — 256 = 4×8×8
        x = x.view(B, self.OUT_CH, 8, 8)

        # Upsample to (B, 4, latent_H, latent_W)
        # bilinear upsampling produces smoother spatial embeddings than nearest
        x = F.interpolate(
            x,
            size=(self.latent_h, self.latent_w),
            mode='bilinear',
            align_corners=False,
        )

        return x   # (B, 4, 64, 48) for 512×384 images


# ─────────────────────────────────────────────────────────────────
# Default measurement factory (for inference when real measurements unavailable)
# ─────────────────────────────────────────────────────────────────

def default_measurements(garment_type: str = 'upper') -> torch.Tensor:
    """
    Returns a neutral fit embedding (Comfortable fit, standard proportions).
    Used during inference when the user doesn't provide measurements.
    """
    person = {
        'chest': 90, 'waist': 74, 'hip': 96,
        'shoulder_width': 40, 'height': 165, 'weight': 62,
    }
    garment = {
        'garment_chest': 94, 'garment_waist': 78, 'garment_hip': 100,
        'garment_length': 65, 'garment_shoulder': 40,
    }
    return normalize_measurements(person, garment, garment_type)


# ─────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    encoder = MeasurementEncoder(target_h=512, target_w=384)
    total = sum(p.numel() for p in encoder.parameters())
    print(f"MeasurementEncoder parameters: {total:,}")   # ~21k

    B = 4
    features = torch.rand(B, 17)
    embed = encoder(features)
    print(f"Input:  {features.shape}")    # (4, 17)
    print(f"Output: {embed.shape}")       # (4, 4, 64, 48)

    # Test with default measurements
    default = default_measurements('upper')
    print(f"Default features (Comfortable fit): {default.numpy().round(3)}")
    embed_single = encoder(default.unsqueeze(0))
    print(f"Default embedding shape: {embed_single.shape}")  # (1, 4, 64, 48)
