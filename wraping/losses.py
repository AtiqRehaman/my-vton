"""
warping/losses.py — Three training losses for the TPS warping module

─────────────────────────────────────────────────────────────────
WHY THREE LOSSES?
─────────────────────────────────────────────────────────────────

1. L1_loss(warped_cloth, gt_worn_region)
   The baseline: pixel-by-pixel accuracy.
   Problem alone: L1 doesn't care about textures/edges — a slightly
   blurry but shape-correct warp scores almost the same as a sharp one.

2. VGG perceptual loss (feature matching)
   Compares intermediate VGG-16 feature activations, not raw pixels.
   Why: VGG features encode textures, edges, and semantic structure.
   A warp that preserves stripe patterns scores high even if individual
   pixels don't exactly match. This is crucial for patterned garments.

3. Second-order grid smoothness loss (TPS regularization)
   Penalizes abrupt changes in the warp field (prevents fold artifacts).
   Why: Without regularization, the network CAN fold the cloth back onto
   itself or create extreme stretching — both produce nonsensical results.
   We penalize the discrete second derivative of the grid: if three
   adjacent control points move inconsistently, it costs.

Loss weights (from spec):
   total_loss = L1 + 0.1 × perceptual + 0.01 × second_order
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision.models import VGG16_Weights


# ─────────────────────────────────────────────────────────────────
# VGG Perceptual Loss
# ─────────────────────────────────────────────────────────────────

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG-16 feature maps.

    Computes L1 difference at 3 VGG feature levels:
      relu1_2  (layer 4 ) — low-level: edges, colours
      relu2_2  (layer 9 ) — mid-level: textures
      relu3_3  (layer 16) — high-level: patterns, shapes

    We use the same VGG-16 pretrained on ImageNet that generated the
    original perceptual loss paper (Johnson et al. 2016). The weights
    are frozen — we only extract features.

    Input normalization:
      VGG expects pixels in [0, 1] normalized with ImageNet stats.
      Our images are in [-1, 1] → we renormalize internally.
    """

    # VGG-16 layer indices for feature extraction
    FEATURE_LAYERS = {
        'relu1_2': 4,
        'relu2_2': 9,
        'relu3_3': 16,
    }

    # Loss weights per feature level (deeper = more weight)
    LAYER_WEIGHTS = {
        'relu1_2': 1.0,
        'relu2_2': 1.0,
        'relu3_3': 1.0,
    }

    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        features = vgg.features

        # Extract sub-networks up to each feature level
        self.slice1 = nn.Sequential(*list(features.children())[:5])   # up to relu1_2
        self.slice2 = nn.Sequential(*list(features.children())[5:10]) # relu1_2 to relu2_2
        self.slice3 = nn.Sequential(*list(features.children())[10:17])# relu2_2 to relu3_3

        # Freeze all VGG parameters — we only use it as a fixed feature extractor
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization (applied to [0,1] images)
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def normalize_imagenet(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [-1,1] to ImageNet-normalized [~-2.1, ~2.6]."""
        x = (x + 1.0) / 2.0          # [-1,1] → [0,1]
        return (x - self.mean) / self.std

    def get_features(self, x: torch.Tensor) -> dict:
        x = self.normalize_imagenet(x)
        f1 = self.slice1(x)
        f2 = self.slice2(f1)
        f3 = self.slice3(f2)
        return {'relu1_2': f1, 'relu2_2': f2, 'relu3_3': f3}

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (B, 3, H, W) in [-1, 1]
        Returns: scalar perceptual loss
        """
        pred_feats   = self.get_features(pred)
        target_feats = self.get_features(target.detach())  # no grad through target

        loss = torch.tensor(0.0, device=pred.device)
        for layer_name, weight in self.LAYER_WEIGHTS.items():
            loss = loss + weight * F.l1_loss(
                pred_feats[layer_name],
                target_feats[layer_name]
            )
        return loss


# ─────────────────────────────────────────────────────────────────
# Second-order grid smoothness loss (TPS regularization)
# ─────────────────────────────────────────────────────────────────

class SecondOrderGridLoss(nn.Module):
    """
    Penalizes non-smooth (folding/tearing) warp fields.

    Why second-order (not first-order)?
    First-order (gradient) penalty would suppress ALL deformation,
    including valid global shifts (sliding cloth left/right).
    Second-order penalizes only CHANGES IN THE GRADIENT — i.e., curves
    and kinks in the warp field. A globally shifted but smooth warp
    has zero second-order loss.

    Implementation:
    Compute finite differences of the grid in x and y directions.
    Then compute differences of those differences (second derivative).
    Penalize with L2 norm.

    grid: (B, H, W, 2) — the warp grid from TPSGridGenerator
    """

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        grid: (B, H, W, 2) where grid[..., 0]=x, grid[..., 1]=y
        Returns: scalar regularization loss
        """
        # First differences along H (vertical direction)
        dy = grid[:, 1:, :, :] - grid[:, :-1, :, :]    # (B, H-1, W, 2)
        # First differences along W (horizontal direction)
        dx = grid[:, :, 1:, :] - grid[:, :, :-1, :]    # (B, H, W-1, 2)

        # Second differences (change in the gradient)
        d2y = dy[:, 1:, :, :] - dy[:, :-1, :, :]       # (B, H-2, W, 2)
        d2x = dx[:, :, 1:, :] - dx[:, :, :-1, :]       # (B, H, W-2, 2)

        # L2 norm of second-order differences (mean over all positions)
        loss = (d2x ** 2).mean() + (d2y ** 2).mean()
        return loss


# ─────────────────────────────────────────────────────────────────
# Combined warping loss
# ─────────────────────────────────────────────────────────────────

class WarpingLoss(nn.Module):
    """
    Combined loss for TPS warping training.

    total = L1(warped, gt)
            + w_perc × perceptual(warped, gt)
            + w_smooth × second_order(grid)

    Spec weights:
        w_perc   = 0.1
        w_smooth = 0.01
    """

    def __init__(
        self,
        w_perceptual: float = 0.1,
        w_smooth:     float = 0.01,
    ):
        super().__init__()
        self.w_perceptual = w_perceptual
        self.w_smooth     = w_smooth

        self.perceptual_loss = VGGPerceptualLoss()
        self.smooth_loss     = SecondOrderGridLoss()

    def forward(
        self,
        warped_cloth: torch.Tensor,   # (B, 3, H, W) — model output
        gt_worn:      torch.Tensor,   # (B, 3, H, W) — ground truth worn region
        grid:         torch.Tensor,   # (B, H, W, 2) — warp grid from model
        cloth_mask:   torch.Tensor,   # (B, 1, H, W) — cloth binary mask
    ) -> dict:
        """
        Returns dict with individual losses and total for logging.
        All images expected in [-1, 1] range.

        cloth_mask is used to focus L1 and perceptual losses on the
        actual garment pixels (ignoring white background in the metric).
        """

        # Apply cloth mask: only compute pixel losses on garment region
        # This avoids penalizing the model for white background differences
        mask = cloth_mask.expand_as(warped_cloth)     # (B, 3, H, W)
        warped_masked = warped_cloth * mask
        gt_masked     = gt_worn * mask

        # L1 pixel loss
        l1 = F.l1_loss(warped_masked, gt_masked)

        # VGG perceptual loss (on full image — context matters for VGG features)
        perc = self.perceptual_loss(warped_cloth, gt_worn)

        # Grid smoothness regularization
        smooth = self.smooth_loss(grid)

        # Weighted total
        total = l1 + self.w_perceptual * perc + self.w_smooth * smooth

        return {
            'total':      total,
            'l1':         l1,
            'perceptual': perc,
            'smooth':     smooth,
        }


# ─────────────────────────────────────────────────────────────────
# Ground truth extraction utility
# ─────────────────────────────────────────────────────────────────

def extract_gt_worn_region(
    worn_person: torch.Tensor,
    parse_map:   torch.Tensor,
    cloth_type:  str = 'upper',
) -> torch.Tensor:
    """
    Extract the ground-truth worn garment region from the full person image.

    The training target for warping is NOT the full person image — it's
    just the clothing region. We use the parse map to isolate it.

    Args:
        worn_person: (B, 3, H, W) — original person photo (GT)
        parse_map:   (B, 1, H, W) — SCHP label map, values 0–17, long
        cloth_type:  'upper', 'lower', or 'overall'

    Returns:
        gt_worn: (B, 3, H, W) — worn region in [-1,1], background=0
    """
    # Labels to extract per cloth type (same as in agnostic_builder.py)
    CLOTH_LABELS = {
        'upper':   [4, 14, 15],           # upper-clothes, left-arm, right-arm
        'lower':   [5, 6, 8],             # skirt, pants, belt
        'overall': [4, 5, 6, 7, 14, 15],  # all clothing
    }
    labels = CLOTH_LABELS[cloth_type]

    # Build mask: 1 where parse_map matches any target label
    mask = torch.zeros_like(parse_map, dtype=torch.float32)
    for lbl in labels:
        mask = mask + (parse_map == lbl).float()
    mask = mask.clamp(0, 1)  # union of all target labels

    # Expand to 3 channels and apply
    mask3 = mask.expand_as(worn_person)
    gt_worn = worn_person * mask3  # background pixels become 0 (neutral grey in [-1,1])

    return gt_worn, mask