"""
warping/model.py — TPS (Thin Plate Spline) Garment Warping Network

─────────────────────────────────────────────────────────────────
WHY TPS WARPING?
─────────────────────────────────────────────────────────────────
The flat garment photo needs to be deformed to match the person's
body pose before the diffusion model runs. Without pre-warping:
  • The diffusion model must both place AND deform the cloth
  • Fine texture patterns (stripes, logos, prints) get destroyed
  • Training converges much slower

TPS warping is a classical geometric transformation:
  • We predict N control points and their offsets (Δx, Δy)
  • A smooth warp field is interpolated from those control points
  • torch.nn.functional.grid_sample() applies the warp efficiently

Why not learned flow / optical flow?
  • TPS is more regularizable — the thin plate spline energy term
    prevents impossible folds (cloth can't fold onto itself)
  • CP-VTON, VITON-HD, and HR-VTON all use TPS for the first stage

─────────────────────────────────────────────────────────────────
ARCHITECTURE OVERVIEW
─────────────────────────────────────────────────────────────────

Input A: cloth_clean   (3, H, W)  — flat garment
Input B: agnostic_img  (3, H, W)  — person body with clothes erased
         pose_img      (3, H, W)  — skeleton visualization
         densepose_iuv (3, H, W)  — 3D body surface map
         → B is concatenated → (12, H, W) to give full body context

FeatureExtractor_A (VGG-like) → cloth_features     [512, H/16, W/16]
FeatureExtractor_B (VGG-like) → body_features      [512, H/16, W/16]

CorrelationLayer:
  For each spatial position in cloth_features, compute dot-product
  similarity against ALL positions in body_features
  Output: correlation_map [H/16 × W/16, H/16 × W/16]
  This tells the network: "cloth pixel at (r,c) matches body at (r',c')"

TPSRegressor (FC layers):
  correlation_map → flatten → FC → FC → 2N control point offsets
  N = 5×5 = 25 control points by default

TPS Grid Generator:
  control_points + offsets → smooth warp grid (H, W, 2)

grid_sample:
  Apply warp grid to cloth_clean → warped_cloth (3, H, W)

─────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


# ─────────────────────────────────────────────────────────────────
# Building blocks: VGG-style feature extractor
# ─────────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    """Conv2d + BatchNorm + ReLU — standard building block."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, pad: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FeatureExtractor(nn.Module):
    """
    VGG-style feature extractor.
    Progressively downsamples input by 16× while expanding channels.

    Input:  (B, in_channels, H, W)
    Output: (B, 512, H/16, W/16)

    Architecture mirrors VGG-11 but lighter (no FC layers needed
    since we only need the spatial feature maps).
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: (B, in_ch, H, W) → (B, 64, H/2, W/2)
            ConvBnRelu(in_channels, 64, 3, 1, 1),
            ConvBnRelu(64, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),                     # /2

            # Block 2: → (B, 128, H/4, W/4)
            ConvBnRelu(64, 128, 3, 1, 1),
            ConvBnRelu(128, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),                     # /4

            # Block 3: → (B, 256, H/8, W/8)
            ConvBnRelu(128, 256, 3, 1, 1),
            ConvBnRelu(256, 256, 3, 1, 1),
            ConvBnRelu(256, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),                     # /8

            # Block 4: → (B, 512, H/16, W/16)
            ConvBnRelu(256, 512, 3, 1, 1),
            ConvBnRelu(512, 512, 3, 1, 1),
            ConvBnRelu(512, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),                     # /16
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


# ─────────────────────────────────────────────────────────────────
# Correlation layer
# ─────────────────────────────────────────────────────────────────

class CorrelationLayer(nn.Module):
    """
    Computes normalized cross-correlation between two feature maps.

    For each spatial position (h, w) in feat_A, compute the dot-product
    similarity with ALL positions in feat_B. This produces a 4D tensor
    that encodes "how much does every cloth location look like every
    body location?"

    Input:
        feat_A: (B, C, Ha, Wa) — cloth features
        feat_B: (B, C, Hb, Wb) — body features

    Output:
        corr:   (B, Ha*Wa, Hb*Wb) — correlation scores

    Note: both inputs are L2-normalized along the channel dim before
    correlation so similarity values stay in [-1, 1].
    """

    def forward(self, feat_A: torch.Tensor, feat_B: torch.Tensor) -> torch.Tensor:
        B, C, Ha, Wa = feat_A.shape
        _,  _, Hb, Wb = feat_B.shape

        # L2-normalize along channel dimension
        feat_A = F.normalize(feat_A, p=2, dim=1)   # (B, C, Ha, Wa)
        feat_B = F.normalize(feat_B, p=2, dim=1)   # (B, C, Hb, Wb)

        # Reshape for batch matrix multiply
        # feat_A: (B, Ha*Wa, C)  ← each spatial position is a C-dim vector
        # feat_B: (B, C, Hb*Wb)
        feat_A_flat = feat_A.view(B, C, -1).permute(0, 2, 1)   # (B, Ha*Wa, C)
        feat_B_flat = feat_B.view(B, C, -1)                     # (B, C, Hb*Wb)

        # Correlation: (B, Ha*Wa, Hb*Wb)
        corr = torch.bmm(feat_A_flat, feat_B_flat)

        # ReLU: only keep positive correlations (suppress noise)
        corr = F.relu(corr)

        return corr


# ─────────────────────────────────────────────────────────────────
# TPS control point regressor
# ─────────────────────────────────────────────────────────────────

class TPSRegressor(nn.Module):
    """
    Predicts TPS control point offsets from the correlation map.

    Input:  correlation map (B, Ha*Wa, Hb*Wb) — flattened to 1D
    Output: (B, 2*N_cp) — Δx, Δy offsets for each of the N_cp control points

    The N_cp control points form a regular grid over the cloth image.
    Their offsets define how the cloth is deformed.
    """

    def __init__(self, feat_h: int, feat_w: int, n_control_points: int = 25):
        super().__init__()
        self.n_cp = n_control_points
        in_dim = feat_h * feat_w * feat_h * feat_w  # Ha*Wa × Hb*Wb

        # Progressive reduction to keep param count reasonable
        self.regressor = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 2 * n_control_points),  # 2× for x and y offsets
        )

        # Initialize final layer to near-zero → small initial deformation
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)

    def forward(self, corr: torch.Tensor) -> torch.Tensor:
        # corr: (B, Ha*Wa, Hb*Wb)
        B = corr.shape[0]
        x = corr.view(B, -1)       # flatten
        offsets = self.regressor(x) # (B, 2*N_cp)
        # Tanh clamps offsets to (-1, 1) in normalized image coordinates
        # This prevents extreme deformations that tear the cloth
        return torch.tanh(offsets) * 0.5   # scale to ±0.5 max offset
        # Note: 0.5 in normalized coords ≈ 192px at 384 width — enough range


# ─────────────────────────────────────────────────────────────────
# TPS Grid Generator
# ─────────────────────────────────────────────────────────────────

class TPSGridGenerator(nn.Module):
    """
    Converts control point offsets to a dense warp grid.

    Thin Plate Spline interpolation: given N control points with
    known positions and displacements, interpolate a smooth warp
    at every pixel location.

    The TPS interpolation minimizes the "bending energy" — it finds
    the smoothest possible warp that passes through all control points.
    This is the key property that prevents physically impossible folds.

    Input:
        offsets: (B, 2*N_cp) — predicted Δx, Δy per control point
        H, W: output grid size

    Output:
        grid: (B, H, W, 2) — sampling grid for F.grid_sample
              grid[b, h, w] = (x, y) position in input to sample from
    """

    def __init__(self, height: int, width: int, n_control_points: int = 25):
        super().__init__()
        self.H = height
        self.W = width
        self.N = n_control_points

        # Control point grid: regular N×N grid over [-1, 1]^2
        # We use sqrt(N) × sqrt(N) grid (N must be a perfect square)
        grid_size = int(np.sqrt(n_control_points))
        assert grid_size * grid_size == n_control_points, \
            f"n_control_points must be a perfect square, got {n_control_points}"

        # Source control point positions (fixed, regular grid)
        xs = np.linspace(-0.9, 0.9, grid_size)
        ys = np.linspace(-0.9, 0.9, grid_size)
        xx, yy = np.meshgrid(xs, ys)
        # Shape: (N, 2) in (x, y) order
        src_pts = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
        self.register_buffer('src_pts', torch.from_numpy(src_pts))

        # Precompute target grid positions (every pixel, normalized)
        # Shape: (H*W, 2)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width),
            indexing='ij'
        )
        target_pts = torch.stack([grid_x.ravel(), grid_y.ravel()], dim=1)  # (H*W, 2)
        self.register_buffer('target_pts', target_pts)

        # Precompute TPS kernel matrix inverse for efficiency
        # (This is constant given fixed source control points)
        K_src = self._tps_kernel(src_pts, src_pts)        # (N, N)
        # Build the linear system: [K | P; P^T | 0] * [w; a] = [y; 0]
        N = n_control_points
        P = np.concatenate([np.ones((N, 1)), src_pts], axis=1)  # (N, 3)
        L = np.zeros((N + 3, N + 3), dtype=np.float32)
        L[:N, :N] = K_src
        L[:N, N:] = P
        L[N:, :N] = P.T
        # Add regularization to avoid singular matrix
        L[:N, :N] += 1e-6 * np.eye(N, dtype=np.float32)
        L_inv = np.linalg.inv(L)
        self.register_buffer('L_inv', torch.from_numpy(L_inv))
        self.register_buffer('P_src', torch.from_numpy(P))

    @staticmethod
    def _tps_kernel(pts_a: np.ndarray, pts_b: np.ndarray) -> np.ndarray:
        """
        TPS radial basis function: U(r) = r² * log(r²)
        where r = distance between two points.
        This is the 2D TPS kernel.
        """
        # pts_a: (M, 2), pts_b: (N, 2) → (M, N) kernel matrix
        diff = pts_a[:, None, :] - pts_b[None, :, :]   # (M, N, 2)
        r2 = (diff ** 2).sum(-1)                        # (M, N)
        # U(r) = r² log(r²),  U(0) = 0 by convention
        with np.errstate(divide='ignore', invalid='ignore'):
            K = np.where(r2 == 0, 0.0, r2 * np.log(r2 + 1e-12))
        return K.astype(np.float32)

    def _tps_kernel_torch(self, pts_a: torch.Tensor, pts_b: torch.Tensor) -> torch.Tensor:
        """
        Torch version of TPS kernel, batched.

        pts_a: (B, M, 2)
        pts_b: (B, N, 2)
        Returns: (B, M, N) — per-batch pairwise kernel values

        NOTE: pts_a and pts_b must both carry the batch dimension and
        must NOT be flattened with .reshape(-1, 2) before calling this —
        doing so merges the batch dim into the points dim and produces
        a (B*M, B*N) cross-batch kernel instead of B independent (M,N)
        kernels, which both inflates memory and computes wrong values.
        """
        diff = pts_a.unsqueeze(2) - pts_b.unsqueeze(1)   # (B, M, N, 2)
        r2 = (diff ** 2).sum(-1)                          # (B, M, N)
        return torch.where(r2 == 0, torch.zeros_like(r2), r2 * torch.log(r2 + 1e-12))

    def forward(self, offsets: torch.Tensor) -> torch.Tensor:
        """
        offsets: (B, 2*N) — Δx, Δy per control point
        Returns: (B, H, W, 2) — warp grid for F.grid_sample
        """
        B = offsets.shape[0]
        N = self.N

        # Split into x and y offsets → target control point positions
        delta = offsets.view(B, N, 2)                        # (B, N, 2)
        src = self.src_pts.unsqueeze(0).expand(B, -1, -1)   # (B, N, 2)
        dst = src + delta                                     # (B, N, 2)

        # For each batch element, solve TPS and evaluate at all pixel positions
        # Build RHS: (N+3, 2) = [dst; 0; 0; 0]
        zeros = torch.zeros(B, 3, 2, device=offsets.device, dtype=offsets.dtype)
        rhs = torch.cat([dst, zeros], dim=1)                 # (B, N+3, 2)

        # Solve for TPS weights: w = L_inv @ rhs
        # L_inv: (N+3, N+3),  rhs: (B, N+3, 2)
        L_inv = self.L_inv.unsqueeze(0).expand(B, -1, -1)   # (B, N+3, N+3)
        w = torch.bmm(L_inv, rhs)                            # (B, N+3, 2)

        # Evaluate TPS at all target pixel positions
        # target_pts: (H*W, 2)
        M = self.H * self.W
        target = self.target_pts.unsqueeze(0).expand(B, -1, -1)   # (B, M, 2)

        # Kernel values at target pts w.r.t. source control pts.
        # target_pts and src_pts are FIXED buffers (identical across the
        # batch), so compute the (M, N) kernel once on the unbatched
        # tensors, then broadcast to the batch — this is both correct
        # and far cheaper than recomputing per batch element.
        K_target_unbatched = self._tps_kernel_torch(
            self.target_pts.unsqueeze(0),           # (1, M, 2)
            self.src_pts.unsqueeze(0).detach(),      # (1, N, 2)
        )                                              # (1, M, N)
        K_target = K_target_unbatched.expand(B, -1, -1)   # (B, M, N)

        # Polynomial part: [1, x, y] for each target point
        ones = torch.ones(B, M, 1, device=offsets.device, dtype=offsets.dtype)
        P_target = torch.cat([ones, target], dim=-1)   # (B, M, 3)

        # TPS displacement at each target pixel:
        #   f(x) = a0 + a1*x + a2*y + Σ w_i * U(|x - c_i|)
        w_kernel = w[:, :N, :]    # (B, N, 2) — kernel weights
        w_affine = w[:, N:, :]    # (B, 3, 2) — affine weights

        disp = (torch.bmm(K_target, w_kernel) +
                torch.bmm(P_target, w_affine))    # (B, M, 2)

        # disp is the DISPLACEMENT from identity — convert to absolute grid
        grid = self.target_pts.unsqueeze(0).expand(B, -1, -1) + disp  # (B, M, 2)
        grid = grid.view(B, self.H, self.W, 2)

        # Clamp to [-1, 1] — out-of-bounds samples return zeros (border mode)
        grid = grid.clamp(-1, 1)

        return grid


# ─────────────────────────────────────────────────────────────────
# Full TPS Warping Network
# ─────────────────────────────────────────────────────────────────

class TPSWarpingNet(nn.Module):
    """
    Full TPS garment warping network.

    Inputs:
        cloth:      (B, 3,  H, W) — flat garment image
        agnostic:   (B, 3,  H, W) — person with clothes erased (grey fill)
        pose:       (B, 3,  H, W) — skeleton visualization
        densepose:  (B, 3,  H, W) — IUV body surface map

    Output:
        warped_cloth: (B, 3, H, W) — garment warped to body shape
        grid:         (B, H, W, 2) — warp grid (for visualization/regularization loss)
        offsets:      (B, 2*N_cp)  — raw control point offsets (for regularization)

    VRAM profile at (B=8, 512×384):
        Feature extractors: ~2.1 GB
        Correlation layer:  ~0.8 GB (feat_h=32, feat_w=24 → 32*24*32*24=589k floats)
        Total:              ~3.2 GB — fits T4 at batch 8
    """

    def __init__(
        self,
        height: int = 512,
        width:  int = 384,
        n_control_points: int = 25,
    ):
        super().__init__()
        self.H = height
        self.W = width
        self.N = n_control_points

        # Feature size after 4× MaxPool downsampling
        self.feat_h = height // 16   # 32 for 512
        self.feat_w = width  // 16   # 24 for 384

        # Stream A: cloth features (3 channels)
        self.feat_extractor_cloth = FeatureExtractor(in_channels=3)

        # Stream B: body features (agnostic + pose + densepose = 9 channels)
        self.feat_extractor_body  = FeatureExtractor(in_channels=9)

        # Correlation
        self.correlation = CorrelationLayer()

        # TPS regressor
        self.regressor = TPSRegressor(
            feat_h=self.feat_h,
            feat_w=self.feat_w,
            n_control_points=n_control_points,
        )

        # TPS grid generator
        self.grid_gen = TPSGridGenerator(
            height=height,
            width=width,
            n_control_points=n_control_points,
        )

    def forward(
        self,
        cloth: torch.Tensor,
        agnostic: torch.Tensor,
        pose: torch.Tensor,
        densepose: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # ── Stream A: extract cloth features ──
        cloth_feat = self.feat_extractor_cloth(cloth)       # (B, 512, H/16, W/16)

        # ── Stream B: extract body features (concat inputs) ──
        body_input = torch.cat([agnostic, pose, densepose], dim=1)  # (B, 9, H, W)
        body_feat  = self.feat_extractor_body(body_input)           # (B, 512, H/16, W/16)

        # ── Correlation ──
        corr = self.correlation(cloth_feat, body_feat)      # (B, Ha*Wa, Hb*Wb)

        # ── TPS regression ──
        offsets = self.regressor(corr)                      # (B, 2*N_cp)

        # ── Generate warp grid ──
        grid = self.grid_gen(offsets)                       # (B, H, W, 2)

        # ── Apply warp ──
        warped_cloth = F.grid_sample(
            cloth,
            grid,
            mode='bilinear',
            padding_mode='border',    # 'border' replicates edge pixels → no black border
            align_corners=True,
        )

        return warped_cloth, grid, offsets


# ─────────────────────────────────────────────────────────────────
# Model summary utility
# ─────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> str:
    total    = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (f"Total params:     {total:,}\n"
            f"Trainable params: {trainable:,} "
            f"({trainable/total*100:.1f}%)")


if __name__ == '__main__':
    # Quick sanity check
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, H, W = 2, 512, 384

    model = TPSWarpingNet(height=H, width=W, n_control_points=25).to(device)
    print(count_parameters(model))

    cloth     = torch.randn(B, 3, H, W, device=device)
    agnostic  = torch.randn(B, 3, H, W, device=device)
    pose      = torch.randn(B, 3, H, W, device=device)
    densepose = torch.randn(B, 3, H, W, device=device)

    warped, grid, offsets = model(cloth, agnostic, pose, densepose)
    print(f"warped_cloth: {warped.shape}")    # (2, 3, 512, 384)
    print(f"grid:         {grid.shape}")      # (2, 512, 384, 2)
    print(f"offsets:      {offsets.shape}")   # (2, 50)

    if device == 'cuda':
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM: {mem:.2f} GB")