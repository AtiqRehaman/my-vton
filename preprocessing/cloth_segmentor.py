"""
cloth_segmentor.py — U2Net cloth segmentation wrapper

What it does:
  Removes the background from a garment product photo and produces:
    1. A binary mask of the garment
    2. A clean garment image on pure white background

Why we need it:
  Garment product photos come on various backgrounds (grey, gradient, model).
  The warping model and generation model both need a clean, isolated cloth
  on a white background — any background bleeds into the warped cloth and
  corrupts generation quality.

  The model also classifies pixels into 3 channels:
    Channel 0: upper_body (shirts, jackets, tops)
    Channel 1: lower_body (pants, skirts)  
    Channel 2: full_body (dresses, jumpsuits)

Two usage modes:
  1. rembg library (easiest): pip install rembg[gpu] — auto-downloads u2net_cloth_seg
  2. Direct ONNX checkpoint: u2net_cloth_seg.onnx from the original repo
"""

import numpy as np
import cv2
from pathlib import Path


class ClothSegmentorRembg:
    """
    Uses the rembg library which wraps u2net_cloth_seg automatically.
    Easiest installation — recommended for Colab.
    
    Install: pip install rembg
    Colab:   pip install rembg==2.0.50
    
    rembg auto-downloads the u2net_cloth_seg.pth checkpoint to:
      ~/.u2net/u2net_cloth_seg.pth (232 MB)
    """

    def __init__(self):
        try:
            from rembg import remove, new_session
            # Use cloth-specific model — NOT the generic u2net
            self.session = new_session('u2net_cloth_seg')
            self.remove_fn = remove
            print("[ClothSegmentor] Loaded u2net_cloth_seg via rembg")
        except ImportError:
            raise ImportError(
                "rembg not installed.\n"
                "Run: pip install rembg\n"
                "Colab: pip install rembg==2.0.50"
            )

    def segment(self, cloth_bgr: np.ndarray) -> dict:
        """
        Segment garment from background.
        
        Args:
            cloth_bgr: OpenCV BGR image of garment product photo
        Returns:
            dict with keys:
              'cloth_mask':  H×W uint8 binary mask (0=bg, 255=cloth)
              'cloth_clean': H×W×3 BGR image, garment on white background
              'cloth_rgba':  H×W×4 RGBA image with transparent background
        """
        import PIL.Image
        import io

        h, w = cloth_bgr.shape[:2]
        img_rgb = cv2.cvtColor(cloth_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(img_rgb)

        # rembg returns RGBA PIL image
        result_pil = self.remove_fn(pil_img, session=self.session)
        result_rgba = np.array(result_pil)  # H×W×4, uint8

        # Alpha channel → binary mask
        alpha = result_rgba[:, :, 3]
        _, cloth_mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)

        # Compose on white background
        cloth_rgb = result_rgba[:, :, :3]
        white_bg = np.ones_like(cloth_rgb) * 255
        alpha_norm = alpha.astype(np.float32)[:, :, np.newaxis] / 255.0
        cloth_on_white = (cloth_rgb.astype(np.float32) * alpha_norm +
                          white_bg.astype(np.float32) * (1 - alpha_norm)).astype(np.uint8)
        cloth_clean_bgr = cv2.cvtColor(cloth_on_white, cv2.COLOR_RGB2BGR)

        # Post-process mask: close small holes, remove thin noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_CLOSE, kernel)
        cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_OPEN, kernel)

        cloth_rgba_bgra = cv2.cvtColor(result_rgba, cv2.COLOR_RGBA2BGRA)

        return {
            'cloth_mask': cloth_mask,
            'cloth_clean': cloth_clean_bgr,
            'cloth_rgba': cloth_rgba_bgra,
        }


class ClothSegmentorONNX:
    """
    Uses u2net_cloth_seg.onnx checkpoint directly.
    More control, no extra library dependencies beyond onnxruntime.
    
    Checkpoint source:
      https://github.com/bsmnyk/u2net_cloth_seg
      Or: HuggingFace yisol/IDM-VTON-DC (ckpt/u2net/u2net_cloth_seg.onnx)
    """

    INPUT_SIZE = (768, 768)  # U2Net cloth seg expects 768×768

    def __init__(self, onnx_path: str = 'pretrained/u2net/u2net_cloth_seg.onnx'):
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX checkpoint not found: {onnx_path}\n"
                "Download: huggingface-cli download yisol/IDM-VTON-DC "
                "--include 'ckpt/u2net/*' --local-dir pretrained/"
            )

        import onnxruntime as ort
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        print(f"[ClothSegmentor] Loaded ONNX: {onnx_path.name}")

    def _preprocess(self, img_bgr: np.ndarray) -> tuple[np.ndarray, tuple]:
        orig_h, orig_w = img_bgr.shape[:2]
        img_resized = cv2.resize(img_bgr, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # U2Net normalization
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img_norm = (img_rgb - mean) / std
        tensor = img_norm.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        return tensor, (orig_h, orig_w)

    def segment(self, cloth_bgr: np.ndarray) -> dict:
        tensor, orig_size = self._preprocess(cloth_bgr)
        outputs = self.session.run(None, {self.input_name: tensor})
        # outputs[0] shape: (1, 3, 768, 768) — 3 channels: upper/lower/full
        pred = outputs[0][0]  # (3, 768, 768)

        # Combine all 3 channels into single mask (any cloth type)
        combined = np.max(pred, axis=0)  # (768, 768)
        combined = (combined * 255).clip(0, 255).astype(np.uint8)

        orig_h, orig_w = orig_size
        cloth_mask = cv2.resize(combined, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        _, cloth_mask = cv2.threshold(cloth_mask, 127, 255, cv2.THRESH_BINARY)

        # Compose on white background
        cloth_rgb = cv2.cvtColor(cloth_bgr, cv2.COLOR_BGR2RGB)
        white_bg = np.ones_like(cloth_rgb) * 255
        alpha = cloth_mask.astype(np.float32)[:, :, np.newaxis] / 255.0
        cloth_on_white = (cloth_rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
        cloth_clean_bgr = cv2.cvtColor(cloth_on_white, cv2.COLOR_RGB2BGR)

        return {
            'cloth_mask': cloth_mask,
            'cloth_clean': cloth_clean_bgr,
        }


def prepare_cloth_for_training(
    cloth_clean_bgr: np.ndarray,
    target_h: int = 512,
    target_w: int = 384
) -> np.ndarray:
    """
    Prepare clean cloth image for training format.
    
    CRITICAL: Do NOT use simple resize_and_padding — it creates a landscape
    rectangle artifact because the cloth gets letterboxed into a wide canvas.
    
    Correct sequence (from hard-won lessons in spec):
      1. Run u2net_cloth_seg → get binary mask (done before this function)
      2. Remove near-white background pixels (R,G,B > 230 → transparent)
      3. Tight crop to bounding box of remaining pixels
      4. Scale so garment fills 85% of target canvas height
      5. Center on white portrait canvas (target_w × target_h)
    
    This matches VITON-HD training format exactly.
    """
    # Step 2: Remove near-white background
    img_rgb = cv2.cvtColor(cloth_clean_bgr, cv2.COLOR_BGR2RGB)
    near_white = (img_rgb[:, :, 0] > 230) & \
                 (img_rgb[:, :, 1] > 230) & \
                 (img_rgb[:, :, 2] > 230)
    alpha = (~near_white).astype(np.uint8) * 255

    # Step 3: Tight crop to bounding box
    ys, xs = np.where(alpha > 0)
    if len(ys) == 0:
        # Fallback: entire image is background — return white canvas
        return np.ones((target_h, target_w, 3), dtype=np.uint8) * 255

    y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
    cropped = img_rgb[y1:y2+1, x1:x2+1]

    # Step 4: Scale so garment fills 85% of target canvas height
    crop_h, crop_w = cropped.shape[:2]
    target_cloth_h = int(target_h * 0.85)
    scale = target_cloth_h / crop_h
    new_h = int(crop_h * scale)
    new_w = int(crop_w * scale)
    # Ensure width fits in canvas
    if new_w > target_w:
        scale = target_w / crop_w
        new_h = int(crop_h * scale)
        new_w = int(crop_w * scale)

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Step 5: Center on white portrait canvas
    canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


def load_cloth_segmentor(mode: str = 'rembg', onnx_path: str = None):
    """Factory function."""
    if mode == 'rembg':
        return ClothSegmentorRembg()
    elif mode == 'onnx':
        if onnx_path is None:
            onnx_path = 'pretrained/u2net/u2net_cloth_seg.onnx'
        return ClothSegmentorONNX(onnx_path)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'rembg' or 'onnx'")