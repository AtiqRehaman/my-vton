"""
densepose_estimator.py — Detectron2 DensePose wrapper

What it does:
  Maps each body pixel to UV coordinates on the SMPL 3D body surface.
  Output is an IUV image:
    I channel (0–24): which body part (0=bg, 1=torso, 2=L-thigh, etc.)
    U channel (0–255): horizontal position on the 3D surface
    V channel (0–255): vertical position on the 3D surface

Why we need it:
  UV coordinates give the generation model 3D body shape understanding.
  Without DensePose, the model sees the person as a flat 2D silhouette.
  With DensePose, it understands that pixel (120, 300) is on the LEFT side
  of the torso at UV=(0.3, 0.7) — this enables proper cloth wrapping around
  3D body contours.

DensePose body part indices (I channel):
  0=Background, 1=Torso, 2=Right-Hand, 3=Left-Hand, 4=L-Lower-Arm,
  5=R-Lower-Arm, 6=L-Upper-Arm, 7=R-Upper-Arm, 8=L-Thigh, 9=R-Thigh,
  10=L-Calf, 11=R-Calf, 12=L-Foot, 13=R-Foot, 14=Upper-Leg-Left,
  15=Upper-Leg-Right, 16=Lower-Leg-Left, 17=Lower-Leg-Right, 18=Upper-Arm-Left,
  19=Upper-Arm-Right, 20=Lower-Arm-Left, 21=Lower-Arm-Right, 22=Left-Hand,
  23=Right-Hand, 24=Head

Install (Colab T4):
  pip install 'git+https://github.com/facebookresearch/detectron2.git'
  pip install 'git+https://github.com/facebookresearch/detectron2.git#subdirectory=projects/DensePose'

Checkpoint:
  wget https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl
  Save to: pretrained/densepose/model_final_162be9.pkl
"""

import numpy as np
import cv2
from pathlib import Path


class DensePoseEstimator:
    """
    Wraps Detectron2 DensePose for IUV prediction.
    
    VRAM: ~500MB on GPU. Runs fine on T4.
    Speed: ~200ms per image on T4 GPU.
    """

    CONFIG_URL = (
        "detectron2://DensePose/densepose_rcnn_R_50_FPN_s1x/"
        "165712039/model_final_162be9.pkl"
    )

    def __init__(self, checkpoint_path: str = 'pretrained/densepose/model_final_162be9.pkl',
                 device: str = 'auto'):
        checkpoint_path = Path(checkpoint_path)

        try:
            import torch
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from densepose import add_densepose_config
            from densepose.vis.extractor import DensePoseResultExtractor
        except ImportError as e:
            raise ImportError(
                f"Detectron2/DensePose import failed: {e}\n"
                "Install with:\n"
                "  pip install 'git+https://github.com/facebookresearch/detectron2.git'\n"
                "  pip install 'git+https://github.com/facebookresearch/detectron2.git"
                "#subdirectory=projects/DensePose'\n"
                "Colab: Runtime → Restart after install"
            )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"DensePose checkpoint not found: {checkpoint_path}\n"
                "Download:\n"
                "  wget https://dl.fbaipublicfiles.com/densepose/"
                "densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl\n"
                "  mkdir -p pretrained/densepose\n"
                "  mv model_final_162be9.pkl pretrained/densepose/"
            )

        if device == 'auto':
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Build Detectron2 config for DensePose RCNN with ResNet-50 FPN backbone
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(self._get_config_file())
        cfg.MODEL.WEIGHTS = str(checkpoint_path)
        cfg.MODEL.DEVICE = device
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # detection confidence threshold

        self.predictor = DefaultPredictor(cfg)
        self.extractor = DensePoseResultExtractor()
        self.device = device
        print(f"[DensePose] Loaded model_final_162be9.pkl on {device}")

    @staticmethod
    def _get_config_file() -> str:
        """Get the DensePose RCNN R50 FPN config file path."""
        try:
            from densepose.model_zoo import ModelZooConfig
            return ModelZooConfig.get_config_file(
                "densepose_rcnn_R_50_FPN_s1x.yaml"
            )
        except Exception:
            # Fallback: locate installed config
            import pkg_resources
            try:
                return pkg_resources.resource_filename(
                    'densepose',
                    'configs/densepose_rcnn_R_50_FPN_s1x.yaml'
                )
            except Exception:
                raise RuntimeError(
                    "Cannot locate DensePose config file.\n"
                    "Try: find / -name 'densepose_rcnn_R_50_FPN_s1x.yaml' 2>/dev/null"
                )

    def estimate(self, img_bgr: np.ndarray) -> dict:
        """
        Run DensePose estimation.
        
        Args:
            img_bgr: OpenCV BGR image (person photo)
        Returns:
            dict with keys:
              'iuv_img':     H×W×3 uint8 image with channels [I, U, V]
              'iuv_colored': H×W×3 BGR colorized visualization for debugging
              'has_detection': bool, whether a person was detected
        """
        h, w = img_bgr.shape[:2]

        outputs = self.predictor(img_bgr)['instances']

        iuv_img = np.zeros((h, w, 3), dtype=np.uint8)

        if len(outputs) == 0:
            # No person detected — return blank IUV
            print("[DensePose] Warning: no person detected in image")
            return {
                'iuv_img': iuv_img,
                'iuv_colored': np.zeros((h, w, 3), dtype=np.uint8),
                'has_detection': False
            }

        # Extract DensePose results for detected instances
        result = self.extractor(outputs)

        # Build IUV image by rendering each detected person
        iuv_img = self._render_iuv(result, h, w)

        return {
            'iuv_img': iuv_img,
            'iuv_colored': self._colorize_iuv(iuv_img),
            'has_detection': True
        }

    def _render_iuv(self, densepose_result, h: int, w: int) -> np.ndarray:
        """
        Render DensePose IUV result to an H×W×3 uint8 image.
        Channel 0 = I (body part index, 0–24)
        Channel 1 = U (0–255)
        Channel 2 = V (0–255)
        """
        iuv = np.zeros((h, w, 3), dtype=np.uint8)

        for i, (boxes, result_encoded) in enumerate(zip(
            densepose_result.pred_boxes_XYXY,
            densepose_result.pred_densepose
        )):
            x1, y1, x2, y2 = [int(c) for c in boxes.tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Decode result: get labels (I) and UV
            labels = result_encoded.labels.cpu().numpy().astype(np.uint8)
            uv = result_encoded.uv.cpu().numpy()  # (2, H_box, W_box)

            box_h = y2 - y1
            box_w = x2 - x1

            # Resize to bounding box size
            labels_resized = cv2.resize(labels, (box_w, box_h), interpolation=cv2.INTER_NEAREST)
            u_resized = cv2.resize(uv[0], (box_w, box_h), interpolation=cv2.INTER_LINEAR)
            v_resized = cv2.resize(uv[1], (box_w, box_h), interpolation=cv2.INTER_LINEAR)

            # Fill IUV image in the bounding box region
            iuv[y1:y2, x1:x2, 0] = labels_resized
            iuv[y1:y2, x1:x2, 1] = (u_resized * 255).clip(0, 255).astype(np.uint8)
            iuv[y1:y2, x1:x2, 2] = (v_resized * 255).clip(0, 255).astype(np.uint8)

        return iuv

    @staticmethod
    def _colorize_iuv(iuv_img: np.ndarray) -> np.ndarray:
        """
        Create a colorful visualization of the IUV map for debugging.
        Each body part (I channel) gets a distinct color, U and V shown as texture.
        """
        # Assign distinct hue per body part
        part_hues = np.linspace(0, 179, 25, dtype=np.uint8)  # 25 parts
        h, w = iuv_img.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)

        I = iuv_img[:, :, 0]
        U = iuv_img[:, :, 1]
        V = iuv_img[:, :, 2]

        for part_idx in range(1, 25):  # skip background (0)
            mask = I == part_idx
            if mask.any():
                hsv[mask, 0] = part_hues[part_idx]
                hsv[mask, 1] = 200
                hsv[mask, 2] = U[mask] // 2 + 128  # brightness encodes U

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)