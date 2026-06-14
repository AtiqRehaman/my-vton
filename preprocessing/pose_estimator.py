"""
pose_estimator.py — OpenPose wrapper for 18-keypoint body pose estimation

What it does:
  Detects 18 body keypoints (nose, neck, shoulders, elbows, wrists,
  hips, knees, ankles, eyes, ears) and renders a skeleton image.

Why we need it:
  The generation model needs to understand HOW the person is posed so it
  can correctly place the garment. Without pose conditioning, the model
  guesses body orientation and produces misaligned results.

Keypoint indices (COCO 18-point format):
  0=Nose, 1=Neck, 2=RShoulder, 3=RElbow, 4=RWrist,
  5=LShoulder, 6=LElbow, 7=LWrist, 8=MidHip, 9=RHip,
  10=RKnee, 11=RAnkle, 12=LHip, 13=LKnee, 14=LAnkle,
  15=REye, 16=LEye, 17=REar, 18=LEar  (index 18 unused in 18-pt format)

Two usage modes:
  1. controlnet-aux (recommended for Colab) — pip install controlnet-aux
  2. Direct .pth checkpoint (for local use with the IDM-VTON checkpoint)
"""

import json
import numpy as np
import cv2
from pathlib import Path


# Skeleton connections for rendering: list of (kp_a, kp_b, BGR_color)
SKELETON = [
    (0, 1, (255, 165, 0)),   # Nose-Neck: orange
    (1, 2, (255, 0, 0)),     # Neck-RShoulder: blue
    (2, 3, (255, 0, 0)),     # RShoulder-RElbow
    (3, 4, (255, 0, 0)),     # RElbow-RWrist
    (1, 5, (0, 0, 255)),     # Neck-LShoulder: red
    (5, 6, (0, 0, 255)),     # LShoulder-LElbow
    (6, 7, (0, 0, 255)),     # LElbow-LWrist
    (1, 8, (0, 255, 0)),     # Neck-MidHip: green
    (8, 9, (0, 255, 165)),   # MidHip-RHip
    (9, 10, (0, 255, 165)),  # RHip-RKnee
    (10, 11, (0, 255, 165)), # RKnee-RAnkle
    (8, 12, (165, 255, 0)),  # MidHip-LHip
    (12, 13, (165, 255, 0)), # LHip-LKnee
    (13, 14, (165, 255, 0)), # LKnee-LAnkle
    (0, 15, (255, 255, 0)),  # Nose-REye
    (15, 17, (255, 255, 0)), # REye-REar
    (0, 16, (255, 0, 255)),  # Nose-LEye
    (16, 18, (255, 0, 255)), # LEye-LEar (18 = extra for full connectivity)
]

NUM_KEYPOINTS = 18


class PoseEstimatorControlnetAux:
    """
    Uses controlnet-aux library (easiest installation on Colab).
    
    Install: pip install controlnet-aux
    
    This wraps OpenposeDetector from controlnet-aux which internally
    uses the same body_pose_model.pth weights.
    """

    def __init__(self):
        try:
            from controlnet_aux import OpenposeDetector
        except ImportError:
            raise ImportError(
                "controlnet-aux not installed.\n"
                "Run: pip install controlnet-aux\n"
                "Colab: pip install controlnet-aux==0.0.7"
            )
        # Downloads weights automatically to ~/.cache/
        self.detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
        print("[PoseEstimator] Loaded OpenposeDetector (controlnet-aux)")

    def estimate(self, img_bgr: np.ndarray) -> dict:
        """
        Run pose estimation.
        
        Args:
            img_bgr: OpenCV BGR image
        Returns:
            dict with keys:
              'keypoints': list of 18 [x, y, confidence] — None if not detected
              'pose_img':  H×W×3 BGR skeleton visualization
              'keypoints_json': JSON string for saving to disk
        """
        import PIL.Image

        h, w = img_bgr.shape[:2]
        img_rgb_pil = PIL.Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        # detect_resolution=512 standardizes detection then resizes back
        pose_pil = self.detector(
            img_rgb_pil,
            detect_resolution=512,
            image_resolution=max(h, w)
        )

        pose_img_bgr = cv2.cvtColor(np.array(pose_pil), cv2.COLOR_RGB2BGR)
        pose_img_bgr = cv2.resize(pose_img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

        # Extract raw keypoints from the detector (returns pose objects)
        keypoints_raw = self._extract_keypoints_controlnet(img_rgb_pil, h, w)

        return {
            'keypoints': keypoints_raw,
            'pose_img': pose_img_bgr,
            'keypoints_json': json.dumps({'keypoints': keypoints_raw, 'img_h': h, 'img_w': w})
        }

    def _extract_keypoints_controlnet(self, img_pil, h, w):
        """Extract numeric keypoints from controlnet-aux detector."""
        try:
            # controlnet-aux v0.0.7+ supports return_pil=False
            result = self.detector.detect_poses(img_pil)
            if result and len(result) > 0:
                pose = result[0]
                keypoints = []
                for kp in pose.body.keypoints:
                    if kp is None:
                        keypoints.append(None)
                    else:
                        # kp.x and kp.y are normalized 0–1
                        keypoints.append([float(kp.x * w), float(kp.y * h), float(kp.score)])
                return keypoints
        except Exception:
            pass
        # Fallback: return None keypoints (pose_img still valid from rendered output)
        return [None] * NUM_KEYPOINTS


class PoseEstimatorDirect:
    """
    Uses body_pose_model.pth directly from IDM-VTON checkpoint.
    Best for local use when you already have the checkpoint.
    
    Checkpoint: pretrained/openpose/body_pose_model.pth
    Source: HuggingFace yisol/IDM-VTON-DC
    """

    def __init__(self, model_path: str):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"OpenPose checkpoint not found: {model_path}\n"
                "Download: huggingface-cli download yisol/IDM-VTON-DC "
                "--include 'ckpt/openpose/*' --local-dir pretrained/"
            )

        import torch
        # Import body_estimation from the openpose submodule
        # This requires cloning: https://github.com/Hzzone/pytorch-openpose
        try:
            from openpose.src.body import Body
            self.body_estimation = Body(str(model_path))
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"[PoseEstimator] Loaded body_pose_model.pth on {self.device}")
        except ImportError:
            raise ImportError(
                "pytorch-openpose not installed.\n"
                "Run: git clone https://github.com/Hzzone/pytorch-openpose\n"
                "Then: sys.path.insert(0, 'pytorch-openpose')"
            )

    def estimate(self, img_bgr: np.ndarray) -> dict:
        h, w = img_bgr.shape[:2]
        candidate, subset = self.body_estimation(img_bgr)
        keypoints = self._parse_candidates(candidate, subset, h, w)
        pose_img = self._render_skeleton(keypoints, h, w)
        return {
            'keypoints': keypoints,
            'pose_img': pose_img,
            'keypoints_json': json.dumps({'keypoints': keypoints, 'img_h': h, 'img_w': w})
        }

    def _parse_candidates(self, candidate, subset, h, w):
        """Convert raw openpose output to list of 18 [x, y, conf] or None."""
        keypoints = [None] * NUM_KEYPOINTS
        if subset is None or len(subset) == 0:
            return keypoints

        # Take the first detected person (highest confidence subset)
        person = subset[0]
        for kp_idx in range(NUM_KEYPOINTS):
            cand_idx = int(person[kp_idx])
            if cand_idx >= 0 and cand_idx < len(candidate):
                x, y, conf, _ = candidate[cand_idx]
                keypoints[kp_idx] = [float(x), float(y), float(conf)]
        return keypoints

    @staticmethod
    def _render_skeleton(keypoints: list, h: int, w: int) -> np.ndarray:
        """
        Draw skeleton on black canvas.
        Same format as controlnet-aux renders — important for consistent conditioning.
        """
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw limb connections
        for (kp_a, kp_b, color) in SKELETON:
            if kp_a < len(keypoints) and kp_b < len(keypoints):
                if keypoints[kp_a] is not None and keypoints[kp_b] is not None:
                    x1, y1, c1 = keypoints[kp_a]
                    x2, y2, c2 = keypoints[kp_b]
                    if c1 > 0.1 and c2 > 0.1:
                        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)),
                                 color, thickness=3, lineType=cv2.LINE_AA)

        # Draw keypoint circles
        for kp in keypoints:
            if kp is not None:
                x, y, conf = kp
                if conf > 0.1:
                    cv2.circle(canvas, (int(x), int(y)), radius=5,
                               color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

        return canvas


def load_pose_estimator(mode: str = 'controlnet', model_path: str = None):
    """
    Factory function — returns the right estimator based on mode.
    
    Args:
        mode: 'controlnet' (Colab recommended) or 'direct' (local .pth)
        model_path: only required for mode='direct'
    """
    if mode == 'controlnet':
        return PoseEstimatorControlnetAux()
    elif mode == 'direct':
        if model_path is None:
            model_path = 'pretrained/openpose/body_pose_model.pth'
        return PoseEstimatorDirect(model_path)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'controlnet' or 'direct'")