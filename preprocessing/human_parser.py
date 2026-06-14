"""
human_parser.py — SCHP (Self-Correction for Human Parsing) ONNX wrapper

What it does:
  Takes a person RGB image and outputs a semantic label map where each pixel
  is assigned one of 20 body-part labels (background, hair, face, clothes, etc.)

Why we need it:
  We need to know WHERE the current clothes are on the person so we can erase
  them and replace with the new garment. The label map drives agnostic_mask creation.

Label map (ATR — 18 classes, LIP — 20 classes):
  0=Background, 1=Hat, 2=Hair, 3=Sunglasses, 4=Upper-clothes, 5=Skirt,
  6=Pants, 7=Dress, 8=Belt, 9=Left-shoe, 10=Right-shoe, 11=Face,
  12=Left-leg, 13=Right-leg, 14=Left-arm, 15=Right-arm, 16=Bag, 17=Scarf
  (LIP adds: 18=Left-hand, 19=Right-hand)
"""

import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path


# Label names for debugging/visualization
ATR_LABELS = [
    'Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes',
    'Skirt', 'Pants', 'Dress', 'Belt', 'Left-shoe', 'Right-shoe',
    'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf'
]

LIP_LABELS = ATR_LABELS + ['Left-hand', 'Right-hand']

# Color palette for visualization (one BGR color per label)
PALETTE = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
]


class HumanParser:
    """
    Wraps SCHP ONNX model for inference.
    Supports both ATR (18-class) and LIP (20-class) checkpoints.
    
    Usage:
        parser = HumanParser('pretrained/humanparsing/parsing_atr.onnx')
        label_map = parser.parse(person_bgr_image)  # returns H×W uint8 array
    """

    INPUT_SIZE = (512, 512)  # SCHP expects 512×512 input

    def __init__(self, onnx_path: str):
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {onnx_path}\n"
                "Download from: https://huggingface.co/yisol/IDM-VTON-DC/tree/main/ckpt/humanparsing"
            )

        # Use CPU provider — SCHP is fast enough (~50ms) and avoids GPU memory pressure
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        print(f"[HumanParser] Loaded: {onnx_path.name} | "
              f"Input: {self.session.get_inputs()[0].shape}")

    def _preprocess(self, img_bgr: np.ndarray) -> tuple[np.ndarray, tuple]:
        """
        Resize to 512×512 and normalize to ImageNet stats.
        Returns: (processed_tensor, original_size)
        
        ImageNet normalization:
          mean = [0.406, 0.456, 0.485]  (BGR order)
          std  = [0.225, 0.224, 0.229]
        """
        orig_h, orig_w = img_bgr.shape[:2]

        img_resized = cv2.resize(img_bgr, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Normalize: convert to float, apply ImageNet stats
        img_float = img_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_float - mean) / std

        # HWC → CHW → NCHW (add batch dim)
        img_chw = img_norm.transpose(2, 0, 1)
        img_nchw = img_chw[np.newaxis, ...]  # shape: (1, 3, 512, 512)

        return img_nchw.astype(np.float32), (orig_h, orig_w)

    def _postprocess(self, logits: np.ndarray, orig_size: tuple) -> np.ndarray:
        """
        Convert raw logits to label map at original resolution.
        
        logits shape: (1, num_classes, 512, 512)
        Returns: H×W uint8 label map at original image size
        """
        # Argmax over class dimension → (1, 512, 512)
        pred = np.argmax(logits[0], axis=0).astype(np.uint8)  # (512, 512)

        # Resize back to original resolution using NEAREST (preserve integer labels)
        orig_h, orig_w = orig_size
        pred_resized = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        return pred_resized

    def parse(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Main inference method.
        
        Args:
            img_bgr: OpenCV BGR image, any size
        Returns:
            label_map: H×W uint8, values 0–17 (ATR) or 0–19 (LIP)
        """
        tensor, orig_size = self._preprocess(img_bgr)
        outputs = self.session.run(None, {self.input_name: tensor})
        # SCHP ONNX output: [logits] where logits.shape = (1, num_classes, 512, 512)
        logits = outputs[0]
        return self._postprocess(logits, orig_size)

    @staticmethod
    def colorize(label_map: np.ndarray) -> np.ndarray:
        """Convert label map to a color visualization image (BGR)."""
        h, w = label_map.shape
        color_img = np.zeros((h, w, 3), dtype=np.uint8)
        for label_idx, color in enumerate(PALETTE):
            color_img[label_map == label_idx] = color
        return color_img

    @staticmethod
    def get_region_mask(label_map: np.ndarray, labels: list[int]) -> np.ndarray:
        """
        Extract binary mask for specific label indices.
        Example: get_region_mask(label_map, [4, 5, 7]) → upper clothes + skirt + dress
        
        Returns: H×W binary uint8 mask (0 or 255)
        """
        mask = np.zeros(label_map.shape, dtype=np.uint8)
        for label in labels:
            mask[label_map == label] = 255
        return mask