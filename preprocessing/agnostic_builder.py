"""
agnostic_builder.py — Builds clothing-agnostic image and mask

What this does:
  Creates two outputs that are central to the VTON pipeline:

  1. agnostic_mask — a binary mask covering the clothing region we want
     to REPLACE (erased, so the model can generate new clothes there)

  2. agnostic_image — the person photo with the clothing region filled
     grey (128, 128, 128). This tells the model: "everything outside this
     grey region is real and should be preserved. Fill in the grey part
     with the new garment."

Why grey specifically?
  Grey (128) is the median of the 0–255 range. Using grey instead of black
  or white minimizes the initialization bias when the VAE encodes this image.
  It's a neutral "unknown" signal.

Which labels to erase depends on cloth_type:
  'upper'  : upper-clothes (4), left-arm (14), right-arm (15)
  'lower'  : pants (6), skirt (5)
  'overall': dress (7), upper-clothes (4), pants (6), left-arm (14), right-arm (15)

Label indices (ATR model):
  0=Background, 1=Hat, 2=Hair, 3=Sunglasses, 4=Upper-clothes,
  5=Skirt, 6=Pants, 7=Dress, 8=Belt, 9=Left-shoe, 10=Right-shoe,
  11=Face, 12=Left-leg, 13=Right-leg, 14=Left-arm, 15=Right-arm,
  16=Bag, 17=Scarf
"""

import numpy as np
import cv2


# Which parsing labels to erase per cloth type
CLOTH_LABELS = {
    'upper':   [4, 14, 15],          # upper-clothes, left-arm, right-arm
    'lower':   [5, 6, 8],            # skirt, pants, belt
    'overall': [4, 5, 6, 7, 14, 15], # everything from top to bottom
}

# Additional labels to always preserve (never erase these)
PRESERVE_LABELS = [0, 1, 2, 11, 17]  # background, hat, hair, face, scarf


class AgnosticBuilder:
    """
    Builds agnostic mask and agnostic image from parsing + pose outputs.
    
    Usage:
        builder = AgnosticBuilder(cloth_type='upper')
        result = builder.build(person_bgr, parse_map, keypoints)
        agnostic_mask  = result['agnostic_mask']
        agnostic_image = result['agnostic_image']
    """

    # Dilation kernel size — 5px dilation to avoid edge leakage
    # (spec requirement: dilate mask by 5px)
    DILATION_SIZE = 5

    # Arm extension multiplier — extend arm mask downward to cover wrists
    ARM_EXTENSION_RATIO = 0.3

    def __init__(self, cloth_type: str = 'upper'):
        if cloth_type not in CLOTH_LABELS:
            raise ValueError(f"cloth_type must be one of {list(CLOTH_LABELS.keys())}")
        self.cloth_type = cloth_type
        self.labels_to_erase = CLOTH_LABELS[cloth_type]

    def build(
        self,
        person_bgr: np.ndarray,
        parse_map: np.ndarray,
        keypoints: list | None = None,
    ) -> dict:
        """
        Build agnostic mask and image.
        
        Args:
            person_bgr: H×W×3 BGR person photo
            parse_map:  H×W uint8 label map from SCHP (A1)
            keypoints:  list of 18 [x,y,conf] or None — used to refine arm mask
                        If None, falls back to parse_map labels only
        Returns:
            dict with keys:
              'agnostic_mask':  H×W uint8 (0 or 255) — region to inpaint
              'agnostic_image': H×W×3 BGR — person with clothing region greyed out
              'agnostic_mask_vis': H×W×3 BGR — debug: colored mask overlay
        """
        h, w = person_bgr.shape[:2]

        # Step 1: Extract raw mask from parsing labels
        raw_mask = self._build_parse_mask(parse_map, h, w)

        # Step 2: Optionally refine arm regions using pose keypoints
        if keypoints is not None and self.cloth_type in ('upper', 'overall'):
            raw_mask = self._refine_arm_mask(raw_mask, keypoints, parse_map, h, w)

        # Step 3: Dilate to avoid edge leakage (spec: 5px dilation)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.DILATION_SIZE * 2 + 1, self.DILATION_SIZE * 2 + 1)
        )
        agnostic_mask = cv2.dilate(raw_mask, kernel, iterations=1)

        # Step 4: Make sure we preserve face/hair/hands regions
        # Do NOT erase face (label 11) or hair (label 2)
        preserve_mask = np.zeros((h, w), dtype=np.uint8)
        for label in PRESERVE_LABELS:
            preserve_mask[parse_map == label] = 255
        # Erode preserve mask slightly to avoid sharp edge artifacts
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        preserve_mask = cv2.erode(preserve_mask, erode_kernel, iterations=2)
        agnostic_mask[preserve_mask > 0] = 0

        # Step 5: Smooth mask edges with Gaussian blur then re-threshold
        # This prevents hard pixel-level boundaries in the generated image
        agnostic_mask_smooth = cv2.GaussianBlur(agnostic_mask, (9, 9), 3)
        _, agnostic_mask_final = cv2.threshold(agnostic_mask_smooth, 127, 255, cv2.THRESH_BINARY)

        # Step 6: Build agnostic image — grey out the masked region
        agnostic_image = person_bgr.copy()
        grey_fill = np.ones_like(person_bgr) * 128
        mask_3ch = (agnostic_mask_final[:, :, np.newaxis] / 255.0)
        agnostic_image = (
            person_bgr.astype(np.float32) * (1 - mask_3ch) +
            grey_fill.astype(np.float32) * mask_3ch
        ).astype(np.uint8)

        # Step 7: Debug visualization — overlay mask on original
        vis = person_bgr.copy()
        vis[agnostic_mask_final > 0] = (vis[agnostic_mask_final > 0] * 0.5 +
                                         np.array([0, 100, 200]) * 0.5).astype(np.uint8)

        return {
            'agnostic_mask': agnostic_mask_final,
            'agnostic_image': agnostic_image,
            'agnostic_mask_vis': vis,
        }

    def _build_parse_mask(self, parse_map: np.ndarray, h: int, w: int) -> np.ndarray:
        """Create binary mask from parsing label indices."""
        mask = np.zeros((h, w), dtype=np.uint8)
        for label in self.labels_to_erase:
            mask[parse_map == label] = 255
        return mask

    def _refine_arm_mask(
        self,
        raw_mask: np.ndarray,
        keypoints: list,
        parse_map: np.ndarray,
        h: int,
        w: int
    ) -> np.ndarray:
        """
        Extend arm mask downward from wrist keypoints.
        
        Problem: The parsing model sometimes misses the lower arm/wrist area
        when the arm is raised or in an unusual pose. We use wrist keypoints
        (indices 4=RWrist, 7=LWrist) to draw a small ellipse covering the
        wrist area, ensuring the hand-wrist transition is always masked.
        
        This only applies to upper and overall cloth types.
        """
        mask = raw_mask.copy()

        # Wrist keypoint indices: 4=RWrist, 7=LWrist
        wrist_indices = [4, 7]
        # Elbow keypoint indices: 3=RElbow, 6=LElbow  
        elbow_indices = [3, 6]

        for wrist_idx, elbow_idx in zip(wrist_indices, elbow_indices):
            wrist = keypoints[wrist_idx] if wrist_idx < len(keypoints) else None
            elbow = keypoints[elbow_idx] if elbow_idx < len(keypoints) else None

            if wrist is None or wrist[2] < 0.1:
                continue

            wx, wy = int(wrist[0]), int(wrist[1])

            # Compute arm segment length for ellipse sizing
            if elbow is not None and elbow[2] > 0.1:
                ex, ey = int(elbow[0]), int(elbow[1])
                arm_len = ((wx - ex)**2 + (wy - ey)**2) ** 0.5
                radius = max(15, int(arm_len * 0.35))
            else:
                radius = 25  # fallback radius

            # Draw ellipse at wrist location
            cv2.ellipse(mask, (wx, wy), (radius, radius), 0, 0, 360, 255, -1)

        return mask


def build_agnostic_pair(
    person_bgr: np.ndarray,
    parse_map: np.ndarray,
    keypoints: list | None,
    cloth_type: str = 'upper',
    target_size: tuple | None = None
) -> dict:
    """
    Convenience function: build agnostic pair and optionally resize.
    
    Args:
        person_bgr:  H×W×3 BGR
        parse_map:   H×W uint8
        keypoints:   list of 18 [x,y,conf] or None
        cloth_type:  'upper', 'lower', or 'overall'
        target_size: (W, H) tuple to resize outputs, or None to keep original
    Returns:
        dict with agnostic_mask, agnostic_image, agnostic_mask_vis
    """
    builder = AgnosticBuilder(cloth_type=cloth_type)
    result = builder.build(person_bgr, parse_map, keypoints)

    if target_size is not None:
        tw, th = target_size
        interp_img = cv2.INTER_LINEAR
        interp_mask = cv2.INTER_NEAREST
        result['agnostic_mask'] = cv2.resize(result['agnostic_mask'], (tw, th), interp_mask)
        result['agnostic_image'] = cv2.resize(result['agnostic_image'], (tw, th), interp_img)
        result['agnostic_mask_vis'] = cv2.resize(result['agnostic_mask_vis'], (tw, th), interp_img)

    return result