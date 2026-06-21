"""
backend/main.py — FastAPI VTON backend

Endpoints:
  POST /api/try-on      — full try-on pipeline (upload person + cloth photos)
  POST /api/fit-check   — body/garment measurement fit assessment
  GET  /api/health      — liveness check

VRAM management strategy:
  Models are loaded once at startup and kept in memory.
  The warping model and generation pipeline run sequentially
  (not in parallel) to avoid OOM on T4/RTX GPUs.

Model loading order:
  1. PreprocessingModels (Phase 1) — loaded on CPU, ~800 MB
  2. WarpingModel       (Phase 2) — GPU, ~200 MB
  3. VTONPipeline       (Phase 4) — GPU, ~7 GB
  4. FitService         (Phase 3) — CPU, ~50 MB
"""

import io
import time
import base64
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VTON API",
    description="Virtual Try-On with fit conditioning",
    version="1.0.0",
)

# CORS — allow the React frontend (and HuggingFace Spaces)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────
# Request/Response models
# ─────────────────────────────────────────────────────────────────

class FitCheckRequest(BaseModel):
    # Person body measurements (cm)
    chest:          float
    waist:          float
    hip:            float
    height:         Optional[float] = 165.0
    weight:         Optional[float] = 62.0
    shoulder_width: Optional[float] = 40.0
    # Garment measurements (cm)
    garment_chest:    float
    garment_waist:    float
    garment_hip:      float
    garment_length:   Optional[float] = 65.0
    garment_shoulder: Optional[float] = 40.0
    garment_type:     str = 'upper'   # 'upper' | 'lower' | 'overall'


class FitCheckResponse(BaseModel):
    fit_label:      str
    ease_values:    dict
    region_labels:  dict
    confidence:     float
    recommendation: str


class TryOnResponse(BaseModel):
    result_image_b64: str       # base64-encoded JPEG
    fit_result:       dict
    processing_time_ms: float


# ─────────────────────────────────────────────────────────────────
# Global model registry (loaded once at startup)
# ─────────────────────────────────────────────────────────────────

class ModelRegistry:
    preprocessing = None
    warping_model = None
    vton_pipeline = None
    fit_service   = None
    device        = None
    dtype         = None


registry = ModelRegistry()


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda'), torch.float16
    return torch.device('cpu'), torch.float32


@app.on_event("startup")
async def load_models():
    """Load all models at startup. Called once when the server starts."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    registry.device, registry.dtype = get_device()
    logger.info(f"Device: {registry.device} | dtype: {registry.dtype}")

    # ── Phase 3: Fit service (fast, CPU, load first) ──
    try:
        from fit.ease_calculator import EaseCalculator
        from fit.synthetic_data_gen import FitClassifier

        calc = EaseCalculator()
        clf  = FitClassifier()
        clf.load('fit/checkpoints/fit_classifier.pkl')

        class _FitService:
            def check(self, person_data, garment_data):
                from fit.ease_calculator import PersonMeasurements, GarmentMeasurements
                person  = PersonMeasurements(**person_data)
                garment = GarmentMeasurements(**garment_data)
                ease = calc.compute_ease(person, garment)
                ml_label, ml_conf = clf.predict(
                    ease.chest, ease.waist, ease.hip, garment.garment_type
                )
                result = calc.assess(person, garment, ml_label, ml_conf)
                return {
                    'fit_label':      result.fit_label,
                    'ease_values':    {'chest': round(ease.chest,1),
                                       'waist': round(ease.waist,1),
                                       'hip':   round(ease.hip,  1)},
                    'region_labels':  result.region_labels,
                    'confidence':     round(result.confidence, 2),
                    'recommendation': result.recommendation,
                }

        registry.fit_service = _FitService()
        logger.info("✅ FitService loaded")
    except Exception as e:
        logger.warning(f"FitService failed to load: {e}")

    # ── Phase 1: Preprocessing models ──
    try:
        from preprocessing.human_parser   import HumanParser
        from preprocessing.pose_estimator import load_pose_estimator
        from preprocessing.cloth_segmentor import load_cloth_segmentor, prepare_cloth_for_training
        from preprocessing.agnostic_builder import build_agnostic_pair

        class _PreprocessingModels:
            def __init__(self):
                self.parser   = HumanParser('pretrained/humanparsing/parsing_atr.onnx')
                self.pose_est = load_pose_estimator(mode='controlnet')
                self.cloth_seg = load_cloth_segmentor(mode='rembg')
                self._build_agnostic = build_agnostic_pair
                self._prep_cloth = prepare_cloth_for_training

        registry.preprocessing = _PreprocessingModels()
        logger.info("✅ Preprocessing models loaded")
    except Exception as e:
        logger.warning(f"Preprocessing failed to load: {e}")

    # ── Phase 2: Warping model ──
    try:
        from warping.model import TPSWarpingNet

        ckpt_paths = sorted(Path('warping/checkpoints').glob('*_best.pth'))
        if ckpt_paths:
            ckpt = torch.load(str(ckpt_paths[-1]), map_location=registry.device)
            warp_model = TPSWarpingNet(height=512, width=384).to(registry.device)
            warp_model.load_state_dict(ckpt['model'])
            warp_model.eval()
            registry.warping_model = warp_model
            logger.info("✅ Warping model loaded")
        else:
            logger.warning("No warping checkpoint found — warped_cloth will equal cloth_clean")
    except Exception as e:
        logger.warning(f"Warping model failed to load: {e}")

    # ── Phase 4: VTON generation pipeline ──
    try:
        from diffusers import AutoencoderKL, DDIMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        from generation.unet_modified import VTONUNet
        from generation.pipeline import VTONPipeline, LatentEncoder
        from fit.measurement_encoder import MeasurementEncoder

        ema_path = 'generation/checkpoints/vton_ema_weights.pth'
        if Path(ema_path).exists():
            mid = 'sd-legacy/stable-diffusion-v1-5'
            vae = AutoencoderKL.from_pretrained(
                mid, subfolder='vae', torch_dtype=registry.dtype
            ).to(registry.device)
            text_enc = CLIPTextModel.from_pretrained(
                mid, subfolder='text_encoder', torch_dtype=registry.dtype
            ).to(registry.device)
            tokenizer = CLIPTokenizer.from_pretrained(mid, subfolder='tokenizer')
            scheduler = DDIMScheduler.from_pretrained(mid, subfolder='scheduler')

            unet = VTONUNet.from_pretrained(mid, torch_dtype=registry.dtype,
                                             device=str(registry.device))
            fit_enc = MeasurementEncoder(target_h=512, target_w=384).to(
                device=registry.device, dtype=registry.dtype
            )

            weights = torch.load(ema_path, map_location=registry.device)
            unet.load_state_dict(weights['unet'], strict=False)
            fit_enc.load_state_dict(weights['fit_encoder'])

            registry.vton_pipeline = VTONPipeline(
                unet, vae, text_enc, tokenizer, scheduler, fit_enc,
                device=str(registry.device), dtype=registry.dtype
            )
            logger.info("✅ VTON generation pipeline loaded")
        else:
            logger.warning(f"EMA weights not found: {ema_path}")
    except Exception as e:
        logger.warning(f"VTON pipeline failed to load: {e}")

    logger.info("Startup complete.")


# ─────────────────────────────────────────────────────────────────
# Helper: decode uploaded image
# ─────────────────────────────────────────────────────────────────

def decode_image(upload: UploadFile, target_h: int = 512, target_w: int = 384) -> np.ndarray:
    """Read UploadFile → OpenCV BGR image at target size."""
    contents = upload.file.read()
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {upload.filename}")
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    return img


def tensor_to_b64_jpeg(tensor: torch.Tensor, quality: int = 90) -> str:
    """(1,3,H,W) [-1,1] tensor → base64 JPEG string."""
    t = tensor[0].float().clamp(-1, 1)
    t = ((t + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
    bgr = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode('utf-8')


def bgr_to_tensor(bgr: np.ndarray, device, dtype) -> torch.Tensor:
    """BGR uint8 → (1,3,H,W) [-1,1] tensor."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return (t * 2 - 1).to(device=device, dtype=dtype)


# ─────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "device": str(registry.device),
        "models": {
            "preprocessing": registry.preprocessing is not None,
            "warping":       registry.warping_model is not None,
            "generation":    registry.vton_pipeline is not None,
            "fit_service":   registry.fit_service   is not None,
        }
    }


@app.post("/api/fit-check", response_model=FitCheckResponse)
async def fit_check(req: FitCheckRequest):
    """
    Compute fit label and ease values from body + garment measurements.
    No image required — pure measurement-based assessment.
    """
    if registry.fit_service is None:
        raise HTTPException(503, "Fit service not loaded")

    person_data = {
        'chest': req.chest, 'waist': req.waist, 'hip': req.hip,
        'shoulder_width': req.shoulder_width,
        'height': req.height, 'weight': req.weight,
    }
    garment_data = {
        'garment_chest':    req.garment_chest,
        'garment_waist':    req.garment_waist,
        'garment_hip':      req.garment_hip,
        'garment_length':   req.garment_length,
        'garment_shoulder': req.garment_shoulder,
        'garment_type':     req.garment_type,
    }

    result = registry.fit_service.check(person_data, garment_data)
    return FitCheckResponse(**result)


@app.post("/api/try-on", response_model=TryOnResponse)
async def try_on(
    person_image: UploadFile = File(..., description="Person photo"),
    cloth_image:  UploadFile = File(..., description="Garment product photo"),
    # Body measurements
    chest:          float = Form(90.0),
    waist:          float = Form(74.0),
    hip:            float = Form(96.0),
    height:         float = Form(165.0),
    weight:         float = Form(62.0),
    shoulder_width: float = Form(40.0),
    # Garment measurements
    garment_chest:    float = Form(94.0),
    garment_waist:    float = Form(77.0),
    garment_hip:      float = Form(100.0),
    garment_length:   float = Form(65.0),
    garment_shoulder: float = Form(40.0),
    garment_type:     str   = Form('upper'),
    # Generation params
    num_steps:      int   = Form(50),
    guidance_scale: float = Form(2.0),
    seed:           Optional[int] = Form(None),
):
    """
    Full virtual try-on pipeline.

    1. Preprocess person image (parse, pose, agnostic, densepose)
    2. Preprocess cloth image (segment, prepare)
    3. Run TPS warping model
    4. Run VTON generation pipeline
    5. Compute fit assessment
    6. Return result image (base64 JPEG) + fit label
    """
    if registry.vton_pipeline is None:
        raise HTTPException(503, "VTON pipeline not loaded — check training status")

    t_start = time.time()
    device  = registry.device
    dtype   = registry.dtype

    # ── 1. Decode uploads ──
    person_bgr = decode_image(person_image)
    cloth_bgr  = decode_image(cloth_image)

    # ── 2. Preprocessing ──
    if registry.preprocessing is None:
        raise HTTPException(503, "Preprocessing models not loaded")

    prep = registry.preprocessing

    parse_map   = prep.parser.parse(person_bgr)
    pose_result = prep.pose_est.estimate(person_bgr)
    agnostic    = prep.build_agnostic(
        person_bgr, parse_map, pose_result['keypoints'], cloth_type=garment_type
    )
    seg_result  = prep.cloth_seg.segment(cloth_bgr)
    cloth_clean = prep.prep_cloth(seg_result['cloth_clean'], 512, 384)

    # ── 3. Warp cloth ──
    def to_t(bgr):
        return bgr_to_tensor(bgr, device, dtype)

    cloth_t    = to_t(cloth_clean)
    agnostic_t = to_t(agnostic['agnostic_image'])
    agn_mask_t = torch.from_numpy(
        (agnostic['agnostic_mask'] > 127).astype(np.float32)
    ).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)

    pose_img_t  = to_t(pose_result['pose_img'])

    # DensePose: use zeros if model not available (graceful degradation)
    dense_bgr = np.zeros((512, 384, 3), dtype=np.uint8)
    dense_t   = to_t(dense_bgr)

    if registry.warping_model is not None:
        with torch.no_grad():
            body_input = torch.cat([agnostic_t, pose_img_t, dense_t], dim=1)
            warped_cloth, _, _ = registry.warping_model(
                cloth_t, agnostic_t, pose_img_t, dense_t
            )
    else:
        warped_cloth = cloth_t

    # ── 4. Fit features ──
    from fit.measurement_encoder import normalize_measurements
    person_meas  = {'chest': chest, 'waist': waist, 'hip': hip,
                    'shoulder_width': shoulder_width, 'height': height, 'weight': weight}
    garment_meas = {'garment_chest': garment_chest, 'garment_waist': garment_waist,
                    'garment_hip': garment_hip, 'garment_length': garment_length,
                    'garment_shoulder': garment_shoulder, 'garment_type': garment_type}
    fit_feats = normalize_measurements(person_meas, garment_meas, garment_type)
    fit_feats = fit_feats.unsqueeze(0).to(device=device, dtype=dtype)

    # ── 5. Generation ──
    prompt = f"a photo of a person wearing {garment_type} clothing, high quality"
    with torch.no_grad():
        result = registry.vton_pipeline(
            person_img    = to_t(person_bgr),
            cloth_img     = cloth_t,
            warped_cloth  = warped_cloth,
            agnostic_img  = agnostic_t,
            agnostic_mask = agn_mask_t,
            pose_img      = pose_img_t,
            densepose_img = dense_t,
            fit_features  = fit_feats,
            prompt        = prompt,
            num_steps     = num_steps,
            guidance_scale= guidance_scale,
            seed          = seed,
        )

    result_b64 = tensor_to_b64_jpeg(result)

    # ── 6. Fit assessment ──
    fit_result = {}
    if registry.fit_service:
        fit_result = registry.fit_service.check(
            {**person_meas},
            {**garment_meas}
        )

    t_total = (time.time() - t_start) * 1000
    logger.info(f"try-on complete in {t_total:.0f} ms")

    return TryOnResponse(
        result_image_b64   = result_b64,
        fit_result         = fit_result,
        processing_time_ms = round(t_total, 1),
    )


# ─────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
