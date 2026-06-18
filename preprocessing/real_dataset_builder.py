"""
preprocessing/real_dataset_builder.py — Build train_manifest.json from
real VITON-HD data (Zalando-HD-resized format)

─────────────────────────────────────────────────────────────────
WHY THIS MODULE EXISTS
─────────────────────────────────────────────────────────────────
The synthetic data generator (Phase 1 Cell 6) draws placeholder
rectangles so the pipeline can be smoke-tested without any dataset.
This module replaces that with the REAL VITON-HD dataset, which
ships with this folder structure (Zalando-HD-resized / "zalando-hd-resized"
naming, used by VITON-HD, HR-VITON, StableVITON, IDM-VTON alike):

  <root>/
    train/
      image/                      person photos (.jpg)
      cloth/                      garment photos (.jpg)
      cloth_mask/                 garment binary masks (.jpg, often present)
      image-densepose/            precomputed DensePose RGB renders (.jpg)
      image-parse-v3/             precomputed SCHP parse maps (.png, paletted)
      openpose_img/               precomputed pose skeleton renders (.jpg)
      openpose_json/              precomputed 18-keypoint COCO json
      agnostic-v3.2/              precomputed agnostic person image (.jpg)
      agnostic-mask/               precomputed agnostic mask (.png) [newer releases]
    test/
      ... same structure ...
    train_pairs.txt               "person.jpg cloth.jpg" per line
    test_pairs.txt

─────────────────────────────────────────────────────────────────
STRATEGY: REUSE WHAT'S PRECOMPUTED, FILL GAPS WITH OUR MODELS
─────────────────────────────────────────────────────────────────
Running SCHP + OpenPose + DensePose + rembg on the full ~13,679-pair
training split would take hours on a T4. VITON-HD ships precomputed
parse maps, pose JSON/images, and DensePose renders for exactly this
reason. We use them directly when present, and only fall back to our
own Group A models (preprocessing/human_parser.py etc.) for whatever
is missing — most commonly the agnostic mask/image, since older
dataset releases didn't include it, and the cloth_clean (white-bg
prepared) version, which the raw dataset never includes.

This keeps Phase 1 fast on real data while still producing output in
EXACTLY the same manifest schema the rest of the pipeline expects, so
nothing downstream (Phase 2 warping, Phase 4 generation) needs to change.

─────────────────────────────────────────────────────────────────
PARSE MAP FORMAT CAVEAT
─────────────────────────────────────────────────────────────────
VITON-HD's image-parse-v3/*.png files are PALETTED PNGs where the
visible colors do NOT correspond to small integer label indices when
read naively with cv2.imread (which decodes the palette into RGB).
We must read them as palette-indexed (mode 'P' in PIL) to recover the
actual 0-19 LIP label indices.

These files use the LIP-20 label scheme (NOT ATR-18) — see
preprocessing/human_parser.py and preprocessing/agnostic_builder.py
for the authoritative label orderings and scheme-aware CLOTH_LABELS /
PRESERVE_LABELS dicts. Whenever this module reuses a dataset-provided
parse map, it must tell build_agnostic_fn to use label_scheme='lip'.
When it falls back to our own HumanParser (loaded with
parsing_atr.onnx in the Phase 1 notebook), it must use
label_scheme='atr' instead, since that model outputs ATR indices.
This module tracks which scheme is in play per-sample via the
`parse_scheme` local variable in process_sample() and passes it
through to build_agnostic_fn accordingly.
"""

import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


class RealDatasetBuilder:
    """
    Builds a train_manifest.json compatible with our pipeline from a
    real VITON-HD-format dataset directory, reusing precomputed files
    where available and computing the rest with our Group A models.
    """

    def __init__(
        self,
        viton_root:   str,
        output_root:  str = 'preprocessed_dataset',
        split:        str = 'train',
        cloth_type:   str = 'upper',
        target_h:     int = 512,
        target_w:     int = 384,
        max_samples:  Optional[int] = None,
    ):
        self.viton_root  = Path(viton_root)
        self.output_root = Path(output_root)
        self.split        = split
        self.cloth_type    = cloth_type
        self.target_h, self.target_w = target_h, target_w
        self.max_samples   = max_samples

        self.src = self.viton_root / split
        assert self.src.exists(), f"Missing split dir: {self.src}"

        self.out_dir = self.output_root / split
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Detect which precomputed folders are actually present
        self.available = {
            'parse':      (self.src / 'image-parse-v3').exists(),
            'densepose':  (self.src / 'image-densepose').exists(),
            'pose_img':   (self.src / 'openpose_img').exists(),
            'pose_json':  (self.src / 'openpose_json').exists(),
            'agnostic':   (self.src / 'agnostic-v3.2').exists(),
            'agnostic_mask': (self.src / 'agnostic-mask').exists(),
            'cloth_mask': (self.src / 'cloth_mask').exists(),
        }
        print(f"[RealDatasetBuilder] split={split} | precomputed available:")
        for k, v in self.available.items():
            print(f"    {k:15s}: {'✅' if v else '❌ (will compute)'}")

    # ── Pairs file ──────────────────────────────────────────────

    def _get_pairs(self) -> list[tuple[str, str]]:
        pairs_file = self.viton_root / f'{self.split}_pairs.txt'
        assert pairs_file.exists(), f"Missing pairs file: {pairs_file}"
        pairs = []
        with open(pairs_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
        if self.max_samples:
            pairs = pairs[:self.max_samples]
        print(f"Loaded {len(pairs)} pairs from {pairs_file}")
        return pairs

    # ── Per-file loaders that resize to target ─────────────────────

    def _resize_save_rgb(self, src_path: Path, dst_path: Path,
                          interp=cv2.INTER_LINEAR) -> bool:
        img = cv2.imread(str(src_path))
        if img is None:
            return False
        img = cv2.resize(img, (self.target_w, self.target_h), interpolation=interp)
        cv2.imwrite(str(dst_path), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95] if dst_path.suffix in ('.jpg', '.jpeg') else [])
        return True

    def _resize_save_parse(self, src_path: Path, dst_path: Path) -> bool:
        """
        VITON-HD parse PNGs are palette-indexed. We must open with PIL
        in mode 'P' and read raw indices — NOT cv2.imread, which would
        decode the palette into RGB color triples and destroy the
        integer label semantics our agnostic_builder.py depends on.
        """
        if not src_path.exists():
            return False
        img = Image.open(src_path)
        if img.mode != 'P':
            # Some re-exports flatten to greyscale already holding labels — use as-is
            img = img.convert('L')
        arr = np.array(img, dtype=np.uint8)
        arr_resized = cv2.resize(arr, (self.target_w, self.target_h),
                                 interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(dst_path), arr_resized)
        return True

    # ── Main per-sample processing ─────────────────────────────────

    def process_sample(
        self,
        person_fname: str,
        cloth_fname:  str,
        parser=None, pose_estimator=None, densepose=None,
        cloth_segmentor=None, prepare_cloth_fn=None,
        build_agnostic_fn=None,
    ) -> Optional[dict]:
        """
        Process one (person, cloth) pair, reusing dataset files when
        present and falling back to the supplied Group A model
        instances for anything missing.

        The model arguments are optional — pass None for any model
        you haven't loaded if you're confident the dataset already
        provides everything needed (common for parse/densepose/pose
        on the official release).
        """
        pid = Path(person_fname).stem
        cid = Path(cloth_fname).stem
        out_dir = self.out_dir / pid
        out_dir.mkdir(parents=True, exist_ok=True)

        record = {'person_id': pid, 'cloth_id': cid}

        # ── person.jpg ──
        person_src = self.src / 'image' / person_fname
        person_dst = out_dir / 'person.jpg'
        if not self._resize_save_rgb(person_src, person_dst, cv2.INTER_LANCZOS4):
            print(f"[skip] missing person image: {person_src}")
            return None
        record['person'] = str(person_dst)

        person_bgr = cv2.imread(str(person_dst))

        # ── parse_map.png ──
        parse_dst = out_dir / 'parse_map.png'
        got_parse = False
        if self.available['parse']:
            parse_src = self.src / 'image-parse-v3' / f'{pid}.png'
            got_parse = self._resize_save_parse(parse_src, parse_dst)

        if got_parse:
            # Dataset-provided image-parse-v3 files use the LIP-20 scheme
            parse_scheme = 'lip'
        else:
            assert parser is not None, (
                f"image-parse-v3 missing for {pid} and no parser model supplied"
            )
            parse_map = parser.parse(person_bgr)
            cv2.imwrite(str(parse_dst), parse_map)
            # Our own HumanParser is loaded with parsing_atr.onnx in the
            # Phase 1 notebook, so the fallback output is ATR-18, not LIP.
            # If you load parsing_lip.onnx instead, change this to 'lip'.
            parse_scheme = 'atr'

        record['parse_map'] = str(parse_dst)
        parse_map = cv2.imread(str(parse_dst), cv2.IMREAD_GRAYSCALE)

        # ── pose_img.png + keypoints.json ──
        pose_img_dst = out_dir / 'pose_img.png'
        kp_dst       = out_dir / 'keypoints.json'
        got_pose_img  = False
        got_pose_json = False

        if self.available['pose_img']:
            pose_src = self.src / 'openpose_img' / f'{pid}_rendered.png'
            if not pose_src.exists():
                pose_src = self.src / 'openpose_img' / f'{pid}.png'
            got_pose_img = self._resize_save_rgb(pose_src, pose_img_dst)

        if self.available['pose_json']:
            kp_src = self.src / 'openpose_json' / f'{pid}_keypoints.json'
            if kp_src.exists():
                keypoints = self._convert_openpose_json(kp_src, person_bgr.shape)
                with open(kp_dst, 'w') as f:
                    json.dump({'keypoints': keypoints,
                              'img_h': self.target_h, 'img_w': self.target_w}, f)
                got_pose_json = True

        if not (got_pose_img and got_pose_json):
            assert pose_estimator is not None, (
                f"openpose files missing for {pid} and no pose_estimator supplied"
            )
            pose_result = pose_estimator.estimate(person_bgr)
            if not got_pose_img:
                cv2.imwrite(str(pose_img_dst), pose_result['pose_img'])
            if not got_pose_json:
                with open(kp_dst, 'w') as f:
                    f.write(pose_result['keypoints_json'])

        record['pose_img']      = str(pose_img_dst)
        record['keypoints_json'] = str(kp_dst)

        with open(kp_dst) as f:
            keypoints = json.load(f)['keypoints']

        # ── densepose_iuv.png ──
        dp_dst = out_dir / 'densepose_iuv.png'
        got_dp = False
        if self.available['densepose']:
            dp_src = self.src / 'image-densepose' / person_fname
            if not dp_src.exists():
                dp_src = self.src / 'image-densepose' / f'{pid}.jpg'
            got_dp = self._resize_save_rgb(dp_src, dp_dst, cv2.INTER_NEAREST)
        if not got_dp:
            if densepose is not None:
                dp_result = densepose.estimate(person_bgr)
                cv2.imwrite(str(dp_dst), dp_result['iuv_img'])
            else:
                # Graceful degradation — zero IUV, pipeline still runs
                cv2.imwrite(str(dp_dst),
                           np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8))
        record['densepose_iuv'] = str(dp_dst)

        # ── agnostic_image.png + agnostic_mask.png ──
        agn_img_dst  = out_dir / 'agnostic_image.png'
        agn_mask_dst = out_dir / 'agnostic_mask.png'
        got_agn_img  = False
        got_agn_mask = False

        if self.available['agnostic']:
            agn_src = self.src / 'agnostic-v3.2' / person_fname
            if not agn_src.exists():
                agn_src = self.src / 'agnostic-v3.2' / f'{pid}.jpg'
            got_agn_img = self._resize_save_rgb(agn_src, agn_img_dst)

        if self.available['agnostic_mask']:
            mask_src = self.src / 'agnostic-mask' / f'{pid}_mask.png'
            if not mask_src.exists():
                mask_src = self.src / 'agnostic-mask' / f'{pid}.png'
            got_agn_mask = self._resize_save_parse(mask_src, agn_mask_dst) \
                          if mask_src.exists() else False

        if not (got_agn_img and got_agn_mask):
            assert build_agnostic_fn is not None, (
                f"agnostic files missing for {pid} and no build_agnostic_fn supplied"
            )
            agnostic = build_agnostic_fn(
                person_bgr=person_bgr, parse_map=parse_map,
                keypoints=keypoints, cloth_type=self.cloth_type,
                label_scheme=parse_scheme,
            )
            if not got_agn_img:
                cv2.imwrite(str(agn_img_dst), agnostic['agnostic_image'])
            if not got_agn_mask:
                cv2.imwrite(str(agn_mask_dst), agnostic['agnostic_mask'])

        record['agnostic_image'] = str(agn_img_dst)
        record['agnostic_mask']  = str(agn_mask_dst)

        # ── cloth_clean.png + cloth_mask.png ──
        # The raw dataset's cloth/ images are ALREADY on white background
        # (Zalando product photos), so we still run prepare_cloth_for_training
        # to get the exact tight-crop + 85%-fill format our warping model
        # was designed around, rather than assuming the raw crop is correct.
        cloth_src = self.src / 'cloth' / cloth_fname
        cloth_clean_dst = out_dir / 'cloth_clean.png'
        cloth_mask_dst  = out_dir / 'cloth_mask.png'

        cloth_bgr = cv2.imread(str(cloth_src))
        if cloth_bgr is None:
            print(f"[skip] missing cloth image: {cloth_src}")
            return None

        got_cloth_mask = False
        if self.available['cloth_mask']:
            mask_src = self.src / 'cloth_mask' / cloth_fname
            if not mask_src.exists():
                mask_src = self.src / 'cloth_mask' / f'{cid}.jpg'
            if mask_src.exists():
                mask_img = cv2.imread(str(mask_src), cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    mask_img = cv2.resize(mask_img, (self.target_w, self.target_h),
                                          interpolation=cv2.INTER_NEAREST)
                    _, mask_img = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
                    cv2.imwrite(str(cloth_mask_dst), mask_img)
                    got_cloth_mask = True

        if not got_cloth_mask or prepare_cloth_fn is not None:
            # Always run our own segmentation+prep for cloth_clean (the
            # raw dataset doesn't ship a "prepared" white-canvas version),
            # and use it for cloth_mask too if the dataset didn't have one.
            assert cloth_segmentor is not None and prepare_cloth_fn is not None, (
                f"cloth_mask missing for {cid} and no cloth_segmentor supplied"
            )
            seg = cloth_segmentor.segment(cloth_bgr)
            cloth_clean = prepare_cloth_fn(seg['cloth_clean'], self.target_h, self.target_w)
            cv2.imwrite(str(cloth_clean_dst), cloth_clean)
            if not got_cloth_mask:
                mask_resized = cv2.resize(seg['cloth_mask'], (self.target_w, self.target_h),
                                          interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(cloth_mask_dst), mask_resized)
        else:
            # We had a mask but still need a clean white-canvas cloth image
            cloth_resized = cv2.resize(cloth_bgr, (self.target_w, self.target_h),
                                       interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(cloth_clean_dst), cloth_resized)

        record['cloth_clean'] = str(cloth_clean_dst)
        record['cloth_mask']  = str(cloth_mask_dst)

        return record

    # ── OpenPose JSON parsing ────────────────────────────────────

    @staticmethod
    def _convert_openpose_json(json_path: Path, orig_shape: tuple) -> list:
        """
        VITON-HD's openpose_json files use the OpenPose COCO-18 format:
          {"people": [{"pose_keypoints_2d": [x0,y0,c0, x1,y1,c1, ...]}]}
        Convert to our [[x,y,conf], ...] list of 18, matching the
        format pose_estimator.py produces, so downstream code (agnostic
        builder, dataset loaders) doesn't need to know the source.
        """
        with open(json_path) as f:
            data = json.load(f)
        if not data.get('people'):
            return [None] * 18
        flat = data['people'][0]['pose_keypoints_2d']
        keypoints = []
        for i in range(0, min(len(flat), 18 * 3), 3):
            x, y, c = flat[i], flat[i+1], flat[i+2]
            keypoints.append(None if c < 0.05 else [float(x), float(y), float(c)])
        while len(keypoints) < 18:
            keypoints.append(None)
        return keypoints

    # ── Main run loop ────────────────────────────────────────────

    def run(self, **model_kwargs) -> list[dict]:
        """
        model_kwargs forwarded to process_sample:
          parser, pose_estimator, densepose, cloth_segmentor,
          prepare_cloth_fn, build_agnostic_fn
        Pass None for any model you haven't loaded — an AssertionError
        will tell you clearly if it turns out to be needed.
        """
        from tqdm import tqdm

        pairs = self._get_pairs()
        records = []
        errors  = []

        for person_fname, cloth_fname in tqdm(pairs, desc=f'Building {self.split} manifest'):
            try:
                rec = self.process_sample(person_fname, cloth_fname, **model_kwargs)
                if rec is not None:
                    records.append(rec)
                else:
                    errors.append(person_fname)
            except Exception as e:
                import traceback
                print(f"[ERROR] {person_fname}: {e}")
                traceback.print_exc()
                errors.append(person_fname)

        manifest_path = self.output_root / f'{self.split}_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(records, f, indent=2)

        print(f"\n✅ {len(records)} samples written to {manifest_path}")
        if errors:
            print(f"⚠️  {len(errors)} failed: {errors[:10]}{'...' if len(errors)>10 else ''}")

        return records
