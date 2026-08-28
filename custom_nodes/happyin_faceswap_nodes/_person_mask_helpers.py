"""
Person Mask helper functions — extracted from happyin_person_mask.py.

Contains:
  - Tensor/PIL conversion utilities
  - Detail refinement methods (GuidedFilter, PyMatting, VITMatte)
  - Model download helpers (MediaPipe, Pose Landmarker)
  - Pose detection (head+shoulders, upper body)
  - InsightFace person targeting
"""

import os
import math
import cv2
import torch
import numpy as np
from PIL import Image


# ── Tensor/PIL conversion helpers ─────────────────────────────

def _tensor2pil(t: torch.Tensor) -> Image.Image:
    if t.dtype != torch.float32:
        t = t.float()
    return Image.fromarray(
        np.clip(255.0 * t.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


def _pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(
        np.array(image).astype(np.float32) / 255.0
    ).unsqueeze(0)


def _image2mask(image: Image.Image) -> torch.Tensor:
    """PIL image -> single-channel mask tensor (B, H, W)."""
    if image.mode != 'L':
        image = image.convert('L')
    arr = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _mask2image(mask: torch.Tensor) -> Image.Image:
    """Mask tensor -> RGBA PIL image (white foreground, black background)."""
    m = np.clip(255.0 * mask.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    _mask = Image.fromarray(m).convert("L")
    fg = Image.new("RGBA", _mask.size, color='white')
    bg = Image.new("RGBA", _mask.size, color='black')
    return Image.composite(fg, bg, _mask)


def _rgb2rgba(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Combine RGB image with mask as alpha channel."""
    R, G, B = image.convert('RGB').split()
    return Image.merge('RGBA', (R, G, B, mask.convert('L')))


def _histogram_remap(image: torch.Tensor, bp: float, wp: float) -> torch.Tensor:
    """Remap histogram black/white points."""
    bp = min(bp, wp - 0.001)
    scale = 1.0 / (wp - bp)
    arr = image.cpu().numpy().copy()
    arr = np.clip((arr - bp) * scale, 0.0, 1.0)
    return torch.from_numpy(arr)


# ── Detail refinement methods ─────────────────────────────────

def _guided_filter_alpha(image: torch.Tensor, mask: torch.Tensor,
                         filter_radius: int) -> torch.Tensor:
    """Edge-aware alpha via guided filter (requires opencv-contrib).

    guidedFilter(guide, src, radius, eps):
      guide = original photo (edges guide the smoothing)
      src   = mask to refine
      radius = neighbourhood size
      eps    = regularization (higher = more smoothing)
    """
    try:
        from cv2.ximgproc import guidedFilter
    except ImportError:
        print("[HappyinPersonMask] cv2.ximgproc unavailable, skipping guided filter")
        return mask

    radius = max(3, filter_radius)
    eps = 0.04
    mask = _pil2tensor(_tensor2pil(mask).convert('RGB'))
    i_dup = image.cpu().numpy().copy()
    a_dup = mask.cpu().numpy().copy()
    for index, img in enumerate(i_dup):
        i_dup[index] = guidedFilter(img, a_dup[index], radius, eps)
    return torch.from_numpy(i_dup)


def _mask_edge_detail(image: torch.Tensor, mask: torch.Tensor,
                      detail_range: int = 8,
                      bp: float = 0.01, wp: float = 0.99) -> torch.Tensor:
    """Alpha refinement via PyMatting closed-form solver."""
    try:
        from pymatting import fix_trimap, estimate_alpha_cf
    except ImportError:
        print("[HappyinPersonMask] pymatting unavailable, skipping")
        return mask

    d = detail_range * 5 + 1
    mask = _pil2tensor(_tensor2pil(mask).convert('RGB'))
    if d % 2 == 0:
        d += 1
    i_dup = image.cpu().numpy().astype(np.float64)
    a_dup = mask.cpu().numpy().astype(np.float64)
    for index, img in enumerate(i_dup):
        trimap = a_dup[index][:, :, 0]
        if detail_range > 0:
            trimap = cv2.GaussianBlur(trimap, (d, d), 0)
        trimap = fix_trimap(trimap, bp, wp)
        alpha = estimate_alpha_cf(img, trimap,
                                  laplacian_kwargs={"epsilon": 1e-6},
                                  cg_kwargs={"maxiter": 500})
        a_dup[index] = np.stack([alpha, alpha, alpha], axis=-1)
    return torch.from_numpy(a_dup.astype(np.float32))


def _generate_trimap(mask: torch.Tensor, erode_size: int,
                     dilate_size: int) -> Image.Image:
    """Generate trimap from binary mask for VITMatte."""
    m = mask.squeeze(0).cpu().detach().numpy().astype(np.uint8) * 255
    erode_kernel = np.ones((erode_size, erode_size), np.uint8)
    dilate_kernel = np.ones((dilate_size, dilate_size), np.uint8)
    eroded = cv2.erode(m, erode_kernel, iterations=5)
    dilated = cv2.dilate(m, dilate_kernel, iterations=5)
    trimap = np.zeros_like(m)
    trimap[dilated == 255] = 128
    trimap[eroded == 255] = 255
    trimap = trimap.astype(np.float32)
    trimap[trimap == 128] = 0.5
    trimap[trimap == 255] = 1.0
    trimap_t = torch.from_numpy(trimap).unsqueeze(0)
    return _tensor2pil(trimap_t).convert('L')


_vitmatte_cache = {}


def _generate_vitmatte(image: Image.Image, trimap: Image.Image,
                       device: str = "cpu",
                       max_megapixels: float = 2.0,
                       method: str = "VITMatte") -> Image.Image:
    """Run VITMatte matting model (HuggingFace transformers)."""
    try:
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor
    except ImportError:
        print("[HappyinPersonMask] transformers unavailable, returning raw trimap")
        return trimap

    if image.mode != 'RGB':
        image = image.convert('RGB')
    if trimap.mode != 'L':
        trimap = trimap.convert('L')

    max_px = max_megapixels * 1048576
    width, height = image.size
    resized = False
    if width * height > max_px:
        ratio = width / height
        tw = int(math.sqrt(ratio * max_px))
        th = int(tw / ratio)
        image = image.resize((tw, th), Image.BILINEAR)
        trimap = trimap.resize((tw, th), Image.BILINEAR)
        resized = True

    if device == "cuda" and torch.cuda.is_available():
        free_bytes = torch.cuda.mem_get_info()[0]
        if free_bytes < 4 * 1024 ** 3:
            print(f"[HappyinPersonMask] VITMatte: only {free_bytes/1024**3:.1f} GB VRAM free, "
                  f"returning flat mask (skip VITMatte)")
            import numpy as _np
            flat = _np.where(_np.array(trimap) > 0, 255, 0).astype(_np.uint8)
            return Image.fromarray(flat, 'L')
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')

    if method == "vitmatte-base-composition-1k":
        model_name = "hustvl/vitmatte-base-composition-1k"
    else:
        model_name = "hustvl/vitmatte-small-composition-1k"

    if model_name not in _vitmatte_cache:
        print(f"[HappyinPersonMask] Loading VITMatte: {model_name}")
        proc = VitMatteImageProcessor.from_pretrained(model_name)
        model = VitMatteForImageMatting.from_pretrained(model_name)
        _vitmatte_cache[model_name] = (proc, model)

    proc, model = _vitmatte_cache[model_name]

    # Free cached VRAM before VITMatte (generation models may still
    # hold fragmented blocks even after offload).
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.to(dev)
    inputs = proc(images=image, trimaps=trimap, return_tensors="pt")
    with torch.no_grad():
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        predictions = model(**inputs).alphas

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = _tensor2pil(predictions).convert('L')
    result = result.crop((0, 0, image.width, image.height))
    if resized:
        result = result.resize((width, height), Image.BILINEAR)
    return result


# ── MediaPipe model download ─────────────────────────────────

_MP_MODEL = "selfie_multiclass_256x256.tflite"
_MP_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "image_segmenter/selfie_multiclass_256x256/float32/latest/"
    + _MP_MODEL
)


def _ensure_mediapipe_model() -> str:
    """Return path to mediapipe segmenter model, download if missing."""
    import folder_paths

    model_dir = os.path.join(folder_paths.models_dir, "mediapipe")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, _MP_MODEL)

    if not os.path.isfile(model_path):
        print(f"[HappyinPersonMask] Downloading {_MP_MODEL} ...")
        import urllib.request
        urllib.request.urlretrieve(_MP_URL, model_path)
        size_kb = os.path.getsize(model_path) / 1024
        print(f"[HappyinPersonMask] Downloaded ({size_kb:.0f} KB)")

    return model_path


# ── Pose Landmarker for head+shoulders ────────────────────────

_POSE_MODEL = "pose_landmarker_lite.task"
_POSE_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    + _POSE_MODEL
)


def _ensure_pose_model() -> str:
    """Return path to pose landmarker model, download if missing (~3MB)."""
    import folder_paths

    model_dir = os.path.join(folder_paths.models_dir, "mediapipe")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, _POSE_MODEL)

    if not os.path.isfile(model_path):
        print(f"[HappyinPersonMask] Downloading {_POSE_MODEL} ...")
        import urllib.request
        urllib.request.urlretrieve(_POSE_URL, model_path)
        size_kb = os.path.getsize(model_path) / 1024
        print(f"[HappyinPersonMask] Downloaded pose model ({size_kb:.0f} KB)")

    return model_path


def _head_shoulders_region(orig_image, segmented, confidence, h, w,
                           target_bbox=None):
    """Compute head+shoulders bounding region.

    Primary: MediaPipe Pose Landmarker — uses actual shoulder landmarks.
    Fallback: geometric expansion from face bbox.

    target_bbox: (x1, y1, x2, y2) of matched face from InsightFace.
    When provided and multiple poses detected, picks the pose whose
    nose is closest to the target face center.

    Returns (y1, x1, y2, x2) or None if no face detected.
    """
    import mediapipe as mp

    # -- Try Pose Landmarker first --
    try:
        pose_path = _ensure_pose_model()
        with open(pose_path, "rb") as f:
            pose_buffer = f.read()

        pose_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=pose_buffer),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )

        # Convert PIL to MediaPipe image
        np_img = np.array(orig_image)
        if np_img.shape[-1] == 4:
            mp_fmt = mp.ImageFormat.SRGBA
        else:
            mp_fmt = mp.ImageFormat.SRGB
        mp_image = mp.Image(image_format=mp_fmt, data=np_img)

        with mp.tasks.vision.PoseLandmarker.create_from_options(
                pose_options) as landmarker:
            result = landmarker.detect(mp_image)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            # Pick the best pose: closest to target_bbox if provided
            if (target_bbox is not None
                    and len(result.pose_landmarks) > 1):
                tbx1, tby1, tbx2, tby2 = target_bbox
                tcx = (tbx1 + tbx2) / 2
                tcy = (tby1 + tby2) / 2
                best_pose_idx = 0
                best_dist = float('inf')
                for pi, plm in enumerate(result.pose_landmarks):
                    nx = plm[0].x * w  # nose x
                    ny = plm[0].y * h  # nose y
                    d = ((nx - tcx) ** 2 + (ny - tcy) ** 2) ** 0.5
                    if d < best_dist:
                        best_dist = d
                        best_pose_idx = pi
                lm = result.pose_landmarks[best_pose_idx]
                print(f"[HappyinPersonMask] head_shoulders: "
                      f"picked pose {best_pose_idx + 1}/"
                      f"{len(result.pose_landmarks)} "
                      f"(dist={best_dist:.0f}px to target face)")
            else:
                lm = result.pose_landmarks[0]

            # Landmark indices: 0=nose, 7=left_ear, 8=right_ear,
            # 11=left_shoulder, 12=right_shoulder
            nose = lm[0]
            l_shoulder = lm[11]
            r_shoulder = lm[12]

            # Convert normalized coords to pixels
            nose_y = int(nose.y * h)
            ls_x = int(l_shoulder.x * w)
            ls_y = int(l_shoulder.y * h)
            rs_x = int(r_shoulder.x * w)
            rs_y = int(r_shoulder.y * h)

            # Shoulder width and midpoint
            shoulder_w = abs(ls_x - rs_x)
            shoulder_mid_y = (ls_y + rs_y) // 2

            # Head top: estimate from nose — head is ~same distance above
            # nose as nose-to-shoulder distance
            head_top = max(0, nose_y - int((shoulder_mid_y - nose_y) * 0.8))

            # Region: from head_top to below shoulders,
            # width = shoulder span + 30% padding each side
            pad_x = int(shoulder_w * 0.3)
            pad_y_bottom = int(shoulder_w * 0.25)  # bit below shoulders

            region_y1 = head_top
            region_y2 = min(h, shoulder_mid_y + pad_y_bottom)
            region_x1 = max(0, min(ls_x, rs_x) - pad_x)
            region_x2 = min(w, max(ls_x, rs_x) + pad_x)

            print(f"[HappyinPersonMask] head_shoulders: Pose Landmarker OK, "
                  f"shoulders=({ls_x},{ls_y})-({rs_x},{rs_y})")
            return (region_y1, region_x1, region_y2, region_x2)

    except Exception as e:
        print(f"[HappyinPersonMask] Pose Landmarker failed: {e}, "
              f"falling back to face bbox")

    # -- Fallback: geometric expansion from face segmentation bbox --
    face_conf = segmented.confidence_masks[3].numpy_view()
    if face_conf.ndim == 3:
        face_conf = face_conf.squeeze(axis=2)
    face_bin = (face_conf > confidence).astype(np.uint8)

    face_ys, face_xs = np.where(face_bin > 0)
    if len(face_ys) == 0:
        return None

    fy1, fy2 = int(face_ys.min()), int(face_ys.max())
    fx1, fx2 = int(face_xs.min()), int(face_xs.max())
    face_h = fy2 - fy1
    face_w = fx2 - fx1
    face_cx = (fx1 + fx2) // 2

    region_y1 = max(0, fy1 - int(face_h * 0.6))
    region_y2 = min(h, fy2 + int(face_h * 1.2))
    region_x1 = max(0, face_cx - int(face_w * 0.8))
    region_x2 = min(w, face_cx + int(face_w * 0.8))

    print(f"[HappyinPersonMask] head_shoulders: fallback (face bbox), "
          f"face=({fx1},{fy1})-({fx2},{fy2})")
    return (region_y1, region_x1, region_y2, region_x2)


def _get_upper_body_data(orig_image, h, w):
    """Get upper body pose data: hip line + arm keep mask.

    Returns (hip_y, arm_keep_mask) or (None, None) on failure.
    hip_y: int pixel Y of hip line.
    arm_keep_mask: (h, w) float32 — 1.0 where arms are, 0.0 elsewhere.
      Used to preserve arm pixels below hip line.

    Landmarks used:
      11,12 = shoulders, 13,14 = elbows, 15,16 = wrists,
      23,24 = hips.
    """
    import mediapipe as mp

    try:
        pose_path = _ensure_pose_model()
        with open(pose_path, "rb") as f:
            pose_buffer = f.read()

        pose_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_buffer=pose_buffer),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )

        np_img = np.array(orig_image)
        if np_img.shape[-1] == 4:
            mp_fmt = mp.ImageFormat.SRGBA
        else:
            mp_fmt = mp.ImageFormat.SRGB
        mp_image = mp.Image(image_format=mp_fmt, data=np_img)

        with mp.tasks.vision.PoseLandmarker.create_from_options(
                pose_options) as landmarker:
            result = landmarker.detect(mp_image)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lm = result.pose_landmarks[0]

            # Hip line
            l_hip = lm[23]
            r_hip = lm[24]
            hip_y = int(((l_hip.y + r_hip.y) / 2) * h)

            # Arm corridors: shoulder -> elbow -> wrist
            # Draw thick polylines so body pixels near arms are kept
            arm_keep = np.zeros((h, w), dtype=np.float32)

            # Arm width proportional to shoulder span
            l_sh = lm[11]
            r_sh = lm[12]
            shoulder_w = abs(int(l_sh.x * w) - int(r_sh.x * w))
            arm_thickness = max(20, shoulder_w // 3)

            for side_name, sh, el, wr in [
                ("L", lm[11], lm[13], lm[15]),
                ("R", lm[12], lm[14], lm[16]),
            ]:
                pts = np.array([
                    [int(sh.x * w), int(sh.y * h)],
                    [int(el.x * w), int(el.y * h)],
                    [int(wr.x * w), int(wr.y * h)],
                ], dtype=np.int32)
                cv2.polylines(
                    arm_keep, [pts], isClosed=False,
                    color=1.0, thickness=arm_thickness)

            print(f"[HappyinPersonMask] upper_body: hip_y={hip_y}, "
                  f"arm_thickness={arm_thickness}px")
            return hip_y, arm_keep

    except Exception as e:
        print(f"[HappyinPersonMask] upper_body detection failed: {e}")

    return None, None


# ── InsightFace: selfie-based person targeting ─────────────────

_face_app = None


def _ensure_face_app():
    """Load InsightFace ArcFace model for face recognition.

    Auto-downloads buffalo_l (~300 MB) on first run.
    Only loads detection + recognition modules (no landmarks/genderage).
    """
    global _face_app
    if _face_app is not None:
        return _face_app

    import folder_paths
    from insightface.app import FaceAnalysis

    model_root = os.path.join(folder_paths.models_dir, "insightface")
    os.makedirs(model_root, exist_ok=True)

    print("[HappyinPersonMask] Loading InsightFace buffalo_l ...")
    app = FaceAnalysis(
        name="buffalo_l",
        root=model_root,
        allowed_modules=["detection", "recognition"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    _face_app = app
    print("[HappyinPersonMask] InsightFace ready")
    return _face_app


def _isolate_target_person(merged, target_idx, faces, h, w,
                           body_full=False):
    """Keep pixels near target person's face (generous region).

    First-pass only — determines the CROP BBOX, not the final mask.
    Uses generous region (3x face size) to capture head+hair area
    without spanning the entire image when multiple people are present.
    The second pass re-segments the crop for a clean final mask.

    When body_full=True, vertical range is unconstrained (full height)
    to avoid cutting off legs/feet — only horizontal isolation applies.

    Always applies region constraint — even with 1 InsightFace face,
    MediaPipe may segment multiple people whose masks overlap.
    """
    if len(faces) == 0:
        return merged

    tf = faces[target_idx]
    bx1, by1, bx2, by2 = map(int, tf.bbox)
    bx1, by1 = max(0, bx1), max(0, by1)
    bx2, by2 = min(w, bx2), min(h, by2)
    face_h = by2 - by1
    face_w = bx2 - bx1
    expand = max(face_w, face_h)

    if body_full:
        # Full body: no vertical constraint, only horizontal isolation
        ry1 = 0
        ry2 = h
        rx1 = max(0, bx1 - expand * 3)
        rx2 = min(w, bx2 + expand * 3)
    else:
        # Generous region: 3x face size — covers long hair, hats,
        # but doesn't span across the whole image to other people.
        ry1 = max(0, by1 - expand * 3)
        ry2 = min(h, by2 + expand * 2)
        rx1 = max(0, bx1 - expand * 2)
        rx2 = min(w, bx2 + expand * 2)

    merged_bin = (merged[:, :, 3] > 127).astype(np.uint8)
    n_before = int(merged_bin.sum())

    keep = np.zeros((h, w), dtype=np.uint8)
    keep[ry1:ry2, rx1:rx2] = 1
    keep = keep * merged_bin

    result = merged * np.stack([keep] * 4, axis=-1)
    n_after = int(keep.sum())
    print(f"[HappyinPersonMask] Selfie isolation: "
          f"face ({bx1},{by1})-({bx2},{by2}), "
          f"region ({rx1},{ry1})-({rx2},{ry2}), "
          f"kept {n_after}/{n_before}px")

    return result
