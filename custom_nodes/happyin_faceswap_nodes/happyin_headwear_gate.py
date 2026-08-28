"""
Happyin Headwear Gate — MediaPipe accessory/headwear detection + routing.

Uses the same MediaPipe selfie_multiclass_256x256 model as PersonMask.
Categories: 0=background, 1=hair, 2=body, 3=face, 4=clothes, 5=accessories.

MediaPipe often classifies headwear as "clothes" (cat 4) rather than
"accessories" (cat 5), so we check BOTH in the head zone (chin line and above).

Optional selfie input: when connected, uses InsightFace to find the
specific person (by face embedding + eye landmarks) and checks headwear
only above THEIR eye line.  Without selfie — checks headwear on any
detected face via MediaPipe (original behavior).

Outputs:
  has_headwear → image if headwear detected (1×1 black otherwise)
  no_headwear  → image if head is bare (1×1 black otherwise)
"""

import numpy as np
import torch
import time

from ._person_mask_helpers import _ensure_mediapipe_model, _ensure_face_app


class HappyinHeadwearGate:
    """Detect headwear using MediaPipe segmentation and route.

    Checks accessories (cat 5) + clothes (cat 4) in the head zone.
    Same model as PersonMask — no extra downloads.

    Two outputs:
      has_headwear → processing branch (remove headwear)
      no_headwear  → normal branch (skip)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "min_pixels": ("INT", {
                    "default": 200, "min": 10, "max": 5000, "step": 50,
                    "tooltip": (
                        "Minimum accessory/clothes pixels in head zone "
                        "to count as headwear. Lower = more sensitive."
                    ),
                }),
                "confidence": ("FLOAT", {
                    "default": 0.4, "min": 0.1, "max": 0.9, "step": 0.05,
                    "tooltip": "MediaPipe segmentation confidence threshold.",
                }),
            },
            "optional": {
                "selfie": ("IMAGE", {
                    "tooltip": (
                        "Reference face photo. When connected, uses "
                        "InsightFace to find the matching person and "
                        "checks headwear only above THEIR eye line. "
                        "Without — checks any detected face via MediaPipe."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("has_headwear", "no_headwear", "detected", "info")
    FUNCTION = "run"
    CATEGORY = "Happyin/Router"

    DESCRIPTION = (
        "Headwear Gate — MediaPipe accessory detection\n\n"
        "has_headwear: image if headwear detected (1x1 black otherwise)\n"
        "no_headwear: image if head is bare (1x1 black otherwise)\n"
        "detected: 1 = headwear, 0 = bare head\n\n"
        "Uses same MediaPipe model as PersonMask (no extra download).\n"
        "Checks accessories (cat 5) + clothes (cat 4) from chin upward.\n"
        "MediaPipe classifies hats as 'clothes', so both are checked.\n\n"
        "Optional selfie input: when connected, InsightFace finds the\n"
        "matching person by face embedding, then checks headwear only\n"
        "above their eye line (precise per-person zone)."
    )

    # ── selfie → matched face info via InsightFace ──────────────
    @staticmethod
    def _match_face(selfie_tensor, img_np, W, H):
        """Find the face in img_np that best matches selfie.

        Returns dict with bbox, nose_y, score — or None on failure.
        InsightFace kps: [right_eye, left_eye, nose, right_mouth, left_mouth].
        """
        import cv2

        try:
            face_app = _ensure_face_app()

            # Encode selfie reference
            s_np = (selfie_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            s_bgr = cv2.cvtColor(s_np, cv2.COLOR_RGB2BGR)
            s_faces = face_app.get(s_bgr)
            if not s_faces:
                print("[HappyinHeadwearGate] No face in selfie")
                return None

            ref = max(s_faces,
                      key=lambda f: (f.bbox[2] - f.bbox[0])
                                  * (f.bbox[3] - f.bbox[1]))
            ref_emb = ref.normed_embedding

            # Detect faces in input image
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            det_faces = face_app.get(img_bgr)
            if not det_faces:
                print("[HappyinHeadwearGate] No faces in image")
                return None

            # Find best match by cosine similarity
            scores = [float(np.dot(ref_emb, f.normed_embedding))
                      for f in det_faces]
            best_idx = int(np.argmax(scores))
            best_score = scores[best_idx]

            bf = det_faces[best_idx]
            bx1, by1, bx2, by2 = map(int, bf.bbox)
            bx1 = max(0, bx1)
            by1 = max(0, by1)
            bx2 = min(W, bx2)
            by2 = min(H, by2)

            # Landmarks from InsightFace
            # kps: [right_eye, left_eye, nose, right_mouth, left_mouth]
            nose_y = int(bf.kps[2][1])
            eye_y = int((bf.kps[0][1] + bf.kps[1][1]) / 2)

            scores_str = ", ".join(f"{s:.3f}" for s in scores)
            print(f"[HappyinHeadwearGate] Selfie match: "
                  f"{best_idx + 1}/{len(det_faces)}, "
                  f"score={best_score:.3f} [{scores_str}], "
                  f"bbox=({bx1},{by1})-({bx2},{by2}), "
                  f"eye_y={eye_y} nose_y={nose_y}")

            return {
                "bbox": (bx1, by1, bx2, by2),
                "nose_y": nose_y,
                "eye_y": eye_y,
                "score": best_score,
            }

        except Exception as e:
            print(f"[HappyinHeadwearGate] InsightFace error: {e}")
            return None

    def run(self, image, min_pixels, confidence, selfie=None):
        import mediapipe as mp

        t0 = time.time()
        # 1×1 black tensor for inactive IMAGE outputs.
        # StreamGuard downstream converts it to ExecutionBlocker.
        blocked = torch.zeros(1, 1, 1, 3)

        img_tensor = image[0]  # [H, W, C]
        H, W, _ = img_tensor.shape

        # ── MediaPipe segmentation ──
        img_np = np.ascontiguousarray(
            np.clip(255.0 * img_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
        )
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)

        model_path = _ensure_mediapipe_model()
        with open(model_path, "rb") as f:
            model_buffer = f.read()

        base_options = mp.tasks.BaseOptions(model_asset_buffer=model_buffer)
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            output_category_mask=True,
        )

        with mp.tasks.vision.ImageSegmenter.create_from_options(options) as seg:
            result = seg.segment(mp_image)

        # ── Extract category masks ──
        # 0=background, 1=hair, 2=body, 3=face, 4=clothes, 5=accessories
        face_conf = result.confidence_masks[3].numpy_view()
        acc_conf = result.confidence_masks[5].numpy_view()
        cloth_conf = result.confidence_masks[4].numpy_view()
        hair_conf = result.confidence_masks[1].numpy_view()

        for arr_name in ("face_conf", "acc_conf", "cloth_conf", "hair_conf"):
            arr = locals()[arr_name]
            if arr.ndim == 3:
                locals()[arr_name] = arr.squeeze(axis=2)
        # Re-bind after possible squeeze
        face_conf = face_conf.squeeze(axis=2) if face_conf.ndim == 3 else face_conf
        acc_conf = acc_conf.squeeze(axis=2) if acc_conf.ndim == 3 else acc_conf
        cloth_conf = cloth_conf.squeeze(axis=2) if cloth_conf.ndim == 3 else cloth_conf
        hair_conf = hair_conf.squeeze(axis=2) if hair_conf.ndim == 3 else hair_conf

        hair_bin = (hair_conf > confidence).astype(np.uint8)
        acc_bin = (acc_conf > confidence).astype(np.uint8)
        cloth_bin = (cloth_conf > confidence).astype(np.uint8)

        # ── Determine head zone ──
        # Two paths: selfie (InsightFace eye line) vs MediaPipe (any face).
        matched = None
        if selfie is not None:
            matched = self._match_face(selfie, img_np, W, H)

            if matched is None:
                # Selfie connected but face match failed →
                # safe default: no headwear (don't guess on wrong person).
                info = "selfie match failed -> no_headwear (safe default)"
                ms = (time.time() - t0) * 1000
                print(f"[HappyinHeadwearGate] {info} ({ms:.0f}ms)")
                return (blocked, image, 0, info)

        if matched is not None:
            # ── InsightFace path: zone from face bbox ──
            bx1, by1, bx2, by2 = matched["bbox"]
            eye_y = matched["eye_y"]
            face_w = bx2 - bx1
            face_h = by2 - by1
            face_cx = (bx1 + bx2) // 2

            # Zone = tight box around head
            # Horizontal: face center ± 1x face_w
            zone_left = max(0, face_cx - face_w)
            zone_right = min(W, face_cx + face_w)
            # Vertical: from above head to eye level
            zone_top = max(0, by1 - int(face_h * 1.5))
            zone_bottom = eye_y

            if zone_bottom <= zone_top:
                info = "zone empty -> no_headwear"
                ms = (time.time() - t0) * 1000
                print(f"[HappyinHeadwearGate] {info} ({ms:.0f}ms)")
                return (blocked, image, 0, info)

            head_zone = np.zeros((H, W), dtype=np.uint8)
            head_zone[zone_top:zone_bottom, zone_left:zone_right] = 1

            # Exclude hair — hair above face is normal, not headwear
            head_zone = head_zone & (~hair_bin)

            face_tag = (f"face=({bx1},{by1})-({bx2},{by2}) "
                        f"eye_y={eye_y} "
                        f"score={matched['score']:.3f} "
                        f"zone=({zone_left},{zone_top})-"
                        f"({zone_right},{zone_bottom})")
            selfie_tag = " [selfie]"

        else:
            # ── MediaPipe path: zone from face segmentation ──
            face_bin = (face_conf > confidence).astype(np.uint8)
            face_rows = np.where(face_bin.any(axis=1))[0]

            if len(face_rows) == 0:
                info = "no face detected -> no_headwear (default)"
                ms = (time.time() - t0) * 1000
                print(f"[HappyinHeadwearGate] {info} ({ms:.0f}ms)")
                return (blocked, image, 0, info)

            face_top = int(face_rows[0])
            chin_y = int(face_rows[-1])
            face_cols = np.where(face_bin.any(axis=0))[0]
            face_left = int(face_cols[0])
            face_right = int(face_cols[-1])
            face_w = face_right - face_left
            face_cx = (face_left + face_right) // 2

            # Zone = proportional to face size
            zone_left = max(0, face_cx - face_w)
            zone_right = min(W, face_cx + face_w)
            # Vertical: top of face + 15% (forehead line)
            zone_bottom = min(H, face_top + int((chin_y - face_top) * 0.15))

            head_zone = np.zeros((H, W), dtype=np.uint8)
            head_zone[:zone_bottom, zone_left:zone_right] = 1

            # Exclude hair
            head_zone = head_zone & (~hair_bin)

            face_tag = (f"face=({face_left},{face_top})-({face_right},{chin_y}) "
                        f"zone=({zone_left},0)-({zone_right},{zone_bottom})")
            selfie_tag = ""

        # ── Count accessories + clothes pixels in head zone ──
        acc_in_head = int((acc_bin & head_zone).sum())
        cloth_in_head = int((cloth_bin & head_zone).sum())
        total_headwear_px = acc_in_head + cloth_in_head
        zone_px = int(head_zone.sum())

        headwear = total_headwear_px >= min_pixels

        ms = (time.time() - t0) * 1000
        verdict = "HEADWEAR" if headwear else "BARE"
        info = (f"{verdict}{selfie_tag} acc={acc_in_head}px "
                f"cloth_head={cloth_in_head}px "
                f"total={total_headwear_px}px min={min_pixels} "
                f"zone={zone_px}px {face_tag} {ms:.0f}ms")
        print(f"[HappyinHeadwearGate] {W}x{H} {info}")

        if headwear:
            return (image, blocked, 1, info)
        else:
            return (blocked, image, 0, info)
