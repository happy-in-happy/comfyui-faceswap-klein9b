"""
Happyin Person Mask — MediaPipe person segmentation.

6 selectable body parts: face, hair, body, clothes, accessories, background.
Detail refinement: VITMatte / PyMatting / GuidedFilter.

Outputs:
  image     — RGB crop (same as crop, controlled by crop_padding)
  mask      — crop-space mask (same as crop_mask)
  crop      — RGB bounding box crop of original (no alpha, for processing)
  crop_mask — mask cropped to same bbox (same size as crop)
  crop_data — all-in-one for PasteBack (original + mask + bbox)

Model selfie_multiclass_256x256.tflite auto-downloads on first run.
"""

import os
import math
import cv2
import torch
import numpy as np
from PIL import Image
from functools import reduce

from .nodes import LATENT_SIZES as _LATENT_SIZES
from ._person_mask_helpers import (
    _tensor2pil, _pil2tensor, _image2mask, _mask2image,
    _rgb2rgba, _histogram_remap,
    _guided_filter_alpha, _mask_edge_detail,
    _generate_trimap, _generate_vitmatte,
    _ensure_mediapipe_model, _ensure_pose_model,
    _head_shoulders_region, _get_upper_body_data,
    _ensure_face_app, _isolate_target_person,
)


# ── Node class ────────────────────────────────────────────────

class HappyinPersonMask:
    """Segment people using MediaPipe multiclass model.

    Selectable body parts: face, hair, body, clothes, accessories, background.
    Detail refinement: VITMatte / PyMatting / GuidedFilter.

    Outputs: RGBA cutout, mask, and original image for compositing back.
    Model auto-downloads on first run.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "face": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled", "label_off": "disabled",
                    "tooltip": "Лицо — кожа лица, брови, губы.",
                }),
                "hair": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled", "label_off": "disabled",
                    "tooltip": "Волосы — все волосы на голове, включая длинные.",
                }),
                "head_shoulders": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled", "label_off": "disabled",
                    "tooltip": "Расширяет КРОП до головы+плеч (через Pose Landmarker).\n"
                               "НЕ добавляет пиксели в маску — только увеличивает область выреза.\n"
                               "Полезно для inpaint: кроп захватывает контекст шеи и плеч.",
                }),
                "body": (["off", "upper", "crop", "full"], {
                    "default": "off",
                    "tooltip": "Тело — открытая кожа (руки, шея, декольте).\n"
                               "off — тело не включается в маску.\n"
                               "upper — только выше линии бёдер (Pose Landmarker).\n"
                               "crop — тело только внутри зоны кропа.\n"
                               "full — всё тело целиком.",
                }),
                "clothes": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled", "label_off": "disabled",
                    "tooltip": "Одежда — рубашки, штаны, платья и т.д.",
                }),
                "accessories": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled", "label_off": "disabled",
                    "tooltip": "Аксессуары — очки, серьги, украшения.\n"
                               "При выключенном strict: очки/шляпы рядом с лицом "
                               "добавляются автоматически.",
                }),
                "background": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled", "label_off": "disabled",
                    "tooltip": "Фон — всё что не является человеком.\n"
                               "Инверсия маски: выделяет окружение вместо персонажа.",
                }),
                "auto_accessories": ("BOOLEAN", {
                    "default": True,
                    "label_on": "auto", "label_off": "manual",
                    "tooltip": "auto: очки, шляпы, головные уборы рядом с лицом "
                               "добавляются в маску автоматически.\n"
                               "manual: только выбранные категории, без авто-добавления.",
                }),
                "mask_expand": ("INT", {
                    "default": 0, "min": 0, "max": 200, "step": 1,
                    "tooltip": "Расширение маски в пикселях.\n"
                               "Увеличивает область сегментации на N px от края.\n"
                               "Полезно когда маска чуть не дотягивает до контура.\n"
                               "0 = без расширения.",
                }),
                "crop_mode": (["crop", "disabled"], {
                    "default": "crop",
                    "tooltip": "crop — обычный кроп по bbox маски.\n"
                               "disabled — без кропа, маска в масштабе исходного изображения.\n"
                               "crop/crop_mask = полное изображение/маска.",
                }),
                "crop_padding_mode": (["px", "%"], {
                    "default": "px",
                    "tooltip": "Режим отступа вокруг кропа:\n"
                               "px — абсолютный (фиксированные пиксели).\n"
                               "% — относительный (процент от размера bbox маски).",
                }),
                "crop_padding": ("INT", {
                    "default": 50, "min": 0, "max": 500, "step": 5,
                    "tooltip": "Отступ вокруг кропа в пикселях (режим px).\n"
                               "Не влияет на маску — только добавляет контекст вокруг выреза.\n"
                               "Больше = больше фона вокруг объекта в кропе.",
                }),
                "crop_padding_pct": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Отступ вокруг кропа в % от размера bbox маски (режим %).\n"
                               "10% = отступ равен 10% от большей стороны bbox.\n"
                               "Масштабируется с размером изображения.",
                }),
                "portrait": (["off", "auto", "force"], {
                    "default": "auto",
                    "tooltip": "Определяет крупные планы лица (портреты).\n"
                               "auto: если лицо занимает >5% кадра — портрет,\n"
                               "  crop = полное изображение, маска = белая.\n"
                               "force: всегда считает портретом (без кропа).\n"
                               "off: всегда кропает + отключает 2-й проход сегментации.",
                }),
                "matting_method": (
                    ["VITMatte", "PyMatting", "GuidedFilter"],
                    {"default": "VITMatte",
                     "tooltip": "Метод уточнения краёв маски (матинг):\n"
                                "VITMatte — нейросеть, лучшее качество.\n"
                                "PyMatting — алгоритмический, быстрый, хорошее качество.\n"
                                "GuidedFilter — самый быстрый, базовое качество."},
                ),
                "matting": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled", "label_off": "disabled",
                    "tooltip": "Матинг краёв маски.\n"
                               "Вкл: плавные, точные края (волосы, мех, прозрачность).\n"
                               "Выкл: быстрая грубая маска без уточнения.",
                }),
                "square_crop": (["off", "crop", "pad", "smart"], {
                    "default": "off",
                    "tooltip": "Квадратный кроп для head swap референсов.\n"
                               "off: обычный прямоугольный кроп.\n"
                               "crop: расширяет короткую сторону пикселями из оригинала,\n"
                               "обрезает только если упёрся в край картинки.\n"
                               "pad: дополняет короткую сторону чёрными пикселями до квадрата.\n"
                               "smart: обрезает длинную сторону до квадрата, паддинг не добавляется.",
                }),
                "latent_snap": (
                    ["off"] + list(_LATENT_SIZES.keys()), {
                    "default": "off",
                    "tooltip": "Подогнать аспект кропа под ближайший латент модели.\n"
                               "Расширяет bbox (добавляет контекст), никогда не обрезает.\n"
                               "Потом LatentSizeSnap масштабирует без искажения.\n"
                               "Перекрывает square_crop если активен.",
                }),
                "remove_islands": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled", "label_off": "disabled",
                    "tooltip": "Удаляет отдельные мелкие фрагменты маски.\n"
                               "Оставляет только главную связную область.\n"
                               "Полезно когда кусочки рук, кистей или тела\n"
                               "попадают в маску отдельно от основной области.",
                }),
            },
            "optional": {
                "selfie": ("IMAGE", {
                    "tooltip": "Фото лица для поиска конкретного человека.\n"
                               "Подключите — и на групповом снимке будет\n"
                               "сегментирован только этот человек.\n"
                               "При первом запуске скачивает InsightFace (~300 МБ).",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "MASK", "CROP_DATA")
    RETURN_NAMES = ("image", "mask", "crop", "crop_mask", "crop_data")
    FUNCTION = "run"
    CATEGORY = "Happyin/Mask"

    DESCRIPTION = (
        "Person Mask — сегментация людей через MediaPipe.\n\n"
        "ВЫХОДЫ:\n"
        "• image — RGB-кроп (= crop, управляется crop_padding).\n"
        "• mask — маска в пространстве кропа (= crop_mask).\n"
        "• crop — RGB-вырез по bbox (для img2img / inpaint).\n"
        "• crop_mask — маска внутри кропа (тот же размер).\n"
        "• crop_data — пакет для PasteBack (оригинал + маска + bbox).\n\n"
        "РЕЖИМЫ:\n"
        "• selfie подключён → двухпроходная сегментация:\n"
        "  1-й проход: находит персону + определяет область.\n"
        "  2-й проход: чистая маска без артефактов.\n"
        "• portrait=auto → крупные планы: crop = полное фото.\n"
        "• matting=off → быстрая грубая маска.\n\n"
        "Модели скачиваются автоматически при первом запуске."
    )

    def _get_mediapipe_image(self, image_pil: Image.Image):
        import mediapipe as mp
        numpy_image = np.asarray(image_pil)
        if numpy_image.shape[-1] == 4:
            fmt = mp.ImageFormat.SRGBA
        else:
            fmt = mp.ImageFormat.SRGB
            # PIL is already RGB — no conversion needed
        return mp.Image(image_format=fmt, data=numpy_image)

    def run(self, images, face, hair, head_shoulders, body, clothes,
            accessories, background, auto_accessories,
            mask_expand, crop_mode, crop_padding_mode, crop_padding, crop_padding_pct,
            portrait,
            matting_method, matting, square_crop, latent_snap,
            remove_islands, selfie=None, crop_align="16"):

        import mediapipe as mp

        # ── Hardcoded parameters (removed from UI) ──
        confidence = 0.4       # segmentation confidence threshold
        detail_erode = 6       # mask erosion for matting zone
        edge_width = 6         # matting edge zone (auto-scaled by resolution)
        black_point = 0.01     # histogram black point
        white_point = 0.99     # histogram white point
        device = "cuda"        # VITMatte device
        max_megapixels = 2.0   # VITMatte resolution limit

        # Latent factor: ×8 for SD/SDXL VAE, ×16 for Flux/Qwen
        _LATENT_F = {"SDXL": 8, "SD 1.5": 8}.get(latent_snap, 16)

        model_path = _ensure_mediapipe_model()
        with open(model_path, "rb") as f:
            model_buffer = f.read()

        base_options = mp.tasks.BaseOptions(model_asset_buffer=model_buffer)
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            output_category_mask=True,
        )

        ret_images = []
        ret_masks = []
        ret_crops = []
        ret_crop_masks = []
        ret_bboxes = []
        ret_crop_sizes = []   # (h, w) actual crop dims (after padding)
        ret_pad_info = []     # (pad_y, pad_x, orig_h, orig_w) or None
        ret_trim_info = []    # (y_off, x_off, side) or None

        # MediaPipe category indices:
        # 0=background, 1=hair, 2=body, 3=face, 4=clothes, 5=accessories
        # Body (cat 2) handled separately per body mode.
        # Accessories (cat 5) deferred when head_shoulders is on
        # to restrict them to the hs_region (avoid purse/shoes at feet).
        _acc_deferred = head_shoulders and accessories
        category_flags = {
            0: background, 1: hair,
            3: face, 4: clothes,
            5: False if _acc_deferred else accessories,
        }
        body_enabled = body != "off"

        # -- Selfie-based face matching: encode reference face --
        ref_embedding = None
        face_app_inst = None
        if selfie is not None:
            try:
                face_app_inst = _ensure_face_app()
                s_np = (selfie[0].cpu().numpy() * 255).astype(np.uint8)
                s_bgr = cv2.cvtColor(s_np, cv2.COLOR_RGB2BGR)
                s_faces = face_app_inst.get(s_bgr)
                if s_faces:
                    ref = max(s_faces,
                              key=lambda f: (f.bbox[2] - f.bbox[0])
                                          * (f.bbox[3] - f.bbox[1]))
                    ref_embedding = ref.normed_embedding
                    print(f"[HappyinPersonMask] Selfie encoded, "
                          f"{len(s_faces)} face(s) detected")
                else:
                    print("[HappyinPersonMask] No face in selfie, "
                          "fallback to default segmentation")
            except Exception as e:
                print(f"[HappyinPersonMask] InsightFace failed: {e}, "
                      "fallback to default segmentation")

        with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
            for img_tensor in images:
                # Single conversion: tensor → PIL RGB (used for both MediaPipe and detail refinement)
                orig_image = Image.fromarray(
                    np.clip(255.0 * img_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
                ).convert('RGB')

                # Segment
                mp_image = self._get_mediapipe_image(orig_image)
                segmented = segmenter.segment(mp_image)

                # Collect enabled category masks
                masks = []
                for idx, enabled in category_flags.items():
                    if enabled:
                        masks.append(segmented.confidence_masks[idx])

                image_data = mp_image.numpy_view()
                h, w = image_data.shape[:2]
                channels = 4
                image_shape = (h, w, channels)

                fg = np.zeros(image_shape, dtype=np.uint8)
                fg[:] = (255, 255, 255, 255)
                bg = np.zeros(image_shape, dtype=np.uint8)
                bg[:] = (0, 0, 0, 0)

                if not masks:
                    mask_arrays = [bg]
                else:
                    mask_arrays = []
                    for mask in masks:
                        m2d = mask.numpy_view()
                        if m2d.ndim == 3 and m2d.shape[2] == 1:
                            m2d = m2d.squeeze(axis=2)
                        elif m2d.ndim != 2:
                            raise ValueError(
                                f"Unexpected mask shape: {m2d.shape}")
                        condition = (
                            np.stack((m2d,) * channels, axis=-1) > confidence
                        )
                        if condition.ndim == 4 and condition.shape[2] == 1:
                            condition = condition.squeeze(2)
                        mask_arr = np.where(condition, fg, bg)
                        mask_arrays.append(mask_arr)

                # Merge masks (maximum of all selected categories)
                merged = reduce(np.maximum, mask_arrays)

                # ── Body mode handling (separate from category_flags) ──
                _first_pass_hip_y = None
                _body_conf_for_crop = None

                if body_enabled:
                    body_conf = segmented.confidence_masks[2].numpy_view()
                    if body_conf.ndim == 3 and body_conf.shape[2] == 1:
                        body_conf = body_conf.squeeze(axis=2)

                    if body == "full":
                        # Include all body pixels
                        body_cond = (
                            np.stack((body_conf,) * channels, axis=-1)
                            > confidence
                        )
                        body_rgba = np.where(body_cond, fg, bg)
                        merged = np.maximum(merged, body_rgba)

                    elif body == "upper":
                        # Include body above hip + full arms
                        _first_pass_hip_y, _arm_keep = (
                            _get_upper_body_data(orig_image, h, w))
                        body_conf_upper = body_conf.copy()
                        if (_first_pass_hip_y is not None
                                and 0 < _first_pass_hip_y < h):
                            # Cut below hip, then restore arm pixels
                            body_conf_upper[_first_pass_hip_y:, :] = 0.0
                            if _arm_keep is not None:
                                # Re-include body near arm landmarks
                                arm_below = _arm_keep[
                                    _first_pass_hip_y:, :]
                                body_conf_upper[
                                    _first_pass_hip_y:, :
                                ] = np.where(
                                    arm_below > 0.5,
                                    body_conf[_first_pass_hip_y:, :],
                                    0.0)
                            print(f"[HappyinPersonMask] body=upper: "
                                  f"hip_y={_first_pass_hip_y}, "
                                  f"arms preserved")
                        else:
                            _arm_keep = None
                            print("[HappyinPersonMask] body=upper: "
                                  "pose not detected, using full body")
                        body_cond = (
                            np.stack((body_conf_upper,) * channels,
                                     axis=-1)
                            > confidence
                        )
                        body_rgba = np.where(body_cond, fg, bg)
                        merged = np.maximum(merged, body_rgba)

                    elif body == "crop":
                        # Defer: body must NOT influence bbox
                        _body_conf_for_crop = body_conf.copy()

                img_h, img_w = img_tensor.shape[0], img_tensor.shape[1]
                total_px = img_h * img_w

                # ── Early portrait detection ──────────────────────
                # Check BEFORE accessories/detail refinement for fast exit.
                # Portrait = close-up face, no cropping needed.
                _is_portrait = False
                if portrait == "force":
                    _is_portrait = True
                elif portrait == "auto" and (face or hair):
                    face_conf = segmented.confidence_masks[3].numpy_view()
                    if face_conf.ndim == 3:
                        face_conf = face_conf.squeeze(axis=2)
                    face_px = int((face_conf > confidence).sum())
                    face_ratio = face_px / max(1, total_px)

                    hair_conf = segmented.confidence_masks[1].numpy_view()
                    if hair_conf.ndim == 3:
                        hair_conf = hair_conf.squeeze(axis=2)
                    fh_px = face_px + int((hair_conf > confidence).sum())
                    fh_ratio = fh_px / max(1, total_px)

                    face_bin = (face_conf > confidence)
                    face_ys, face_xs = np.where(face_bin)
                    if len(face_ys) > 0:
                        face_cy = float(face_ys.mean()) / h
                        face_cx = float(face_xs.mean()) / w
                        centered = (0.25 < face_cx < 0.75
                                    and 0.15 < face_cy < 0.65)
                    else:
                        centered = False

                    _is_portrait = (
                        (face_ratio > 0.05 and fh_ratio > 0.15 and centered)
                        or face_ratio > 0.10
                    )

                # NOTE: portrait suppression removed — when face is large
                # enough for portrait, white mask is always correct
                # (head_swap needs full-frame white mask even with
                # head_shoulders=enabled and body=crop).

                if _is_portrait:
                    # Fast path: portrait = full frame is the crop.
                    # ALL masks are WHITE — entire image gets replaced.
                    # No segmentation shape needed (PasteBack uses
                    # portrait passthrough when bbox >= 80% of image).
                    white_mask = Image.new('L', (img_w, img_h), 255)
                    ret_image = _rgb2rgba(orig_image, white_mask)
                    ret_images.append(_pil2tensor(ret_image))
                    final_mask_t = torch.ones(1, img_h, img_w)
                    ret_masks.append(final_mask_t)

                    # Full image as crop (latent-aligned for VAE)
                    aligned_h = (img_h // _LATENT_F) * _LATENT_F
                    aligned_w = (img_w // _LATENT_F) * _LATENT_F
                    crop = img_tensor[:aligned_h, :aligned_w, :]
                    crop_mask_t = torch.ones(aligned_h, aligned_w)
                    ret_crops.append(crop.unsqueeze(0))
                    ret_crop_masks.append(crop_mask_t.unsqueeze(0))
                    ret_bboxes.append((0, 0, aligned_h, aligned_w))
                    ret_crop_sizes.append((aligned_h, aligned_w))
                    ret_pad_info.append(None)   # no padding
                    ret_trim_info.append(None)  # no trim

                    if portrait == "force":
                        print("[HappyinPersonMask] Portrait fast path: "
                              "mode=force → skip detail/accessories")
                    else:
                        print(f"[HappyinPersonMask] Portrait fast path: "
                              f"face={face_ratio:.1%}, "
                              f"face+hair={fh_ratio:.1%}, "
                              f"centered={centered} "
                              f"→ skip detail/accessories")
                    continue

                # -- Selfie targeting: isolate the matching person --
                #    MUST run BEFORE auto-accessories, otherwise accessories
                #    connect all people's masks into one blob and connected
                #    components can't separate them.
                _matched_face_bbox = None  # (x1,y1,x2,y2) of matched face
                if ref_embedding is not None and face_app_inst is not None:
                    try:
                        img_np = (img_tensor.cpu().numpy() * 255).astype(
                            np.uint8)
                        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                        det_faces = face_app_inst.get(img_bgr)
                        if det_faces:
                            scores = [
                                float(np.dot(ref_embedding,
                                             f.normed_embedding))
                                for f in det_faces
                            ]
                            best_idx = int(np.argmax(scores))
                            bf = det_faces[best_idx]
                            _matched_face_bbox = (
                                max(0, int(bf.bbox[0])),
                                max(0, int(bf.bbox[1])),
                                min(w, int(bf.bbox[2])),
                                min(h, int(bf.bbox[3])),
                            )
                            print(
                                f"[HappyinPersonMask] Face match: "
                                f"{best_idx + 1}/{len(det_faces)}, "
                                f"score={scores[best_idx]:.3f}, "
                                f"bbox={_matched_face_bbox}")
                            merged = _isolate_target_person(
                                merged, best_idx, det_faces, h, w,
                                body_full=(body == "full"))
                        else:
                            print("[HappyinPersonMask] No faces detected "
                                  "in image, using full mask")
                    except Exception as e:
                        print(f"[HappyinPersonMask] Face match error: {e}, "
                              "using full mask")

                # -- Head + shoulders: compute EARLY so auto_accessories
                #    can be constrained to hs_region (avoids spreading
                #    to other people on group photos).
                _hs_region = None
                if head_shoulders:
                    hs_region = _head_shoulders_region(
                        orig_image, segmented, confidence, h, w,
                        target_bbox=_matched_face_bbox)
                    if hs_region is not None:
                        _hs_region = hs_region
                        print(f"[HappyinPersonMask] head_shoulders: "
                              f"bbox expand to "
                              f"y={hs_region[0]}..{hs_region[2]}, "
                              f"x={hs_region[1]}..{hs_region[3]}")

                # -- Auto-include accessories/clothes in HEAD ZONE --
                #    Accessories (cat 5) AND clothes (cat 4) that overlap
                #    with the face/hair zone are auto-included.
                #    MediaPipe often classifies hats/balaclavas as "clothes"
                #    rather than "accessories", so both must be checked.
                #    Dilation is proportional to face size (not fixed px).
                #    When hs_region is available, constrain auto-accessories
                #    to that zone + margin (prevents spreading to other people).
                #    (only when auto_accessories=True)
                if (face or hair) and auto_accessories:
                    fh_2d = (merged[:, :, 3] > 127).astype(np.uint8)
                    fh_ys = np.where(fh_2d.any(axis=1))[0]

                    if len(fh_ys) > 0:
                        # Dilation proportional to face/hair height
                        # (covers tall hats, balaclavas, headwear)
                        fh_height = int(fh_ys[-1] - fh_ys[0])
                        dilate_px = max(31, int(fh_height * 0.5))
                        # distanceTransform: O(n) — instant for any radius
                        # (cv2.dilate with 1535x1535 kernel was very slow)
                        dist = cv2.distanceTransform(
                            1 - fh_2d, cv2.DIST_L2, 5)
                        fh_dilated = (dist <= dilate_px).astype(np.uint8)

                        # Constrain dilated zone to hs_region + margin
                        # when head_shoulders is active (prevents
                        # accessories from other people being included).
                        if _hs_region is not None:
                            hy1, hx1, hy2, hx2 = _hs_region
                            hs_margin = int(max(hy2 - hy1, hx2 - hx1)
                                            * 0.3)
                            hs_zone = np.zeros_like(fh_dilated)
                            zy1 = max(0, hy1 - hs_margin)
                            zy2 = min(h, hy2 + hs_margin)
                            zx1 = max(0, hx1 - hs_margin)
                            zx2 = min(w, hx2 + hs_margin)
                            hs_zone[zy1:zy2, zx1:zx2] = 1
                            fh_dilated = fh_dilated & hs_zone
                            print(f"[HappyinPersonMask] Auto-acc "
                                  f"constrained to hs_region+"
                                  f"{hs_margin}px margin")

                        # Check accessories (cat 5) — glasses, earrings,
                        # hair clips, pins.  Use a lower confidence floor
                        # (0.2) so small accessories partially occluded by
                        # hair (e.g. zakolki in an updo) are caught even
                        # when their confidence is below the main threshold.
                        if not accessories:
                            acc_conf = segmented.confidence_masks[5].numpy_view()
                            if acc_conf.ndim == 3:
                                acc_conf = acc_conf.squeeze(axis=2)
                            acc_thr = min(confidence, 0.2)
                            acc_bin = (acc_conf > acc_thr).astype(np.uint8)
                            acc_intersect = acc_bin & fh_dilated

                            if acc_intersect.any():
                                acc_rgba = np.zeros_like(merged)
                                acc_rgba[acc_intersect > 0] = (
                                    255, 255, 255, 255)
                                merged = np.maximum(merged, acc_rgba)
                                n_px = int(acc_intersect.sum())
                                print(f"[HappyinPersonMask] Auto-added "
                                      f"{n_px} accessory px "
                                      f"(intersect with face/hair, "
                                      f"dilate={dilate_px}px, "
                                      f"acc_thr={acc_thr:.2f})")

                        # Check clothes (cat 4) in HEAD zone only —
                        # hats/balaclavas/headbands often classified as clothes.
                        # Only include clothes ABOVE face center to avoid
                        # adding shirts/jackets.
                        if not clothes:
                            cloth_conf = segmented.confidence_masks[4].numpy_view()
                            if cloth_conf.ndim == 3:
                                cloth_conf = cloth_conf.squeeze(axis=2)
                            cloth_bin = (cloth_conf > confidence).astype(
                                np.uint8)

                            # Head zone: above face center only
                            face_center_y = int(
                                (fh_ys[0] + fh_ys[-1]) / 2)
                            head_zone = np.zeros_like(fh_dilated)
                            head_zone[:face_center_y, :] = fh_dilated[
                                :face_center_y, :]
                            cloth_head = cloth_bin & head_zone

                            if cloth_head.any():
                                ch_rgba = np.zeros_like(merged)
                                ch_rgba[cloth_head > 0] = (
                                    255, 255, 255, 255)
                                merged = np.maximum(merged, ch_rgba)
                                n_px = int(cloth_head.sum())
                                print(f"[HappyinPersonMask] Auto-added "
                                      f"{n_px} headwear px "
                                      f"(clothes in head zone, "
                                      f"dilate={dilate_px}px)")

                # ── Deferred accessories: restrict above shoulders ──
                # Purpose: exclude purses/shoes at feet while keeping
                # veils, hats, and other head accessories that extend
                # well beyond the tight hs_region bbox.
                # Zone: everything from y=0 to shoulder line (hs_region
                # bottom + margin), full image width.
                if _acc_deferred:
                    acc_conf = segmented.confidence_masks[5].numpy_view()
                    if acc_conf.ndim == 3:
                        acc_conf = acc_conf.squeeze(axis=2)
                    if _hs_region is not None:
                        hy1, hx1, hy2, hx2 = _hs_region
                        # Above shoulders: y=0 to hs bottom + 20% margin
                        margin_y = int((hy2 - hy1) * 0.2)
                        cutoff_y = min(h, hy2 + margin_y)
                        acc_mask = (acc_conf > confidence).astype(
                            np.uint8)
                        # Zero out everything below shoulder line
                        acc_mask[cutoff_y:, :] = 0
                    else:
                        acc_mask = (acc_conf > confidence).astype(
                            np.uint8)
                    if acc_mask.any():
                        acc_rgba = np.zeros_like(merged)
                        acc_rgba[acc_mask > 0] = (255, 255, 255, 255)
                        merged = np.maximum(merged, acc_rgba)
                        print(f"[HappyinPersonMask] Accessories "
                              f"above shoulders: "
                              f"{int(acc_mask.sum())}px "
                              f"(cutoff y={cutoff_y if _hs_region else 'none'})")

                # ── Fill junction gaps between categories ──────────
                # Hair/face/body masks may not overlap perfectly at
                # neck/forehead → small gaps.  Morphological close
                # fills gaps BEFORE matting so VITMatte/GuidedFilter
                # sees a continuous mask (no seam artifacts).
                merged_bin = (merged[:, :, 3] > 127).astype(np.uint8)
                close_k = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (7, 7))
                closed = cv2.morphologyEx(
                    merged_bin, cv2.MORPH_CLOSE, close_k)
                fill_zone = (closed > 0) & (merged_bin == 0)
                if fill_zone.any():
                    merged[fill_zone] = [255, 255, 255, 255]
                    n_filled = int(fill_zone.sum())
                    if n_filled > 50:
                        print(f"[HappyinPersonMask] Junction close: "
                              f"filled {n_filled}px gaps")

                # Fill interior holes: body/clothes boundary can leave
                # enclosed voids that close can't bridge.
                # RETR_EXTERNAL traces outer silhouette only → fill all
                # interior voids without affecting arm-torso gaps
                # (which connect to background).
                contours_ext, _ = cv2.findContours(
                    closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours_ext:
                    filled_mask = np.zeros_like(closed)
                    cv2.drawContours(
                        filled_mask, contours_ext, -1, 1, cv2.FILLED)
                    interior_holes = (
                        (filled_mask > 0) & (closed == 0))
                    if interior_holes.any():
                        merged[interior_holes] = [255, 255, 255, 255]
                        n_holes = int(interior_holes.sum())
                        if n_holes > 50:
                            print(f"[HappyinPersonMask] Hole fill: "
                                  f"filled {n_holes}px interior")

                # ── Remove disconnected islands ──────────────────
                # Keep the largest connected component + any fragment
                # that overlaps with hair (so disconnected hair strands
                # are preserved while stray body fragments are dropped).
                if remove_islands:
                    _isl = (merged[:, :, 3] > 127).astype(np.uint8)
                    n_lab, labels = cv2.connectedComponents(
                        _isl, connectivity=8)
                    if n_lab > 2:  # bg(0) + main + fragments
                        areas = np.bincount(labels.ravel())
                        areas[0] = 0  # ignore background
                        largest = int(np.argmax(areas))

                        # Hair overlap check: keep fragments that are
                        # predominantly hair (>30% overlap with hair mask)
                        _hair_conf = (
                            segmented.confidence_masks[1].numpy_view())
                        if _hair_conf.ndim == 3:
                            _hair_conf = _hair_conf.squeeze(axis=2)
                        _hair_bin = (_hair_conf > confidence)

                        kept_hair = 0
                        removed_px = 0
                        for lab_id in range(1, n_lab):
                            if lab_id == largest:
                                continue
                            frag = (labels == lab_id)
                            frag_px = int(frag.sum())
                            hair_overlap = int(
                                (frag & _hair_bin).sum())
                            if frag_px > 0 and (
                                    hair_overlap / frag_px > 0.3):
                                # Hair island — keep it
                                kept_hair += frag_px
                            else:
                                # Stray fragment — remove
                                merged[frag] = [0, 0, 0, 0]
                                removed_px += frag_px

                        if removed_px > 0 or kept_hair > 0:
                            print(
                                f"[HappyinPersonMask] Islands: "
                                f"removed {removed_px}px, "
                                f"kept {kept_hair}px (hair)")

                # Crop bbox from final mask (after auto-accessories,
                # junction close, hole fill, island removal).
                # Headwear/balaclavas added by auto_accessories
                # expand the crop correctly.
                _core_2d = (merged[:, :, 3] > 127).astype(np.uint8)
                _core_ys = np.where(_core_2d.any(axis=1))[0]
                _core_xs = np.where(_core_2d.any(axis=0))[0]
                if len(_core_ys) > 0 and len(_core_xs) > 0:
                    _core_bbox = (
                        int(_core_ys[0]), int(_core_xs[0]),
                        int(_core_ys[-1]), int(_core_xs[-1]))
                else:
                    _core_bbox = None

                mask_pil = Image.fromarray(merged)

                # Convert to tensor mask
                tensor_mask = (
                    np.array(mask_pil.convert("RGB")).astype(np.float32) / 255.0
                )
                tensor_mask = torch.from_numpy(tensor_mask)[None,]
                _mask = tensor_mask.squeeze(3)[..., 0]

                # mask_expand: dilate mask BEFORE matting so VITMatte
                # refines the new expanded edges (not raw dilated edge).
                if mask_expand > 0:
                    _me_np = (_mask.squeeze(0).numpy() * 255).astype(
                        np.uint8)
                    _me_kern = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (mask_expand * 2 + 1, mask_expand * 2 + 1))
                    _me_np = cv2.dilate(
                        _me_np, _me_kern, iterations=1)
                    _mask = torch.from_numpy(
                        _me_np.astype(np.float32) / 255.0
                    ).unsqueeze(0)
                    print(f"[HappyinPersonMask] mask_expand: "
                          f"+{mask_expand}px (before matting)")

                # Detail refinement
                # Scale matting zone to image resolution so VITMatte/GF
                # get a wide enough uncertain zone after internal resize.
                # At 1MP: use raw values.  At 16MP: 4× wider.
                _mp = (img_h * img_w) / 1_048_576
                _rs = max(1.0, math.sqrt(_mp))
                _s_erode = max(detail_erode, int(detail_erode * _rs))
                _s_edge = max(edge_width, int(edge_width * _rs))
                detail_range = _s_erode + _s_edge
                if matting:
                    if matting_method == 'GuidedFilter':
                        _mask = _guided_filter_alpha(
                            _pil2tensor(orig_image), _mask,
                            _s_edge)
                        _mask = _tensor2pil(
                            _histogram_remap(_mask, black_point, white_point))
                    elif matting_method == 'PyMatting':
                        _mask = _tensor2pil(
                            _mask_edge_detail(
                                _pil2tensor(orig_image), _mask,
                                detail_range // 8 + 1,
                                black_point, white_point))
                    else:
                        _trimap = _generate_trimap(
                            _mask, _s_erode, _s_edge)
                        _mask = _generate_vitmatte(
                            orig_image, _trimap,
                            device=device,
                            max_megapixels=max_megapixels,
                            method=matting_method)
                        _mask = _tensor2pil(
                            _histogram_remap(
                                _pil2tensor(_mask),
                                black_point, white_point))
                else:
                    _mask = _mask2image(_mask)

                ret_image = _rgb2rgba(orig_image, _mask)
                ret_images.append(_pil2tensor(ret_image))
                final_mask_t = _image2mask(_mask)
                ret_masks.append(final_mask_t)

                # Portrait was already handled by fast path (continue) above.
                # If we're here, it's NOT a portrait — do normal crop.
                mask_2d = final_mask_t.squeeze(0) if final_mask_t.dim() == 3 else final_mask_t

                # ── No-crop mode: full image + full mask ──
                if crop_mode == "disabled":
                    ret_crops.append(img_tensor.unsqueeze(0))
                    _fm = mask_2d if mask_2d.dim() == 2 else mask_2d.squeeze(0)
                    ret_crop_masks.append(_fm.unsqueeze(0))
                    ret_bboxes.append((0, 0, img_h, img_w))
                    ret_crop_sizes.append((img_h, img_w))
                    ret_pad_info.append(None)
                    ret_trim_info.append(None)
                    print(f"[HappyinPersonMask] crop_mode=disabled: "
                          f"full image {img_w}x{img_h}")
                    continue

                # ── Crop: bounding box ──
                # Use _core_bbox (final mask after auto-accessories,
                # junction close, hole fill — headwear included).
                # _MIN_MARGIN ensures the mask never touches crop edges
                # even when crop_padding=0 (matting, alignment artefacts).
                _MIN_MARGIN = 16
                nonzero = torch.nonzero(mask_2d > 0.01)
                if nonzero.numel() > 0:
                    # Base bbox (before padding)
                    if _core_bbox is not None:
                        cy1, cx1, cy2, cx2 = _core_bbox
                    else:
                        cy1 = int(nonzero[:, 0].min())
                        cy2 = int(nonzero[:, 0].max())
                        cx1 = int(nonzero[:, 1].min())
                        cx2 = int(nonzero[:, 1].max())

                    # Compute padding (px or % of bbox)
                    if crop_padding_mode == "%":
                        bbox_h = cy2 - cy1
                        bbox_w = cx2 - cx1
                        _pad = int(max(bbox_h, bbox_w)
                                   * crop_padding_pct / 100.0)
                    else:
                        _pad = crop_padding
                    _pad = max(_pad, _MIN_MARGIN + mask_expand)
                    if crop_padding_mode == "%":
                        print(f"[HappyinPersonMask] crop_pad: "
                              f"{crop_padding_pct}% of "
                              f"{max(cy2-cy1, cx2-cx1)}px "
                              f"= {_pad}px")

                    y1 = max(0, cy1 - _pad)
                    y2 = min(img_h, cy2 + 1 + _pad)
                    x1 = max(0, cx1 - _pad)
                    x2 = min(img_w, cx2 + 1 + _pad)

                    # Expand to include head+shoulders region
                    if _hs_region is not None:
                        hs_y1, hs_x1, hs_y2, hs_x2 = _hs_region
                        y1 = min(y1, max(0, hs_y1 - _pad))
                        y2 = max(y2, min(img_h, hs_y2 + _pad))
                        x1 = min(x1, max(0, hs_x1 - _pad))
                        x2 = max(x2, min(img_w, hs_x2 + _pad))

                    # body=upper: ensure crop bottom reaches the hip line.
                    # The mask may not extend to hips if person wears clothes
                    # (body confidence is 0 in clothed areas), so the crop
                    # would be a tight face/neck crop instead of upper-body.
                    if body == "upper" and _first_pass_hip_y is not None:
                        y2 = max(y2, min(img_h, _first_pass_hip_y + _pad))
                        print(f"[HappyinPersonMask] body=upper: "
                              f"crop extended to hip_y={_first_pass_hip_y}")

                    # ── Square crop: adjust bbox before alignment ──
                    if square_crop != "off":
                        sq_h = y2 - y1
                        sq_w = x2 - x1
                        if sq_h != sq_w:
                            # Both crop and pad: expand short side
                            # using original image pixels first.
                            # Never lose head/hair content.
                            # Remaining deficit → pad with black (post-crop).
                            side = max(sq_h, sq_w)
                            if sq_h > sq_w:
                                need = side - sq_w
                                x1 = max(0, x1 - need // 2)
                                x2 = min(img_w, x1 + side)
                                if x2 - x1 < side:
                                    x1 = max(0, x2 - side)
                            else:
                                need = side - sq_h
                                y1 = max(0, y1 - need // 2)
                                y2 = min(img_h, y1 + side)
                                if y2 - y1 < side:
                                    y1 = max(0, y2 - side)

                    # ── SAFETY: ensure mask is fully inside crop ──
                    # Runs FIRST so ×16 alignment has guaranteed
                    # margins to work with.
                    mask_top = int(nonzero[:, 0].min())
                    mask_left = int(nonzero[:, 1].min())
                    mask_bottom = int(nonzero[:, 0].max())
                    mask_right = int(nonzero[:, 1].max())
                    safety_margin = _MIN_MARGIN

                    if mask_top < y1 + safety_margin:
                        shift = (y1 + safety_margin) - mask_top
                        old_y1 = y1
                        y1 = max(0, y1 - shift)
                        y2 = y2 - (old_y1 - y1)
                        if y2 > img_h:
                            y2 = img_h
                        print(f"[HappyinPersonMask] SAFETY: mask top "
                              f"clipped, shifted crop up by {shift}px")

                    if mask_bottom > y2 - safety_margin:
                        shift = mask_bottom - (y2 - safety_margin)
                        y2 = min(img_h, y2 + shift)
                        print(f"[HappyinPersonMask] SAFETY: mask bottom "
                              f"clipped, expanded crop down by {shift}px")

                    if mask_left < x1 + safety_margin:
                        shift = (x1 + safety_margin) - mask_left
                        x1 = max(0, x1 - shift)
                        print(f"[HappyinPersonMask] SAFETY: mask left "
                              f"clipped, shifted crop left by {shift}px")

                    if mask_right > x2 - safety_margin:
                        shift = mask_right - (x2 - safety_margin)
                        x2 = min(img_w, x2 + shift)
                        print(f"[HappyinPersonMask] SAFETY: mask right "
                              f"clipped, expanded crop right by {shift}px")

                    # ── Smart latent alignment ────────────────────────
                    # After safety guarantees mask margin, nudge
                    # crop toward ×N (N = latent factor of model).
                    # Compare shrink-inward vs expand-outward —
                    # pick the smaller change.  Never crop into
                    # mask.  Partial expand OK (more real content
                    # = less VAE padding).
                    _ALIGN_GAP = 2   # min px mask↔crop after trim
                    crop_h = y2 - y1
                    crop_w = x2 - x1

                    def _snap_latent(lo, hi, dim_max, m_lo, m_hi):
                        sz = hi - lo
                        rem = sz % _LATENT_F
                        if rem == 0:
                            return lo, hi
                        trim_n = rem            # shrink → prev ×N
                        grow_n = _LATENT_F - rem  # expand → next ×N
                        # Inward clearance (crop edge → mask)
                        mg_lo = max(0, m_lo - lo - _ALIGN_GAP)
                        mg_hi = max(0, hi - m_hi - 1 - _ALIGN_GAP)
                        can_trim = mg_lo + mg_hi >= trim_n
                        # Outward room (crop edge → image edge)
                        sp_lo, sp_hi = lo, dim_max - hi
                        can_grow = sp_lo + sp_hi >= grow_n

                        if can_trim and (trim_n < grow_n
                                         or not can_grow):
                            # Trim is cheaper (or expand impossible)
                            t_lo = min(trim_n // 2, mg_lo)
                            t_hi = trim_n - t_lo
                            if t_hi > mg_hi:
                                t_hi = mg_hi
                                t_lo = trim_n - t_hi
                            return lo + t_lo, hi - t_hi
                        if can_grow:
                            a_lo = min(grow_n // 2, sp_lo)
                            a_hi = grow_n - a_lo
                            if a_hi > sp_hi:
                                a_hi = sp_hi
                                a_lo = grow_n - a_hi
                            return lo - a_lo, hi + a_hi
                        # Can't reach exact ×N — partial expand
                        avail = sp_lo + sp_hi
                        if avail > 0:
                            a_lo = min(avail // 2, sp_lo)
                            a_hi = min(avail - a_lo, sp_hi)
                            return lo - a_lo, hi + a_hi
                        return lo, hi

                    if latent_snap != "off":
                        x1, x2 = _snap_latent(
                            x1, x2, img_w, mask_left, mask_right)
                        y1, y2 = _snap_latent(
                            y1, y2, img_h, mask_top, mask_bottom)

                    new_w = x2 - x1
                    new_h = y2 - y1
                    if latent_snap != "off" and (crop_w != new_w
                                                  or crop_h != new_h):
                        print(f"[HappyinPersonMask] ×{_LATENT_F} align: "
                              f"{crop_w}x{crop_h} → "
                              f"{new_w}x{new_h}")

                    crop = img_tensor[y1:y2, x1:x2, :]
                    crop_mask = mask_2d[y1:y2, x1:x2]
                    _crop_pad_info = None   # track padding for PasteBack
                    _crop_trim_info = None  # track trim for PasteBack

                    # ── Square pad fallback: bbox hit image edge ──
                    if square_crop in ("crop", "pad", "smart"):
                        ch, cw = crop.shape[0], crop.shape[1]
                        if ch != cw:
                            if square_crop == "smart":
                                # Try trim first (no padding).
                                # Fall back to bg-color pad if face mask
                                # would be cut.
                                side = min(ch, cw)
                                y_off = (ch - side) // 2
                                x_off = (cw - side) // 2
                                trim_top  = crop_mask[:y_off, :].any()        if y_off > 0          else False
                                trim_bot  = crop_mask[y_off+side:, :].any()   if y_off+side < ch    else False
                                trim_left = crop_mask[:, :x_off].any()        if x_off > 0          else False
                                trim_right= crop_mask[:, x_off+side:].any()   if x_off+side < cw    else False
                                can_trim  = not (trim_top or trim_bot or
                                                 trim_left or trim_right)
                                if can_trim:
                                    crop = crop[y_off:y_off+side,
                                                x_off:x_off+side, :]
                                    crop_mask = crop_mask[y_off:y_off+side,
                                                          x_off:x_off+side]
                                    _crop_trim_info = (y_off, x_off, side)
                                    print(f"[HappyinPersonMask] square_crop=smart: "
                                          f"{cw}x{ch} → {side}x{side} (trim)")
                                else:
                                    # Face in trim zone → pad with bg color
                                    side = max(ch, cw)
                                    pad_y = (side - ch) // 2
                                    pad_x = (side - cw) // 2
                                    ps = max(1, min(8, ch // 8, cw // 8))
                                    tl = crop[:ps, :ps, :].reshape(-1, crop.shape[2])
                                    tr = crop[:ps, -ps:, :].reshape(-1, crop.shape[2])
                                    bl = crop[-ps:, :ps, :].reshape(-1, crop.shape[2])
                                    br = crop[-ps:, -ps:, :].reshape(-1, crop.shape[2])
                                    bg = torch.cat([tl, tr, bl, br], dim=0).mean(dim=0)
                                    padded_img = bg.view(1, 1, -1).expand(
                                        side, side, -1).clone()
                                    padded_img[pad_y:pad_y+ch,
                                               pad_x:pad_x+cw, :] = crop
                                    crop = padded_img
                                    padded_m = torch.zeros(side, side)
                                    padded_m[pad_y:pad_y+ch,
                                             pad_x:pad_x+cw] = crop_mask
                                    crop_mask = padded_m
                                    _crop_pad_info = (pad_y, pad_x, ch, cw)
                                    print(f"[HappyinPersonMask] square_crop=smart: "
                                          f"{cw}x{ch} → {side}x{side} "
                                          f"(bg-pad, face in trim zone)")
                            else:
                                side = max(ch, cw)
                                pad_y = (side - ch) // 2
                                pad_x = (side - cw) // 2
                                # Background-color fill: same logic as smart mode.
                                # Black zeros → diffusion model bleeds dark artifacts
                                # into content near padding boundary = black squares.
                                _ps = max(1, min(8, ch // 8, cw // 8))
                                _tl = crop[:_ps, :_ps, :].reshape(-1, crop.shape[2])
                                _tr = crop[:_ps, -_ps:, :].reshape(-1, crop.shape[2])
                                _bl = crop[-_ps:, :_ps, :].reshape(-1, crop.shape[2])
                                _br = crop[-_ps:, -_ps:, :].reshape(-1, crop.shape[2])
                                _bg = torch.cat([_tl, _tr, _bl, _br], dim=0).mean(dim=0)
                                padded_img = _bg.view(1, 1, -1).expand(
                                    side, side, -1).clone()
                                padded_img[pad_y:pad_y+ch,
                                           pad_x:pad_x+cw, :] = crop
                                crop = padded_img
                                padded_m = torch.zeros(side, side)
                                padded_m[pad_y:pad_y+ch,
                                         pad_x:pad_x+cw] = crop_mask
                                crop_mask = padded_m
                                _crop_pad_info = (pad_y, pad_x, ch, cw)
                                print(f"[HappyinPersonMask] square_crop={square_crop}: "
                                      f"{cw}x{ch} → {side}x{side} "
                                      f"(+{pad_x}px x, +{pad_y}px y)")

                    # ── "crop" mode: add body in head+shoulders zone ─
                    # Body restricted to head_shoulders region (not full
                    # bbox which can be huge due to long hair + padding).
                    # Only adds skin within the head+shoulders area.
                    if (body == "crop"
                            and _body_conf_for_crop is not None):
                        # Determine body inclusion zone
                        if _hs_region is not None:
                            by1 = max(y1, _hs_region[0])
                            bx1 = max(x1, _hs_region[1])
                            by2 = min(y2, _hs_region[2])
                            bx2 = min(x2, _hs_region[3])
                        else:
                            # No head_shoulders → use upper 60% of bbox
                            bbox_h = y2 - y1
                            by1, bx1 = y1, x1
                            by2 = min(y2, y1 + int(bbox_h * 0.6))
                            bx2 = x2
                        if by2 > by1 and bx2 > bx1:
                            body_zone = _body_conf_for_crop[
                                by1:by2, bx1:bx2]
                            body_zone_bin = (
                                body_zone > confidence
                            ).astype(np.float32)
                            # Map to crop coordinates
                            cy1 = by1 - y1
                            cx1 = bx1 - x1
                            cy2 = by2 - y1
                            cx2 = bx2 - x1
                            body_in_crop = torch.from_numpy(
                                body_zone_bin)
                            crop_mask[cy1:cy2, cx1:cx2] = (
                                torch.maximum(
                                    crop_mask[cy1:cy2, cx1:cx2],
                                    body_in_crop))
                            # Update full-size mask + RGBA
                            full_body = torch.zeros(img_h, img_w)
                            full_body[by1:by2, bx1:bx2] = (
                                body_in_crop)
                            ret_masks[-1] = torch.maximum(
                                ret_masks[-1],
                                full_body.unsqueeze(0))
                            upd_mask_pil = Image.fromarray(
                                (ret_masks[-1].squeeze(0).numpy()
                                 * 255).astype(np.uint8), 'L')
                            ret_images[-1] = _pil2tensor(
                                _rgb2rgba(orig_image, upd_mask_pil))
                            n_body = int(body_zone_bin.sum())
                            print(f"[HappyinPersonMask] body=crop: "
                                  f"{n_body}px in hs_region "
                                  f"({bx1},{by1})-({bx2},{by2})")

                    # ── Second-pass segmentation (selfie mode) ──────
                    # When selfie is connected, first-pass may have
                    # artifacts from person isolation.  Re-segment the
                    # CROP region only — it contains one person, so no
                    # selfie filtering is needed → clean mask.
                    # Only runs when portrait is enabled (auto/force)
                    # to save time; portrait="off" skips this.
                    if (ref_embedding is not None
                            and portrait != "off"):
                        try:
                            crop_h2 = y2 - y1
                            crop_w2 = x2 - x1
                            crop_np2 = np.clip(
                                255.0 * img_tensor[
                                    y1:y2, x1:x2, :].cpu().numpy(),
                                0, 255).astype(np.uint8)
                            crop_pil2 = Image.fromarray(
                                crop_np2).convert('RGB')
                            mp_crop = self._get_mediapipe_image(crop_pil2)
                            seg2 = segmenter.segment(mp_crop)

                            # Merge same categories as first pass
                            mask2_list = []
                            for idx, enabled in category_flags.items():
                                if enabled:
                                    m2d = seg2.confidence_masks[
                                        idx].numpy_view()
                                    if (m2d.ndim == 3
                                            and m2d.shape[2] == 1):
                                        m2d = m2d.squeeze(axis=2)
                                    mask2_list.append(
                                        (m2d > confidence).astype(
                                            np.float32))

                            # Deferred accessories in crop space
                            # Same logic: above shoulders, not tight bbox
                            if _acc_deferred:
                                a2c = seg2.confidence_masks[
                                    5].numpy_view()
                                if (a2c.ndim == 3
                                        and a2c.shape[2] == 1):
                                    a2c = a2c.squeeze(axis=2)
                                a2_bin = (
                                    a2c > confidence
                                ).astype(np.float32)
                                if _hs_region is not None:
                                    # Cutoff: below shoulders in crop
                                    margin_y2 = int(
                                        (_hs_region[2] - _hs_region[0])
                                        * 0.2)
                                    cutoff_crop = min(
                                        crop_h2,
                                        _hs_region[2] - y1 + margin_y2)
                                    if 0 < cutoff_crop < crop_h2:
                                        a2_bin[cutoff_crop:, :] = 0.0
                                mask2_list.append(a2_bin)

                            # Body in second pass (crop-space)
                            if body_enabled:
                                body2_m = seg2.confidence_masks[
                                    2].numpy_view()
                                if (body2_m.ndim == 3
                                        and body2_m.shape[2] == 1):
                                    body2_m = body2_m.squeeze(axis=2)
                                body2_bin = (
                                    body2_m > confidence
                                ).astype(np.float32)

                                if (body == "upper"
                                        and _first_pass_hip_y
                                        is not None):
                                    hip_y_crop = (
                                        _first_pass_hip_y - y1)
                                    if 0 < hip_y_crop < crop_h2:
                                        body2_bin[
                                            hip_y_crop:, :] = 0.0
                                        # Restore arm pixels
                                        if _arm_keep is not None:
                                            ak_crop = _arm_keep[
                                                y1:y2, x1:x2]
                                            arm_zone = (
                                                ak_crop[hip_y_crop:, :]
                                                > 0.5)
                                            body2_raw = (
                                                body2_m[hip_y_crop:, :]
                                                > confidence
                                            ).astype(np.float32)
                                            body2_bin[
                                                hip_y_crop:, :
                                            ] = np.where(
                                                arm_zone, body2_raw,
                                                0.0)
                                    elif hip_y_crop <= 0:
                                        body2_bin[:] = 0.0
                                elif body == "crop":
                                    # Restrict to hs_region in crop space
                                    if _hs_region is not None:
                                        hs_top = max(
                                            0, _hs_region[0] - y1)
                                        hs_bot = min(
                                            crop_h2, _hs_region[2] - y1)
                                        hs_left = max(
                                            0, _hs_region[1] - x1)
                                        hs_right = min(
                                            crop_w2, _hs_region[3] - x1)
                                        keep = np.zeros_like(body2_bin)
                                        keep[hs_top:hs_bot,
                                             hs_left:hs_right] = 1.0
                                        body2_bin = body2_bin * keep
                                # "full": include all body in crop

                                mask2_list.append(body2_bin)

                            if mask2_list:
                                mask2_np = reduce(
                                    np.maximum, mask2_list)

                                # Auto-accessories on crop
                                if ((face or hair)
                                        and auto_accessories):
                                    fh2 = (mask2_np > 0.5).astype(
                                        np.uint8)
                                    fh2_ys = np.where(
                                        fh2.any(axis=1))[0]
                                    if len(fh2_ys) > 0:
                                        fh2_h = int(
                                            fh2_ys[-1] - fh2_ys[0])
                                        dil2 = max(
                                            31, int(fh2_h * 0.5))
                                        dist2 = cv2.distanceTransform(
                                            1 - fh2, cv2.DIST_L2, 5)
                                        fh2_dil = (
                                            dist2 <= dil2
                                        ).astype(np.uint8)

                                        if not accessories:
                                            a2 = seg2.confidence_masks[
                                                5].numpy_view()
                                            if a2.ndim == 3:
                                                a2 = a2.squeeze(
                                                    axis=2)
                                            a2_hit = (
                                                (a2 > confidence
                                                 ).astype(np.uint8)
                                                & fh2_dil)
                                            if a2_hit.any():
                                                mask2_np = np.maximum(
                                                    mask2_np,
                                                    a2_hit.astype(
                                                        np.float32))

                                        if not clothes:
                                            c2 = seg2.confidence_masks[
                                                4].numpy_view()
                                            if c2.ndim == 3:
                                                c2 = c2.squeeze(
                                                    axis=2)
                                            c2_bin = (
                                                c2 > confidence
                                            ).astype(np.uint8)
                                            fcy2 = int(
                                                (fh2_ys[0]
                                                 + fh2_ys[-1]) / 2)
                                            hz2 = np.zeros_like(
                                                fh2_dil)
                                            hz2[:fcy2, :] = (
                                                fh2_dil[:fcy2, :])
                                            c2_hit = c2_bin & hz2
                                            if c2_hit.any():
                                                mask2_np = np.maximum(
                                                    mask2_np,
                                                    c2_hit.astype(
                                                        np.float32))

                                # Junction close on crop mask
                                m2_bin = (mask2_np > 0.5).astype(
                                    np.uint8)
                                m2_closed = cv2.morphologyEx(
                                    m2_bin, cv2.MORPH_CLOSE,
                                    close_k)
                                m2_fill = ((m2_closed > 0)
                                           & (m2_bin == 0))
                                if m2_fill.any():
                                    mask2_np = np.maximum(
                                        mask2_np,
                                        m2_fill.astype(np.float32))

                                # Fill interior holes in crop
                                cnt2, _ = cv2.findContours(
                                    m2_closed, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
                                if cnt2:
                                    hf2 = np.zeros_like(m2_closed)
                                    cv2.drawContours(
                                        hf2, cnt2, -1, 1,
                                        cv2.FILLED)
                                    ih2 = ((hf2 > 0)
                                           & (m2_closed == 0))
                                    if ih2.any():
                                        mask2_np = np.maximum(
                                            mask2_np,
                                            ih2.astype(np.float32))

                                # Remove islands in crop (hair-aware)
                                if remove_islands:
                                    m2_isl = (mask2_np > 0.5).astype(
                                        np.uint8)
                                    n2l, lab2 = cv2.connectedComponents(
                                        m2_isl, connectivity=8)
                                    if n2l > 2:
                                        a2 = np.bincount(
                                            lab2.ravel())
                                        a2[0] = 0
                                        lg2 = int(np.argmax(a2))
                                        # Hair mask from crop segmentation
                                        _hc2 = seg2.confidence_masks[
                                            1].numpy_view()
                                        if _hc2.ndim == 3:
                                            _hc2 = _hc2.squeeze(2)
                                        _hb2 = (_hc2 > confidence)
                                        for _l2 in range(1, n2l):
                                            if _l2 == lg2:
                                                continue
                                            _fr2 = (lab2 == _l2)
                                            _fp2 = int(_fr2.sum())
                                            _ho2 = int(
                                                (_fr2 & _hb2).sum())
                                            if _fp2 > 0 and (
                                                    _ho2 / _fp2 > 0.3):
                                                continue  # hair — keep
                                            mask2_np[_fr2] = 0.0

                                # mask_expand before matting (crop)
                                if mask_expand > 0:
                                    _me2 = (mask2_np > 0.5).astype(
                                        np.uint8)
                                    _me2k = cv2.getStructuringElement(
                                        cv2.MORPH_ELLIPSE,
                                        (mask_expand * 2 + 1,
                                         mask_expand * 2 + 1))
                                    _me2 = cv2.dilate(
                                        _me2, _me2k, iterations=1)
                                    mask2_np = np.maximum(
                                        mask2_np,
                                        _me2.astype(np.float32))

                                # Detail refinement on crop
                                _m2 = torch.from_numpy(
                                    mask2_np).unsqueeze(0)
                                # Scale matting zone for crop res
                                _mp2 = (crop_h2 * crop_w2) / 1_048_576
                                _rs2 = max(1.0, math.sqrt(_mp2))
                                _se2 = max(detail_erode,
                                           int(detail_erode * _rs2))
                                _sw2 = max(edge_width,
                                           int(edge_width * _rs2))
                                _dr2 = _se2 + _sw2
                                if matting:
                                    if matting_method == 'GuidedFilter':
                                        _m2 = _guided_filter_alpha(
                                            _pil2tensor(crop_pil2),
                                            _m2,
                                            _sw2)
                                        _m2 = _tensor2pil(
                                            _histogram_remap(
                                                _m2, black_point,
                                                white_point))
                                    elif matting_method == 'PyMatting':
                                        _m2 = _tensor2pil(
                                            _mask_edge_detail(
                                                _pil2tensor(crop_pil2),
                                                _m2,
                                                _dr2 // 8 + 1,
                                                black_point,
                                                white_point))
                                    else:
                                        _tri2 = _generate_trimap(
                                            _m2, _se2,
                                            _sw2)
                                        _m2 = _generate_vitmatte(
                                            crop_pil2, _tri2,
                                            device=device,
                                            max_megapixels=(
                                                max_megapixels),
                                            method=matting_method)
                                        _m2 = _tensor2pil(
                                            _histogram_remap(
                                                _pil2tensor(_m2),
                                                black_point,
                                                white_point))
                                else:
                                    _m2 = _mask2image(_m2)

                                cm2 = _image2mask(_m2).squeeze(0)
                                crop_mask = cm2

                                # Update full-size mask + RGBA
                                full2 = torch.zeros(img_h, img_w)
                                full2[y1:y2, x1:x2] = cm2
                                ret_masks[-1] = full2.unsqueeze(0)

                                fm2_pil = Image.fromarray(
                                    (full2.numpy() * 255).astype(
                                        np.uint8), 'L')
                                ret_images[-1] = _pil2tensor(
                                    _rgb2rgba(orig_image, fm2_pil))

                                n2 = int((cm2 > 0.01).sum())
                                print(
                                    f"[HappyinPersonMask] "
                                    f"Second-pass: "
                                    f"crop {crop_w2}x{crop_h2}, "
                                    f"clean mask {n2}px")

                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            print(
                                f"[HappyinPersonMask] "
                                f"Second-pass failed: {e}, "
                                f"using first-pass mask")

                    # ── Sync crop_mask size with crop (after pad) ──
                    if (crop.shape[0] != crop_mask.shape[0]
                            or crop.shape[1] != crop_mask.shape[1]):
                        synced = torch.zeros(
                            crop.shape[0], crop.shape[1])
                        mh = min(crop_mask.shape[0], crop.shape[0])
                        mw = min(crop_mask.shape[1], crop.shape[1])
                        off_y = (crop.shape[0] - mh) // 2
                        off_x = (crop.shape[1] - mw) // 2
                        synced[off_y:off_y+mh, off_x:off_x+mw] = (
                            crop_mask[:mh, :mw])
                        crop_mask = synced

                    ret_crops.append(crop.unsqueeze(0))
                    ret_crop_masks.append(crop_mask.unsqueeze(0))
                    ret_bboxes.append((y1, x1, y2, x2))
                    ret_crop_sizes.append(
                        (crop.shape[0], crop.shape[1]))
                    ret_pad_info.append(_crop_pad_info)
                    ret_trim_info.append(_crop_trim_info)
                else:
                    ret_crops.append(img_tensor.unsqueeze(0))
                    ret_crop_masks.append(torch.zeros(1, img_h, img_w))
                    ret_bboxes.append((0, 0, img_h, img_w))
                    ret_crop_sizes.append((img_h, img_w))
                    ret_pad_info.append(None)
                    ret_trim_info.append(None)

        # Handle different crop/crop_mask sizes across batch (pad to max)
        if ret_crops:
            max_ch = max(c.shape[1] for c in ret_crops)
            max_cw = max(c.shape[2] for c in ret_crops)
            padded_crops = []
            padded_cmasks = []
            for c, cm in zip(ret_crops, ret_crop_masks):
                if c.shape[1] != max_ch or c.shape[2] != max_cw:
                    p = torch.zeros(1, max_ch, max_cw, c.shape[3])
                    p[:, :c.shape[1], :c.shape[2], :] = c
                    padded_crops.append(p)
                    pm = torch.zeros(1, max_ch, max_cw)
                    pm[:, :cm.shape[1], :cm.shape[2]] = cm
                    padded_cmasks.append(pm)
                else:
                    padded_crops.append(c)
                    padded_cmasks.append(cm)
            all_crops = torch.cat(padded_crops, dim=0)
            all_crop_masks = torch.cat(padded_cmasks, dim=0)
        else:
            all_crops = images
            all_crop_masks = torch.zeros(images.shape[0],
                                         images.shape[1], images.shape[2])

        # CROP_DATA: all-in-one package for PasteBack
        # Contains everything PasteBack needs — one wire instead of four
        crop_data = {
            'bboxes': ret_bboxes,                    # (y1, x1, y2, x2) per image
            'originals': [images[i] for i in range(images.shape[0])],  # original tensors
            'masks': [ret_masks[i] for i in range(len(ret_masks))],    # full-size masks
            'crop_masks': [ret_crop_masks[i] for i in range(len(ret_crop_masks))],  # bbox masks
            'crop_sizes': ret_crop_sizes,            # (h, w) actual crop dims (after padding)
            'pad_info': ret_pad_info,                # (pad_y, pad_x, orig_h, orig_w) or None
            'trim_info': ret_trim_info,              # (y_off, x_off, side) or None
        }

        bbox_str = f"{ret_bboxes[0]}" if ret_bboxes else "none"
        print(f"[HappyinPersonMask] Processed {len(ret_images)} image(s), "
              f"bbox={bbox_str}")

        # image = crop (RGB), mask = crop_mask
        # crop_padding controls how much context around the mask.
        # At padding=0: tight around mask. With padding: more spread.
        # PasteBack uses crop_data (has full-size masks internally).
        return (
            all_crops,
            all_crop_masks,
            all_crops,
            all_crop_masks,
            crop_data,
        )
