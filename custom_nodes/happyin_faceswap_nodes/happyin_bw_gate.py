"""
Happyin BW Gate — conditional desaturation.

Checks if the source image is black & white.
If yes — desaturates the passthrough image.
If no  — passes the image unchanged.

Detection: 16×16 palette (INTER_AREA averaging cancels JPEG chroma noise),
then two independent checks — either is enough to call it COLOR:
  1. mean_s > bw_threshold  → COLOR  (overall saturation present)
  2. skin_pixels >= 5/256   → COLOR  (skin-tone hue detected: H=0–25, S>25)
If neither triggers → B&W → desaturate output to gray.
"""

import cv2
import torch
import numpy as np


class HappyinBWGate:
    """Conditional desaturation based on source image color.

    Detection uses a 16×16 downsampled "palette" (256 pixels) in HSV space.
    INTER_AREA averaging naturally cancels JPEG chroma subsampling noise.

    Two independent COLOR signals (OR logic):
      A. mean_s > bw_threshold  — overall saturation (catches vivid colors,
         colored backgrounds, etc.)
      B. skin_pixels >= 5       — warm-hue pixels with moderate saturation,
         H∈[0,25] S∈[25,180] V>40. Catches portraits in white/light outfits
         where overall mean_s stays low but skin tone is clearly present.
         JPEG B&W noise never produces S>25 in this exact hue band → safe.

    Result: COLOR if A or B; B&W otherwise (desaturate + is_bw=True).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE", {
                    "tooltip": "Исходное фото — по нему определяем ЧБ или цвет",
                }),
                "image": ("IMAGE", {
                    "tooltip": "Рабочее изображение — десатурируем если source ЧБ",
                }),
                "bw_threshold": ("FLOAT", {
                    "default": 25.0, "min": 1.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Порог средней насыщенности 16×16 палитры.\n"
                               "Цветное фото: mean_s ~40-80+\n"
                               "ЧБ (любое): mean_s ~0-20\n"
                               "Default 25 = надёжное разделение.\n"
                               "Дополнительно: skin-tone детектор (H=0-25, S>25)\n"
                               "автоматически защищает портреты от десатурации.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("image", "is_bw")
    FUNCTION = "run"
    CATEGORY = "Happyin/Utils"

    DESCRIPTION = (
        "BW Gate — определение ЧБ по средней насыщенности + skin-tone guard.\n\n"
        "COLOR если: mean_s > порога  ИЛИ  ≥5 пикселей телесного цвета.\n"
        "B&W   если: оба условия не выполнены → IMAGE десатурируется.\n\n"
        "Skin-tone guard защищает портреты в белой/светлой одежде:\n"
        "JPEG-шум ч/б фото никогда не даёт H=0-25 + S>25 одновременно."
    )

    def run(self, source, image, bw_threshold):
        src_np = (source[0].cpu().numpy() * 255.0).astype(np.uint8)

        # ── 16×16 palette: INTER_AREA averages out JPEG chroma noise ──
        small = cv2.resize(src_np, (16, 16), interpolation=cv2.INTER_AREA)
        small_hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)

        h_vals = small_hsv[:, :, 0].flatten().astype(np.float32)  # 0-179
        s_vals = small_hsv[:, :, 1].flatten().astype(np.float32)  # 0-255
        v_vals = small_hsv[:, :, 2].flatten().astype(np.float32)  # 0-255

        mean_s = float(np.mean(s_vals))
        max_s  = float(np.max(s_vals))

        # ── Skin-tone guard: warm hue + moderate saturation ───────────
        # H=0-25 in OpenCV (0-179) = 0-50° real = orange/yellow/red = skin
        # S=25-180 = not pure noise, not oversaturated
        # V>40 = not in shadow
        skin_mask  = (h_vals <= 25) & (s_vals >= 25) & (s_vals <= 180) & (v_vals >= 40)
        skin_count = int(np.sum(skin_mask))
        has_skin   = skin_count >= 5  # ≥5/256 pixels (~2%) with skin hue

        # ── Decision ──────────────────────────────────────────────────
        if mean_s > bw_threshold:
            mode   = "COLOR"
            reason = f"mean_s={mean_s:.1f}>{bw_threshold}"
            is_bw  = False
        elif has_skin:
            mode   = "COLOR"
            reason = f"skin={skin_count}px (mean_s={mean_s:.1f} low but skin detected)"
            is_bw  = False
        else:
            mode   = "B&W"
            reason = f"mean_s={mean_s:.1f} skin={skin_count}px"
            is_bw  = True

        print(f"[HappyinBWGate] 16×16 palette: {reason} max={max_s:.0f} → {mode}")

        if not is_bw:
            print(f"[HappyinBWGate] → Passed unchanged")
            return (image, False)

        # ── Desaturate the passthrough image (luminance-weighted) ─────
        img  = image.clone()
        gray = (img[:, :, :, 0:1] * 0.2126
                + img[:, :, :, 1:2] * 0.7152
                + img[:, :, :, 2:3] * 0.0722)
        img  = gray.repeat(1, 1, 1, 3)
        print(f"[HappyinBWGate] → Desaturated to B&W")
        return (img, True)
