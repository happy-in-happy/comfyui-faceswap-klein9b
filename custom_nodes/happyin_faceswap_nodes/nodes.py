"""
ComfyUI Pixel Alignment Nodes — LatentSizeSnap
"""

import torch
import torch.nn.functional as F
import numpy as np


# =============================================================================
# UTILS
# =============================================================================

def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# LATENT SIZE TABLES
# =============================================================================

LATENT_SIZES = {
    "FLUX Klein": [
        # 1.0 MP — основной тир FLUX Klein 9B
        (1024, 1024),  # 1:1
        (1216, 832),   # 3:2
        (832, 1216),   # 2:3
        (1152, 896),   # 4:3
        (896, 1152),   # 3:4
        (1344, 768),   # 16:9
        (768, 1344),   # 9:16
        (1408, 704),   # 2:1
        (704, 1408),   # 1:2
    ],
    "FLUX Dev / Schnell": [
        # FLUX.1 Dev/Schnell — 1.0 MP
        (1024, 1024),  # 1:1
        (1216, 832),   # 3:2
        (832, 1216),   # 2:3
        (1152, 896),   # 4:3
        (896, 1152),   # 3:4
        (1344, 768),   # 16:9
        (768, 1344),   # 9:16
        (1536, 640),   # 2.4:1
        (640, 1536),   # 1:2.4
    ],
    "FLUX 2MP": [
        # FLUX at 2 megapixels (high detail)
        (1408, 1408),  # 1:1
        (1728, 1152),  # 3:2
        (1152, 1728),  # 2:3
        (1664, 1216),  # 4:3
        (1216, 1664),  # 3:4
        (1920, 1088),  # 16:9
        (1088, 1920),  # 9:16
        (2048, 1024),  # 2:1
        (1024, 2048),  # 1:2
    ],
    "SDXL": [
        # SDXL trained resolutions — 1.0 MP (1024x1024 base)
        (1024, 1024),  # 1:1
        (1152, 896),   # 4:3
        (896, 1152),   # 3:4
        (1216, 832),   # 3:2
        (832, 1216),   # 2:3
        (1344, 768),   # 16:9
        (768, 1344),   # 9:16
        (1536, 640),   # 2.4:1
        (640, 1536),   # 1:2.4
    ],
    "SD 1.5": [
        # Stable Diffusion 1.5 — 0.25 MP (512x512 base)
        (512, 512),    # 1:1
        (640, 384),    # 5:3
        (384, 640),    # 3:5
        (576, 448),    # 4:3
        (448, 576),    # 3:4
        (768, 320),    # 2.4:1
        (320, 768),    # 1:2.4
    ],
    "Qwen Edit": [
        # Qwen-Image-Edit-2511 — official trained resolutions ~1.6 MP
        # https://github.com/QwenLM/Qwen-Image/issues/7
        # No square — square causes head elongation and quality degradation
        (1584, 1056),  # 3:2   (1.67 MP)
        (1056, 1584),  # 2:3   (1.67 MP)
        (1472, 1104),  # 4:3   (1.62 MP)
        (1104, 1472),  # 3:4   (1.62 MP)
        (1664, 928),   # 16:9  (1.54 MP)
        (928, 1664),   # 9:16  (1.54 MP)
    ],
    "Qwen Edit 1MP": [
        # Qwen-Image-Edit — community-tested ~1.0 MP sizes
        # Use when TextEncodeQwenImageEdit forces 1MP clip encoding
        # No square — square causes quality degradation
        (1184, 880),   # 4:3
        (880, 1184),   # 3:4
        (1248, 832),   # 3:2
        (832, 1248),   # 2:3
        (1392, 752),   # ~16:9
        (752, 1392),   # ~9:16
        (1568, 672),   # 21:9
        (672, 1568),   # 9:21
    ],
}


# =============================================================================
# NODE: Latent Size Snap
# =============================================================================

class LatentSizeSnap:
    """
    Подгоняет картинку под ближайший размер латента для выбранной модели.

    Выбираешь модель (FLUX Klein, FLUX Dev, SDXL, SD 1.5 или Custom).
    Нода находит ближайший размер по аспекту картинки из таблицы,
    и ТЯНЕТ непропорционально (stretch, не crop) чтобы попасть точно в латент.

    Все размеры кратны 16 — готовы для VAE.
    """

    @classmethod
    def INPUT_TYPES(cls):
        models = list(LATENT_SIZES.keys()) + ["custom"]
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (models, {
                    "default": "FLUX Klein",
                    "tooltip": "Модель для которой подгоняем размер. custom = задать max_side вручную."}),
            },
            "optional": {
                "max_side": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16,
                                     "tooltip": "Макс. сторона (пикселей). 0 = размер из таблицы (для именных моделей) или ×16 align (для custom). >0 = масштабировать до этого размера."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "orig_width", "orig_height", "info")
    FUNCTION = "snap"
    CATEGORY = "Happyin/PixelAlign"

    def _find_best_size(self, w, h, sizes):
        """
        Находит ближайший размер из таблицы по аспекту картинки.
        Минимизирует разницу аспектов в лог-пространстве (симметричная метрика).
        """
        aspect = w / h
        best = None
        best_dist = float('inf')
        for sw, sh in sizes:
            s_aspect = sw / sh
            dist = abs(np.log(aspect) - np.log(s_aspect))
            if dist < best_dist:
                best_dist = dist
                best = (sw, sh)
        return best

    def snap(self, image: torch.Tensor, model: str, max_side: int = 1024):
        div = 16
        b, h, w, c = image.shape
        orig_w, orig_h = w, h
        device = _get_device()

        # Backward compatibility: old workflows may send a numeric string as model
        # (e.g. '1504' from old max_side-only version). Treat as custom.
        if model not in LATENT_SIZES and model != "custom":
            try:
                max_side = int(model)
                model = "custom"
            except (ValueError, TypeError):
                model = "FLUX Klein"  # fallback

        if model == "custom":
            if max_side == 0:
                # max_side=0: stretch до ближайшего ×16
                new_w = max(div, round(w / div) * div)
                new_h = max(div, round(h / div) * div)
                mode_info = "custom align ×16 only"
            else:
                # Custom: max_side + округление до кратного 16
                max_side_snapped = max(div, round(max_side / div) * div)
                if w >= h:
                    new_w = max_side_snapped
                    new_h = max(div, round(h * (max_side_snapped / w) / div) * div)
                else:
                    new_h = max_side_snapped
                    new_w = max(div, round(w * (max_side_snapped / h) / div) * div)
                mode_info = f"custom max_side={max_side}"
        else:
            # Модель: ищем ближайший размер по аспекту из таблицы
            sizes = LATENT_SIZES[model]
            new_w, new_h = self._find_best_size(w, h, sizes)

            if max_side > 0:
                # max_side задан: пропорциональное масштабирование + ×16
                # Без привязки к таблице — минимальный stretch
                max_side_snapped = max(div, round(max_side / div) * div)
                if w >= h:
                    new_w = max_side_snapped
                    new_h = max(div, round(h * (max_side_snapped / w) / div) * div)
                else:
                    new_h = max_side_snapped
                    new_w = max(div, round(w * (max_side_snapped / h) / div) * div)
                mode_info = f"{model} @{max_side_snapped}"
            else:
                # max_side=0: размер из таблицы как есть
                mode_info = model

        # Считаем деформацию аспекта
        orig_aspect = w / h
        new_aspect = new_w / new_h
        aspect_change = abs(orig_aspect - new_aspect) / orig_aspect * 100

        info = f"{w}x{h} -> {new_w}x{new_h} | {mode_info} | stretch {aspect_change:.1f}%"

        if new_w == w and new_h == h:
            return (image, new_w, new_h, orig_w, orig_h,
                    info + " (no change)")

        # Непропорциональный stretch на GPU — bicubic
        img_bchw = image.permute(0, 3, 1, 2).to(device=device,
                                                  dtype=torch.float32)
        resized = F.interpolate(
            img_bchw, size=(new_h, new_w),
            mode='bicubic', align_corners=False).clamp(0, 1)
        result = resized.permute(0, 2, 3, 1).cpu()

        print(f"[LatentSizeSnap] {info}")
        return (result, new_w, new_h, orig_w, orig_h, info)
