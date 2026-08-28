"""
Happyin Auto Upscale — conditional GAN upscale.

If the longest side of the input image is below a configurable threshold,
upscales with 4xNomos8k_atd_jpg (ATD architecture, JPEG-optimized).
Otherwise passes through unchanged with zero overhead.

Model auto-downloads on first run (~78 MB).
Optimized for H100: fp16 tiled inference, OOM-safe fallback.
"""

import os
import torch
import numpy as np
from comfy import model_management
import comfy.utils
import folder_paths

_MODEL_NAME = "4xNomos8k_atd_jpg.pth"
_MODEL_URL = (
    "https://github.com/Phhofm/models/releases/download/"
    "4xNomos8k_atd_jpg/4xNomos8k_atd_jpg.pth"
)

_upscale_model_cache = None


def _ensure_upscale_model():
    """Load 4xNomos8k_atd_jpg via spandrel, download if missing."""
    global _upscale_model_cache
    if _upscale_model_cache is not None:
        return _upscale_model_cache

    from spandrel import ModelLoader, ImageModelDescriptor

    model_dir = os.path.join(folder_paths.models_dir, "upscale_models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, _MODEL_NAME)

    if not os.path.isfile(model_path):
        print(f"[HappyinAutoUpscale] Downloading {_MODEL_NAME} (~78 MB) ...")
        import urllib.request
        urllib.request.urlretrieve(_MODEL_URL, model_path)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"[HappyinAutoUpscale] Downloaded ({size_mb:.1f} MB)")

    sd = comfy.utils.load_torch_file(model_path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
    model = ModelLoader().load_from_state_dict(sd).eval()

    if not isinstance(model, ImageModelDescriptor):
        raise Exception(f"{_MODEL_NAME} is not a valid upscale model")

    _upscale_model_cache = model
    print(f"[HappyinAutoUpscale] Model loaded: "
          f"scale={model.scale}x, arch={type(model.model).__name__}")
    return model


class HappyinAutoUpscale:
    """Conditional GAN upscale: small images get upscaled, large pass through.

    Uses 4xNomos8k_atd_jpg — ATD transformer trained on Nomos8k dataset,
    optimized for JPEG artifact removal + 4x super-resolution.
    Model auto-downloads on first run (~78 MB).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "max_side": ("INT", {
                    "default": 1500, "min": 256, "max": 8192, "step": 8,
                    "tooltip": "Порог по большей стороне (px). "
                               "Если макс. сторона < порога — апскейл 4x через GAN до ~max_side. "
                               "Если >= порога — пропуск без изменений.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "Happyin/Image"

    DESCRIPTION = (
        "Auto Upscale — условный апскейл через 4xNomos8k_atd_jpg\n\n"
        "Если макс. сторона < max_side → апскейл 4x через GAN.\n"
        "Если >= max_side → пропуск без задержки (zero overhead).\n"
        "Модель ATD (Adaptive Token Dictionary) — JPEG-оптимизирована.\n"
        "Скачивается автоматически (~78 МБ)."
    )

    def run(self, images, max_side):
        h, w = images.shape[1], images.shape[2]
        longest = max(h, w)

        if longest >= max_side:
            print(f"[HappyinAutoUpscale] {w}x{h} >= {max_side}px, pass-through")
            return (images,)

        print(f"[HappyinAutoUpscale] {w}x{h} < {max_side}px, "
              "upscaling with 4xNomos8k_atd_jpg ...")

        model = _ensure_upscale_model()
        device = model_management.get_torch_device()

        # Memory estimate (same formula as ComfyUI built-in)
        memory_required = model_management.module_size(model.model)
        memory_required += (512 * 512 * 3) * images.element_size() \
            * max(model.scale, 1.0) * 384.0
        memory_required += images.nelement() * images.element_size()
        model_management.free_memory(memory_required, device)

        model.to(device)
        in_img = images.movedim(-1, -3).to(device)

        # H100 80GB: start with larger tile for fewer passes
        tile = 768
        overlap = 32

        oom = True
        try:
            while oom:
                try:
                    steps = in_img.shape[0] \
                        * comfy.utils.get_tiled_scale_steps(
                            in_img.shape[3], in_img.shape[2],
                            tile_x=tile, tile_y=tile, overlap=overlap)
                    pbar = comfy.utils.ProgressBar(steps)
                    s = comfy.utils.tiled_scale(
                        in_img,
                        lambda a: model(a),
                        tile_x=tile, tile_y=tile, overlap=overlap,
                        upscale_amount=model.scale, pbar=pbar)
                    oom = False
                except model_management.OOM_EXCEPTION as e:
                    tile //= 2
                    if tile < 128:
                        raise e
        finally:
            model.to("cpu")

        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        out_h, out_w = s.shape[1], s.shape[2]

        # Cap GAN output to max_side (4x upscale can overshoot)
        out_longest = max(out_h, out_w)
        if out_longest > max_side:
            scale = max_side / out_longest
            cap_w = max(1, round(out_w * scale))
            cap_h = max(1, round(out_h * scale))
            s = torch.nn.functional.interpolate(
                s.movedim(-1, -3),
                size=(cap_h, cap_w),
                mode="bicubic",
                align_corners=False,
            ).movedim(-3, -1).clamp(0, 1)
            print(f"[HappyinAutoUpscale] Done: {w}x{h} → {out_w}x{out_h} "
                  f"→ capped {cap_w}x{cap_h}")
        else:
            print(f"[HappyinAutoUpscale] Done: {w}x{h} → {out_w}x{out_h}")
        return (s,)
