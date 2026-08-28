"""
Happyin Mask: Expand — expand / shrink mask with guided-filter edge refinement.

Pipeline:
  1. Morphological dilation (or erosion if negative) by `expand_px` pixels
  2. Guided filter (cv2.ximgproc) — snaps mask edges to real image
     boundaries BEFORE blur (needs hard edges to snap to)
  3. Gaussian blur for soft falloff AFTER guided filter

Key: guided filter MUST run on the hard dilated mask, not the blurred
one. Blurring first destroys the edge signal the filter needs.
"""

import numpy as np
import cv2
import torch

# ── Guided filter: ximgproc with box-filter fallback ──────────────
_HAS_XIMGPROC = hasattr(cv2, 'ximgproc') and hasattr(
    getattr(cv2, 'ximgproc', None), 'guidedFilter')


def _guided_filter_single(guide, src, radius, eps):
    """Single-pass guided filter (He, Sun, Tang 2010).

    Uses cv2.ximgproc if available (opencv-contrib), otherwise falls
    back to a box-filter implementation that gives identical results.

    guide : (H, W, 3) uint8 or float32 — reference image
    src   : (H, W) float32             — signal to filter
    radius: int                         — window radius
    eps   : float                       — regularisation (edge sensitivity)
    Returns: (H, W) float32
    """
    if _HAS_XIMGPROC:
        return cv2.ximgproc.guidedFilter(guide, src, radius, eps)

    # Box-filter fallback — grayscale guide approximation
    if guide.dtype == np.uint8:
        g = guide.astype(np.float32) / 255.0
    else:
        g = guide.astype(np.float32).copy()

    if g.ndim == 3:
        g = 0.299 * g[:, :, 0] + 0.587 * g[:, :, 1] + 0.114 * g[:, :, 2]

    p = src.astype(np.float32)
    ksize = (2 * radius + 1, 2 * radius + 1)

    mean_I = cv2.boxFilter(g, -1, ksize)
    mean_p = cv2.boxFilter(p, -1, ksize)
    corr_Ip = cv2.boxFilter(g * p, -1, ksize)
    corr_II = cv2.boxFilter(g * g, -1, ksize)

    var_I = corr_II - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, ksize)
    mean_b = cv2.boxFilter(b, -1, ksize)

    return (mean_a * g + mean_b).astype(np.float32)


class HappyinMaskExpand:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask":      ("MASK",),
                "image":     ("IMAGE",),
                "expand_px": ("INT", {
                    "default": 10, "min": -100, "max": 200, "step": 1,
                    "tooltip": "Pixels to expand (positive) or shrink (negative)"
                }),
                "blur_radius": ("INT", {
                    "default": 5, "min": 0, "max": 50, "step": 1,
                    "tooltip": "Gaussian blur radius for soft falloff (0 = skip)"
                }),
                "guided_radius": ("INT", {
                    "default": 20, "min": 0, "max": 100, "step": 1,
                    "tooltip": "Guided filter radius — how far edges snap (0 = skip)"
                }),
                "edge_sensitivity": ("FLOAT", {
                    "default": 0.01, "min": 0.001, "max": 1.0, "step": 0.001,
                    "tooltip": "Lower = stricter edge snapping (eps for guided filter)"
                }),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "expand"
    CATEGORY = "Happyin/Mask"

    DESCRIPTION = (
        "Mask: Expand — расширение/сужение маски + guided filter.\n\n"
        "1. Morphological dilate/erode по expand_px\n"
        "2. Guided filter (cv2.ximgproc) — прищёлкивает края маски\n"
        "   к реальным границам изображения (волосы, одежда, кожа)\n"
        "3. Gaussian blur — мягкий переход\n\n"
        "guided_radius: радиус окна guided filter (больше = сильнее snap)\n"
        "edge_sensitivity: eps — меньше = точнее следует за краями"
    )

    # ── Main ───────────────────────────────────────────────────────

    def expand(self, mask, image, expand_px, blur_radius,
               guided_radius, edge_sensitivity):
        B = mask.shape[0]
        out = []

        for i in range(B):
            m = mask[i].cpu().numpy().astype(np.float32)

            # 1. Morphological expand / shrink
            if expand_px != 0:
                kernel_size = abs(expand_px) * 2 + 1
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
                )
                if expand_px > 0:
                    m = cv2.dilate(m, kernel, iterations=1)
                else:
                    m = cv2.erode(m, kernel, iterations=1)

            # 2. Guided filter — snap edges to image boundaries
            #    BEFORE blur: needs hard mask edges to snap properly
            if guided_radius > 0:
                img_np = image[min(i, image.shape[0] - 1)].cpu().numpy()
                img_np = img_np.astype(np.float32)
                if img_np.shape[:2] != m.shape[:2]:
                    img_np = cv2.resize(
                        img_np, (m.shape[1], m.shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )
                # Convert guide to uint8 for guided filter
                guide_u8 = (img_np * 255).clip(0, 255).astype(np.uint8)

                # Two-pass: fine (hair strands) + coarse (contours)
                fine_r = max(2, guided_radius // 3)
                fine_eps = edge_sensitivity * 0.1  # stricter for fine detail
                fine = _guided_filter_single(
                    guide_u8, m, fine_r, fine_eps)

                coarse = _guided_filter_single(
                    guide_u8, m, guided_radius, edge_sensitivity)

                m = np.clip(0.5 * fine + 0.5 * coarse, 0.0, 1.0)

            # 3. Gaussian blur for soft falloff (AFTER guided filter)
            if blur_radius > 0:
                ksize = blur_radius * 2 + 1
                m = cv2.GaussianBlur(m, (ksize, ksize), 0)

            m = np.clip(m, 0.0, 1.0).astype(np.float32)
            out.append(torch.from_numpy(m))

        return (torch.stack(out),)
