"""
Happyin Paste Back — composite edited crop back into original image.

CROP_DATA from PersonMask carries everything: original, mask, bbox.
Only two wires needed: crop_data + edited_crop.

Flow: PersonMask.crop -> process -> PasteBack(edited_crop, crop_data)
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2

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

    result = (mean_a * g + mean_b).astype(np.float32)
    return np.nan_to_num(result, nan=0.0)


class HappyinPasteBack:
    """Paste an edited crop back into the original image.

    blend_mode:
      guided          — two-pass Guided Filter on seg mask (recommended)
      box             — blend entire bbox rectangle (Gaussian-feathered edges)
      seamless        — Laplacian pyramid + GuidedFilter
      seamless_guided — GuidedFilter + Laplacian pyramid (best overall)
      laplacian       — multi-scale Laplacian pyramid (best for hair)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "edited_crop": ("IMAGE",),
                "crop_data": ("CROP_DATA",),
                "blend_mode": (["guided", "box", "seamless",
                               "seamless_guided", "laplacian"], {
                    "default": "seamless_guided",
                    "tooltip": "guided = Guided Filter двухпроходный "
                               "(рекомендуется), "
                               "box = bbox с растушёвкой, "
                               "seamless = Laplacian pyramid + GuidedFilter, "
                               "seamless_guided = GuidedFilter + Laplacian "
                               "(лучший общий результат, без ореолов), "
                               "laplacian = пирамидальное смешивание "
                               "(лучший для волос)",
                }),
                "feather": ("INT", {
                    "default": 30, "min": 0, "max": 200, "step": 1,
                    "tooltip": "Расширение маски сегментации (пикселей). "
                               "Создаёт переходную зону вокруг объекта. "
                               "Для box — Gaussian растушёвка прямоугольника.",
                }),
                "guide_radius": ("INT", {
                    "default": 16, "min": 1, "max": 128, "step": 1,
                    "tooltip": "Радиус guided filter (coarse pass). "
                               "Fine pass = radius//4 (для волос). "
                               "16 = хороший баланс. Меньше = точнее края.",
                }),
                "guide_sigma": ("FLOAT", {
                    "default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Чувствительность guided filter к краям. "
                               "Меньше = точнее следует за деталями. "
                               "0.25 = хорошо для волос.",
                }),
            },
            "optional": {
                "original": ("IMAGE", {
                    "tooltip": "Подменить оригинал из crop_data своей картинкой. "
                               "Если не подключено — берётся оригинал из crop_data.",
                }),
                "color_match": ("BOOLEAN", {
                    "default": True,
                    "label_on": "on", "label_off": "off",
                    "tooltip": "Color harmonization: перед блендингом подгоняет "
                               "средний цвет/яркость кропа к оригиналу в зоне "
                               "перехода (LAB mean-shift). Убирает ореол от "
                               "разницы тона между AI и оригиналом.",
                }),
                "crop_mask": ("MASK", {
                    "tooltip": "Маска защиты кропа (размер = edited_crop).\n"
                               "Белое (1.0) = защищённая зона — 100% "
                               "отредактированный кроп, без перехода.\n"
                               "Чёрное (0.0) = зона смешивания.\n"
                               "Если не подключено — используется "
                               "автоматическая маска сегментации.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "Happyin/Detect"

    DESCRIPTION = (
        "Paste Back — вклейка обработанного кропа обратно в оригинал.\n\n"
        "blend_mode:\n"
        "  guided — Guided Filter двухпроходный (рекомендуется)\n"
        "  box — весь прямоугольник с Gaussian растушёвкой\n"
        "  seamless — Laplacian pyramid + GuidedFilter\n"
        "  seamless_guided — GuidedFilter + Laplacian пирамида\n"
        "  laplacian — пирамидальное смешивание (лучший для волос)\n\n"
        "Вклейка ВСЕГДА в ТОЧНО то же место откуда вырезали.\n"
        "Маска встаёт на то же место. Режим смешивания —\n"
        "только формула микширования, ничего не двигает."
    )

    # ── Fill black gaps in bbox ────────────────────────────────

    def _fill_black_gaps(self, result, orig, resized, y1, x1, y2, x2,
                         feather):
        """Fill near-black pixels that leaked from original into result.

        In head-swap workflows, the original image often has a black
        rectangle where the head was removed.  If the blend mask
        (VITMatte / seg) covers only the person *silhouette* (not the
        full rectangle), pixels between the silhouette edge and the
        bbox edge remain black.

        Detection: pixel was black in *original* AND is still black
        in the composited *result*.  This avoids touching legitimately
        dark areas that were already present in both images.

        Fill: replace those gaps with the edited crop (resized to bbox).
        A small Gaussian blur on the gap mask prevents a hard seam.
        """
        resized = resized.to(result.device)  # guard: caller may pass CPU tensor
        orig_region = orig[y1:y2, x1:x2, :]
        # Pixels that are near-black in the ORIGINAL (< ~2.5 / 255)
        black_in_orig = (orig_region.mean(dim=-1) < 0.01)
        # ... and are still near-black after compositing
        final_region = result[y1:y2, x1:x2, :]
        still_black = (final_region.mean(dim=-1) < 0.02)
        gaps = black_in_orig & still_black
        n_gaps = int(gaps.sum())
        if n_gaps < 50:
            return  # nothing to fill

        gap_np = gaps.float().cpu().numpy()
        # Soft edge to avoid hard seam at gap boundary
        blur_r = max(feather, 5)
        ks = blur_r * 2 + 1
        if ks % 2 == 0:
            ks += 1
        gap_soft = cv2.GaussianBlur(gap_np, (ks, ks), blur_r / 2.0)
        gap_soft = np.clip(np.maximum(gap_soft, gap_np), 0, 1)
        gap_t = torch.from_numpy(gap_soft).to(result.device).unsqueeze(-1)
        result[y1:y2, x1:x2, :] = (
            final_region * (1.0 - gap_t) + resized * gap_t)
        print(f"[HappyinPasteBack]   filled {n_gaps} black gap px "
              f"in bbox (head-swap fix)")

    # ── Gaussian feathered box ──────────────────────────────────

    def _make_feathered_box(self, h, w, feather, device):
        """Gaussian-feathered rectangular mask.

        Creates a hard rectangle of ones, pads with zeros, applies
        separable Gaussian blur. The convolution without padding naturally
        crops back to the original hxw size.
        """
        if feather <= 0:
            return torch.ones(h, w, device=device)

        feather = min(feather, h // 2, w // 2)
        if feather <= 0:
            return torch.ones(h, w, device=device)

        pad = feather
        ks = 2 * pad + 1

        # Hard rectangle padded with zeros
        padded = torch.zeros(h + 2 * pad, w + 2 * pad, device=device)
        padded[pad:pad + h, pad:pad + w] = 1.0

        # 1D Gaussian kernel (sigma ≈ feather/2.5 for smooth falloff)
        sigma = feather / 2.5
        x = torch.arange(ks, dtype=torch.float32, device=device) - pad
        k1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        k1d = k1d / k1d.sum()

        m = padded.unsqueeze(0).unsqueeze(0)  # [1, 1, H+2p, W+2p]

        # Separable Gaussian blur (no conv padding -> auto-crops to hxw)
        m = F.conv2d(m, k1d.view(1, 1, 1, -1))  # horizontal -> [1,1, H+2p, W]
        m = F.conv2d(m, k1d.view(1, 1, -1, 1))  # vertical   -> [1,1, H, W]

        return m[0, 0].clamp(0, 1)

    # ── Guided Filter (He, Sun, Tang 2010) ──────────────────────

    def _guided_filter(self, guide_rgb, mask_2d, radius, sigma):
        """Multi-pass edge-aware mask refinement using guided filter.

        guide_rgb: (H, W, 3) numpy float32 — guide image (RGB, 0-1 range)
        mask_2d:   (H, W) numpy float32     — blend mask to refine
        radius:    filter window radius (coarse pass)
        sigma:     edge sensitivity (eps — lower = stricter edge snapping)

        Two-pass strategy:
          1) Fine pass (radius//3, eps*0.1) — hair strands, thin edges
          2) Coarse pass (full radius, full eps) — jaw, forehead, shoulders
        Result = 50% fine + 50% coarse.
        """
        src = mask_2d.astype(np.float32)

        # Convert guide to uint8
        guide_u8 = (guide_rgb * 255).clip(0, 255).astype(np.uint8)

        # Fine pass: small window, very edge-sensitive — hair strands
        fine_r = max(2, radius // 3)
        fine_eps = sigma * 0.1
        fine = _guided_filter_single(guide_u8, src, fine_r, fine_eps)

        # Coarse pass: large window — overall contour
        coarse = _guided_filter_single(guide_u8, src, radius, sigma)

        result = 0.5 * fine + 0.5 * coarse
        return np.clip(result, 0, 1)

    # ── Color harmonization ──────────────────────────────────────

    def _color_harmonize(self, edited_np, orig_np, edge_mask_np,
                         strength=1.0):
        """Mean-shift color harmonization in LAB space.

        Computes the average color difference between edited and original
        in the edge zone, then shifts the entire edited crop to match.
        Preserves AI edits (texture, local changes) — only removes the
        global tone/brightness offset that causes visible seams.

        edited_np : [H, W, 3] float32 0-1
        orig_np   : [H, W, 3] float32 0-1
        edge_mask_np : [H, W] float32 0-1 (weights for edge zone)
        strength  : float 0-1 (1.0 = full correction, 0.5 = half)
        Returns   : [H, W, 3] float32 0-1 (corrected edited)
        """
        eps = 1e-6
        w = edge_mask_np
        w_sum = w.sum() + eps

        # Need enough pixels in edge zone for reliable statistics
        if w_sum < 100:
            return edited_np

        edited_u8 = (np.clip(edited_np, 0, 1) * 255).astype(np.uint8)
        orig_u8 = (np.clip(orig_np, 0, 1) * 255).astype(np.uint8)

        edited_lab = cv2.cvtColor(edited_u8, cv2.COLOR_RGB2LAB).astype(
            np.float32)
        orig_lab = cv2.cvtColor(orig_u8, cv2.COLOR_RGB2LAB).astype(
            np.float32)

        # Mean color difference per LAB channel in edge zone
        # Clamp shifts to prevent over-correction (L=±12, a/b=±8)
        # Without clamping, large AI brightness differences (L shift -30+)
        # cause the entire crop to darken/brighten dramatically -> visible seam
        MAX_SHIFT_L = 12.0   # lightness: ±12 LAB units max
        MAX_SHIFT_AB = 8.0   # chroma: ±8 LAB units max

        shifts = []
        clamped = []
        for c in range(3):
            e_mean = (edited_lab[:, :, c] * w).sum() / w_sum
            o_mean = (orig_lab[:, :, c] * w).sum() / w_sum
            raw_shift = o_mean - e_mean
            if c == 0:  # L channel
                shift = np.clip(raw_shift, -MAX_SHIFT_L, MAX_SHIFT_L)
            else:  # a, b channels
                shift = np.clip(raw_shift, -MAX_SHIFT_AB, MAX_SHIFT_AB)
            shift *= strength
            edited_lab[:, :, c] += shift
            shifts.append(shift)
            clamped.append(abs(raw_shift) > abs(shift) + 0.5)

        edited_lab = np.clip(edited_lab, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(edited_lab, cv2.COLOR_LAB2RGB).astype(
            np.float32) / 255.0

        clamp_note = " [CLAMPED]" if any(clamped) else ""
        str_note = f" @{strength:.0%}" if strength < 1.0 else ""
        print(f"[HappyinPasteBack]   color_match: LAB shift "
              f"L={shifts[0]:+.1f} a={shifts[1]:+.1f} "
              f"b={shifts[2]:+.1f}{clamp_note}{str_note}")
        return result

    # ── Bbox edge fade ──────────────────────────────────────────

    def _bbox_edge_fade(self, mask_np, margin):
        """Fade mask to 0 near bbox edges to prevent hard rectangle seam.

        Without this, the blend mask can reach the bbox boundary and
        create a visible hard rectangle where it abruptly stops.
        The fade ensures a smooth transition to the original at all
        bbox edges.

        mask_np : (H, W) float32 [0..1]
        margin  : fade width in pixels
        Returns : (H, W) float32 [0..1] with edges faded
        """
        if margin <= 0:
            return mask_np
        h, w = mask_np.shape
        margin = min(margin, h // 3, w // 3)
        if margin <= 0:
            return mask_np

        fade = np.ones((h, w), dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, margin, dtype=np.float32)

        for i in range(margin):
            fade[i, :] = np.minimum(fade[i, :], ramp[i])
            fade[h - 1 - i, :] = np.minimum(fade[h - 1 - i, :], ramp[i])
            fade[:, i] = np.minimum(fade[:, i], ramp[i])
            fade[:, w - 1 - i] = np.minimum(fade[:, w - 1 - i], ramp[i])

        return mask_np * fade

    # ── Final edge safety fade ────────────────────────────────────

    # ── Segmentation mask helpers ────────────────────────────────

    def _get_seg_mask_region(self, masks, y1, x1, y2, x2, feather):
        """Extract segmentation mask for the bbox region, dilated by feather.

        Returns (H, W) float32 numpy array in [0, 1].
        The mask follows the actual hair/face contour from PersonMask,
        dilated outward to create a transition zone for guided filter.
        """
        m = masks[0]
        if m.dim() == 3:
            m = m.squeeze(0)
        region = m[y1:y2, x1:x2].cpu().numpy().astype(np.float32)

        # Dilate to expand the mask boundary — creates a transition zone
        # that the guided filter can then refine using image edges
        if feather > 0:
            dilate_px = feather
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dilate_px * 2 + 1, dilate_px * 2 + 1))
            region_bin = (region > 0.1).astype(np.uint8)
            dilated = cv2.dilate(region_bin, kernel, iterations=1)
            # Gaussian blur the dilated zone for soft transition
            sigma = feather / 2.5
            ks = feather * 2 + 1
            if ks % 2 == 0:
                ks += 1
            soft = cv2.GaussianBlur(
                dilated.astype(np.float32), (ks, ks), sigma)
            # Keep original mask interior at 1.0, soft edge outside
            result = np.maximum(region, soft)
        else:
            result = region

        return np.clip(result, 0, 1)

    # ── Laplacian Pyramid Blending ────────────────────────────

    def _laplacian_blend(self, src_np, dst_np, mask_np, levels=6):
        """Multi-scale blending via Laplacian pyramids (Burt & Adelson 1983).

        Fine details (hair strands) get sharp transitions while large-scale
        colour/lighting differences transition smoothly. Significantly
        better for hair than single-scale alpha blending.

        src_np, dst_np: (H, W, 3) float32 [0..1]
        mask_np:        (H, W) float32 [0..1]
        """
        h, w = src_np.shape[:2]
        max_levels = int(np.log2(min(h, w))) - 1
        levels = min(levels, max_levels)
        if levels < 2:
            # Fallback to simple alpha blend for tiny regions
            m3 = mask_np[:, :, np.newaxis]
            return np.clip(m3 * src_np + (1 - m3) * dst_np, 0, 1)

        def _gaussian_pyramid(img, n):
            pyr = [img.astype(np.float32)]
            for _ in range(n):
                img = cv2.pyrDown(img)
                pyr.append(img.astype(np.float32))
            return pyr

        def _laplacian_pyramid(gauss_pyr):
            lap = []
            for i in range(len(gauss_pyr) - 1):
                up = cv2.pyrUp(
                    gauss_pyr[i + 1],
                    dstsize=(gauss_pyr[i].shape[1], gauss_pyr[i].shape[0]))
                lap.append(gauss_pyr[i] - up)
            lap.append(gauss_pyr[-1])
            return lap

        gp_src = _gaussian_pyramid(src_np, levels)
        gp_dst = _gaussian_pyramid(dst_np, levels)
        gp_mask = _gaussian_pyramid(mask_np, levels)

        lp_src = _laplacian_pyramid(gp_src)
        lp_dst = _laplacian_pyramid(gp_dst)

        # Blend at each level
        blended_pyr = []
        for ls, ld, gm in zip(lp_src, lp_dst, gp_mask):
            m = gm[:, :, np.newaxis] if gm.ndim == 2 else gm
            if m.shape[:2] != ls.shape[:2]:
                m = cv2.resize(m, (ls.shape[1], ls.shape[0]))
                if m.ndim == 2:
                    m = m[:, :, np.newaxis]
            blended_pyr.append(m * ls + (1 - m) * ld)

        # Reconstruct
        result = blended_pyr[-1]
        for i in range(len(blended_pyr) - 2, -1, -1):
            result = cv2.pyrUp(
                result,
                dstsize=(blended_pyr[i].shape[1], blended_pyr[i].shape[0]))
            result += blended_pyr[i]

        return np.clip(np.nan_to_num(result, nan=0.0), 0, 1)

    # ── Main ────────────────────────────────────────────────────

    def run(self, edited_crop, crop_data, blend_mode, feather,
            guide_radius, guide_sigma, original=None,
            color_match=True, crop_mask=None):
        # ── 1×1 blocked signal → passthrough original ──────────────
        # In the Happyin routing system, blocked branches output 1×1
        # black tensors.  If edited_crop is 1×1, there is nothing to
        # paste — return the original image unchanged.
        ec_h, ec_w = edited_crop.shape[1], edited_crop.shape[2]
        if ec_h <= 1 or ec_w <= 1:
            originals = crop_data['originals']
            orig = originals[0] if original is None else original[0]
            print(f"[HappyinPasteBack] edited_crop is {ec_w}x{ec_h} "
                  f"(blocked) -> passthrough original")
            return (orig.unsqueeze(0),)

        bboxes = crop_data['bboxes']
        originals = crop_data['originals']
        masks = crop_data['masks']
        pad_info_list  = crop_data.get('pad_info',  [None])
        trim_info_list = crop_data.get('trim_info', [None])

        crop = edited_crop[0]   # [cH, cW, C]

        # ── Undo square_crop padding ─────────────────────────────
        # BGpad: strip the padding so resize targets real content.
        pad_info = pad_info_list[0] if pad_info_list else None
        if pad_info is not None:
            pad_y, pad_x, orig_h, orig_w = pad_info
            cH_sq, cW_sq = crop.shape[:2]
            crop = crop[pad_y:pad_y + orig_h, pad_x:pad_x + orig_w, :]
            print(f"[HappyinPasteBack]   undo sq-pad: "
                  f"{cW_sq}x{cH_sq} -> {orig_w}x{orig_h}")

        # ── Undo square_crop trim ─────────────────────────────────
        # Trim: edited square covers only the center of the bbox.
        # Expand back: fill trimmed zones with original image content,
        # place edited square at its original offset. No resize needed.
        trim_info = trim_info_list[0] if trim_info_list else None

        # Optional override for the canvas image
        if original is not None:
            orig = original[0]  # [H, W, C]
        else:
            orig = originals[0]

        # Segmentation mask (from PersonMask — includes hair/face detail)
        seg_full = masks[0]
        if seg_full.dim() == 3:
            seg_full = seg_full.squeeze(0)    # [1, H, W] -> [H, W]

        y1, x1, y2, x2 = bboxes[0]

        H, W, C = orig.shape

        # ── Sanity check: detect wrong crop_data ──────────────────
        # If crop_data.originals[0] is close in size to edited_crop,
        # the user likely connected crop_data from a PersonMask that
        # analyzed the EDITED CROP (not the original image).
        # bbox coordinates from such crop_data are in crop-space,
        # NOT in original-image space -> paste lands in wrong place.
        cd_orig = originals[0]
        cd_H, cd_W = cd_orig.shape[:2]
        ec_H, ec_W = edited_crop[0].shape[:2]
        if original is not None:
            # original is explicitly connected — check for mismatch
            if (abs(cd_H - ec_H) < max(20, ec_H * 0.05) and
                    abs(cd_W - ec_W) < max(20, ec_W * 0.05) and
                    (abs(cd_H - H) > H * 0.1 or abs(cd_W - W) > W * 0.1)):
                print(f"[HappyinPasteBack] ⚠ WARNING: crop_data.original "
                      f"({cd_W}x{cd_H}) ≈ edited_crop ({ec_W}x{ec_H}) "
                      f"but differs from connected original ({W}x{H}). "
                      f"crop_data likely from WRONG PersonMask "
                      f"(analyzed crop, not original). "
                      f"bbox ({x1},{y1})-({x2},{y2}) may be in crop-space!")

        # Also warn if bbox exceeds original image bounds significantly
        bbox_oob = (y1 < 0 or x1 < 0 or y2 > H or x2 > W)
        if bbox_oob:
            print(f"[HappyinPasteBack] ⚠ WARNING: bbox ({x1},{y1})-"
                  f"({x2},{y2}) exceeds original {W}x{H}!")

        # Mask must be EXACTLY the same spatial size as orig so that
        # masks[0][y1:y2, x1:x2] is pixel-perfect aligned with
        # result[y1:y2, x1:x2].  Matting / GuidedFilter may occasionally
        # produce a mask that is off by 1px — resize to match.
        mH, mW = seg_full.shape[-2], seg_full.shape[-1]
        if mH != H or mW != W:
            print(f"[HappyinPasteBack]   mask size {mW}x{mH} != image "
                  f"{W}x{H} -> resize to align")
            seg_full = torch.from_numpy(
                cv2.resize(seg_full.cpu().numpy(), (W, H),
                           interpolation=cv2.INTER_LINEAR)
            ).to(seg_full.device)
            # Patch masks list so downstream code uses the aligned mask
            if isinstance(masks, list):
                masks[0] = seg_full.unsqueeze(0) if seg_full.dim() == 2 \
                    else seg_full

        # Clamp bbox to image bounds (PersonDetect may add padding that
        # exceeds image edge by 1+ px; downstream slice would clip silently
        # but blend_mask is built from region_w/h -> size mismatch crash).
        y1 = max(0, min(int(y1), H))
        x1 = max(0, min(int(x1), W))
        y2 = max(0, min(int(y2), H))
        x2 = max(0, min(int(x2), W))

        region_h = y2 - y1
        region_w = x2 - x1

        # Degenerate bbox (PersonMask emitted y1==y2 or x1==x2 on edge image)
        # -> cv2.resize crashes with assertion dsize>0.  Return original.
        if region_h <= 0 or region_w <= 0:
            print(f"[HappyinPasteBack]   degenerate bbox "
                  f"({region_w}x{region_h}) -> passthrough")
            return (orig.unsqueeze(0),)

        # ── Apply trim undo (needs orig + bbox) ───────────────────
        if trim_info is not None:
            t_y, t_x, t_side = trim_info
            cH_sq, cW_sq = crop.shape[:2]
            # If crop was resized (e.g. by diffusion model), resize back
            # to expected trim square first, then apply trim undo normally.
            if cH_sq != t_side or cW_sq != t_side:
                crop_np = crop.cpu().numpy()
                crop_np = cv2.resize(
                    crop_np, (t_side, t_side),
                    interpolation=cv2.INTER_LANCZOS4)
                crop = torch.from_numpy(crop_np).to(crop.device)
                print(f"[HappyinPasteBack]   trim: resize crop "
                      f"{cW_sq}x{cH_sq} -> {t_side}x{t_side}")
            orig_bbox = orig[y1:y2, x1:x2, :].clone()
            orig_bbox[t_y:t_y + t_side, t_x:t_x + t_side, :] = crop
            crop = orig_bbox
            print(f"[HappyinPasteBack]   undo trim: "
                  f"{t_side}x{t_side} -> {region_w}x{region_h} "
                  f"(offset y={t_y} x={t_x})")

        cH, cW = crop.shape[:2]

        print(f"[HappyinPasteBack] === START ===")
        print(f"[HappyinPasteBack]   original: {W}x{H}")
        print(f"[HappyinPasteBack]   edited_crop: {cW}x{cH}")
        print(f"[HappyinPasteBack]   bbox: y={y1}..{y2}, x={x1}..{x2} "
              f"({region_w}x{region_h})")
        print(f"[HappyinPasteBack]   blend_mode: {blend_mode}, "
              f"feather: {feather}")

        # Portrait passthrough: just paste edited crop directly
        # Triggers when:
        #   a) crop_mask is fully white (≥99%) — portrait, no bg to preserve
        #   b) bbox covers ≥95% of image and no crop_mask at all
        bbox_area = region_h * region_w
        orig_area = H * W
        is_mask_fully_white = False
        if crop_mask is not None:
            _cm_check = crop_mask[0]
            if _cm_check.dim() == 3:
                _cm_check = _cm_check.squeeze(0)
            _white_ratio = float(
                (_cm_check > 0.99).sum()) / max(1, _cm_check.numel())
            is_mask_fully_white = _white_ratio > 0.99
            if is_mask_fully_white:
                print(f"[HappyinPasteBack]   crop_mask fully white "
                      f"({_white_ratio:.1%}) -> portrait passthrough")

        if is_mask_fully_white or (
                bbox_area >= orig_area * 0.95 and crop_mask is None):
            # Resize edited crop to match original if sizes differ
            if cH != H or cW != W:
                crop_np = crop.cpu().numpy()
                resized_np = cv2.resize(
                    crop_np, (W, H),
                    interpolation=cv2.INTER_LANCZOS4)
                result = torch.from_numpy(resized_np).to(crop.device)
                print(f"[HappyinPasteBack]   portrait passthrough "
                      f"(Lanczos {cW}x{cH} -> {W}x{H})")
            else:
                result = crop
                print(f"[HappyinPasteBack]   portrait passthrough (as-is)")

            # Color match even in passthrough — compare border strip
            # Portrait = full-frame replacement -> softer correction (50%)
            if color_match:
                border_w = max(20, min(H, W) // 20)
                border_w = min(border_w, H // 4, W // 4)
                border_mask = np.zeros((H, W), dtype=np.float32)
                border_mask[:border_w, :] = 1.0
                border_mask[-border_w:, :] = 1.0
                border_mask[:, :border_w] = 1.0
                border_mask[:, -border_w:] = 1.0
                edited_np = result.cpu().numpy().astype(np.float32)
                orig_np = orig.cpu().numpy().astype(np.float32)
                corrected = self._color_harmonize(
                    edited_np, orig_np, border_mask, strength=0.25)
                result = torch.from_numpy(corrected).to(result.device)

            print(f"[HappyinPasteBack] === END ===")
            return (result.unsqueeze(0),)

        result = orig.clone()

        # Resize edited crop to match the bbox region (Lanczos)
        if cH != region_h or cW != region_w:
            crop_np = crop.cpu().numpy()
            resized_np = cv2.resize(
                crop_np, (region_w, region_h),
                interpolation=cv2.INTER_LANCZOS4)
            resized = torch.from_numpy(resized_np).to(crop.device)
        else:
            resized = crop

        if cH != region_h or cW != region_w:
            print(f"[HappyinPasteBack]   resize: {cW}x{cH} -> "
                  f"{region_w}x{region_h}")


        # ── Crop mask (optional explicit control) ─────────────
        #    White = protected (100% edited), Black = blend zone.
        #    Resized to match bbox region, same as edited_crop.
        crop_mask_np = None
        if crop_mask is not None:
            cm = crop_mask[0]  # [H, W] or [1, H, W]
            if cm.dim() == 3:
                cm = cm.squeeze(0)
            cm_np = cm.cpu().numpy().astype(np.float32)
            cm_h, cm_w = cm_np.shape
            # If crop_mask is full-image size, crop to bbox region FIRST.
            # MaskExpand returns a full-size mask; resizing it directly to
            # bbox dims would squish the entire image -> mask misaligned with hair.
            orig_H, orig_W = orig.shape[:2]
            if cm_h == orig_H and cm_w == orig_W:
                cm_np = cm_np[y1:y2, x1:x2]
                cm_h, cm_w = cm_np.shape
            if cm_h != region_h or cm_w != region_w:
                cm_np = cv2.resize(
                    cm_np, (region_w, region_h),
                    interpolation=cv2.INTER_LINEAR)
            crop_mask_np = np.clip(cm_np, 0, 1)
            protected_px = int((crop_mask_np > 0.5).sum())
            print(f"[HappyinPasteBack]   crop_mask: {protected_px}px "
                  f"protected ({100 * protected_px / max(1, region_h * region_w):.1f}%)")

        # Segmentation mask cropped to bbox region — used by most modes
        # Expand at least guide_radius outward so guided filter has
        # enough edge context around the segmentation boundary
        gf_feather = max(feather, guide_radius)
        seg_region_np = self._get_seg_mask_region(
            masks, y1, x1, y2, x2, gf_feather)
        seg_core_np = masks[0]
        if seg_core_np.dim() == 3:
            seg_core_np = seg_core_np.squeeze(0)
        seg_core_np = seg_core_np[y1:y2, x1:x2].cpu().numpy().astype(
            np.float32)
        seg_px = int((seg_core_np > 0.1).sum())
        print(f"[HappyinPasteBack]   seg mask: {seg_px}px "
              f"({100 * seg_px / max(1, region_h * region_w):.1f}%)")

        # ── crop_mask override ─────────────────────────────────
        #    When crop_mask is connected, it becomes the PRIMARY mask
        #    for ALL blending operations: color matching edge zone,
        #    blend mask construction, and transition zone.
        #    Without this, color_match and blending use the original
        #    seg mask boundaries which don't match the actual edit
        #    area (especially after multi-step edits like head swap
        #    where the user combines before/after masks).
        if crop_mask_np is not None:
            seg_core_np = crop_mask_np
            # Wide Gaussian blur -> black zone becomes gradual blend zone
            # (sigma = feather -> transition extends ~3x feather pixels,
            #  so feather=30 -> ~90px smooth gradient into black zone
            #  for noise/texture continuity — no hard cutoff)
            if gf_feather > 0:
                _cm_sigma = float(gf_feather)
                _cm_ks = int(_cm_sigma * 6) | 1  # odd kernel, 6σ wide
                _cm_ks = max(_cm_ks, 3)
                _cm_blurred = cv2.GaussianBlur(
                    crop_mask_np, (_cm_ks, _cm_ks), _cm_sigma)
                seg_region_np = np.clip(
                    np.maximum(crop_mask_np, _cm_blurred), 0, 1)
            else:
                seg_region_np = crop_mask_np.copy()
            cm_core_px = int((seg_core_np > 0.1).sum())
            cm_region_px = int((seg_region_np > 0.01).sum())
            print(f"[HappyinPasteBack]   crop_mask overrides seg: "
                  f"core={cm_core_px}px, "
                  f"region={cm_region_px}px (blur sigma={gf_feather}px)")

        # ── Color harmonization ────────────────────────────────
        #    Match mean color of edited crop to original in edge zone.
        #    Removes global tone/brightness shift that causes halo.
        #    Portrait detection: if crop_mask covers >90% of bbox,
        #    it's a close-up -> softer correction (50%) to preserve
        #    AI-generated skin tones / lighting.
        if color_match:
            # Detect portrait from crop_mask coverage
            cm_strength = 1.0
            if crop_mask_np is not None:
                cm_coverage = float(
                    (crop_mask_np > 0.5).sum()) / max(
                    1, region_h * region_w)
                if cm_coverage > 0.90:
                    cm_strength = 0.25
                    print(f"[HappyinPasteBack]   crop_mask coverage "
                          f"{cm_coverage:.0%} -> portrait, "
                          f"color_match @25%")

            # Edge zone = dilated mask minus core mask
            edge_zone_np = np.clip(seg_region_np - seg_core_np, 0, 1)

            # Fallback: if edge zone is empty (e.g. crop_mask=100%),
            # use a border strip at bbox edges where crop meets original
            if edge_zone_np.sum() <= 50:
                border_w = max(feather, 10)
                border_w = min(border_w, region_h // 4, region_w // 4)
                edge_zone_np = np.zeros(
                    (region_h, region_w), dtype=np.float32)
                edge_zone_np[:border_w, :] = 1.0       # top
                edge_zone_np[-border_w:, :] = 1.0      # bottom
                edge_zone_np[:, :border_w] = 1.0       # left
                edge_zone_np[:, -border_w:] = 1.0      # right
                print(f"[HappyinPasteBack]   color_match: edge_zone empty "
                      f"-> border strip {border_w}px")

            if edge_zone_np.sum() > 50:
                edited_np = resized.cpu().numpy().astype(np.float32)
                orig_np = result[y1:y2, x1:x2, :].cpu().numpy().astype(
                    np.float32)
                corrected = self._color_harmonize(
                    edited_np, orig_np, edge_zone_np,
                    strength=cm_strength)
                resized = torch.from_numpy(corrected).to(resized.device)

        # ── Edge-touch detection (per-edge) ──────────────────────
        # Where the hair/person mask reaches a bbox edge, there is
        # no background to transition to.  Force seg_region_np = 1.0
        # at those edges so ALL downstream blend modes see "no
        # transition needed" there.  Untouched edges blend normally.
        # Gradient ramp avoids a hard seam at the boundary.
        _et_bw = max(min(feather, 20), 8)
        _et_bw = min(_et_bw, region_h // 6, region_w // 6)
        _et_check = max(5, _et_bw // 2)  # narrower strip for detection
        _et_thresh = 0.25
        _et_forced = []
        if _et_bw > 0 and _et_check > 0:
            # Ramp: 1.0 at bbox edge → 0.0 at _et_bw pixels in
            _ramp_v = np.linspace(1.0, 0.0, _et_bw, dtype=np.float32)
            _ramp_h = np.linspace(1.0, 0.0, _et_bw, dtype=np.float32)

            if seg_core_np[:_et_check, :].mean() > _et_thresh:
                # Top edge touched: force seg_region to 1.0 with ramp
                for i in range(_et_bw):
                    seg_region_np[i, :] = np.maximum(
                        seg_region_np[i, :], _ramp_v[i])
                _et_forced.append("top")

            if seg_core_np[-_et_check:, :].mean() > _et_thresh:
                for i in range(_et_bw):
                    seg_region_np[-(i + 1), :] = np.maximum(
                        seg_region_np[-(i + 1), :], _ramp_v[i])
                _et_forced.append("bottom")

            if seg_core_np[:, :_et_check].mean() > _et_thresh:
                for i in range(_et_bw):
                    seg_region_np[:, i] = np.maximum(
                        seg_region_np[:, i], _ramp_h[i])
                _et_forced.append("left")

            if seg_core_np[:, -_et_check:].mean() > _et_thresh:
                for i in range(_et_bw):
                    seg_region_np[:, -(i + 1)] = np.maximum(
                        seg_region_np[:, -(i + 1)], _ramp_h[i])
                _et_forced.append("right")

        # Build persistent overlay mask for re-applying edge-touch
        # after any GF/blur step that might reduce forced values.
        _et_overlay = np.zeros((region_h, region_w), dtype=np.float32)
        if _et_forced and _et_bw > 0:
            _ov_rv = np.linspace(1.0, 0.0, _et_bw, dtype=np.float32)
            _ov_rh = np.linspace(1.0, 0.0, _et_bw, dtype=np.float32)
            if "top" in _et_forced:
                for i in range(_et_bw):
                    _et_overlay[i, :] = np.maximum(
                        _et_overlay[i, :], _ov_rv[i])
            if "bottom" in _et_forced:
                for i in range(_et_bw):
                    _et_overlay[-(i + 1), :] = np.maximum(
                        _et_overlay[-(i + 1), :], _ov_rv[i])
            if "left" in _et_forced:
                for i in range(_et_bw):
                    _et_overlay[:, i] = np.maximum(
                        _et_overlay[:, i], _ov_rh[i])
            if "right" in _et_forced:
                for i in range(_et_bw):
                    _et_overlay[:, -(i + 1)] = np.maximum(
                        _et_overlay[:, -(i + 1)], _ov_rh[i])

        if _et_forced:
            print(f"[HappyinPasteBack]   edge-touch: mask reaches "
                  f"{', '.join(_et_forced)} -> no blend at "
                  f"those edges ({_et_bw}px ramp)")

        # ── Seamless — Laplacian pyramid blend at EXACT bbox position ──
        # No cv2.seamlessClone (requires center point calculation).
        # Instead: multi-scale Laplacian pyramid directly at [y1:y2, x1:x2].
        # Same principle as all other modes: bbox = source of truth,
        # paste at exact position, blend with mask.

        if blend_mode == "seamless":
            guide_np = result[y1:y2, x1:x2, :].cpu().numpy()
            _sm_mask = self._guided_filter(
                guide_np, seg_region_np, guide_radius, guide_sigma)
            _sm_mask = np.nan_to_num(_sm_mask, nan=0.0)
            # Clamp: GF can only ADD mask at edges, never REDUCE inside person
            _sm_mask = np.maximum(_sm_mask, seg_core_np)
            _sm_mask = np.clip(_sm_mask, 0, 1)
            # Laplacian pyramid blend — multi-scale, no position calculation
            src_np = resized.cpu().numpy().astype(np.float32)
            dst_np = result[y1:y2, x1:x2, :].cpu().numpy().astype(
                np.float32)
            blended_np = self._laplacian_blend(
                src_np, dst_np, _sm_mask, levels=6)
            # Soft edge blend
            ks = max(feather * 2 + 1, 5)
            if ks % 2 == 0:
                ks += 1
            soft = cv2.GaussianBlur(_sm_mask, (ks, ks), feather / 2.0)
            soft = np.clip(np.maximum(soft, _sm_mask), 0, 1)
            # Final edge-touch override on soft mask
            soft = np.maximum(soft, _et_overlay)
            m3 = soft[:, :, np.newaxis]
            blended_np = np.nan_to_num(
                m3 * blended_np + (1.0 - m3) * dst_np, nan=0.0)
            blended_np = np.clip(blended_np, 0, 1)
            result[y1:y2, x1:x2, :] = torch.from_numpy(blended_np).to(
                orig.device)
            print(f"[HappyinPasteBack]   seamless: Laplacian pyramid "
                  f"at bbox (6 levels, GuidedFilter "
                  f"r={guide_radius}, s={guide_sigma})")
            self._fill_black_gaps(
                result, orig, resized.to(orig.device), y1, x1, y2, x2, feather)
            print(f"[HappyinPasteBack] === END ===")
            return (result.unsqueeze(0),)

        # ── Seamless Guided: 3-layer with Laplacian edge ─────────
        #    Inner zone  -> 100% edited crop (exact pixels)
        #    Edge zone   -> Laplacian pyramid blend (multi-scale)
        #    Outside     -> 100% original
        #    No Poisson -> no gradient-domain halos.

        if blend_mode == "seamless_guided":
            resized = resized.to(orig.device)  # ensure same device after color_match numpy round-trip
            guide_np = result[y1:y2, x1:x2, :].cpu().numpy()

            # Inner mask: core seg refined by guided filter
            # -> boundary follows real photo edges (hair, skin)
            if crop_mask_np is not None:
                inner_np = crop_mask_np
            else:
                inner_np = self._guided_filter(
                    guide_np, seg_core_np, guide_radius, guide_sigma)
                inner_np = np.nan_to_num(inner_np, nan=0.0)
                # Clamp: GF can only ADD, never REDUCE inside person
                inner_np = np.maximum(inner_np, seg_core_np)

            # Re-apply edge-touch to inner_np — where mask reaches
            # bbox edge, inner must also be 1.0 so edge_zone = 0
            # (no blend at touched edges).
            inner_np = np.maximum(inner_np, _et_overlay)

            # Outer mask: dilated seg with gentle Gaussian blur.
            # NO bbox_edge_fade — with tight crop the person fills
            # the bbox and any fixed fade clips the person.
            ks = max(feather * 2 + 1, 5)
            if ks % 2 == 0:
                ks += 1
            outer_np = cv2.GaussianBlur(
                seg_region_np, (ks, ks), feather / 2.0)
            outer_np = np.clip(np.maximum(outer_np, seg_region_np), 0, 1)

            inner_mask = torch.from_numpy(inner_np).to(
                orig.device).unsqueeze(-1)
            outer_mask = torch.from_numpy(outer_np).to(
                orig.device).unsqueeze(-1)

            # Edge zone = transition strip between inner and outer
            edge_zone = (outer_mask - inner_mask).clamp(0, 1)

            # 3-layer composite
            orig_region = result[y1:y2, x1:x2, :]
            blended = orig_region.clone()

            # Layer 1: core = 100% edited crop (exact pixels)
            blended = blended * (1.0 - inner_mask) + resized * inner_mask

            # Layer 2: edge zone = Laplacian pyramid blend
            if edge_zone.max() > 0.01:
                src_np = resized.cpu().numpy().astype(np.float32)
                dst_np = orig_region.cpu().numpy().astype(np.float32)
                lap_np = self._laplacian_blend(
                    src_np, dst_np, outer_np, levels=6)
                lap_t = torch.from_numpy(lap_np).to(orig.device)
                blended = blended * (1.0 - edge_zone) + lap_t * edge_zone

            result[y1:y2, x1:x2, :] = blended

            edge_px = int((edge_zone.squeeze(-1) > 0.01).sum())
            print(f"[HappyinPasteBack]   seamless_guided: 3-layer "
                  f"(GuidedFilter + Laplacian edge, "
                  f"edge_zone={edge_px}px, "
                  f"f={feather}px)")
            self._fill_black_gaps(
                result, orig, resized, y1, x1, y2, x2, feather)
            print(f"[HappyinPasteBack] === END ===")
            return (result.unsqueeze(0),)

        # ── Build blend mask ────────────────────────────────────
        #    All modes: blending ONLY at edges (feather zone).
        #    Interior = 1.0 (100% edited), exterior = 0.0 (100% orig).

        if blend_mode == "box":
            # Box: feathered rectangle (legacy fallback)
            blend_mask = self._make_feathered_box(
                region_h, region_w, feather, orig.device)
            print(f"[HappyinPasteBack]   box (gaussian): "
                  f"{region_w}x{region_h}, feather={feather}px")

        elif blend_mode == "guided":
            guide_np = result[y1:y2, x1:x2, :].cpu().numpy()
            refined = self._guided_filter(
                guide_np, seg_region_np, guide_radius, guide_sigma)
            refined = np.nan_to_num(refined, nan=0.0)
            # Clamp: GF can only ADD mask at edges, never REDUCE inside person
            refined = np.maximum(refined, seg_core_np)
            blend_mask = torch.from_numpy(refined).to(orig.device)
            print(f"[HappyinPasteBack]   guided: seg mask + GuidedFilter "
                  f"(r={guide_radius}, s={guide_sigma}, f={feather}px)")

        elif blend_mode == "laplacian":
            guide_np = result[y1:y2, x1:x2, :].cpu().numpy()
            refined = self._guided_filter(
                guide_np, seg_region_np, guide_radius, guide_sigma)
            refined = np.nan_to_num(refined, nan=0.0)
            # Clamp: GF can only ADD mask at edges, never REDUCE inside person
            refined = np.maximum(refined, seg_core_np)
            # Merge crop_mask: protected areas forced to 1.0
            if crop_mask_np is not None:
                refined = np.maximum(refined, crop_mask_np.astype(np.float32))
                print(f"[HappyinPasteBack]   crop_mask merged into "
                      f"laplacian mask")
            # Laplacian pyramid blend in the bbox region
            src_np = resized.cpu().numpy().astype(np.float32)
            dst_np = result[y1:y2, x1:x2, :].cpu().numpy().astype(
                np.float32)
            blended_np = self._laplacian_blend(
                src_np, dst_np, refined, levels=6)

            # ── Extra edge softening ──
            # Laplacian pyramid can leave sharp fine-detail transitions
            # at mask boundary. Gentle Gaussian blur to smooth them.
            if crop_mask_np is not None:
                edge_soft = max(feather, 15)
            else:
                edge_soft = max(feather // 2, 8)
            ks = edge_soft * 2 + 1
            if ks % 2 == 0:
                ks += 1
            soft_mask = cv2.GaussianBlur(
                refined, (ks, ks), edge_soft / 2.0)
            soft_mask = np.clip(soft_mask, 0, 1)
            soft_mask = np.maximum(soft_mask, refined)
            # Final edge-touch override on soft mask
            soft_mask = np.maximum(soft_mask, _et_overlay)
            m3 = soft_mask[:, :, np.newaxis]
            blended_np = np.nan_to_num(
                m3 * blended_np + (1.0 - m3) * dst_np, nan=0.0)
            blended_np = np.clip(blended_np, 0, 1)

            result[y1:y2, x1:x2, :] = torch.from_numpy(blended_np).to(
                orig.device)
            print(f"[HappyinPasteBack]   laplacian: seg mask + "
                  f"GuidedFilter + Laplacian pyramid (6 levels, "
                  f"r={guide_radius}, s={guide_sigma}, f={feather}px) "
                  f"+ edge_soft={edge_soft}px")
            self._fill_black_gaps(
                result, orig, resized, y1, x1, y2, x2, feather)
            print(f"[HappyinPasteBack] === END ===")
            return (result.unsqueeze(0),)

        else:
            # Fallback: guided
            guide_np = result[y1:y2, x1:x2, :].cpu().numpy()
            refined = self._guided_filter(
                guide_np, seg_region_np, guide_radius, guide_sigma)
            refined = np.nan_to_num(refined, nan=0.0)
            refined = np.maximum(refined, seg_core_np)
            blend_mask = torch.from_numpy(refined).to(orig.device)
            print(f"[HappyinPasteBack]   fallback->guided "
                  f"(r={guide_radius}, s={guide_sigma})")

        # ── Merge crop_mask into blend mask ──────────────────────
        if crop_mask_np is not None:
            cm_t = torch.from_numpy(crop_mask_np.astype(np.float32)).to(orig.device)
            # Align cm_t to exact region dims
            if cm_t.shape[0] != region_h or cm_t.shape[1] != region_w:
                print(f"[HappyinPasteBack]   align cm_t "
                      f"{cm_t.shape[1]}x{cm_t.shape[0]} -> {region_w}x{region_h}")
                cm_t = torch.from_numpy(
                    cv2.resize(cm_t.cpu().numpy(), (region_w, region_h),
                               interpolation=cv2.INTER_LINEAR)
                ).to(orig.device)
            # Align blend_mask before merge
            if blend_mask.shape[0] != region_h or blend_mask.shape[1] != region_w:
                print(f"[HappyinPasteBack]   align blend_mask "
                      f"{blend_mask.shape[1]}x{blend_mask.shape[0]} -> {region_w}x{region_h}")
                blend_mask = torch.from_numpy(
                    cv2.resize(blend_mask.cpu().numpy(), (region_w, region_h),
                               interpolation=cv2.INTER_LINEAR)
                ).to(orig.device)

            bm_np = blend_mask.cpu().numpy()

            if blend_mode == "box":
                # Use seg_region_np (smoothed crop_mask) as authoritative mask.
                bm_np = seg_region_np.copy()
                print(f"[HappyinPasteBack]   crop_mask merged (box): "
                      f"seg_region as blend mask")

            # bbox_edge_fade ONLY for rectangular masks (box, seamless)
            # where hard bbox boundary would be visible.
            # For guided/laplacian — guided filter already handles edge
            # transitions; bbox_edge_fade with crop_padding=0 clips the
            # person (fills entire bbox).
            if blend_mode == "box":
                _fade_m = max(feather, 30)
                _bbox_f = self._bbox_edge_fade(
                    np.ones((bm_np.shape[0], bm_np.shape[1]),
                            dtype=np.float32),
                    _fade_m)
                bm_np = np.minimum(bm_np, _bbox_f)
                # Re-apply edge-touch AFTER bbox_edge_fade
                bm_np = np.maximum(bm_np, _et_overlay)
                print(f"[HappyinPasteBack]   bbox fade applied "
                      f"(margin={_fade_m}px)")

            blend_mask = torch.from_numpy(bm_np).to(orig.device)
        else:
            # No crop_mask path.
            # For box: apply bbox edge fade (rectangular mask needs it).
            # For guided/laplacian: guided filter output IS the
            # authoritative mask — bbox_edge_fade would clip person.
            bm_np = blend_mask.cpu().numpy()
            if blend_mode == "box":
                fade_margin = max(feather, 30)
                bbox_fade = self._bbox_edge_fade(
                    np.ones((region_h, region_w), dtype=np.float32),
                    fade_margin)
                bm_np = np.minimum(bm_np, bbox_fade)
                # Re-apply edge-touch AFTER bbox_edge_fade
                bm_np = np.maximum(bm_np, _et_overlay)
            blend_mask = torch.from_numpy(bm_np).to(orig.device)

        # Always align blend_mask to exact region dims (bbox is the truth)
        if blend_mask.shape[0] != region_h or blend_mask.shape[1] != region_w:
            print(f"[HappyinPasteBack]   align blend_mask "
                  f"{blend_mask.shape[1]}x{blend_mask.shape[0]} -> {region_w}x{region_h}")
            blend_mask = torch.from_numpy(
                cv2.resize(blend_mask.cpu().numpy(), (region_w, region_h),
                           interpolation=cv2.INTER_LINEAR)
            ).to(orig.device)

        blend_mask = blend_mask.unsqueeze(-1)  # [rH, rW, 1]

        # ── Composite ───────────────────────────────────────────

        # Always align resized to exact region dims (bbox is the truth)
        if resized.shape[0] != region_h or resized.shape[1] != region_w:
            print(f"[HappyinPasteBack]   align resized "
                  f"{resized.shape[1]}x{resized.shape[0]} -> {region_w}x{region_h}")
            resized = torch.from_numpy(
                cv2.resize(resized.cpu().numpy(), (region_w, region_h),
                           interpolation=cv2.INTER_LANCZOS4)
            ).to(orig.device)

        # Final composite — realign blend_mask if size still doesn't match
        # (edge case: guided filter / resize chain left 1px discrepancy)
        orig_region = result[y1:y2, x1:x2, :]
        if blend_mask.shape[0] != orig_region.shape[0] or \
                blend_mask.shape[1] != orig_region.shape[1]:
            # Strange situation — sizes still don't match after all
            # alignment passes.  Use simple feathered box: predictable,
            # no additional processing that could introduce new errors.
            _rH, _rW = orig_region.shape[:2]
            print(f"[HappyinPasteBack]   composite size mismatch "
                  f"bm={blend_mask.shape[:2]} vs region={(_rH, _rW)} "
                  f"-> box fallback")
            fb = max(min(feather, _rH // 4, _rW // 4), 0)
            blend_mask = self._make_feathered_box(
                _rH, _rW, fb, orig.device).unsqueeze(-1)
        if resized.shape[0] != orig_region.shape[0] or \
                resized.shape[1] != orig_region.shape[1]:
            resized = torch.from_numpy(
                cv2.resize(resized.cpu().numpy(),
                           (orig_region.shape[1], orig_region.shape[0]),
                           interpolation=cv2.INTER_LANCZOS4)
            ).to(orig.device)

        resized = resized.to(orig.device)  # guarantee same device after any numpy round-trip
        blended = orig_region * (1.0 - blend_mask) + resized * blend_mask
        blended = torch.nan_to_num(blended, nan=0.0)
        result[y1:y2, x1:x2, :] = blended

        self._fill_black_gaps(
            result, orig, resized, y1, x1, y2, x2, feather)
        print(f"[HappyinPasteBack] === END ===")

        return (result.unsqueeze(0),)
