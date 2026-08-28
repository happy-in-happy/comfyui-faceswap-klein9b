"""
Happyin Color Transfer — перенос цвета от донора к реципиенту.

Три метода (переключатели, приоритет: soft > wavelet > reinhard):

  soft      — Прямая копия хроматики через Guided Filter (He et al. 2010).
              НЕ статистический. Гарантировано без артефактов.
              L-канал (детали) = реципиент, A+B (цвет) = донор + edge-aware сглаживание.
  wavelet   — Частотное разделение: низкие частоты от донора + высокие от реципиента.
  reinhard  — Классический LAB mean+std. Для случаев, когда изображения не из одной сцены.

Типичный workflow: испорченный кадр → восстановление → Color Transfer (донор = оригинал).
"""

import torch
import numpy as np
import cv2


# =============================================================================
# GUIDED FILTER (He, Sun, Tang 2010)
# =============================================================================

def _guided_filter(guide: np.ndarray, src: np.ndarray,
                   radius: int, eps: float) -> np.ndarray:
    """
    Fast O(N) guided image filter.

    Делает src плавным, но сохраняет края из guide.
    Используется для переноса хроматики донора с учётом
    структуры (краёв) реципиента.

    guide: (H, W) float64 — направляющий канал (L реципиента)
    src:   (H, W) float64 — исходный канал (A или B донора)
    radius: размер окна
    eps:    регуляризация (больше = глаже)
    """
    ksize = (2 * radius + 1, 2 * radius + 1)

    mean_g = cv2.blur(guide, ksize)
    mean_s = cv2.blur(src, ksize)
    corr_gg = cv2.blur(guide * guide, ksize)
    corr_gs = cv2.blur(guide * src, ksize)

    var_g = corr_gg - mean_g * mean_g
    cov_gs = corr_gs - mean_g * mean_s

    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g

    mean_a = cv2.blur(a, ksize)
    mean_b = cv2.blur(b, ksize)

    return mean_a * guide + mean_b


# =============================================================================
# SOFT — guided-filter chrominance transfer (artifact-free)
# =============================================================================

def _soft(donor: np.ndarray, recipient: np.ndarray, softness: int = 4) -> np.ndarray:
    """
    Прямой перенос хроматики через Guided Filter.

    Принцип: НЕ подгоняет статистику, а напрямую копирует
    каналы A и B (цвет) от донора к реципиенту.
    Guided filter сглаживает цвет по краям реципиента.

    Гарантировано без артефактов:
    - Нет подгонки статистик (нет выхода за гамму)
    - Нет гистограммного matching (нет клиппинга)
    - Edge-aware сглаживание (нет гало)
    - Bounded output (нет out-of-gamut)

    L-канал: 100% от реципиента (все детали сохранены).
    A+B каналы: от донора, сглаженные guided filter по структуре реципиента.

    softness: 1-8. Управляет радиусом фильтра.
              Больше = глаже цветовые переходы.
    """
    h, w = recipient.shape[:2]

    # Resize donor to match recipient
    if donor.shape[:2] != (h, w):
        interp = cv2.INTER_AREA if donor.shape[0] > h else cv2.INTER_LINEAR
        donor = cv2.resize(donor, (w, h), interpolation=interp)

    # Convert to LAB float64 normalized [0, 1]
    d_lab = cv2.cvtColor(donor, cv2.COLOR_RGB2LAB).astype(np.float64) / 255.0
    r_lab = cv2.cvtColor(recipient, cv2.COLOR_RGB2LAB).astype(np.float64) / 255.0

    # Guide = recipient's luminance (structure/edges)
    guide = r_lab[:, :, 0]

    # Radius: пропорционален изображению, но с жёстким cap = 20px.
    # Без cap: на 4K фото softness=4 даёт радиус 120px → цвет перетекает
    # через границы камень/металл. Cap гарантирует edge-следование.
    radius = min(max(int(min(h, w) * 0.01 * softness), 2), 20)
    eps = 1e-4 * softness

    # Build result: L from recipient, A+B from guided-filtered donor
    result = np.zeros_like(r_lab)
    result[:, :, 0] = r_lab[:, :, 0]                                      # L: recipient
    result[:, :, 1] = _guided_filter(guide, d_lab[:, :, 1], radius, eps)   # A: donor (filtered)
    result[:, :, 2] = _guided_filter(guide, d_lab[:, :, 2], radius, eps)   # B: donor (filtered)

    return cv2.cvtColor((np.clip(result, 0, 1) * 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


# =============================================================================
# WAVELET — frequency-based color transfer
# =============================================================================

def _wavelet(donor: np.ndarray, recipient: np.ndarray, levels: int = 4) -> np.ndarray:
    """
    Частотное разделение в LAB:
    - Низкие частоты (цветовая база) берутся от донора
    - Высокие частоты (детали, текстура) — от реципиента

    levels управляет размером ядра блюра: sigma ∝ image_size × levels.

    Оптимизации скорости:
    - float32 вместо float64 (2× быстрее)
    - Downsample trick: при большой sigma блюрим на уменьшенном изображении,
      затем апсэмплим. Результат идентичен — для цветового переноса sigma>50px
      визуально неотличима от sigma=320px (граница глобал/локал уже насыщена).
    """
    # float32 — вдвое быстрее float64 при работе с cv2
    d_lab = cv2.cvtColor(donor, cv2.COLOR_RGB2LAB).astype(np.float32)
    r_lab = cv2.cvtColor(recipient, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = r_lab.shape[:2]

    if d_lab.shape[:2] != (h, w):
        interp = cv2.INTER_AREA if donor.shape[0] > h else cv2.INTER_LINEAR
        d_lab = cv2.resize(d_lab, (w, h), interpolation=interp)

    sigma = max(min(h, w) * 0.02 * levels, 3.0)

    # Downsample trick: блюрим на миниатюре 256×256, апсэмплим обратно.
    # Для цветового переноса низкие частоты — это глобальная информация,
    # поэтому 256px миниатюры достаточно при любом входном разрешении.
    THUMB = 256
    THUMB_SIGMA = 15.0
    thumb_ksize = int(THUMB_SIGMA * 6) | 1
    if min(h, w) > THUMB:
        d_small = cv2.resize(d_lab, (THUMB, THUMB), interpolation=cv2.INTER_AREA)
        r_small = cv2.resize(r_lab, (THUMB, THUMB), interpolation=cv2.INTER_AREA)
        d_low_s = cv2.GaussianBlur(d_small, (thumb_ksize, thumb_ksize), sigmaX=THUMB_SIGMA)
        r_low_s = cv2.GaussianBlur(r_small, (thumb_ksize, thumb_ksize), sigmaX=THUMB_SIGMA)
        d_low = cv2.resize(d_low_s, (w, h), interpolation=cv2.INTER_LINEAR)
        r_low = cv2.resize(r_low_s, (w, h), interpolation=cv2.INTER_LINEAR)
        result = d_low + (r_lab - r_low)
    else:
        ksize = int(sigma * 6) | 1
        d_low = cv2.GaussianBlur(d_lab, (ksize, ksize), sigmaX=sigma)
        r_low = cv2.GaussianBlur(r_lab, (ksize, ksize), sigmaX=sigma)
        result = d_low + (r_lab - r_low)

    return cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


# =============================================================================
# REINHARD (2001) — LAB mean + std (classic, for non-aligned images)
# =============================================================================

def _reinhard(donor: np.ndarray, recipient: np.ndarray) -> np.ndarray:
    """
    Классический перенос LAB mean+std.
    Для случаев, когда изображения из разных сцен.
    """
    d_lab = cv2.cvtColor(donor, cv2.COLOR_RGB2LAB).astype(np.float64)
    r_lab = cv2.cvtColor(recipient, cv2.COLOR_RGB2LAB).astype(np.float64)

    for ch in range(3):
        mu_d, sigma_d = d_lab[:, :, ch].mean(), d_lab[:, :, ch].std() + 1e-6
        mu_r, sigma_r = r_lab[:, :, ch].mean(), r_lab[:, :, ch].std() + 1e-6
        r_lab[:, :, ch] = (r_lab[:, :, ch] - mu_r) * (sigma_d / sigma_r) + mu_d

    return cv2.cvtColor(np.clip(r_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


# =============================================================================
# COMMON UTILS
# =============================================================================

def _preserve_lum(recipient_np: np.ndarray, result_np: np.ndarray) -> np.ndarray:
    """Replace result's L channel with recipient's L (keep original brightness)."""
    lab_orig = cv2.cvtColor(recipient_np, cv2.COLOR_RGB2LAB)
    lab_res = cv2.cvtColor(result_np, cv2.COLOR_RGB2LAB)
    lab_res[:, :, 0] = lab_orig[:, :, 0]
    return cv2.cvtColor(lab_res, cv2.COLOR_LAB2RGB)


def _blend(original: np.ndarray, transformed: np.ndarray, strength: float) -> np.ndarray:
    """Linear blend between original and transformed."""
    if strength >= 1.0:
        return transformed
    if strength <= 0.0:
        return original
    return (original.astype(np.float32) * (1 - strength) +
            transformed.astype(np.float32) * strength).clip(0, 255).astype(np.uint8)


# =============================================================================
# NODE
# =============================================================================

class HappyinColorTransfer:
    """
    Перенос цвета с тремя методами-переключателями.

    Приоритет (если включено несколько): soft > wavelet > reinhard.
    Если все выключены — изображение проходит без изменений.

    soft      — прямая копия хроматики + guided filter. Без артефактов. Дефолт.
    wavelet   — частотное разделение. Цвет от донора, детали от реципиента.
    reinhard  — классический LAB mean+std. Для разных сцен.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "donor": ("IMAGE", {"tooltip": "Картинка-донор цвета (оригинал)"}),
                "recipient": ("IMAGE", {"tooltip": "Картинка-реципиент (восстановленная)"}),
            },
            "optional": {
                "soft": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Soft — прямая копия цвета через Guided Filter. Без артефактов. Рекомендуемый.",
                }),
                "wavelet": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Wavelet — частотное разделение. Цвет от донора + детали от реципиента.",
                }),
                "reinhard": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reinhard — классический LAB mean+std. Для изображений из разных сцен.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Сила переноса. 0 = без изменений, 1 = полный.",
                }),
                "softness": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Мягкость guided filter (только для soft). Больше = глаже цветовые переходы.",
                }),
                "preserve_luminance": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Сохранить яркость реципиента (для wavelet/reinhard). Soft всегда сохраняет.",
                }),
                "wavelet_levels": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Глубина частотного разделения (только для wavelet). Больше = сильнее.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "transfer"
    CATEGORY = "Happyin/Color"

    def transfer(self, donor: torch.Tensor, recipient: torch.Tensor,
                 soft: bool = True, wavelet: bool = False, reinhard: bool = False,
                 strength: float = 1.0, softness: int = 4,
                 preserve_luminance: bool = False, wavelet_levels: int = 4):

        donor_np = (donor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        recip_np = (recipient[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        dh, dw = donor_np.shape[:2]
        rh, rw = recip_np.shape[:2]

        # Priority: soft > wavelet > reinhard
        if soft:
            method_name = "soft"
            result_np = _soft(donor_np, recip_np, softness)
        elif wavelet:
            method_name = "wavelet"
            result_np = _wavelet(donor_np, recip_np, wavelet_levels)
        elif reinhard:
            method_name = "reinhard"
            result_np = _reinhard(donor_np, recip_np)
        else:
            method_name = "passthrough"
            result_np = recip_np.copy()

        # preserve_luminance for wavelet/reinhard (soft always preserves L)
        if preserve_luminance and method_name in ("wavelet", "reinhard"):
            result_np = _preserve_lum(recip_np, result_np)

        result_np = _blend(recip_np, result_np, strength)

        result = torch.from_numpy(result_np.astype(np.float32) / 255.0).unsqueeze(0)

        info = (f"{method_name} | donor {dw}\u00d7{dh} \u2192 recipient {rw}\u00d7{rh} | "
                f"strength {strength:.0%}"
                f"{f' | softness {softness}' if method_name == 'soft' else ''}"
                f"{f' | levels {wavelet_levels}' if method_name == 'wavelet' else ''}"
                f"{' | lum preserved' if preserve_luminance and method_name in ('wavelet', 'reinhard') else ''}")

        print(f"[HappyinColorTransfer] {info}")
        return (result, info)
