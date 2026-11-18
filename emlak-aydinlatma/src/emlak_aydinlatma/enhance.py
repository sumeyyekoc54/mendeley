"""Deterministic lighting enhancements (Lab L / soft curves / masks)."""

from __future__ import annotations

import cv2
import numpy as np

from .analysis import ImageAnalysis, luminance_rec709


def _lab_l(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Lab float image and L channel 0..100."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # OpenCV Lab: L in 0..255 for uint8; after float convert still 0..255 scale
    L = lab[:, :, 0] * (100.0 / 255.0)
    return lab, L


def _merge_lab_l(lab: np.ndarray, L: np.ndarray) -> np.ndarray:
    out = lab.copy()
    out[:, :, 0] = np.clip(L * (255.0 / 100.0), 0, 255)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def apply_exposure(L: np.ndarray, stops: float) -> np.ndarray:
    """Global exposure on Lab L (0..100), stops of light."""
    if abs(stops) < 1e-6:
        return L
    # Convert L≈0..100 to approx linear relative, multiply, back
    lin = np.clip(L / 100.0, 0.0, 1.0) ** 2.2
    lin = lin * (2.0 ** stops)
    return np.clip(lin ** (1.0 / 2.2), 0.0, 1.0) * 100.0


def soft_shadow_curve(L: np.ndarray, strength: float) -> np.ndarray:
    """Lift only low luminances with a smooth shadow curve."""
    if strength <= 0:
        return L
    x = np.clip(L / 100.0, 0.0, 1.0)
    # Lift amount peaks in deep shadows and falls off by midtones
    lift = strength * (1.0 - x) ** 2.2
    # Soft knead: never push shadows past ~mid
    y = x + lift * (0.55 - 0.35 * x)
    return np.clip(y, 0.0, 1.0) * 100.0


def highlight_rolloff(
    L: np.ndarray,
    original_L: np.ndarray,
    protect: bool = True,
    protect_strength: float = 0.85,
) -> np.ndarray:
    """Limit highlight *lift*; never crush below the original luminance.

    Soft shoulder only applies to values we pushed up; optional window mask
    blends boosted bright regions back toward the source.
    """
    x = np.clip(L / 100.0, 0.0, 1.0)
    src = np.clip(original_L / 100.0, 0.0, 1.0)

    # Soft shoulder on the working image above ~0.78 (only compress overshoot)
    knee = 0.78
    over = np.maximum(x - knee, 0.0)
    rolled = np.where(x > knee, knee + over / (1.0 + over * 2.8), x)

    # Never darker than the source (windows / sky fidelity)
    rolled = np.maximum(rolled, src)

    if protect:
        bright = np.clip((src - 0.62) / 0.32, 0.0, 1.0)
        k = max(15, (min(src.shape[:2]) // 40) | 1)
        if k % 2 == 0:
            k += 1
        mask = cv2.GaussianBlur(bright.astype(np.float32), (k, k), k * 0.35)
        # In bright regions, keep most of the original; allow only a fraction of lift
        lift = np.maximum(rolled - src, 0.0)
        allowed = lift * (1.0 - protect_strength * mask)
        rolled = src + allowed

    return np.clip(rolled, 0.0, 1.0) * 100.0


def mild_local_contrast(L: np.ndarray, amount: float) -> np.ndarray:
    """Very mild multi-scale local contrast on L (no aggressive CLAHE)."""
    if amount <= 0.01:
        return L
    x = (L / 100.0).astype(np.float32)
    blur = cv2.GaussianBlur(x, (0, 0), max(3.0, min(x.shape[:2]) / 80.0))
    detail = x - blur
    # Soften detail in deep shadows to limit noise
    shadow_w = np.clip(x / 0.18, 0.0, 1.0)
    out = x + detail * amount * shadow_w
    return np.clip(out, 0.0, 1.0) * 100.0


def optional_white_balance(bgr: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """Mild gray-world WB; strength 0..1. Preserves overall brightness."""
    if strength <= 0:
        return bgr
    img = bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0) + 1e-6
    gray = float(np.mean(means))
    gains = gray / means
    # Limit gain swing
    gains = 1.0 + (gains - 1.0) * strength
    gains = np.clip(gains, 0.85, 1.15)
    balanced = img * gains.reshape(1, 1, 3)
    # Renormalize mean luminance roughly
    before = luminance_rec709(bgr).mean()
    out = np.clip(balanced, 0, 255).astype(np.uint8)
    after = luminance_rec709(out).mean() + 1e-6
    scale = before / after
    out = np.clip(out.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return out


def enhance_lighting(
    bgr: np.ndarray,
    analysis: ImageAnalysis,
    *,
    exposure_stops: float,
    shadow_strength: float,
    local_amount: float,
    window_protection: bool = True,
    white_balance: bool = False,
) -> np.ndarray:
    """Full lighting pass on a BGR uint8 image; geometry unchanged."""
    work = bgr
    if white_balance:
        work = optional_white_balance(work, strength=0.30)

    lab, L0 = _lab_l(work)
    L = apply_exposure(L0, exposure_stops)
    L = soft_shadow_curve(L, shadow_strength)
    L = highlight_rolloff(L, L0, protect=window_protection)
    L = mild_local_contrast(L, local_amount)
    return _merge_lab_l(lab, L)
