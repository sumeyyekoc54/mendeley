"""iPhone-style look profiles as a post-pass after Doğal Aydınlatma.

Tone/color *feel* only — not an Apple algorithm clone. Deterministic OpenCV/NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from .analysis import ImageAnalysis, analyze_image, luminance_rec709
from .enhance import (
    _lab_l,
    _merge_lab_l,
    highlight_rolloff,
    mild_local_contrast,
    soft_shadow_curve,
)


class LookProfile(str, Enum):
    """Look layer on top of Doğal Aydınlatma."""

    NONE = "none"  # yalnızca doğal aydınlatma (varsayılan)
    NATURAL = "natural"  # hafif polish
    IPHONE_NATURAL_HDR = "iphone_natural_hdr"
    IPHONE_VIBRANT = "iphone_vibrant"


LOOK_LABELS_TR: dict[LookProfile, str] = {
    LookProfile.NONE: "Doğal Aydınlatma (look yok)",
    LookProfile.NATURAL: "Natural",
    LookProfile.IPHONE_NATURAL_HDR: "iPhone Natural HDR",
    LookProfile.IPHONE_VIBRANT: "iPhone Vibrant",
}


@dataclass(frozen=True)
class LookParams:
    """Strength-scaled amounts for one look (0..1 internal)."""

    shadow_lift: float
    black_hold: float
    white_clean: float
    cast_fix: float
    vibrance: float
    micro_contrast: float
    noise_control: float
    midtone_polish: float


def _params_for(look: LookProfile, strength01: float, analysis: ImageAnalysis) -> LookParams:
    """Per-look base recipes × user strength; tempered by analysis."""
    s = float(np.clip(strength01, 0.0, 1.0))
    noise = analysis.noise_risk
    cast_r, cast_g, cast_b = analysis.channel_cast
    # Yellow/warm: R high, B low; green: G high
    warm = float(np.clip((cast_r - cast_b) * 2.0, 0.0, 1.0))
    greenish = float(np.clip((cast_g - (cast_r + cast_b) * 0.5) * 3.0, 0.0, 1.0))
    cast_signal = max(warm, greenish)

    if look == LookProfile.NATURAL:
        return LookParams(
            shadow_lift=0.08 * s * (0.7 + analysis.shadow_ratio),
            black_hold=0.04 * s,
            white_clean=0.10 * s,
            cast_fix=0.12 * s * cast_signal,
            vibrance=0.06 * s,
            micro_contrast=0.05 * s * (1.0 - 0.5 * noise),
            noise_control=0.15 * s * noise,
            midtone_polish=0.04 * s,
        )
    if look == LookProfile.IPHONE_NATURAL_HDR:
        return LookParams(
            shadow_lift=0.14 * s * (0.6 + analysis.shadow_ratio),
            black_hold=0.07 * s,
            white_clean=0.22 * s,
            cast_fix=0.22 * s * cast_signal,
            vibrance=0.08 * s,
            micro_contrast=0.09 * s * (1.0 - 0.45 * noise),
            noise_control=0.25 * s * noise,
            midtone_polish=0.07 * s,
        )
    if look == LookProfile.IPHONE_VIBRANT:
        return LookParams(
            shadow_lift=0.12 * s * (0.55 + analysis.shadow_ratio),
            black_hold=0.06 * s,
            white_clean=0.16 * s,
            cast_fix=0.18 * s * cast_signal,
            vibrance=0.18 * s,
            micro_contrast=0.12 * s * (1.0 - 0.4 * noise),
            noise_control=0.20 * s * noise,
            midtone_polish=0.06 * s,
        )
    return LookParams(0, 0, 0, 0, 0, 0, 0, 0)


def _black_point_hold(L: np.ndarray, amount: float) -> np.ndarray:
    """Pull deepest blacks slightly down after lift to avoid washed look."""
    if amount <= 1e-4:
        return L
    x = np.clip(L / 100.0, 0.0, 1.0)
    # Soft crush only near floor
    crush = amount * np.clip(1.0 - x / 0.12, 0.0, 1.0) ** 1.5
    return np.clip(x - crush * 0.08, 0.0, 1.0) * 100.0


def _clean_whites(L: np.ndarray, lab: np.ndarray, amount: float) -> tuple[np.ndarray, np.ndarray]:
    """Brighten near-neutral high-L regions toward cleaner whites (luminance-biased)."""
    if amount <= 1e-4:
        return L, lab
    L01 = np.clip(L / 100.0, 0.0, 1.0)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    chroma = np.sqrt(a * a + b * b)
    # Neutral-ish + bright → wall/ceiling candidates
    neutral = np.clip(1.0 - chroma / 28.0, 0.0, 1.0)
    bright = np.clip((L01 - 0.55) / 0.35, 0.0, 1.0)
    mask = (neutral * bright).astype(np.float32)
    k = max(9, (min(mask.shape) // 50) | 1)
    if k % 2 == 0:
        k += 1
    mask = cv2.GaussianBlur(mask, (k, k), k * 0.3)
    # Lift L gently; desaturate a/b slightly toward 128
    L_out = L01 + amount * 0.12 * mask * (1.0 - L01)
    lab_out = lab.copy()
    lab_out[:, :, 1] = lab[:, :, 1] + (128.0 - lab[:, :, 1]) * (amount * 0.35 * mask)
    lab_out[:, :, 2] = lab[:, :, 2] + (128.0 - lab[:, :, 2]) * (amount * 0.35 * mask)
    return np.clip(L_out, 0.0, 1.0) * 100.0, lab_out


def _mild_cast_fix(bgr: np.ndarray, amount: float, analysis: ImageAnalysis) -> np.ndarray:
    """Tiny yellow/green cast nudge when cast is detectable."""
    if amount <= 1e-4:
        return bgr
    cast_r, cast_g, cast_b = analysis.channel_cast
    img = bgr.astype(np.float32)
    # Target slight neutralization of R/B imbalance and G spike
    gains = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    # Warm cast → reduce R a touch, lift B
    warm = cast_r - cast_b
    if warm > 0.03:
        gains[2] += amount * warm * 0.35  # B
        gains[0] -= amount * warm * 0.20  # R
    green = cast_g - (cast_r + cast_b) * 0.5
    if green > 0.02:
        gains[1] -= amount * green * 0.40
    gains = np.clip(gains, 0.92, 1.08)
    before = float(luminance_rec709(bgr).mean())
    out = np.clip(img * gains.reshape(1, 1, 3), 0, 255).astype(np.uint8)
    after = float(luminance_rec709(out).mean()) + 1e-6
    out = np.clip(out.astype(np.float32) * (before / after), 0, 255).astype(np.uint8)
    return out


def _vibrance(bgr: np.ndarray, amount: float) -> np.ndarray:
    """Vibrance-like: boost low-sat more than already-saturated pixels."""
    if amount <= 1e-4:
        return bgr
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[:, :, 1]
    # Less boost where already saturated
    weight = (1.0 - s / 255.0) ** 1.2
    hsv[:, :, 1] = np.clip(s + amount * 45.0 * weight, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _shadow_noise_control(bgr: np.ndarray, amount: float) -> np.ndarray:
    """Mild bilateral in shadows only — keep texture elsewhere."""
    if amount <= 0.02:
        return bgr
    luma = luminance_rec709(bgr)
    mask = np.clip((0.18 - luma) / 0.18, 0.0, 1.0).astype(np.float32)
    k = max(5, (min(bgr.shape[:2]) // 120) | 1)
    if k % 2 == 0:
        k += 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    # Light bilateral
    d = 5
    sigma = 25.0 + 40.0 * amount
    soft = cv2.bilateralFilter(bgr, d=d, sigmaColor=sigma, sigmaSpace=sigma)
    m = (mask * amount)[..., None]
    out = bgr.astype(np.float32) * (1.0 - m) + soft.astype(np.float32) * m
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_look(
    lit_bgr: np.ndarray,
    look: LookProfile,
    strength: float = 50.0,
    *,
    analysis: ImageAnalysis | None = None,
    source_bgr: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply look post-pass to an already-lit image.

    strength: 0..100 (UI). look=NONE → identity.
    source_bgr: original pre-lighting (for highlight protection reference); optional.
    """
    if look == LookProfile.NONE or strength <= 0.5:
        return lit_bgr

    strength01 = float(np.clip(strength, 0.0, 100.0) / 100.0)
    analysis = analysis or analyze_image(source_bgr if source_bgr is not None else lit_bgr)
    p = _params_for(look, strength01, analysis)
    if max(p.__dict__.values()) < 1e-5:
        return lit_bgr

    work = lit_bgr
    if p.cast_fix > 0:
        work = _mild_cast_fix(work, p.cast_fix, analysis)

    lab, L0 = _lab_l(work)
    L = soft_shadow_curve(L0, p.shadow_lift)
    L = _black_point_hold(L, p.black_hold)
    # Protect highlights vs lit image (and source if given)
    ref_L = L0
    if source_bgr is not None:
        _, src_L = _lab_l(source_bgr)
        # Blend: don't exceed lit path much in windows — use min of refs for protect mask base
        ref_L = np.maximum(src_L, L0 * 0.85)
    L = highlight_rolloff(L, ref_L, protect=True, protect_strength=0.80)
    L = mild_local_contrast(L, p.micro_contrast)
    # Midtone polish: gentle S toward mid
    if p.midtone_polish > 0:
        x = np.clip(L / 100.0, 0.0, 1.0)
        # Soft contrast around mid without crushing blacks/whites
        mid = 0.5
        x = x + p.midtone_polish * (x - mid) * (1.0 - 2.0 * np.abs(x - mid))
        L = np.clip(x, 0.0, 1.0) * 100.0
    L, lab = _clean_whites(L, lab, p.white_clean)
    work = _merge_lab_l(lab, L)

    if p.vibrance > 0:
        work = _vibrance(work, p.vibrance)
    if p.noise_control > 0:
        work = _shadow_noise_control(work, p.noise_control)

    assert work.shape == lit_bgr.shape
    return work


def apply_all_looks(
    lit_bgr: np.ndarray,
    strength: float = 50.0,
    *,
    analysis: ImageAnalysis | None = None,
    source_bgr: np.ndarray | None = None,
) -> dict[LookProfile, np.ndarray]:
    """Produce Natural / iPhone Natural HDR / iPhone Vibrant from the same lit base."""
    out: dict[LookProfile, np.ndarray] = {}
    for look in (
        LookProfile.NATURAL,
        LookProfile.IPHONE_NATURAL_HDR,
        LookProfile.IPHONE_VIBRANT,
    ):
        out[look] = apply_look(
            lit_bgr, look, strength, analysis=analysis, source_bgr=source_bgr
        )
    return out
