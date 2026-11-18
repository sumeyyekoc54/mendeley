"""iPhone-style look profiles as a post-pass after Doğal Aydınlatma.

Tone/color *feel* only — not an Apple algorithm clone. Deterministic OpenCV/NumPy.
Strength 0–100 scales the full look recipe; 0 = lighting-only identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

import cv2
import numpy as np

from .analysis import ImageAnalysis, analyze_image, luminance_rec709
from .enhance import _lab_l, _merge_lab_l


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
    """Strength-scaled amounts for one look (internal units, already × strength)."""

    shadow_lift: float
    black_hold: float
    white_clean: float
    cast_fix: float
    vibrance: float
    micro_contrast: float
    noise_control: float
    midtone_contrast: float  # S-curve amount
    clarity: float  # mid-scale local contrast


@dataclass
class LookDebugMetrics:
    """Developer-facing metrics for the look post-pass."""

    look: str
    intensity: float  # 0..100
    exposure_delta_stops: float  # intentionally ~0 for HDR look
    shadow_lift: float
    black_hold: float
    white_clean: float
    cast_fix: float
    midtone_contrast: float
    micro_contrast: float
    clarity: float
    vibrance: float
    noise_control: float
    mean_abs_delta_vs_lit: float
    mean_luma_lit: float
    mean_luma_look: float
    mean_luma_delta: float
    effect_band: str  # low | medium | high
    notes: list[str] = field(default_factory=list)

    def as_lines(self) -> list[str]:
        lines = [
            f"Look debug: {self.look} @ intensity={self.intensity:.0f}",
            f"  effect_band={self.effect_band} | mean|Δ|vs lit={self.mean_abs_delta_vs_lit:.2f} "
            f"| luma {self.mean_luma_lit:.3f}→{self.mean_luma_look:.3f} (Δ{self.mean_luma_delta:+.3f})",
            f"  midtone={self.midtone_contrast:.3f} micro={self.micro_contrast:.3f} "
            f"clarity={self.clarity:.3f} white_clean={self.white_clean:.3f}",
            f"  shadow_lift={self.shadow_lift:.3f} black_hold={self.black_hold:.3f} "
            f"vibrance={self.vibrance:.3f} cast_fix={self.cast_fix:.3f}",
            f"  exposure_delta_stops≈{self.exposure_delta_stops:.3f} (look should not re-expose)",
        ]
        lines.extend(f"  • {n}" for n in self.notes)
        return lines


def effect_band_from_delta(mean_abs_delta: float) -> str:
    """Map mean |Δ| (0–255) to low/medium/high visibility band."""
    if mean_abs_delta < 4.0:
        return "low"
    if mean_abs_delta < 12.0:
        return "medium"
    return "high"


def _params_for(look: LookProfile, strength01: float, analysis: ImageAnalysis) -> LookParams:
    """Per-look base recipes × user strength (linear 0..1)."""
    s = float(np.clip(strength01, 0.0, 1.0))
    noise = analysis.noise_risk
    cast_r, cast_g, cast_b = analysis.channel_cast
    warm = float(np.clip((cast_r - cast_b) * 2.5, 0.0, 1.0))
    greenish = float(np.clip((cast_g - (cast_r + cast_b) * 0.5) * 3.0, 0.0, 1.0))
    cast_signal = max(warm * 0.85 + 0.15, greenish)  # keep a floor so cast fix can act

    # Recipes tuned so strength=50 is clearly visible vs lit-only (medium band),
    # without pushing global exposure (bathroom rule).
    if look == LookProfile.NATURAL:
        return LookParams(
            shadow_lift=0.22 * s,
            black_hold=0.18 * s,
            white_clean=0.35 * s,
            cast_fix=0.35 * s * cast_signal,
            vibrance=0.22 * s,
            micro_contrast=0.18 * s * (1.0 - 0.35 * noise),
            noise_control=0.20 * s * noise,
            midtone_contrast=0.28 * s,
            clarity=0.16 * s * (1.0 - 0.3 * noise),
        )
    if look == LookProfile.IPHONE_NATURAL_HDR:
        return LookParams(
            shadow_lift=0.32 * s,
            black_hold=0.32 * s,
            white_clean=0.70 * s,
            cast_fix=0.50 * s * cast_signal,
            vibrance=0.34 * s,
            micro_contrast=0.38 * s * (1.0 - 0.28 * noise),
            noise_control=0.22 * s * max(noise, 0.25),
            midtone_contrast=0.62 * s,
            clarity=0.38 * s * (1.0 - 0.22 * noise),
        )
    if look == LookProfile.IPHONE_VIBRANT:
        return LookParams(
            shadow_lift=0.24 * s,
            black_hold=0.22 * s,
            white_clean=0.42 * s,
            cast_fix=0.40 * s * cast_signal,
            vibrance=0.48 * s,
            micro_contrast=0.36 * s * (1.0 - 0.25 * noise),
            noise_control=0.18 * s * max(noise, 0.2),
            midtone_contrast=0.40 * s,
            clarity=0.28 * s * (1.0 - 0.2 * noise),
        )
    return LookParams(0, 0, 0, 0, 0, 0, 0, 0, 0)


def _midtone_s_curve(L: np.ndarray, amount: float) -> np.ndarray:
    """Contrast S-curve centered on midtones; preserves ends more than mids."""
    if amount <= 1e-4:
        return L
    x = np.clip(L / 100.0, 0.0, 1.0)
    # Stronger mid contrast: sigmoid-ish blend with identity
    # amount 0..~0.7 maps to visible pop without posterizing
    contrast = 1.0 + amount * 2.2
    centered = x - 0.5
    y = 0.5 + centered * contrast / (1.0 + (contrast - 1.0) * 2.0 * np.abs(centered))
    # Keep ends from collapsing via soft mix
    y = (1.0 - 0.15 * amount) * y + (0.15 * amount) * x
    return np.clip(y, 0.0, 1.0) * 100.0


def _shadow_lift_hold(L: np.ndarray, lift: float, hold: float) -> np.ndarray:
    """Soft shadow open + black-point hold (HDR pop without wash)."""
    x = np.clip(L / 100.0, 0.0, 1.0)
    if lift > 1e-4:
        # Lift deep/mid-shadows; fall off before midtones
        w = np.clip(1.0 - x / 0.45, 0.0, 1.0) ** 1.6
        x = x + lift * 0.22 * w * (1.0 - x)
    if hold > 1e-4:
        crush = hold * np.clip(1.0 - x / 0.10, 0.0, 1.0) ** 1.4
        x = x - crush * 0.12
    return np.clip(x, 0.0, 1.0) * 100.0


def _soft_highlight_protect(L: np.ndarray, ref_L: np.ndarray, amount: float = 0.55) -> np.ndarray:
    """Limit bright-region *lift* vs reference; do not crush below lit image."""
    x = np.clip(L / 100.0, 0.0, 1.0)
    ref = np.clip(ref_L / 100.0, 0.0, 1.0)
    bright = np.clip((ref - 0.68) / 0.28, 0.0, 1.0)
    k = max(11, (min(ref.shape[:2]) // 45) | 1)
    if k % 2 == 0:
        k += 1
    mask = cv2.GaussianBlur(bright.astype(np.float32), (k, k), k * 0.35)
    lift = np.maximum(x - ref, 0.0)
    x = ref + lift * (1.0 - amount * mask)
    # Never darker than the lit reference
    x = np.maximum(x, ref)
    return np.clip(x, 0.0, 1.0) * 100.0


def _clean_whites(L: np.ndarray, lab: np.ndarray, amount: float) -> tuple[np.ndarray, np.ndarray]:
    """Clean near-neutral bright walls/ceilings (luminance + mild desat)."""
    if amount <= 1e-4:
        return L, lab
    L01 = np.clip(L / 100.0, 0.0, 1.0)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    chroma = np.sqrt(a * a + b * b)
    neutral = np.clip(1.0 - chroma / 32.0, 0.0, 1.0)
    bright = np.clip((L01 - 0.48) / 0.40, 0.0, 1.0)
    mask = (neutral * bright).astype(np.float32)
    k = max(9, (min(mask.shape) // 45) | 1)
    if k % 2 == 0:
        k += 1
    mask = cv2.GaussianBlur(mask, (k, k), k * 0.28)
    # Stronger L lift toward clean white; desat a/b
    L_out = L01 + amount * 0.28 * mask * (1.0 - L01)
    lab_out = lab.copy()
    lab_out[:, :, 1] = lab[:, :, 1] + (128.0 - lab[:, :, 1]) * (amount * 0.55 * mask)
    lab_out[:, :, 2] = lab[:, :, 2] + (128.0 - lab[:, :, 2]) * (amount * 0.55 * mask)
    return np.clip(L_out, 0.0, 1.0) * 100.0, lab_out


def _micro_and_clarity(L: np.ndarray, micro: float, clarity: float) -> np.ndarray:
    """Two-scale local contrast on L; shadow-weighted to limit noise."""
    if micro <= 1e-4 and clarity <= 1e-4:
        return L
    x = (L / 100.0).astype(np.float32)
    shadow_w = np.clip(x / 0.15, 0.35, 1.0)  # still allow some pop in mid shadows
    if micro > 1e-4:
        blur = cv2.GaussianBlur(x, (0, 0), max(1.2, min(x.shape[:2]) / 220.0))
        x = x + (x - blur) * micro * shadow_w
    if clarity > 1e-4:
        # Mid-radius "clarity" without aggressive halo
        blur2 = cv2.GaussianBlur(x, (0, 0), max(4.0, min(x.shape[:2]) / 55.0))
        detail = x - blur2
        # Soften halo: attenuate detail where |detail| is large next to bright edges
        x = x + detail * clarity * 0.85 * shadow_w
    return np.clip(x, 0.0, 1.0) * 100.0


def _mild_cast_fix(bgr: np.ndarray, amount: float, analysis: ImageAnalysis) -> np.ndarray:
    """Yellow/green cast cleanup when present."""
    if amount <= 1e-4:
        return bgr
    cast_r, cast_g, cast_b = analysis.channel_cast
    img = bgr.astype(np.float32)
    gains = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    warm = cast_r - cast_b
    if warm > 0.02:
        gains[2] += amount * warm * 0.55
        gains[0] -= amount * warm * 0.28
    green = cast_g - (cast_r + cast_b) * 0.5
    if green > 0.015:
        gains[1] -= amount * green * 0.55
    gains = np.clip(gains, 0.88, 1.12)
    before = float(luminance_rec709(bgr).mean())
    out = np.clip(img * gains.reshape(1, 1, 3), 0, 255).astype(np.uint8)
    after = float(luminance_rec709(out).mean()) + 1e-6
    return np.clip(out.astype(np.float32) * (before / after), 0, 255).astype(np.uint8)


def _vibrance_protect_beige(bgr: np.ndarray, amount: float) -> np.ndarray:
    """Vibrance boost with protection for low-sat warm beige tiles."""
    if amount <= 1e-4:
        return bgr
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, _v = cv2.split(hsv)
    # Beige/warm low-sat: hue roughly 10–45 (OpenCV H 0–180)
    beige = ((h >= 8) & (h <= 45) & (s < 70)).astype(np.float32)
    k = max(5, (min(beige.shape) // 80) | 1)
    if k % 2 == 0:
        k += 1
    beige = cv2.GaussianBlur(beige, (k, k), 0)
    weight = (1.0 - s / 255.0) ** 1.15
    weight = weight * (1.0 - 0.75 * beige)  # protect tiles
    hsv[:, :, 1] = np.clip(s + amount * 70.0 * weight, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _shadow_noise_control(bgr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.03:
        return bgr
    luma = luminance_rec709(bgr)
    mask = np.clip((0.16 - luma) / 0.16, 0.0, 1.0).astype(np.float32)
    k = max(5, (min(bgr.shape[:2]) // 120) | 1)
    if k % 2 == 0:
        k += 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    soft = cv2.bilateralFilter(
        bgr, d=5, sigmaColor=20.0 + 35.0 * amount, sigmaSpace=20.0 + 35.0 * amount
    )
    m = (mask * amount * 0.85)[..., None]
    return np.clip(
        bgr.astype(np.float32) * (1.0 - m) + soft.astype(np.float32) * m, 0, 255
    ).astype(np.uint8)


def apply_look(
    lit_bgr: np.ndarray,
    look: LookProfile,
    strength: float = 50.0,
    *,
    analysis: ImageAnalysis | None = None,
    source_bgr: np.ndarray | None = None,
    return_metrics: bool = False,
) -> np.ndarray | tuple[np.ndarray, LookDebugMetrics]:
    """
    Apply look post-pass to an already-lit image.

    strength: 0..100 (UI). 0 or look=NONE → identity (lighting only).
    Does not re-apply global exposure — tonal rendering only.
    """
    if look == LookProfile.NONE or strength <= 0.5:
        metrics = LookDebugMetrics(
            look=look.value,
            intensity=float(strength),
            exposure_delta_stops=0.0,
            shadow_lift=0.0,
            black_hold=0.0,
            white_clean=0.0,
            cast_fix=0.0,
            midtone_contrast=0.0,
            micro_contrast=0.0,
            clarity=0.0,
            vibrance=0.0,
            noise_control=0.0,
            mean_abs_delta_vs_lit=0.0,
            mean_luma_lit=float(luminance_rec709(lit_bgr).mean()),
            mean_luma_look=float(luminance_rec709(lit_bgr).mean()),
            mean_luma_delta=0.0,
            effect_band="low",
            notes=["identity (look kapalı veya güç≈0)"],
        )
        return (lit_bgr, metrics) if return_metrics else lit_bgr

    strength01 = float(np.clip(strength, 0.0, 100.0) / 100.0)
    analysis = analysis or analyze_image(source_bgr if source_bgr is not None else lit_bgr)
    p = _params_for(look, strength01, analysis)

    work = lit_bgr
    if p.cast_fix > 0:
        work = _mild_cast_fix(work, p.cast_fix, analysis)

    lab, L0 = _lab_l(work)
    L = _shadow_lift_hold(L0, p.shadow_lift, p.black_hold)
    L = _midtone_s_curve(L, p.midtone_contrast)
    L = _soft_highlight_protect(L, L0, amount=0.65)
    L = _micro_and_clarity(L, p.micro_contrast, p.clarity)
    L, lab = _clean_whites(L, lab, p.white_clean)
    work = _merge_lab_l(lab, L)

    if p.vibrance > 0:
        work = _vibrance_protect_beige(work, p.vibrance)
    if p.noise_control > 0:
        work = _shadow_noise_control(work, p.noise_control)

    assert work.shape == lit_bgr.shape

    mean_abs = float(np.mean(np.abs(work.astype(np.float32) - lit_bgr.astype(np.float32))))
    luma_lit = float(luminance_rec709(lit_bgr).mean())
    luma_look = float(luminance_rec709(work).mean())
    band = effect_band_from_delta(mean_abs)
    notes: list[str] = []
    if band == "low" and strength01 >= 0.4:
        notes.append("Uyarı: güç≥40 ama etki hâlâ düşük — parametreleri kontrol edin.")
    if abs(luma_look - luma_lit) > 0.08:
        notes.append("Ortalama luminans kayması yüksek; look aşırı parlatıyor olabilir.")

    metrics = LookDebugMetrics(
        look=look.value,
        intensity=float(np.clip(strength, 0.0, 100.0)),
        exposure_delta_stops=0.0,
        shadow_lift=p.shadow_lift,
        black_hold=p.black_hold,
        white_clean=p.white_clean,
        cast_fix=p.cast_fix,
        midtone_contrast=p.midtone_contrast,
        micro_contrast=p.micro_contrast,
        clarity=p.clarity,
        vibrance=p.vibrance,
        noise_control=p.noise_control,
        mean_abs_delta_vs_lit=mean_abs,
        mean_luma_lit=luma_lit,
        mean_luma_look=luma_look,
        mean_luma_delta=luma_look - luma_lit,
        effect_band=band,
        notes=notes,
    )
    return (work, metrics) if return_metrics else work


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


def params_dict(look: LookProfile, strength: float, analysis: ImageAnalysis) -> dict:
    """Helper for debug UI / reports."""
    return asdict(_params_for(look, float(np.clip(strength, 0, 100)) / 100.0, analysis))
