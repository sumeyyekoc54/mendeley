"""Per-image luminance / exposure analysis for real-estate photos."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageAnalysis:
    """Measurable scene stats used to pick exposure / shadow strength."""

    mean_luminance: float
    median_luminance: float
    shadow_ratio: float
    midtone_ratio: float
    highlight_ratio: float
    clipped_highlight_pct: float
    contrast: float
    channel_cast: tuple[float, float, float]
    noise_risk: float
    suggested_exposure: float
    suggested_shadow: float
    suggested_local: float


def _srgb_to_linear(channel: np.ndarray) -> np.ndarray:
    """Approximate sRGB → linear (channel in 0..1)."""
    c = np.clip(channel, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def luminance_rec709(bgr: np.ndarray) -> np.ndarray:
    """Rec.709 luma from BGR uint8, returned as float32 0..1 linear-ish."""
    bgr_f = bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(bgr_f)
    r_lin = _srgb_to_linear(r)
    g_lin = _srgb_to_linear(g)
    b_lin = _srgb_to_linear(b)
    return (0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin).astype(np.float32)


def analyze_image(bgr: np.ndarray) -> ImageAnalysis:
    """Compute histogram ratios, cast, noise risk, and suggested gains."""
    if bgr is None or bgr.size == 0:
        raise ValueError("Boş görüntü")

    luma = luminance_rec709(bgr)
    mean_l = float(np.mean(luma))
    median_l = float(np.median(luma))

    shadow_ratio = float(np.mean(luma < 0.12))
    midtone_ratio = float(np.mean((luma >= 0.12) & (luma <= 0.70)))
    highlight_ratio = float(np.mean(luma > 0.70))
    clipped_highlight_pct = float(np.mean(luma > 0.97) * 100.0)

    p10, p90 = np.percentile(luma, [10, 90])
    contrast = float(p90 - p10)

    b, g, r = cv2.split(bgr.astype(np.float32))
    means = np.array([np.mean(r), np.mean(g), np.mean(b)], dtype=np.float64)
    gray_mean = float(np.mean(means)) + 1e-6
    cast = tuple(float(x / gray_mean) for x in means)  # type: ignore[assignment]

    # Noise risk: high-frequency energy in shadow regions (downscaled)
    small = cv2.resize(luma, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (0, 0), 1.2)
    hf = small - blur
    shadow_mask = (small < 0.15).astype(np.float32)
    shadow_pixels = float(np.sum(shadow_mask)) + 1e-6
    noise_energy = float(np.sqrt(np.sum((hf * shadow_mask) ** 2) / shadow_pixels))
    noise_risk = float(np.clip(noise_energy / 0.04, 0.0, 1.0))

    # Prefer median / shadow-aware target so bright windows don't hide dark rooms
    shadow_mean = float(np.mean(luma[luma < 0.20])) if np.any(luma < 0.20) else mean_l
    # Blend mean with median and shadow mean for exposure decision
    exposure_anchor = float(0.35 * mean_l + 0.40 * median_l + 0.25 * shadow_mean)
    target = 0.34
    raw_exposure = np.log2((target + 1e-4) / (exposure_anchor + 1e-4))
    # Cap exposure aggressively for MVP safety
    suggested_exposure = float(np.clip(raw_exposure, 0.0, 1.25))

    # More shadows / low floor → more shadow lift, tempered by noise
    # Also lift when contrast is high (bright window + dark floor) even if mean looks OK
    contrast_boost = float(np.clip((contrast - 0.35) * 0.5, 0.0, 0.25))
    suggested_shadow = float(
        np.clip(
            0.20 + shadow_ratio * 0.55 + contrast_boost - noise_risk * 0.15,
            0.10,
            0.72,
        )
    )
    suggested_local = float(
        np.clip(0.08 * (1.0 - noise_risk) * (0.5 + shadow_ratio), 0.0, 0.18)
    )

    # If already bright overall and few shadows, dial back
    if mean_l > 0.45 and shadow_ratio < 0.25:
        suggested_exposure *= 0.35
        suggested_shadow *= 0.5
    if clipped_highlight_pct > 8.0:
        suggested_exposure = min(suggested_exposure, 0.55)

    return ImageAnalysis(
        mean_luminance=mean_l,
        median_luminance=median_l,
        shadow_ratio=shadow_ratio,
        midtone_ratio=midtone_ratio,
        highlight_ratio=highlight_ratio,
        clipped_highlight_pct=clipped_highlight_pct,
        contrast=contrast,
        channel_cast=cast,  # type: ignore[arg-type]
        noise_risk=noise_risk,
        suggested_exposure=suggested_exposure,
        suggested_shadow=suggested_shadow,
        suggested_local=suggested_local,
    )
