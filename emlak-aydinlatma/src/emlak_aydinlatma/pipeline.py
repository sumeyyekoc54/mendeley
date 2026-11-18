"""End-to-end enhancement pipeline and mode presets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import ImageAnalysis, analyze_image
from .enhance import enhance_lighting
from .io_exif import load_image, output_path_for, save_image
from .looks import LookDebugMetrics, LookProfile, apply_look
from .quality import QualityReport, check_quality


class Mode(str, Enum):
    OTOMATIK = "otomatik"
    HAFIF = "hafif"
    ORTA = "orta"
    GUCLU = "guclu"


# Multipliers on analysis-suggested values
_MODE_SCALE: dict[Mode, dict[str, float]] = {
    Mode.HAFIF: {"exposure": 0.45, "shadow": 0.50, "local": 0.0},
    Mode.ORTA: {"exposure": 0.75, "shadow": 0.85, "local": 0.5},
    Mode.GUCLU: {"exposure": 1.05, "shadow": 1.15, "local": 1.0},
    Mode.OTOMATIK: {"exposure": 0.85, "shadow": 0.90, "local": 0.35},
}


@dataclass
class EnhanceSettings:
    mode: Mode = Mode.OTOMATIK
    intensity: float = 1.0  # manual slider 0.3..1.5
    window_protection: bool = True
    white_balance: bool = False
    apply_local: bool | None = None  # None → mode decides
    # Look layer (post-pass). Default NONE keeps Doğal Aydınlatma unchanged.
    look: LookProfile = LookProfile.NONE
    look_strength: float = 50.0  # 0..100


@dataclass
class EnhanceResult:
    original_bgr: np.ndarray
    enhanced_bgr: np.ndarray
    analysis: ImageAnalysis
    quality: QualityReport
    settings: EnhanceSettings
    applied: dict[str, float]
    meta: dict[str, Any]
    lit_bgr: np.ndarray | None = None  # Doğal aydınlatma çıktısı (look öncesi)
    look_metrics: LookDebugMetrics | None = None


def resolve_strengths(
    analysis: ImageAnalysis,
    settings: EnhanceSettings,
) -> dict[str, float]:
    """Map mode + intensity + analysis → concrete exposure/shadow/local."""
    scale = _MODE_SCALE[settings.mode]
    intensity = float(np.clip(settings.intensity, 0.3, 1.5))

    exposure = analysis.suggested_exposure * scale["exposure"] * intensity
    shadow = analysis.suggested_shadow * scale["shadow"] * intensity
    local = analysis.suggested_local * scale["local"] * intensity

    if settings.apply_local is False:
        local = 0.0
    elif settings.apply_local is True and local < 0.05:
        local = 0.08 * intensity

    # Hard caps for natural look
    exposure = float(np.clip(exposure, 0.0, 1.35))
    shadow = float(np.clip(shadow, 0.0, 0.75))
    local = float(np.clip(local, 0.0, 0.20))

    # Auto: reduce if noise high (keep a usable shadow floor)
    if settings.mode == Mode.OTOMATIK and analysis.noise_risk > 0.55:
        exposure *= 0.80
        shadow = max(shadow * 0.75, 0.12)
        local *= 0.45

    return {"exposure": exposure, "shadow": shadow, "local": local}


def process_bgr(
    bgr: np.ndarray,
    settings: EnhanceSettings | None = None,
    meta: dict[str, Any] | None = None,
) -> EnhanceResult:
    """Analyze + Doğal Aydınlatma, then optional look post-pass."""
    settings = settings or EnhanceSettings()
    analysis = analyze_image(bgr)
    strengths = resolve_strengths(analysis, settings)
    lit = enhance_lighting(
        bgr,
        analysis,
        exposure_stops=strengths["exposure"],
        shadow_strength=strengths["shadow"],
        local_amount=strengths["local"],
        window_protection=settings.window_protection,
        white_balance=settings.white_balance,
    )
    assert lit.shape == bgr.shape

    enhanced, look_metrics = apply_look(
        lit,
        settings.look,
        settings.look_strength,
        analysis=analysis,
        source_bgr=bgr,
        return_metrics=True,
    )
    assert enhanced.shape == bgr.shape

    applied = dict(strengths)
    applied["look_strength"] = float(settings.look_strength)
    applied["look_mean_abs_delta"] = float(look_metrics.mean_abs_delta_vs_lit)
    quality = check_quality(bgr, enhanced)
    return EnhanceResult(
        original_bgr=bgr,
        enhanced_bgr=enhanced,
        analysis=analysis,
        quality=quality,
        settings=settings,
        applied=applied,
        meta=meta or {},
        lit_bgr=lit,
        look_metrics=look_metrics,
    )


def process_file(
    path: str | Path,
    settings: EnhanceSettings | None = None,
) -> EnhanceResult:
    bgr, meta = load_image(path)
    return process_bgr(bgr, settings=settings, meta=meta)


def save_result(
    result: EnhanceResult,
    out_dir: str | Path,
    source_path: str | Path | None = None,
) -> Path:
    """User-triggered save only — writes `_AYDINLIK` under out_dir."""
    src = source_path or result.meta.get("path")
    if not src:
        raise ValueError("Kaynak dosya yolu yok")
    out = output_path_for(src, out_dir)
    exif = result.meta.get("exif")
    return save_image(result.enhanced_bgr, out, quality=95, exif=exif)


def process_batch(
    paths: list[str | Path],
    settings: EnhanceSettings | None = None,
    out_dir: str | Path | None = None,
    save: bool = False,
) -> list[EnhanceResult]:
    """Process multiple files independently (per-image analysis)."""
    results: list[EnhanceResult] = []
    for p in paths:
        r = process_file(p, settings=settings)
        results.append(r)
        if save and out_dir is not None:
            save_result(r, out_dir, source_path=p)
    return results
