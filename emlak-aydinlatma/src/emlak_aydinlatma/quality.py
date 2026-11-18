"""Post-enhance quality gates with measurable thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .analysis import luminance_rec709


@dataclass
class QualityReport:
    ok: bool
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def needs_manual_review(self) -> bool:
        return not self.ok or bool(self.warnings)


# Thresholds (MVP — conservative)
MAX_CLIPPED_HIGHLIGHT_INCREASE_PP = 2.5  # percentage points
MAX_MEAN_LUMA_DELTA = 0.28
MAX_MEAN_DELTA_E = 8.0  # chroma-only; L changes tracked via mean_luma_delta
MAX_SHADOW_NOISE_RATIO = 2.2


def _delta_e_approx(bgr_a: np.ndarray, bgr_b: np.ndarray) -> float:
    """Mean chromatic ΔE (a/b only) — luminance lift is checked separately."""
    lab_a = cv2.cvtColor(bgr_a, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(bgr_b, cv2.COLOR_BGR2LAB).astype(np.float32)
    # OpenCV a/b: 128 ≈ 0; difference already ≈ Lab a/b units
    da = lab_a[:, :, 1] - lab_b[:, :, 1]
    db = lab_a[:, :, 2] - lab_b[:, :, 2]
    de = np.sqrt(da ** 2 + db ** 2)
    return float(np.mean(de))


def _shadow_noise(luma: np.ndarray) -> float:
    small = cv2.resize(luma, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (0, 0), 1.2)
    hf = small - blur
    mask = (small < 0.15).astype(np.float32)
    denom = float(np.sum(mask)) + 1e-6
    return float(np.sqrt(np.sum((hf * mask) ** 2) / denom))


def check_quality(original_bgr: np.ndarray, enhanced_bgr: np.ndarray) -> QualityReport:
    """Compare original vs enhanced; flag unrecoverable / risky results."""
    warnings: list[str] = []
    metrics: dict[str, float] = {}

    if original_bgr.shape != enhanced_bgr.shape:
        return QualityReport(
            ok=False,
            warnings=["Çözünürlük veya kanal sayısı değişti — işlem iptal edilmeli."],
            metrics={},
        )

    luma0 = luminance_rec709(original_bgr)
    luma1 = luminance_rec709(enhanced_bgr)

    clip0 = float(np.mean(luma0 > 0.97) * 100.0)
    clip1 = float(np.mean(luma1 > 0.97) * 100.0)
    clip_delta = clip1 - clip0
    metrics["clipped_highlight_orig_pct"] = clip0
    metrics["clipped_highlight_out_pct"] = clip1
    metrics["clipped_highlight_delta_pp"] = clip_delta

    mean0 = float(np.mean(luma0))
    mean1 = float(np.mean(luma1))
    mean_delta = mean1 - mean0
    metrics["mean_luma_orig"] = mean0
    metrics["mean_luma_out"] = mean1
    metrics["mean_luma_delta"] = mean_delta

    de = _delta_e_approx(original_bgr, enhanced_bgr)
    metrics["mean_delta_e"] = de

    n0 = _shadow_noise(luma0)
    n1 = _shadow_noise(luma1)
    noise_ratio = n1 / (n0 + 1e-6)
    metrics["shadow_noise_ratio"] = noise_ratio

    ok = True

    if clip_delta > MAX_CLIPPED_HIGHLIGHT_INCREASE_PP:
        ok = False
        warnings.append(
            f"Parlak alanlarda ek yanma riski (+{clip_delta:.1f} puan). "
            "Pencere korumasını açın veya şiddeti düşürün."
        )
    elif clip1 > 12.0 and clip_delta > 0.5:
        warnings.append(
            "Kaynakta zaten yanmış highlight'lar var; detay geri getirilemez. "
            "Manuel kontrol önerilir."
        )

    if mean_delta > MAX_MEAN_LUMA_DELTA:
        ok = False
        warnings.append(
            f"Ortalama parlaklık artışı yüksek ({mean_delta:.2f}). "
            "Daha hafif mod deneyin."
        )

    if de > MAX_MEAN_DELTA_E:
        warnings.append(
            f"Renk sapması belirgin (ΔE≈{de:.1f}). "
            "Beyaz dengesi kapalıyken bile kontrol edin."
        )
        if de > MAX_MEAN_DELTA_E * 1.5:
            ok = False

    if noise_ratio > MAX_SHADOW_NOISE_RATIO:
        warnings.append(
            "Gölge bölgelerinde gürültü artışı yüksek. "
            "Hafif mod veya daha düşük şiddet önerilir."
        )
        if noise_ratio > MAX_SHADOW_NOISE_RATIO * 1.4:
            ok = False

    # Unrecoverable: large dark regions that remain near black after lift
    dark_left = float(np.mean((luma0 < 0.04) & (luma1 < 0.06)))
    metrics["unrecoverable_shadow_ratio"] = dark_left
    if dark_left > 0.08:
        warnings.append(
            "Bazı koyu bölgelerde detay kaynakta yok; aydınlatma uydurma yapmaz. "
            "Manuel kontrol önerilir."
        )

    if not warnings and ok:
        # Soft advisory when clip was already high in source
        if clip0 > 5.0:
            warnings.append(
                "Orijinalde yanmış highlight oranı yüksek; çıktı da sınırlı kalabilir."
            )

    return QualityReport(ok=ok, warnings=warnings, metrics=metrics)
