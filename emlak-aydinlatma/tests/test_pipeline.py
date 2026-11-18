"""Synthetic unit tests for analysis / enhance / quality / geometry."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emlak_aydinlatma.analysis import analyze_image, luminance_rec709  # noqa: E402
from emlak_aydinlatma.enhance import enhance_lighting  # noqa: E402
from emlak_aydinlatma.io_exif import output_path_for, save_image  # noqa: E402
from emlak_aydinlatma.looks import LookProfile, apply_look  # noqa: E402
from emlak_aydinlatma.pipeline import (  # noqa: E402
    EnhanceSettings,
    Mode,
    process_bgr,
    resolve_strengths,
)
from emlak_aydinlatma.quality import check_quality  # noqa: E402


FIXTURES = Path(__file__).parent / "fixtures"


def _dark_room_with_window(h: int = 240, w: int = 320) -> np.ndarray:
    """Synthetic: dark interior + bright window strip on the right."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Dark walls ~ RGB 25
    img[:, :] = (22, 24, 28)
    # Floor a bit brighter
    img[int(h * 0.65) :, :] = (35, 38, 42)
    # Bright window (near clip)
    img[:, int(w * 0.72) :] = (245, 248, 250)
    # Soft lamp hotspot
    cy, cx = int(h * 0.2), int(w * 0.35)
    for y in range(h):
        for x in range(max(0, cx - 25), min(w, cx + 25)):
            d = ((y - cy) ** 2 + (x - cx) ** 2) ** 0.5
            if d < 25:
                v = int(180 * (1 - d / 25))
                img[y, x] = np.clip(img[y, x].astype(np.int16) + v, 0, 255).astype(
                    np.uint8
                )
    return img


def _mid_bright(h: int = 128, w: int = 128) -> np.ndarray:
    img = np.full((h, w, 3), 140, dtype=np.uint8)
    return img


@pytest.fixture(scope="session", autouse=True)
def write_fixtures() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    dark = _dark_room_with_window()
    mid = _mid_bright()
    cv2.imwrite(str(FIXTURES / "dark_room_window.png"), dark)
    cv2.imwrite(str(FIXTURES / "mid_bright.png"), mid)


def test_geometry_preserved() -> None:
    img = _dark_room_with_window()
    r = process_bgr(img, EnhanceSettings(mode=Mode.OTOMATIK))
    assert r.enhanced_bgr.shape == img.shape
    assert r.enhanced_bgr.dtype == np.uint8


def test_dark_room_gets_brighter() -> None:
    img = _dark_room_with_window()
    r = process_bgr(img, EnhanceSettings(mode=Mode.ORTA, intensity=1.0))
    luma0 = luminance_rec709(img)
    luma1 = luminance_rec709(r.enhanced_bgr)
    # Shadow / wall region (left 70%) must lift meaningfully
    shadow0 = float(np.mean(luma0[:, : int(img.shape[1] * 0.7)]))
    shadow1 = float(np.mean(luma1[:, : int(img.shape[1] * 0.7)]))
    assert shadow1 > shadow0 * 1.4
    # Overall mean should not collapse
    assert float(np.mean(luma1)) >= float(np.mean(luma0)) * 0.98


def test_window_protection_limits_clip_increase() -> None:
    img = _dark_room_with_window()
    protected = process_bgr(
        img, EnhanceSettings(mode=Mode.GUCLU, intensity=1.2, window_protection=True)
    )
    unprotected = process_bgr(
        img, EnhanceSettings(mode=Mode.GUCLU, intensity=1.2, window_protection=False)
    )
    clip_p = protected.quality.metrics["clipped_highlight_delta_pp"]
    clip_u = unprotected.quality.metrics["clipped_highlight_delta_pp"]
    # Protection should not increase clip more than unprotected
    assert clip_p <= clip_u + 0.5


def test_modes_scale_strength() -> None:
    img = _dark_room_with_window()
    a = analyze_image(img)
    hafif = resolve_strengths(a, EnhanceSettings(mode=Mode.HAFIF, intensity=1.0))
    guclu = resolve_strengths(a, EnhanceSettings(mode=Mode.GUCLU, intensity=1.0))
    assert guclu["exposure"] >= hafif["exposure"]
    assert guclu["shadow"] >= hafif["shadow"]


def test_quality_flags_resolution_change() -> None:
    a = _mid_bright()
    b = cv2.resize(a, (64, 64))
    q = check_quality(a, b)
    assert q.ok is False


def test_output_naming() -> None:
    p = output_path_for("oda.jpg", "/tmp/out")
    assert p.name == "oda_AYDINLIK.jpg"


def test_save_same_pixel_size(tmp_path: Path) -> None:
    img = _dark_room_with_window()
    r = process_bgr(img)
    out = tmp_path / "x_AYDINLIK.jpg"
    save_image(r.enhanced_bgr, out, quality=95)
    loaded = cv2.imread(str(out))
    assert loaded is not None
    assert loaded.shape == img.shape


def test_enhance_lighting_no_nan() -> None:
    img = _dark_room_with_window()
    a = analyze_image(img)
    out = enhance_lighting(
        img,
        a,
        exposure_stops=0.8,
        shadow_strength=0.4,
        local_amount=0.1,
        window_protection=True,
    )
    assert np.isfinite(out).all()


def test_default_look_matches_natural_lighting_only() -> None:
    """Look=NONE must leave Doğal Aydınlatma path unchanged."""
    img = _dark_room_with_window()
    plain = process_bgr(img, EnhanceSettings(mode=Mode.OTOMATIK, look=LookProfile.NONE))
    assert plain.lit_bgr is not None
    assert np.array_equal(plain.enhanced_bgr, plain.lit_bgr)


def test_look_postpass_preserves_geometry() -> None:
    img = _dark_room_with_window()
    r = process_bgr(
        img,
        EnhanceSettings(
            mode=Mode.OTOMATIK,
            look=LookProfile.IPHONE_NATURAL_HDR,
            look_strength=50.0,
        ),
    )
    assert r.enhanced_bgr.shape == img.shape
    assert np.isfinite(r.enhanced_bgr).all()


def test_look_strength_zero_is_identity_on_lit() -> None:
    img = _dark_room_with_window()
    lit = process_bgr(img, EnhanceSettings(look=LookProfile.NONE))
    assert lit.lit_bgr is not None
    out = apply_look(lit.lit_bgr, LookProfile.IPHONE_VIBRANT, strength=0.0)
    assert np.array_equal(out, lit.lit_bgr)
