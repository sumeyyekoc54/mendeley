#!/usr/bin/env python3
"""Process sample photos and write before/after/side-by-side into media/mvp-results."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emlak_aydinlatma.pipeline import EnhanceSettings, Mode, process_file  # noqa: E402
from emlak_aydinlatma.io_exif import save_image  # noqa: E402


def side_by_side(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    # Thin separator
    sep = np.full((a.shape[0], 4, 3), 40, dtype=np.uint8)
    return np.concatenate([a, sep, b], axis=1)


def main() -> int:
    sample_dir = Path(
        "/cursor/stores/bc-61f26783-e4ef-4d62-9f36-cee9386a2393/media/sample-photos"
    )
    out_dir = Path(
        "/cursor/stores/bc-61f26783-e4ef-4d62-9f36-cee9386a2393/media/mvp-results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = EnhanceSettings(
        mode=Mode.OTOMATIK,
        intensity=1.0,
        window_protection=True,
        white_balance=False,
    )

    paths = sorted(sample_dir.glob("*.jpg"))
    if not paths:
        print("No sample photos found", file=sys.stderr)
        return 1

    short_names = {
        "4c0ee860": "ic-giris",
        "877bec3f": "dis-avlu",
        "e2940f06": "merdiven-koridor",
    }

    for p in paths:
        key = p.stem[:8]
        label = short_names.get(key, key)
        print(f"Processing {p.name} → {label}")
        r = process_file(p, settings=settings)
        before = out_dir / f"{label}_01_before.jpg"
        after = out_dir / f"{label}_02_after.jpg"
        sbs = out_dir / f"{label}_03_side-by-side.jpg"
        save_image(r.original_bgr, before, quality=92)
        save_image(r.enhanced_bgr, after, quality=92)
        save_image(side_by_side(r.original_bgr, r.enhanced_bgr), sbs, quality=90)
        print(
            f"  mean {r.quality.metrics.get('mean_luma_orig', 0):.3f} → "
            f"{r.quality.metrics.get('mean_luma_out', 0):.3f} | "
            f"warnings={len(r.quality.warnings)}"
        )
        for w in r.quality.warnings:
            print(f"  UYARI: {w}")

    print("Wrote:", out_dir)
    for f in sorted(out_dir.iterdir()):
        print(" ", f.name, f.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
