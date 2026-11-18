#!/usr/bin/env python3
"""Emlak Fotoğraf Aydınlatma — giriş noktası.

Kullanım:
  python main.py              # Tkinter arayüzü
  python main.py --cli foto1.jpg foto2.jpg --out duzenlenmis_fotograflar
  python main.py --cli --mode orta --intensity 1.0 path/*.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without install: add src/ to path
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from emlak_aydinlatma.looks import LookProfile  # noqa: E402
from emlak_aydinlatma.pipeline import (  # noqa: E402
    EnhanceSettings,
    Mode,
    process_batch,
)
from emlak_aydinlatma.ui_app import run_app  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Emlak fotoğraf aydınlatma MVP")
    p.add_argument(
        "--cli",
        action="store_true",
        help="Arayüzsüz (headless) toplu işleme",
    )
    p.add_argument(
        "paths",
        nargs="*",
        help="Girdi görüntü dosyaları",
    )
    p.add_argument(
        "--out",
        type=str,
        default="duzenlenmis_fotograflar",
        help="Çıktı klasörü (varsayılan: duzenlenmis_fotograflar)",
    )
    p.add_argument(
        "--mode",
        choices=[m.value for m in Mode],
        default=Mode.OTOMATIK.value,
    )
    p.add_argument("--intensity", type=float, default=1.0)
    p.add_argument("--no-window-protect", action="store_true")
    p.add_argument("--wb", action="store_true", help="Hafif beyaz dengesi")
    p.add_argument(
        "--look",
        choices=[m.value for m in LookProfile],
        default=LookProfile.NONE.value,
        help="Look katmanı (varsayılan: none = yalnızca Doğal Aydınlatma)",
    )
    p.add_argument(
        "--look-strength",
        type=float,
        default=50.0,
        help="Look gücü 0–100 (varsayılan 50)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.cli:
        run_app()
        return 0

    if not args.paths:
        print("CLI için en az bir görüntü yolu verin.", file=sys.stderr)
        return 2

    settings = EnhanceSettings(
        mode=Mode(args.mode),
        intensity=args.intensity,
        window_protection=not args.no_window_protect,
        white_balance=args.wb,
        look=LookProfile(args.look),
        look_strength=args.look_strength,
    )
    out = Path(args.out)
    results = process_batch(args.paths, settings=settings, out_dir=out, save=True)
    for path, r in zip(args.paths, results):
        print(f"OK {path}")
        print(
            f"  mean_luma {r.analysis.mean_luminance:.3f} → "
            f"{r.quality.metrics.get('mean_luma_out', 0):.3f} | "
            f"exp={r.applied['exposure']:.2f} shadow={r.applied['shadow']:.2f}"
        )
        if r.quality.warnings:
            for w in r.quality.warnings:
                print(f"  UYARI: {w}")
    print(f"Kayıt klasörü: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
