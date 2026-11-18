#!/usr/bin/env python3
"""PyInstaller öncesi import / freeze kontrolü.

Windows'ta build_windows.ps1 bunu çağırır.
Linux bulut VM'de tkinter olmayabilir; çekirdek paketler yine doğrulanır.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CORE = [
    "numpy",
    "cv2",
    "PIL",
    "PIL.Image",
    "PIL.ImageOps",
    "PIL.ImageDraw",
    "emlak_aydinlatma",
    "emlak_aydinlatma.analysis",
    "emlak_aydinlatma.enhance",
    "emlak_aydinlatma.quality",
    "emlak_aydinlatma.io_exif",
    "emlak_aydinlatma.pipeline",
    "emlak_aydinlatma.looks",
]

UI = [
    "tkinter",
    "tkinter.ttk",
    "tkinter.filedialog",
    "tkinter.messagebox",
    "PIL.ImageTk",
    "emlak_aydinlatma.ui_app",
]


def _check(mods: list[str]) -> list[str]:
    failed: list[str] = []
    for name in mods:
        try:
            importlib.import_module(name)
            print(f"  OK  {name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {name}: {exc}")
            failed.append(name)
    return failed


def main() -> int:
    print("Core imports:")
    failed = _check(CORE)
    print("UI imports (Windows exe için gerekli):")
    ui_failed = _check(UI)

    if failed:
        print("ÇEKİRDEK IMPORT HATASI:", ", ".join(failed), file=sys.stderr)
        return 1

    if ui_failed:
        # Linux CI / headless VM: uyarı; Windows build script bunu hata saymalı
        if sys.platform.startswith("win"):
            print("UI IMPORT HATASI (Windows):", ", ".join(ui_failed), file=sys.stderr)
            return 2
        print(
            "Uyarı: UI importları bu ortamda yok (muhtemelen Linux/headless). "
            "Windows derlemesinde Tkinter zorunlu.",
            file=sys.stderr,
        )
    else:
        print("Tüm importlar tamam.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
