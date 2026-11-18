#!/usr/bin/env python3
"""Write Natural | iPhone Natural HDR | iPhone Vibrant triples for sample photos."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emlak_aydinlatma.io_exif import save_image  # noqa: E402
from emlak_aydinlatma.looks import LookProfile, apply_all_looks  # noqa: E402
from emlak_aydinlatma.pipeline import EnhanceSettings, Mode, process_file  # noqa: E402

STORE = Path("/cursor/stores/bc-61f26783-e4ef-4d62-9f36-cee9386a2393/media")
OUT = STORE / "iphone-look-compare"

LABELS = {
    "4c0ee860": "ic-giris",
    "877bec3f": "dis-avlu",
    "e2940f06": "merdiven-koridor",
    "35d73b0d": "bos-oda",
    "b44f8318": "balkon",
    "d224a1c3": "gardirob-koridor",
    "1581e7c8": "mutfak",
}

LOOK_FILE = {
    LookProfile.NATURAL: "natural",
    LookProfile.IPHONE_NATURAL_HDR: "iphone-natural-hdr",
    LookProfile.IPHONE_VIBRANT: "iphone-vibrant",
}


def _hconcat(images: list[np.ndarray], gap: int = 4) -> np.ndarray:
    h = min(im.shape[0] for im in images)
    resized = []
    for im in images:
        if im.shape[0] != h:
            scale = h / im.shape[0]
            im = cv2.resize(im, (int(im.shape[1] * scale), h))
        resized.append(im)
    sep = np.full((h, gap, 3), 40, dtype=np.uint8)
    out = resized[0]
    for im in resized[1:]:
        out = np.concatenate([out, sep, im], axis=1)
    return out


def _preview_with_labels(panels: list[tuple[str, np.ndarray]], max_w: int = 1600) -> Image.Image:
    pil_panels = []
    for title, bgr in panels:
        rgb = bgr[:, :, ::-1]
        im = Image.fromarray(rgb)
        bar = 24
        canvas = Image.new("RGB", (im.width, im.height + bar), (28, 28, 28))
        canvas.paste(im, (0, bar))
        ImageDraw.Draw(canvas).text((8, 5), title, fill=(240, 240, 240))
        pil_panels.append(canvas)
    h = min(p.height for p in pil_panels)
    norm = []
    for p in pil_panels:
        if p.height != h:
            p = p.resize((int(p.width * h / p.height), h), Image.Resampling.LANCZOS)
        norm.append(p)
    gap = 4
    w = sum(p.width for p in norm) + gap * (len(norm) - 1)
    row = Image.new("RGB", (w, h), (20, 20, 20))
    x = 0
    for i, p in enumerate(norm):
        row.paste(p, (x, 0))
        x += p.width + (gap if i < len(norm) - 1 else 0)
    row.thumbnail((max_w, 900), Image.Resampling.LANCZOS)
    return row


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for folder in (STORE / "sample-photos", STORE / "sample-photos-batch2"):
        paths.extend(sorted(folder.glob("*.jpg")))
    if not paths:
        print("No samples", file=sys.stderr)
        return 1

    settings = EnhanceSettings(
        mode=Mode.OTOMATIK,
        intensity=1.0,
        window_protection=True,
        look=LookProfile.NONE,  # get lit base only
        look_strength=50.0,
    )
    strength = 50.0

    for p in paths:
        label = LABELS.get(p.stem[:8], p.stem[:8])
        print(f"{p.name} → {label}")
        r = process_file(p, settings=settings)
        lit = r.lit_bgr if r.lit_bgr is not None else r.enhanced_bgr
        looks = apply_all_looks(
            lit, strength=strength, analysis=r.analysis, source_bgr=r.original_bgr
        )

        # Save lit (doğal) + each look
        save_image(lit, OUT / f"{label}_00_dogal-aydinlatma.jpg", quality=92)
        for look, name in LOOK_FILE.items():
            save_image(looks[look], OUT / f"{label}_01_{name}.jpg", quality=92)

        triple = _hconcat(
            [
                looks[LookProfile.NATURAL],
                looks[LookProfile.IPHONE_NATURAL_HDR],
                looks[LookProfile.IPHONE_VIBRANT],
            ]
        )
        save_image(triple, OUT / f"{label}_triple.jpg", quality=90)
        preview = _preview_with_labels(
            [
                ("Natural", looks[LookProfile.NATURAL]),
                ("iPhone Natural HDR", looks[LookProfile.IPHONE_NATURAL_HDR]),
                ("iPhone Vibrant", looks[LookProfile.IPHONE_VIBRANT]),
            ]
        )
        preview_path = OUT / f"{label}_triple_preview.jpg"
        preview.save(preview_path, quality=85, optimize=True)
        print(f"  wrote {preview_path.name}")

    print("OUT", OUT)
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name} {f.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
