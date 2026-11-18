"""Load / save images with EXIF orientation handling."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

# Optional HEIC
_HEIF_OK = False
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    _HEIF_OK = True
except Exception:
    _HEIF_OK = False


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
if _HEIF_OK:
    SUPPORTED_EXTENSIONS |= {".heic", ".heif"}


def heif_available() -> bool:
    return _HEIF_OK


def load_image(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load image as BGR uint8 (OpenCV convention) after applying EXIF orientation.
    Returns (bgr, meta) where meta holds original size, mode, and stripped EXIF.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    img = Image.open(path)
    exif = img.getexif()
    oriented = ImageOps.exif_transpose(img)
    if oriented.mode not in ("RGB", "RGBA"):
        oriented = oriented.convert("RGB")
    elif oriented.mode == "RGBA":
        background = Image.new("RGB", oriented.size, (255, 255, 255))
        background.paste(oriented, mask=oriented.split()[3])
        oriented = background

    rgb = np.asarray(oriented, dtype=np.uint8)
    # RGB → BGR for OpenCV pipeline
    bgr = rgb[:, :, ::-1].copy()

    meta: dict[str, Any] = {
        "path": str(path),
        "size": oriented.size,  # (w, h)
        "mode": oriented.mode,
        "exif": exif,
        "format": (img.format or path.suffix.lstrip(".")).upper(),
    }
    return bgr, meta


def save_image(
    bgr: np.ndarray,
    path: str | Path,
    *,
    quality: int = 95,
    exif: Any | None = None,
) -> Path:
    """
    Save BGR uint8 image. Orientation already applied → do not re-write Orientation tag
    that would double-rotate. Keep other EXIF when possible (JPEG).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rgb = bgr[:, :, ::-1]
    img = Image.fromarray(rgb, mode="RGB")

    save_kwargs: dict[str, Any] = {}
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        save_kwargs.update(quality=quality, subsampling=0, optimize=True)
        # Strip Orientation (tag 274) to avoid double rotation on viewers
        if exif is not None:
            try:
                exif_copy = exif.copy() if hasattr(exif, "copy") else exif
                if hasattr(exif_copy, "__delitem__") and 274 in exif_copy:
                    del exif_copy[274]
                save_kwargs["exif"] = exif_copy
            except Exception:
                pass
    elif suffix == ".png":
        save_kwargs.update(optimize=True)

    img.save(path, **save_kwargs)
    return path


def output_path_for(
    source: str | Path,
    out_dir: str | Path,
    suffix: str = "_AYDINLIK",
) -> Path:
    """Build `duzenlenmis_fotograflar/<stem>_AYDINLIK.jpg` style path."""
    source = Path(source)
    out_dir = Path(out_dir)
    ext = source.suffix.lower()
    if ext in {".heic", ".heif"}:
        ext = ".jpg"
    if ext not in {".jpg", ".jpeg", ".png"}:
        ext = ".jpg"
    name = f"{source.stem}{suffix}{ext}"
    return out_dir / name


def default_output_dir(near: str | Path | None = None) -> Path:
    """Prefer cwd/duzenlenmis_fotograflar, or next to source folder."""
    if near is not None:
        base = Path(near).resolve().parent
        return base / "duzenlenmis_fotograflar"
    return Path(os.getcwd()) / "duzenlenmis_fotograflar"
