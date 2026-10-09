"""Camera RAW photographs (.NEF and friends).

ffmpeg and Pillow cannot decode them, and a RAW file is a negative: for finding the right photo, the
full-size JPEG every camera embeds in it is exactly as good, and much cheaper to read. So the preview
is pulled out with libraw (the `rawpy` package) and analysed in the RAW's place. The RAW file itself is
never changed.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

RAW_SUFFIXES = {
    ".nef", ".nrw", ".arw", ".srf", ".sr2", ".cr2", ".cr3", ".crw", ".dng", ".raf", ".orf",
    ".rw2", ".pef", ".srw", ".x3f", ".3fr", ".erf", ".mef", ".mrw", ".kdc", ".dcr",
}


def is_raw(path: str | Path) -> bool:
    return Path(path).suffix.lower() in RAW_SUFFIXES


def rawpy_available() -> bool:
    try:
        import rawpy  # noqa: F401
    except ImportError:
        return False
    return True


def extract_preview(raw_path: Path, out_dir: Path) -> Path | None:
    """The photograph inside a RAW file, as a JPEG in `out_dir`. None if it cannot be read."""
    try:
        import rawpy
    except ImportError:
        log.warning("RAW support needs the 'raw' extra (pip install 'broll-librarian[raw]')")
        return None

    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{raw_path.stem}_preview.jpg"
    try:
        with rawpy.imread(str(raw_path)) as raw:
            flip = getattr(raw.sizes, "flip", 0)
            image = None
            try:
                thumb = raw.extract_thumb()
            except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
                thumb = None
            if thumb is not None and thumb.format == rawpy.ThumbFormat.JPEG:
                target.write_bytes(thumb.data)
                with Image.open(target) as opened:
                    image = opened.convert("RGB")
                # An embedded preview is stored sideways for a portrait shot; libraw says how.
                image = _orient(image, flip)
            elif thumb is not None and thumb.format == rawpy.ThumbFormat.BITMAP:
                image = _orient(Image.fromarray(thumb.data).convert("RGB"), flip)
            else:
                # No usable preview: develop a small version of the RAW itself.
                rgb = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
                image = Image.fromarray(rgb).convert("RGB")
        image.save(target, "JPEG", quality=90)
        return target
    except Exception as exc:  # noqa: BLE001 - a damaged RAW is a file to flag, not a crash
        log.warning("could not read %s: %s", raw_path.name, exc)
        target.unlink(missing_ok=True)
        return None


def _orient(image, flip: int):
    """libraw's flip codes: 3 = upside down, 5 = turned left, 6 = turned right."""
    if flip == 3:
        return image.rotate(180, expand=True)
    if flip == 5:
        return image.rotate(90, expand=True)
    if flip == 6:
        return image.rotate(-90, expand=True)
    return image
