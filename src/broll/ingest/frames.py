"""Keyframe extraction and quality filtering.

Three frames per shot at 20/50/80% of its duration, downscaled to 768px on the
long edge before they are sent anywhere - resolution beyond that buys nothing
and costs tokens. Frames that are near-black, near-white or nearly flat (a fade
or a blur) are rejected and resampled from a nearby timestamp.
"""

from __future__ import annotations

import logging
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..config import check_ffmpeg

log = logging.getLogger(__name__)

DEFAULT_POSITIONS = (0.2, 0.5, 0.8)
# Retry offsets, as a fraction of shot duration, when a sample is rejected.
RESAMPLE_OFFSETS = (0.06, -0.06, 0.12, -0.12, 0.2)

NEAR_BLACK = 18.0
NEAR_WHITE = 240.0
MIN_STDDEV = 8.0


@dataclass
class FrameStats:
    mean: float
    stddev: float

    @property
    def usable(self) -> bool:
        if self.mean < NEAR_BLACK or self.mean > NEAR_WHITE:
            return False
        return self.stddev >= MIN_STDDEV

    @property
    def reason(self) -> str:
        if self.mean < NEAR_BLACK:
            return "near-black"
        if self.mean > NEAR_WHITE:
            return "near-white"
        if self.stddev < MIN_STDDEV:
            return "low-variance"
        return "ok"


def frame_stats(path: Path) -> FrameStats:
    with Image.open(path) as img:
        grey = img.convert("L")
        # A thumbnail is enough to judge exposure and variance, and is far
        # cheaper than reading every pixel of a 4K still.
        grey.thumbnail((160, 160))
        # mode 'L' packs one byte per pixel, so tobytes() is the pixel list.
        pixels = list(grey.tobytes())
    mean = statistics.fmean(pixels)
    stddev = statistics.pstdev(pixels) if len(pixels) > 1 else 0.0
    return FrameStats(mean=mean, stddev=stddev)


def _extract_one(
    video: Path, timestamp: float, out_path: Path, max_edge: int, seek: bool = True
) -> bool:
    ffmpeg, _ = check_ffmpeg()
    scale = (
        f"scale='if(gt(iw,ih),min({max_edge},iw),-2)':"
        f"'if(gt(iw,ih),-2,min({max_edge},ih))'"
    )
    # A still has exactly one frame, and seeking - even to 0 - lands past it:
    # ffmpeg then exits 0 having written nothing. So don't seek into a still.
    seek_args = ["-ss", f"{max(timestamp, 0):.3f}"] if seek else []
    proc = subprocess.run(
        [
            ffmpeg, "-nostdin", "-loglevel", "error",
            *seek_args, "-i", str(video),
            "-frames:v", "1", "-vf", scale, "-q:v", "3", "-y", str(out_path),
        ],
        capture_output=True, text=True,
    )
    return proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0


def extract_still(
    image: Path, out_dir: Path, max_edge: int = 768, prefix: str = "still"
) -> list[Path]:
    """The one frame of a photograph, downscaled.

    An iPhone HEIC is stored as a grid of 512x512 tiles - 99 streams for one
    photo - which ffmpeg stitches with an implicit complex filtergraph. A -vf
    scale on top of that is refused ("Simple and complex filtering cannot be
    used together"), and the extraction produced nothing at all. So decode
    first, at full size, and let Pillow do the scaling.
    """
    ffmpeg, _ = check_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    full = out_dir / f"{prefix}_full.jpg"
    proc = subprocess.run(
        [ffmpeg, "-nostdin", "-loglevel", "error", "-i", str(image),
         "-frames:v", "1", "-q:v", "2", "-y", str(full)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not full.exists() or full.stat().st_size == 0:
        log.warning("could not decode %s: %s", image.name, proc.stderr.strip()[:200])
        return []

    frame = out_dir / f"{prefix}_00.jpg"
    try:
        with Image.open(full) as img:
            img = img.convert("RGB")
            img.thumbnail((max_edge, max_edge))
            img.save(frame, quality=88)
    except OSError as exc:
        log.warning("could not scale %s: %s", image.name, exc)
        return []
    finally:
        full.unlink(missing_ok=True)
    return [frame]


def extract_frames(
    video: Path,
    out_dir: Path,
    start_s: float = 0.0,
    duration_s: float | None = None,
    count: int = 3,
    max_edge: int = 768,
    prefix: str = "frame",
) -> list[Path]:
    """Extract ``count`` usable frames spread across a shot.

    Returns the frames that passed the quality filter, in chronological order.
    If every sample at a position is rejected, the best of the rejected ones is
    kept rather than dropping the position entirely - an all-dark clip should
    still be analysed and flagged, not silently skipped.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = duration_s or 0.0
    positions = _positions(count)
    kept: list[Path] = []
    seek = duration > 0

    for index, fraction in enumerate(positions):
        base = start_s + (duration * fraction if duration else 0.0)
        chosen: Path | None = None
        fallback: tuple[float, Path] | None = None

        for attempt, offset in enumerate((0.0, *RESAMPLE_OFFSETS)):
            timestamp = base + offset * duration
            if duration and not (start_s <= timestamp <= start_s + duration):
                continue
            candidate = out_dir / f"{prefix}_{index:02d}_{attempt}.jpg"
            if not _extract_one(video, timestamp, candidate, max_edge, seek):
                continue
            stats = frame_stats(candidate)
            if stats.usable:
                chosen = candidate
                break
            score = stats.stddev
            if fallback is None or score > fallback[0]:
                fallback = (score, candidate)

        if chosen is None and fallback is not None:
            chosen = fallback[1]
        if chosen is not None:
            kept.append(chosen)

    # Clean up rejected samples we did not keep.
    for leftover in out_dir.glob(f"{prefix}_*.jpg"):
        if leftover not in kept:
            leftover.unlink(missing_ok=True)
    return kept


def _positions(count: int) -> tuple[float, ...]:
    if count == len(DEFAULT_POSITIONS):
        return DEFAULT_POSITIONS
    if count <= 1:
        return (0.5,)
    step = 1.0 / (count + 1)
    return tuple(step * (i + 1) for i in range(count))


def save_thumbnail(frame: Path, out_path: Path, max_edge: int = 640) -> Path:
    """Save the best keyframe as a JPEG for the search UI."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(frame) as img:
        img = img.convert("RGB")
        img.thumbnail((max_edge, max_edge))
        img.save(out_path, "JPEG", quality=82, optimize=True)
    return out_path


def best_frame(frames: list[Path]) -> Path | None:
    """The frame with the most visual variety - the least likely to be a fade."""
    if not frames:
        return None
    return max(frames, key=lambda f: frame_stats(f).stddev)
