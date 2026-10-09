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

EXTRACT_TIMEOUT_S = 300

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


def ffmpeg_input(video) -> list[str]:
    """The ffmpeg arguments that open `video`: a local path, or a remote source that knows how."""
    opener = getattr(video, "ffmpeg_input", None)
    return list(opener()) if opener else ["-i", str(video)]


def _extract_one(
    video, timestamp: float, out_path: Path, max_edge: int, seek: bool = True, fast: bool = False
) -> bool:
    ffmpeg, _ = check_ffmpeg()
    scale = (
        f"scale='if(gt(iw,ih),min({max_edge},iw),-2)':"
        f"'if(gt(iw,ih),-2,min({max_edge},ih))'"
    )
    # A still has exactly one frame, and seeking - even to 0 - lands past it:
    # ffmpeg then exits 0 having written nothing. So don't seek into a still.
    seek_args = ["-ss", f"{max(timestamp, 0):.3f}"] if seek else []
    if fast and seek:
        # Decode only keyframes: the frame is the first keyframe after the time asked for, up to a
        # second or so late. Seeking exactly means decoding every frame since the last keyframe, and in
        # 4K at 120 fps that is sixty frames (5 s) for one picture; this is 0.7 s.
        seek_args += ["-skip_frame", "nokey"]
    try:
        proc = subprocess.run(
            [
                ffmpeg, "-nostdin", "-loglevel", "error",
                *seek_args, *ffmpeg_input(video),
                "-frames:v", "1", "-vf", scale, "-q:v", "3", "-y", str(out_path),
            ],
            capture_output=True, text=True, timeout=EXTRACT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        # A frame that cannot be had in this long (a stalled connection) is a frame to skip, not a
        # worker to lose.
        log.warning("timed out reading a frame at %.1fs", timestamp)
        out_path.unlink(missing_ok=True)
        return False
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


@dataclass
class Frame:
    path: Path
    #: Seconds from the start of the file.
    t: float


def extract_frames_timed(
    video,
    out_dir: Path,
    start_s: float = 0.0,
    duration_s: float | None = None,
    count: int = 3,
    max_edge: int = 768,
    prefix: str = "frame",
    centred: bool = False,
    fast: bool = False,
    times: list[float] | None = None,
) -> list[Frame]:
    """Extract ``count`` usable frames spread across a stretch, each with where it came from.

    Returns the frames that passed the quality filter, in chronological order.
    If every sample at a position is rejected, the best of the rejected ones is
    kept rather than dropping the position entirely - an all-dark clip should
    still be analysed and flagged, not silently skipped.

    ``times`` takes frames at those moments (seconds in the file) instead of spreading `count` of them.
    ``fast`` reads only keyframes (see `_extract_one`): for sources too big to decode frame by frame.
    ``centred`` puts each frame in the middle of an equal slice (so the first and last frames sit
    close to the two ends), which is what finding a setup at the start needs. The default keeps the
    old 20/50/80% positions for a shot that is analysed as a whole.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = duration_s or 0.0
    if times and duration:
        # Exact moments asked for: the fractions are where each falls in the stretch.
        positions = tuple(min(1.0, max(0.0, (t - start_s) / duration)) for t in times)
    else:
        positions = _centred_positions(count) if centred else _positions(count)
    kept: list[Frame] = []
    seek = duration > 0

    for index, fraction in enumerate(positions):
        base = start_s + (duration * fraction if duration else 0.0)
        chosen: Frame | None = None
        fallback: tuple[float, Frame] | None = None

        for attempt, offset in enumerate((0.0, *RESAMPLE_OFFSETS)):
            timestamp = base + offset * duration
            if duration and not (start_s <= timestamp <= start_s + duration):
                continue
            candidate = out_dir / f"{prefix}_{index:02d}_{attempt}.jpg"
            if not _extract_one(video, timestamp, candidate, max_edge, seek, fast):
                continue
            stats = frame_stats(candidate)
            frame = Frame(candidate, max(0.0, timestamp))
            if stats.usable:
                chosen = frame
                break
            score = stats.stddev
            if fallback is None or score > fallback[0]:
                fallback = (score, frame)

        if chosen is None and fallback is not None:
            chosen = fallback[1]
        if chosen is not None:
            kept.append(chosen)

    # Clean up rejected samples we did not keep.
    keep_paths = {f.path for f in kept}
    for leftover in out_dir.glob(f"{prefix}_*.jpg"):
        if leftover not in keep_paths:
            leftover.unlink(missing_ok=True)
    return sorted(kept, key=lambda f: f.t)


def extract_frames(
    video,
    out_dir: Path,
    start_s: float = 0.0,
    duration_s: float | None = None,
    count: int = 3,
    max_edge: int = 768,
    prefix: str = "frame",
) -> list[Path]:
    """`extract_frames_timed`, when only the files are wanted."""
    return [
        f.path for f in extract_frames_timed(
            video, out_dir, start_s, duration_s, count, max_edge, prefix
        )
    ]


def frame_count_for(duration_s: float, minimum: int = 3, maximum: int = 8, every_s: float = 4.0) -> int:
    """How many frames a stretch of this length deserves: one per `every_s`, within limits.

    Three frames describe a 4-second clip well and a 40-second one badly: the middle of a long shot
    is where the subject changes.
    """
    if duration_s <= 0:
        return minimum
    return max(minimum, min(maximum, round(duration_s / every_s)))


def _centred_positions(count: int) -> tuple[float, ...]:
    count = max(1, count)
    return tuple((i + 0.5) / count for i in range(count))


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


def best_frame_timed(frames: list[Frame], prefer: tuple[float, float] | None = None) -> Path | None:
    """The frame to use as a thumbnail: the most varied one, from the best part when there is one."""
    if not frames:
        return None
    pool = frames
    if prefer is not None:
        inside = [f for f in frames if prefer[0] <= f.t <= prefer[1]]
        pool = inside or frames
    return max(pool, key=lambda f: frame_stats(f.path).stddev).path


#: Footage at least this many pixels on its long edge is read keyframe by keyframe.
FAST_SEEK_EDGE = 2500


def wants_fast_seek(width: int, height: int) -> bool:
    return max(width or 0, height or 0) >= FAST_SEEK_EDGE


def times_weighted_to(
    start_s: float, end_s: float, count: int, best: tuple[float, float] | None, share: float = 0.4
) -> list[float] | None:
    """Moments to take frames at, with most of them inside the shot's strongest stretch.

    Describing a 34-second shot from eight evenly spaced frames lets a few seconds at the edges (someone
    glancing at a laptop) weigh as much as the ten seconds that are the shot. With the best part known,
    about `share` of the frames come from it (the rest still cover the whole shot, so a shot is never judged by its best seconds alone), spread evenly, and the rest from before and after it, so
    the description follows what an editor would actually use. None when there is no best part, or it
    is most of the shot anyway.
    """
    duration = end_s - start_s
    if not best or duration <= 0 or count < 4:
        return None
    lo, hi = max(start_s, best[0]), min(end_s, best[1])
    if hi - lo < 1.0 or (hi - lo) >= 0.8 * duration:
        return None
    inside = max(2, round(count * share))
    outside = count - inside
    picked = [lo + (hi - lo) * (i + 0.5) / inside for i in range(inside)]
    before, after = lo - start_s, end_s - hi
    if outside:
        # Split what is left between the two sides in proportion to how much shot there is on each.
        total = before + after
        n_before = round(outside * before / total) if total > 0 else 0
        n_before = min(max(n_before, 1 if before > 0.5 else 0), outside)
        n_after = outside - n_before
        picked += [start_s + before * (i + 0.5) / n_before for i in range(n_before)] if n_before and before > 0 else []
        picked += [hi + after * (i + 0.5) / n_after for i in range(n_after)] if n_after and after > 0 else []
    return sorted(picked)
