"""Shot boundary detection.

Typical B-roll here is under three minutes and often a single take, so this is
tuned to avoid over-splitting: a minimum shot length, and a fallback that treats
the whole file as one shot when detection produces a spray of very short ones.
A single-shot file must produce exactly one shot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 27.0


@dataclass
class DetectedShot:
    index: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def scenedetect_available() -> bool:
    try:
        import scenedetect  # noqa: F401
    except ImportError:
        return False
    return True


def detect_shots(
    video: Path,
    duration_s: float,
    min_shot_length_s: float = 1.5,
    min_average_shot_length_s: float = 2.0,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[DetectedShot]:
    """Split a video into shots, falling back to one shot for the whole file."""
    single = [DetectedShot(index=0, start_s=0.0, end_s=duration_s)]

    if duration_s <= min_shot_length_s * 2:
        return single
    if not scenedetect_available():
        log.info("scenedetect not installed - treating %s as a single shot", video.name)
        return single

    from scenedetect import ContentDetector, detect

    try:
        scenes = detect(
            str(video),
            ContentDetector(threshold=threshold, min_scene_len=1),
            show_progress=False,
        )
    except Exception as exc:  # a codec scenedetect cannot open is not fatal
        log.warning("shot detection failed for %s (%s) - using one shot", video.name, exc)
        return single

    if not scenes:
        return single

    spans = _merge_short_shots(
        [(_seconds(s), _seconds(e)) for s, e in scenes],
        min_shot_length_s,
    )
    if not spans:
        return single

    average = sum(e - s for s, e in spans) / len(spans)
    if average < min_average_shot_length_s:
        log.info(
            "%s split into %d shots averaging %.1fs - below the %.1fs floor, "
            "treating it as one shot",
            video.name, len(spans), average, min_average_shot_length_s,
        )
        return single

    # Snap the ends so the shots cover the file exactly.
    spans[0] = (0.0, spans[0][1])
    spans[-1] = (spans[-1][0], max(spans[-1][1], duration_s))
    return [DetectedShot(index=i, start_s=s, end_s=e) for i, (s, e) in enumerate(spans)]


def _seconds(timecode) -> float:
    """PySceneDetect renamed get_seconds() to a `seconds` property in 0.7."""
    value = getattr(timecode, "seconds", None)
    return float(value) if value is not None else float(timecode.get_seconds())


def _merge_short_shots(
    spans: list[tuple[float, float]], minimum: float
) -> list[tuple[float, float]]:
    """Fold any shot shorter than ``minimum`` into its predecessor."""
    merged: list[tuple[float, float]] = []
    for start, end in spans:
        if merged and (end - start) < minimum:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    # A short first shot has no predecessor; fold it forward instead.
    while len(merged) > 1 and (merged[0][1] - merged[0][0]) < minimum:
        merged[1] = (merged[0][0], merged[1][1])
        merged.pop(0)
    return merged


def primary_index(shots: list[DetectedShot]) -> int:
    """The longest shot. Its facets drive the source's Drive filename."""
    if not shots:
        return 0
    return max(shots, key=lambda s: s.duration_s).index
