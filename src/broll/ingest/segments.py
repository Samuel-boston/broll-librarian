"""Looking through a file for the shots in it.

For a clip long enough to hold more than one thing, `plan_segments` samples frames across it, asks
the model which stretches are usable and which are setup or dead air, and returns segments that tile
the file. See analysis/segmentation.py for why, and for the rules that clean up the model's answer.

A file the model cannot be asked about (a provider with no support, an error that is not worth
retrying) is not lost: it becomes one segment covering the whole file, which is what the library did
before it looked inside clips at all.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import ValidationError

from ..analysis.providers.base import ProviderError, TransientProviderError, VisionProvider
from ..analysis.segmentation import (
    MIN_SEGMENT_S,
    Segment,
    SegmentContext,
    SegmentPlan,
    combine_windows,
    normalise_segments,
    plan_windows,
)
from ..config import WorkspaceConfig
from .frames import extract_frames_timed

log = logging.getLogger(__name__)

# Frames for finding setup and scene changes are smaller than the ones that describe a shot: they only
# have to show what is going on, and a long recording sends a lot of them.
SEGMENT_FRAME_EDGE = 512


def _whole(start_s: float, end_s: float, reason: str) -> SegmentPlan:
    return SegmentPlan(segments=[Segment(start_s, end_s, "usable")], fallback_reason=reason)


async def plan_segments(
    provider: VisionProvider,
    video,
    *,
    filename: str,
    start_s: float,
    end_s: float,
    file_duration_s: float,
    width: int,
    height: int,
    config: WorkspaceConfig,
    work_dir: Path,
    limiter=None,
    prefix: str = "seg",
) -> SegmentPlan:
    """Segments for the stretch [start_s, end_s] of a file. Never raises for a model that cannot help."""
    ingest = config.ingest
    windows = plan_windows(start_s, end_s, max_frames=ingest.segment_frames)
    if not windows:
        return SegmentPlan()

    per_window: list[list[Segment]] = []
    cost = 0.0
    for number, window in enumerate(windows):
        frames = extract_frames_timed(
            video, work_dir, start_s=window.start_s, duration_s=window.end_s - window.start_s,
            count=window.frames, max_edge=min(ingest.frame_max_edge, SEGMENT_FRAME_EDGE),
            prefix=f"{prefix}{number:03d}", centred=True,
        )
        try:
            if len(frames) < 2:
                # One frame cannot show a change, and no frame at all cannot show anything.
                per_window.append([Segment(window.start_s, window.end_s, "usable")])
                continue
            context = SegmentContext(
                source_filename=filename, file_duration_s=file_duration_s,
                window_start_s=window.start_s, window_end_s=window.end_s,
                width=width, height=height,
            )
            times = [round(f.t, 2) for f in frames]
            paths = [f.path for f in frames]
            proposed = None
            retry_error: str | None = None
            for attempt in (1, 2):
                try:
                    if limiter is not None:
                        await limiter.acquire()
                    proposed = await provider.segment(paths, times, context, retry_error)
                    cost += provider.estimate_segment_cost(len(paths))
                    break
                except ValidationError as exc:
                    retry_error = "; ".join(
                        f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:5]
                    )
                except TransientProviderError:
                    raise  # the queue retries the whole file with backoff
                except ProviderError as exc:
                    log.warning("could not look through %s (%s)", filename, exc)
                    return _whole(start_s, end_s, str(exc))
            if proposed is None:
                return _whole(start_s, end_s, retry_error or "no usable answer")
            per_window.append(
                normalise_segments(
                    proposed.segments, window.start_s, window.end_s,
                    min_segment_s=MIN_SEGMENT_S, max_usable=10_000,
                )
            )
        finally:
            for f in frames:
                f.path.unlink(missing_ok=True)

    segments = combine_windows(per_window, MIN_SEGMENT_S, ingest.max_segments_per_source)
    if not any(s.kind == "usable" for s in segments):
        segments = [Segment(start_s, end_s, "usable", unsure=True)]
    # Setup and dead air stay in the plan, so a person can see what was left out and why.
    return SegmentPlan(segments=segments, cost_usd=cost)


def number_usable(segments: list[Segment]) -> list[Segment]:
    """Give each usable segment its shot index, in time order, and return just those."""
    usable = [s for s in sorted(segments, key=lambda s: s.start_s) if s.kind == "usable"]
    for index, segment in enumerate(usable):
        segment.index = index
    return usable


def primary_segment_index(usable: list[Segment]) -> int:
    """The longest usable segment: its facets name the file and decide the folder it lives in."""
    if not usable:
        return 0
    return max(usable, key=lambda s: s.duration_s).index
