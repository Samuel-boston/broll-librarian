"""Finding the shot inside a raw clip.

Raw B-roll rarely starts at the shot. The first seconds are someone setting up the camera, walking
away from it, or counting down; the last are the camera coming down. A long take can also hold
several separate scenes. Describing the whole file as one shot gives it tags for the setup, and
cuts it into an edit from the wrong second.

So a clip is first looked at as a sequence: frames spread through it, each with its time, and the
model splits it into segments and labels each `usable`, `setup` or `dead`. Only the usable ones are
analysed and indexed, each as its own shot with its own in and out point, and the model also names the
strongest few seconds inside it.

The model proposes; `normalise_segments` decides. It has to cope with a model that overlaps its
segments, leaves gaps, invents times outside the file, or returns nothing usable - without ever
discarding footage because the model was sloppy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field


class SegmentKind(str, Enum):
    usable = "usable"
    setup = "setup"
    dead = "dead"


class ModelSegment(BaseModel):
    start_s: float = Field(description="Where it starts, in seconds from the start of the file.")
    end_s: float = Field(description="Where it ends, in seconds from the start of the file.")
    kind: SegmentKind
    summary: str = Field(default="", description="One short line: what happens in it.")
    best_start_s: float | None = Field(
        default=None,
        description="For a usable segment: where its strongest 3-10 seconds start.",
    )
    best_end_s: float | None = Field(default=None, description="...and where they end.")


class SegmentationResult(BaseModel):
    segments: list[ModelSegment] = Field(default_factory=list)


class SegmentContext(BaseModel):
    """What the model is told about the stretch it is looking at."""

    source_filename: str
    file_duration_s: float
    window_start_s: float = 0.0
    window_end_s: float | None = None
    width: int = 0
    height: int = 0

    @property
    def end_s(self) -> float:
        return self.window_end_s if self.window_end_s is not None else self.file_duration_s

    @property
    def is_first_window(self) -> bool:
        return self.window_start_s <= 0.01

    @property
    def is_last_window(self) -> bool:
        return self.end_s >= self.file_duration_s - 0.01


SEGMENT_SYSTEM_PROMPT = """\
You are a video editor's assistant looking through raw footage for the parts worth keeping. \
You are shown frames sampled in order from one file, each labelled with its time in seconds.

Divide the stretch you are shown into consecutive segments that cover it with no gaps and no \
overlaps, and label each one:
- usable: a stable, intentional shot with a clear subject that an editor could cut into a \
video.
- setup: the camera being positioned, handled or adjusted; someone walking up to start it or \
away from it at the end; a slate, a countdown or a "rolling" call; focus or exposure being \
set; the operator checking the shot.
- dead: black or near-black, the lens covered, the ground or ceiling by accident, a long blur, \
or nothing in frame.

Start a new segment only when the subject, the place, the action or the kind of shot clearly \
changes - not for small changes inside one continuous shot. Most clips are a single usable \
segment, often with some setup before it and a tail after it. Say so when that is what you \
see, and do not split a good continuous shot into pieces.

For each usable segment also give best_start_s and best_end_s: the strongest 3 to 10 seconds \
inside it - stable, well framed, the subject doing the thing. It is the part an editor would \
cut first. It must lie inside the segment.

Put a boundary between the two frames that show the change; if you cannot tell where, put it \
halfway between them. Use the frame times you are given, not frame numbers.
"""


def build_segment_prompt(
    context: SegmentContext, times: list[float], retry_error: str | None = None
) -> str:
    """The user turn. The frames themselves are attached by the provider, each after its label."""
    parts = [
        f"File: {context.source_filename}",
        f"Whole file: {context.file_duration_s:.1f}s, {context.width}x{context.height}",
        f"You are looking at {context.window_start_s:.1f}s to {context.end_s:.1f}s of it.",
        "Frame times (seconds): " + ", ".join(f"{t:.1f}" for t in times),
    ]
    if not context.is_first_window:
        parts.append("This is not the start of the file, so do not expect setup at the beginning.")
    if not context.is_last_window:
        parts.append("This is not the end of the file; it carries on after the last frame.")
    parts.append(
        "Return the segments for this stretch only, covering "
        f"{context.window_start_s:.1f}s to {context.end_s:.1f}s."
    )
    if retry_error:
        parts += [
            "",
            "YOUR PREVIOUS ANSWER FAILED VALIDATION. Fix exactly this and return the whole "
            "result again:",
            retry_error,
        ]
    return "\n".join(parts)


# --------------------------------------------------------------------------
# What the pipeline works with
# --------------------------------------------------------------------------


@dataclass
class Segment:
    """One stretch of a file, after the model's proposal has been checked."""

    start_s: float
    end_s: float
    kind: str = "usable"            # usable | setup | dead | skipped
    summary: str = ""
    best_start_s: float | None = None
    best_end_s: float | None = None
    index: int = 0                  # shot index, for usable segments
    #: Set when nothing in the file looked usable and the whole of it was kept for a person to judge.
    unsure: bool = False

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def to_dict(self) -> dict:
        return {
            "start_s": round(self.start_s, 2),
            "end_s": round(self.end_s, 2),
            "kind": self.kind,
            "summary": self.summary,
            "best_start_s": None if self.best_start_s is None else round(self.best_start_s, 2),
            "best_end_s": None if self.best_end_s is None else round(self.best_end_s, 2),
            "index": self.index,
            "unsure": self.unsure,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Segment":
        return cls(
            start_s=float(data["start_s"]),
            end_s=float(data["end_s"]),
            kind=str(data.get("kind", "usable")),
            summary=str(data.get("summary", "")),
            best_start_s=data.get("best_start_s"),
            best_end_s=data.get("best_end_s"),
            index=int(data.get("index", 0)),
            unsure=bool(data.get("unsure", False)),
        )


MIN_SEGMENT_S = 1.5


def combine_windows(
    per_window: list[list[Segment]],
    min_segment_s: float = MIN_SEGMENT_S,
    max_usable: int = 8,
) -> list[Segment]:
    """Join the segments of consecutive windows into one tiling, settling the seams."""
    segments = [s for window in per_window for s in window]
    segments = _absorb_short(segments, min_segment_s)
    _check_best_windows(segments)
    usable = [s for s in segments if s.kind == "usable"]
    if len(usable) > max_usable:
        keep = {id(s) for s in sorted(usable, key=lambda s: -s.duration_s)[:max_usable]}
        for s in usable:
            if id(s) not in keep:
                s.kind = "skipped"
                s.best_start_s = s.best_end_s = None
    return segments


def normalise_segments(
    proposed: list[ModelSegment],
    start_s: float,
    end_s: float,
    min_segment_s: float = MIN_SEGMENT_S,
    max_usable: int = 8,
) -> list[Segment]:
    """Turn a model's proposal into segments that tile [start_s, end_s] exactly.

    * times outside the stretch are clamped, and segments left empty by that are dropped;
    * overlaps and gaps are settled at the midpoint between the two segments;
    * a segment shorter than `min_segment_s` is absorbed by a neighbour, so a flicker in the
      model's answer never becomes a one-second shot;
    * the best window must sit inside its segment, and be at least a second long;
    * more than `max_usable` usable segments keeps the longest and marks the rest `skipped`;
    * nothing usable at all keeps the whole stretch, flagged `unsure`, rather than losing it.
    """
    if end_s <= start_s:
        return []
    ordered = sorted(
        (
            (max(start_s, min(end_s, s.start_s)), max(start_s, min(end_s, s.end_s)), s)
            for s in proposed
        ),
        key=lambda t: (t[0], t[1]),
    )
    ordered = [t for t in ordered if t[1] - t[0] > 1e-6]
    if not ordered:
        return [Segment(start_s, end_s, "usable", unsure=True)]

    # Tile: settle every boundary at the midpoint of the overlap or gap between neighbours.
    spans: list[list] = [[a, b, s] for a, b, s in ordered]
    spans[0][0] = start_s
    spans[-1][1] = end_s
    for left, right in zip(spans, spans[1:]):
        edge = (left[1] + right[0]) / 2
        left[1] = right[0] = min(max(edge, left[0]), right[1])

    segments = [
        Segment(
            a, b, s.kind.value, s.summary.strip(),
            best_start_s=s.best_start_s if s.kind is SegmentKind.usable else None,
            best_end_s=s.best_end_s if s.kind is SegmentKind.usable else None,
        )
        for a, b, s in spans
    ]
    segments = _absorb_short(segments, min_segment_s)
    _check_best_windows(segments)

    usable = [s for s in segments if s.kind == "usable"]
    if not usable:
        return [Segment(start_s, end_s, "usable", unsure=True)]
    if len(usable) > max_usable:
        keep = {id(s) for s in sorted(usable, key=lambda s: -s.duration_s)[:max_usable]}
        for s in usable:
            if id(s) not in keep:
                s.kind = "skipped"
                s.best_start_s = s.best_end_s = None
    return segments


def _absorb_short(segments: list[Segment], minimum: float) -> list[Segment]:
    """Fold a segment shorter than `minimum` into the neighbour it is most like, or the longer one."""
    changed = True
    while changed and len(segments) > 1:
        changed = False
        for i, seg in enumerate(segments):
            if seg.duration_s >= minimum:
                continue
            left = segments[i - 1] if i > 0 else None
            right = segments[i + 1] if i + 1 < len(segments) else None
            same = [n for n in (left, right) if n is not None and n.kind == seg.kind]
            pool = same or [n for n in (left, right) if n is not None]
            target = max(pool, key=lambda n: n.duration_s)
            target.start_s = min(target.start_s, seg.start_s)
            target.end_s = max(target.end_s, seg.end_s)
            segments.pop(i)
            changed = True
            break
    return segments


def _check_best_windows(segments: list[Segment]) -> None:
    for seg in segments:
        if seg.kind != "usable":
            continue
        a, b = seg.best_start_s, seg.best_end_s
        if a is None or b is None:
            seg.best_start_s = seg.best_end_s = None
            continue
        a, b = max(seg.start_s, a), min(seg.end_s, b)
        if b - a < 1.0:
            seg.best_start_s = seg.best_end_s = None
        else:
            seg.best_start_s, seg.best_end_s = a, b


@dataclass
class Window:
    """A stretch of a file looked at in one model call."""

    start_s: float
    end_s: float
    frames: int


def plan_windows(
    start_s: float, end_s: float, max_frames: int = 12, window_s: float = 60.0, frame_gap_s: float = 3.0
) -> list[Window]:
    """How to look at a stretch: one call if it is short, one per minute if it is long.

    A short clip gets a frame about every three seconds; a long recording is read a minute at a time
    with up to `max_frames` frames each, so a two-hour file is a couple of hundred frames rather than
    one impossible request.
    """
    duration = max(0.0, end_s - start_s)
    if duration <= 0:
        return []
    if duration <= window_s:
        return [Window(start_s, end_s, max(4, min(max_frames, round(duration / frame_gap_s))))]
    windows: list[Window] = []
    cursor = start_s
    while cursor < end_s - 1e-6:
        stop = min(end_s, cursor + window_s)
        # A sliver at the end joins the window before it rather than costing a call.
        if end_s - stop < window_s * 0.25:
            stop = end_s
        windows.append(Window(cursor, stop, max(4, min(max_frames, round((stop - cursor) / 5.0)))))
        cursor = stop
    return windows


@dataclass
class SegmentPlan:
    """Everything decided about one file: the stretches, and why."""

    segments: list[Segment] = field(default_factory=list)
    cost_usd: float = 0.0
    #: Why the model could not be asked (no provider support, an error): the whole file is one shot.
    fallback_reason: str | None = None

    @property
    def usable(self) -> list[Segment]:
        return [s for s in self.segments if s.kind == "usable"]
