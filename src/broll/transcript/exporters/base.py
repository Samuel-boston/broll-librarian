"""Turning matched beats into a timeline an NLE can open.

Rules that are not obvious, stated here once because every exporter obeys them:

* The sequence timebase comes from config, defaulting to the modal fps across
  the suggested clips. Every source's timecodes are converted into it.
* Each clip is placed at its beat's start, trimmed to the beat's duration,
  starting from the shot's ``start_s``.
* If the shot is shorter than the beat, the remainder is left as a gap and
  flagged in the export report - never stretched or frozen.
* Mixed frame rates in one library are normal. The conversion is explicit.

And the practical detail that decides whether the export is usable at all: an
NLE cannot link media from a Google Drive URL. It needs a local path, so each
Drive file id is mapped through ``drive_local_mount_path``. Without that, the
export is still produced but every clip imports offline.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from ...config import WorkspaceConfig
from ...ingest.probe import snap_fps
from ..matcher import BeatMatch, Suggestion

DEFAULT_FPS = 25.0
NTSC_RATES = {23.976: 24, 23.98: 24, 29.97: 30, 59.94: 60, 119.88: 120}


@dataclass
class TimelineItem:
    index: int
    match: BeatMatch
    suggestion: Suggestion
    start_frame: int          # sequence position, in sequence frames
    end_frame: int
    source_in_frame: int      # in the source's own timebase
    source_out_frame: int
    sequence_in_frame: int    # the same in/out counted in sequence frames,
    sequence_out_frame: int   # which is what FCP7 XML and EDL expect
    gap_frames: int           # unfilled remainder of the beat
    media_path: str | None
    offline: bool
    source_fps: float

    @property
    def name(self) -> str:
        return Path(self.suggestion.source.original_filename).name

    @property
    def duration_frames(self) -> int:
        return max(1, self.end_frame - self.start_frame)


@dataclass
class Timeline:
    name: str
    fps: float
    width: int
    height: int
    items: list[TimelineItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    gaps: list[BeatMatch] = field(default_factory=list)

    @property
    def duration_frames(self) -> int:
        return max((item.end_frame for item in self.items), default=0)

    @property
    def timebase(self) -> int:
        return ntsc_timebase(self.fps)[0]

    @property
    def is_ntsc(self) -> bool:
        return ntsc_timebase(self.fps)[1]


def ntsc_timebase(fps: float) -> tuple[int, bool]:
    """FCP7 XML wants an integer timebase plus an NTSC flag."""
    for rate, timebase in NTSC_RATES.items():
        if abs(fps - rate) < 0.01:
            return timebase, True
    return int(round(fps)), False


def seconds_to_frames(seconds: float, fps: float) -> int:
    return int(round(seconds * fps))


def frames_to_timecode(frames: int, fps: float) -> str:
    """Non-drop-frame timecode. Drop-frame is deliberately not emitted."""
    timebase = max(1, int(round(fps)))
    frame = int(frames % timebase)
    total_seconds = int(frames // timebase)
    seconds = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frame:02d}"


def resolve_media_path(suggestion: Suggestion, config: WorkspaceConfig) -> tuple[str | None, bool]:
    """(local path, offline). Offline means the NLE will ask for a relink."""
    source = suggestion.source
    mount = config.drive_local_mount_path
    if mount and source.drive_path:
        return str(Path(mount).expanduser() / source.drive_path), False
    if source.origin_path and Path(source.origin_path).exists():
        return source.origin_path, False
    if source.drive_path:
        return source.drive_path, True
    return source.origin_path, True


def path_to_url(path: str | None) -> str:
    if not path:
        return ""
    return "file://localhost" + quote(str(Path(path).as_posix()))


def choose_fps(matches: list[BeatMatch], config: WorkspaceConfig) -> tuple[float, list[str]]:
    """Config wins; otherwise the modal fps of the clips actually used."""
    warnings: list[str] = []
    if config.transcript.sequence_fps:
        return float(config.transcript.sequence_fps), warnings

    rates = [
        round(float(snap_fps(m.chosen.source.fps)), 3)
        for m in matches
        if m.chosen and m.chosen.source.fps
    ]
    if not rates:
        return DEFAULT_FPS, [f"No source frame rates known - defaulting to {DEFAULT_FPS} fps."]

    fps = statistics.mode(rates)
    distinct = sorted(set(rates))
    if len(distinct) > 1:
        warnings.append(
            f"Mixed frame rates in this timeline ({', '.join(str(r) for r in distinct)}); "
            f"conforming everything to {fps} fps."
        )
    return fps, warnings


def build_timeline(
    matches: list[BeatMatch],
    config: WorkspaceConfig,
    name: str = "B-Roll Suggestions",
    width: int = 1920,
    height: int = 1080,
) -> Timeline:
    fps, warnings = choose_fps(matches, config)
    timeline = Timeline(name=name, fps=fps, width=width, height=height, warnings=list(warnings))

    if not config.drive_local_mount_path:
        timeline.warnings.append(
            "drive_local_mount_path is not set, so clips stored only in Drive will "
            "import offline and need relinking. Set it to your Google Drive for "
            "Desktop path in the workspace config."
        )

    for index, match in enumerate(matches):
        suggestion = match.chosen
        if suggestion is None:
            timeline.gaps.append(match)
            continue

        beat = match.beat
        start_frame = seconds_to_frames(beat.start_s, fps)
        wanted_frames = max(1, seconds_to_frames(beat.duration_s, fps))
        source_fps = float(snap_fps(suggestion.source.fps) or fps)

        available_frames = max(1, seconds_to_frames(suggestion.shot.duration_s, fps))
        used_frames = min(wanted_frames, available_frames)
        gap_frames = wanted_frames - used_frames
        if gap_frames > 0:
            timeline.warnings.append(
                f"Beat {beat.index + 1} ({beat.duration_s:.1f}s) is longer than "
                f"{Path(suggestion.source.original_filename).name} "
                f"({suggestion.shot.duration_s:.1f}s); "
                f"{gap_frames / fps:.1f}s is left as a gap."
            )

        source_in = seconds_to_frames(suggestion.shot.start_s, source_fps)
        source_out = source_in + max(1, seconds_to_frames(used_frames / fps, source_fps))
        sequence_in = seconds_to_frames(suggestion.shot.start_s, fps)
        media_path, offline = resolve_media_path(suggestion, config)
        if offline:
            timeline.warnings.append(
                f"{Path(suggestion.source.original_filename).name} has no local path - "
                "it will import offline."
            )

        timeline.items.append(
            TimelineItem(
                index=index,
                match=match,
                suggestion=suggestion,
                start_frame=start_frame,
                end_frame=start_frame + used_frames,
                source_in_frame=source_in,
                source_out_frame=source_out,
                sequence_in_frame=sequence_in,
                sequence_out_frame=sequence_in + used_frames,
                gap_frames=gap_frames,
                media_path=media_path,
                offline=offline,
                source_fps=source_fps,
            )
        )

    for match in matches:
        if match.no_good_match and match not in timeline.gaps:
            timeline.gaps.append(match)
    return timeline
