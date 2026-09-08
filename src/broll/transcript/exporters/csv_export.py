"""CSV export - for anyone who just wants the list."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from ..matcher import BeatMatch
from .base import Timeline, frames_to_timecode

COLUMNS = [
    "beat", "beat_start", "beat_end", "beat_duration_s", "narration",
    "rank", "filename", "shot_timecode", "shot_start_s", "shot_duration_s",
    "shot_type", "camera_movement", "setting", "mood", "reason", "confidence",
    "reused", "drive_link", "local_path", "sequence_timecode", "gap_s", "status",
]


def build(matches: list[BeatMatch], timeline: Timeline) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()

    placed = {item.match.beat.index: item for item in timeline.items}

    for match in matches:
        beat = match.beat
        if not match.suggestions:
            writer.writerow({
                "beat": beat.index + 1,
                "beat_start": f"{beat.start_s:.3f}",
                "beat_end": f"{beat.end_s:.3f}",
                "beat_duration_s": f"{beat.duration_s:.3f}",
                "narration": beat.text,
                "status": "no good match",
                "reason": match.missing_footage or "",
            })
            continue

        item = placed.get(beat.index)
        for rank, suggestion in enumerate(match.suggestions, start=1):
            writer.writerow({
                "beat": beat.index + 1,
                "beat_start": f"{beat.start_s:.3f}",
                "beat_end": f"{beat.end_s:.3f}",
                "beat_duration_s": f"{beat.duration_s:.3f}",
                "narration": beat.text,
                "rank": rank,
                "filename": suggestion.source.original_filename,
                "shot_timecode": _timecode(suggestion.shot.start_s),
                "shot_start_s": f"{suggestion.shot.start_s:.3f}",
                "shot_duration_s": f"{suggestion.shot.duration_s:.3f}",
                "shot_type": suggestion.shot.shot_type or "",
                "camera_movement": suggestion.shot.camera_movement or "",
                "setting": suggestion.shot.setting or "",
                "mood": "; ".join(suggestion.shot.mood),
                "reason": suggestion.reason,
                "confidence": f"{suggestion.confidence:.2f}",
                "reused": "yes" if suggestion.reused else "",
                "drive_link": suggestion.drive_link or "",
                "local_path": (item.media_path or "") if rank == 1 and item else "",
                "sequence_timecode": (
                    frames_to_timecode(item.start_frame, timeline.fps)
                    if rank == 1 and item else ""
                ),
                "gap_s": (
                    f"{item.gap_frames / timeline.fps:.2f}"
                    if rank == 1 and item and item.gap_frames else ""
                ),
                "status": "placed" if rank == 1 else "alternative",
            })
    return buffer.getvalue()


def _timecode(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s"


def write(matches: list[BeatMatch], timeline: Timeline, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build(matches, timeline), encoding="utf-8")
    return path
