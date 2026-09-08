"""CMX3600 EDL export - the universal fallback.

An EDL has one timebase, so every source timecode is expressed in the sequence
rate. Where a source was shot at a different rate that conversion is lossy at
the frame level, which is why the FCP7 XML is the preferred export and why the
report says so out loud.
"""

from __future__ import annotations

from pathlib import Path

from .base import Timeline, frames_to_timecode


def build(timeline: Timeline) -> str:
    lines = [f"TITLE: {timeline.name}", "FCM: NON-DROP FRAME", ""]

    for position, item in enumerate(timeline.items, start=1):
        # Source timecodes are conformed to the sequence rate.
        source_in = int(round(item.source_in_frame / item.source_fps * timeline.fps))
        source_out = source_in + item.duration_frames
        reel = _reel(item.name)
        lines.append(
            f"{position:03d}  {reel:<8} V     C        "
            f"{frames_to_timecode(source_in, timeline.fps)} "
            f"{frames_to_timecode(source_out, timeline.fps)} "
            f"{frames_to_timecode(item.start_frame, timeline.fps)} "
            f"{frames_to_timecode(item.end_frame, timeline.fps)}"
        )
        lines.append(f"* FROM CLIP NAME: {item.name}")
        if item.media_path:
            lines.append(f"* SOURCE FILE: {item.media_path}")
        lines.append(f"* BEAT: {item.match.beat.text[:70]}")
        if item.suggestion.reason:
            lines.append(f"* REASON: {item.suggestion.reason[:70]}")
        if item.gap_frames:
            lines.append(
                f"* GAP AFTER: {item.gap_frames} frames "
                f"({item.gap_frames / timeline.fps:.1f}s) - clip is shorter than the beat"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _reel(name: str) -> str:
    """EDL reel names are 8 characters of A-Z0-9."""
    cleaned = "".join(c for c in Path(name).stem.upper() if c.isalnum())
    return (cleaned or "AX")[:8].ljust(3, "X")


def write(timeline: Timeline, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build(timeline), encoding="utf-8")
    return path
