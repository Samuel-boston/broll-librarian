"""ffprobe wrapper."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from pydantic import BaseModel

from ..config import check_ffmpeg


class NotAVideoError(ValueError):
    pass


class ProbeResult(BaseModel):
    duration_s: float
    width: int
    height: int
    fps: float | None = None
    codec: str | None = None
    filesize_bytes: int
    container: str | None = None
    has_audio: bool = False


def _parse_fps(rate: str | None) -> float | None:
    if not rate or rate in ("0/0", "0"):
        return None
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else None
        except ValueError:
            return None
    try:
        return float(rate)
    except ValueError:
        return None


# Rates an NLE sequence can actually be set to.
STANDARD_RATES = (23.976, 24.0, 25.0, 29.97, 30.0, 48.0, 50.0, 59.94, 60.0, 100.0, 119.88, 120.0)


def snap_fps(fps: float | None, tolerance: float = 0.015) -> float | None:
    """The standard rate a measured rate is really meant to be.

    Phone footage is variable frame rate, so its *average* comes out as 29.58
    or 30.04 fps. Handed to an NLE as-is, that makes a sequence nobody can set.
    """
    if not fps:
        return fps
    nearest = min(STANDARD_RATES, key=lambda rate: abs(rate - fps))
    return nearest if abs(nearest - fps) / nearest <= tolerance else fps


def nominal_fps(r_frame_rate: str | None, avg_frame_rate: str | None) -> float | None:
    """Prefer the stream's declared rate; fall back to the snapped average."""
    declared = _parse_fps(r_frame_rate)
    if declared and any(abs(declared - rate) < 0.01 for rate in STANDARD_RATES):
        return declared
    return snap_fps(_parse_fps(avg_frame_rate) or declared)


def probe(path: Path) -> ProbeResult:
    _, ffprobe = check_ffmpeg()
    if not path.exists():
        raise FileNotFoundError(path)
    proc = subprocess.run(
        [
            ffprobe, "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise NotAVideoError(f"ffprobe could not read {path.name}: {proc.stderr.strip()[:200]}")

    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise NotAVideoError(f"{path.name} contains no video stream - skipping.")

    fmt = data.get("format", {})
    duration = video.get("duration") or fmt.get("duration")
    fps = nominal_fps(video.get("r_frame_rate"), video.get("avg_frame_rate"))

    return ProbeResult(
        duration_s=float(duration) if duration else 0.0,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=fps,
        codec=video.get("codec_name"),
        filesize_bytes=int(fmt.get("size") or path.stat().st_size),
        container=fmt.get("format_name"),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )
