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
    fps = _parse_fps(video.get("avg_frame_rate")) or _parse_fps(video.get("r_frame_rate"))

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
