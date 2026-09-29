"""Getting a shot's footage onto this machine, for an edit or a render.

The library holds metadata and a Drive file id, never the footage, so a caller
that has chosen a shot still needs the bytes. In order of preference:

1. the file it was indexed from, if it is still there and still the same file
   (its content hash matches);
2. the same file through Google Drive for Desktop (``drive_local_mount_path``),
   which streams only the parts that are read;
3. a download through the Drive API, into the workspace temp directory (kept
   as a cache) or into the folder the caller asked for.

``trim`` then cuts just the shot plus handles, so rendering a four-second
cutaway does not pull a 2 GB camera file into the project. It stream-copies
when a keyframe sits close enough before the in-point - bit-exact and instant -
and otherwise re-encodes at high quality, because a stream copy can only start
on a keyframe and a long camera GOP would hand back seconds of the wrong shot.

Whatever it returns, the caller gets the offset: the source time at which the
returned file starts. A render that addresses the shot in source time
subtracts it and lands on the same frame.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import WorkspaceConfig
from .db.models import Source
from .db.store import Store
from .ingest.hashing import content_hash

log = logging.getLogger(__name__)

DEFAULT_HANDLES_S = 0.5
# A stream copy starts on the keyframe at or before the in-point. Up to this
# much extra lead is worth it for a bit-exact, instant cut; beyond it the file
# carries too much of whatever came before, so re-encode instead.
MAX_COPY_LEAD_S = 1.0
# How far back to look for that keyframe. Camera files put one every 0.5-2 s;
# a stream with none in half a minute is re-encoded.
KEYFRAME_SEARCH_S = 30.0
# Full downloads are kept as a cache for the next trim of the same file. They
# are the only thing this module keeps, so it is also the only thing it evicts.
DOWNLOAD_CACHE_BYTES = 20 * 1024**3
# Visually lossless at a modest size. Only used when a stream copy cannot start
# close enough to the shot.
REENCODE_CRF = "12"
COPY_CONTAINERS = {".mov", ".mp4", ".m4v"}


class FetchError(RuntimeError):
    """Something the caller can act on, with the HTTP status that fits it."""

    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


@dataclass
class FetchResult:
    shot_id: str
    client: str
    source_id: str
    filename: str
    media: str
    path: str               # the file to use
    via: str                # local | drive_mount | drive_download
    source_path: str        # the full-length original that was read
    trimmed: bool
    trim_mode: str | None   # copy | reencode | None
    offset_s: float         # source time at which `path` starts
    shot_start_s: float     # the shot, in source time
    shot_end_s: float
    in_s: float             # the shot, in `path`'s own time
    out_s: float
    handles_s: float
    duration_s: float | None
    bytes: int
    verified: bool          # the original's content hash was checked against the library

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Finding a full copy of the file
# --------------------------------------------------------------------------


def drive_client_for(config: WorkspaceConfig):
    """A Drive client for this client's library. Replaced in tests."""
    from .drive.auth import DriveAuthError, load_credentials
    from .drive.client import DriveClient

    try:
        credentials = load_credentials(config)
    except DriveAuthError as exc:
        raise FetchError(
            f"Drive is not connected for client {config.id!r}: {exc} "
            f"Run `broll drive login -w {config.id}`.", status=503,
        ) from exc
    if credentials is None:
        raise FetchError(
            f"Drive is not connected for client {config.id!r}. "
            f"Run `broll drive login -w {config.id}`.", status=503,
        )
    return DriveClient(credentials)


def mount_candidates(config: WorkspaceConfig, source: Source) -> list[Path]:
    """Where Drive for Desktop would show this file.

    ``drive_path`` is relative to the client's root folder, and the setting is
    normally the "My Drive" folder, so the root folder's name goes in between.
    Someone may equally have pointed the setting at the root folder itself, so
    that is tried second. The content hash decides which, if either, is it.
    """
    if not (config.drive_local_mount_path and source.drive_path):
        return []
    mount = Path(config.drive_local_mount_path).expanduser()
    return [mount / config.drive_root_folder_name / source.drive_path, mount / source.drive_path]


def _same_file(path: Path, source: Source) -> bool:
    try:
        return content_hash(path) == source.content_hash
    except OSError:
        return False


def _safe_name(name: str, limit: int = 60) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "clip"
    return stem[:limit]


def _download_dir(config: WorkspaceConfig) -> Path:
    return config.temp_dir / "fetched"


def _evict_downloads(directory: Path, keep: Path, limit: int = DOWNLOAD_CACHE_BYTES) -> None:
    """Least recently used first, never the file just fetched."""
    files = sorted((p for p in directory.glob("*") if p.is_file()), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in files)
    for path in files:
        if total <= limit:
            break
        if path == keep:
            continue
        total -= path.stat().st_size
        path.unlink(missing_ok=True)


def download(config: WorkspaceConfig, source: Source, directory: Path, drive=None) -> Path:
    """The file from Drive, verified against the library. Reused when already there."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{_safe_name(source.original_filename)}_{source.id[:8]}{Path(source.original_filename).suffix.lower() or '.mp4'}"
    if target.is_file() and _same_file(target, source):
        os.utime(target, None)  # the cache is least-recently-used
        return target
    client = drive or drive_client_for(config)
    partial = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.part")
    try:
        client.download(source.drive_file_id, partial)
        if not _same_file(partial, source):
            raise FetchError(
                f"The copy of {source.original_filename} in Drive is not the file that was "
                "indexed (its content differs). Re-index it, or restore the original.",
            )
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def locate(
    config: WorkspaceConfig, source: Source, download_to: Path | None = None, drive=None,
) -> tuple[Path, str]:
    """A full, verified copy of the source on this machine, and how it was reached."""
    tried: list[str] = []
    if source.origin_path:
        original = Path(source.origin_path)
        if original.is_file():
            if _same_file(original, source):
                return original, "local"
            tried.append(f"the file at {original} has changed since it was indexed")
        else:
            tried.append(f"{original} is no longer there")

    if config.drive_local_mount_path:
        for candidate in mount_candidates(config, source):
            if candidate.is_file() and _same_file(candidate, source):
                return candidate, "drive_mount"
        if source.drive_path:
            tried.append(f"it is not under the Drive for Desktop folder {config.drive_local_mount_path}")
    else:
        tried.append("no Drive for Desktop folder is set (drive_local_mount_path)")

    if source.drive_file_id:
        try:
            path = download(config, source, download_to or _download_dir(config), drive)
        except FetchError as exc:
            if exc.status == 503:  # Drive not connected: say what else was tried, too
                raise FetchError(
                    f"No copy of {source.original_filename} is reachable: "
                    + "; ".join(tried) + f"; and {exc}", status=503,
                ) from exc
            raise
        if download_to is None:
            _evict_downloads(_download_dir(config), keep=path)
        return path, "drive_download"

    tried.append("it has not been filed into Drive yet")
    raise FetchError(
        f"No copy of {source.original_filename} is reachable: " + "; ".join(tried) + ".",
    )


# --------------------------------------------------------------------------
# Cutting the shot out
# --------------------------------------------------------------------------


def _run(cmd: list[str], timeout: float = 900) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _number(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def probe_video(path: Path) -> dict:
    """Duration, pixel format, and where the first keyframe sits on the timeline
    ffmpeg's -ss uses.

    That timeline starts at the file's start time, not at zero: a stream copy of
    B-frame footage begins at 0.08 s, and ffmpeg (so a render) calls that 0. The
    first keyframe's distance from the start time is what anchors a cut.
    """
    cp = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "format=duration,start_time:stream=pix_fmt,codec_name", "-of", "json", str(path),
    ], timeout=60)
    data = json.loads(cp.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    frames = json.loads(_run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#8",
        "-show_entries", "frame=best_effort_timestamp_time,pts_time,key_frame", "-of", "json", str(path),
    ], timeout=60).stdout or "{}").get("frames") or []
    start = _number(fmt.get("start_time")) or 0.0
    key = next((f for f in frames if str(f.get("key_frame")) == "1"), frames[0] if frames else {})
    key_pts = _number(key.get("best_effort_timestamp_time")) or _number(key.get("pts_time"))
    return {
        "duration": _number(fmt.get("duration")),
        "pix_fmt": stream.get("pix_fmt") or "",
        "codec": stream.get("codec_name") or "",
        "key_at": (key_pts - start) if key_pts is not None else 0.0,
    }


def _frame_times(path: Path, interval: str, keyframes_only: bool = False) -> list[float]:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if keyframes_only:
        cmd += ["-skip_frame", "nokey"]
    cmd += ["-read_intervals", interval, "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(path)]
    times = []
    for line in _run(cmd, timeout=120).stdout.splitlines():
        try:
            times.append(float(line.strip().rstrip(",")))
        except ValueError:
            continue
    return times


def keyframe_at_or_before(path: Path, t: float) -> float | None:
    """The last keyframe at or before `t`, looking back KEYFRAME_SEARCH_S at most."""
    start = max(0.0, t - KEYFRAME_SEARCH_S)
    before = [k for k in _frame_times(path, f"{start:.3f}%{t + 0.5:.3f}", keyframes_only=True)
              if k <= t + 1e-3]
    return max(before) if before else None


def first_frame_at_or_after(path: Path, t: float) -> float | None:
    """The presentation time of the first frame at or after `t`."""
    after = [f for f in _frame_times(path, f"{t:.3f}%+0.5") if f >= t - 1e-4]
    return min(after) if after else None


def _x264_pix_fmt(source_pix_fmt: str) -> str:
    """Keep the source's chroma and bit depth: a 10-bit log file stays 10-bit."""
    deep = any(depth in source_pix_fmt for depth in ("10", "12", "16"))
    chroma = "444" if "444" in source_pix_fmt else "422" if "422" in source_pix_fmt else "420"
    return f"yuv{chroma}p10le" if deep else f"yuv{chroma}p"


def cut(original: Path, dest: Path, start: float, end: float) -> tuple[str, float]:
    """Cut [start, end] of `original` into `dest`. Returns (mode, offset_s).

    The offset maps the two timelines: the frame a render gets by seeking to t
    in `dest` is the frame at t + offset in the original. It is measured, not
    assumed, because both kinds of cut move the start (checked frame by frame
    on test footage):

    * a stream copy starts on a keyframe. Seeking exactly onto the keyframe's
      printed time can land a whole GOP early, so the seek goes half a
      millisecond past it, and the keyframe's own time is the anchor.
    * a re-encode starts on the first frame at or after the in-point, and
      rounds its timestamps to the frame grid, so the seek goes to just before
      that frame and the frame's own time is the anchor.

    The anchor frame is found in the result as its first keyframe, measured on
    the timeline ffmpeg seeks on (see probe_video).

    Every cut is probed afterwards; a copy that comes out the wrong length is
    redone as a re-encode.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    info = probe_video(original)
    key = keyframe_at_or_before(original, start)
    attempts: list[str] = []
    if key is not None and start - key <= MAX_COPY_LEAD_S:
        attempts.append("copy")
    attempts.append("reencode")

    errors: list[str] = []
    for mode in attempts:
        if mode == "copy":
            anchor, seek = key, key + 0.0005
        else:
            anchor = first_frame_at_or_after(original, start)
            anchor = start if anchor is None else anchor
            seek = max(0.0, anchor - 0.001)
        partial = dest.with_name(f".{dest.stem}.{uuid.uuid4().hex[:8]}.part{dest.suffix}")
        # Two threads each way: a cut is a few seconds of footage, and the machine
        # it runs on is usually busy with something that matters more.
        common = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-threads", "2", "-ss", f"{seek:.6f}",
                  "-i", str(original), "-t", f"{end - anchor:.6f}", "-map", "0:v:0", "-map", "0:a:0?",
                  "-sn", "-dn", "-threads", "2"]
        if mode == "copy":
            cmd = common + ["-c", "copy", "-avoid_negative_ts", "make_zero"]
        else:
            cmd = common + ["-c:v", "libx264", "-preset", "medium", "-crf", REENCODE_CRF,
                            "-pix_fmt", _x264_pix_fmt(info["pix_fmt"]), "-c:a", "aac", "-b:a", "192k"]
        if dest.suffix.lower() in (".mp4", ".mov", ".m4v"):
            cmd += ["-movflags", "+faststart"]
        try:
            cp = _run(cmd + [str(partial)])
            if cp.returncode != 0 or not partial.is_file():
                errors.append(f"{mode}: {(cp.stderr or '').strip()[-300:]}")
                continue
            got = probe_video(partial)
            expected = end - anchor
            if got["duration"] is None or abs(got["duration"] - expected) > 0.25:
                errors.append(f"{mode}: came out {got['duration']}s long, expected {expected:.2f}s")
                continue
            os.replace(partial, dest)
            return mode, round(anchor - got["key_at"], 6)
        finally:
            partial.unlink(missing_ok=True)
    raise FetchError(f"Could not cut {original.name}: " + " | ".join(errors), status=500)


# --------------------------------------------------------------------------
# The whole fetch
# --------------------------------------------------------------------------


def fetch_shot(
    config: WorkspaceConfig,
    store: Store,
    shot_id: str,
    *,
    dest_dir: Path | None = None,
    trim: bool = False,
    handles_s: float = DEFAULT_HANDLES_S,
    copy: bool = False,
    drive=None,
) -> FetchResult:
    """A local file for one shot. See the module docstring for the order."""
    shot = store.get_shot(shot_id)
    if shot is None:
        raise FetchError(f"No shot {shot_id!r} in client {config.id!r}.", status=404)
    source = store.get_source(shot.source_id)
    if source is None:
        raise FetchError(f"Shot {shot_id!r} has no source file on record.", status=404)
    if dest_dir is not None and not dest_dir.is_absolute():
        raise FetchError(f"dest_dir must be an absolute path, got {dest_dir}.", status=422)

    # A trim only needs the full file as an intermediate, so it goes to the
    # cache; an untrimmed download goes wherever the caller asked.
    original, via = locate(config, source, None if trim else dest_dir, drive)
    is_image = source.media_kind == "image"
    duration = source.duration_s or 0.0
    start = max(0.0, shot.start_s - handles_s)
    end = min(duration, shot.end_s + handles_s) if duration else shot.end_s + handles_s
    whole_file = start <= 0.05 and (not duration or end >= duration - 0.05)

    if trim and not is_image and not whole_file:
        target_dir = dest_dir or (config.temp_dir / "trims")
        suffix = original.suffix.lower() if original.suffix.lower() in COPY_CONTAINERS else ".mov"
        dest = target_dir / (
            f"{_safe_name(source.original_filename)}_{source.id[:8]}-{shot.shot_index}"
            f"_{start:.2f}-{end:.2f}{suffix}"
        )
        mode, offset = cut(original, dest, start, end)
        if mode == "reencode" and dest.suffix != ".mp4":
            # A re-encode is H.264 whatever the source was; say so in the name.
            final = dest.with_suffix(".mp4")
            os.replace(dest, final)
            dest = final
        info = probe_video(dest)
        return FetchResult(
            shot_id=shot.id, client=config.id, source_id=source.id,
            filename=source.original_filename, media=source.media_kind, path=str(dest),
            via=via, source_path=str(original), trimmed=True, trim_mode=mode,
            offset_s=offset, shot_start_s=shot.start_s, shot_end_s=shot.end_s,
            in_s=round(shot.start_s - offset, 6), out_s=round(shot.end_s - offset, 6),
            handles_s=handles_s, duration_s=info["duration"], bytes=dest.stat().st_size,
            verified=True,
        )

    path = original
    if copy and dest_dir is not None and original.parent.resolve() != dest_dir.resolve():
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / f"{_safe_name(source.original_filename)}_{source.id[:8]}{original.suffix.lower()}"
        if not (path.is_file() and _same_file(path, source)):
            partial = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.part")
            try:
                shutil.copy2(original, partial)
                os.replace(partial, path)
            finally:
                partial.unlink(missing_ok=True)
    return FetchResult(
        shot_id=shot.id, client=config.id, source_id=source.id,
        filename=source.original_filename, media=source.media_kind, path=str(path),
        via=via, source_path=str(original), trimmed=False, trim_mode=None, offset_s=0.0,
        shot_start_s=shot.start_s, shot_end_s=shot.end_s, in_s=shot.start_s, out_s=shot.end_s,
        handles_s=handles_s, duration_s=None if is_image else source.duration_s,
        bytes=path.stat().st_size, verified=True,
    )
