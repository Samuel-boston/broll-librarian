"""A pack: the clips chosen for a script, cut out of Drive, plus a timeline that opens with them online.

The library holds Drive ids, not footage, so an exported timeline imports offline. A pack fixes that for the
editor: for every line's chosen clip the server cuts just the shot plus handles - straight out of Drive over
HTTPS for a video, so a 100 GB camera file costs only the seconds that are read - and zips the cuts with a
timeline whose media paths point at the folder the editor will unzip into.

It is a background job because a script may have thirty-odd lines and each cut takes a while. Jobs run one at
a time (the server is small), and the working files are deleted as soon as each is in the zip. A finished zip is
deleted after ``transcript.pack_keep_hours``. A pack stops growing at ``transcript.pack_max_gb``; the lines that
did not fit are listed, never silently dropped.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import shutil
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .. import fetch as fetch_module
from ..config import WorkspaceConfig
from ..db.store import Store
from ..drive.fetcher import GB, reserve_room
from ..ingest.errors import DiskSpaceError
from .exporters import edl, fcp7xml
from .exporters.base import PackedClip, Timeline, build_timeline, clean_folder
from .matcher import BeatMatch

log = logging.getLogger(__name__)

# Something a person cannot do much with if it is bigger: a handle of more than this is a different clip.
MAX_HANDLES_S = 10.0
# What a cut is expected to weigh next to the source's own data rate. A re-encode at high quality can come out
# larger than a camera file, so leave room.
SIZE_MARGIN = 1.6
MIN_CLIP_BYTES = 5 * 1024 * 1024
IMAGE_GUESS_BYTES = 60 * 1024 * 1024


def safe_stem(name: str, limit: int = 50) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "clip"
    return stem[:limit]


def zip_base_name(run_name: str) -> str:
    """The pack's name, and the name of the folder a double-clicked zip unpacks into."""
    return f"{safe_stem(run_name, 40)}-broll-pack"


def clip_file_name(number: int, beat_start_s: float, original_name: str, suffix: str) -> str:
    """001_00m12s_Walking_alone.mp4 - the line it was first chosen for, and where that line starts."""
    minutes, seconds = divmod(int(beat_start_s), 60)
    return f"{number:03d}_{minutes:02d}m{seconds:02d}s_{safe_stem(original_name)}{suffix.lower()}"


@dataclass
class PackSkip:
    number: int
    timecode: str
    text: str
    reason: str


@dataclass
class PackJob:
    id: str
    run_id: str
    name: str
    folder: str
    handles_s: float
    state: str = "queued"          # queued | running | done | failed
    total: int = 0                 # distinct clips to cut
    done: int = 0
    current: str = ""
    lines: int = 0                 # lines that have a chosen clip
    gaps: list[PackSkip] = field(default_factory=list)       # lines with "No good match"
    skipped: list[PackSkip] = field(default_factory=list)    # lines with a clip that could not be packed
    packed_lines: int = 0
    bytes: int = 0
    limit_hit: bool = False
    error: str | None = None
    zip_path: Path | None = None
    zip_name: str = ""
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    keep_hours: float = 6.0
    max_gb: float = 5.0

    @property
    def percent(self) -> int:
        return int(100 * self.done / self.total) if self.total else 0

    @property
    def finished(self) -> bool:
        return self.state in ("done", "failed")

    @property
    def expires_at(self) -> float | None:
        return self.finished_at + self.keep_hours * 3600 if self.finished_at else None


class PackManager:
    """Runs pack jobs one at a time and deletes what they leave behind."""

    def __init__(self, config: WorkspaceConfig):
        self.config = config
        self.jobs: dict[str, PackJob] = {}
        self._pool: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()
        self.sweep(startup=True)

    @property
    def root(self) -> Path:
        return self.config.temp_dir / "packs"

    def start(self, matches: list[BeatMatch], run_id: str, run_name: str, folder: str, handles_s: float) -> PackJob:
        """Queue a pack for these matches. ValueError if the folder or handles are not usable."""
        folder = clean_folder(folder)
        if not 0 <= handles_s <= MAX_HANDLES_S:
            raise ValueError(f"Handles must be between 0 and {MAX_HANDLES_S:g} seconds.")
        transcript = self.config.transcript
        job = PackJob(
            id=uuid.uuid4().hex[:10], run_id=run_id, name=run_name, folder=folder, handles_s=handles_s,
            keep_hours=transcript.pack_keep_hours, max_gb=transcript.pack_max_gb,
            zip_name=zip_base_name(run_name) + ".zip",
        )
        # A copy of the choices as they are now: the person may keep swapping clips while this runs.
        snapshot = [dataclasses.replace(m) for m in matches]
        job.lines = sum(1 for m in snapshot if m.chosen)
        job.total = len({m.chosen.shot.id for m in snapshot if m.chosen})
        with self._lock:
            self.sweep()
            self.jobs[job.id] = job
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pack")
            self._pool.submit(self._run, job, snapshot)
        return job

    def get(self, job_id: str) -> PackJob | None:
        job = self.jobs.get(job_id)
        if job and job.state == "done" and (job.zip_path is None or not job.zip_path.is_file()):
            return None  # swept
        return job

    def _run(self, job: PackJob, matches: list[BeatMatch]) -> None:
        job.state = "running"
        try:
            build_pack(self.config, matches, job, self.root / job.id)
        except Exception as exc:  # noqa: BLE001 - whatever it was, the person gets a sentence, not a hang
            log.exception("pack %s failed", job.id)
            job.state = "failed"
            job.error = fetch_module._scrub(str(exc)) or exc.__class__.__name__
            job.finished_at = time.time()
            shutil.rmtree(self.root / job.id, ignore_errors=True)

    def sweep(self, startup: bool = False) -> int:
        """Delete finished packs older than the keep time, and (at start-up) whatever a killed run left."""
        root = self.root
        if not root.is_dir():
            return 0
        removed = 0
        now = time.time()
        keep_s = self.config.transcript.pack_keep_hours * 3600
        live = {j.id for j in self.jobs.values() if not j.finished}
        for entry in root.iterdir():
            if entry.name in live:
                continue
            try:
                age = now - entry.stat().st_mtime
                stale = startup and not any(entry.glob("*.zip")) or age > keep_s
                if stale:
                    shutil.rmtree(entry, ignore_errors=True) if entry.is_dir() else entry.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                continue
        return removed


def _estimate_bytes(source, duration_s: float, image: bool) -> int:
    if image:
        return int(source.filesize_bytes or IMAGE_GUESS_BYTES)
    if source.filesize_bytes and source.duration_s:
        rate = source.filesize_bytes / source.duration_s
        return max(MIN_CLIP_BYTES, int(rate * duration_s * SIZE_MARGIN))
    return MIN_CLIP_BYTES * 4


def _short(text: str, limit: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def build_pack(config: WorkspaceConfig, matches: list[BeatMatch], job: PackJob, job_dir: Path) -> None:
    """Cut every chosen clip, zip them with the timeline, and finish the job. Runs in a worker thread."""
    work = job_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    cap = int(job.max_gb * GB)
    headroom = config.ingest.disk_headroom_gb
    handles = job.handles_s
    zip_part = job_dir / f".{job.zip_name}.part"
    final_zip = job_dir / job.zip_name

    # The first line each distinct shot is chosen for names its file; later lines share the file.
    first_line: dict[str, BeatMatch] = {}
    # The longest stretch any line takes from a shot. Nothing past it is cut: a 60 s shot under a 4 s line
    # would otherwise be a minute of re-encoded footage the timeline never uses.
    needed: dict[str, float] = {}
    for match in matches:
        number = match.beat.index + 1
        if match.chosen is None:
            job.gaps.append(PackSkip(number, match.beat.timecode, match.beat.text,
                                     match.missing_footage or "No good match in the library."))
        else:
            if match.chosen.shot.id not in first_line:
                first_line[match.chosen.shot.id] = match
            needed[match.chosen.shot.id] = max(needed.get(match.chosen.shot.id, 0.0), match.beat.duration_s)

    job.total = len(first_line)
    packed: dict[str, PackedClip] = {}
    failed: dict[str, str] = {}
    store = Store.for_config(config)
    drive = None
    total_bytes = 0
    stopped: str | None = None
    try:
        with zipfile.ZipFile(zip_part, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
            for shot_id, match in first_line.items():
                suggestion = match.chosen
                source, shot = suggestion.source, suggestion.shot
                job.current = f"Line {match.beat.index + 1}: {_short(match.beat.text, 50)}"
                if stopped:
                    failed[shot_id] = stopped
                    job.done += 1
                    continue

                image = source.media_kind == "image"
                length = min(shot.duration_s, needed[shot_id]) if not image else 0.0
                estimate = _estimate_bytes(source, length + 2 * handles + 1.0, image)
                if total_bytes + estimate > cap:
                    job.limit_hit = True
                    stopped = (f"The pack reached its {job.max_gb:g} GB limit before this clip. "
                               "Take the rest from Drive by hand.")
                    failed[shot_id] = stopped
                    job.done += 1
                    continue

                try:
                    # Room for the cut and for its copy in the zip, while both exist.
                    with reserve_room(work, 2 * estimate, headroom):
                        if source.drive_file_id and drive is None and not _has_local_copy(config, source):
                            drive = fetch_module.drive_client_for(config)
                        result = fetch_module.fetch_shot(
                            config, store, shot_id, dest_dir=work, trim=not image, handles_s=handles,
                            copy=True, drive=drive, stream=True,
                            max_len_s=needed[shot_id],
                        )
                        path = Path(result.path)
                        if total_bytes + result.bytes > cap:
                            job.limit_hit = True
                            stopped = (f"The pack reached its {job.max_gb:g} GB limit before this clip. "
                                       "Take the rest from Drive by hand.")
                            failed[shot_id] = stopped
                            _discard(path, work)
                            job.done += 1
                            continue
                        name = clip_file_name(match.beat.index + 1, match.beat.start_s,
                                              source.original_filename, path.suffix)
                        archive.write(path, f"clips/{name}")
                        total_bytes += result.bytes
                        packed[shot_id] = PackedClip(
                            shot_id=shot_id, file=name, in_s=0.0 if image else result.in_s,
                            out_s=0.0 if image else result.out_s,
                            duration_s=None if image else result.duration_s,
                            media=source.media_kind, trim_mode=result.trim_mode, bytes=result.bytes,
                        )
                        _discard(path, work)
                except DiskSpaceError as exc:
                    stopped = f"The server ran low on disk space ({_short(str(exc), 160)})"
                    failed[shot_id] = stopped
                except fetch_module.FetchError as exc:
                    failed[shot_id] = _short(fetch_module._scrub(str(exc)), 240)
                except Exception as exc:  # noqa: BLE001 - one bad file must not sink the other thirty
                    log.exception("pack %s: shot %s failed", job.id, shot_id)
                    failed[shot_id] = _short(fetch_module._scrub(str(exc)) or exc.__class__.__name__, 240)
                job.done += 1

            for match in matches:
                if match.chosen and match.chosen.shot.id in failed:
                    job.skipped.append(PackSkip(match.beat.index + 1, match.beat.timecode, match.beat.text,
                                                failed[match.chosen.shot.id]))
            job.packed_lines = sum(1 for m in matches if m.chosen and m.chosen.shot.id in packed)

            if not packed:
                reasons = sorted({s.reason for s in job.skipped})
                raise RuntimeError(
                    "No clip could be packed." + (" " + " | ".join(reasons[:3]) if reasons else
                                                  " None of the lines has a chosen clip.")
                )

            job.current = "Writing the timeline"
            timeline = build_timeline(matches, config, name=job.name, packed=packed, pack_folder=job.folder)
            compressed = zipfile.ZIP_DEFLATED
            archive.writestr("timeline.xml", fcp7xml.build(timeline, job.folder), compress_type=compressed)
            archive.writestr("timeline.edl", edl.build(timeline), compress_type=compressed)
            archive.writestr("README.txt", readme(job, timeline, packed), compress_type=compressed)
        os.replace(zip_part, final_zip)
    finally:
        store.close()
        shutil.rmtree(work, ignore_errors=True)
        zip_part.unlink(missing_ok=True)

    job.zip_path = final_zip
    job.bytes = final_zip.stat().st_size
    job.current = ""
    job.state = "done"
    job.finished_at = time.time()


def _has_local_copy(config: WorkspaceConfig, source) -> bool:
    found, _ = fetch_module.find_local(config, source)
    return found is not None


def _discard(path: Path, work: Path) -> None:
    """Delete a working file - but never the library's own original, which `copy` can hand back."""
    try:
        if path.resolve().is_relative_to(work.resolve()):
            path.unlink(missing_ok=True)
    except OSError:
        pass


# --------------------------------------------------------------------------
# README
# --------------------------------------------------------------------------


def readme(job: PackJob, timeline: Timeline, packed: dict[str, PackedClip]) -> str:
    made = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    folder = job.folder
    windows = bool(re.match(r"^[A-Za-z]:", folder) or folder.startswith("//"))
    clips_at = f"{folder}/clips".replace("/", "\\") if windows else f"{folder}/clips"
    clip_files = {c.file for c in packed.values()}
    out = [
        f"B-ROLL PACK: {job.name}",
        f"Made {made}. {len(clip_files)} clip file(s) for {job.packed_lines} line(s) of the script, "
        f"{timeline.fps:g} fps, {job.handles_s:g} s of handle either side of every shot.",
        "",
        "WHAT IS IN THIS ZIP",
        "  clips/         the chosen clip for each line, cut down to just the shot plus handles",
        "  timeline.xml   a Final Cut Pro 7 XML timeline; Premiere Pro and DaVinci Resolve both open it",
        "  timeline.edl   the same cuts as an EDL, if you need one",
        "  README.txt     this file",
        "",
        "HOW TO IMPORT",
        f"  1. Unzip so that the clips folder is exactly here:  {clips_at}",
        "     (Double-clicking a zip on a Mac, or Extract All on Windows, makes a new folder named after the zip.",
        "     The timeline expects the folder you typed on the page, so unzip into that one, or move the",
        "     clips folder there afterwards.)",
        "  2. Premiere Pro: File > Import, choose timeline.xml. The clips come in online.",
        "     DaVinci Resolve: File > Import > Timeline, choose timeline.xml.",
        "  3. The timeline starts at the first line of the script. Put it on top of your cut and slide it so",
        "     time 0 lines up with the start of your script.",
        "",
        "THE HANDLES",
        f"  Each file runs {job.handles_s:g} s longer than the shot at both ends. The timeline uses only the shot,",
        "  so you can slide or extend a clip by up to that much without going back to Drive.",
        "",
        "IF THE CLIPS COME IN OFFLINE (the folder was not the one typed on the page)",
        "  Premiere Pro: in the Project panel select the offline clips, right-click > Link Media, press Locate,",
        "  pick any one file in the clips folder, and keep \"Relink others automatically\" ticked.",
        "  DaVinci Resolve: in the Media Pool select the clips, right-click > Relink Selected Clips, and choose",
        "  the clips folder.",
        "  Or run the page's button again with the right folder - the clips are the same.",
        "",
    ]
    if job.gaps:
        out += ["LINES WITH NO CLIP (no good match in the library - these are gaps in the timeline)"]
        for gap in job.gaps:
            out.append(f"  line {gap.number} at {gap.timecode}: \"{_short(gap.text, 90)}\"")
            out.append(f"      footage to find or shoot: {_short(gap.reason, 160)}")
        out.append("")
    if job.skipped:
        out += ["LINES WHOSE CLIP COULD NOT BE PACKED (they are not in the timeline)"]
        for skip in job.skipped:
            out.append(f"  line {skip.number} at {skip.timecode}: \"{_short(skip.text, 90)}\"")
            out.append(f"      why: {skip.reason}")
        out.append("")
    if job.limit_hit:
        out += [f"SIZE LIMIT: a pack holds at most {job.max_gb:g} GB, and this one reached it.",
                "  The lines listed above stopped there. Take those clips from Drive by hand.", ""]
    if timeline.warnings:
        out += ["NOTES FROM THE TIMELINE"]
        out += [f"  - {w}" for w in timeline.warnings]
        out.append("")
    out += [
        "CAN'T BE CHECKED FROM HERE",
        "  The timeline follows the FCP7 XML format, but it was built without opening it in Premiere or Resolve.",
        "  If something looks off (a clip a few frames short, a wrong frame), tell whoever sent you this pack.",
        "  Still photos are copied as they are; a RAW photo (NEF, DNG...) may need Camera Raw to open.",
    ]
    return "\n".join(out) + "\n"
