"""The ingest pipeline for one source file.

Every stage is independently resumable: the process will be killed halfway
through a 4,000-clip run and must pick up cleanly. Resumability comes from the
database, not from in-memory state - a source that already has an indexed shot
for a given index is not re-analysed.

Stages: fetch -> dedupe -> probe -> detect shots -> frames -> analyse -> embed
-> thumbnail -> (organise, M3) -> clean up.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .. import attention
from ..analysis.analyzer import Analyzer
from ..analysis.embedder import Embedder
from ..analysis.prompt import PROMPT_VERSION
from ..analysis.schema import ShotContext
from ..analysis.providers.base import TransientProviderError
from ..analysis.segmentation import Segment, cap_usable
from ..config import WorkspaceConfig
from ..db.models import Shot, Source
from ..db.store import Store, new_id
from .frames import Frame, best_frame_timed, save_thumbnail
from .hashing import content_hash
from .limits import Verdict, check_limits
from .probe import NotAVideoError, ProbeResult, probe
from .raw import extract_preview, is_raw, rawpy_available
from .remote import RemoteDriveVideo
from .scanner import DiscoveredFile, media_kind
from .segments import number_usable, plan_segments, primary_segment_index
from .shots import detect_shots

log = logging.getLogger(__name__)


def _is_corrected(shot: Shot) -> bool:
    return bool((shot.raw_analysis or {}).get("corrected_by_operator"))


@dataclass
class IngestResult:
    source_id: str | None = None
    status: str = "pending"
    filename: str = ""
    shots_analysed: int = 0
    shots_skipped: int = 0
    cost_usd: float = 0.0
    deduped: bool = False
    organised: bool = False
    error: str | None = None
    messages: list[str] = field(default_factory=list)
    #: Set when the file was turned away, not failed: the kind it was listed under.
    skipped_kind: str | None = None


class IngestPipeline:
    def __init__(
        self,
        config: WorkspaceConfig,
        store: Store,
        analyzer: Analyzer | None = None,
        embedder: Embedder | None = None,
        organise: Callable[[str], object] | None = None,
        drive_client: Callable[[], object] | None = None,
    ):
        self.config = config
        self.store = store
        self.analyzer = analyzer or Analyzer(config)
        self._embedder = embedder
        # How to get a Drive client for reading a video in place. Overridable for tests.
        self._drive_client = drive_client or self._default_drive_client
        # Runs in a worker thread, so it must open its own database connection
        # and Drive client - see drive_organise_callable().
        self._organise = organise

    @property
    def embedder(self) -> Embedder | None:
        return self._embedder

    # -- entry point --------------------------------------------------------

    async def ingest(
        self,
        discovered: DiscoveredFile,
        force: bool = False,
        overwrite_corrections: bool = False,
        allow_long: bool = False,
    ) -> IngestResult:
        progress: dict[str, str] = {}
        try:
            return await self._ingest(discovered, force, overwrite_corrections, progress, allow_long)
        except BaseException:
            # A failure partway through a source - a 503, a kill - must not
            # strand it at "analysing": the shots already written decide its
            # status (and its prompt version). Then let the queue retry.
            if "source_id" in progress:
                self.store.recompute_source_status(progress["source_id"])
            raise

    def _turn_away(
        self, discovered: DiscoveredFile, verdict: Verdict, result: IngestResult,
        size_bytes: int | None = None, duration_s: float | None = None,
    ) -> IngestResult:
        """Not indexed, and not a failure either: listed for a person, with a link."""
        attention.flag_file(
            self.store, discovered, verdict.kind, verdict.detail,
            size_bytes=size_bytes, duration_s=duration_s,
        )
        result.status = "skipped"
        result.skipped_kind = verdict.kind
        result.messages.append(f"{attention.heading(verdict.kind)}: {verdict.detail}")
        return result

    def _default_drive_client(self):
        from ..drive.auth import load_credentials
        from ..drive.client import DriveClient

        credentials = load_credentials(self.config)
        if credentials is None:
            raise RuntimeError("Drive is not connected for this workspace. Run `broll drive login`.")
        return DriveClient(credentials)

    def _streams(self, discovered: DiscoveredFile) -> bool:
        """Is this a video big enough, and in Drive, to be read in place rather than downloaded?"""
        limit = self.config.ingest.stream_above_gb
        return bool(
            limit and discovered.origin == "drive" and discovered.drive_file_id
            and discovered.size_bytes and discovered.size_bytes >= limit * 1_000_000_000
            and media_kind(discovered.filename) == "video"
        )

    def _free_bytes(self) -> int | None:
        try:
            self.config.temp_dir.mkdir(parents=True, exist_ok=True)
            return shutil.disk_usage(self.config.temp_dir).free
        except OSError:
            return None

    async def _ingest(
        self,
        discovered: DiscoveredFile,
        force: bool,
        overwrite_corrections: bool,
        progress: dict[str, str],
        allow_long: bool = False,
    ) -> IngestResult:
        result = IngestResult(filename=discovered.filename)
        ingest = self.config.ingest
        is_image = media_kind(discovered.filename) == "image"

        # 1. Turn away what Drive already says is too long or too big - before downloading a byte.
        size = discovered.size_bytes
        if size is None and discovered.path and discovered.path.exists():
            size = discovered.path.stat().st_size
        stream = self._streams(discovered)
        on_disk = bool(discovered.path and discovered.path.exists())
        verdict = check_limits(
            ingest, size_bytes=None if stream else size, duration_s=discovered.duration_s,
            is_image=is_image, forced=allow_long,
            # A file already on the disk takes no more of it; only a download needs room.
            free_bytes=None if (on_disk or stream) else self._free_bytes(),
        )
        if verdict:
            return self._turn_away(discovered, verdict, result, size, discovered.duration_s)

        if stream:
            # Too big to download sensibly: read frames straight from Drive.
            video = RemoteDriveVideo(
                await asyncio.to_thread(self._drive_client), discovered.drive_file_id,
                discovered.filename, discovered.size_bytes,
            )
            result.messages.append("Read in place from Drive (too big to download)")
        else:
            video = await self._fetch(discovered)
        if video is None:
            result.status = "failed"
            result.error = "could not obtain the file's bytes"
            return result

        # 2. A RAW photograph is read through the JPEG inside it; the RAW itself is never touched.
        working = video
        raw_dir: Path | None = None
        if isinstance(video, Path) and is_raw(video):
            raw_dir = self.config.temp_dir / f"raw-{new_id()[:8]}"
            preview = await asyncio.to_thread(extract_preview, video, raw_dir)
            if preview is None:
                kind = "unreadable" if rawpy_available() else "unsupported"
                detail = (
                    "The photo inside this RAW file could not be read."
                    if kind == "unreadable"
                    else "RAW photos need the 'raw' extra installed on the server."
                )
                self._cleanup_raw(raw_dir)
                self._cleanup(None, video, discovered, None)
                return self._turn_away(discovered, Verdict(kind, detail), result, size)
            working = preview

        try:
            return await self._index(
                discovered, video, working, force, overwrite_corrections, progress, result,
                allow_long,
            )
        finally:
            self._cleanup_raw(raw_dir)

    async def _index(
        self,
        discovered: DiscoveredFile,
        video: Path | RemoteDriveVideo,
        working: Path | RemoteDriveVideo,
        force: bool,
        overwrite_corrections: bool,
        progress: dict[str, str],
        result: IngestResult,
        allow_long: bool,
    ) -> IngestResult:
        digest = await asyncio.to_thread(
            video.content_hash if isinstance(video, RemoteDriveVideo) else (lambda: content_hash(video))
        )
        existing = self.store.source_by_hash(digest)
        if existing and not force and self._is_complete(existing):
            result.source_id = existing.id
            result.status = existing.status
            result.deduped = True
            self.store.resolve_attention(attention.key_for(discovered))
            self._cleanup(None, video, discovered, existing.id)  # a duplicate's download is not kept
            return result

        try:
            meta: ProbeResult = await asyncio.to_thread(probe, working)
        except NotAVideoError as exc:
            result.status = "failed"
            result.error = str(exc)
            return result
        if isinstance(video, RemoteDriveVideo):
            meta.filesize_bytes = video.size or meta.filesize_bytes
        elif working is not video:
            meta.filesize_bytes = video.stat().st_size  # the RAW's size, not its preview's

        # The length is known for certain now: a file Drive gave no length for is caught here.
        remote = isinstance(video, RemoteDriveVideo)
        verdict = check_limits(
            self.config.ingest,
            # Read in place from Drive, a file takes no disk: its size is no reason to turn it away.
            size_bytes=None if remote else meta.filesize_bytes, duration_s=meta.duration_s,
            is_image=meta.media_kind == "image", forced=allow_long, free_bytes=None,
        )
        if verdict:
            self._cleanup(None, video, discovered, None)
            return self._turn_away(discovered, verdict, result, meta.filesize_bytes, meta.duration_s)

        source = existing or self._create_source(discovered, video, digest, meta)
        result.source_id = source.id
        progress["source_id"] = source.id
        self.store.update_source(
            source.id, status="analysing", analysis_version=PROMPT_VERSION, error_message=None
        )
        work_dir = self.config.temp_dir / f"src-{source.id[:8]}"

        # 3. Decide which stretches of the file are shots. A plan made once is kept, so a run killed
        #    half way resumes with the same shots instead of asking the model again and getting a
        #    different answer.
        plan = await self._plan(source, working, meta, force, work_dir, allow_long)
        result.cost_usd += plan["cost"]
        result.messages.extend(plan["messages"])
        segments: list[Segment] = plan["segments"]
        usable = number_usable(segments)
        self.store.update_source(source.id, segments_json=json.dumps([s.to_dict() for s in segments]))
        self.store.prune_shots(source.id, keep=len(usable))
        primary = primary_segment_index(usable)

        for seg in usable:
            shot_id = f"{source.id}-{seg.index}"
            existing_shot = self.store.get_shot(shot_id)
            if existing_shot and existing_shot.status == "indexed" and not force:
                result.shots_skipped += 1
                continue
            if existing_shot and _is_corrected(existing_shot) and not overwrite_corrections:
                # A human already fixed this shot. Re-analysis must not quietly
                # throw that away; --overwrite-corrections is the explicit opt-in.
                result.shots_skipped += 1
                result.messages.append(
                    f"kept operator correction for shot {seg.index} of {discovered.filename}"
                )
                continue

            shot = existing_shot or Shot(
                id=shot_id,
                workspace_id=self.config.id,
                source_id=source.id,
                shot_index=seg.index,
            )
            shot.is_primary = seg.index == primary
            shot.start_s = seg.start_s
            shot.end_s = seg.end_s
            shot.duration_s = seg.duration_s
            shot.best_start_s = seg.best_start_s
            shot.best_end_s = seg.best_end_s

            context = ShotContext(
                source_filename=discovered.filename,
                duration_s=seg.duration_s,
                width=meta.width,
                height=meta.height,
                media_kind=meta.media_kind,
                fps=meta.fps,
                shot_index=seg.index,
                shot_count=len(usable),
                start_s=seg.start_s,
                end_s=seg.end_s,
            )

            frames: list[Frame] = await asyncio.to_thread(
                self.analyzer.extract_timed, working, context, work_dir
            )
            if not frames and isinstance(working, RemoteDriveVideo):
                # Drive stopped answering: nothing is wrong with the clip, so do not write it down as
                # one that could not be described. The queue tries the file again.
                raise TransientProviderError(
                    f"could not read any frames of {discovered.filename} from Drive "
                    f"({seg.start_s:.0f}s-{seg.end_s:.0f}s)"
                )
            context = context.model_copy(update={"frame_times": [round(f.t, 2) for f in frames]})
            outcome = await self.analyzer.analyse_frames([f.path for f in frames], context)
            result.cost_usd += outcome.cost_usd

            if outcome.result is not None:
                shot.apply_analysis(outcome.result, PROMPT_VERSION)
                shot.status = outcome.status
                shot.error_message = outcome.error
            else:
                shot.status = "needs_review"
                shot.error_message = outcome.error
            reasons = list(outcome.review_reasons)
            if seg.unsure and "no_usable_part" not in reasons:
                # The model found nothing usable in this file; the whole of it is kept for a person.
                reasons.append("no_usable_part")
                shot.status = "needs_review"
            shot.review_reasons = reasons

            keyframe = await asyncio.to_thread(
                best_frame_timed, frames,
                (seg.best_start_s, seg.best_end_s) if seg.best_start_s is not None else None,
            )
            if keyframe:
                thumbnail = self.config.thumbnails_dir / f"{shot.id}.jpg"
                await asyncio.to_thread(
                    save_thumbnail, keyframe, thumbnail,
                    self.config.ingest.thumbnail_max_edge,
                )
                shot.thumbnail_path = str(thumbnail)

            if existing_shot:
                self.store.update_shot(shot)
            else:
                self.store.insert_shot(shot)
            if outcome.oov:
                self.store.record_vocabulary_candidates(outcome.oov)
            if outcome.proposal:
                self.store.record_folder_proposal(outcome.proposal[0], outcome.proposal[1], shot.id)

            await self._embed(shot.id)
            result.shots_analysed += 1

            for frame in frames:
                frame.path.unlink(missing_ok=True)

        result.status = self.store.recompute_source_status(source.id)
        self.store.resolve_attention(attention.key_for(discovered))

        # Step 10: organise into Drive, *before* cleanup. For an upload the
        # staged file is the only copy, so it has to reach Drive first.
        if self._organise is not None and result.status in ("indexed", "needs_review"):
            try:
                await asyncio.to_thread(self._organise, source.id)
            except Exception as exc:  # a Drive hiccup must not lose the analysis
                log.warning("filing %s into Drive failed: %s", discovered.filename, exc)
                result.messages.append(f"Drive filing failed: {exc}")
            refreshed = self.store.get_source(source.id)
            result.organised = bool(refreshed and refreshed.drive_file_id)

        self._cleanup(work_dir, video, discovered, source.id)
        return result

    async def _plan(
        self, source: Source, working: Path, meta: ProbeResult, force: bool, work_dir: Path,
        allow_long: bool,
    ) -> dict:
        """The stretches of this file that are shots."""
        ingest = self.config.ingest
        messages: list[str] = []
        stored = [] if force else source.segments
        if stored:
            try:
                return {"segments": [Segment.from_dict(d) for d in stored], "cost": 0.0, "messages": messages}
            except (KeyError, ValueError, TypeError):
                log.warning("stored segments for %s are unreadable; planning again", source.id)

        if meta.media_kind == "image":
            # Nothing to detect: a still is one "shot" of no duration.
            return {"segments": [Segment(0.0, 0.0, "usable")], "cost": 0.0, "messages": messages}

        duration = meta.duration_s
        if ingest.scene_detect_max_s and 0 < duration <= ingest.scene_detect_max_s and isinstance(working, Path):
            spans = await asyncio.to_thread(
                detect_shots, working, duration,
                ingest.min_shot_length_s, ingest.min_average_shot_length_s,
            )
            ranges = [(sp.start_s, sp.end_s) for sp in spans]
        else:
            ranges = [(0.0, duration)]

        segments: list[Segment] = []
        cost = 0.0
        for start, end in ranges:
            if end - start < ingest.segment_min_s:
                segments.append(Segment(start, end, "usable"))
                continue
            plan = await plan_segments(
                self.analyzer.provider, working, filename=source.original_filename,
                start_s=start, end_s=end, file_duration_s=duration, width=meta.width,
                height=meta.height, config=self.config, work_dir=work_dir,
                limiter=self.analyzer.limiter, prefix=f"seg{len(segments):02d}-",
            )
            cost += plan.cost_usd
            if plan.fallback_reason:
                messages.append(f"Looked at the whole of it as one shot: {plan.fallback_reason}")
            segments.extend(plan.segments)
        if not segments:
            segments = [Segment(0.0, duration, "usable")]
        cap_usable(segments, ingest.max_segments_per_source)
        left_out = sum(1 for s in segments if s.kind == "skipped")
        if left_out:
            messages.append(
                f"{left_out} shorter stretch(es) were not indexed: a file keeps at most "
                f"{ingest.max_segments_per_source} shots (the longest)."
            )
        return {"segments": segments, "cost": cost, "messages": messages}

    def _is_complete(self, source) -> bool:
        """Only a finished source short-circuits the pipeline.

        A source left mid-ingest by a killed worker is in `analysing` with some
        or none of its shots written; that must resume, not be skipped as a
        duplicate.
        """
        if source.status not in ("indexed", "needs_review"):
            return False
        shots = self.store.shots_for_source(source.id)
        planned = {int(s.get("index", 0)) for s in source.segments if s.get("kind") == "usable"}
        have = {s.shot_index for s in shots}
        # A planned shot with no row means the run was cut short; so does one the model never
        # managed to describe (it is tried again, not left as the file's history).
        retry = any("analysis_failed" in s.review_reasons for s in shots)
        return bool(shots) and planned <= have and not retry

    # -- stages -------------------------------------------------------------

    async def _fetch(self, discovered: DiscoveredFile) -> Path | None:
        """Drive-only sources are downloaded first; local and staged ones are ready."""
        if discovered.path and discovered.path.exists():
            return discovered.path
        if discovered.origin == "drive" and discovered.drive_file_id:
            from ..drive.fetcher import fetch_drive_file

            return await asyncio.to_thread(
                fetch_drive_file, self.config, discovered.drive_file_id, discovered.filename,
                None, discovered.size_bytes,
            )
        return None

    def _create_source(self, discovered, video: Path, digest: str, meta) -> Source:
        return self.store.insert_source(
            Source(
                id=new_id(),
                workspace_id=self.config.id,
                content_hash=digest,
                original_filename=discovered.filename,
                origin=discovered.origin,
                origin_path=discovered.origin_path or str(video),
                drive_file_id=discovered.drive_file_id,
                media_kind=meta.media_kind,
                duration_s=meta.duration_s,
                width=meta.width,
                height=meta.height,
                fps=meta.fps,
                codec=meta.codec,
                filesize_bytes=meta.filesize_bytes,
                status="analysing",
            )
        )

    async def _embed(self, shot_id: str) -> None:
        if self._embedder is None:
            return
        text = self.store.recompute_search_text(shot_id)
        if not text:
            return
        vector = await asyncio.to_thread(self._embedder.embed_documents, [text])
        self.store.vectors.upsert(shot_id, vector[0])

    def _cleanup_raw(self, raw_dir: Path | None) -> None:
        if raw_dir is not None and raw_dir.exists():
            for leftover in raw_dir.glob("*"):
                leftover.unlink(missing_ok=True)
            raw_dir.rmdir()

    def _cleanup(self, work_dir: Path | None, video, discovered: DiscoveredFile,
                 source_id: str | None) -> None:
        """Only ever delete inside the workspace temp and staging directories,
        and never delete the only copy of an uploaded file.

        A Drive-origin temp copy can always go (the original is in Drive). An
        upload's staged copy only goes once it has reached Drive; until then it
        is the footage, and a later `broll organise` needs it.
        """
        if work_dir is not None and work_dir.exists():
            for leftover in work_dir.glob("*"):
                leftover.unlink(missing_ok=True)
            work_dir.rmdir()
        if discovered.origin not in ("upload", "drive"):
            return
        if discovered.origin == "upload":
            source = self.store.get_source(source_id) if source_id else None
            if not (source and source.drive_file_id):
                return  # not in Drive yet - keep it
        if not isinstance(video, Path):
            return  # read in place from Drive: there is nothing of ours to delete
        staging = self.config.staging_dir.resolve()
        tmp = self.config.temp_dir.resolve()
        resolved = video.resolve()
        if any(resolved.is_relative_to(root) for root in (staging, tmp)):
            resolved.unlink(missing_ok=True)


def drive_organise_callable(config: WorkspaceConfig) -> Callable[[str], object] | None:
    """A thread-safe "file this source into Drive" function, or None.

    None when Drive is not connected or auto_organise is off. Each call opens
    its own Store, because it runs in a worker thread and a sqlite3 connection
    may not cross threads; the Drive client is shared behind a lock.
    """
    if not config.ingest.auto_organise or not config.drive_token_path.exists():
        return None

    from ..drive.session import DriveSession

    # One session for the whole run: a shared client and folder-id map, so
    # filing clip N does not re-walk the client's tree from scratch.
    return DriveSession(config).organise
