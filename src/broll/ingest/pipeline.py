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
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..analysis.analyzer import Analyzer
from ..analysis.embedder import Embedder
from ..analysis.prompt import PROMPT_VERSION
from ..analysis.schema import ShotContext
from ..config import WorkspaceConfig
from ..db.models import Shot, Source
from ..db.store import Store, new_id
from .frames import best_frame, save_thumbnail
from .hashing import content_hash
from .probe import NotAVideoError, ProbeResult, probe
from .scanner import DiscoveredFile
from .shots import detect_shots, primary_index

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


class IngestPipeline:
    def __init__(
        self,
        config: WorkspaceConfig,
        store: Store,
        analyzer: Analyzer | None = None,
        embedder: Embedder | None = None,
        organise: Callable[[str], object] | None = None,
    ):
        self.config = config
        self.store = store
        self.analyzer = analyzer or Analyzer(config)
        self._embedder = embedder
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
    ) -> IngestResult:
        result = IngestResult(filename=discovered.filename)

        video = await self._fetch(discovered)
        if video is None:
            result.status = "failed"
            result.error = "could not obtain the file's bytes"
            return result

        digest = await asyncio.to_thread(content_hash, video)
        existing = self.store.source_by_hash(digest)
        if existing and not force and self._is_complete(existing):
            result.source_id = existing.id
            result.status = existing.status
            result.deduped = True
            return result

        try:
            meta: ProbeResult = await asyncio.to_thread(probe, video)
        except NotAVideoError as exc:
            result.status = "failed"
            result.error = str(exc)
            return result

        source = existing or self._create_source(discovered, video, digest, meta)
        result.source_id = source.id
        self.store.update_source(
            source.id, status="analysing", analysis_version=PROMPT_VERSION, error_message=None
        )

        spans = await asyncio.to_thread(
            detect_shots,
            video,
            meta.duration_s,
            self.config.ingest.min_shot_length_s,
            self.config.ingest.min_average_shot_length_s,
        )
        primary = primary_index(spans)
        work_dir = self.config.temp_dir / f"src-{source.id[:8]}"

        for span in spans:
            shot_id = f"{source.id}-{span.index}"
            existing_shot = self.store.get_shot(shot_id)
            if existing_shot and existing_shot.status == "indexed" and not force:
                result.shots_skipped += 1
                continue
            if existing_shot and _is_corrected(existing_shot) and not overwrite_corrections:
                # A human already fixed this shot. Re-analysis must not quietly
                # throw that away; --overwrite-corrections is the explicit opt-in.
                result.shots_skipped += 1
                result.messages.append(
                    f"kept operator correction for shot {span.index} of {discovered.filename}"
                )
                continue

            shot = existing_shot or Shot(
                id=shot_id,
                workspace_id=self.config.id,
                source_id=source.id,
                shot_index=span.index,
            )
            shot.is_primary = span.index == primary
            shot.start_s = span.start_s
            shot.end_s = span.end_s
            shot.duration_s = span.duration_s

            context = ShotContext(
                source_filename=discovered.filename,
                duration_s=span.duration_s,
                width=meta.width,
                height=meta.height,
                fps=meta.fps,
                shot_index=span.index,
                shot_count=len(spans),
                start_s=span.start_s,
                end_s=span.end_s,
            )

            frames = await asyncio.to_thread(self.analyzer.extract, video, context, work_dir)
            outcome = await self.analyzer.analyse_frames(frames, context)
            result.cost_usd += outcome.cost_usd

            if outcome.result is not None:
                shot.apply_analysis(outcome.result, PROMPT_VERSION)
                shot.status = outcome.status
                shot.error_message = outcome.error
            else:
                shot.status = "needs_review"
                shot.error_message = outcome.error

            keyframe = await asyncio.to_thread(best_frame, frames)
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

            await self._embed(shot.id)
            result.shots_analysed += 1

            for frame in frames:
                frame.unlink(missing_ok=True)

        result.status = self.store.recompute_source_status(source.id)

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

    def _is_complete(self, source) -> bool:
        """Only a finished source short-circuits the pipeline.

        A source left mid-ingest by a killed worker is in `analysing` with some
        or none of its shots written; that must resume, not be skipped as a
        duplicate.
        """
        if source.status not in ("indexed", "needs_review"):
            return False
        return bool(self.store.shots_for_source(source.id))

    # -- stages -------------------------------------------------------------

    async def _fetch(self, discovered: DiscoveredFile) -> Path | None:
        """Drive-only sources are downloaded first; local and staged ones are ready."""
        if discovered.path and discovered.path.exists():
            return discovered.path
        if discovered.origin == "drive" and discovered.drive_file_id:
            from ..drive.fetcher import fetch_drive_file

            return await asyncio.to_thread(
                fetch_drive_file, self.config, discovered.drive_file_id, discovered.filename
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
        vector = await asyncio.to_thread(self._embedder.embed_one, text)
        self.store.vectors.upsert(shot_id, vector)

    def _cleanup(self, work_dir: Path, video: Path, discovered: DiscoveredFile,
                 source_id: str) -> None:
        """Only ever delete inside the workspace temp and staging directories,
        and never delete the only copy of an uploaded file.

        A Drive-origin temp copy can always go (the original is in Drive). An
        upload's staged copy only goes once it has reached Drive; until then it
        is the footage, and a later `broll organise` needs it.
        """
        if work_dir.exists():
            for leftover in work_dir.glob("*"):
                leftover.unlink(missing_ok=True)
            work_dir.rmdir()
        if discovered.origin not in ("upload", "drive"):
            return
        if discovered.origin == "upload":
            source = self.store.get_source(source_id)
            if not (source and source.drive_file_id):
                return  # not in Drive yet - keep it
        staging = self.config.staging_dir.resolve()
        tmp = self.config.temp_dir.resolve()
        resolved = video.resolve()
        if any(resolved.is_relative_to(root) for root in (staging, tmp)):
            resolved.unlink(missing_ok=True)


def drive_organise_callable(config: WorkspaceConfig) -> Callable[[str], object] | None:
    """A thread-safe "file this source into Drive" function, or None.

    None when Drive is not connected or auto_organise is off. Each call opens
    its own Store and Drive client, because it runs in a worker thread and
    neither a sqlite3 connection nor an httplib2 client may cross threads.
    """
    if not config.ingest.auto_organise or not config.drive_token_path.exists():
        return None

    def organise(source_id: str):
        from ..drive.auth import load_credentials
        from ..drive.client import DriveClient
        from ..drive.organizer import Organizer

        credentials = load_credentials(config)
        if credentials is None:
            return None
        store = Store.for_config(config)
        try:
            report = Organizer(config, store, DriveClient(credentials)).organise_source(source_id)
            if report.errors:
                raise RuntimeError("; ".join(report.errors))
            return report
        finally:
            store.close()

    return organise
