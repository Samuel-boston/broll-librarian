"""Removing things: one file, the whole library, or jobs waiting in the queue.

All of this is local. Nothing in Drive is ever deleted: files already filed there, and the
shortcuts the organiser made, stay exactly where they are. The dashboard's Footage index is
kept in step (best effort), so a clip removed here does not linger there.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import WorkspaceConfig
from .db.models import Job
from .db.store import Store
from .sync.dashboard import DashboardSync, DashboardSyncError, is_connected

log = logging.getLogger(__name__)


@dataclass
class RemoveResult:
    sources: int = 0
    shots: int = 0
    jobs: int = 0
    thumbnails: int = 0
    staged: int = 0
    dashboard_removed: int = 0
    #: Set when the library was cleared but the dashboard couldn't be told; the next sync fixes a single file.
    dashboard_error: str | None = None
    filename: str | None = None


@dataclass
class CancelResult:
    cancelled: int = 0
    #: Jobs a worker has already started. They finish; they can't be stopped half way.
    still_running: int = 0
    staged: int = 0
    names: list[str] = field(default_factory=list)


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except (ValueError, OSError):
        return False


def _unlink_inside(path: str | Path | None, directory: Path) -> bool:
    """Delete a file only if it lives inside `directory`. Nothing else is ever removed."""
    if not path:
        return False
    candidate = Path(path)
    if candidate.is_file() and _inside(candidate, directory):
        candidate.unlink(missing_ok=True)
        return True
    return False


def _tell_dashboard(config: WorkspaceConfig, shot_ids: list[str] | None, result: RemoveResult) -> None:
    if not is_connected(config):
        return
    try:
        result.dashboard_removed = DashboardSync(config).remove(shot_ids)
    except DashboardSyncError as exc:
        result.dashboard_error = str(exc)
        log.warning("dashboard not updated: %s", exc)


def remove_source(config: WorkspaceConfig, store: Store, source_id: str) -> RemoveResult | None:
    """Remove one file (and all its shots) from the library. None if there is no such file."""
    gone = store.delete_source(source_id)
    if gone is None:
        return None
    result = RemoveResult(sources=1, shots=len(gone["shot_ids"]), filename=gone["source"].original_filename)
    for shot_id, path in gone["thumbnails"].items():
        if _unlink_inside(path, config.thumbnails_dir) or _unlink_inside(config.thumbnails_dir / f"{shot_id}.jpg", config.thumbnails_dir):
            result.thumbnails += 1
    _tell_dashboard(config, gone["shot_ids"], result)
    return result


def running_jobs(store: Store) -> int:
    return store.job_counts().get("running", 0)


def clear_library(config: WorkspaceConfig, store: Store) -> RemoveResult:
    """Empty the library: every file, shot, job, vector, thumbnail and staged upload. Drive is untouched."""
    counts = store.clear_library()
    result = RemoveResult(sources=counts["sources"], shots=counts["shots"], jobs=counts["jobs"])
    for thumbnail in config.thumbnails_dir.glob("*.jpg"):
        thumbnail.unlink(missing_ok=True)
        result.thumbnails += 1
    for staged in config.staging_dir.glob("*"):
        if staged.is_file():
            staged.unlink(missing_ok=True)
            result.staged += 1
    _tell_dashboard(config, None, result)
    return result


def _discard_staged(config: WorkspaceConfig, job: Job) -> bool:
    """A cancelled upload leaves its copy in the staging folder; a folder being indexed in place is never touched."""
    if job.payload.get("origin") != "upload":
        return False
    return _unlink_inside(job.payload.get("path"), config.staging_dir)


def cancel_job(config: WorkspaceConfig, store: Store, job_id: str) -> CancelResult:
    job = store.cancel_job(job_id)
    result = CancelResult()
    if job is None:
        current = store.get_job(job_id)
        result.still_running = 1 if current is not None and current.status == "running" else 0
        return result
    result.cancelled = 1
    result.names.append(job.payload.get("filename") or job.id[:8])
    result.staged = int(_discard_staged(config, job))
    return result


def clear_queue(config: WorkspaceConfig, store: Store) -> CancelResult:
    """Remove every waiting job from the queue. Running jobs finish; done and failed history is kept."""
    still_running = running_jobs(store)
    jobs = store.cancel_queued_jobs()
    result = CancelResult(cancelled=len(jobs), still_running=still_running)
    for job in jobs:
        result.names.append(job.payload.get("filename") or job.id[:8])
        result.staged += int(_discard_staged(config, job))
    return result
