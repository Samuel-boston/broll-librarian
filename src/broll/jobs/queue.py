"""The SQLite-backed job queue.

The queue lives in the workspace database rather than in memory so that a run
killed halfway through resumes from the same place, with no duplicates: jobs are
claimed atomically, and the pipeline itself skips shots that are already
indexed.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..db.models import Job
from ..db.store import Store
from ..ingest.scanner import DiscoveredFile

KIND_INDEX_SOURCE = "index_source"
KIND_REEMBED = "reembed_shot"

MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 5.0
BACKOFF_CAP_S = 900.0


@dataclass
class QueueStats:
    queued: int = 0
    running: int = 0
    done: int = 0
    failed: int = 0
    cancelled: int = 0

    @property
    def outstanding(self) -> int:
        return self.queued + self.running

    @property
    def total(self) -> int:
        return self.queued + self.running + self.done + self.failed + self.cancelled


def enqueue_files(
    store: Store,
    files: list[DiscoveredFile],
    force: bool = False,
    overwrite_corrections: bool = False,
) -> list[Job]:
    """Queue one job per discovered file, skipping files already queued."""
    pending = _pending_paths(store)
    jobs: list[Job] = []
    for discovered in files:
        key = discovered.origin_path or discovered.filename
        if key in pending:
            continue
        payload = discovered.payload()
        payload["force"] = force
        payload["overwrite_corrections"] = overwrite_corrections
        jobs.append(store.enqueue(KIND_INDEX_SOURCE, payload))
        pending.add(key)
    return jobs


def _pending_paths(store: Store) -> set[str]:
    rows = store.conn.execute(
        """SELECT json_extract(payload_json, '$.origin_path') AS p,
                  json_extract(payload_json, '$.filename') AS f
           FROM jobs
           WHERE workspace_id = ? AND kind = ? AND status IN ('queued','running')""",
        (store.workspace_id, KIND_INDEX_SOURCE),
    ).fetchall()
    return {(r["p"] or r["f"]) for r in rows if (r["p"] or r["f"])}


def backoff_delay(attempts: int) -> float:
    return min(BACKOFF_BASE_S * (2 ** max(0, attempts - 1)), BACKOFF_CAP_S)


def queue_stats(store: Store) -> QueueStats:
    counts = store.job_counts()
    return QueueStats(
        queued=counts.get("queued", 0),
        running=counts.get("running", 0),
        done=counts.get("done", 0),
        failed=counts.get("failed", 0),
        cancelled=counts.get("cancelled", 0),
    )
