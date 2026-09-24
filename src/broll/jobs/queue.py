"""The SQLite-backed job queue.

The queue lives in the workspace database rather than in memory so that a run
killed halfway through resumes from the same place, with no duplicates: jobs are
claimed atomically, and the pipeline itself skips shots that are already
indexed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..db.models import Job
from ..db.store import Store
from ..ingest.scanner import DiscoveredFile

KIND_INDEX_SOURCE = "index_source"
KIND_REEMBED = "reembed_shot"

# A file the model cannot read is worth three tries. A model that answers 503
# is worth waiting out: the spike lasts minutes, and the old budget - three
# attempts, 5s then 10s apart - gave up on it inside twenty seconds.
MAX_ATTEMPTS = 3
MAX_TRANSIENT_ATTEMPTS = 8
BACKOFF_BASE_S = 5.0
TRANSIENT_BACKOFF_BASE_S = 20.0
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


def max_attempts(transient: bool) -> int:
    return MAX_TRANSIENT_ATTEMPTS if transient else MAX_ATTEMPTS


def backoff_delay(attempts: int, transient: bool = False) -> float:
    """Exponential, with jitter so parallel workers do not retry in lockstep."""
    base = TRANSIENT_BACKOFF_BASE_S if transient else BACKOFF_BASE_S
    delay = min(base * (2 ** max(0, attempts - 1)), BACKOFF_CAP_S)
    return delay * random.uniform(0.8, 1.2)


def queue_stats(store: Store) -> QueueStats:
    counts = store.job_counts()
    return QueueStats(
        queued=counts.get("queued", 0),
        running=counts.get("running", 0),
        done=counts.get("done", 0),
        failed=counts.get("failed", 0),
        cancelled=counts.get("cancelled", 0),
    )
