"""Two workers on one queue must not steal each other's jobs.

Regression, found on the first real client run: the web app (started from the
Desktop launcher) and a CLI command shared the queue. A worker starting up
requeued *every* "running" job as if it had been orphaned by a crash -
including one a live worker was in the middle of - so the job ran again with
stale code, overwrote the source's prompt version and left it stuck at
"analysing".
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys

import pytest

from broll.analysis.analyzer import Analyzer
from broll.analysis.providers.base import TransientProviderError
from broll.ingest.pipeline import IngestPipeline
from broll.ingest.scanner import DiscoveredFile


def _dead_pid() -> int:
    """A pid that certainly belonged to a process that has now exited."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def test_a_starting_worker_leaves_a_live_workers_job_alone(store):
    job = store.enqueue("index_source", {"filename": "a.mp4"})
    live_sibling = f"{socket.gethostname()}:{os.getppid()}"  # our parent: alive
    store.claim_job(worker_id=live_sibling)

    assert store.reset_stale_jobs() == 0
    assert store.get_job(job.id).status == "running"


def test_a_dead_workers_job_is_requeued(store):
    job = store.enqueue("index_source", {"filename": "a.mp4"})
    store.claim_job(worker_id=f"{socket.gethostname()}:{_dead_pid()}")

    assert store.reset_stale_jobs() == 1
    assert store.get_job(job.id).status == "queued"


def test_a_running_job_with_no_recorded_owner_is_requeued(store):
    """Rows from before ownership was recorded, or a hard kill mid-claim."""
    job = store.enqueue("index_source", {"filename": "a.mp4"})
    store.conn.execute("UPDATE jobs SET status = 'running' WHERE id = ?", (job.id,))

    assert store.reset_stale_jobs() == 1


def test_a_claim_records_this_process_as_the_owner(store):
    store.enqueue("index_source", {"filename": "a.mp4"})
    job = store.claim_job()
    assert job.claimed_by == f"{socket.gethostname()}:{os.getpid()}"


async def test_a_failed_reanalysis_leaves_the_source_status_consistent(workspace, store, clips):
    """A transient failure mid-source must not strand it at "analysing"."""
    clip = clips["single_static_bars"]
    discovered = DiscoveredFile(origin="local", path=clip, filename=clip.name,
                                origin_path=str(clip))
    first = await IngestPipeline(workspace, store).ingest(discovered)
    assert first.status == "indexed"

    class Overloaded:
        name = "overloaded"

        async def analyse(self, frames, context, retry_error=None):
            raise TransientProviderError("gemini request failed: 503 UNAVAILABLE")

        def estimate_cost(self, frames):
            return 0.0

    pipeline = IngestPipeline(workspace, store, analyzer=Analyzer(workspace, provider=Overloaded()))
    with pytest.raises(TransientProviderError):
        await pipeline.ingest(discovered, force=True)

    # Its shot is still the good, indexed one from the first run.
    assert store.get_source(first.source_id).status == "indexed"
