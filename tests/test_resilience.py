"""What happens when the provider is busy, and how a run is cleared down.

Hundreds of files through a shared model means transient failures are normal,
not exceptional. The queue has to outlast them rather than give up in seconds.
"""

from __future__ import annotations

import asyncio

import pytest

from broll.analysis.limiter import RateLimiter
from broll.analysis.providers.base import TransientProviderError
from broll.db.models import Shot, Source
from broll.db.store import new_id
from broll.jobs.queue import (
    KIND_INDEX_SOURCE,
    MAX_ATTEMPTS,
    MAX_TRANSIENT_ATTEMPTS,
    backoff_delay,
    max_attempts,
    queue_stats,
)


def test_a_503_run_is_waited_out_not_given_up_on():
    """Old policy: 3 tries, 5s then 10s apart - dead inside twenty seconds."""
    transient_budget = sum(backoff_delay(n, transient=True)
                           for n in range(1, MAX_TRANSIENT_ATTEMPTS))
    assert transient_budget > 15 * 60
    assert max_attempts(True) == MAX_TRANSIENT_ATTEMPTS
    assert max_attempts(False) == MAX_ATTEMPTS


async def test_a_busy_model_keeps_its_job_queued(workspace, store, tmp_path):
    """A 503 requeues; only a real fault burns the short budget."""
    from broll.ingest.pipeline import IngestPipeline
    from broll.jobs.worker import Worker

    clip = tmp_path / "a.mp4"
    clip.write_bytes(b"not really a video")
    store.enqueue(KIND_INDEX_SOURCE, {"origin": "local", "path": str(clip),
                                      "filename": "a.mp4", "origin_path": str(clip)})

    class Busy(IngestPipeline):
        async def ingest(self, discovered, force=False, overwrite_corrections=False):
            raise TransientProviderError("gemini request failed: 503 UNAVAILABLE")

    worker = Worker(workspace, store, pipeline=Busy(workspace, store))
    await worker.run(drain=True)

    stats = queue_stats(store)
    assert stats.failed == 0, "a busy model must not fail the job"
    assert stats.queued == 1
    job = store.conn.execute("SELECT attempts, not_before FROM jobs").fetchone()
    assert job["attempts"] == 1
    assert job["not_before"], "it waits before trying again"


async def test_the_request_cap_spaces_calls_out():
    limiter = RateLimiter(per_minute=120)   # one every 0.5s
    loop = asyncio.get_running_loop()
    start = loop.time()
    await asyncio.gather(*(limiter.acquire() for _ in range(3)))
    assert loop.time() - start >= 1.0      # 0s, 0.5s, 1.0s

    assert await RateLimiter(per_minute=0).acquire() == 0.0


def test_failed_jobs_can_be_put_back_in_the_queue(store):
    job = store.enqueue(KIND_INDEX_SOURCE, {"filename": "a.mp4"})
    store.finish_job(job.id, "failed", error="503 UNAVAILABLE")
    assert queue_stats(store).failed == 1

    assert store.requeue_failed_jobs() == 1
    assert queue_stats(store).failed == 0
    assert queue_stats(store).queued == 1
    assert store.requeue_failed_jobs() == 0


def test_clearing_the_library_empties_everything_local(workspace, store, tmp_path):
    source = store.insert_source(Source(
        id=new_id(), workspace_id=workspace.id, content_hash="h1",
        original_filename="a.mp4", origin="local", origin_path=str(tmp_path / "a.mp4")))
    store.insert_shot(Shot(id=f"{source.id}-0", workspace_id=workspace.id,
                           source_id=source.id, caption="A clip.", status="indexed"))
    store.vectors.upsert(f"{source.id}-0", [0.1] * workspace.embedder.dimensions)
    store.enqueue(KIND_INDEX_SOURCE, {"filename": "a.mp4"})

    counts = store.clear_library()

    assert counts["sources"] == 1 and counts["shots"] == 1 and counts["vectors"] == 1
    assert store.list_sources() == []
    assert store.list_shots() == []
    assert store.vectors.count() == 0
    assert queue_stats(store).total == 0
