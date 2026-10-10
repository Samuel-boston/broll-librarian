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


def test_a_gemini_request_that_never_answers_is_abandoned_and_retried(monkeypatch):
    """Caught live: a free-key request hung for minutes with the socket open, and with
    the always-on server indexing one file at a time nothing else could be indexed."""
    pytest.importorskip("google.genai")
    from google import genai

    from broll.analysis.providers import gemini
    from broll.analysis.providers.base import TransientProviderError, classify_error

    made = {}

    class Client:
        def __init__(self, **kwargs):
            made.update(kwargs)

    monkeypatch.setattr(genai, "Client", Client)
    gemini._client("a-key")
    assert made["http_options"].timeout == gemini.REQUEST_TIMEOUT_MS
    assert gemini.REQUEST_TIMEOUT_MS <= 300_000
    assert classify_error("gemini request failed: The read operation timed out") is TransientProviderError


async def test_a_dead_sign_in_pauses_the_queue_instead_of_failing_every_file(workspace, store, tmp_path):
    """invalid_grant (an expired Google sign-in) hits every file the same way. None may be marked failed,
    none may lose an attempt, and nothing lands on Needs attention for it."""
    from broll.ingest.pipeline import IngestPipeline
    from broll.jobs.worker import Worker

    for n in range(5):
        clip = tmp_path / f"{n}.mp4"
        clip.write_bytes(b"x")
        store.enqueue(KIND_INDEX_SOURCE, {"origin": "local", "path": str(clip), "filename": clip.name,
                                          "origin_path": str(clip)})
    tried = []

    class DeadSignIn(IngestPipeline):
        async def ingest(self, discovered, force=False, overwrite_corrections=False, **kw):
            tried.append(discovered.filename)
            raise RuntimeError("invalid_grant: Token has been expired or revoked.")

    worker = Worker(workspace, store, pipeline=DeadSignIn(workspace, store), concurrency=1)
    await worker.run(drain=True)

    stats = queue_stats(store)
    assert stats.failed == 0 and stats.queued == 5 and len(tried) == 1, "it stopped after the first"
    assert worker.pause_reason and "invalid_grant" in worker.pause_reason
    assert store.conn.execute("SELECT MAX(attempts) FROM jobs").fetchone()[0] == 0
    assert store.list_attention("open") == []


def test_environment_errors_are_told_apart_from_a_bad_file():
    from broll.analysis.providers.base import (
        ProviderError, TransientProviderError, classify_error, is_environment_error)

    for message in ("invalid_grant: Bad Request", "API key not valid. Please pass a valid API key.",
                    "429 RESOURCE_EXHAUSTED: Quota exceeded ... per day"):
        assert is_environment_error(message) and classify_error(message) is TransientProviderError
    assert not is_environment_error("ffprobe could not read a.mp4: Invalid data")
    assert classify_error("ffprobe could not read a.mp4: Invalid data") is ProviderError


async def test_a_failed_attempt_leaves_no_download_behind(workspace, store, tmp_path, monkeypatch):
    from broll.ingest.pipeline import IngestPipeline
    from broll.ingest.scanner import DiscoveredFile

    pipeline = IngestPipeline(workspace, store)
    downloaded = workspace.temp_dir / "drive-abc.mp4"
    downloaded.parent.mkdir(parents=True, exist_ok=True)
    downloaded.write_bytes(b"x" * 100)

    async def fetch(_self, _discovered):
        return downloaded

    async def broken(*args, **kwargs):
        raise TransientProviderError("gemini request failed: 503 UNAVAILABLE")

    monkeypatch.setattr(IngestPipeline, "_fetch", fetch)
    monkeypatch.setattr(IngestPipeline, "_index", broken)
    with pytest.raises(TransientProviderError):
        await pipeline.ingest(DiscoveredFile(origin="drive", path=None, filename="abc.mp4",
                                             drive_file_id="abc", origin_path="drive:abc", size_bytes=100))
    assert not downloaded.exists(), "the Drive download must not stay on the disk after a failed attempt"


async def test_an_embedding_failure_is_retried_not_swallowed(workspace, store):
    from broll.ingest.pipeline import IngestPipeline
    from tests.test_precision import add_shot

    class Quota:
        name = "fake"
        dimensions = 4

        def embed_documents(self, texts):
            raise RuntimeError("429 RESOURCE_EXHAUSTED: slow down")

    shot = add_shot(store, "a.mp4", caption="A man walks on a beach.")
    pipeline = IngestPipeline(workspace, store, embedder=Quota())
    with pytest.raises(TransientProviderError):
        await pipeline._embed(shot)
