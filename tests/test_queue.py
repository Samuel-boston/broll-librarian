"""Job queue and worker: resumability is the point of this module.

The M2 acceptance criterion is that killing the worker mid-run and restarting
completes the remaining files with no duplicates.
"""

from __future__ import annotations

import asyncio

import pytest

from broll.ingest.pipeline import IngestPipeline
from broll.ingest.scanner import scan_local
from broll.jobs.queue import backoff_delay, enqueue_files, queue_stats
from broll.jobs.worker import Worker
from tests.conftest import CLIP_DIR


def _files():
    return scan_local(CLIP_DIR)


def _worker(workspace, store, stop_after: int | None = None, concurrency: int = 2):
    pipeline = IngestPipeline(workspace, store)
    finished = {"n": 0}

    def on_progress(event, job, result):
        if event == "finished":
            finished["n"] += 1
            if stop_after and finished["n"] >= stop_after:
                worker.stop()

    worker = Worker(workspace, store, pipeline, concurrency, on_progress)
    return worker


def test_enqueue_is_idempotent(store, workspace):
    files = _files()
    first = enqueue_files(store, files)
    second = enqueue_files(store, files)
    assert len(first) == len(files)
    assert second == []  # already queued, not queued twice
    assert queue_stats(store).queued == len(files)


def test_worker_drains_the_queue(store, workspace):
    files = _files()
    enqueue_files(store, files)
    stats = asyncio.run(_worker(workspace, store).run())

    assert stats.failed == 0
    final = queue_stats(store)
    assert final.queued == 0 and final.running == 0
    assert final.done == len(files)
    assert len(store.list_sources()) == len(files)


def test_multi_shot_file_produces_one_shot_row_per_shot(store, workspace):
    multi = [f for f in _files() if f.filename == "multi_shot_three_cuts.mp4"]
    single = [f for f in _files() if f.filename == "single_static_bars.mp4"]
    enqueue_files(store, multi + single)
    asyncio.run(_worker(workspace, store, concurrency=1).run())

    sources = {s.original_filename: s for s in store.list_sources()}
    multi_shots = store.shots_for_source(sources["multi_shot_three_cuts.mp4"].id)
    single_shots = store.shots_for_source(sources["single_static_bars.mp4"].id)

    assert len(multi_shots) == 3
    assert len(single_shots) == 1
    assert sum(1 for s in multi_shots if s.is_primary) == 1
    assert [s.shot_index for s in multi_shots] == [0, 1, 2]
    assert multi_shots[0].start_s == 0.0
    assert multi_shots[-1].end_s >= 5.9


def test_killed_worker_resumes_with_no_duplicates(store, workspace):
    """The M2 acceptance criterion."""
    files = _files()
    enqueue_files(store, files)

    # First run: stopped after two files, as if the process were killed.
    first = asyncio.run(_worker(workspace, store, stop_after=2).run())
    assert first.done >= 2
    mid = queue_stats(store)
    assert mid.outstanding > 0, "the run should not have drained the queue"

    # Simulate a hard kill: a job left mid-flight stays 'running' on disk.
    store.conn.execute(
        "UPDATE jobs SET status = 'running' WHERE workspace_id = ? AND status = 'queued'"
        " AND id = (SELECT id FROM jobs WHERE workspace_id = ? AND status = 'queued' LIMIT 1)",
        (store.workspace_id, store.workspace_id),
    )

    # Second run: a fresh worker picks up everything that is left.
    asyncio.run(_worker(workspace, store).run())

    final = queue_stats(store)
    assert final.outstanding == 0
    assert final.done == len(files)
    assert final.failed == 0

    sources = store.list_sources()
    assert len(sources) == len(files)
    hashes = [s.content_hash for s in sources]
    assert len(set(hashes)) == len(hashes), "a source was ingested twice"

    shot_ids = [s.id for s in store.list_shots()]
    assert len(set(shot_ids)) == len(shot_ids)
    assert all(s.status in ("indexed", "needs_review") for s in store.list_shots())

    # A third run must be a no-op.
    third = asyncio.run(_worker(workspace, store).run())
    assert third.done == 0 and third.shots == 0


def test_rerunning_an_indexed_file_skips_analysed_shots(store, workspace):
    files = [f for f in _files() if f.filename == "multi_shot_three_cuts.mp4"]
    enqueue_files(store, files)
    asyncio.run(_worker(workspace, store).run())

    # Queue it again: dedupe on content hash means no second source.
    enqueue_files(store, files)
    asyncio.run(_worker(workspace, store).run())
    assert len(store.list_sources()) == 1
    assert len(store.list_shots()) == 3


def test_failed_jobs_retry_with_backoff_then_give_up(store, workspace, monkeypatch):
    files = _files()[:1]
    enqueue_files(store, files)

    async def boom(self, discovered, force=False, overwrite_corrections=False):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(IngestPipeline, "ingest", boom)

    for expected_attempt in (1, 2, 3):
        # not_before backoff would otherwise hold the job back
        store.conn.execute(
            "UPDATE jobs SET not_before = NULL WHERE workspace_id = ?", (store.workspace_id,)
        )
        asyncio.run(_worker(workspace, store).run())
        job = store.conn.execute(
            "SELECT attempts, status, last_error FROM jobs WHERE workspace_id = ?",
            (store.workspace_id,),
        ).fetchone()
        assert job["attempts"] == expected_attempt
        assert "provider exploded" in job["last_error"]

    assert job["status"] == "failed"
    assert queue_stats(store).failed == 1


@pytest.mark.parametrize(
    "attempts,expected", [(1, 5.0), (2, 10.0), (3, 20.0), (12, 900.0)]
)
def test_backoff_is_exponential_and_capped(attempts, expected):
    assert backoff_delay(attempts) == expected


def test_a_source_left_mid_ingest_resumes_rather_than_deduping(store, workspace):
    """A worker killed inside a file leaves a source in 'analysing'.

    Dedupe must not treat that as an already-indexed file and skip it, or the
    shots are never written and the source is stuck forever.
    """
    from broll.db.models import Source
    from broll.db.store import new_id
    from broll.ingest.hashing import content_hash

    multi = [f for f in _files() if f.filename == "multi_shot_three_cuts.mp4"]
    digest = content_hash(multi[0].path)
    store.insert_source(
        Source(
            id=new_id(), workspace_id=workspace.id, content_hash=digest,
            original_filename=multi[0].filename, origin="local",
            origin_path=str(multi[0].path), status="analysing",
        )
    )

    enqueue_files(store, multi)
    asyncio.run(_worker(workspace, store).run())

    sources = store.list_sources()
    assert len(sources) == 1, "the half-finished source must be reused, not duplicated"
    assert sources[0].status == "indexed"
    assert len(store.shots_for_source(sources[0].id)) == 3


def test_a_partially_analysed_source_only_finishes_the_missing_shots(store, workspace):
    multi = [f for f in _files() if f.filename == "multi_shot_three_cuts.mp4"]
    enqueue_files(store, multi)
    asyncio.run(_worker(workspace, store).run())
    source = store.list_sources()[0]

    # Throw away the last shot and put the source back into 'analysing' -
    # exactly the state a worker killed mid-file leaves behind.
    store.conn.execute(
        "DELETE FROM shots WHERE workspace_id = ? AND source_id = ? AND shot_index = 2",
        (store.workspace_id, source.id),
    )
    store.update_source(source.id, status="analysing")

    enqueue_files(store, multi)
    stats = asyncio.run(_worker(workspace, store).run())

    assert stats.shots == 1, "only the missing shot should be re-analysed"
    assert len(store.shots_for_source(source.id)) == 3


def test_reanalysis_keeps_operator_corrections_unless_told_otherwise(store, workspace):
    """A human correction is ground truth; re-analysis must not silently undo it."""
    from broll.review import apply_correction

    files = [f for f in _files() if f.filename == "single_static_bars.mp4"]
    enqueue_files(store, files)
    asyncio.run(_worker(workspace, store).run())

    shot = store.list_shots()[0]
    apply_correction(workspace, store, shot.id,
                     {"caption": "SMPTE colour bars.", "setting": "studio"})

    enqueue_files(store, files, force=True)
    asyncio.run(_worker(workspace, store).run())
    assert store.get_shot(shot.id).caption == "SMPTE colour bars."

    enqueue_files(store, files, force=True, overwrite_corrections=True)
    asyncio.run(_worker(workspace, store).run())
    assert store.get_shot(shot.id).caption != "SMPTE colour bars."
