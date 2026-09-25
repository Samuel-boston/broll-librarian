"""Removing things: one file, the whole library, and files waiting in the queue.

Everything is local: nothing in Drive is ever touched. The dashboard's copy is kept in step.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from broll.db.models import Shot
from broll.db.store import Store, new_id
from broll.ingest.scanner import DiscoveredFile
from broll.jobs.queue import KIND_INDEX_SOURCE, enqueue_files, queue_stats
from broll.library_admin import cancel_job, clear_library, clear_queue, remove_source, running_jobs
from broll.sync.dashboard import KEY_ENV, DashboardSync
from broll.web.app import create_app
from tests.test_dashboard_sync import URL, FakeSupabase, add_shot, make_sync


@pytest.fixture()
def connected(workspace, monkeypatch):
    workspace.dashboard.enabled = True
    workspace.dashboard.supabase_url = URL
    workspace.save()
    monkeypatch.setenv(KEY_ENV, "service-key")
    return workspace


def _local(name: str, path: str) -> DiscoveredFile:
    from pathlib import Path

    return DiscoveredFile(origin="local", path=Path(path), filename=name, origin_path=path)


def _upload(workspace, name: str) -> DiscoveredFile:
    staged = workspace.staging_dir / name
    staged.write_bytes(b"clip bytes")
    return DiscoveredFile(origin="upload", path=staged, filename=name, origin_path=str(staged))


# ---- the queue ------------------------------------------------------------


def test_cancel_one_waiting_job(store, workspace):
    jobs = enqueue_files(store, [_local("a.mp4", "/x/a.mp4"), _local("b.mp4", "/x/b.mp4")])
    result = cancel_job(workspace, store, jobs[0].id)

    assert result.cancelled == 1 and result.names == ["a.mp4"]
    stats = queue_stats(store)
    assert (stats.queued, stats.cancelled) == (1, 1)
    # a cancelled job is never handed to a worker
    assert store.claim_job().id == jobs[1].id
    assert store.claim_job() is None


def test_a_running_job_cannot_be_cancelled(store, workspace):
    (job,) = enqueue_files(store, [_local("a.mp4", "/x/a.mp4")])
    store.claim_job()
    result = cancel_job(workspace, store, job.id)
    assert result.cancelled == 0 and result.still_running == 1
    assert queue_stats(store).running == 1


def test_cancelling_twice_or_a_missing_job_is_harmless(store, workspace):
    (job,) = enqueue_files(store, [_local("a.mp4", "/x/a.mp4")])
    assert cancel_job(workspace, store, job.id).cancelled == 1
    assert cancel_job(workspace, store, job.id).cancelled == 0
    assert cancel_job(workspace, store, "nope").cancelled == 0


def test_a_cancelled_file_can_be_queued_again(store, workspace):
    (job,) = enqueue_files(store, [_local("a.mp4", "/x/a.mp4")])
    cancel_job(workspace, store, job.id)
    again = enqueue_files(store, [_local("a.mp4", "/x/a.mp4")])
    assert len(again) == 1


def test_clear_queue_removes_waiting_jobs_and_keeps_history(store, workspace):
    jobs = enqueue_files(store, [_local(f"{n}.mp4", f"/x/{n}.mp4") for n in "abcd"])
    store.claim_job()  # a is now running
    store.finish_job(jobs[1].id, "done")  # b is finished history
    store.claim_job()  # c running too
    result = clear_queue(workspace, store)

    assert result.cancelled == 1  # only d was still waiting
    assert result.still_running == 2
    stats = queue_stats(store)
    assert (stats.queued, stats.running, stats.done, stats.cancelled) == (0, 2, 1, 1)


def test_cancelling_an_upload_deletes_its_staged_copy_but_never_a_folder_file(store, workspace, tmp_path):
    upload = _upload(workspace, "dropped.mp4")
    folder_file = tmp_path / "my-footage" / "keep.mp4"
    folder_file.parent.mkdir()
    folder_file.write_bytes(b"precious")
    enqueue_files(store, [upload, _local("keep.mp4", str(folder_file))])

    result = clear_queue(workspace, store)
    assert result.cancelled == 2 and result.staged == 1
    assert not upload.path.exists()
    assert folder_file.read_bytes() == b"precious"


def test_a_staged_path_outside_staging_is_never_deleted(store, workspace, tmp_path):
    outside = tmp_path / "elsewhere.mp4"
    outside.write_bytes(b"x")
    job = store.enqueue(KIND_INDEX_SOURCE, {"origin": "upload", "path": str(outside), "filename": "elsewhere.mp4"})
    cancel_job(workspace, store, job.id)
    assert outside.exists()


# ---- one file --------------------------------------------------------------


def _shot_count(store, source_id: str) -> int:
    return len(store.shots_for_source(source_id))


def test_remove_source_takes_out_shots_tags_vectors_search_and_thumbnails(store, workspace):
    first = add_shot(store, workspace, "a.mp4", caption="A calm beach at sunrise")
    second = add_shot(store, workspace, "b.mp4", caption="A busy street market")
    source_id = first.rsplit("-", 1)[0]
    store.set_shot_tags(first, ["sunrise", "beach"])
    store.vectors.upsert(first, [0.1] * 384)
    store.vectors.upsert(second, [0.2] * 384)
    store.record_shortcut(source_id, first, "Travel", "folder-1", "a.mp4", "sc-1", "drive-a.mp4")
    thumb = workspace.thumbnails_dir / f"{first}.jpg"
    assert thumb.exists()

    result = remove_source(workspace, store, source_id)

    assert result.sources == 1 and result.shots == 1 and result.filename == "a.mp4"
    assert store.get_source(source_id) is None
    assert store.get_shot(first) is None
    assert store.vectors.count() == 1
    assert store.shortcuts_for_source(source_id) == []
    assert not thumb.exists()
    # the search index followed
    hits = store.conn.execute("SELECT rowid FROM shots_fts WHERE shots_fts MATCH 'sunrise'").fetchall()
    assert hits == []
    # the other file is untouched
    assert store.get_shot(second) is not None
    assert (workspace.thumbnails_dir / f"{second}.jpg").exists()


def test_removing_a_file_that_is_not_there_returns_none(store, workspace):
    assert remove_source(workspace, store, "nope") is None


def test_a_removed_file_can_be_indexed_again(store, workspace):
    shot_id = add_shot(store, workspace, "a.mp4")
    source_id = shot_id.rsplit("-", 1)[0]
    remove_source(workspace, store, source_id)
    assert store.source_by_hash("a.mp4") is None  # so re-adding it is not skipped as a duplicate


# ---- everything ------------------------------------------------------------


def test_clear_library_empties_everything_and_leaves_no_files(store, workspace):
    add_shot(store, workspace, "a.mp4")
    add_shot(store, workspace, "b.mp4")
    (workspace.staging_dir / "left.mp4").write_bytes(b"x")
    enqueue_files(store, [_local("c.mp4", "/x/c.mp4")])

    result = clear_library(workspace, store)

    assert result.sources == 2 and result.shots == 2 and result.jobs == 1
    assert result.thumbnails == 2 and result.staged == 1
    assert store.list_sources() == [] and store.count_shots() == 0 and store.vectors.count() == 0
    assert queue_stats(store).total == 0
    assert not list(workspace.thumbnails_dir.glob("*.jpg")) and not list(workspace.staging_dir.glob("*"))


def test_running_jobs_reports_files_being_indexed(store):
    enqueue_files(store, [_local("a.mp4", "/x/a.mp4")])
    assert running_jobs(store) == 0
    store.claim_job()
    assert running_jobs(store) == 1


# ---- the dashboard follows -------------------------------------------------


def test_remove_source_also_removes_it_from_the_dashboard(connected, store):
    keep = add_shot(store, connected, "keep.mp4")
    gone = add_shot(store, connected, "gone.mp4")
    fake = FakeSupabase()
    make_sync(connected, fake).run()
    assert set(fake.rows) == {keep, gone}

    sync = make_sync(connected, fake)
    assert sync.remove([gone]) == 1
    assert set(fake.rows) == {keep}
    assert f"{gone}.jpg" not in fake.thumbs
    # the sync's memory forgot it too, so a later sync doesn't try again or resend it
    assert make_sync(connected, fake).run().removed == 0


def test_remove_all_takes_everything_this_library_sent(connected, store):
    add_shot(store, connected, "a.mp4")
    add_shot(store, connected, "b.mp4")
    fake = FakeSupabase()
    make_sync(connected, fake).run()
    assert len(fake.rows) == 2
    assert make_sync(connected, fake).remove(None) == 2
    assert fake.rows == {} and fake.thumbs == set()


def test_remove_only_touches_shots_this_library_sent(connected, store):
    fake = FakeSupabase()
    fake.rows["someone-elses-shot"] = {"id": "someone-elses-shot"}
    add_shot(store, connected, "a.mp4")
    make_sync(connected, fake).run()
    make_sync(connected, fake).remove(None)
    assert "someone-elses-shot" in fake.rows


def test_remove_is_a_no_op_when_not_connected(workspace):
    assert DashboardSync(workspace).remove(["x"]) == 0


def test_the_library_is_still_cleared_when_the_dashboard_cannot_be_reached(connected, store, monkeypatch):
    add_shot(store, connected, "a.mp4")
    make_sync(connected, FakeSupabase()).run()

    def unreachable(self, *args, **kwargs):
        raise httpx.ConnectError("no route")

    monkeypatch.setattr(httpx.Client, "delete", unreachable)
    result = clear_library(connected, store)
    assert result.sources == 1 and store.list_sources() == []
    assert result.dashboard_error and "dashboard" in result.dashboard_error.lower()


# ---- the web buttons -------------------------------------------------------


def _client(workspace) -> TestClient:
    return TestClient(create_app(workspace, run_worker=False))


def test_delete_button_on_a_card_removes_the_file(workspace, store):
    shot_id = add_shot(store, workspace, "a.mp4")
    source_id = shot_id.rsplit("-", 1)[0]
    store.close()
    with _client(workspace) as client:
        page = client.get("/library").text
        assert f"/sources/{source_id}/delete" in page and "Delete all" in page
        response = client.post(f"/sources/{source_id}/delete")
        assert response.status_code == 200 and response.text == ""
        assert client.post(f"/sources/{source_id}/delete").status_code == 404
    check = Store.for_config(workspace)
    try:
        assert check.get_source(source_id) is None
    finally:
        check.close()


def test_delete_all_needs_the_word_delete(workspace, store):
    add_shot(store, workspace, "a.mp4")
    store.close()
    with _client(workspace) as client:
        wrong = client.post("/library/delete-all", headers={"HX-Prompt": "yes"})
        assert wrong.headers["HX-Reswap"] == "none" and "DELETE" in wrong.headers["HX-Trigger"]
        none = client.post("/library/delete-all")
        assert none.headers["HX-Reswap"] == "none"
        assert Store.for_config(workspace).count_shots() == 1

        right = client.post("/library/delete-all", headers={"HX-Prompt": "DELETE"})
        assert right.headers["HX-Redirect"] == "/library"
    check = Store.for_config(workspace)
    try:
        assert check.count_shots() == 0 and check.list_sources() == []
    finally:
        check.close()


def test_delete_all_refuses_while_a_file_is_being_indexed(workspace, store):
    add_shot(store, workspace, "a.mp4")
    enqueue_files(store, [_local("b.mp4", "/x/b.mp4")])
    store.claim_job()
    store.close()
    with _client(workspace) as client:
        response = client.post("/library/delete-all", headers={"HX-Prompt": "DELETE"})
        assert response.headers["HX-Reswap"] == "none" and "indexed right now" in response.headers["HX-Trigger"]
    check = Store.for_config(workspace)
    try:
        assert check.count_shots() == 1
    finally:
        check.close()


def test_queue_buttons(workspace, store):
    jobs = enqueue_files(store, [_local("a.mp4", "/x/a.mp4"), _local("b.mp4", "/x/b.mp4"), _local("c.mp4", "/x/c.mp4")])
    store.close()
    with _client(workspace) as client:
        html = client.get("/ingest/queue").text
        assert f"/ingest/jobs/{jobs[0].id}/cancel" in html and "Clear the queue (3 waiting)" in html

        one = client.post(f"/ingest/jobs/{jobs[0].id}/cancel").text
        assert "Removed a.mp4 from the queue" in one and "Clear the queue (2 waiting)" in one

        cleared = client.post("/ingest/clear").text
        assert "Removed 2 file(s) from the queue" in cleared
        assert "Clear the queue" not in cleared
        assert client.post("/ingest/clear").text.count("Nothing was waiting") == 1
