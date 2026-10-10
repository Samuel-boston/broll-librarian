"""The Needs attention page, the folder suggestions on Review, and their API."""

from __future__ import annotations

from fastapi.testclient import TestClient

from broll import attention
from broll.db.store import Store
from broll.ingest.scanner import DiscoveredFile
from broll.web.app import create_app
from tests.test_client_tree import tree_config

GB = 1_000_000_000


def _client(workspace) -> TestClient:
    return TestClient(create_app(workspace, run_worker=False))


def _flag(workspace, name="podcast.mp4", kind="too_long", link="https://drive.google.com/file/d/1/view"):
    store = Store.for_config(workspace)
    try:
        return attention.flag_file(
            store,
            DiscoveredFile(origin="drive", path=None, filename=name, drive_file_id=f"id-{name}",
                           origin_path=f"drive:id-{name}", link=link, size_bytes=30 * GB, duration_s=5400),
            kind, "1 h 30 min long; the limit is 10 min 00 s.",
        )
    finally:
        store.close()


def test_the_page_says_so_when_nothing_needs_attention(workspace):
    html = _client(workspace).get("/attention").text
    assert "No videos need attention" in html


def test_each_file_is_listed_with_its_reason_and_a_link_to_open_it(workspace):
    _flag(workspace)
    html = _client(workspace).get("/attention").text
    assert "Too long to be B-roll" in html and "podcast.mp4" in html
    assert "1 h 30 min" in html and "30.0 GB" in html
    assert 'href="https://drive.google.com/file/d/1/view"' in html and "Index anyway" in html


def test_a_file_too_big_for_the_disk_does_not_offer_to_force_it(workspace):
    _flag(workspace, name="giant.mp4", kind="too_big")
    html = _client(workspace).get("/attention").text
    assert "giant.mp4" in html and "Index anyway" not in html and "Try again" not in html


def test_the_nav_badge_counts_what_is_waiting(workspace):
    client = _client(workspace)
    assert client.get("/attention/badge").text == "Needs attention"
    _flag(workspace)
    assert "tag-warn" in client.get("/attention/badge").text and ">1<" in client.get("/attention/badge").text


def test_index_anyway_queues_the_file_with_the_limit_lifted(workspace):
    item = _flag(workspace)
    client = _client(workspace)
    page = client.post(f"/attention/{item}/index").text
    assert "Queued podcast.mp4" in page
    store = Store.for_config(workspace)
    try:
        job = store.claim_job()
        assert job.payload["allow_long"] is True and job.payload["drive_file_id"] == "id-podcast.mp4"
        assert store.get_attention(item)["status"] == "requeued"
    finally:
        store.close()


def test_dismissing_removes_it_from_the_page(workspace):
    item = _flag(workspace)
    client = _client(workspace)
    assert "Dismissed" in client.post(f"/attention/{item}/dismiss").text
    assert "podcast.mp4" not in client.get("/attention").text


def test_the_api_lists_and_acts_on_the_same_items(workspace):
    item = _flag(workspace)
    client = _client(workspace)
    body = client.get("/api/attention").json()
    assert body["count"] == 1 and body["items"][0]["filename"] == "podcast.mp4"
    assert body["items"][0]["link"].startswith("https://drive.google.com")
    assert client.get("/api/status").json()["needs_attention"] == 1
    assert client.post(f"/api/attention/{item}/dismiss").json() == {"dismissed": True, "id": item}
    assert client.get("/api/attention").json()["count"] == 0
    assert client.post("/api/attention/999/index").status_code == 404


def test_suggested_folders_appear_on_the_review_page_and_can_be_approved(workspace):
    workspace.taxonomy = tree_config()
    workspace.save()
    from tests.test_precision import add_shot

    store = Store.for_config(workspace)
    try:
        shot = add_shot(store, "w.mp4", caption="A man wakeboards.", category="05_Travel & Adventure/Beach & Water")
        store.record_folder_proposal("05_Travel & Adventure/Wakeboarding", "Towed water sports.", shot)
        proposal = store.list_folder_proposals()[0]["id"]
    finally:
        store.close()

    client = _client(workspace)
    html = client.get("/review").text
    assert "Suggested folders" in html and "Wakeboarding" in html and "Towed water sports." in html

    after = client.post(f"/review/folders/{proposal}/approve").text
    assert "Created 05_Travel &amp; Adventure/Wakeboarding and moved 1 clip(s)" in after
    store = Store.for_config(workspace)
    try:
        assert store.get_shot(shot).category == "05_Travel & Adventure/Wakeboarding"
    finally:
        store.close()


def test_a_suggestion_can_be_dismissed(workspace):
    from tests.test_precision import add_shot

    store = Store.for_config(workspace)
    try:
        shot = add_shot(store, "w.mp4", caption="x")
        store.record_folder_proposal("a/b", None, shot)
        proposal = store.list_folder_proposals()[0]["id"]
    finally:
        store.close()
    html = _client(workspace).post(f"/review/folders/{proposal}/dismiss").text
    assert "Suggestion dismissed" in html and "Suggested folders" not in html


def test_the_tag_audit_is_available_to_an_agent(workspace):
    body = _client(workspace).get("/api/audit/tags").json()
    assert body == {"shots": 0}
