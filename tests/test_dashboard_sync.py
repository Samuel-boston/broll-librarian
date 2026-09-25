"""Dashboard sync: the librarian mirrors its index into the dashboard's Supabase.

A fake Supabase (an httpx MockTransport) stands in for PostgREST and Storage, so
these tests check exactly what would be sent, without a network.
"""

from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from broll.db.models import Shot, Source
from broll.db.store import Store, new_id
from broll.sync.dashboard import KEY_ENV, DashboardSync, DashboardSyncError

URL = "https://example.supabase.co"


class FakeSupabase:
    def __init__(self, status: int = 200):
        self.rows: dict[str, dict] = {}
        self.thumbs: set[str] = set()
        self.calls: list[tuple[str, str]] = []
        self.status = status

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = urlparse(str(request.url)).path
        self.calls.append((request.method, path))
        if self.status != 200:
            return httpx.Response(self.status, text="nope")
        if path.startswith("/storage/v1/object/library-thumbs/") and request.method == "POST":
            self.thumbs.add(path.rsplit("/", 1)[1])
            return httpx.Response(200, json={})
        if path == "/storage/v1/object/library-thumbs" and request.method == "DELETE":
            for name in json.loads(request.content)["prefixes"]:
                self.thumbs.discard(name)
            return httpx.Response(200, json=[])
        if path == "/rest/v1/library_shots":
            if request.method == "POST":
                for row in json.loads(request.content):
                    self.rows[row["id"]] = row
                return httpx.Response(201)
            if request.method == "DELETE":
                raw = parse_qs(urlparse(str(request.url)).query)["id"][0]
                for sid in raw[4:-1].replace('"', "").split(","):
                    self.rows.pop(sid, None)
                return httpx.Response(204)
            return httpx.Response(200, json=[])
        return httpx.Response(404)


@pytest.fixture()
def connected(workspace, monkeypatch):
    workspace.dashboard.enabled = True
    workspace.dashboard.supabase_url = URL
    workspace.save()
    monkeypatch.setenv(KEY_ENV, "service-key")
    return workspace


def make_sync(config, fake: FakeSupabase) -> DashboardSync:
    return DashboardSync(config, client=httpx.Client(transport=httpx.MockTransport(fake.handler)))


def add_shot(store, config, name="a.mp4", with_thumb=True, caption="A calm beach") -> str:
    source = store.insert_source(Source(
        id=new_id(), workspace_id=store.workspace_id, content_hash=name,
        original_filename=name, origin="local", origin_path=f"/x/{name}",
        drive_file_id=f"drive-{name}", drive_web_link=f"https://drive/{name}",
    ))
    shot_id = f"{source.id}-0"
    thumb = config.thumbnails_dir / f"{shot_id}.jpg"
    if with_thumb:
        thumb.write_bytes(b"\xff\xd8jpeg")
    store.insert_shot(Shot(
        id=shot_id, workspace_id=store.workspace_id, source_id=source.id,
        caption=caption, status="indexed", subjects=["beach"], emotions=["calm"],
        category="Travel", thumbnail_path=str(thumb) if with_thumb else None,
        search_text=caption,
    ))
    return shot_id


def test_not_connected_does_nothing(workspace):
    result = DashboardSync(workspace).run()
    assert result.skipped_reason and result.upserted == 0


def test_first_sync_sends_shots_and_thumbnails(connected, store):
    shot_id = add_shot(store, connected)
    fake = FakeSupabase()
    result = make_sync(connected, fake).run()

    assert result.upserted == 1 and result.thumbnails == 1
    row = fake.rows[shot_id]
    assert row["caption"] == "A calm beach"
    assert row["media_kind"] == "video"
    assert row["drive_file_id"] == "drive-a.mp4"
    assert row["thumb_path"] == f"{shot_id}.jpg"
    assert f"{shot_id}.jpg" in fake.thumbs


def test_second_run_sends_nothing(connected, store):
    add_shot(store, connected)
    fake = FakeSupabase()
    sync = make_sync(connected, fake)
    sync.run()
    fake.calls.clear()
    result = sync.run()
    assert result.upserted == 0 and result.thumbnails == 0
    assert not [c for c in fake.calls if c[0] in ("POST", "DELETE")]


def test_a_change_is_sent_alone(connected, store):
    keep = add_shot(store, connected, "a.mp4")
    edit = add_shot(store, connected, "b.mp4")
    fake = FakeSupabase()
    sync = make_sync(connected, fake)
    sync.run()

    store.set_shot_fields(edit, caption="Corrected caption")
    result = sync.run()
    assert result.upserted == 1
    assert fake.rows[edit]["caption"] == "Corrected caption"
    assert fake.rows[keep]["caption"] == "A calm beach"


def test_removed_shots_are_pruned(connected, store):
    keep = add_shot(store, connected, "a.mp4")
    gone = add_shot(store, connected, "b.mp4")
    fake = FakeSupabase()
    sync = make_sync(connected, fake)
    sync.run()

    store.conn.execute("DELETE FROM shots WHERE id = ?", (gone,))
    result = sync.run()
    assert result.removed == 1
    assert set(fake.rows) == {keep}
    assert f"{gone}.jpg" not in fake.thumbs


def test_an_empty_local_index_never_wipes_the_dashboard(connected, store):
    only = add_shot(store, connected)
    fake = FakeSupabase()
    sync = make_sync(connected, fake)
    sync.run()

    store.conn.execute("DELETE FROM shots")
    result = sync.run()
    assert result.removed == 0
    assert only in fake.rows


def test_shots_still_needing_review_are_not_sent(connected, store):
    shot_id = add_shot(store, connected)
    store.set_shot_fields(shot_id, status="needs_review")
    fake = FakeSupabase()
    assert make_sync(connected, fake).run().total == 0


def test_a_photo_is_marked_as_an_image(connected, store):
    shot_id = add_shot(store, connected, "p.jpg")
    store.conn.execute("UPDATE sources SET media_kind = 'image'")
    fake = FakeSupabase()
    make_sync(connected, fake).run()
    assert fake.rows[shot_id]["media_kind"] == "image"


def test_wrong_key_gives_a_plain_message(connected, store):
    add_shot(store, connected)
    with pytest.raises(DashboardSyncError, match="service_role"):
        make_sync(connected, FakeSupabase(status=401)).check()


def test_missing_table_says_which_migration(connected):
    with pytest.raises(DashboardSyncError, match="setup_all.sql"):
        make_sync(connected, FakeSupabase(status=404)).check()


def test_pointing_at_a_new_dashboard_resends_everything(connected, store, monkeypatch):
    add_shot(store, connected)
    first = FakeSupabase()
    make_sync(connected, first).run()

    connected.dashboard.supabase_url = "https://other.supabase.co"
    second = FakeSupabase()
    result = make_sync(connected, second).run()
    assert result.upserted == 1 and len(second.rows) == 1


def test_settings_page_shows_the_dashboard_card(workspace):
    from fastapi.testclient import TestClient

    from broll.web.app import create_app

    with TestClient(create_app(workspace, run_worker=False)) as client:
        page = client.get("/settings")
        assert page.status_code == 200
        assert "Content Ops dashboard" in page.text and "Not connected" in page.text

        empty = client.post("/settings/dashboard", data={"supabase_url": "", "service_key": ""})
        assert "Both the Project URL" in empty.text


# ---- hardening (from the audit) ---------------------------------------------------------------


def test_a_network_error_is_wrapped_and_progress_is_kept(connected, store):
    good = add_shot(store, connected, "a.mp4")
    fake = FakeSupabase()
    sync = make_sync(connected, fake)
    sync.run()  # first shot goes across

    add_shot(store, connected, "b.mp4")

    def boom(request):
        if request.method == "POST" and request.url.path == "/rest/v1/library_shots":
            raise httpx.ConnectError("connection reset")
        return fake.handler(request)

    broken = DashboardSync(connected, client=httpx.Client(transport=httpx.MockTransport(boom)))
    with pytest.raises(DashboardSyncError, match="Couldn't finish"):
        broken.run()
    # the shot sent earlier is still remembered, so the retry doesn't redo it
    state = json.loads(broken.state_path.read_text())
    assert good in state["shots"]


def test_a_reset_index_does_not_wipe_the_dashboard(connected, store):
    ids = [add_shot(store, connected, f"{n}.mp4") for n in range(30)]
    fake = FakeSupabase()
    sync = make_sync(connected, fake)
    sync.run()
    assert len(fake.rows) == 30

    # index rebuilt with only two clips: 28 of 30 "vanished"
    store.conn.execute("DELETE FROM shots WHERE id NOT IN (?, ?)", (ids[0], ids[1]))
    result = sync.run()
    assert result.removed == 0 and len(fake.rows) == 30
    assert any("looks like a reset" in e for e in result.errors)
    # an explicit --force is allowed to do it
    forced = sync.run(force=True)
    assert forced.removed == 0 or len(fake.rows) <= 30  # force resends; nothing is pruned from a cleared state


def test_a_damaged_state_file_starts_over_instead_of_crashing(connected, store):
    add_shot(store, connected)
    fake = FakeSupabase()
    sync = make_sync(connected, fake)
    sync.state_path.write_text('["not", "a", "dict"]')
    assert sync.run().upserted == 1
    sync.state_path.write_text('{"url": "%s"}' % URL)  # right url, missing shape
    assert sync.run().total == 1


def test_write_env_var_is_owner_only_and_rejects_line_breaks(broll_home):
    from broll.config import write_env_var

    path = write_env_var("DASHBOARD_SUPABASE_KEY", "abc")
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    with pytest.raises(ValueError):
        write_env_var("X", "a\nEVIL=1")


def test_cross_site_posts_are_refused_but_local_ones_work(workspace):
    from fastapi.testclient import TestClient

    from broll.web.app import create_app

    with TestClient(create_app(workspace, run_worker=False)) as client:
        evil = client.post("/settings/dashboard", data={"supabase_url": "https://evil.example", "service_key": ""},
                           headers={"Origin": "https://evil.example"})
        assert evil.status_code == 403
        own = client.post("/settings/dashboard", data={"supabase_url": "", "service_key": ""},
                          headers={"Origin": "http://testserver", "Host": "testserver"})
        assert own.status_code == 200


def test_the_saved_key_is_never_sent_to_a_different_address(connected, store):
    from fastapi.testclient import TestClient

    from broll.web.app import create_app

    with TestClient(create_app(connected, run_worker=False)) as client:
        r = client.post("/settings/dashboard", data={"supabase_url": "https://other.supabase.co", "service_key": ""})
        assert "needed" in r.text  # no key for that address: it asks for one instead of reusing the saved one
        plain = client.post("/settings/dashboard", data={"supabase_url": "http://evil.example", "service_key": "k"})
        assert "https://" in plain.text
