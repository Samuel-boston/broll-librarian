"""Running on a server: the shared-password gate, Connect Google Drive from the browser, database
backups, and setting up a workspace that uses an embeddings API instead of a local model."""

from __future__ import annotations

import json
import sqlite3
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from broll import cli
from broll.analysis.embedder import GeminiEmbedder, get_embedder
from broll.backup import backup_database, backups_dir, list_backups
from broll.config import WorkspaceConfig
from broll.db.store import Store
from broll.web import access
from broll.web.app import create_app
from tests.test_dashboard_sync import add_shot

runner = CliRunner()


def _client(workspace) -> TestClient:
    return TestClient(create_app(workspace, run_worker=False), follow_redirects=False)


@pytest.fixture(autouse=True)
def _clean_gate(monkeypatch):
    monkeypatch.delenv(access.PASSWORD_ENV, raising=False)
    access.reset_failures()
    yield
    access.reset_failures()


# ---- health check ----------------------------------------------------------


def test_healthz_is_always_open(workspace, monkeypatch):
    monkeypatch.setenv(access.PASSWORD_ENV, "hunter2")
    with _client(workspace) as client:
        response = client.get("/healthz")
    assert response.status_code == 200 and response.text == "ok"


# ---- the password gate -----------------------------------------------------


def test_no_password_means_no_gate(workspace):
    with _client(workspace) as client:
        assert client.get("/library").status_code == 200
        assert client.get("/login").status_code == 303  # nothing to sign in to


def test_with_a_password_pages_redirect_to_login(workspace, monkeypatch):
    monkeypatch.setenv(access.PASSWORD_ENV, "hunter2")
    with _client(workspace) as client:
        for path in ("/library", "/ingest", "/settings", "/search?q=beach", "/thumbnails/x.jpg"):
            response = client.get(path)
            assert response.status_code == 303 and response.headers["location"].startswith("/login?next="), path
        assert client.post("/library/delete-all", headers={"HX-Prompt": "DELETE"}).status_code == 303


def test_htmx_requests_get_a_401_that_sends_the_browser_to_login(workspace, monkeypatch):
    monkeypatch.setenv(access.PASSWORD_ENV, "hunter2")
    with _client(workspace) as client:
        response = client.get("/ingest/queue", headers={"HX-Request": "true"})
    assert response.status_code == 401 and response.headers["HX-Redirect"] == "/login"


def test_login_with_the_right_password_opens_the_app(workspace, monkeypatch):
    monkeypatch.setenv(access.PASSWORD_ENV, "hunter2")
    with _client(workspace) as client:
        page = client.get("/login?next=/ingest")
        assert page.status_code == 200 and 'value="/ingest"' in page.text
        response = client.post("/login", data={"password": "hunter2", "next": "/ingest"})
        assert response.status_code == 303 and response.headers["location"] == "/ingest"
        cookie = response.headers["set-cookie"]
        assert "HttpOnly" in cookie and "samesite=lax" in cookie.lower()
        assert client.get("/ingest").status_code == 200  # the cookie is kept
        assert client.get("/library").status_code == 200
        out = client.post("/logout")
        assert out.headers["location"] == "/login"
        assert client.get("/library").status_code == 303


def test_a_wrong_password_is_refused(workspace, monkeypatch):
    monkeypatch.setenv(access.PASSWORD_ENV, "hunter2")
    with _client(workspace) as client:
        response = client.post("/login", data={"password": "nope", "next": ""})
        assert response.status_code == 401 and "isn&#x27;t the password" in response.text or "isn't the password" in response.text
        assert "set-cookie" not in response.headers
        assert client.get("/library").status_code == 303


def test_repeated_wrong_passwords_are_slowed_down_even_for_the_right_one(workspace, monkeypatch):
    monkeypatch.setenv(access.PASSWORD_ENV, "hunter2")
    with _client(workspace) as client:
        for _ in range(access.MAX_PER_CLIENT):
            assert client.post("/login", data={"password": "no"}).status_code == 401
        locked = client.post("/login", data={"password": "hunter2"})
        assert locked.status_code == 429 and "set-cookie" not in locked.headers


def test_login_never_redirects_off_site(workspace, monkeypatch):
    monkeypatch.setenv(access.PASSWORD_ENV, "hunter2")
    assert access.safe_next("//evil.example") == "/library"
    assert access.safe_next("https://evil.example") == "/library"
    assert access.safe_next("/\\evil.example") == "/library"
    assert access.safe_next("/ingest?x=1") == "/ingest?x=1"
    with _client(workspace) as client:
        response = client.post("/login", data={"password": "hunter2", "next": "//evil.example"})
    assert response.headers["location"] == "/library"


def test_the_login_page_escapes_what_it_echoes(workspace, monkeypatch):
    monkeypatch.setenv(access.PASSWORD_ENV, "hunter2")
    with _client(workspace) as client:
        page = client.get('/login?next=/x"><script>alert(1)</script>').text
    assert "<script>alert(1)</script>" not in page


def test_tokens_expire_and_cannot_be_forged_or_reused_after_a_password_change():
    now = 1_000_000.0
    token = access.make_token("pw", now)
    assert access.valid_token("pw", token, now + 10)
    assert not access.valid_token("pw", token, now + access.TTL_S + 1)
    assert not access.valid_token("other", token, now + 10)  # changing the password signs everyone out
    expires, _, signature = token.partition(".")
    assert not access.valid_token("pw", f"{int(expires) + 99999}.{signature}", now + 10)
    assert not access.valid_token("pw", "garbage", now) and not access.valid_token("pw", None, now)


# ---- connect Google Drive from the browser -------------------------------


class FakeFlow:
    def __init__(self, refresh_token="refresh-1"):
        self.code_verifier = "verifier-1"
        self.credentials = SimpleNamespace(
            refresh_token=refresh_token,
            to_json=lambda: json.dumps({"client_id": "cid", "client_secret": "sec", "refresh_token": refresh_token}),
        )
        self.fetched = None

    def authorization_url(self, **kwargs):
        assert kwargs["access_type"] == "offline" and kwargs["prompt"] == "consent"
        return "https://accounts.google.com/o/oauth2/auth?client_id=cid&state=abc123", "abc123"

    def fetch_token(self, code):
        self.fetched = code


@pytest.fixture()
def google(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "cid")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "sec")
    monkeypatch.setenv("BROLL_PUBLIC_URL", "https://library.example.com/")
    flow = FakeFlow()
    seen = {}

    def new_flow(uri):
        seen["uri"] = uri
        return flow

    monkeypatch.setattr("broll.web.routes.drive_connect.new_flow", new_flow)
    return flow, seen


def test_connect_sends_the_person_to_google_with_the_public_callback(workspace, google):
    _, seen = google
    with _client(workspace) as client:
        response = client.get("/drive/connect")
    assert response.status_code == 303 and response.headers["location"].startswith("https://accounts.google.com/")
    assert seen["uri"] == "https://library.example.com/drive/callback"


def test_the_callback_stores_the_login(workspace, google):
    flow, _ = google
    with _client(workspace) as client:
        client.get("/drive/connect")
        assert not workspace.drive_token_path.exists()
        response = client.get("/drive/callback?code=the-code&state=abc123")
        assert response.status_code == 303 and "Google%20Drive%20is%20connected" in response.headers["location"]
        settings = client.get("/settings").text
    assert flow.fetched == "the-code"
    assert flow.code_verifier == "verifier-1"
    assert json.loads(workspace.drive_token_path.read_text())["refresh_token"] == "refresh-1"
    assert oct(workspace.drive_token_path.stat().st_mode)[-3:] == "600"
    assert "Connected." in settings


def test_the_callback_refuses_an_unknown_or_reused_state(workspace, google):
    with _client(workspace) as client:
        forged = client.get("/drive/callback?code=x&state=not-started-here")
        assert "expired" in forged.headers["location"]
        client.get("/drive/connect")
        client.get("/drive/callback?code=x&state=abc123")
        again = client.get("/drive/callback?code=x&state=abc123")
        assert "expired" in again.headers["location"]


def test_the_callback_handles_a_declined_sign_in_and_a_missing_refresh_token(workspace, google, monkeypatch):
    with _client(workspace) as client:
        declined = client.get("/drive/callback?error=access_denied")
        assert "declined" in declined.headers["location"]
        monkeypatch.setattr("broll.web.routes.drive_connect.new_flow", lambda uri: FakeFlow(refresh_token=None))
        client.get("/drive/connect")
        no_refresh = client.get("/drive/callback?code=x&state=abc123")
        assert "long-lived" in no_refresh.headers["location"]
    assert not workspace.drive_token_path.exists()


def test_connect_without_a_saved_client_explains(workspace, monkeypatch):
    for name in ("GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET", "GOOGLE_OAUTH_CLIENT_SECRETS_FILE"):
        monkeypatch.delenv(name, raising=False)
    with _client(workspace) as client:
        response = client.get("/drive/connect")
    assert response.status_code == 303 and "No%20Google%20OAuth%20client" in response.headers["location"]


def test_the_settings_page_saves_the_google_client_and_never_shows_the_secret(workspace, broll_home, monkeypatch):
    for name in ("GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BROLL_PUBLIC_URL", "https://library.example.com")
    with _client(workspace) as client:
        page = client.get("/settings").text
        assert "https://library.example.com/drive/callback" in page and 'href="/drive/connect"' not in page
        response = client.post("/settings/google-client", data={"client_id": "abc.apps.googleusercontent.com", "client_secret": "s3cret-value"})
        assert "Google client saved" in response.text and "s3cret-value" not in response.text
        assert 'href="/drive/connect"' in client.get("/settings").text
    env = (broll_home / ".env").read_text()
    assert "GOOGLE_OAUTH_CLIENT_ID=abc.apps.googleusercontent.com" in env and "GOOGLE_OAUTH_CLIENT_SECRET=s3cret-value" in env


def test_drive_import_takes_a_login_made_elsewhere(workspace):
    from broll.drive.auth import DriveAuthError, import_token

    import_token(workspace, {"client_id": "a", "client_secret": "b", "refresh_token": "c", "token": "ignored"})
    saved = json.loads(workspace.drive_token_path.read_text())
    assert saved["refresh_token"] == "c" and saved["type"] == "authorized_user" and "token" not in saved
    with pytest.raises(DriveAuthError, match="refresh_token"):
        import_token(workspace, {"client_id": "a", "client_secret": "b"})


# ---- backups ---------------------------------------------------------------


def test_a_backup_is_a_complete_copy_and_only_the_newest_are_kept(workspace, store):
    shot_id = add_shot(store, workspace, "a.mp4", caption="A calm beach")
    first = backup_database(workspace, keep=2)
    with sqlite3.connect(first) as conn:
        assert conn.execute("SELECT caption FROM shots WHERE id = ?", (shot_id,)).fetchone()[0] == "A calm beach"
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert oct(first.stat().st_mode)[-3:] == "600"

    for stamp in ("20200101-000000", "20200102-000000", "20200103-000000"):
        (backups_dir(workspace) / f"library-{stamp}.db").write_bytes(b"old")
    backup_database(workspace, keep=2)
    names = [p.name for p in list_backups(workspace)]
    # the new copy and the newest old one survive; the two older ones are gone
    assert len(names) == 2 and "library-20200103-000000.db" in names
    assert "library-20200101-000000.db" not in names and "library-20200102-000000.db" not in names
    assert not list(backups_dir(workspace).glob("*.part"))


def test_backup_cli(broll_home):
    assert runner.invoke(cli.app, ["init", "--name", "Cli", "--provider", "mock"]).exit_code == 0
    result = runner.invoke(cli.app, ["backup", "-w", "cli"])
    assert result.exit_code == 0 and "Backed up to" in result.output, result.output
    listing = runner.invoke(cli.app, ["backup", "-w", "cli", "--list"])
    assert "library-" in listing.output


def test_backups_run_on_their_own_when_switched_on(workspace, store, monkeypatch):
    from broll.backup import backups_enabled

    assert not backups_enabled(workspace)
    monkeypatch.setenv("BROLL_BACKUPS", "1")
    assert backups_enabled(workspace)
    monkeypatch.delenv("BROLL_BACKUPS")
    workspace.backup.enabled = True
    with TestClient(create_app(workspace, run_worker=False)) as client:
        client.get("/healthz")
        deadline = time.monotonic() + 10
        while not list_backups(workspace) and time.monotonic() < deadline:
            time.sleep(0.1)
    assert len(list_backups(workspace)) == 1


# ---- a workspace that uses an embeddings API -----------------------------


def test_init_with_a_gemini_embedder_sizes_the_vector_table_to_match(broll_home):
    result = runner.invoke(cli.app, ["init", "--name", "Hosted", "--provider", "gemini", "--embedder", "gemini", "--hosted"])
    assert result.exit_code == 0, result.output
    config = WorkspaceConfig.load("hosted") if hasattr(WorkspaceConfig, "load") else cli.resolve_workspace("hosted")
    assert (config.embedder.kind, config.embedder.model, config.embedder.dimensions) == ("gemini", "gemini-embedding-001", 768)
    assert config.backup.enabled is True
    store = Store.for_config(config)
    try:
        assert store.vectors.dimensions == 768 if hasattr(store.vectors, "dimensions") else store._dimensions == 768
        embedder = get_embedder(config)
        assert isinstance(embedder, GeminiEmbedder) and embedder.dimensions == 768
        # the embedder and the table agree, so a vector can be stored
        store.vectors.upsert("shot-1", [0.1] * embedder.dimensions)
        assert store.vectors.count() == 1
    finally:
        store.close()


def test_init_rejects_an_unknown_embedder(broll_home):
    result = runner.invoke(cli.app, ["init", "--name", "X", "--provider", "mock", "--embedder", "magic"])
    assert result.exit_code != 0 and "Unknown embedder" in result.output


def test_the_login_throttle_trusts_only_the_address_the_proxy_appended():
    def request(forwarded):
        return SimpleNamespace(headers={"x-forwarded-for": forwarded}, client=SimpleNamespace(host="172.18.0.2"))

    assert access._client(request("1.1.1.1, 9.9.9.9")) == access._client(request("2.2.2.2, 9.9.9.9")) == "9.9.9.9"


def test_a_cookie_with_odd_digits_is_rejected_not_a_crash():
    assert access.valid_token("pw", "²³.abc") is False
    assert access.valid_token("pw", "٣٣.abc") is False


def test_the_entrypoint_refuses_a_public_domain_without_a_password():
    import subprocess
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "deploy" / "entrypoint.sh"
    done = subprocess.run(["sh", str(script)], env={"PATH": "/usr/bin:/bin", "BROLL_DOMAIN": "example.com"},
                          capture_output=True, text=True)
    assert done.returncode == 1 and "BROLL_ACCESS_PASSWORD" in done.stderr


def test_gemini_embeddings_are_sent_in_batches_the_api_accepts(monkeypatch):
    """The API rejects more than 100 texts in a request; a rebuild of a big library sends more."""
    import sys
    import types as pytypes

    seen = []

    class FakeModels:
        def embed_content(self, model, contents, config):
            seen.append((len(contents), config.task_type))
            return pytypes.SimpleNamespace(
                embeddings=[pytypes.SimpleNamespace(values=[1.0, 0.0]) for _ in contents])

    genai = pytypes.ModuleType("google.genai")
    genai.Client = lambda api_key: pytypes.SimpleNamespace(models=FakeModels())
    genai.types = pytypes.SimpleNamespace(
        EmbedContentConfig=lambda **kw: pytypes.SimpleNamespace(**kw))
    google = pytypes.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)

    from broll.analysis.embedder import GeminiEmbedder
    from broll.config import EmbedderConfig

    embedder = GeminiEmbedder(EmbedderConfig(kind="gemini", model="gemini-embedding-001", dimensions=2), "k")
    assert len(embedder.embed_documents(["t"] * 250)) == 250
    assert [n for n, _ in seen] == [96, 96, 58] and {t for _, t in seen} == {"RETRIEVAL_DOCUMENT"}
    seen.clear()
    embedder.embed_query("q")
    assert seen == [(1, "RETRIEVAL_QUERY")], "a query is embedded as a query"


def test_a_stranger_cannot_make_the_server_read_a_big_body(monkeypatch, tmp_path):
    """An unauthenticated upload or login is refused before its body is read (and so before it reaches disk)."""
    from broll.web import access

    monkeypatch.setenv("BROLL_ACCESS_PASSWORD", "hunter2")
    monkeypatch.delenv("BROLL_API_TOKEN", raising=False)
    from broll.config import WorkspaceConfig
    from broll.web.app import create_app

    config = WorkspaceConfig(id="t", name="T")
    config.provider.vision = "mock"
    config.save()
    config.ensure_dirs()
    with TestClient(create_app(config, run_worker=False), client=("203.0.113.9", 5000)) as client:
        spilled = {"n": 0}
        original = access._api_refusal

        def counting(request):
            spilled["n"] += 1
            return original(request)

        monkeypatch.setattr(access, "_api_refusal", counting)
        upload = client.post("/api/upload", files=[("files", ("a.mp4", b"x" * 200_000, "video/mp4"))])
        assert upload.status_code == 403 and spilled["n"] == 1
        big_login = client.post("/login", content=b"password=" + b"x" * 200_000,
                                headers={"content-type": "application/x-www-form-urlencoded"})
        assert big_login.status_code == 413
        assert client.post("/login", data={"password": "nope"}).status_code == 401


def test_restore_puts_the_backup_back_and_does_not_replay_newer_writes(workspace, store):
    import sqlite3

    from broll.backup import RestoreError, backup_database, restore_database
    from broll.db.models import Shot, Source
    from broll.db.store import Store, new_id

    def add(n):
        source = store.insert_source(Source(id=new_id(), workspace_id=workspace.id, content_hash=f"h{n}",
                                            original_filename=f"{n}.mov", origin="local"))
        store.insert_shot(Shot(id=f"{source.id}-0", workspace_id=workspace.id, source_id=source.id, status="indexed"))

    for n in range(5):
        add(n)
    backup = backup_database(workspace)
    for n in range(5, 12):
        add(n)  # newer rows, sitting in the -wal as far as the backup is concerned
    store.conn.commit()
    with pytest.raises(RestoreError):
        restore_database(workspace, backup.parent / "nope.db")
    store.close()
    done = restore_database(workspace, backup)
    assert done["counts"] == {"sources": 5, "shots": 5} and done["kept"].exists()
    again = Store.for_config(workspace)
    try:
        assert again.count_shots() == 5
    finally:
        again.close()
    kept = sqlite3.connect(str(done["kept"]))
    assert kept.execute("SELECT COUNT(*) FROM shots").fetchone()[0] == 12  # nothing was thrown away
    kept.close()
