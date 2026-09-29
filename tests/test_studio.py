"""Many clients in one process, and the JSON API an agent drives it with.

A studio has one library per client: separate databases, separate Drive trees,
separate folder structures. What is shared is the process, the embedding model
and the worker pool. Nothing may leak between clients except when the caller
explicitly asks to search across all of them.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from broll.config import CategoryNode, WorkspaceConfig
from broll.db.models import Shot, Source
from broll.db.store import Store, new_id
from broll.web.app import create_studio_app


@pytest.fixture()
def two_clients(broll_home):
    """Two client libraries with one clip each, describing different things."""
    made = []
    for client_id, name, caption, tags in (
        ("adam", "Adam Kunder", "A man meditates on a beach at sunrise.",
         ["meditation", "beach", "calm"]),
        ("luqman", "Luqman", "A salesman closes a deal in a glass office.",
         ["sales", "office", "handshake"]),
    ):
        config = WorkspaceConfig(id=client_id, name=name)
        config.provider.vision = "mock"
        config.taxonomy.mode = "tree"
        config.taxonomy.tree = [CategoryNode(name=f"01_{name.split()[0]} Folder")]
        config.save()
        config.ensure_dirs()

        store = Store.for_config(config)
        try:
            source = store.insert_source(Source(
                id=new_id(), workspace_id=client_id, content_hash=f"hash-{client_id}",
                original_filename=f"{client_id}.mp4", origin="local",
                duration_s=12.0, status="indexed"))
            store.insert_shot(Shot(
                id=f"{source.id}-0", workspace_id=client_id, source_id=source.id,
                caption=caption, tags=tags, emotions=["calm"], duration_s=12.0,
                start_s=0.0, end_s=12.0, status="indexed"))
            store.recompute_search_text(f"{source.id}-0")
        finally:
            store.close()
        made.append(config)
    return made


@pytest.fixture()
def studio(two_clients) -> TestClient:
    with TestClient(create_studio_app(two_clients, run_worker=False)) as client:
        yield client


# -- switching between clients ---------------------------------------------


def test_the_header_offers_every_client(studio):
    page = studio.get("/library").text
    assert "Adam Kunder" in page and "Luqman" in page


def test_a_chosen_client_is_remembered(studio):
    """?client= switches, and the cookie keeps you there for the next click."""
    first = studio.get("/library?client=luqman")
    assert first.status_code == 200
    assert "closes a deal" in first.text

    # No ?client= this time: the cookie decides, not the default.
    again = studio.get("/library")
    assert "closes a deal" in again.text
    assert "meditates on a beach" not in again.text


def test_an_unknown_client_falls_back_rather_than_erroring(studio):
    page = studio.get("/library?client=nobody")
    assert page.status_code == 200


def test_one_client_never_sees_another_clients_footage(studio):
    adam = studio.get("/api/search?client=adam&q=meditating").json()
    assert adam["count"] == 1
    assert "beach" in adam["results"][0]["caption"]

    luqman = studio.get("/api/search?client=luqman&q=meditating").json()
    assert luqman["count"] == 0, "a search in one client must not reach another"


# -- the agent API ----------------------------------------------------------


def test_clients_endpoint_lists_what_exists(studio):
    body = studio.get("/api/clients").json()
    ids = [c["id"] for c in body["clients"]]
    assert ids == ["adam", "luqman"]
    assert body["clients"][0]["shots"] == 1
    assert body["clients"][0]["folders"] == ["01_Adam Folder"]


def test_searching_every_client_at_once_says_which_is_which(studio):
    body = studio.get("/api/search?q=office deal&all_clients=true").json()
    assert body["count"] >= 1
    assert body["results"][0]["client"] == "luqman"


def test_a_shortlist_costs_nothing_and_carries_the_line_with_it(studio):
    body = studio.post("/api/shortlist", json={
        "client": "adam",
        "text": "He meditates on the beach until the morning feels calm again.",
    }).json()

    assert body["client"] == "adam"
    assert body["beats"] >= 1
    beat = body["shortlist"][0]
    assert beat["line"]
    assert beat["duration_s"] > 0
    assert beat["candidates"], "the agent needs something to choose between"
    assert beat["candidates"][0]["caption"]


def test_a_shot_already_used_is_not_offered_again(studio):
    first = studio.post("/api/shortlist", json={
        "client": "adam", "text": "A calm beach meditation at sunrise.",
    }).json()
    used = first["shortlist"][0]["candidates"][0]["shot_id"]

    second = studio.post("/api/shortlist", json={
        "client": "adam", "text": "A calm beach meditation at sunrise.",
        "exclude_shots": [used],
    }).json()
    offered = [c["shot_id"] for beat in second["shortlist"] for c in beat["candidates"]]
    assert used not in offered


def test_the_api_names_the_clients_it_knows_when_asked_for_a_stranger(studio):
    response = studio.get("/api/search?client=ghost&q=anything")
    assert response.status_code == 404
    assert "adam" in response.json()["detail"]


def test_ingest_queues_against_the_named_client(studio, tmp_path):
    clip = tmp_path / "new.mp4"
    clip.write_bytes(b"pretend video")

    body = studio.post("/api/ingest", json={
        "client": "luqman", "paths": [str(clip), str(tmp_path / "missing.mov")],
    }).json()

    assert body["client"] == "luqman"
    assert body["queued"] == 1
    assert body["skipped"] == [str(tmp_path / "missing.mov")]

    status = studio.get("/api/status?client=luqman").json()
    assert status["queued"] == 1
    assert studio.get("/api/status?client=adam").json()["queued"] == 0


def test_a_token_is_required_once_one_is_set(studio, monkeypatch):
    monkeypatch.setenv("BROLL_API_TOKEN", "s3cret")
    assert studio.get("/api/clients").status_code == 401
    assert studio.get("/api/clients", headers={"X-Broll-Token": "wrong"}).status_code == 401
    ok = studio.get("/api/clients", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
