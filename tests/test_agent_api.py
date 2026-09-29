"""What an editing agent needs after it has a shortlist: a verdict on each
candidate, memory of where shots were used, a library for every client (empty
ones included), and a health check that says what is wrong without saying
anything secret.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from broll.clients import register
from broll.config import CategoryNode, WorkspaceConfig
from broll.db.models import Shot, Source
from broll.db.store import Registry, Store, canonical_time, new_id
from broll.web.app import create_studio_app


def make_library(client_id: str, name: str, clips: list[tuple[str, list[str], int]]) -> WorkspaceConfig:
    """A registered library. Each clip is (caption, tags, number of shots)."""
    config = WorkspaceConfig(id=client_id, name=name)
    config.provider.vision = "mock"
    config.ingest.requests_per_minute = 5.0
    config.taxonomy.mode = "tree"
    config.taxonomy.tree = [CategoryNode(name=f"{name} Folder")]
    registry = Registry()
    try:
        register(config, registry)
    finally:
        registry.close()
    config.ensure_dirs()
    store = Store.for_config(config)
    try:
        for caption, tags, shots in clips:
            source = store.insert_source(Source(
                id=new_id(), workspace_id=client_id, content_hash=new_id(),
                original_filename=f"{tags[0]}.mp4", origin="local",
                duration_s=6.0 * shots, width=1920, height=1080, fps=25.0, status="indexed"))
            for index in range(shots):
                shot_id = f"{source.id}-{index}"
                store.insert_shot(Shot(
                    id=shot_id, workspace_id=client_id, source_id=source.id, shot_index=index,
                    caption=caption, tags=tags, duration_s=6.0, start_s=6.0 * index,
                    end_s=6.0 * (index + 1), status="indexed"))
                store.recompute_search_text(shot_id)
    finally:
        store.close()
    return config


def shot_ids(config: WorkspaceConfig, word: str) -> list[str]:
    store = Store.for_config(config)
    try:
        return [s.id for s in store.list_shots() if word in (s.caption or "")]
    finally:
        store.close()


@pytest.fixture()
def libraries(broll_home):
    adam = make_library("adam", "Adam Kunder", [
        ("A man meditates on a beach at sunrise.", ["beach", "meditation", "calm"], 3),
        ("A man journals by a window in the morning.", ["journal", "writing", "morning"], 1),
        ("Two men shake hands to close a deal in an office.", ["office", "handshake", "deal"], 2),
    ])
    luqman = make_library("luqman", "Luqman", [
        ("A salesman closes a deal in a glass office.", ["sales", "office", "handshake"], 1),
    ])
    return adam, luqman


@pytest.fixture()
def api(libraries):
    with TestClient(create_studio_app(list(libraries), run_worker=False)) as client:
        yield client


def beats_shortlist(api, **body):
    body.setdefault("client", "adam")
    response = api.post("/api/shortlist", json=body)
    assert response.status_code == 200, response.text
    return {b["beat"]: b for b in response.json()["shortlist"]}


# -- a verdict on every candidate --------------------------------------------


def test_every_candidate_says_whether_it_is_really_about_the_beat(api):
    beats = beats_shortlist(api, beats=[{"id": "b1", "queries": ["beach meditation"]}])
    candidates = beats["b1"]["candidates"]
    assert candidates, "an agent needs something to choose between"
    assert all(c["relevance"] in ("match", "near", "weak") for c in candidates)
    assert candidates[0]["relevance"] == "match"
    assert "beach" in candidates[0]["caption"]
    assert beats["b1"]["matches"] >= 1
    assert beats["b1"]["reason"] is None


def test_each_idea_in_a_beat_gets_its_own_candidates(api):
    """A chained beat wants a clip per idea, not the two best matches for one of them."""
    beats = beats_shortlist(api, candidates_per_beat=2, beats=[
        {"id": "b1", "queries": ["beach meditation sunrise", "office handshake deal"]},
    ])
    captions = " ".join(c["caption"] for c in beats["b1"]["candidates"])
    assert "beach" in captions and "office" in captions
    assert {c["query"] for c in beats["b1"]["candidates"]} == {
        "beach meditation sunrise", "office handshake deal",
    }


def test_one_file_cannot_fill_a_beat_on_its_own(api):
    beats = beats_shortlist(api, max_per_source=1, beats=[{"id": "b1", "queries": ["beach meditation"]}])
    sources = [c["source_id"] for c in beats["b1"]["candidates"]]
    assert len(sources) == len(set(sources))


def test_a_beat_with_nothing_on_topic_says_so_rather_than_forcing_a_clip(api):
    beats = beats_shortlist(api, beats=[{"id": "b1", "queries": ["a mariachi band at a wedding"]}])
    assert beats["b1"]["matches"] == 0
    assert "really about" in beats["b1"]["reason"] or "passes" in beats["b1"]["reason"]


def test_a_script_shortlist_labels_its_candidates_too(api):
    body = api.post("/api/shortlist", json={"client": "adam", "text": "He meditates on the beach."}).json()
    beat = body["shortlist"][0]
    assert beat["candidates"][0]["relevance"] == "match"
    assert body["library_shots"] == 6


# -- memory of where shots were used ------------------------------------------


def test_a_video_s_shots_are_not_offered_to_its_other_beats(api, libraries):
    adam, _ = libraries
    used = shot_ids(adam, "beach")[0]
    recorded = api.post("/api/usage", json={
        "client": "adam", "project": "v1", "shot_ids": [used], "beats": {used: "b1"},
    }).json()
    assert recorded["recorded"] == 1

    beats = beats_shortlist(api, exclude_used_in_project="v1", beats=[
        {"id": "b1", "queries": ["beach meditation"]},
        {"id": "b2", "queries": ["beach meditation"]},
    ])
    assert used in [c["shot_id"] for c in beats["b1"]["candidates"]], "its own beat keeps it"
    assert used not in [c["shot_id"] for c in beats["b2"]["candidates"]], "never twice in a video"


def test_shots_another_video_used_recently_are_held_back(api, libraries):
    adam, _ = libraries
    used = shot_ids(adam, "beach")[0]
    api.post("/api/usage", json={"client": "adam", "project": "last-week", "shot_ids": [used]})

    other = beats_shortlist(api, project="this-week", exclude_used_within_days=30,
                            beats=[{"id": "b1", "queries": ["beach meditation"]}])
    assert used not in [c["shot_id"] for c in other["b1"]["candidates"]]

    own = beats_shortlist(api, project="last-week", exclude_used_within_days=30,
                          beats=[{"id": "b1", "queries": ["beach meditation"]}])
    assert used in [c["shot_id"] for c in own["b1"]["candidates"]], \
        "a video's own choices never count against it"


def test_use_long_ago_does_not_count_as_recent(api, libraries):
    adam, _ = libraries
    used = shot_ids(adam, "beach")[0]
    long_ago = (datetime.now(UTC) - timedelta(days=60)).isoformat()
    api.post("/api/usage", json={"client": "adam", "project": "old", "shot_ids": [used], "used_at": long_ago})

    beats = beats_shortlist(api, project="new", exclude_used_within_days=30,
                            beats=[{"id": "b1", "queries": ["beach meditation"]}])
    assert used in [c["shot_id"] for c in beats["b1"]["candidates"]]


def test_candidates_carry_where_they_have_been_used(api, libraries):
    adam, _ = libraries
    used = shot_ids(adam, "journal")[0]
    api.post("/api/usage", json={"client": "adam", "project": "v7", "shot_ids": [used]})
    beats = beats_shortlist(api, beats=[{"id": "b1", "queries": ["journal writing"]}])
    mine = next(c for c in beats["b1"]["candidates"] if c["shot_id"] == used)
    assert mine["use_count"] == 1
    assert mine["used_in"][0]["project"] == "v7"


def test_replacing_a_video_s_list_releases_the_shots_it_dropped(api, libraries):
    adam, _ = libraries
    first, second = shot_ids(adam, "beach")[:2]
    api.post("/api/usage", json={"client": "adam", "project": "v1", "shot_ids": [first, second]})
    result = api.post("/api/usage", json={
        "client": "adam", "project": "v1", "shot_ids": [second], "replace": True,
    }).json()
    assert result["released"] == 1
    listed = api.get("/api/usage?client=adam&project=v1").json()["usage"]
    assert [row["shot_id"] for row in listed] == [second]


def test_usage_reports_ids_the_library_does_not_hold(api):
    result = api.post("/api/usage", json={"client": "adam", "project": "v1", "shot_ids": ["nope"]}).json()
    assert result["recorded"] == 0
    assert result["unknown"] == ["nope"]


def test_recording_usage_must_name_the_client_when_there_are_several(api, libraries):
    adam, _ = libraries
    response = api.post("/api/usage", json={"project": "v1", "shot_ids": shot_ids(adam, "beach")[:1]})
    assert response.status_code == 400
    assert "adam" in response.json()["detail"]


def test_a_search_can_leave_out_what_a_video_already_uses(api, libraries):
    adam, _ = libraries
    used = shot_ids(adam, "journal")[0]
    api.post("/api/usage", json={"client": "adam", "project": "v1", "shot_ids": [used]})
    plain = api.get("/api/search?client=adam&q=journal").json()
    assert used in [r["shot_id"] for r in plain["results"]]
    filtered = api.get("/api/search?client=adam&q=journal&exclude_used_in_project=v1").json()
    assert used not in [r["shot_id"] for r in filtered["results"]]


def test_usage_goes_when_its_shot_is_removed(libraries):
    adam, _ = libraries
    store = Store.for_config(adam)
    try:
        shot = store.get_shot(shot_ids(adam, "journal")[0])
        store.record_usage([shot.id], "v1")
        assert store.list_usage(project="v1")
        store.delete_source(shot.source_id)
        assert store.list_usage(project="v1") == []
    finally:
        store.close()


def test_usage_dates_are_stored_in_one_form():
    assert canonical_time("2026-09-29T10:00:00+01:00") == "2026-09-29T09:00:00+00:00"
    assert canonical_time("2026-09-29T10:00:00") == "2026-09-29T10:00:00+00:00"
    with pytest.raises(ValueError):
        canonical_time("last tuesday")


def test_a_bad_usage_date_is_refused_with_the_reason(api, libraries):
    adam, _ = libraries
    response = api.post("/api/usage", json={
        "client": "adam", "project": "v1", "shot_ids": shot_ids(adam, "beach")[:1], "used_at": "yesterday",
    })
    assert response.status_code == 422
    assert "ISO 8601" in response.json()["detail"]


# -- a library for every client ----------------------------------------------


def test_a_new_client_gets_an_empty_library_it_can_use_at_once(api):
    made = api.post("/api/clients", json={"name": "Knversion"}).json()
    assert made == {"id": "knversion", "name": "Knversion", "created": True}
    assert "knversion" in [c["id"] for c in api.get("/api/clients").json()["clients"]]

    body = api.post("/api/shortlist", json={
        "client": "knversion", "beats": [{"id": "b1", "queries": ["a founder at a whiteboard"]}],
    }).json()
    assert body["library_shots"] == 0
    assert body["shortlist"][0]["candidates"] == []
    assert "empty" in body["shortlist"][0]["reason"]

    text = api.post("/api/shortlist", json={"client": "knversion", "text": "We grew fast."}).json()
    assert "empty" in text["shortlist"][0]["reason"]
    assert api.get("/api/search?client=knversion&q=anything").json()["count"] == 0


def test_asking_for_a_client_s_library_twice_never_makes_a_second(api):
    first = api.post("/api/clients", json={"name": "Knversion"}).json()
    again = api.post("/api/clients", json={"name": "  knversion "}).json()
    assert again == {"id": first["id"], "name": "Knversion", "created": False}


def test_a_library_made_by_hand_is_found_by_the_client_s_name(api):
    """Adam's library has the id 'adam'; an app that only knows "Adam Kunder" gets it."""
    found = api.post("/api/clients", json={"name": "adam kunder"}).json()
    assert found == {"id": "adam", "name": "Adam Kunder", "created": False}


def test_a_library_renamed_in_settings_is_found_by_the_name_people_see(api, libraries):
    """Caught live: the registry keeps the name a library was created with ("Nathan Test"),
    Settings renames only its config ("Adam Kunder"), and matching the registry's name made
    a second library for Adam."""
    adam, _ = libraries
    registry = Registry()
    try:
        registry.update("adam", name="Nathan Test")
    finally:
        registry.close()
    found = api.post("/api/clients", json={"name": "Adam Kunder"}).json()
    assert found == {"id": "adam", "name": "Adam Kunder", "created": False}
    assert [c["id"] for c in api.get("/api/clients").json()["clients"]] == ["adam", "luqman"]


def test_a_new_library_shares_the_server_s_key_settings_but_not_its_client(api, broll_home):
    api.post("/api/clients", json={"name": "Knversion"})
    from broll.config import load_workspace_config

    made = load_workspace_config("knversion")
    assert made.provider.vision == "mock"
    assert made.ingest.requests_per_minute == 5.0, "a free key's rate cap must hold for every library"
    assert made.taxonomy.tree == [], "the folder tree is the new client's own"


def test_an_app_that_suggests_its_own_id_still_gets_the_existing_library(api):
    """The Editing Joe app sends an id made from its client slug; Adam's library is 'adam'."""
    found = api.post("/api/clients", json={"name": "Adam Kunder", "id": "adam-kunder"}).json()
    assert found == {"id": "adam", "name": "Adam Kunder", "created": False}
    made = api.post("/api/clients", json={"name": "Knversion", "id": "knv"}).json()
    assert made == {"id": "knv", "name": "Knversion", "created": True}


def test_a_client_id_has_to_be_a_slug(api):
    response = api.post("/api/clients", json={"name": "X", "id": "Not A Slug"})
    assert response.status_code == 422


# -- is it working ------------------------------------------------------------


def test_health_reports_each_client_without_revealing_a_key(api, libraries, monkeypatch):
    adam, _ = libraries
    adam.provider.vision = "gemini"
    monkeypatch.setenv("GEMINI_API_KEY", "sk-this-must-never-appear")
    response = api.get("/api/health")
    assert response.status_code == 200
    assert "sk-this-must-never-appear" not in response.text

    body = response.json()
    assert body["ok"] is True and body["api"] >= 2
    clients = {c["id"]: c for c in body["clients"]}
    assert clients["adam"]["provider_key"] is True
    assert clients["adam"]["shots"] == 6
    assert clients["luqman"]["drive_token"] is False
    assert any("broll drive login -w luqman" in p for p in body["problems"])


def test_health_says_when_a_drive_login_has_stopped_working(api, libraries, monkeypatch):
    adam, _ = libraries
    adam.drive_token_path.write_text("{}")
    from broll.drive import auth

    def expired(config):
        raise auth.DriveAuthError("Token has been expired or revoked.")

    monkeypatch.setattr(auth, "load_credentials", expired)
    body = api.get("/api/health?refresh=true").json()
    adam_state = next(c for c in body["clients"] if c["id"] == "adam")
    assert adam_state["drive_token"] is True
    assert adam_state["drive_connected"] is False
    assert any("no longer works" in p and "adam" in p for p in body["problems"])


def test_health_explains_a_free_key_that_has_run_out_for_the_day(api, libraries):
    adam, _ = libraries
    store = Store.for_config(adam)
    try:
        job = store.enqueue("index_source", {"filename": "beach.mp4"})
        store.retry_job(job.id, "429 RESOURCE_EXHAUSTED: Quota exceeded for metric "
                                "generate_content_free_tier_requests, limit: "
                                "GenerateRequestsPerDayPerProjectPerModel-FreeTier", 600)
    finally:
        store.close()
    body = api.get("/api/health").json()
    adam_state = next(c for c in body["clients"] if c["id"] == "adam")
    assert adam_state["provider_waiting"] == 1
    assert adam_state["provider_daily_quota_used"] is True
    assert any("daily quota" in p for p in body["problems"])


def test_health_of_an_empty_library_is_a_normal_answer(api):
    api.post("/api/clients", json={"name": "Knversion"})
    body = api.get("/api/health").json()
    empty = next(c for c in body["clients"] if c["id"] == "knversion")
    assert empty["shots"] == 0
    assert not any("knversion" in p and "empty" in p for p in body["problems"])


# -- adding footage from another app -------------------------------------------


def test_a_dropped_file_sent_by_another_app_is_queued_for_its_client(api, libraries):
    _, luqman = libraries
    response = api.post(
        "/api/upload?client=luqman",
        files=[("files", ("deal.mp4", b"pretend video", "video/mp4")),
               ("files", ("notes.txt", b"not footage", "text/plain"))],
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["client"] == "luqman"
    assert body["queued"] == 1
    assert body["skipped"] == ["notes.txt"]
    assert (luqman.staging_dir / "deal.mp4").exists()

    queue = api.get("/api/queue?client=luqman").json()
    assert queue["queued"] == 1
    assert queue["jobs"][0]["filename"] == "deal.mp4"
    assert queue["jobs"][0]["status"] == "queued"
    assert api.get("/api/queue?client=adam").json()["jobs"] == []


def test_an_upload_must_name_the_client_when_there_are_several(api):
    response = api.post("/api/upload", files=[("files", ("a.mp4", b"x", "video/mp4"))])
    assert response.status_code == 400
