"""Corrections, the review queue, and the settings screen."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from broll.db.models import Shot, Source
from broll.db.store import new_id
from broll.review import CorrectionError, apply_correction, parse_list, review_queue
from broll.search.filters import SearchFilters
from broll.search.query import SearchEngine
from broll.web.app import create_app


@pytest.fixture()
def flagged(store, workspace):
    source = store.insert_source(
        Source(id=new_id(), workspace_id=workspace.id, content_hash="h1",
               original_filename="mystery.mp4", origin="local", duration_s=8.0)
    )
    shot = Shot(
        id=f"{source.id}-0", workspace_id=workspace.id, source_id=source.id,
        shot_index=0, is_primary=True, duration_s=8.0, end_s=8.0,
        caption="Something out of focus, hard to tell.",
        setting="indoors", shot_type="medium", camera_movement="handheld",
        time_of_day="unknown", colour_profile="neutral", people_count="none",
        pace="moderate", quality_flags=["out_of_focus"], confidence=0.22,
        tags=["blurry"], status="needs_review",
    )
    store.insert_shot(shot)
    store.recompute_source_status(source.id)
    return workspace, store, shot


def test_review_queue_explains_why_each_shot_is_flagged(flagged):
    workspace, store, shot = flagged
    queue = review_queue(store)
    assert len(queue) == 1
    assert queue[0]["filename"] == "mystery.mp4"
    reasons = " ".join(queue[0]["reasons"])
    assert "out_of_focus" in reasons and "low confidence" in reasons


def test_a_correction_updates_facets_search_text_and_status(flagged):
    workspace, store, shot = flagged
    corrected = apply_correction(
        workspace, store, shot.id,
        {
            "caption": "A cyclist crests a hill at golden hour.",
            "setting": "mountain", "action": "cycling", "shot_type": "wide",
            "time_of_day": "golden_hour", "mood": "energetic, adventurous",
            "tags": "cycling, bike, hill, endurance", "quality_flags": "",
        },
    )
    assert corrected.status == "indexed"
    assert corrected.setting == "mountain"
    assert corrected.mood == ["energetic", "adventurous"]
    assert corrected.quality_flags == []
    assert "cycling" in corrected.search_text and "golden hour" in corrected.search_text
    assert store.get_source(shot.source_id).status == "indexed"
    assert corrected.raw_analysis is None or corrected.raw_analysis.get("corrected_by_operator")


def test_a_corrected_shot_is_immediately_findable(flagged):
    workspace, store, shot = flagged
    apply_correction(workspace, store, shot.id,
                     {"caption": "A cyclist crests a hill.", "tags": "cycling, bike"})
    results = SearchEngine(store, embedder=None).search("cycling", SearchFilters(), 5)
    assert [r.shot.id for r in results] == [shot.id]


def test_an_invalid_enum_is_refused_rather_than_silently_dropped(flagged):
    workspace, store, shot = flagged
    with pytest.raises(CorrectionError):
        apply_correction(workspace, store, shot.id, {"shot_type": "banana"})
    assert store.get_shot(shot.id).shot_type == "medium"


def test_correcting_a_missing_shot_is_an_error(flagged):
    workspace, store, _ = flagged
    with pytest.raises(CorrectionError):
        apply_correction(workspace, store, "nope", {"caption": "x"})


def test_parse_list_normalises():
    assert parse_list("Calm,  Peaceful ,,") == ["calm", "peaceful"]
    assert parse_list(["A", "b"]) == ["a", "b"]
    assert parse_list(None) == []


# -- web -------------------------------------------------------------------


def test_review_screen_lists_and_saves(flagged):
    workspace, store, shot = flagged
    store.close()
    with TestClient(create_app(workspace, run_worker=False)) as client:
        page = client.get("/review")
        assert page.status_code == 200
        assert "mystery.mp4" in page.text
        assert "out_of_focus" in page.text

        saved = client.post(
            f"/review/{shot.id}",
            data={"caption": "A cyclist crests a hill.", "setting": "mountain",
                  "shot_type": "wide", "tags": "cycling, bike", "quality_flags": "",
                  "status": "indexed"},
        )
        assert saved.status_code == 200
        assert "Nothing needs review" in saved.text

        bad = client.post(f"/review/{shot.id}", data={"shot_type": "banana"})
        assert bad.status_code == 400


def test_settings_screen_saves_config_but_never_echoes_a_key(workspace):
    # The broll_home fixture already points BROLL_HOME at a temp directory,
    # so the .env this writes is isolated.
    with TestClient(create_app(workspace, run_worker=False)) as client:
        page = client.get("/settings")
        assert page.status_code == 200
        assert "Vocabulary candidates" in page.text

        saved = client.post("/settings", data={
            "provider": "mock", "model": "", "drive_root_folder_id": "",
            "drive_local_mount_path": "/Volumes/GoogleDrive/My Drive",
            "min_clips_for_subfolder": 8, "max_folders_per_level": 30,
            "words_per_minute": 160, "concurrency": 2,
        })
        assert saved.status_code == 200
        assert "/Volumes/GoogleDrive/My Drive" in saved.text

        from broll.config import load_workspace_config

        reloaded = load_workspace_config(workspace.id)
        assert reloaded.taxonomy.min_clips_for_subfolder == 8
        assert reloaded.transcript.words_per_minute == 160

        response = client.post("/settings/key",
                               data={"provider": "openai", "api_key": "sk-secret-value"})
        assert response.status_code == 200
        assert "sk-secret-value" not in response.text

        from broll.config import broll_home

        env = (broll_home() / ".env").read_text()
        assert "OPENAI_API_KEY=sk-secret-value" in env
        assert (broll_home() / ".env").stat().st_mode & 0o777 == 0o600


def test_settings_promotes_a_vocabulary_candidate(workspace, store):
    store.record_vocabulary_candidates([("subjects", "hydrofoil")] * 3)
    store.close()

    with TestClient(create_app(workspace, run_worker=False)) as client:
        assert "hydrofoil" in client.get("/settings").text
        response = client.post("/settings/vocab",
                               data={"field": "subjects", "term": "hydrofoil"})
        # Jinja escapes the quotes in the flash message, so match on the parts.
        assert "Promoted" in response.text and "hydrofoil" in response.text

    from broll.config import load_workspace_config

    assert "hydrofoil" in load_workspace_config(workspace.id).vocabulary_overrides["subjects"]
