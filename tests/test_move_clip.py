"""A folder lists what is filed in it, and a person can move any clip to another folder by hand.

The model files a clip in one folder and may name up to two more it could also fit. Those used to be listed
in the folder as if they were filed there (a beach clip showed up under Men's Work), which looked like a
mistake. Now they are one click away, and the card has a Move menu that settles it.
"""

from __future__ import annotations

import re
from urllib.parse import quote

from fastapi.testclient import TestClient

from broll.db.models import Shot, Source
from broll.db.store import new_id
from broll.review import CorrectionError, move_shot
from broll.web.app import create_app
from tests.test_client_tree import tree_config

BEACH = "05_Travel & Adventure/Beach & Water"
GYM = "02_Gym & Training"
REFLECTIVE = "07_Cinematic & Mood/Reflective"


def clip(workspace, store, name: str, category: str, secondary: list[str]) -> Shot:
    source = store.insert_source(Source(
        id=new_id(), workspace_id=workspace.id, content_hash=new_id(), original_filename=name,
        origin="local", origin_path=f"/x/{name}", status="indexed"))
    return store.insert_shot(Shot(
        id=f"{source.id}-0", workspace_id=workspace.id, source_id=source.id, duration_s=3, end_s=3,
        caption=f"{name} caption", status="indexed", category=category, secondary_categories=secondary))


def page(workspace, url: str) -> str:
    app = create_app(workspace, run_worker=False)
    with TestClient(app) as client:
        return client.get(url).text


def test_a_folder_lists_only_what_is_filed_in_it_and_offers_the_rest(workspace, store):
    workspace.taxonomy = tree_config()
    clip(workspace, store, "spikeball.mov", BEACH, [GYM])
    clip(workspace, store, "squats.mov", GYM, [])
    store.close()

    gym = page(workspace, f"/library?kind=video&path={quote(GYM)}")
    assert "squats.mov" in gym and "spikeball.mov" not in gym
    assert "1 more might also fit here" in gym

    gym_also = page(workspace, f"/library?kind=video&path={quote(GYM)}&also=1")
    assert "squats.mov" in gym_also and "spikeball.mov" in gym_also
    assert "hide them" in gym_also

    beach = page(workspace, f"/library?kind=video&path={quote(BEACH)}")
    assert "spikeball.mov" in beach and "might also fit" not in beach


def test_moving_a_clip_files_it_there_and_drops_the_models_second_choices(workspace, store):
    workspace.taxonomy = tree_config()
    shot = clip(workspace, store, "spikeball.mov", BEACH, [GYM, REFLECTIVE])
    moved = move_shot(workspace, store, shot.id, REFLECTIVE)
    assert moved.category == REFLECTIVE and moved.secondary_categories == []
    assert moved.status == "indexed" and (moved.raw_analysis or {}).get("corrected_by_operator")


def test_a_clip_cannot_be_moved_to_a_folder_that_does_not_exist(workspace, store):
    workspace.taxonomy = tree_config()
    shot = clip(workspace, store, "a.mov", BEACH, [])
    for bad in ("", "Nowhere", "05_Travel & Adventure", "00_START HERE"):  # a container is not a destination
        try:
            move_shot(workspace, store, shot.id, bad)
        except CorrectionError:
            continue
        raise AssertionError(f"{bad!r} should be refused")
    assert store.get_shot(shot.id).category == BEACH


def test_the_move_button_works_and_a_wrong_folder_says_why(workspace, store):
    workspace.taxonomy = tree_config()
    shot = clip(workspace, store, "spikeball.mov", BEACH, [GYM])
    store.close()
    app = create_app(workspace, run_worker=False)
    with TestClient(app) as client:
        html = client.get(f"/library?kind=video&path={quote(BEACH)}").text
        assert "Move to…" in html and 'class="folder-list"' in html and "02_Gym \\u0026 Training" in html

        done = client.post(f"/shots/{shot.id}/move", data={"category": GYM})
        assert done.status_code == 200 and done.json() == {"ok": True, "path": GYM, "label": "02_Gym & Training"}
        assert "spikeball.mov" not in client.get(f"/library?kind=video&path={quote(BEACH)}").text
        assert "spikeball.mov" in client.get(f"/library?kind=video&path={quote(GYM)}").text

        bad = client.post(f"/shots/{shot.id}/move", data={"category": "Nowhere"})
        assert bad.status_code == 400 and "not one of this library's folders" in bad.json()["detail"]
