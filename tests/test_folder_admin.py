"""Changing the client's folder tree: rename, add, change what a folder is for."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from broll.cli import app
from broll.config import load_workspace_config
from broll.db.store import Store
from broll.review import CorrectionError, add_folder, remove_folder, rename_folder, set_folder_note
from tests.test_client_tree import tree_config
from tests.test_precision import add_shot

runner = CliRunner()


@pytest.fixture()
def tree(workspace, store):
    workspace.taxonomy = tree_config()
    workspace.save()
    return workspace


def test_renaming_a_folder_carries_its_clips_with_it(tree, store):
    fire = add_shot(store, "a.mp4", caption="x", category="07_Cinematic & Mood/Reflective",
                    secondary_categories=["07_Cinematic & Mood/Struggle & Upset", "02_Gym & Training"])
    other = add_shot(store, "b.mp4", caption="y", category="02_Gym & Training")
    store.record_folder_proposal("07_Cinematic & Mood/Moods", None, fire)

    done = rename_folder(tree, store, "07_Cinematic & Mood", "07_Moods & Atmosphere")

    assert done == {"old": "07_Cinematic & Mood", "new": "07_Moods & Atmosphere", "clips": 1}
    moved = store.get_shot(fire)
    assert moved.category == "07_Moods & Atmosphere/Reflective"
    assert moved.secondary_categories == ["07_Moods & Atmosphere/Struggle & Upset", "02_Gym & Training"]
    assert store.get_shot(other).category == "02_Gym & Training", "unrelated clips are left alone"
    assert store.list_folder_proposals()[0]["path"] == "07_Moods & Atmosphere/Moods"
    paths = dict(load_workspace_config(tree.id).taxonomy.category_leaves())
    assert "07_Moods & Atmosphere/Reflective" in paths and "07_Cinematic & Mood/Reflective" not in paths


def test_renaming_a_leaf_updates_the_words_clips_are_searched_by(tree, store):
    shot = add_shot(store, "a.mp4", caption="x", category="02_Gym & Training")
    rename_folder(tree, store, "02_Gym & Training", "02_Training")
    row = store.conn.execute("SELECT concept_text FROM shots WHERE id = ?", (shot,)).fetchone()
    assert "Training" in row["concept_text"] and "Gym" not in row["concept_text"]


def test_a_rename_is_refused_when_it_would_collide_or_is_not_a_name(tree, store):
    with pytest.raises(CorrectionError, match="already a folder"):
        rename_folder(tree, store, "02_Gym & Training", "01_Nervous System Practices")
    with pytest.raises(CorrectionError, match="no folder"):
        rename_folder(tree, store, "Nope", "x")
    with pytest.raises(CorrectionError, match="can't be empty"):
        rename_folder(tree, store, "02_Gym & Training", "a/b")


def test_a_folder_can_be_added_at_the_top_or_inside_another(tree):
    top = add_folder(tree, "", "09_Landscaping", "Lawns and gardens.")
    inner = add_folder(tree, "07_Cinematic & Mood", "Windows", "Looking out of one.")
    leaves = dict(load_workspace_config(tree.id).taxonomy.category_leaves())
    assert top == "09_Landscaping" and leaves["09_Landscaping"].startswith("Lawns and gardens")
    assert inner == "07_Cinematic & Mood/Windows" and inner in leaves
    with pytest.raises(CorrectionError, match="already a folder"):
        add_folder(tree, "", "09_Landscaping")
    with pytest.raises(CorrectionError, match="no folder"):
        add_folder(tree, "Nope", "x")


def test_what_a_folder_is_for_can_be_changed(tree):
    set_folder_note(tree, "02_Gym & Training", "Weights only.")
    assert dict(load_workspace_config(tree.id).taxonomy.category_leaves())["02_Gym & Training"] == "Weights only."


def test_the_commands(cli_workspace_with_tree):
    ws = cli_workspace_with_tree
    out = runner.invoke(app, ["folders", "add", "07_Cinematic & Mood", "Windows", "-n", "Looking out.", "-w", ws.id])
    assert out.exit_code == 0 and "Added 07_Cinematic & Mood/Windows" in out.stdout
    out = runner.invoke(app, ["folders", "note", "07_Cinematic & Mood/Windows", "Looking out of a window.", "-w", ws.id])
    assert out.exit_code == 0
    out = runner.invoke(app, ["folders", "rename", "07_Cinematic & Mood", "07_Moods", "-w", ws.id])
    assert out.exit_code == 0 and "Renamed 07_Cinematic & Mood to 07_Moods" in out.stdout
    leaves = dict(load_workspace_config(ws.id).taxonomy.category_leaves())
    assert leaves["07_Moods/Windows"].startswith("Looking out of a window")
    assert runner.invoke(app, ["folders", "rename", "Nope", "x", "-w", ws.id]).exit_code == 1


@pytest.fixture()
def cli_workspace_with_tree(broll_home):
    assert runner.invoke(app, ["init", "--name", "Fixtures", "--provider", "mock"]).exit_code == 0
    ws = load_workspace_config("fixtures")
    ws.taxonomy = tree_config()
    ws.save()
    return ws


def test_removing_a_folder_moves_its_clips_where_asked(tree, store):
    a = add_shot(store, "a.mp4", caption="x", category="07_Cinematic & Mood/Reflective", status="indexed")
    b = add_shot(store, "b.mp4", caption="y", category="02_Gym & Training",
                 secondary_categories=["07_Cinematic & Mood/Reflective"])
    done = remove_folder(tree, store, "07_Cinematic & Mood/Reflective", move_to="05_Travel & Adventure/Beach & Water")
    assert done["clips"] == 2
    assert store.get_shot(a).category == "05_Travel & Adventure/Beach & Water"
    assert store.get_shot(a).status == "indexed"
    assert store.get_shot(b).category == "02_Gym & Training"
    assert store.get_shot(b).secondary_categories == ["05_Travel & Adventure/Beach & Water"]
    assert "07_Cinematic & Mood/Reflective" not in dict(load_workspace_config(tree.id).taxonomy.category_leaves())


def test_removing_a_folder_without_a_destination_flags_its_clips_for_a_person(tree, store):
    a = add_shot(store, "a.mp4", caption="x", category="07_Cinematic & Mood/Reflective", status="indexed")
    remove_folder(tree, store, "07_Cinematic & Mood/Reflective")
    shot = store.get_shot(a)
    assert shot.category is None and shot.status == "needs_review" and "category_unmatched" in shot.review_reasons


def test_a_folder_with_folders_inside_it_cannot_be_removed_and_so_cannot_be_a_destination(tree, store):
    with pytest.raises(CorrectionError, match="has folders inside"):
        remove_folder(tree, store, "05_Travel & Adventure")
    with pytest.raises(CorrectionError, match="not a folder clips can be filed in"):
        remove_folder(tree, store, "07_Cinematic & Mood/Reflective", move_to="01_Nervous System Practices")
    with pytest.raises(CorrectionError, match="no folder"):
        remove_folder(tree, store, "Nope")


def test_a_top_level_folder_can_be_removed(tree, store):
    remove_folder(tree, store, "02_Gym & Training")
    assert "02_Gym & Training" not in dict(load_workspace_config(tree.id).taxonomy.category_leaves())
