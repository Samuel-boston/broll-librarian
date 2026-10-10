"""`broll folders auto`: after the library is filed, a suggested folder is created only when enough different
files belong in it. Near-duplicate suggestions are one folder; a suggestion that matches a folder already in
the tree moves its clips there; a lone clip's suggestion creates nothing."""

from __future__ import annotations

from broll.db.models import Shot, Source
from broll.db.store import new_id
from broll.review import create_wanted_folders, plan_new_folders
from tests.test_client_tree import tree_config

PARENT = "05_Travel & Adventure"
START = "05_Travel & Adventure/Beach & Water"


def add_clip(workspace, store, name: str, suggested: str | None, shots: int = 1) -> list[str]:
    """A clip filed in Beach & Water whose analysis suggested `suggested`; `shots` shots from one file."""
    source = store.insert_source(Source(
        id=new_id(), workspace_id=workspace.id, content_hash=new_id(), original_filename=name,
        origin="local", origin_path=f"/x/{name}", status="indexed"))
    ids = []
    for n in range(shots):
        shot = store.insert_shot(Shot(
            id=f"{source.id}-{n}", workspace_id=workspace.id, source_id=source.id, shot_index=n,
            start_s=n * 3.0, end_s=n * 3.0 + 3.0, duration_s=3.0, category=START, status="indexed"))
        ids.append(shot.id)
        if suggested:
            store.record_folder_proposal(f"{PARENT}/{suggested}", "Water sports on a board.", shot.id)
    return ids


def test_spellings_of_one_folder_add_up_and_the_folder_is_made(workspace, store):
    workspace.taxonomy = tree_config()
    for i, spelling in enumerate(["Jet Ski", "Jetski", "Jet Skiing", "Jet Ski", "Jetskis"]):
        add_clip(workspace, store, f"jet_{i}.mov", spelling)
    assert len(store.list_folder_proposals()) == 4, "four spellings, so four suggestions"

    [plan] = plan_new_folders(workspace, store)
    assert plan.files == 5 and plan.ready and not plan.existing
    assert plan.path == f"{PARENT}/Jet Ski", "named after the spelling most clips used"

    done = create_wanted_folders(workspace, store)
    assert done["created"] == [(f"{PARENT}/Jet Ski", 5)] and len(done["moved"]) == 5
    assert all(store.get_shot(i).category == f"{PARENT}/Jet Ski" for i in done["moved"])
    assert f"{PARENT}/Jet Ski" in dict(workspace.taxonomy.category_leaves())
    assert store.list_folder_proposals() == [], "every merged suggestion is closed"


def test_too_few_files_create_nothing_and_the_clips_stay_put(workspace, store):
    workspace.taxonomy = tree_config()
    for i in range(2):
        add_clip(workspace, store, f"kite_{i}.mov", "Kitesurfing")
    before = dict(workspace.taxonomy.category_leaves())
    done = create_wanted_folders(workspace, store)
    assert done["created"] == [] and done["waiting"] == [(f"{PARENT}/Kitesurfing", 2)]
    assert dict(workspace.taxonomy.category_leaves()) == before
    assert len(store.list_folder_proposals()) == 1, "still a suggestion, for later"
    assert all(s.category == START for s in store.list_shots())


def test_many_shots_of_one_file_are_one_thing(workspace, store):
    """A long clip cut into six shots is still one file: it does not earn a folder alone."""
    workspace.taxonomy = tree_config()
    add_clip(workspace, store, "long_wake.mov", "Wakeboarding", shots=6)
    [plan] = plan_new_folders(workspace, store)
    assert plan.files == 1 and not plan.ready


def test_the_threshold_can_be_changed_per_run_and_in_the_config(workspace, store):
    workspace.taxonomy = tree_config()
    for i in range(3):
        add_clip(workspace, store, f"surf_{i}.mov", "Surfing")
    assert not plan_new_folders(workspace, store)[0].ready
    assert plan_new_folders(workspace, store, min_files=3)[0].ready
    workspace.taxonomy.new_folder_min_clips = 3
    assert plan_new_folders(workspace, store)[0].ready


def test_a_suggestion_that_matches_a_real_folder_moves_its_clips_there_even_if_few(workspace, store):
    workspace.taxonomy = tree_config()
    add_clip(workspace, store, "one.mov", "Beach Water")   # spelt differently from "Beach & Water"
    [plan] = plan_new_folders(workspace, store)
    assert plan.existing and plan.ready and plan.path == START
    done = create_wanted_folders(workspace, store)
    assert done["created"] == [(START, 1)]
    assert len(dict(workspace.taxonomy.category_leaves())) == len(dict(tree_config().category_leaves()))


def test_a_dry_run_changes_nothing(workspace, store):
    workspace.taxonomy = tree_config()
    for i in range(5):
        add_clip(workspace, store, f"s_{i}.mov", "Sailing")
    done = create_wanted_folders(workspace, store, dry_run=True)
    assert done["created"] == [(f"{PARENT}/Sailing", 5)] and done["moved"] == []
    assert len(store.list_folder_proposals()) == 1
    assert f"{PARENT}/Sailing" not in dict(workspace.taxonomy.category_leaves())


def test_nothing_suggested_is_nothing_to_do(workspace, store):
    workspace.taxonomy = tree_config()
    assert create_wanted_folders(workspace, store) == {"created": [], "waiting": [], "moved": [], "sources": []}
