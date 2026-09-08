"""The organiser must be idempotent, and reorganise must rebuild from the DB."""

from __future__ import annotations

from pathlib import Path

import pytest

from broll.db.models import Source
from broll.db.store import new_id
from broll.drive.organizer import Organizer
from tests.fakes import FakeDriveClient
from tests.test_search import seed_library


@pytest.fixture()
def organised(workspace, store, tmp_path):
    """A seeded library whose sources have real local files to 'upload'."""
    seed_library(store)
    for source in store.list_sources(limit=100):
        local = tmp_path / source.original_filename
        local.write_bytes(b"video")
        store.update_source(source.id, origin_path=str(local), filesize_bytes=5)
    workspace.drive_root_folder_id = None
    client = FakeDriveClient()
    return workspace, store, client


def test_first_pass_uploads_and_creates_shortcuts(organised):
    workspace, store, client = organised
    organizer = Organizer(workspace, store, client)
    report = organizer.reorganise()

    assert report.errors == []
    assert report.sources == len(store.list_sources(limit=100))

    tree = client.tree()
    assert any(p.startswith("B-Roll/_Library/") for p in tree)
    assert any(p.startswith("B-Roll/By Setting/Beach") for p in tree)
    assert any(p.startswith("B-Roll/By Shot Type/Aerial") for p in tree)

    uploads = [a for a in report.actions if a.kind == "upload"]
    assert len(uploads) == 14, "one canonical copy per source"
    assert len(client.shortcuts()) > len(uploads), "each clip appears in many folders"


def test_second_pass_performs_zero_writes(organised):
    """The M3 acceptance criterion."""
    workspace, store, client = organised
    Organizer(workspace, store, client).reorganise()

    writes_after_first = client.writes
    assert writes_after_first > 0

    second = Organizer(workspace, store, client).reorganise()
    assert client.writes == writes_after_first, f"second pass wrote: {client.calls[writes_after_first:]}"
    assert second.writes == 0
    assert second.errors == []


def test_reorganise_rebuilds_an_emptied_tree_identically(organised):
    workspace, store, client = organised
    Organizer(workspace, store, client).reorganise()
    before = client.tree()

    # Empty the facet tree, keeping _Library and the real files: every
    # shortcut, and every "By ..." folder, goes.
    for entry in list(client._files.values()):
        if entry.is_shortcut or (entry.is_folder and entry.name.startswith("By ")):
            del client._files[entry.id]

    assert client.tree() != before

    Organizer(workspace, store, client).reorganise()
    assert client.tree() == before


def test_dry_run_touches_nothing(organised):
    workspace, store, client = organised
    report = Organizer(workspace, store, client, dry_run=True).reorganise()

    assert report.writes > 0, "the plan should be non-empty"
    assert client.writes == 0, "dry run must not write"
    assert client.tree() == set()
    assert store.list_sources(limit=100)[0].drive_file_id is None


def test_filename_change_is_a_rename_not_a_second_upload(organised):
    workspace, store, client = organised
    Organizer(workspace, store, client).reorganise()
    source = next(
        s for s in store.list_sources(limit=100)
        if s.original_filename == "beach_meditation_sunrise.mp4"
    )
    shot = store.shots_for_source(source.id)[0]

    store.set_shot_fields(shot.id, setting="office")
    store.recompute_search_text(shot.id)

    report = Organizer(workspace, store, client).organise_source(source.id)
    kinds = [a.kind for a in report.actions]
    assert "rename" in kinds
    assert "upload" not in kinds


def test_a_shortcut_that_is_no_longer_justified_is_removed(organised):
    workspace, store, client = organised
    # Named explicitly: this clip is 'aerial', so re-facetting it to close_up
    # is guaranteed to move it.
    source = next(
        s for s in store.list_sources(limit=100)
        if s.original_filename == "mountain_drone_flyover.mp4"
    )
    Organizer(workspace, store, client).organise_source(source.id)
    assert any("By Shot Type/Aerial" in p for p in client.tree())

    shot = store.shots_for_source(source.id)[0]
    store.set_shot_fields(shot.id, shot_type="close_up")

    report = Organizer(workspace, store, client).organise_source(source.id)
    assert any(a.kind == "shortcut" for a in report.actions)
    assert any(a.kind == "delete_shortcut" for a in report.actions)

    paths = client.tree()
    assert any("By Shot Type/Close Up" in p for p in paths)
    assert not any(p.endswith("mp4") and "By Shot Type/Aerial" in p for p in paths)


def test_the_organiser_never_deletes_a_non_shortcut(organised):
    workspace, store, client = organised
    Organizer(workspace, store, client).reorganise()
    real_file = next(f for f in client._files.values() if f.mime_type == "video/mp4")
    with pytest.raises(Exception):
        client.delete_shortcut(real_file.id)


def test_needs_review_shots_land_in_the_review_folder(organised):
    workspace, store, client = organised
    source = store.list_sources(limit=100)[0]  # any source will do
    shot = store.shots_for_source(source.id)[0]
    store.set_shot_fields(shot.id, status="needs_review")

    Organizer(workspace, store, client).organise_source(source.id)
    assert any("_Needs Review" in p for p in client.tree())
