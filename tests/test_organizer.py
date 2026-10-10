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


def test_dry_run_on_an_empty_drive_never_queries_placeholder_ids(organised):
    """Regression: the first real dry run 404'd asking Drive to list the
    children of a folder that only existed in the plan."""
    workspace, store, client = organised
    assert client.tree() == set(), "precondition: nothing in Drive yet"

    report = Organizer(workspace, store, client, dry_run=True).reorganise()

    assert report.errors == []
    assert any(a.kind == "create_folder" and a.path == "B-Roll" for a in report.actions)
    assert any(a.kind == "upload" for a in report.actions)
    assert client.writes == 0


def test_a_full_drive_stops_cleanly_with_a_useful_message(organised):
    """Found on the first real run: a full Drive crashed with a traceback after
    retrying a non-retryable 403 for 40 seconds."""
    from broll.drive.client import DriveStorageFullError

    workspace, store, client = organised

    def full(path, name, parent_id, app_properties=None):
        raise DriveStorageFullError("This Google Drive is full")

    client.upload = full
    report = Organizer(workspace, store, client).reorganise()

    assert report.aborted
    assert len(report.errors) == 1, "it should stop at the first full-Drive error"
    assert "full" in report.errors[0]


def test_storage_quota_403_is_not_retried_but_rate_limit_403_is():
    from broll.drive.client import _is_retryable

    class Resp:
        status = 403

    class Err(Exception):
        def __init__(self, reason):
            self.resp = Resp()
            self.error_details = [{"reason": reason}]

    assert _is_retryable(Err("userRateLimitExceeded"))
    assert not _is_retryable(Err("storageQuotaExceeded"))


def test_only_files_the_library_made_are_filed_when_asked(organised):
    """Index the original footage folder by mistake and nothing in it is renamed or moved."""
    workspace, store, client = organised
    from broll.drive.copier import PROVENANCE_KEY

    Organizer(workspace, store, client).reorganise()  # the library uploads its own files, stamped
    workspace.ingest.organise_only_copies = True
    source = next(s for s in store.list_sources(limit=10) if s.drive_file_id)
    entry = client.get(source.drive_file_id)
    assert entry.app_properties and PROVENANCE_KEY in entry.app_properties
    entry.app_properties = None  # an original, not a copy
    entry.name = "IMG_0001.MOV"  # still carrying its camera name: filing would rename it
    report = Organizer(workspace, store, client).organise_source(source.id)
    assert any("no copy stamp" in e for e in report.errors)
    assert not any(a.kind in ("rename", "move") for a in report.actions)
    entry.app_properties = {PROVENANCE_KEY: "origin-id"}
    again = Organizer(workspace, store, client).organise_source(source.id)
    assert not any("no copy stamp" in e for e in again.errors)


def test_a_dry_run_counts_the_moves_into_folders_that_do_not_exist_yet(organised):
    workspace, store, client = organised
    first = Organizer(workspace, store, client, dry_run=True).reorganise()
    real = Organizer(workspace, store, client).reorganise()
    assert [a.kind for a in first.actions].count("move") == [a.kind for a in real.actions].count("move")


def test_ten_failures_in_a_row_stop_the_run_with_a_clear_message(organised):
    workspace, store, client = organised

    def broken(path, name, parent_id, app_properties=None):
        raise RuntimeError("503 backendError")

    client.upload = broken
    report = Organizer(workspace, store, client).reorganise()
    assert report.aborted or len(report.errors) <= 11
