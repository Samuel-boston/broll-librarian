"""`broll organise --shortcuts-only`: a tree of shortcuts to untouched originals."""

from __future__ import annotations

import pytest

from broll.db.models import Shot, Source
from broll.db.store import new_id
from broll.drive.client import DriveError
from broll.drive.shortcuts_only import (
    ShortcutsOnlyOrganizer,
    ShortcutsOnlyViolation,
    _GuardedClient,
)
from tests.fakes import FakeDriveClient
from tests.test_client_tree import tree_config

MED = "01_Nervous System Practices/Meditation & Stillness"
GYM = "02_Gym & Training"
BEACH = "05_Travel & Adventure/Beach & Water"


class SpyDrive(FakeDriveClient):
    """Fails the test if anything but list/create_folder/create_shortcut is called."""

    def _forbidden(self, *a, **k):
        raise AssertionError("an original was touched")

    move = rename = copy_file = upload = download = delete_shortcut = _forbidden


def add_source(store, client, name, shots, content_hash=None, folder="srcfolder"):
    original = client.add_file(name, folder)
    source = store.insert_source(Source(
        id=new_id(), workspace_id=store.workspace_id, content_hash=content_hash or name,
        original_filename=name, origin="drive", drive_file_id=original.id,
        source_alt_id="copy-should-never-be-used"))
    for i, (category, secondary, start, primary) in enumerate(shots):
        store.insert_shot(Shot(
            id=f"{source.id}-{i}", workspace_id=store.workspace_id, source_id=source.id,
            shot_index=i, is_primary=primary, start_s=start, end_s=start + 5, duration_s=5,
            status="indexed", category=category, secondary_categories=secondary, action="walking",
            emotions=["calm"]))
    store.recompute_source_status(source.id)
    return source, original


@pytest.fixture()
def lib(workspace, store):
    workspace.taxonomy = tree_config()
    workspace.ingest.shortcuts_writes_per_second = 1000.0
    client = SpyDrive()
    client.add_folder("Adam", "root")
    a, orig_a = add_source(store, client, "IMG_0001.MOV", [(MED, [GYM], 0.0, True)])
    b, orig_b = add_source(store, client, "IMG_0002.MOV",
                           [(GYM, [], 0.0, True), (BEACH, [], 42.0, False)])
    return workspace, store, client, [orig_a, orig_b]


def run(workspace, store, client, **kw):
    kw.setdefault("sleep", lambda s: None)
    return ShortcutsOnlyOrganizer(workspace, store, client, **kw).run()


def test_creates_new_root_tree_and_shortcuts_to_originals(lib):
    workspace, store, client, originals = lib
    before = {o.id: (o.name, tuple(o.parents)) for o in originals}
    report = run(workspace, store, client)
    assert report.errors == [] and not report.aborted

    tree = client.tree()
    root = "B-Roll Library (Organised)"
    assert root in tree
    assert f"{root}/{MED}" in tree and f"{root}/{BEACH}" in tree  # whole tree, incl. empty folders
    targets = {s.shortcut_target_id for s in client.shortcuts()}
    assert targets == {o.id for o in originals}, "only ORIGINAL ids, never the copy"
    # primary home + one secondary for A; home + beach (timecoded) for B
    assert len(client.shortcuts()) == 4
    assert any(p.endswith("_at_0m42s.mov") and BEACH in p for p in tree)
    assert any(p.startswith(f"{root}/{MED}/") for p in tree)
    # originals: same name, same parent
    assert before == {o.id: (client.get(o.id).name, tuple(client.get(o.id).parents)) for o in originals}
    assert all(c.split(":")[0] in {"create_folder", "shortcut"} for c in client.calls)


def test_second_run_writes_nothing(lib):
    workspace, store, client, _ = lib
    run(workspace, store, client)
    writes = client.writes
    again = run(workspace, store, client)
    assert client.writes == writes
    assert again.write_calls == 0 and again.created_shortcuts == 0 and again.already_done == 4


def test_dry_run_prints_plan_and_touches_nothing(lib):
    workspace, store, client, _ = lib
    report = run(workspace, store, client, dry_run=True)
    assert client.writes == 0
    assert report.root_to_create == 1 and len(report.shortcuts) == 4
    assert report.write_calls == 1 + len(report.folders_to_create) + 4
    assert store.shortcuts_for_source(report.shortcuts[0].source_id) == []


def test_resumes_after_a_stop(lib):
    workspace, store, client, _ = lib
    first = run(workspace, store, client, limit=6)
    assert first.aborted and client.writes == 6
    run(workspace, store, client)
    assert len(client.shortcuts()) == 4
    assert len(client.shortcuts()) == len({(tuple(s.parents), s.name) for s in client.shortcuts()})


def test_adopts_shortcuts_when_the_db_record_was_lost(lib):
    workspace, store, client, _ = lib
    run(workspace, store, client)
    store.conn.execute("DELETE FROM drive_shortcuts")
    writes = client.writes
    run(workspace, store, client)
    assert client.writes == writes and len(client.shortcuts()) == 4


def test_duplicate_content_gets_distinct_names(workspace, store):
    workspace.taxonomy = tree_config()
    workspace.ingest.shortcuts_writes_per_second = 1000.0
    client = SpyDrive()
    client.add_folder("Adam", "root")
    for n, h in (("a.MOV", "samehash0000"), ("b.MOV", "samehash0001")):  # same 8-char suffix -> same name
        add_source(store, client, n, [(GYM, [], 0.0, True)], content_hash=h)
    report = run(workspace, store, client)
    assert report.errors == []
    names = [s.name for s in client.shortcuts()]
    assert len(names) == 2 and len(set(names)) == 2


def test_stops_after_consecutive_errors(lib, monkeypatch):
    workspace, store, client, _ = lib
    workspace.ingest.shortcuts_max_consecutive_errors = 3

    def boom(*a, **k):
        raise DriveError("500 backend")

    monkeypatch.setattr(client, "create_shortcut", boom)
    report = run(workspace, store, client)
    assert report.aborted and "3 failed writes in a row" in report.errors[-1]
    assert client.shortcuts() == []


def test_quota_error_stops_at_once(lib, monkeypatch):
    workspace, store, client, _ = lib

    def quota(*a, **k):
        raise DriveError("downloadQuotaExceeded: limit reached")

    monkeypatch.setattr(client, "create_shortcut", quota)
    calls = []
    monkeypatch.setattr(client, "create_shortcut",
                        lambda *a, **k: (calls.append(1), quota())[1])
    report = run(workspace, store, client)
    assert report.aborted and len(calls) == 1


def test_one_failure_is_skipped_and_retried_next_run(lib, monkeypatch):
    workspace, store, client, _ = lib
    real = client.create_shortcut
    state = {"n": 0}

    def flaky(*a, **k):
        state["n"] += 1
        if state["n"] == 2:
            raise DriveError("404 target")
        return real(*a, **k)

    monkeypatch.setattr(client, "create_shortcut", flaky)
    first = run(workspace, store, client)
    assert not first.aborted and len(first.errors) == 1 and len(client.shortcuts()) == 3
    run(workspace, store, client)
    assert len(client.shortcuts()) == 4


def test_write_rate_is_limited(lib):
    workspace, store, client, _ = lib
    workspace.ingest.shortcuts_writes_per_second = 4.0
    sleeps = []
    run(workspace, store, client, sleep=sleeps.append)
    assert sleeps and all(0 < s <= 0.25 + 1e-6 for s in sleeps)


def test_guard_blocks_every_method_that_could_touch_an_original():
    guard = _GuardedClient(FakeDriveClient())
    for name in ("move", "rename", "copy_file", "upload", "download", "delete_shortcut", "create_doc"):
        with pytest.raises(ShortcutsOnlyViolation):
            getattr(guard, name)


def test_mode_never_calls_move_rename_or_update(lib):
    """SpyDrive raises if any of these is reached; the run must also record none."""
    workspace, store, client, originals = lib
    run(workspace, store, client)
    run(workspace, store, client, dry_run=True)
    assert not any(c.split(":")[0] in {"move", "rename", "upload", "copy", "delete_shortcut"} for c in client.calls)
    assert all(client.get(o.id).is_shortcut is False for o in originals)
