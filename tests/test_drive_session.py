"""Filing clip after clip must not re-walk the client's whole tree each time.

Regression: three clips took 464 seconds, because every clip opened a fresh
Drive client and re-checked all 53 of Adam's folders.
"""

from __future__ import annotations

import time

from broll.db.models import Shot, Source
from broll.db.store import new_id
from broll.drive.session import DriveSession
from tests.fakes import FakeDriveClient
from tests.test_client_tree import tree_config


def _seed(store, workspace, tmp_path, categories):
    ids = []
    for index, category in enumerate(categories):
        local = tmp_path / f"clip{index}.mp4"
        local.write_bytes(b"video")
        source = store.insert_source(Source(
            id=new_id(), workspace_id=workspace.id, content_hash=f"hash{index:08d}",
            original_filename=local.name, origin="local", origin_path=str(local)))
        store.insert_shot(Shot(
            id=f"{source.id}-0", workspace_id=workspace.id, source_id=source.id,
            caption="A clip.", category=category, action="breathing",
            emotions=["calm"], status="indexed"))
        store.recompute_source_status(source.id)
        ids.append(source.id)
    return ids


def _counting_fake():
    fake = FakeDriveClient()
    calls: list[str] = []
    real = fake.list_children

    def counting(parent_id, refresh=False):
        calls.append(parent_id)
        return real(parent_id, refresh)

    fake.list_children = counting
    return fake, calls


def test_the_second_clip_does_not_rewalk_the_tree(workspace, store, tmp_path):
    workspace.taxonomy = tree_config()
    ids = _seed(store, workspace, tmp_path, [
        "01_Nervous System Practices/Breathwork", "02_Gym & Training"])

    fake, calls = _counting_fake()
    session = DriveSession(workspace)
    session._client = fake

    session.organise(ids[0])
    first = len(calls)
    session.organise(ids[1])
    second = len(calls) - first

    assert first >= 5, "the first clip builds the tree"
    assert second <= 2, f"the second clip re-walked the tree ({second} listings)"
    files = [p for p in fake.tree() if p.endswith(".mp4")]
    assert len(files) == 2


def test_the_session_forgets_its_caches_after_the_ttl(workspace):
    session = DriveSession(workspace, ttl_s=0.0)
    session.folder_ids["a"] = "b"
    session.root_id = "root-id"
    session.tree_ready = True
    time.sleep(0.01)
    session._expire_if_stale()
    assert session.folder_ids == {} and session.root_id is None and not session.tree_ready


def test_a_dry_run_never_writes_placeholders_into_a_session(workspace, store, tmp_path):
    from broll.drive.organizer import Organizer

    workspace.taxonomy = tree_config()
    ids = _seed(store, workspace, tmp_path, ["02_Gym & Training"])
    session = DriveSession(workspace)
    Organizer(workspace, store, FakeDriveClient(), dry_run=True, session=session).organise_source(ids[0])
    assert session.folder_ids == {} and session.root_id is None
