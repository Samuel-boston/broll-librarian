"""Copying a Drive folder: the original is never touched, only media is copied, and a repeat or a
resumed run never makes a second copy of anything."""

from __future__ import annotations

import json

import pytest

from broll.drive.client import DriveError, DriveStorageFullError
from broll.drive.copier import copy_folder, human_size, parse_folder_id
from tests.fakes import FakeDriveClient


@pytest.fixture()
def drive():
    fake = FakeDriveClient()
    top = fake.add_folder("Footage", "root")
    gym = fake.add_folder("Gym", top.id)
    deep = fake.add_folder("Squats", gym.id)
    fake.add_file("beach.mov", top.id, "video/quicktime", 5_000)
    fake.add_file("beach.mov", top.id, "video/quicktime", 7_000)       # same name, different file
    fake.add_file("selfie.jpg", gym.id, "image/jpeg", 300)
    fake.add_file("rep1.mp4", deep.id, "video/mp4", 2_000)
    fake.add_file("notes.pdf", top.id, "application/pdf", 50)          # not media
    fake.add_folder("Empty", top.id)
    fake.create_shortcut("src1", "link to beach", top.id)               # shortcut: skipped
    fake.writes = 0
    fake.calls.clear()
    fake.top = top
    return fake


def source_snapshot(fake, top):
    """Everything under the source folder, including ids, names and parents."""
    out = {}
    stack = [top.id]
    while stack:
        pid = stack.pop()
        for f in fake._files.values():
            if pid in f.parents:
                out[f.id] = (f.name, tuple(f.parents), f.size)
                if f.is_folder:
                    stack.append(f.id)
    return out


def run(fake, tmp_path, **kw):
    return copy_folder(fake, fake.top.id, "root", state_path=tmp_path / "state.json", **kw)


def test_dry_run_counts_and_sizes_and_writes_nothing(drive, tmp_path):
    drive.quota = (2_000_000, 500_000)
    report = run(drive, tmp_path, dry_run=True)
    assert (report.media_files, report.media_bytes) == (4, 14_300)
    assert report.folders == 4                      # Footage, Gym, Squats, Empty
    assert (report.other_files, report.shortcuts) == (1, 1)
    assert drive.writes == 0 and not (tmp_path / "state.json").exists()
    text = report.summary()
    assert "4 photo/video" in text and "Nothing was copied" in text and "used of" in text


def test_copy_mirrors_the_tree_and_only_media_and_leaves_the_original_alone(drive, tmp_path):
    before = source_snapshot(drive, drive.top)
    report = run(drive, tmp_path)
    assert source_snapshot(drive, drive.top) == before           # the original is exactly as it was
    assert (report.copied, report.failed) == (4, [])
    tree = drive.tree()
    for path in ("Footage (working copy)/beach.mov", "Footage (working copy)/Gym/selfie.jpg",
                 "Footage (working copy)/Gym/Squats/rep1.mp4", "Footage (working copy)/Empty"):
        assert path in tree, path
    assert "Footage (working copy)/notes.pdf" not in tree
    assert not any("link to beach" in p and "working copy" in p for p in tree)
    # the two files with the same name both came across
    copies = [f for f in drive._files.values() if f.name == "beach.mov" and f.id.startswith("copy")]
    assert sorted(f.size for f in copies) == [5_000, 7_000]
    # each copy records where it came from
    assert all(f.app_properties.get("broll_copy_of") for f in drive._files.values() if f.id.startswith("copy"))
    assert "Next" not in report.summary() and report.dest_root_id


def test_a_second_run_copies_nothing_and_new_footage_is_picked_up(drive, tmp_path):
    run(drive, tmp_path)
    drive.writes = 0
    again = run(drive, tmp_path)
    assert (again.copied, again.already_copied) == (0, 4) and drive.writes == 0

    drive.add_file("new.mp4", drive.top.id, "video/mp4", 900)
    third = run(drive, tmp_path)
    assert (third.copied, third.already_copied) == (1, 4)
    assert [f.name for f in drive._files.values() if f.id.startswith("copy")].count("new.mp4") == 1


def copies(fake):
    return [f for f in fake._files.values() if f.id.startswith("copy")]


def test_a_killed_run_resumes_without_copying_anything_twice(drive, tmp_path):
    real = drive.copy_file
    seen = {"n": 0}

    def killed(*a, **k):
        seen["n"] += 1
        if seen["n"] == 3:
            raise KeyboardInterrupt  # killed after two copies, before any progress was written down
        return real(*a, **k)

    drive.copy_file = killed
    with pytest.raises(KeyboardInterrupt):
        run(drive, tmp_path)
    drive.copy_file = real
    assert len(copies(drive)) == 2
    saved = json.loads((tmp_path / "state.json").read_text())[drive.top.id]["files"]
    assert len(saved) == 0                         # the two copies were never written down

    report = run(drive, tmp_path)
    assert (report.copied, report.already_copied) == (2, 2)
    assert len(copies(drive)) == 4                 # exactly four: nothing doubled


def test_a_lost_state_file_cannot_cause_duplicates(drive, tmp_path):
    run(drive, tmp_path)
    (tmp_path / "state.json").unlink()
    report = run(drive, tmp_path)
    assert report.copied == 0 and len(copies(drive)) == 4
    folders = [f for f in drive._files.values() if f.name == "Footage (working copy)"]
    assert len(folders) == 1                       # the copy folder was reused, not duplicated


def test_one_failing_file_does_not_stop_the_rest(drive, tmp_path):
    real = drive.copy_file

    def picky(file_id, name, parent_id, props=None):
        if name == "selfie.jpg":
            raise RuntimeError("The user does not have sufficient permissions")
        return real(file_id, name, parent_id, props)

    drive.copy_file = picky
    report = run(drive, tmp_path)
    assert report.copied == 3
    assert [n for n, _ in report.failed] == ["selfie.jpg"] and "permissions" in report.failed[0][1]
    drive.copy_file = real
    assert run(drive, tmp_path).copied == 1        # the failed one is retried next time


def test_a_full_drive_stops_cleanly_and_keeps_progress(drive, tmp_path):
    real = drive.copy_file
    seen = {"n": 0}

    def full(*a, **k):
        seen["n"] += 1
        if seen["n"] == 3:
            raise DriveStorageFullError("full")
        return real(*a, **k)

    drive.copy_file = full
    with pytest.raises(DriveStorageFullError):
        run(drive, tmp_path)
    saved = json.loads((tmp_path / "state.json").read_text())[drive.top.id]["files"]
    assert len(saved) == 2
    drive.copy_file = real
    assert run(drive, tmp_path).copied == 2


def test_refuses_a_destination_inside_the_source(drive, tmp_path):
    gym = next(f for f in drive._files.values() if f.name == "Gym")
    with pytest.raises(DriveError, match="inside the folder being copied"):
        copy_folder(drive, drive.top.id, gym.id, state_path=tmp_path / "state.json")
    with pytest.raises(DriveError, match="inside the folder being copied"):
        copy_folder(drive, drive.top.id, drive.top.id, state_path=tmp_path / "state.json")
    assert drive.writes == 0


def test_a_missing_or_unshared_folder_gets_a_plain_message(drive, tmp_path):
    with pytest.raises(DriveError, match="shared with the account"):
        copy_folder(drive, "nope", "root", state_path=tmp_path / "state.json")


def test_all_files_includes_documents_and_a_custom_name(drive, tmp_path):
    report = run(drive, tmp_path, media_only=False, dest_name="Adam archive")
    assert report.copied == 5
    assert "Adam archive/notes.pdf" in drive.tree()


def test_parsing_links_and_sizes():
    link = "https://drive.google.com/drive/folders/1buuMF9LU4tH0nbIuVnN3IL48lXPrnhKb?usp=drive_link"
    assert parse_folder_id(link) == "1buuMF9LU4tH0nbIuVnN3IL48lXPrnhKb"
    assert parse_folder_id("1buuMF9LU4tH0nbIuVnN3IL48lXPrnhKb") == "1buuMF9LU4tH0nbIuVnN3IL48lXPrnhKb"
    assert parse_folder_id("https://drive.google.com/drive/u/1/folders/abc123DEF456ghi") == "abc123DEF456ghi"
    with pytest.raises(ValueError):
        parse_folder_id("https://example.com/not-a-folder")
    assert human_size(0) == "0 B" and human_size(1536) == "1.5 KB" and human_size(5 * 1024**3) == "5.0 GB"
