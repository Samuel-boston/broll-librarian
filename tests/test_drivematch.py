"""Indexing files obtained outside Drive, tied to their Drive originals by md5 and size.

No Drive byte is ever read: the Drive side is a fake that only lists metadata.
"""

from __future__ import annotations

import hashlib
import shutil

import pytest
from typer.testing import CliRunner

from broll.cli import app
from broll.drive.organizer import Organizer
from broll.ingest import drivematch
from broll.ingest.pipeline import IngestPipeline
from broll.jobs.queue import enqueue_files
from tests.fakes import FakeDriveClient
from tests.test_images import _photo

runner = CliRunner()


def md5_of(path):
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324


@pytest.fixture()
def drive(tmp_path):
    """A fake Drive source folder whose listing carries the md5 and size of the real local fixtures."""
    originals = tmp_path / "originals"
    originals.mkdir()
    beach = _photo(originals / "IMG_0001.jpg")
    other = _photo(originals / "IMG_0002.jpg")
    other.write_bytes(other.read_bytes() + b"\0different")
    fake = FakeDriveClient()
    top = fake.add_folder("Footage", "root")
    sub = fake.add_folder("Day 1", top.id)
    entries = {}
    for name, path, parent in (("IMG_0001.jpg", beach, top.id), ("IMG_0002.jpg", other, sub.id)):
        e = fake.add_file(name, parent, "image/jpeg", path.stat().st_size)
        e.md5 = md5_of(path)
        entries[name] = e
    fake.add_file("clip.mp4", top.id, "video/mp4", 999).md5 = "f" * 32  # never downloaded by anyone
    fake.top, fake.entries, fake.originals = top, entries, originals
    return fake


@pytest.fixture()
def downloads(tmp_path, drive):
    """What Nathan gets from the website: the same bytes, browser-style names, plus a stranger."""
    folder = tmp_path / "inbox"
    folder.mkdir()
    shutil.copy(drive.originals / "IMG_0001.jpg", folder / "IMG_0001 (1).jpg")
    shutil.copy(drive.originals / "IMG_0002.jpg", folder / "IMG_0002.jpg")
    (folder / "stranger.jpg").write_bytes(b"not in drive")
    (folder / "notes.txt").write_text("x")
    return folder


def listing(workspace, drive):
    items, _ = drivematch.load_inventory(workspace, drive.top.id, drive)
    return items


def test_listing_is_metadata_only_and_cached(workspace, drive):
    first, listed = drivematch.load_inventory(workspace, drive.top.id, drive)
    assert listed and {i.name for i in first} == {"IMG_0001.jpg", "IMG_0002.jpg", "clip.mp4"}
    assert next(i for i in first if i.name == "IMG_0002.jpg").path == "Day 1"
    assert drive.writes == 0 and not any(c.startswith("download") for c in drive.calls)
    again, listed = drivematch.load_inventory(workspace, drive.top.id, None)  # no client: cache only
    assert not listed and [i.id for i in again] == [i.id for i in first]


def test_files_match_by_md5_and_size_and_strangers_are_reported(workspace, store, drive, downloads):
    cache = drivematch.Md5Cache(workspace.dir / "m.json")
    report = drivematch.match_local_files(
        drivematch.scan_files(downloads), listing(workspace, drive), store, cache)
    by_name = {d.filename: d for d in report.matched}
    assert set(by_name) == {"IMG_0001.jpg", "IMG_0002.jpg"}  # the Drive name, not the browser's
    one = by_name["IMG_0001.jpg"]
    assert one.drive_file_id == drive.entries["IMG_0001.jpg"].id
    assert one.link == f"https://drive.google.com/file/d/{one.drive_file_id}/view"
    assert report.unmatched == ["stranger.jpg"]
    assert report.not_media == 1
    assert report.renamed == ["IMG_0001 (1).jpg is IMG_0001.jpg in Drive"]


def test_same_name_and_size_but_different_bytes_is_not_a_match(workspace, store, drive, tmp_path):
    folder = tmp_path / "fake"
    folder.mkdir()
    original = drive.originals / "IMG_0001.jpg"
    data = bytearray(original.read_bytes())
    data[-1] ^= 0xFF  # same size, same name, one byte different
    (folder / "IMG_0001.jpg").write_bytes(bytes(data))
    report = drivematch.match_local_files(
        drivematch.scan_files(folder), listing(workspace, drive), store,
        drivematch.Md5Cache(workspace.dir / "m.json"))
    assert not report.matched and report.unmatched == ["IMG_0001.jpg"]


def test_a_matched_file_is_indexed_as_the_drive_original_and_filing_never_uploads(
        workspace, store, drive, downloads):
    report = drivematch.match_local_files(
        drivematch.scan_files(downloads), listing(workspace, drive), store,
        drivematch.Md5Cache(workspace.dir / "m.json"))
    import asyncio

    async def run():
        pipeline = IngestPipeline(workspace, store)
        return [await pipeline.ingest(d) for d in report.matched]

    results = asyncio.run(run())
    assert all(r.status == "indexed" for r in results)
    for d, r in zip(report.matched, results):
        source = store.get_source(r.source_id)
        assert source.drive_file_id == d.drive_file_id
        assert source.drive_web_link == f"https://drive.google.com/file/d/{d.drive_file_id}/view"
        assert source.original_filename == d.filename
        assert d.path.exists(), "the downloaded copy is never deleted"

    # Run it again: everything is already indexed, nothing is queued twice.
    again = drivematch.match_local_files(
        drivematch.scan_files(downloads), listing(workspace, drive), store,
        drivematch.Md5Cache(workspace.dir / "m.json"))
    assert not again.matched and len(again.already_indexed) == 2

    # Filing treats them as the Drive originals: rename/move, never an upload or a copy.
    organised = Organizer(workspace, store, drive).reorganise()
    assert not [a for a in organised.actions if a.kind == "upload"]
    assert not [c for c in drive.calls if c.startswith(("upload", "copy"))]
    assert organised.errors == []
    for d in report.matched:
        assert drive.get(d.drive_file_id) is not None  # same Drive file, not replaced


def test_filing_refuses_matched_originals_only_when_organise_only_copies_is_on(
        workspace, store, drive, downloads):
    import asyncio

    report = drivematch.match_local_files(
        drivematch.scan_files(downloads), listing(workspace, drive), store,
        drivematch.Md5Cache(workspace.dir / "m.json"))
    asyncio.run(IngestPipeline(workspace, store).ingest(report.matched[0]))
    workspace.ingest.organise_only_copies = True
    guarded = Organizer(workspace, store, drive).reorganise()
    assert any("no copy stamp" in e for e in guarded.errors)
    assert not [a for a in guarded.actions if a.kind in ("rename", "move")]


def test_cli_match_drive_dry_run_then_real(broll_home, drive, downloads, monkeypatch):
    assert runner.invoke(app, ["init", "--name", "Fixtures", "--provider", "mock"]).exit_code == 0
    monkeypatch.setattr("broll.cli._drive_client", lambda config: drive)
    base = ["index", str(downloads), "--match-drive", "--match-folder", drive.top.id,
            "-w", "fixtures"]
    dry = runner.invoke(app, base + ["--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "2 matched to Drive" in dry.output and "NOT IN DRIVE" in dry.output and "stranger.jpg" in dry.output
    assert "Nothing was queued" in dry.output

    queued = runner.invoke(app, base + ["--no-wait"])
    assert queued.exit_code == 0, queued.output
    assert "Queued 2 file(s)" in queued.output
    # The second run reads the saved listing: it needs no Drive client at all.
    monkeypatch.setattr("broll.cli._drive_client", lambda config: pytest.fail("listed Drive twice"))
    again = runner.invoke(app, base + ["--dry-run"])
    assert again.exit_code == 0 and "from the saved listing" in again.output


def test_queued_jobs_carry_the_drive_id(workspace, store, drive, downloads):
    report = drivematch.match_local_files(
        drivematch.scan_files(downloads), listing(workspace, drive), store,
        drivematch.Md5Cache(workspace.dir / "m.json"))
    jobs = enqueue_files(store, report.matched)
    assert {j.payload["drive_file_id"] for j in jobs} == {
        drive.entries["IMG_0001.jpg"].id, drive.entries["IMG_0002.jpg"].id}
    assert all(j.payload["origin"] == "local" for j in jobs)


def test_the_upload_screen_ties_a_drive_original_to_its_file_and_leaves_new_footage_alone(
        workspace, store, drive, downloads):
    from broll.jobs.queue import queue_stats
    from tests.test_web import _client

    workspace.drive_source_folder_id = drive.top.id
    workspace.save()
    drivematch.load_inventory(workspace, drive.top.id, drive)  # the saved listing the upload reads
    store.close()
    stranger = downloads / "stranger.jpg"
    original = downloads / "IMG_0002.jpg"
    with _client(workspace, run_worker=False) as client:
        response = client.post("/ingest/upload", files=[
            ("files", (original.name, original.read_bytes(), "image/jpeg")),
            ("files", (stranger.name, stranger.read_bytes(), "image/jpeg")),
        ])
    assert response.status_code == 200 and "1 matched to their Drive original" in response.text
    from broll.db.store import Store

    s = Store.for_config(workspace)
    try:
        rows = s.conn.execute("SELECT json_extract(payload_json,'$.filename'), "
                              "json_extract(payload_json,'$.drive_file_id') FROM jobs").fetchall()
        assert dict((r[0], r[1]) for r in rows) == {
            "IMG_0002.jpg": drive.entries["IMG_0002.jpg"].id, "stranger.jpg": None}
        assert queue_stats(s).total == 2
    finally:
        s.close()


# -- copies made inside Drive, attributed to the originals ---------------------------------------


@pytest.fixture()
def copies(drive):
    """Nathan's Make-a-copy folder: same md5 and size as the originals, 'Copy of' names, new ids."""
    folder = drive.add_folder("My copies", "root")
    made = {}
    for name, entry in drive.entries.items():
        c = drive.add_file(f"Copy of {name}", folder.id, "image/jpeg", entry.size)
        c.md5 = entry.md5
        made[name] = c
    stray = drive.add_file("Copy of stray.jpg", folder.id, "image/jpeg", 77)
    stray.md5 = "a" * 32
    drive.copy_folder, drive.copy_entries = folder, made
    return drive


def test_copies_are_matched_to_originals_and_identified_as_them(workspace, store, copies):
    originals, _ = drivematch.load_inventory(workspace, copies.top.id, copies)
    folder, _ = drivematch.load_inventory(workspace, copies.copy_folder.id, copies)
    report = drivematch.match_copies(folder, originals, store)
    assert len(report.matched) == 2 and report.unmatched == ["Copy of stray.jpg"]
    for d in report.matched:
        original = copies.entries[d.filename]
        assert d.drive_file_id == original.id
        assert d.byte_source_id == copies.copy_entries[d.filename].id
        assert d.link == f"https://drive.google.com/file/d/{original.id}/view"
        assert d.origin == "drive" and d.origin_path == f"drive:{original.id}"
        assert d.payload()["byte_source_id"] == d.byte_source_id


def test_a_copy_is_read_for_bytes_but_the_source_is_the_original(workspace, store, copies, monkeypatch):
    import asyncio

    originals, _ = drivematch.load_inventory(workspace, copies.top.id, copies)
    folder, _ = drivematch.load_inventory(workspace, copies.copy_folder.id, copies)
    report = drivematch.match_copies(folder, originals, store)
    asked = []

    def fake_fetch(config, file_id, filename, client=None, expected_size=None):
        asked.append(file_id)
        return copies.originals / filename

    monkeypatch.setattr("broll.drive.fetcher.fetch_drive_file", fake_fetch)
    pipeline = IngestPipeline(workspace, store)
    results = asyncio.run(_run_all(pipeline, report.matched))
    assert sorted(asked) == sorted(c.id for c in copies.copy_entries.values())
    for d, r in zip(report.matched, results):
        source = store.get_source(r.source_id)
        assert (source.drive_file_id, source.source_alt_id) == (d.drive_file_id, d.byte_source_id)
        assert source.drive_web_link.endswith(f"{d.drive_file_id}/view")
    assert drivematch.match_copies(folder, originals, store).already_indexed  # not queued twice

    # Filing moves the ORIGINAL and never touches the copies.
    before = {c.id: (c.name, tuple(c.parents)) for c in copies.copy_entries.values()}
    organised = Organizer(workspace, store, copies).reorganise()
    assert organised.errors == [] and not [c for c in copies.calls if c.startswith(("upload", "copy"))]
    assert before == {c.id: (c.name, tuple(c.parents)) for c in copies.copy_entries.values()}


async def _run_all(pipeline, files):
    return [await pipeline.ingest(d) for d in files]


def test_later_reads_try_the_original_then_the_copy(workspace, store):
    from broll import fetch
    from broll.db.models import Source
    from broll.ingest.hashing import content_hash

    data = tmp = workspace.dir / "real.jpg"
    _photo(tmp)
    source = Source(id="s1", workspace_id=workspace.id, content_hash=content_hash(data),
                    original_filename="real.jpg", origin="drive", drive_file_id="orig",
                    source_alt_id="copy")
    assert fetch.drive_ids(source) == ["orig", "copy"]

    class Client:
        tried = []

        def download(self, file_id, destination):
            self.tried.append(file_id)
            if file_id == "orig":
                raise RuntimeError("Too many users have viewed or downloaded this file recently")
            destination.write_bytes(data.read_bytes())

    client = Client()
    path = fetch.download(workspace, source, workspace.dir / "dl", client)
    assert client.tried == ["orig", "copy"] and path.read_bytes() == data.read_bytes()
    client.tried.clear()
    source2 = source.model_copy(update={"source_alt_id": None, "id": "s2"})
    with pytest.raises(RuntimeError):
        fetch.download(workspace, source2, workspace.dir / "dl", client)


def test_cli_copies_folder_dry_run_lists_both_folders_and_queues_nothing(broll_home, copies, monkeypatch):
    assert runner.invoke(app, ["init", "--name", "Fixtures", "--provider", "mock"]).exit_code == 0
    monkeypatch.setattr("broll.cli._drive_client", lambda config: copies)
    result = runner.invoke(app, ["index", "--copies-folder", copies.copy_folder.id,
                                 "--match-folder", copies.top.id, "--dry-run", "-w", "fixtures"])
    assert result.exit_code == 0, result.output
    assert "2 copies matched to their originals" in result.output
    assert "NO ORIGINAL" in result.output and "Nothing was queued" in result.output
    assert copies.writes == 0
