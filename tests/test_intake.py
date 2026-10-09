"""What gets turned away, what gets flagged, and downloads that are never mistaken for the clip."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from broll import attention
from broll.config import IngestConfig
from broll.db.models import Source
from broll.drive.fetcher import GB, ensure_room, fetch_drive_file, sweep_partial_downloads
from broll.ingest.errors import DiskSpaceError, IncompleteDownloadError
from broll.ingest.limits import check_limits, human_duration
from broll.ingest.pipeline import IngestPipeline
from broll.ingest.raw import extract_preview, is_raw
from broll.ingest.scanner import DiscoveredFile, media_kind
from broll.jobs.queue import enqueue_files, queue_stats
from broll.jobs.worker import Worker
from tests.fakes import FakeDriveClient

CFG = IngestConfig()


# -- the limits ------------------------------------------------------------------------


def test_a_two_hour_recording_is_turned_away_before_anything_is_downloaded():
    verdict = check_limits(CFG, size_bytes=40 * GB, duration_s=7200)
    assert verdict.kind == "too_long" and "2 h" in verdict.detail


def test_a_big_file_is_turned_away_by_size_when_its_length_is_unknown():
    verdict = check_limits(CFG, size_bytes=20 * GB, duration_s=None)
    assert verdict.kind == "too_big" and "20.0 GB" in verdict.detail


def test_an_ordinary_clip_passes():
    assert check_limits(CFG, size_bytes=468_000_000, duration_s=45) is None


def test_a_photograph_is_never_too_long():
    assert check_limits(CFG, size_bytes=20_000_000, duration_s=9999, is_image=True) is None


def test_forcing_a_file_lifts_the_length_limit_but_not_the_laws_of_disk_space():
    assert check_limits(CFG, size_bytes=3 * GB, duration_s=1800, forced=True, free_bytes=30 * GB) is None
    too_big = check_limits(CFG, size_bytes=58 * GB, duration_s=1800, forced=True, free_bytes=30 * GB)
    assert too_big.kind == "too_big" and "more than this server can hold" in too_big.detail
    # even forced, a ten-hour file is past the ceiling
    assert check_limits(CFG, size_bytes=GB, duration_s=11 * 3600, forced=True, free_bytes=30 * GB).kind == "too_long"


def test_a_limit_of_zero_means_no_limit():
    cfg = IngestConfig(max_duration_s=0, max_file_gb=0)
    assert check_limits(cfg, size_bytes=90 * GB, duration_s=90_000) is None


def test_durations_read_naturally():
    assert human_duration(45) == "45 s" and human_duration(125) == "2 min 05 s"
    assert human_duration(3900) == "1 h 05 min"


# -- downloads -------------------------------------------------------------------------


class Client(FakeDriveClient):
    """A Drive whose download we control."""

    def __init__(self, payload: bytes = b"x" * 1000, fail_after: int | None = None):
        super().__init__()
        self.payload, self.fail_after, self.downloads = payload, fail_after, 0

    def download(self, file_id, destination):
        self.downloads += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = self.payload if self.fail_after is None else self.payload[: self.fail_after]
        destination.write_bytes(data)
        return destination


def test_a_complete_download_lands_under_its_real_name(workspace):
    got = fetch_drive_file(workspace, "abc", "clip.mov", Client(), expected_size=1000)
    assert got.name == "drive-abc-clip.mov" and got.stat().st_size == 1000
    assert not list(workspace.temp_dir.glob("*.part"))


def test_a_download_that_stops_short_is_deleted_never_kept_as_the_clip(workspace):
    client = Client(fail_after=400)
    with pytest.raises(IncompleteDownloadError, match="400 of 1000"):
        fetch_drive_file(workspace, "abc", "clip.mov", client, expected_size=1000)
    assert not list(workspace.temp_dir.glob("drive-abc*"))


def test_a_half_file_left_by_a_killed_run_is_not_reused(workspace):
    """The old behaviour: a file on disk was assumed complete. It was the bug."""
    workspace.ensure_dirs()
    (workspace.temp_dir / "drive-abc-clip.mov").write_bytes(b"x" * 300)  # cut short
    client = Client()
    got = fetch_drive_file(workspace, "abc", "clip.mov", client, expected_size=1000)
    assert got.stat().st_size == 1000 and client.downloads == 1


def test_a_complete_file_already_on_disk_is_reused(workspace):
    workspace.ensure_dirs()
    (workspace.temp_dir / "drive-abc-clip.mov").write_bytes(b"x" * 1000)
    client = Client()
    fetch_drive_file(workspace, "abc", "clip.mov", client, expected_size=1000)
    assert client.downloads == 0


def test_the_expected_size_is_asked_of_drive_when_not_given(workspace):
    client = Client(fail_after=10)
    entry = client.add_file("clip.mov", "root", size=1000)
    with pytest.raises(IncompleteDownloadError, match="10 of 1000"):
        fetch_drive_file(workspace, entry.id, "clip.mov", client)  # no expected_size given


def test_stale_partial_files_are_swept_but_a_download_in_progress_is_left(workspace):
    workspace.ensure_dirs()
    old, fresh = workspace.temp_dir / "old.part", workspace.temp_dir / "fresh.part"
    old.write_bytes(b"x"); fresh.write_bytes(b"x")
    long_ago = time.time() - 3 * 3600
    os.utime(old, (long_ago, long_ago))
    assert sweep_partial_downloads(workspace) == 1
    assert not old.exists() and fresh.exists()


def test_a_download_that_would_fill_the_disk_waits_instead(workspace):
    with pytest.raises(DiskSpaceError, match="Not enough free disk"):
        ensure_room(workspace.temp_dir, 10**18, headroom_gb=6)
    ensure_room(workspace.temp_dir, 1000, headroom_gb=0)  # fits: no error


# -- the pipeline turns files away, and lists them ----------------------------------------


def _pipeline(workspace, store):
    return IngestPipeline(workspace, store)


async def test_a_long_drive_file_is_listed_with_a_link_and_never_fetched(workspace, store):
    discovered = DiscoveredFile(
        origin="drive", path=None, filename="podcast_ep12.mp4", drive_file_id="d1",
        origin_path="drive:d1", size_bytes=30 * GB, duration_s=5400, link="https://drive/d1",
    )
    result = await _pipeline(workspace, store).ingest(discovered)

    assert result.status == "skipped" and result.skipped_kind == "too_long"
    [item] = store.list_attention()
    assert item["filename"] == "podcast_ep12.mp4" and item["link"] == "https://drive/d1"
    assert item["kind"] == "too_long" and item["duration_s"] == 5400
    assert store.list_sources() == [], "turned away, not indexed"
    assert not list(workspace.temp_dir.glob("*")), "nothing was downloaded"


async def test_the_worker_counts_a_skip_as_done_not_failed(workspace, store):
    discovered = DiscoveredFile(origin="local", path=None, filename="big.mp4",
                                origin_path="/footage/big.mp4", size_bytes=50 * GB, duration_s=100)
    enqueue_files(store, [discovered])
    worker = Worker(workspace, store, _pipeline(workspace, store))
    stats = await worker.run()
    assert (stats.skipped, stats.failed) == (1, 0)
    assert queue_stats(store).failed == 0 and queue_stats(store).done == 1


async def test_a_local_file_too_long_for_the_limit_is_kept_where_it_is(workspace, store, tmp_path):
    clip = tmp_path / "long_take.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=160x120:rate=5:duration=30", "-y", str(clip)], check=True)
    workspace.ingest.max_duration_s = 10
    discovered = DiscoveredFile(origin="local", path=clip, filename=clip.name, origin_path=str(clip))
    result = await _pipeline(workspace, store).ingest(discovered)
    assert result.skipped_kind == "too_long" and clip.exists()
    assert store.list_attention()[0]["detail"].startswith("30 s long; the limit is 10 s")


async def test_index_anyway_lifts_the_length_limit_for_that_file(workspace, store, tmp_path):
    clip = tmp_path / "long_take.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=160x120:rate=5:duration=30", "-y", str(clip)], check=True)
    workspace.ingest.max_duration_s = 10
    pipeline = _pipeline(workspace, store)
    discovered = DiscoveredFile(origin="local", path=clip, filename=clip.name, origin_path=str(clip))
    await pipeline.ingest(discovered)
    [item] = store.list_attention()

    assert attention.requeue(workspace, store, item["id"], forced=True)
    assert store.get_attention(item["id"])["status"] == "requeued"
    stats = await Worker(workspace, store, pipeline).run()
    assert stats.done == 1 and stats.skipped == 0 and stats.shots >= 1
    assert store.list_attention() == [], "once indexed it is off the list"
    assert store.get_attention(item["id"])["status"] == "resolved"


async def test_a_dismissed_file_is_not_queued_again(workspace, store):
    discovered = DiscoveredFile(origin="local", path=None, filename="x.mp4",
                                origin_path="/footage/x.mp4", size_bytes=50 * GB)
    await _pipeline(workspace, store).ingest(discovered)
    [item] = store.list_attention()
    attention.dismiss(store, item["id"])
    assert enqueue_files(store, [discovered]) == []
    assert len(enqueue_files(store, [discovered], force=True)) == 1


async def test_a_file_that_keeps_failing_to_download_ends_up_on_the_list(workspace, store, monkeypatch):
    from broll.ingest import pipeline as pipeline_module

    async def broken(self, discovered):
        raise IncompleteDownloadError("clip.mov: the download stopped at 5 of 1000 bytes")

    monkeypatch.setattr(pipeline_module.IngestPipeline, "_fetch", broken)
    discovered = DiscoveredFile(origin="drive", path=None, filename="clip.mov", drive_file_id="d4",
                                origin_path="drive:d4", link="https://drive/d4")
    enqueue_files(store, [discovered])
    worker = Worker(workspace, store, _pipeline(workspace, store))
    for _ in range(3):
        store.conn.execute("UPDATE jobs SET not_before = NULL WHERE workspace_id = ?", (store.workspace_id,))
        await worker.run()
    [item] = store.list_attention()
    assert item["kind"] == "download_incomplete" and item["link"] == "https://drive/d4"


def test_failures_are_sorted_into_the_right_list():
    assert attention.classify_failure("x: the download stopped at 5 of 9 bytes") == "download_incomplete"
    assert attention.classify_failure("ffprobe could not read a.mp4: Invalid data") == "unreadable"
    assert attention.classify_failure("gemini returned no structured output") == "analysis_failed"


def test_flagging_twice_keeps_one_row_and_dismissal_sticks(store):
    f = DiscoveredFile(origin="drive", path=None, filename="a.mp4", drive_file_id="z", origin_path="drive:z")
    first = attention.flag_file(store, f, "too_long", "one")
    attention.dismiss(store, first)
    second = attention.flag_file(store, f, "too_long", "two")
    assert first == second and store.get_attention(first)["status"] == "dismissed"
    assert store.get_attention(first)["detail"] == "two"


def test_the_drive_mirror_adds_one_shortcut_per_file_in_a_folder_per_reason(workspace, store):
    client = FakeDriveClient()
    for name, kind in (("a.mp4", "too_long"), ("b.mp4", "too_long"), ("c.mov", "download_incomplete")):
        entry = client.add_file(name, "root")
        attention.flag_file(
            store, DiscoveredFile(origin="drive", path=None, filename=name, drive_file_id=entry.id,
                                  origin_path=f"drive:{entry.id}"), kind, "why",
        )
    assert attention.sync_drive_shortcuts(workspace, store, client) == 3
    tree = client.tree()
    assert "ADAM/_Needs Attention/Too long to be B-roll/a.mp4" not in tree  # root is the workspace's own
    assert any(p.endswith("_Needs Attention/Too long to be B-roll/a.mp4") for p in tree)
    assert any(p.endswith("_Needs Attention/Download didn't finish/c.mov") for p in tree)
    assert attention.sync_drive_shortcuts(workspace, store, client) == 0, "the second pass changes nothing"


# -- RAW photographs --------------------------------------------------------------------------


def test_raw_files_are_photographs():
    assert media_kind("DSC_5387.NEF") == "image" and media_kind("a.CR3") == "image"
    assert is_raw("x.nef") and not is_raw("x.jpg")


def _fake_rawpy(monkeypatch, *, thumb, flip=0, boom=False):
    """A stand-in for libraw: the real one needs a real camera file."""
    from PIL import Image

    module = types.ModuleType("rawpy")
    class NoThumb(Exception): ...
    class Unsupported(Exception): ...
    module.LibRawNoThumbnailError, module.LibRawUnsupportedThumbnailError = NoThumb, Unsupported
    module.ThumbFormat = types.SimpleNamespace(JPEG="jpeg", BITMAP="bitmap")

    class Raw:
        sizes = types.SimpleNamespace(flip=flip)
        def __enter__(self):
            if boom:
                raise OSError("corrupt")
            return self
        def __exit__(self, *a): return False
        def extract_thumb(self):
            if thumb is None:
                raise NoThumb()
            return thumb
        def postprocess(self, **k):
            import numpy as np
            return np.full((30, 40, 3), 120, dtype="uint8")

    module.imread = lambda path: Raw()
    monkeypatch.setitem(sys.modules, "rawpy", module)
    return module


def _jpeg_bytes(width=60, height=40):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (200, 80, 40)).save(buf, "JPEG")
    return buf.getvalue()


def test_a_raw_photo_is_read_through_the_jpeg_inside_it(monkeypatch, tmp_path):
    from PIL import Image
    _fake_rawpy(monkeypatch, thumb=types.SimpleNamespace(format="jpeg", data=_jpeg_bytes()))
    raw = tmp_path / "DSC_1.NEF"
    raw.write_bytes(b"raw")
    preview = extract_preview(raw, tmp_path / "out")
    assert preview and preview.suffix == ".jpg"
    with Image.open(preview) as img:
        assert img.size == (60, 40)
    assert raw.read_bytes() == b"raw", "the RAW itself is never touched"


def test_a_portrait_raw_is_turned_upright(monkeypatch, tmp_path):
    from PIL import Image
    _fake_rawpy(monkeypatch, thumb=types.SimpleNamespace(format="jpeg", data=_jpeg_bytes(60, 40)), flip=6)
    raw = tmp_path / "DSC_2.NEF"; raw.write_bytes(b"raw")
    with Image.open(extract_preview(raw, tmp_path / "out")) as img:
        assert img.size == (40, 60)


def test_a_raw_with_no_preview_is_developed_small(monkeypatch, tmp_path):
    from PIL import Image
    _fake_rawpy(monkeypatch, thumb=None)
    raw = tmp_path / "DSC_3.NEF"; raw.write_bytes(b"raw")
    with Image.open(extract_preview(raw, tmp_path / "out")) as img:
        assert img.size == (40, 30)


def test_a_damaged_raw_gives_none_not_a_crash(monkeypatch, tmp_path):
    _fake_rawpy(monkeypatch, thumb=None, boom=True)
    raw = tmp_path / "DSC_4.NEF"; raw.write_bytes(b"raw")
    assert extract_preview(raw, tmp_path / "out") is None


async def test_a_raw_photo_is_indexed_from_its_preview_and_the_raw_is_left_alone(
    workspace, store, monkeypatch, tmp_path
):
    _fake_rawpy(monkeypatch, thumb=types.SimpleNamespace(format="jpeg", data=_jpeg_bytes(640, 480)))
    raw = tmp_path / "DSC_9.NEF"
    raw.write_bytes(b"not really a raw file" * 100)
    discovered = DiscoveredFile(origin="local", path=raw, filename=raw.name, origin_path=str(raw))
    result = await IngestPipeline(workspace, store).ingest(discovered)

    assert result.status in ("indexed", "needs_review") and result.shots_analysed == 1
    [source] = store.list_sources()
    assert source.media_kind == "image" and source.filesize_bytes == raw.stat().st_size
    assert raw.exists() and raw.read_bytes().startswith(b"not really")
    assert not list(workspace.temp_dir.glob("raw-*")), "the preview is cleaned up"


async def test_a_raw_that_cannot_be_read_is_listed_not_failed(workspace, store, monkeypatch, tmp_path):
    _fake_rawpy(monkeypatch, thumb=None, boom=True)
    raw = tmp_path / "DSC_10.NEF"; raw.write_bytes(b"junk")
    discovered = DiscoveredFile(origin="local", path=raw, filename=raw.name, origin_path=str(raw))
    result = await IngestPipeline(workspace, store).ingest(discovered)
    assert result.skipped_kind == "unreadable"
    assert store.list_attention()[0]["kind"] == "unreadable" and raw.exists()


def test_two_files_with_the_same_name_in_one_folder_are_both_found(workspace):
    """Camera footage is full of them; a listing keyed by name silently lost one of each pair."""
    from broll.ingest.scanner import scan_drive_folder

    client = FakeDriveClient()
    folder = client.add_folder("Day 1", "root")
    client.add_file("IMG_0001.MOV", folder.id)
    client.add_file("IMG_0001.MOV", folder.id)
    client.add_file("IMG_0002.MOV", folder.id)
    found = scan_drive_folder(client, folder.id)
    assert [f.filename for f in found] == ["IMG_0001.MOV", "IMG_0001.MOV", "IMG_0002.MOV"]
    assert len({f.drive_file_id for f in found}) == 3


# -- big Drive videos are read in place, not downloaded --------------------------------------------


class RangeServer:
    """A local web server that serves one file and honours Range requests, the way Drive does."""

    def __init__(self, path):
        import http.server
        import re
        import threading

        data = path.read_bytes()
        self.data, self.requests = data, []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.requests.append((self.headers.get("Range"), self.headers.get("Authorization")))
                if self.headers.get("Authorization") != "Bearer tok":
                    self.send_response(401); self.end_headers(); return
                rng = self.headers.get("Range")
                if rng:
                    a, b = re.match(r"bytes=(\d+)-(\d*)", rng).groups()
                    a, b = int(a), int(b) if b else len(data) - 1
                    chunk = data[a:b + 1]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {a}-{a + len(chunk) - 1}/{len(data)}")
                else:
                    chunk = data
                    self.send_response(200)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(len(chunk)))
                self.send_header("Content-Type", "video/mp4")
                self.end_headers()
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/clip.mp4"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


class TokenClient:
    def access_token(self):
        return "tok"


@pytest.fixture()
def served_clip(tmp_path):
    path = tmp_path / "remote.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=320x240:rate=15:duration=24", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "-y", str(path)], check=True)
    server = RangeServer(path)
    yield path, server
    server.close()


def test_a_remote_video_has_the_same_fingerprint_as_the_downloaded_file(served_clip):
    from broll.ingest.hashing import content_hash
    from broll.ingest.remote import RemoteDriveVideo

    path, server = served_clip
    remote = RemoteDriveVideo(TokenClient(), "id", "remote.mp4", size=path.stat().st_size, url=server.url)
    assert remote.content_hash() == content_hash(path), "streamed or fetched, it is the same file"


def test_ffprobe_and_ffmpeg_read_a_video_over_https_with_the_token(served_clip, tmp_path):
    from broll.ingest.frames import extract_frames_timed
    from broll.ingest.probe import probe
    from broll.ingest.remote import RemoteDriveVideo

    path, server = served_clip
    remote = RemoteDriveVideo(TokenClient(), "id", "remote.mp4", size=path.stat().st_size, url=server.url)
    meta = probe(remote)
    assert meta.duration_s == pytest.approx(24, abs=0.5) and (meta.width, meta.height) == (320, 240)
    frames = extract_frames_timed(remote, tmp_path / "f", start_s=0, duration_s=24, count=4, centred=True)
    assert len(frames) == 4 and all(f.path.stat().st_size > 0 for f in frames)
    assert all(auth == "Bearer tok" for _, auth in server.requests), "every request carried the token"
    fetched = sum(1 for rng, _ in server.requests if rng)
    assert fetched >= 4, "it asked for pieces of the file, not the whole thing"


async def test_a_big_drive_video_is_indexed_without_being_downloaded(workspace, store, served_clip, monkeypatch):
    path, server = served_clip
    workspace.ingest.stream_above_gb = 0.000001   # treat this small file as "big"
    monkeypatch.setattr("broll.ingest.remote.API", server.url + "?{id}")
    pipeline = IngestPipeline(workspace, store, drive_client=lambda: TokenClient())
    from broll.ingest import pipeline as pipeline_module
    original = pipeline_module.RemoteDriveVideo

    class Served(original):
        def __init__(self, client, file_id, name, size=None, url=None):
            super().__init__(client, file_id, name, size, url=server.url)

    monkeypatch.setattr(pipeline_module, "RemoteDriveVideo", Served)
    discovered = DiscoveredFile(origin="drive", path=None, filename="remote.mp4", drive_file_id="abc",
                                origin_path="drive:abc", size_bytes=path.stat().st_size, link="https://drive/abc")
    result = await pipeline.ingest(discovered)

    assert result.status in ("indexed", "needs_review") and result.shots_analysed >= 1
    assert "Read in place from Drive" in " ".join(result.messages)
    [source] = store.list_sources()
    assert source.drive_file_id == "abc" and source.duration_s == pytest.approx(24, abs=0.5)
    assert not list(workspace.temp_dir.glob("drive-*")), "nothing was downloaded"
    assert server.requests and all(rng for rng, _ in server.requests), "only ever asked for ranges"


async def test_a_streamed_video_over_the_length_limit_is_listed_after_one_look(workspace, store, served_clip, monkeypatch):
    path, server = served_clip
    workspace.ingest.stream_above_gb = 0.000001
    workspace.ingest.max_duration_s = 10
    from broll.ingest import pipeline as pipeline_module
    original = pipeline_module.RemoteDriveVideo

    class Served(original):
        def __init__(self, client, file_id, name, size=None, url=None):
            super().__init__(client, file_id, name, size, url=server.url)

    monkeypatch.setattr(pipeline_module, "RemoteDriveVideo", Served)
    pipeline = IngestPipeline(workspace, store, drive_client=lambda: TokenClient())
    discovered = DiscoveredFile(origin="drive", path=None, filename="remote.mp4", drive_file_id="abc",
                                origin_path="drive:abc", size_bytes=path.stat().st_size)
    result = await pipeline.ingest(discovered)
    assert result.skipped_kind == "too_long" and store.list_sources() == []


def test_a_drive_video_is_streamed_only_when_it_is_big_and_a_video(workspace, store):
    pipeline = IngestPipeline(workspace, store)
    def d(name, size, origin="drive"):
        return DiscoveredFile(origin=origin, path=None, filename=name, drive_file_id="x", origin_path="drive:x", size_bytes=size)
    assert pipeline._streams(d("a.mov", 3 * GB))
    assert not pipeline._streams(d("a.mov", GB)), "small enough to download"
    assert not pipeline._streams(d("a.jpg", 3 * GB)), "photos are downloaded"
    assert not pipeline._streams(d("a.mov", 3 * GB, origin="local")), "only Drive files can be read in place"


def test_a_recording_is_read_in_fewer_bigger_windows():
    from broll.analysis.segmentation import plan_windows

    assert len(plan_windows(0, 600)) == 10                 # ten minutes: a minute at a time
    long = plan_windows(0, 3 * 3600)                       # three hours
    assert len(long) <= 20 and all(w.end_s - w.start_s <= 601 for w in long) and long[-1].end_s == 3 * 3600
