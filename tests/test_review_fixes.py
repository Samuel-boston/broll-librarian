"""Regression tests for problems found by an independent review of the precision/intake change.

Each test reproduces a failure the reviewer described.
"""

from __future__ import annotations

import asyncio
import json
import random
import subprocess
import threading
import time
from pathlib import Path

import pytest

from broll import attention
from broll.analysis.analyzer import Analyzer
from broll.analysis.providers.base import TransientProviderError
from broll.analysis.providers.mock import MockVisionProvider
from broll.analysis.segmentation import (
    ModelSegment, Segment, SegmentKind, combine_windows, normalise_segments,
)
from broll.db.models import ShotFacets
from broll.drive.fetcher import GB, reserve_room
from broll.drive.organizer import Organizer
from broll.drive.taxonomy import plan_tree
from broll.ingest.errors import DiskSpaceError
from broll.ingest.pipeline import IngestPipeline
from broll.ingest.scanner import DiscoveredFile
from broll.jobs.queue import enqueue_files
from broll.review import apply_correction, approve_folder_proposal
from tests.fakes import FakeDriveClient
from tests.test_client_tree import tree_config
from tests.test_intake import RangeServer, TokenClient
from tests.test_precision import add_shot


def seg(a, b, kind="usable", **kw):
    return ModelSegment(start_s=a, end_s=b, kind=SegmentKind(kind), **kw)


@pytest.fixture()
def clip(tmp_path):
    def make(name="remote.mp4", seconds=24):
        path = tmp_path / name
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
                        f"testsrc2=size=320x240:rate=15:duration={seconds}", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", "-y", str(path)], check=True)
        return path
    return make


# 1. planning must not freeze the web app ------------------------------------------------------------


async def test_looking_through_a_clip_does_not_freeze_the_event_loop(workspace, clip, tmp_path, monkeypatch):
    from broll.ingest import segments as segments_module

    real = segments_module.extract_frames_timed

    def slow(*a, **k):
        time.sleep(0.4)
        return real(*a, **k)

    monkeypatch.setattr(segments_module, "extract_frames_timed", slow)
    path = clip("plain_loop.mp4", 24)
    gaps, stop = [], False

    async def heartbeat():
        last = time.monotonic()
        while not stop:
            await asyncio.sleep(0.02)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    task = asyncio.create_task(heartbeat())
    await segments_module.plan_segments(
        MockVisionProvider(), path, filename=path.name, start_s=0, end_s=24, file_duration_s=24,
        width=320, height=240, config=workspace, work_dir=tmp_path / "w")
    stop = True
    await task
    assert max(gaps) < 0.25, f"the loop stalled for {max(gaps):.2f}s while ffmpeg ran"


# 2. a streamed file is never "too big" ----------------------------------------------------------------


async def test_a_streamed_file_over_the_size_limit_is_still_read_in_place(workspace, store, clip, monkeypatch):
    path = clip()
    server = RangeServer(path)
    try:
        workspace.ingest.stream_above_gb = 0.000001
        workspace.ingest.max_file_gb = 0.000001      # smaller than the file: only a download would care
        from broll.ingest import pipeline as pipeline_module
        original = pipeline_module.RemoteDriveVideo

        class Served(original):
            def __init__(self, client, file_id, name, size=None, url=None):
                super().__init__(client, file_id, name, size, url=server.url)

        monkeypatch.setattr(pipeline_module, "RemoteDriveVideo", Served)
        result = await IngestPipeline(workspace, store, drive_client=lambda: TokenClient()).ingest(
            DiscoveredFile(origin="drive", path=None, filename="remote.mp4", drive_file_id="a",
                           origin_path="drive:a", size_bytes=path.stat().st_size))
        assert result.skipped_kind is None and result.shots_analysed >= 1
        assert store.list_attention() == []
    finally:
        server.close()


# 3. Drive hiccups are retried, not written into the clip's history -------------------------------------


async def test_drive_going_quiet_while_looking_through_a_clip_is_retried_not_recorded(
    workspace, store, clip, monkeypatch
):
    path = clip()
    server = RangeServer(path)
    try:
        workspace.ingest.stream_above_gb = 0.000001
        from broll.ingest import pipeline as pipeline_module
        from broll.ingest import segments as segments_module
        original = pipeline_module.RemoteDriveVideo

        class Served(original):
            def __init__(self, client, file_id, name, size=None, url=None):
                super().__init__(client, file_id, name, size, url=server.url)

        monkeypatch.setattr(pipeline_module, "RemoteDriveVideo", Served)
        real = segments_module.extract_frames_timed
        monkeypatch.setattr(segments_module, "extract_frames_timed", lambda *a, **k: [])  # Drive stops answering
        pipeline = IngestPipeline(workspace, store, drive_client=lambda: TokenClient())
        discovered = DiscoveredFile(origin="drive", path=None, filename="remote.mp4", drive_file_id="a",
                                    origin_path="drive:a", size_bytes=path.stat().st_size)
        with pytest.raises(TransientProviderError, match="could not read any frames"):
            await pipeline.ingest(discovered)
        assert store.count_shots() == 0, "nothing about this clip was recorded"

        monkeypatch.setattr(segments_module, "extract_frames_timed", real)   # Drive is back
        result = await pipeline.ingest(discovered)
        assert result.shots_analysed >= 1 and not result.deduped
        assert all("analysis_failed" not in s.review_reasons for s in store.list_shots())
    finally:
        server.close()


async def test_a_clip_the_model_could_not_describe_is_tried_again_not_deduped(workspace, store, clip):
    class Flaky(MockVisionProvider):
        calls = 0

        async def analyse(self, frames, context, retry_error=None):
            Flaky.calls += 1
            if Flaky.calls <= 2:
                from broll.analysis.providers.base import ProviderError
                raise ProviderError("no structured output")
            return await super().analyse(frames, context, retry_error)

    path = clip("quick.mp4", 5)
    d = DiscoveredFile(origin="local", path=path, filename="quick.mp4", origin_path=str(path))
    pipeline = IngestPipeline(workspace, store, analyzer=Analyzer(workspace, provider=Flaky()))
    await pipeline.ingest(d)
    [shot] = store.list_shots()
    assert shot.review_reasons == ["analysis_failed"] and not shot.caption
    result = await pipeline.ingest(d)
    assert not result.deduped and result.shots_analysed == 1
    assert store.list_shots()[0].caption and "analysis_failed" not in store.list_shots()[0].review_reasons


def test_a_clip_that_could_not_be_described_is_left_alone_in_drive(workspace, store):
    """Renaming the original to clip_<hash>.mov and moving it away would be all cost."""
    workspace.taxonomy = tree_config()
    source_id = _shot_in_drive(workspace, store, review_reasons=["analysis_failed"], status="needs_review")
    client = FakeDriveClient()
    report = Organizer(workspace, store, client).organise_source(source_id)
    assert report.skipped == 1 and client.writes == 0


def _shot_in_drive(workspace, store, drive_file_id=None, **fields):
    from broll.db.models import Source
    from broll.db.store import new_id

    source = store.insert_source(Source(id=new_id(), workspace_id=workspace.id, content_hash="h",
                                        original_filename="IMG_1.mov", origin="local", origin_path="/x/IMG_1.mov",
                                        drive_file_id=drive_file_id))
    from broll.db.models import Shot
    store.insert_shot(Shot(id=f"{source.id}-0", workspace_id=workspace.id, source_id=source.id, shot_index=0,
                           duration_s=5, end_s=5, **fields))
    store.recompute_source_status(source.id)
    return source.id


# 4. windows: seams and the "nothing usable" fallback -----------------------------------------------------


def test_one_dead_minute_in_a_long_file_is_not_a_usable_shot():
    first = normalise_segments([seg(0, 60, "usable", summary="a man walks on a beach")], 0, 60, unsure_if_none=False)
    second = normalise_segments([seg(60, 120, "dead")], 60, 120, unsure_if_none=False)
    out = combine_windows([first, second])
    assert [(s.kind, s.start_s, s.end_s) for s in out] == [("usable", 0, 60), ("dead", 60, 120)]
    assert not any(s.unsure for s in out)


def test_a_continuous_take_is_not_cut_in_two_at_every_window_seam():
    a = normalise_segments([seg(0, 60, summary="a man walks along a beach at sunset")], 0, 60, unsure_if_none=False)
    b = normalise_segments([seg(60, 120, summary="a man walking along the beach at sunset")], 60, 120, unsure_if_none=False)
    out = combine_windows([a, b])
    assert [(s.start_s, s.end_s) for s in out] == [(0, 120)]


def test_two_different_scenes_either_side_of_a_seam_stay_two_shots():
    a = normalise_segments([seg(0, 60, summary="a man walks along a beach")], 0, 60, unsure_if_none=False)
    b = normalise_segments([seg(60, 120, summary="crowded city market stalls at night")], 60, 120, unsure_if_none=False)
    assert len(combine_windows([a, b])) == 2


def test_dead_air_across_a_seam_is_one_stretch():
    a = normalise_segments([seg(0, 60, "dead")], 0, 60, unsure_if_none=False)
    b = normalise_segments([seg(60, 120, "dead")], 60, 120, unsure_if_none=False)
    out = combine_windows([a, b])
    assert [(s.kind, s.start_s, s.end_s) for s in out] == [("dead", 0, 120)]


def test_whatever_the_model_says_the_segments_tile_the_stretch_exactly():
    rng = random.Random(11)
    for _ in range(3000):
        proposed = []
        for _ in range(rng.randint(1, 7)):
            a = rng.uniform(-10, 70)
            proposed.append(seg(a, a + rng.uniform(0, 50), rng.choice(["usable", "setup", "dead"])))
        out = normalise_segments(proposed, 0, 60)
        assert out[0].start_s == 0 and out[-1].end_s == 60
        for left, right in zip(out, out[1:]):
            assert left.end_s == pytest.approx(right.start_s), proposed
        assert all(s.end_s > s.start_s for s in out)


async def test_at_most_the_configured_number_of_shots_per_file_across_scenes(workspace, store, clip):
    workspace.ingest.max_segments_per_source = 2

    class Many(MockVisionProvider):
        async def segment(self, frames, times, context, retry_error=None):
            from broll.analysis.segmentation import SegmentationResult
            span = (context.end_s - context.window_start_s) / 4
            return SegmentationResult(segments=[
                ModelSegment(start_s=context.window_start_s + i * span, end_s=context.window_start_s + (i + 1) * span,
                             kind=SegmentKind.usable, summary=f"scene {i}") for i in range(4)])

    path = clip("many_scenes.mp4", 24)
    d = DiscoveredFile(origin="local", path=path, filename="many_scenes.mp4", origin_path=str(path))
    result = await IngestPipeline(workspace, store, analyzer=Analyzer(workspace, provider=Many())).ingest(d)
    assert store.count_shots() == 2
    assert any("not indexed" in m and "at most 2" in m for m in result.messages)


# 5. approving a folder really moves the clips ------------------------------------------------------------


def test_approving_a_suggested_folder_ends_the_doubt_that_kept_the_clip_in_review(workspace, store):
    workspace.taxonomy = tree_config()
    client = FakeDriveClient()
    original = client.add_file("IMG_1.mov", "root")     # the clip is already in Drive
    source_id = _shot_in_drive(
        workspace, store, drive_file_id=original.id, caption="A man wakeboards.",
        category="05_Travel & Adventure/Beach & Water", category_confidence=0.4, confidence=0.9,
        status="needs_review", review_reasons=["low_category_confidence"])
    [shot] = store.shots_for_source(source_id)
    store.record_folder_proposal("05_Travel & Adventure/Wakeboarding", "Towed.", shot.id)
    proposal = store.list_folder_proposals()[0]["id"]

    Organizer(workspace, store, client).organise_source(source_id)
    assert any("_Needs Review/" in p and p.endswith(".mov") for p in client.tree())

    approve_folder_proposal(workspace, store, proposal)
    moved = store.get_shot(shot.id)
    assert moved.status == "indexed" and moved.review_reasons == []
    Organizer(workspace, store, client).organise_source(source_id)
    files = [p for p in client.tree() if p.endswith(".mov")]
    assert any("Wakeboarding/" in p for p in files) and not any("_Needs Review/" in p for p in files)


async def test_a_folder_approved_while_the_worker_runs_is_offered_to_the_next_clip(workspace, tmp_path):
    workspace.taxonomy = tree_config()
    workspace.save()
    analyzer = Analyzer(workspace, provider=MockVisionProvider())
    assert not any("Wakeboarding" in o for o in analyzer.category_options)
    # another process (`broll folders approve`) adds the folder and saves the config
    from broll.config import load_workspace_config
    other = load_workspace_config(workspace.id)
    other.taxonomy.add_folder("05_Travel & Adventure", "Wakeboarding", "Towed water sports.")
    other.save()
    import os
    os.utime(workspace.config_path, (time.time() + 5, time.time() + 5))
    analyzer._prepare(__import__("broll.analysis.schema", fromlist=["ShotContext"]).ShotContext(
        source_filename="a.mp4", duration_s=3, width=1, height=1))
    assert any("Wakeboarding" in o for o in analyzer.category_options)


# 6. re-planning never throws away what a person has touched ------------------------------------------------


def test_a_new_plan_never_deletes_a_starred_or_used_shot_or_one_edited_by_hand(workspace, store):
    source_id = _shot_in_drive(workspace, store, caption="zero")
    from broll.db.models import Shot
    ids = {}
    for index, label in ((1, "plain"), (2, "starred"), (3, "used"), (4, "edited")):
        shot = store.insert_shot(Shot(id=f"{source_id}-{index}", workspace_id=workspace.id, source_id=source_id,
                                      shot_index=index, caption=label, status="indexed"))
        ids[label] = shot.id
    store.set_shot_fields(ids["starred"], top_pick=1)
    store.record_usage([ids["used"]], "video-1")
    apply_correction(workspace, store, ids["edited"], {"caption": "by hand"})   # a shot with no raw analysis

    assert store.prune_shots(source_id, keep=1) == 1
    remaining = {s.caption for s in store.shots_for_source(source_id)}
    assert remaining == {"zero", "starred", "used", "by hand"}


# 7. disk -----------------------------------------------------------------------------------------------


def test_two_downloads_cannot_together_overshoot_the_disk(tmp_path, monkeypatch):
    import shutil
    free = 30 * GB
    monkeypatch.setattr(shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(100 * GB, 70 * GB, free))
    with reserve_room(tmp_path, 14 * GB, headroom_gb=6):
        with pytest.raises(DiskSpaceError):
            with reserve_room(tmp_path, 14 * GB, headroom_gb=6):   # 30 - 14 - 14 = 2 < 6
                pass
    with reserve_room(tmp_path, 14 * GB, headroom_gb=6):            # the first finished: room again
        pass


def test_a_file_already_on_the_disk_is_not_charged_against_it_again():
    from broll.config import IngestConfig
    from broll.ingest.limits import check_limits

    cfg = IngestConfig()
    assert check_limits(cfg, size_bytes=9 * GB, duration_s=100, forced=True, free_bytes=None) is None
    assert check_limits(cfg, size_bytes=9 * GB, duration_s=100, forced=True, free_bytes=14 * GB).kind == "too_big"


def test_an_upload_is_refused_when_the_disk_is_nearly_full(workspace, monkeypatch):
    from fastapi.testclient import TestClient
    from broll.web import app as app_module
    from broll.web.routes import ingest as ingest_routes

    monkeypatch.setattr(ingest_routes, "_has_room", lambda config: False)
    client = TestClient(app_module.create_app(workspace, run_worker=False))
    page = client.post("/ingest/upload", files=[("files", ("a.mp4", b"x" * 100, "video/mp4"))]).text
    assert "disk is too full" in page and "a.mp4" in page
    assert not list(workspace.staging_dir.glob("*"))
    body = client.post("/api/upload", files=[("files", ("b.mp4", b"x" * 100, "video/mp4"))]).json()
    assert body["queued"] == 0 and "almost out of disk" in body["skipped"][0]


# 8. the Needs Attention mirror in Drive ----------------------------------------------------------------------


def _flag(store, client, name, kind="too_long", **kw):
    entry = client.add_file(name, "root")
    return attention.flag_file(store, DiscoveredFile(
        origin="drive", path=None, filename=name, drive_file_id=entry.id, origin_path=f"drive:{entry.id}", **kw),
        kind, "why"), entry


def test_one_bad_file_does_not_stop_the_others_being_mirrored(workspace, store):
    client = FakeDriveClient()
    _flag(store, client, "a.mp4"); _flag(store, client, "b.mp4"); _flag(store, client, "c.mp4")
    real = client.create_shortcut

    def flaky(target, name, parent):
        if name == "a.mp4":
            raise RuntimeError("Drive said no")
        return real(target, name, parent)

    client.create_shortcut = flaky
    assert attention.sync_drive_shortcuts(workspace, store, client) == 2
    assert {i["filename"] for i in store.list_attention() if i["shortcut_id"]} == {"b.mp4", "c.mp4"}
    client.create_shortcut = real
    assert attention.sync_drive_shortcuts(workspace, store, client) == 1, "the one that failed is tried again"


def test_rescanning_after_a_clear_does_not_double_the_shortcuts(workspace, store):
    client = FakeDriveClient()
    item, entry = _flag(store, client, "a.mp4")
    attention.sync_drive_shortcuts(workspace, store, client)
    store.conn.execute("UPDATE attention SET shortcut_id = NULL")       # the library was cleared and rescanned
    assert attention.sync_drive_shortcuts(workspace, store, client) == 0
    assert len(client.shortcuts()) == 1


def test_a_shortcut_goes_when_its_file_is_indexed_or_dismissed_and_the_file_is_never_touched(workspace, store):
    client = FakeDriveClient()
    one, e1 = _flag(store, client, "a.mp4")
    two, e2 = _flag(store, client, "b.mp4")
    attention.sync_drive_shortcuts(workspace, store, client)
    assert len(client.shortcuts()) == 2
    store.resolve_attention(f"drive:{e1.id}")
    attention.dismiss(store, two)
    attention.sync_drive_shortcuts(workspace, store, client)
    assert client.shortcuts() == []
    assert e1.id in client._files and e2.id in client._files, "the footage itself is untouched"


def test_dismissing_an_upload_frees_the_server_copy_but_never_a_local_file(workspace, store, tmp_path):
    workspace.ensure_dirs()
    staged = workspace.staging_dir / "big.mp4"
    staged.write_bytes(b"x")
    elsewhere = tmp_path / "keep.mp4"
    elsewhere.write_bytes(b"y")
    up = attention.flag_file(store, DiscoveredFile(origin="upload", path=staged, filename="big.mp4",
                                                   origin_path=str(staged)), "too_big", "x")
    local = attention.flag_file(store, DiscoveredFile(origin="local", path=elsewhere, filename="keep.mp4",
                                                      origin_path=str(elsewhere)), "too_long", "x")
    attention.dismiss(store, up, workspace)
    attention.dismiss(store, local, workspace)
    assert not staged.exists() and elsewhere.exists()


def test_asking_twice_to_index_the_same_file_queues_it_once(workspace, store):
    f = DiscoveredFile(origin="drive", path=None, filename="a.mp4", drive_file_id="a", origin_path="drive:a")
    item = attention.flag_file(store, f, "too_long", "x")
    assert attention.requeue(workspace, store, item) and attention.requeue(workspace, store, item)
    assert store.conn.execute("SELECT COUNT(*) FROM jobs WHERE status = 'queued'").fetchone()[0] == 1


async def test_a_listed_file_that_turns_out_to_be_indexed_already_comes_off_the_list(workspace, store, clip):
    path = clip("quick.mp4", 5)
    d = DiscoveredFile(origin="local", path=path, filename="quick.mp4", origin_path=str(path))
    pipeline = IngestPipeline(workspace, store)
    await pipeline.ingest(d)
    item = attention.flag_file(store, d, "too_long", "x")        # listed by mistake or earlier
    result = await pipeline.ingest(d)
    assert result.deduped and store.get_attention(item)["status"] == "resolved"


# 9. reading from Drive ------------------------------------------------------------------------------------


def test_a_server_that_ignores_the_range_header_cannot_fill_the_memory(clip):
    import http.server
    from broll.ingest.remote import RemoteDriveVideo

    path = clip()
    data = path.read_bytes()

    class Greedy(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            self.send_response(200)                       # ignores Range: sends everything
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Greedy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        remote = RemoteDriveVideo(TokenClient(), "i", "x.mp4", size=len(data),
                                  url=f"http://127.0.0.1:{server.server_address[1]}/x")
        assert len(remote._range(0, 99)) == 100, "only what was asked for is kept"
        with pytest.raises(RuntimeError, match="ignored a byte-range request"):
            remote._range(len(data) - 50, len(data) - 1)
    finally:
        server.shutdown()


def test_a_drive_hiccup_is_retried(clip, monkeypatch):
    import http.server
    from broll.ingest.remote import RemoteDriveVideo

    monkeypatch.setattr("broll.ingest.remote.time.sleep", lambda s: None)
    hits = {"n": 0}

    class Hiccup(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            hits["n"] += 1
            if hits["n"] < 3:
                self.send_response(503); self.end_headers(); return
            self.send_response(206)
            self.send_header("Content-Length", "10")
            self.end_headers()
            self.wfile.write(b"0123456789")

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Hiccup)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        remote = RemoteDriveVideo(TokenClient(), "i", "x.mp4", size=100,
                                  url=f"http://127.0.0.1:{server.server_address[1]}/x")
        assert remote._range(0, 9) == b"0123456789" and hits["n"] == 3
    finally:
        server.shutdown()


# 10. small things --------------------------------------------------------------------------------------------


def test_the_cli_can_be_run_as_a_module():
    out = subprocess.run([".venv/bin/python", "-m", "broll.cli", "attention", "--help"],
                         capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert out.returncode == 0 and "Files the library did not index" in out.stdout


def test_a_pilot_of_nothing_is_refused(cli_workspace_for_review):
    from typer.testing import CliRunner
    from broll.cli import app

    result = CliRunner().invoke(app, ["index", "--drive-folder", "F", "--sample", "0", "-w", "fixtures"])
    assert result.exit_code == 1


@pytest.fixture()
def cli_workspace_for_review(broll_home):
    from typer.testing import CliRunner
    from broll.cli import app

    assert CliRunner().invoke(app, ["init", "--name", "Fixtures", "--provider", "mock"]).exit_code == 0


async def test_fetching_the_best_part_cuts_to_the_strongest_stretch(workspace, tmp_path):
    from tests.test_fetch import add_shot as add_fetch_shot, fetch, ramp_clip

    clip_path = ramp_clip(tmp_path / "ramp.mp4", seconds=5.0)
    shot = add_fetch_shot(workspace, clip_path, 0.0, 5.0)
    store = __import__("broll.db.store", fromlist=["Store"]).Store.for_config(workspace)
    store.set_shot_fields(shot.id, best_start_s=2.0, best_end_s=3.0)
    store.close()
    whole = fetch(workspace, shot, trim=True, handles_s=0.0)
    best = fetch(workspace, shot, trim=True, handles_s=0.0, best_part=True)
    assert (best.shot_start_s, best.shot_end_s) == (2.0, 3.0)
    assert best.duration_s == pytest.approx(1.0, abs=0.2) and (whole.shot_start_s, whole.shot_end_s) == (0.0, 5.0)
