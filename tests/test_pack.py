"""The download pack: every chosen clip cut out of Drive, plus a timeline that imports with the media online.

The timing tests need no ffmpeg. The cutting tests use clips whose every frame's brightness is its frame number
(see test_fetch), so they can ask what a render asks: when the trimmed file is read at the timeline's in-point,
is the first frame really the shot's first frame? Drive is a local web server that honours Range requests, and a
fake client whose download() fails the test, so "never downloads the whole file" is checked, not assumed.
"""

from __future__ import annotations

import dataclasses
import re
import shutil
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from fastapi.testclient import TestClient

from broll.config import WorkspaceConfig
from broll.db.models import Shot, Source
from broll.db.store import Store, new_id
from broll.fetch import fetch_shot
from broll.ingest.hashing import content_hash
from broll.transcript import pack as pack_module
from broll.transcript.exporters import edl, fcp7xml
from broll.transcript.exporters.base import PackedClip, build_timeline, clean_folder, packed_pathurl
from broll.transcript.matcher import BeatMatch, Suggestion
from broll.transcript.pack import PackJob, build_pack, clip_file_name, zip_base_name
from broll.transcript.parser import Beat
from broll.web.app import create_app
from broll.web.routes.transcript import TranscriptRun
from tests.test_fetch import FPS, add_shot, frame_luma, luma_at, ramp_clip
from tests.test_intake import RangeServer, TokenClient
from tests.timeline_fixture import build_config, build_matches

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is needed to cut footage")


# -- where the media is, as the editor's machine sees it ------------------------------------------


def test_a_mac_folder_becomes_a_file_url_inside_clips():
    assert packed_pathurl("/Users/sam/Downloads/my pack/", "001_00m00s_a b.mp4") == (
        "file://localhost/Users/sam/Downloads/my%20pack/clips/001_00m00s_a%20b.mp4")


def test_a_windows_folder_becomes_the_url_form_premiere_writes():
    assert packed_pathurl("C:\\Users\\Sam Lee\\Downloads\\pack\\", "001.mp4") == (
        "file://localhost/C%3a/Users/Sam%20Lee/Downloads/pack/clips/001.mp4")
    assert packed_pathurl("D:/footage", "x.mov") == "file://localhost/D%3a/footage/clips/x.mov"


@pytest.mark.parametrize("typed", ["", "   ", "Downloads/pack", "~/Downloads", "pack"])
def test_a_folder_that_is_not_a_full_path_is_refused_with_a_hint(typed):
    with pytest.raises(ValueError) as caught:
        clean_folder(typed)
    assert "folder" in str(caught.value).lower()


def test_files_are_named_for_their_line_and_start():
    assert clip_file_name(1, 12.9, "Walking alone (final).MOV", ".MP4") == "001_00m12s_Walking_alone_final.mp4"
    assert clip_file_name(12, 3725, "x.mov", ".mov") == "012_62m05s_x.mov"
    assert zip_base_name("My script: v2") == "My_script_v2-broll-pack"


# -- the timeline for trimmed files -----------------------------------------------------------------


HANDLE = 1.0


def packed_for(matches, handle=HANDLE):
    """What a pack would hold: each shot cut with `handle` seconds before it, so the shot starts `handle` in."""
    clips = {}
    for number, match in enumerate(matches, start=1):
        if match.chosen is None:
            continue
        shot = match.chosen.shot
        clips[shot.id] = PackedClip(
            shot_id=shot.id, file=clip_file_name(number, match.beat.start_s, match.chosen.source.original_filename, ".mp4"),
            in_s=handle, out_s=handle + shot.duration_s, duration_s=shot.duration_s + 2 * handle)
    return clips


def clipitems(xml: str) -> list[ET.Element]:
    return ET.fromstring(xml.split("<!DOCTYPE xmeml>\n", 1)[1]).findall(".//clipitem")


def test_the_in_point_is_the_handle_not_the_shots_place_in_the_original():
    """The beach shot starts 2.0 s into its camera file; in the pack the file starts 1 s before the shot."""
    matches = build_matches()
    packed = packed_for(matches)
    timeline = build_timeline(matches, build_config(), packed=packed, pack_folder="/Users/sam/Pack")
    items = clipitems(fcp7xml.build(timeline, "/Users/sam/Pack"))

    assert len(items) == 3
    beach, office, coffee = items
    assert (beach.findtext("in"), beach.findtext("out")) == ("25", "175"), "1 s handle, 6 s line, 25 fps"
    assert (beach.findtext("start"), beach.findtext("end")) == ("0", "150")
    assert (office.findtext("in"), office.findtext("out")) == ("25", "225"), "8 s line at the 25 fps sequence rate"
    assert (coffee.findtext("in"), coffee.findtext("out")) == ("25", "100"), "a 3 s shot: shorter than its 8 s line"
    for item in items:
        assert int(item.findtext("out")) - int(item.findtext("in")) == int(item.findtext("end")) - int(item.findtext("start"))


def test_every_clip_points_into_the_clips_folder_with_the_media_online():
    matches = build_matches()
    timeline = build_timeline(matches, build_config(mount=None), packed=packed_for(matches),
                              pack_folder="C:\\Users\\Sam\\Pack")
    xml = fcp7xml.build(timeline, "C:\\Users\\Sam\\Pack")
    urls = re.findall(r"<pathurl>([^<]+)</pathurl>", xml)
    assert urls == [
        "file://localhost/C%3a/Users/Sam/Pack/clips/001_00m00s_beach_meditation.mp4",
        "file://localhost/C%3a/Users/Sam/Pack/clips/002_00m06s_office_meeting.mp4",
        "file://localhost/C%3a/Users/Sam/Pack/clips/003_00m14s_coffee_pour.mp4",
    ]
    assert all(not item.offline for item in timeline.items)
    assert not any("offline" in w or "drive_local_mount_path" in w for w in timeline.warnings)
    # The file's length is given so the NLE knows the handles exist, and it starts at zero timecode.
    assert re.search(r"<file id=\"file-1\">.*?<duration>350</duration>", xml, re.S)
    assert xml.count("<string>00:00:00:00</string>") >= 3


def test_the_same_shot_chosen_twice_is_one_file():
    matches = build_matches()
    again = dataclasses.replace(matches[0], beat=Beat(index=4, start_s=30.0, end_s=34.0, text="Again.", words=1))
    matches.append(again)
    timeline = build_timeline(matches, build_config(), packed=packed_for(matches[:4]), pack_folder="/Users/sam/Pack")
    xml = fcp7xml.build(timeline, "/Users/sam/Pack")
    assert len(timeline.items) == 4
    assert xml.count("<pathurl>") == 3, "a second use refers back to the first file"
    assert xml.count('<file id="file-1"') == 2 and xml.count("<name>001_00m00s_beach_meditation.mp4</name>") >= 1


def test_two_shots_of_one_camera_file_are_two_pack_files():
    matches = build_matches()
    other = matches[0].chosen.shot.model_copy(update={"id": "src-beach-1", "start_s": 20.0})
    second = BeatMatch(beat=Beat(index=4, start_s=30.0, end_s=34.0, text="Later.", words=1),
                       suggestions=[dataclasses.replace(matches[0].chosen, shot=other)])
    matches.append(second)
    packed = packed_for(matches)
    timeline = build_timeline(matches, build_config(), packed=packed, pack_folder="/Users/sam/Pack")
    xml = fcp7xml.build(timeline, "/Users/sam/Pack")
    assert xml.count("<pathurl>") == 4
    assert packed["src-beach-0"].file != packed["src-beach-1"].file


def test_a_line_whose_clip_was_not_packed_is_left_off_the_timeline():
    matches = build_matches()
    packed = packed_for(matches)
    del packed["src-office-0"]
    timeline = build_timeline(matches, build_config(), packed=packed, pack_folder="/Users/sam/Pack")
    assert [i.suggestion.shot.id for i in timeline.items] == ["src-beach-0", "src-coffee-0"]
    assert len(timeline.gaps) == 1, "only the line with no good match is a gap here; the README lists the other"


def test_the_edl_counts_from_the_trimmed_file_too():
    matches = build_matches()
    timeline = build_timeline(matches, build_config(), packed=packed_for(matches), pack_folder="/Users/sam/Pack")
    text = edl.build(timeline)
    assert "001  MSBE0001 V     C        00:00:01:00 00:00:07:00 00:00:00:00 00:00:06:00" in text
    assert "* FROM CLIP NAME: 001_00m00s_beach_meditation.mp4" in text


def test_without_a_pack_the_timeline_is_unchanged():
    matches = build_matches()
    items = clipitems(fcp7xml.build(build_timeline(matches, build_config())))
    assert items[0].findtext("in") == "50", "the shot's own place in the original, 2 s at 25 fps"


# -- cutting, with real footage --------------------------------------------------------------------


@pytest.fixture(scope="module")
def footage(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is needed to cut footage")
    folder = tmp_path_factory.mktemp("pack-footage")
    return {"a": ramp_clip(folder / "a_clip.mp4"), "b": ramp_clip(folder / "b_clip.mp4", seconds=6.0)}


def match_for(config, shot: Shot, index: int, start: float, end: float, text: str) -> BeatMatch:
    store = Store.for_config(config)
    try:
        source = store.get_source(shot.source_id)
    finally:
        store.close()
    return BeatMatch(beat=Beat(index=index, start_s=start, end_s=end, text=text, words=3),
                     suggestions=[Suggestion(shot=shot, source=source, reason="fits")])


def job_for(config, folder="/Users/sam/Pack", handles=1.0, name="demo") -> PackJob:
    return PackJob(id="job1", run_id="r", name=name, folder=folder, handles_s=handles,
                   zip_name=zip_base_name(name) + ".zip", keep_hours=config.transcript.pack_keep_hours,
                   max_gb=config.transcript.pack_max_gb)


def frame_at_in(zf: zipfile.ZipFile, name: str, in_s: float, tmp: Path) -> float:
    target = tmp / Path(name).name
    target.write_bytes(zf.read(name))
    return luma_at(target, in_s)


@needs_ffmpeg
def test_a_pack_holds_trimmed_files_and_a_timeline_that_starts_on_each_shot(workspace, footage, tmp_path):
    first = add_shot(workspace, footage["a"], 2.0, 3.6)
    second = add_shot(workspace, footage["b"], 3.04, 4.0)
    matches = [match_for(workspace, first, 0, 0.0, 4.0, "First line."),
               match_for(workspace, second, 1, 4.0, 7.0, "Second line."),
               BeatMatch(beat=Beat(index=2, start_s=7.0, end_s=10.0, text="Nothing for this.", words=3),
                         no_good_match=True, missing_footage="Close-up of a hand signing.")]
    job = job_for(workspace)
    job_dir = tmp_path / "job"
    build_pack(workspace, matches, job, job_dir)

    assert job.state == "done" and job.packed_lines == 2
    assert job.zip_path.is_file()
    assert [p.name for p in job_dir.iterdir()] == [job.zip_name], "the working files are gone"
    with zipfile.ZipFile(job.zip_path) as zf:
        names = sorted(zf.namelist())
        assert names == ["README.txt", "clips/001_00m00s_a_clip.mp4", "clips/002_00m04s_b_clip.mp4",
                         "timeline.edl", "timeline.xml"]
        xml = zf.read("timeline.xml").decode()
        items = clipitems(xml)
        # Reading each trimmed file at the timeline's in-point lands on the shot's first frame.
        for item, (name, shot_start) in zip(items, [("clips/001_00m00s_a_clip.mp4", 2.0),
                                                    ("clips/002_00m04s_b_clip.mp4", 3.04)]):
            in_s = int(item.findtext("in")) / FPS
            assert frame_at_in(zf, name, in_s, tmp_path) == pytest.approx(frame_luma(shot_start), abs=2.5)
            assert in_s <= 1.5 + 1e-6 and in_s >= 1.0 - 0.05, "the handle (a copy may start up to a second earlier)"
        assert "file://localhost/Users/sam/Pack/clips/001_00m00s_a_clip.mp4" in xml
        readme = zf.read("README.txt").decode()
        assert "LINES WITH NO CLIP" in readme and "Close-up of a hand signing." in readme
        assert "/Users/sam/Pack/clips" in readme and "Link Media" in readme
        assert "without opening it in Premiere" in readme, "the README says what was not verified"
    assert [g.number for g in job.gaps] == [3]


@needs_ffmpeg
def test_the_same_shot_for_two_lines_is_cut_once(workspace, footage, tmp_path):
    shot = add_shot(workspace, footage["a"], 1.0, 2.0)
    matches = [match_for(workspace, shot, 0, 0.0, 3.0, "One."), match_for(workspace, shot, 1, 3.0, 6.0, "Two.")]
    job = job_for(workspace)
    build_pack(workspace, matches, job, tmp_path / "job")
    with zipfile.ZipFile(job.zip_path) as zf:
        assert [n for n in zf.namelist() if n.startswith("clips/")] == ["clips/001_00m00s_a_clip.mp4"]
        assert len(clipitems(zf.read("timeline.xml").decode())) == 2
    assert job.total == 1 and job.packed_lines == 2


@needs_ffmpeg
def test_only_the_stretch_a_line_uses_is_cut_from_a_long_shot(workspace, footage, tmp_path):
    """A 4 s shot under a 1.5 s line: the file holds the 1.5 s plus handles, not the whole shot."""
    from broll.fetch import probe_video

    shot = add_shot(workspace, footage["b"], 1.0, 5.0)
    job = job_for(workspace, handles=0.5)
    build_pack(workspace, [match_for(workspace, shot, 0, 0.0, 1.5, "A short line.")], job, tmp_path / "job")
    with zipfile.ZipFile(job.zip_path) as zf:
        name = "clips/001_00m00s_b_clip.mp4"
        target = tmp_path / "cut.mp4"
        target.write_bytes(zf.read(name))
        length = probe_video(target)["duration"]
        item = clipitems(zf.read("timeline.xml").decode())[0]
        in_s, out_s = int(item.findtext("in")) / FPS, int(item.findtext("out")) / FPS
    assert length < 3.5, f"{length}s: the whole 4 s shot plus handles would be 5 s"
    assert out_s - in_s == pytest.approx(1.5, abs=0.05)
    assert out_s <= length + 0.05, "the whole slot is inside the file"
    assert luma_at(target, in_s) == pytest.approx(frame_luma(1.0), abs=2.5)
    assert luma_at(target, out_s - 0.08) == pytest.approx(frame_luma(2.42), abs=4)


@needs_ffmpeg
def test_a_still_image_is_copied_as_it_is(workspace, tmp_path):
    photo = tmp_path / "beach photo.jpg"
    import subprocess
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=64x48", "-frames:v", "1",
                    str(photo)], check=True)
    store = Store.for_config(workspace)
    source = store.insert_source(Source(
        id=new_id(), workspace_id=workspace.id, content_hash=content_hash(photo),
        original_filename=photo.name, origin="local", media_kind="image", origin_path=str(photo),
        width=64, height=48, status="indexed"))
    shot = store.insert_shot(Shot(id=f"{source.id}-0", workspace_id=workspace.id, source_id=source.id,
                                  start_s=0, end_s=0, duration_s=0, caption="a photo", status="indexed"))
    store.close()
    matches = [match_for(workspace, shot, 0, 0.0, 4.0, "A photo line.")]
    job = job_for(workspace)
    build_pack(workspace, matches, job, tmp_path / "job")
    with zipfile.ZipFile(job.zip_path) as zf:
        assert zf.read("clips/001_00m00s_beach_photo.jpg") == photo.read_bytes()
        item = clipitems(zf.read("timeline.xml").decode())[0]
        assert (item.findtext("in"), item.findtext("out")) == ("0", "100"), "held for the whole 4 s line"
    assert photo.is_file(), "the library's own file is never deleted"


@needs_ffmpeg
def test_a_pack_that_hits_its_size_limit_says_which_lines_it_left_out(workspace, footage, tmp_path, monkeypatch):
    monkeypatch.setattr(pack_module, "_estimate_bytes", lambda *a, **k: 1)
    first = add_shot(workspace, footage["a"], 1.0, 1.4)
    store = Store.for_config(workspace)
    shots = [first] + [store.insert_shot(first.model_copy(update={
        "id": f"{first.source_id}-{n}", "shot_index": n, "start_s": 1.0 + n * 0.4, "end_s": 1.4 + n * 0.4}))
        for n in (1, 2)]
    store.close()
    matches = [match_for(workspace, s, i, i * 4.0, i * 4.0 + 4.0, f"Line {i + 1}.") for i, s in enumerate(shots)]

    roomy = job_for(workspace)
    build_pack(workspace, matches, roomy, tmp_path / "roomy")
    with zipfile.ZipFile(roomy.zip_path) as zf:
        sizes = [zf.getinfo(n).file_size for n in zf.namelist() if n.startswith("clips/")]
    assert len(sizes) == 3 and not roomy.limit_hit

    tight = job_for(workspace)
    tight.max_gb = (sizes[0] + sizes[1] / 2) / 1e9
    build_pack(workspace, matches, tight, tmp_path / "tight")
    assert tight.limit_hit and tight.packed_lines == 1
    assert [s.number for s in tight.skipped] == [2, 3]
    assert all("limit" in s.reason for s in tight.skipped)
    with zipfile.ZipFile(tight.zip_path) as zf:
        assert [n for n in zf.namelist() if n.startswith("clips/")] == ["clips/001_00m00s_a_clip.mp4"]
        readme = zf.read("README.txt").decode()
        assert "SIZE LIMIT" in readme and "Line 2." in readme and "Line 3." in readme
        assert len(clipitems(zf.read("timeline.xml").decode())) == 1


@needs_ffmpeg
def test_a_server_short_of_disk_packs_nothing_and_says_so(workspace, footage, tmp_path):
    workspace.ingest.disk_headroom_gb = 10_000_000.0  # more than any disk has
    shot = add_shot(workspace, footage["a"], 1.0, 2.0)
    job = job_for(workspace)
    with pytest.raises(RuntimeError) as caught:
        build_pack(workspace, [match_for(workspace, shot, 0, 0.0, 3.0, "One.")], job, tmp_path / "job")
    assert "No clip could be packed" in str(caught.value) and "disk" in str(caught.value)
    assert not (tmp_path / "job" / "work").exists()


@needs_ffmpeg
def test_one_missing_file_does_not_sink_the_rest(workspace, footage, tmp_path):
    good = add_shot(workspace, footage["a"], 1.0, 2.0)
    gone = add_shot(workspace, footage["b"], 1.0, 2.0, origin_path=str(tmp_path / "gone.mp4"))
    matches = [match_for(workspace, gone, 0, 0.0, 3.0, "Gone."), match_for(workspace, good, 1, 3.0, 6.0, "Good.")]
    job = job_for(workspace)
    build_pack(workspace, matches, job, tmp_path / "job")
    assert job.packed_lines == 1
    assert [(s.number, "is no longer there" in s.reason) for s in job.skipped] == [(1, True)]
    with zipfile.ZipFile(job.zip_path) as zf:
        assert "Gone." in zf.read("README.txt").decode()
        assert len(clipitems(zf.read("timeline.xml").decode())) == 1


# -- Drive: cut in place, never downloaded ---------------------------------------------------------


class NoDownloadDrive(TokenClient):
    """A Drive that hands out a token and fails the test if anyone asks it for a whole file."""

    def __init__(self):
        self.downloads = 0

    def download(self, file_id, destination):
        self.downloads += 1
        raise AssertionError("the whole file must never be downloaded")

    def get(self, file_id):
        return None


@pytest.fixture()
def drive_clip(footage, tmp_path, monkeypatch):
    """A Drive-only video served over HTTP, and the plumbing so fetch.py reads it from there."""
    served = tmp_path / "served.mp4"
    shutil.copyfile(footage["b"], served)
    server = RangeServer(served)
    import broll.fetch as fetch_module

    class Served(fetch_module.RemoteDriveVideo):
        def __init__(self, client, file_id, name, size=None, url=None):
            super().__init__(client, file_id, name, size, url=server.url)

    drive = NoDownloadDrive()
    monkeypatch.setattr(fetch_module, "RemoteDriveVideo", Served)
    monkeypatch.setattr(fetch_module, "drive_client_for", lambda config: drive)
    yield served, server, drive
    server.close()


def add_drive_shot(config, served: Path, start: float, end: float) -> Shot:
    store = Store.for_config(config)
    try:
        source = store.insert_source(Source(
            id=new_id(), workspace_id=config.id, content_hash=content_hash(served),
            original_filename="Camera A 0042.mp4", origin="drive", origin_path=None,
            drive_file_id="drive-42", drive_path="Videos/Camera A 0042.mp4", filesize_bytes=served.stat().st_size,
            duration_s=6.0, width=160, height=90, fps=FPS, status="indexed"))
        return store.insert_shot(Shot(id=f"{source.id}-0", workspace_id=config.id, source_id=source.id,
                                      start_s=start, end_s=end, duration_s=end - start, caption="a ramp",
                                      status="indexed"))
    finally:
        store.close()


@needs_ffmpeg
def test_a_drive_video_is_cut_where_it_lives_with_only_ranges_read(workspace, drive_clip, tmp_path):
    served, server, drive = drive_clip
    shot = add_drive_shot(workspace, served, 3.04, 4.2)
    store = Store.for_config(workspace)
    try:
        result = fetch_shot(workspace, store, shot.id, dest_dir=tmp_path / "out", trim=True, handles_s=0.5,
                            stream=True, drive=drive)
    finally:
        store.close()
    assert result.via == "drive_stream" and result.trimmed
    assert drive.downloads == 0
    assert luma_at(result.path, result.in_s) == pytest.approx(frame_luma(3.04), abs=2.5)
    assert luma_at(result.path, result.in_s + 0.4) == pytest.approx(frame_luma(3.44), abs=2.5)
    assert all(auth == "Bearer tok" for _, auth in server.requests), "every read carried the token"
    assert server.requests and all(rng for rng, _ in server.requests), "only byte ranges were asked for"


@needs_ffmpeg
def test_a_drive_copy_that_is_not_the_indexed_file_is_not_packed(workspace, drive_clip, footage, tmp_path):
    served, server, drive = drive_clip
    shot = add_drive_shot(workspace, served, 3.0, 4.0)
    store = Store.for_config(workspace)
    store.conn.execute("UPDATE sources SET content_hash = 'something-else' WHERE id = ?", (shot.source_id,))
    store.conn.commit()
    store.close()
    job = job_for(workspace)
    with pytest.raises(RuntimeError):
        build_pack(workspace, [match_for(workspace, shot, 0, 0.0, 3.0, "One.")], job, tmp_path / "job")
    assert "not the file that was indexed" in job.skipped[0].reason


@needs_ffmpeg
def test_a_pack_of_drive_clips_never_downloads_them(workspace, drive_clip, tmp_path):
    served, server, drive = drive_clip
    shot = add_drive_shot(workspace, served, 3.04, 4.2)
    job = job_for(workspace)
    build_pack(workspace, [match_for(workspace, shot, 0, 0.0, 3.0, "From Drive.")], job, tmp_path / "job")
    assert job.state == "done" and drive.downloads == 0
    with zipfile.ZipFile(job.zip_path) as zf:
        name = "clips/001_00m00s_Camera_A_0042.mp4"
        assert name in zf.namelist()
        item = clipitems(zf.read("timeline.xml").decode())[0]
        in_s = int(item.findtext("in")) / FPS
        assert frame_at_in(zf, name, in_s, tmp_path) == pytest.approx(frame_luma(3.04), abs=2.5)


# -- the screen and the background job -------------------------------------------------------------


def wait_for(client, url, text, seconds=60):
    deadline = time.time() + seconds
    while time.time() < deadline:
        body = client.get(url).text
        if text in body:
            return body
        assert "could not be built" not in body, body
        time.sleep(0.2)
    raise AssertionError(f"never saw {text!r} at {url}; last: {body}")


@needs_ffmpeg
def test_the_button_builds_a_pack_in_the_background_and_offers_the_zip(workspace, footage, tmp_path):
    shot = add_shot(workspace, footage["a"], 2.0, 3.0)
    gap = BeatMatch(beat=Beat(index=1, start_s=4.0, end_s=8.0, text="No clip here.", words=3), no_good_match=True,
                    missing_footage="A hand signing.")
    app = create_app(workspace, run_worker=False)
    with TestClient(app) as client:
        state = app.state.broll
        state.runs["r1"] = TranscriptRun(id="r1", name="my script", matches=[
            match_for(workspace, shot, 0, 0.0, 4.0, "The first line."), gap])
        page = client.get("/transcript")
        assert page.status_code == 200 and "broll.pack." in page.text

        started = client.post("/transcript/r1/pack", data={"folder": "/Users/sam/Downloads/pack", "handles": "1"})
        assert started.status_code == 200
        assert 'hx-trigger="every 2s"' in started.text or "Download" in started.text
        job_id = re.search(r"/transcript/r1/pack/([0-9a-f]+)", started.text).group(1)

        done = wait_for(client, f"/transcript/r1/pack/{job_id}", "my_script-broll-pack.zip")
        assert 'hx-trigger="every 2s"' not in done, "polling stops when the pack is ready"
        assert "/Users/sam/Downloads/pack/clips" in done
        assert "listed in the README as gaps" in done

        download = client.get(f"/transcript/r1/pack/{job_id}/download")
        assert download.status_code == 200 and download.headers["content-type"] == "application/zip"
        assert "my_script-broll-pack.zip" in download.headers["content-disposition"]
        archive = tmp_path / "got.zip"
        archive.write_bytes(download.content)
        with zipfile.ZipFile(archive) as zf:
            assert "timeline.xml" in zf.namelist() and "clips/001_00m00s_a_clip.mp4" in zf.namelist()

        assert client.get("/transcript/r1/pack/nope/download").status_code == 404
        assert client.get("/transcript/other/pack/nope").status_code == 404, "an unknown run is an expired run"


def test_the_button_refuses_a_folder_that_is_not_a_full_path(workspace):
    app = create_app(workspace, run_worker=False)
    with TestClient(app) as client:
        state = app.state.broll
        state.runs["r1"] = TranscriptRun(id="r1", name="x", matches=build_matches())
        body = client.post("/transcript/r1/pack", data={"folder": "Downloads/pack"}).text
        assert "full path" in body and "hx-trigger" not in body
        body = client.post("/transcript/r1/pack", data={"folder": "/Users/sam/pack", "handles": "99"}).text
        assert "Handles must be between" in body


def test_with_no_chosen_clip_there_is_nothing_to_pack(workspace):
    app = create_app(workspace, run_worker=False)
    with TestClient(app) as client:
        state = app.state.broll
        state.runs["r1"] = TranscriptRun(id="r1", name="x", matches=[
            BeatMatch(beat=Beat(index=0, start_s=0, end_s=3, text="x", words=1), no_good_match=True)])
        assert "nothing to pack" in client.post("/transcript/r1/pack", data={"folder": "/Users/sam/p"}).text


def test_old_packs_are_swept_and_a_swept_pack_cannot_be_downloaded(workspace, tmp_path):
    manager = pack_module.PackManager(workspace)
    old = manager.root / "oldjob"
    old.mkdir(parents=True)
    (old / "x.zip").write_bytes(b"zip")
    fresh = manager.root / "freshjob"
    fresh.mkdir()
    (fresh / "y.zip").write_bytes(b"zip")
    import os
    long_ago = time.time() - (workspace.transcript.pack_keep_hours + 1) * 3600
    os.utime(old, (long_ago, long_ago))
    assert manager.sweep() == 1
    assert not old.exists() and fresh.exists()

    job = PackJob(id="j", run_id="r", name="n", folder="/x", handles_s=1, state="done", zip_path=fresh / "gone.zip")
    manager.jobs["j"] = job
    assert manager.get("j") is None


def test_a_restart_clears_what_a_killed_pack_left_behind(workspace):
    leftovers = workspace.temp_dir / "packs" / "killed" / "work"
    leftovers.mkdir(parents=True)
    (leftovers / "half.mp4").write_bytes(b"x")
    pack_module.PackManager(workspace)
    assert not (workspace.temp_dir / "packs" / "killed").exists()
