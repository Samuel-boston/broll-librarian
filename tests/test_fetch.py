"""Turning a chosen shot into a file a render can read.

The clips here are generated so that every frame's brightness is its frame
number (frame n has luma 2n), which lets a test ask the question a render
cares about: when the file is read from `in_s`, is the first frame really the
shot's first frame? Times on both sides are read the way the render reads
them, with ffmpeg's -ss.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from broll.clients import register
from broll.db.models import Shot, Source
from broll.db.store import Registry, Store, new_id
from broll.fetch import FetchError, fetch_shot
from broll.ingest.hashing import content_hash
from broll.web.app import create_studio_app

FPS = 25


def ramp_clip(path: Path, seconds: float = 5.0, gop: int = FPS) -> Path:
    """Frame n has luma 2n; a keyframe every `gop` frames; B-frames, like camera files."""
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi",
        "-i", f"color=c=black:s=160x90:r={FPS}:d={seconds},geq=lum='N*2':cb=128:cr=128",
        "-c:v", "libx264", "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
        "-bf", "2", "-pix_fmt", "yuv420p", str(path),
    ], check=True)
    return path


def luma_at(path: str | Path, t: float) -> float:
    """The brightness of the frame a render gets when it seeks to `t`."""
    cp = subprocess.run([
        "ffmpeg", "-v", "error", "-ss", f"{t:.4f}", "-i", str(path), "-frames:v", "1",
        "-vf", "signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-", "-f", "null", "-",
    ], capture_output=True, text=True)
    return float(re.search(r"YAVG=([\d.]+)", cp.stdout).group(1))


def frame_luma(t: float) -> float:
    """Luma of the first source frame at or after t."""
    return 2 * round(t * FPS + 0.4999)


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is needed to cut footage")
    folder = tmp_path_factory.mktemp("ramp")
    return {
        "short_gop": ramp_clip(folder / "short_gop.mp4"),
        "long_gop": ramp_clip(folder / "long_gop.mp4", gop=250),
    }


def add_shot(config, clip: Path, start: float, end: float, *, origin_path: str | None = "same",
             drive_file_id: str | None = None, drive_path: str | None = None,
             media_kind: str = "video") -> Shot:
    store = Store.for_config(config)
    try:
        source = store.insert_source(Source(
            id=new_id(), workspace_id=config.id, content_hash=content_hash(clip),
            original_filename=clip.name, origin="local", media_kind=media_kind,
            origin_path=str(clip) if origin_path == "same" else origin_path,
            drive_file_id=drive_file_id, drive_path=drive_path,
            duration_s=5.0, width=160, height=90, fps=FPS, status="indexed"))
        return store.insert_shot(Shot(
            id=f"{source.id}-0", workspace_id=config.id, source_id=source.id,
            start_s=start, end_s=end, duration_s=end - start, caption="a ramp", status="indexed"))
    finally:
        store.close()


def fetch(config, shot, **kwargs):
    store = Store.for_config(config)
    try:
        return fetch_shot(config, store, shot.id, **kwargs)
    finally:
        store.close()


class CopyingDrive:
    """Stands in for Drive: a download copies `bytes_from`, and is counted."""

    def __init__(self, bytes_from: Path):
        self.bytes_from = bytes_from
        self.downloads = 0

    def download(self, file_id: str, destination: Path) -> Path:
        self.downloads += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.bytes_from, destination)
        return destination


# -- finding the file ---------------------------------------------------------


def test_an_original_that_is_still_there_is_used_where_it_is(workspace, clips):
    shot = add_shot(workspace, clips["short_gop"], 1.0, 2.0)
    result = fetch(workspace, shot)
    assert result.via == "local"
    assert result.path == str(clips["short_gop"])
    assert (result.offset_s, result.in_s, result.trimmed) == (0.0, 1.0, False)


def test_a_changed_original_is_not_trusted(workspace, clips, tmp_path):
    changed = tmp_path / "changed.mp4"
    shutil.copyfile(clips["short_gop"], changed)
    shot = add_shot(workspace, changed, 1.0, 2.0)
    with changed.open("ab") as handle:
        handle.write(b"edited since")
    with pytest.raises(FetchError) as caught:
        fetch(workspace, shot)
    assert "has changed since it was indexed" in str(caught.value)


def test_the_drive_for_desktop_copy_stands_in_for_a_missing_original(workspace, clips, tmp_path):
    """drive_path is relative to the client's root folder, which sits inside My Drive."""
    mount = tmp_path / "My Drive"
    workspace.drive_local_mount_path = str(mount)
    workspace.drive_root_folder_name = "ADAM - B-ROLL"
    synced = mount / "ADAM - B-ROLL" / "Videos" / "Practices" / "ramp.mp4"
    synced.parent.mkdir(parents=True)
    shutil.copyfile(clips["short_gop"], synced)
    shot = add_shot(workspace, clips["short_gop"], 1.0, 2.0, origin_path=str(tmp_path / "gone.mp4"),
                    drive_path="Videos/Practices/ramp.mp4")

    result = fetch(workspace, shot)
    assert result.via == "drive_mount"
    assert result.path == str(synced)


def test_a_drive_only_shot_is_downloaded_once_and_checked(workspace, clips, tmp_path):
    drive = CopyingDrive(clips["short_gop"])
    shot = add_shot(workspace, clips["short_gop"], 1.0, 2.0, origin_path=str(tmp_path / "gone.mp4"),
                    drive_file_id="drive-1")

    first = fetch(workspace, shot, drive=drive)
    assert first.via == "drive_download"
    assert Path(first.path).is_relative_to(workspace.temp_dir)
    second = fetch(workspace, shot, drive=drive)
    assert second.path == first.path
    assert drive.downloads == 1, "the download is kept for the next fetch"


def test_a_drive_copy_that_is_not_the_indexed_file_is_refused(workspace, clips, tmp_path):
    shot = add_shot(workspace, clips["short_gop"], 1.0, 2.0, origin_path=str(tmp_path / "gone.mp4"),
                    drive_file_id="drive-1")
    with pytest.raises(FetchError) as caught:
        fetch(workspace, shot, drive=CopyingDrive(clips["long_gop"]))
    assert "not the file that was indexed" in str(caught.value)
    leftovers = list((workspace.temp_dir / "fetched").glob("*"))
    assert leftovers == [], "a wrong download is not kept"


def test_without_a_drive_login_the_error_says_what_was_tried_and_how_to_connect(workspace, clips, tmp_path):
    shot = add_shot(workspace, clips["short_gop"], 1.0, 2.0, origin_path=str(tmp_path / "gone.mp4"),
                    drive_file_id="drive-1")
    with pytest.raises(FetchError) as caught:
        fetch(workspace, shot)
    message = str(caught.value)
    assert caught.value.status == 503
    assert "is no longer there" in message
    assert f"broll drive login -w {workspace.id}" in message


def test_a_file_never_filed_into_drive_and_since_moved_is_reported_plainly(workspace, clips, tmp_path):
    shot = add_shot(workspace, clips["short_gop"], 1.0, 2.0, origin_path=str(tmp_path / "gone.mp4"))
    with pytest.raises(FetchError) as caught:
        fetch(workspace, shot)
    assert "has not been filed into Drive yet" in str(caught.value)


# -- cutting just the shot ------------------------------------------------------


def test_a_trim_from_a_keyframe_is_a_stream_copy_that_starts_on_the_shot(workspace, clips):
    shot = add_shot(workspace, clips["short_gop"], 2.0, 3.0)
    result = fetch(workspace, shot, trim=True, handles_s=0.0)
    assert result.trimmed and result.trim_mode == "copy"
    assert result.duration_s == pytest.approx(1.0, abs=0.15)
    assert luma_at(result.path, result.in_s) == pytest.approx(frame_luma(2.0), abs=1.5)
    assert luma_at(result.path, result.in_s + 0.4) == pytest.approx(frame_luma(2.4), abs=1.5)


def test_a_trim_between_keyframes_still_lands_on_the_shot(workspace, clips):
    """The copy starts on the keyframe before; `in_s` says where the shot begins inside it."""
    shot = add_shot(workspace, clips["short_gop"], 2.52, 3.52)
    result = fetch(workspace, shot, trim=True, handles_s=0.2)
    assert result.trim_mode == "copy"
    assert result.in_s > 0.2, "the file starts before the shot"
    assert luma_at(result.path, result.in_s) == pytest.approx(frame_luma(2.52), abs=1.5)


def test_a_long_gop_is_re_encoded_rather_than_padded_with_the_wrong_shot(workspace, clips):
    shot = add_shot(workspace, clips["long_gop"], 3.0, 4.0)
    result = fetch(workspace, shot, trim=True, handles_s=0.0)
    assert result.trim_mode == "reencode"
    assert result.path.endswith(".mp4")
    assert result.duration_s == pytest.approx(1.0, abs=0.15), "not three seconds of lead-in"
    assert luma_at(result.path, result.in_s) == pytest.approx(frame_luma(3.0), abs=1.5)
    assert luma_at(result.path, result.in_s + 0.6) == pytest.approx(frame_luma(3.6), abs=1.5)


def test_a_trim_lands_in_the_folder_it_was_asked_for(workspace, clips, tmp_path):
    assets = tmp_path / "project" / "04_ASSETS" / "BROLL"
    shot = add_shot(workspace, clips["short_gop"], 2.0, 3.0)
    result = fetch(workspace, shot, trim=True, dest_dir=assets)
    assert Path(result.path).parent == assets
    assert result.handles_s == 0.5
    assert result.in_s >= 0.5 - 1e-3, "at least the handle asked for before the shot"
    assert luma_at(result.path, result.in_s) == pytest.approx(frame_luma(2.0), abs=1.5)


def test_a_copy_puts_the_whole_file_in_the_folder_when_asked(workspace, clips, tmp_path):
    assets = tmp_path / "assets"
    shot = add_shot(workspace, clips["short_gop"], 0.0, 5.0)
    result = fetch(workspace, shot, trim=True, dest_dir=assets, copy=True)
    assert not result.trimmed, "the shot is the whole file, so there is nothing to cut"
    assert Path(result.path).parent == assets
    assert content_hash(Path(result.path)) == content_hash(clips["short_gop"])


def test_a_photograph_is_handed_over_whole(workspace, clips, tmp_path):
    still = tmp_path / "still.jpg"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clips["short_gop"]), "-frames:v", "1",
                    str(still)], check=True)
    shot = add_shot(workspace, still, 0.0, 0.0, media_kind="image")
    result = fetch(workspace, shot, trim=True)
    assert result.trimmed is False
    assert result.path == str(still)
    assert result.media == "image"


# -- the endpoint ---------------------------------------------------------------


@pytest.fixture()
def two_libraries(broll_home):
    from broll.config import WorkspaceConfig

    configs = []
    registry = Registry()
    try:
        for client_id in ("adam", "luqman"):
            config = WorkspaceConfig(id=client_id, name=client_id.title())
            config.provider.vision = "mock"
            register(config, registry)
            configs.append(config)
    finally:
        registry.close()
    return configs


def test_the_endpoint_finds_the_shot_in_whichever_client_holds_it(two_libraries, clips, tmp_path):
    _, luqman = two_libraries
    shot = add_shot(luqman, clips["short_gop"], 2.0, 3.0)
    with TestClient(create_studio_app(two_libraries, run_worker=False)) as api:
        response = api.post("/api/fetch", json={
            "shot_id": shot.id, "trim": True, "dest_dir": str(tmp_path / "BROLL"), "handles_s": 0.25,
        })
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["client"] == "luqman"
        assert body["trimmed"] is True
        assert Path(body["path"]).parent == tmp_path / "BROLL"
        assert body["in_s"] == pytest.approx(body["shot_start_s"] - body["offset_s"])

        missing = api.post("/api/fetch", json={"shot_id": "nope"})
        assert missing.status_code == 404
        relative = api.post("/api/fetch", json={"shot_id": shot.id, "dest_dir": "relative/dir"})
        assert relative.status_code == 422


def test_the_endpoint_passes_on_why_a_shot_cannot_be_fetched(two_libraries, clips, tmp_path):
    adam, _ = two_libraries
    shot = add_shot(adam, clips["short_gop"], 2.0, 3.0, origin_path=str(tmp_path / "gone.mp4"),
                    drive_file_id="drive-9")
    with TestClient(create_studio_app(two_libraries, run_worker=False)) as api:
        response = api.post("/api/fetch", json={"shot_id": shot.id, "client": "adam"})
    assert response.status_code == 503
    assert "broll drive login -w adam" in response.json()["detail"]
