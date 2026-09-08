"""Probe, hashing and frame extraction against the fixture clips."""

from __future__ import annotations

import pytest

from broll.ingest.frames import extract_frames, frame_stats, best_frame, save_thumbnail
from broll.ingest.hashing import content_hash, short_hash
from broll.ingest.probe import NotAVideoError, probe


def test_probe_reads_metadata(clips):
    meta = probe(clips["single_static_bars"])
    assert 2.5 < meta.duration_s < 3.5
    assert (meta.width, meta.height) == (640, 360)
    assert meta.fps and 24 < meta.fps < 26
    assert meta.codec == "h264"
    assert meta.filesize_bytes > 0


def test_probe_rejects_non_video(tmp_path):
    junk = tmp_path / "notes.txt"
    junk.write_text("this is not a video")
    with pytest.raises(NotAVideoError):
        probe(junk)


def test_content_hash_is_stable_and_distinct(clips, tmp_path):
    a = clips["single_static_bars"]
    b = clips["out_of_focus"]
    assert content_hash(a) == content_hash(a)
    assert content_hash(a) != content_hash(b)

    copy = tmp_path / "renamed.mp4"
    copy.write_bytes(a.read_bytes())
    assert content_hash(copy) == content_hash(a)  # dedupe ignores the filename
    assert len(short_hash(content_hash(a))) == 8


def test_extract_frames_returns_three_downscaled_frames(clips, tmp_path):
    clip = clips["moving_push_in"]
    meta = probe(clip)
    frames = extract_frames(clip, tmp_path, 0.0, meta.duration_s, count=3, max_edge=768)
    assert len(frames) == 3
    from PIL import Image

    for frame in frames:
        with Image.open(frame) as img:
            assert max(img.size) <= 768


def test_frames_avoid_the_black_edges_of_a_fade(clips, tmp_path):
    """The 20% sample lands inside a fade-in; resampling should find a usable frame."""
    clip = clips["fade_black_edges"]
    meta = probe(clip)
    frames = extract_frames(clip, tmp_path, 0.0, meta.duration_s, count=3)
    assert len(frames) == 3
    assert all(frame_stats(f).usable for f in frames)


def test_all_dark_clip_still_yields_frames(tmp_path):
    """A clip with no usable frame is analysed and flagged, not silently skipped."""
    import subprocess

    black = tmp_path / "black.mp4"
    subprocess.run(
        ["ffmpeg", "-f", "lavfi", "-i", "color=c=black:size=320x180:rate=25:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
         "-y", "-loglevel", "error", str(black)],
        check=True,
    )
    frames = extract_frames(black, tmp_path / "frames", 0.0, 2.0, count=3)
    assert len(frames) == 3
    assert not any(frame_stats(f).usable for f in frames)


def test_best_frame_and_thumbnail(clips, tmp_path):
    clip = clips["single_static_bars"]
    frames = extract_frames(clip, tmp_path / "f", 0.0, 3.0, count=3)
    chosen = best_frame(frames)
    assert chosen in frames
    thumb = save_thumbnail(chosen, tmp_path / "thumbs" / "t.jpg", max_edge=640)
    assert thumb.exists()
    from PIL import Image

    with Image.open(thumb) as img:
        assert max(img.size) <= 640
