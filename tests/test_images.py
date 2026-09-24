"""Stills are footage too.

A photograph runs the same pipeline as a clip with the shot detection removed:
one "shot", one frame, no duration, no camera movement. What is worth testing is
the places where that difference leaks - probing, filing, filtering, exporting.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from broll.analysis.analyzer import Analyzer
from broll.analysis.schema import AnalysisResult, ShotContext
from broll.db.models import ShotFacets
from broll.drive.taxonomy import plan_tree
from broll.ingest.pipeline import IngestPipeline
from broll.ingest.probe import probe
from broll.ingest.scanner import DiscoveredFile, media_kind, scan_local
from broll.search.filters import SearchFilters
from broll.search.query import SearchEngine


def _photo(path: Path, size: tuple[int, int] = (1200, 800)) -> Path:
    """A still with enough variation to pass the frame quality filter."""
    image = Image.new("RGB", size)
    pixels = image.load()
    for x in range(size[0]):
        for y in range(0, size[1], 4):
            shade = (x * 255 // size[0], (x + y) % 255, 120)
            for dy in range(4):
                if y + dy < size[1]:
                    pixels[x, y + dy] = shade
    image.save(path, quality=92)
    return path


def test_file_types_are_sorted_into_clips_and_stills(tmp_path):
    assert media_kind("a.MP4") == "video"
    assert media_kind("holiday.HEIC") == "image"
    assert media_kind("notes.pdf") is None

    _photo(tmp_path / "still.jpg")
    (tmp_path / "notes.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "clip.mov").write_bytes(b"")
    found = {f.filename for f in scan_local(tmp_path)}
    assert found == {"still.jpg", "clip.mov"}


def test_a_still_probes_with_no_duration_and_no_frame_rate(tmp_path):
    meta = probe(_photo(tmp_path / "still.jpg"))
    assert meta.media_kind == "image"
    assert meta.duration_s == 0.0
    assert meta.fps is None
    assert (meta.width, meta.height) == (1200, 800)


async def test_a_photo_is_indexed_as_one_shot(workspace, store, tmp_path):
    photo = _photo(tmp_path / "beach.jpg")
    pipeline = IngestPipeline(workspace, store)
    result = await pipeline.ingest(
        DiscoveredFile(origin="local", path=photo, filename="beach.jpg",
                       origin_path=str(photo))
    )

    assert result.status == "indexed"
    assert result.shots_analysed == 1
    source = store.get_source(result.source_id)
    assert source.media_kind == "image"
    assert source.duration_s == 0.0

    shot = store.shots_for_source(source.id)[0]
    assert shot.duration_s == 0.0
    assert shot.thumbnail_path, "a photo still needs a thumbnail to browse by"


async def test_a_still_is_never_described_as_moving(workspace, tmp_path):
    """A model that says a photograph pans left is describing an illusion."""

    class Drifting:
        name = "drifting"

        async def analyse(self, frames, context, retry_error=None):
            return AnalysisResult.model_validate({
                "caption": "A man sits on a beach.", "setting": "beach",
                "shot_type": "wide", "camera_movement": "pan_left",
                "time_of_day": "golden_hour", "colour_profile": "warm",
                "people_count": "one", "pace": "slow", "confidence": 0.9,
            })

        def estimate_cost(self, frames):
            return 0.0

    context = ShotContext(source_filename="a.jpg", duration_s=0.0, width=1200,
                          height=800, media_kind="image")
    outcome = await Analyzer(workspace, provider=Drifting()).analyse_frames(
        [_photo(tmp_path / "f.jpg")], context
    )
    assert outcome.result.camera_movement.value == "static"
    assert outcome.result.pace.value == "still"


def test_one_frame_is_extracted_for_a_still(workspace, tmp_path):
    workspace.ingest.frames_per_shot = 3
    context = ShotContext(source_filename="a.jpg", duration_s=0.0, width=1200,
                          height=800, media_kind="image")
    frames = Analyzer(workspace).extract(
        _photo(tmp_path / "a.jpg"), context, tmp_path / "work"
    )
    assert len(frames) == 1


def test_the_tree_hangs_under_images_or_videos(workspace):
    taxonomy = workspace.taxonomy
    taxonomy.mode = "tree"
    taxonomy.media_split = True
    from broll.config import CategoryNode

    taxonomy.tree = [CategoryNode(name="01_Practices", children=[CategoryNode(name="Meditation")])]
    facets = [ShotFacets(shot_id="s1", source_id="src1",
                        category="01_Practices/Meditation", is_primary=True)]

    video_home, _ = plan_tree(facets, "a.mp4", taxonomy, taxonomy.media_prefix("video"))
    photo_home, _ = plan_tree(facets, "a.jpg", taxonomy, taxonomy.media_prefix("image"))
    assert str(video_home) == "Videos/01_Practices/Meditation"
    assert str(photo_home) == "Images/01_Practices/Meditation"

    taxonomy.media_split = False
    plain_home, _ = plan_tree(facets, "a.mp4", taxonomy, taxonomy.media_prefix("video"))
    assert str(plain_home) == "01_Practices/Meditation"


async def test_a_duration_filter_does_not_hide_photos(workspace, store, tmp_path):
    photo = _photo(tmp_path / "beach.jpg")
    pipeline = IngestPipeline(workspace, store)
    await pipeline.ingest(
        DiscoveredFile(origin="local", path=photo, filename="beach.jpg",
                       origin_path=str(photo))
    )
    engine = SearchEngine(store, embedder=None)

    # A photo has no length, so "at least 4 seconds" is not a question about it.
    assert engine.browse(SearchFilters(duration_min_s=4.0), 10)
    assert engine.browse(SearchFilters(media_kind=["image"]), 10)
    assert not engine.browse(SearchFilters(media_kind=["video"]), 10)


def test_a_photo_is_filed_under_images_and_a_clip_under_videos(workspace, store, tmp_path):
    from broll.db.models import Shot, Source
    from broll.db.store import new_id
    from broll.drive.organizer import Organizer
    from tests.fakes import FakeDriveClient
    from tests.test_client_tree import tree_config

    workspace.taxonomy = tree_config(media_split=True)
    client = FakeDriveClient()

    for index, (name, kind) in enumerate((("clip.mp4", "video"), ("still.jpg", "image"))):
        local = tmp_path / name
        local.write_bytes(b"bytes")
        source = store.insert_source(Source(
            id=new_id(), workspace_id=workspace.id, content_hash=f"hash{index:08d}",
            original_filename=name, origin="local", origin_path=str(local),
            media_kind=kind))
        store.insert_shot(Shot(
            id=f"{source.id}-0", workspace_id=workspace.id, source_id=source.id,
            caption="Sitting quietly.", action="meditating", emotions=["calm"],
            category="01_Nervous System Practices/Meditation & Stillness",
            status="indexed"))
        store.recompute_source_status(source.id)

    Organizer(workspace, store, client).reorganise()

    filed = {s.original_filename: s.drive_path for s in store.list_sources()}
    assert filed["clip.mp4"].startswith("Videos/01_Nervous System Practices/")
    assert filed["still.jpg"].startswith("Images/01_Nervous System Practices/")
    # Both trees exist in full, so an editor can browse either side.
    folders = {f.name for f in client._files.values() if f.is_folder}
    assert {"Videos", "Images", "02_Gym & Training"} <= folders


def test_a_still_fills_its_beat_rather_than_leaving_a_gap(workspace):
    from broll.db.models import Shot, Source
    from broll.transcript.exporters.base import build_timeline
    from broll.transcript.matcher import BeatMatch, Suggestion
    from broll.transcript.parser import Beat

    photo = Source(id="src", workspace_id=workspace.id, content_hash="h", origin="local",
                   original_filename="still.jpg", media_kind="image", duration_s=0.0)
    shot = Shot(id="src-0", workspace_id=workspace.id, source_id="src", duration_s=0.0)
    match = BeatMatch(
        beat=Beat(index=0, start_s=0.0, end_s=6.0, text="A quiet morning."),
        suggestions=[Suggestion(shot=shot, source=photo, reason="", confidence=0.9, score=1.0)],
    )

    timeline = build_timeline([match], workspace, name="Stills")
    item = timeline.items[0]
    assert item.gap_frames == 0
    assert item.end_frame - item.start_frame == item.sequence_out_frame - item.sequence_in_frame
    assert not any("left as a gap" in w for w in timeline.warnings)


def test_a_tiled_heic_reports_the_whole_picture_not_one_tile():
    """An iPhone HEIC is 48 tiles of 512x512; stream 0 is a tile, not the photo."""
    from broll.ingest.probe import _tile_grid

    probed = {
        "stream_groups": [{
            "type": "Tile Grid",
            "components": [{"nb_tiles": 48, "coded_width": 4096, "coded_height": 3072,
                            "width": 4032, "height": 3024}],
        }],
    }
    assert _tile_grid(probed) == (4032, 3024)
    assert _tile_grid({"streams": [{"width": 512}]}) is None


def test_extract_still_scales_without_a_video_filter(tmp_path, monkeypatch):
    """-vf is refused on a tiled HEIC, so the still path must not pass one."""
    from broll.ingest import frames as frames_module

    source = _photo(tmp_path / "big.jpg", size=(2400, 1600))
    seen: list[list[str]] = []
    real_run = frames_module.subprocess.run

    def spy(command, *args, **kwargs):
        seen.append(command)
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(frames_module.subprocess, "run", spy)
    out = frames_module.extract_still(source, tmp_path / "work", max_edge=768)

    assert len(out) == 1
    assert max(Image.open(out[0]).size) == 768
    assert not any("-vf" in command for command in seen)
    assert not list((tmp_path / "work").glob("*_full.jpg")), "the full-size decode is temporary"
