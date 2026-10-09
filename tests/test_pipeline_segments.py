"""The pipeline looks inside clips: skips setup, splits scenes, keeps its plan, files doubtful clips for review."""

from __future__ import annotations

import subprocess

import pytest

from broll.analysis.providers.mock import MockVisionProvider
from broll.analysis.analyzer import Analyzer
from broll.analysis.providers.base import TransientProviderError
from broll.analysis.segmentation import Segment
from broll.ingest.pipeline import IngestPipeline
from broll.ingest.scanner import DiscoveredFile
from broll.review import apply_correction
from broll.drive.organizer import Organizer
from tests.fakes import FakeDriveClient
from tests.test_client_tree import tree_config


@pytest.fixture()
def make_clip(tmp_path):
    def make(name: str, seconds: int = 24):
        path = tmp_path / name
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
             f"testsrc2=size=320x240:rate=15:duration={seconds}", "-pix_fmt", "yuv420p", "-y", str(path)],
            check=True,
        )
        return DiscoveredFile(origin="local", path=path, filename=name, origin_path=str(path))
    return make


class Counting(MockVisionProvider):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.segment_calls = 0

    async def segment(self, *a, **k):
        self.segment_calls += 1
        return await super().segment(*a, **k)


def pipeline_with(workspace, store, provider):
    return IngestPipeline(workspace, store, analyzer=Analyzer(workspace, provider=provider))


async def test_the_setup_at_the_start_is_not_indexed(workspace, store, make_clip):
    result = await pipeline_with(workspace, store, Counting()).ingest(make_clip("beach_setup_a.mp4"))
    [source] = store.list_sources()
    [shot] = store.shots_for_source(source.id)
    assert result.shots_analysed == 1
    assert shot.start_s == pytest.approx(4.8, abs=0.2) and shot.end_s == pytest.approx(24)
    assert shot.best_start_s is not None and shot.start_s <= shot.best_start_s < shot.best_end_s <= shot.end_s
    kinds = [s["kind"] for s in source.segments]
    assert kinds == ["setup", "usable"], "what was left out is recorded, not forgotten"


async def test_a_take_with_two_scenes_becomes_two_shots(workspace, store, make_clip):
    await pipeline_with(workspace, store, Counting()).ingest(make_clip("market_scenes_a.mp4"))
    [source] = store.list_sources()
    first, second = store.shots_for_source(source.id)
    assert (first.shot_index, second.shot_index) == (0, 1)
    assert first.end_s == pytest.approx(second.start_s)
    assert first.start_s == 0 and second.end_s == pytest.approx(24)
    assert sum(s.is_primary for s in (first, second)) == 1


async def test_a_short_clip_is_not_sent_for_looking_through(workspace, store, make_clip):
    provider = Counting()
    await pipeline_with(workspace, store, provider).ingest(make_clip("quick_setup_a.mp4", seconds=5))
    assert provider.segment_calls == 0
    [shot] = store.list_shots()
    assert shot.start_s == 0, "under the minimum length there is nothing to look inside"


async def test_a_killed_run_resumes_with_the_same_plan(workspace, store, make_clip):
    """Asking the model again could give a different answer, and shot numbers would stop matching."""
    provider = Counting()
    pipeline = pipeline_with(workspace, store, provider)
    clip = make_clip("market_scenes_b.mp4")

    original = pipeline.analyzer.analyse_frames
    calls = {"n": 0}

    async def flaky(frames, context):
        calls["n"] += 1
        if calls["n"] == 2:
            raise TransientProviderError("503")
        return await original(frames, context)

    pipeline.analyzer.analyse_frames = flaky  # type: ignore[method-assign]
    with pytest.raises(TransientProviderError):
        await pipeline.ingest(clip)
    assert provider.segment_calls == 1 and store.count_shots() == 1

    pipeline.analyzer.analyse_frames = original  # type: ignore[method-assign]
    result = await pipeline.ingest(clip)
    assert provider.segment_calls == 1, "the stored plan was used"
    assert result.shots_skipped == 1 and result.shots_analysed == 1 and store.count_shots() == 2


async def test_forcing_a_new_plan_removes_shots_that_no_longer_exist(workspace, store, make_clip):
    clip = make_clip("market_scenes_c.mp4")
    await pipeline_with(workspace, store, Counting()).ingest(clip)
    assert store.count_shots() == 2

    class OneShot(Counting):
        async def segment(self, frames, times, context, retry_error=None):
            from broll.analysis.segmentation import ModelSegment, SegmentationResult, SegmentKind
            return SegmentationResult(segments=[ModelSegment(
                start_s=context.window_start_s, end_s=context.end_s, kind=SegmentKind.usable)])

    await pipeline_with(workspace, store, OneShot()).ingest(clip, force=True)
    assert store.count_shots() == 1
    [source] = store.list_sources()
    assert store.vectors.count() == 0 or len(store.shots_for_source(source.id)) == 1


async def test_a_shot_a_person_corrected_survives_a_new_plan(workspace, store, make_clip):
    clip = make_clip("market_scenes_d.mp4")
    await pipeline_with(workspace, store, Counting()).ingest(clip)
    [source] = store.list_sources()
    second = store.shots_for_source(source.id)[1]
    apply_correction(workspace, store, second.id, {"caption": "Hand-written caption"})

    class OneShot(Counting):
        async def segment(self, frames, times, context, retry_error=None):
            from broll.analysis.segmentation import ModelSegment, SegmentationResult, SegmentKind
            return SegmentationResult(segments=[ModelSegment(
                start_s=context.window_start_s, end_s=context.end_s, kind=SegmentKind.usable)])

    await pipeline_with(workspace, store, OneShot()).ingest(clip, force=True)
    assert store.get_shot(second.id).caption == "Hand-written caption"


async def test_a_file_the_model_found_nothing_usable_in_is_kept_for_a_person(workspace, store, make_clip):
    class AllSetup(Counting):
        async def segment(self, frames, times, context, retry_error=None):
            from broll.analysis.segmentation import ModelSegment, SegmentationResult, SegmentKind
            return SegmentationResult(segments=[ModelSegment(
                start_s=context.window_start_s, end_s=context.end_s, kind=SegmentKind.setup)])

    await pipeline_with(workspace, store, AllSetup()).ingest(make_clip("empty_a.mp4"))
    [shot] = store.list_shots()
    assert shot.status == "needs_review" and "no_usable_part" in shot.review_reasons
    assert (shot.start_s, shot.end_s) == (0, pytest.approx(24))


async def test_the_frames_carry_their_times_to_the_model(workspace, store, make_clip):
    seen = {}

    class Spy(Counting):
        async def analyse(self, frames, context, retry_error=None):
            seen["times"] = context.frame_times
            seen["count"] = len(frames)
            return await super().analyse(frames, context, retry_error)

    await pipeline_with(workspace, store, Spy()).ingest(make_clip("plain_long_a.mp4"))
    assert seen["count"] == len(seen["times"]) >= 5, "a 24-second shot gets more than three frames"
    assert seen["times"] == sorted(seen["times"]) and seen["times"][-1] <= 24


# -- a doubtful clip is filed for review, not guessed into a folder ----------------------------


def filed_tree_state(workspace, store, tmp_path, confidence, category_confidence):
    """Index one clip with a given confidence and file it into a fake Drive."""
    tree = tree_config()
    workspace.taxonomy = tree
    return tree


async def test_a_doubtful_clip_goes_to_the_review_folder_and_moves_when_confirmed(
    workspace, store, make_clip
):
    workspace.taxonomy = tree_config()

    class Doubtful(MockVisionProvider):
        async def analyse(self, frames, context, retry_error=None):
            result = await super().analyse(frames, context, retry_error)
            result.category = "01_Nervous System Practices/Breathwork"
            result.category_confidence = 0.35
            result.confidence = 0.9
            return result

    clip = make_clip("breath_a.mp4", seconds=5)
    await pipeline_with(workspace, store, Doubtful()).ingest(clip)
    [source] = store.list_sources()
    [shot] = store.shots_for_source(source.id)
    assert shot.status == "needs_review" and shot.review_reasons == ["low_category_confidence"]
    assert shot.category == "01_Nervous System Practices/Breathwork", "the guess is kept for the person to see"

    client = FakeDriveClient()
    Organizer(workspace, store, client).organise_source(source.id)
    tree = client.tree()
    assert any(p.startswith("B-Roll/_Needs Review/") and p.endswith(".mp4") for p in tree)
    assert not any("Breathwork/" in p for p in tree), "no shortcut into the guessed folder either"

    apply_correction(workspace, store, shot.id, {}, status="indexed")
    Organizer(workspace, store, client).organise_source(source.id)
    tree = client.tree()
    assert not any("_Needs Review/" in p and p.endswith(".mp4") for p in tree)
    assert any(p.startswith("B-Roll/01_Nervous System Practices/Breathwork/") and p.endswith(".mp4") for p in tree)
    assert store.get_shot(shot.id).review_reasons == []


async def test_a_shaky_clip_is_flagged_but_stays_in_its_folder(workspace, store, make_clip):
    """Quality is not doubt about the folder: a shaky clip of the gym still belongs in the gym."""
    workspace.taxonomy = tree_config()

    class Shaky(MockVisionProvider):
        async def analyse(self, frames, context, retry_error=None):
            result = await super().analyse(frames, context, retry_error)
            result.category = "02_Gym & Training"
            result.category_confidence = 0.95
            result.confidence = 0.95
            result.quality_flags = ["shaky"]
            return result

    await pipeline_with(workspace, store, Shaky()).ingest(make_clip("gym_a.mp4", seconds=5))
    [source] = store.list_sources()
    [shot] = store.shots_for_source(source.id)
    assert shot.status == "needs_review" and shot.review_reasons == ["quality_defect"]
    client = FakeDriveClient()
    Organizer(workspace, store, client).organise_source(source.id)
    assert any(p.startswith("B-Roll/02_Gym & Training/") for p in client.tree())
    assert not any("_Needs Review" in p for p in client.tree())


def test_routing_can_be_switched_off(workspace):
    from broll.db.models import ShotFacets
    from broll.drive.taxonomy import plan_tree

    facets = ShotFacets(shot_id="s", source_id="x", category="02_Gym & Training",
                        review_reasons_dummy=None) if False else ShotFacets(
        shot_id="s", source_id="x", category="02_Gym & Training", misfile_risk=True)
    on, _ = plan_tree([facets], "c.mov", tree_config())
    off, _ = plan_tree([facets], "c.mov", tree_config(route_uncertain_to_review=False))
    assert str(on) == "_Needs Review" and str(off) == "02_Gym & Training"


async def test_a_suggested_folder_is_held_until_approved_then_the_clips_move(workspace, store, make_clip):
    from broll.review import approve_folder_proposal, dismiss_folder_proposal

    workspace.taxonomy = tree_config()

    class Wakeboard(MockVisionProvider):
        async def analyse(self, frames, context, retry_error=None):
            result = await super().analyse(frames, context, retry_error)
            result.category = "05_Travel & Adventure/Beach & Water"
            result.category_confidence = 0.9
            result.confidence = 0.9
            result.new_category = "05_Travel & Adventure/wakeboarding"
            result.new_category_note = "Towed water sports."
            return result

    for name, seconds in (("wake_a.mp4", 5), ("wake_b.mp4", 6)):  # different lengths: different files
        await pipeline_with(workspace, store, Wakeboard()).ingest(make_clip(name, seconds=seconds))
    [proposal] = store.list_folder_proposals()
    assert proposal["path"] == "05_Travel & Adventure/Wakeboarding" and len(proposal["shot_ids"]) == 2
    assert "05_Travel & Adventure/Wakeboarding" not in dict(workspace.taxonomy.category_leaves())

    done = approve_folder_proposal(workspace, store, proposal["id"])
    assert len(done["moved"]) == 2 and done["created"]
    assert all(store.get_shot(i).category == "05_Travel & Adventure/Wakeboarding" for i in done["moved"])
    assert "05_Travel & Adventure/Wakeboarding" in dict(workspace.taxonomy.category_leaves())
    assert store.list_folder_proposals() == []
    assert not dismiss_folder_proposal(store, proposal["id"]), "already closed"
