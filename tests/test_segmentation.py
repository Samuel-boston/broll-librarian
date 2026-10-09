"""Finding the shot inside a raw clip: skipping setup, splitting scenes, naming the best part."""

from __future__ import annotations

import subprocess

import pytest

from broll.analysis.providers.mock import MockVisionProvider
from broll.analysis.segmentation import (
    ModelSegment,
    Segment,
    SegmentKind,
    build_segment_prompt,
    SegmentContext,
    combine_windows,
    normalise_segments,
    plan_windows,
)
from broll.ingest.frames import frame_count_for
from broll.ingest.segments import number_usable, plan_segments, primary_segment_index


def seg(a, b, kind="usable", **kw):
    return ModelSegment(start_s=a, end_s=b, kind=SegmentKind(kind), **kw)


# -- cleaning up what the model says -------------------------------------------------


def test_segments_tile_the_stretch_exactly():
    out = normalise_segments([seg(2, 10, "setup"), seg(10, 30), seg(30, 40, "dead")], 0, 40)
    assert [(s.start_s, s.end_s, s.kind) for s in out] == [
        (0, 10, "setup"), (10, 30, "usable"), (30, 40, "dead"),
    ]


def test_overlaps_and_gaps_are_settled_at_the_midpoint():
    out = normalise_segments([seg(0, 12, "setup"), seg(8, 28), seg(32, 40, "dead")], 0, 40)
    assert [round(s.start_s, 1) for s in out] == [0.0, 10.0, 30.0]
    assert [round(s.end_s, 1) for s in out] == [10.0, 30.0, 40.0]


def test_times_outside_the_file_are_clamped_not_trusted():
    out = normalise_segments([seg(-5, 20), seg(20, 400)], 0, 45)
    assert out[0].start_s == 0 and out[-1].end_s == 45


def test_a_flicker_never_becomes_a_one_second_shot():
    out = normalise_segments([seg(0, 20), seg(20, 20.8, "dead"), seg(20.8, 40)], 0, 40)
    assert all(s.duration_s >= 1.5 for s in out)
    assert sum(s.duration_s for s in out) == pytest.approx(40)


def test_the_best_part_must_sit_inside_its_segment():
    out = normalise_segments([seg(5, 30, best_start_s=2, best_end_s=60)], 5, 30)
    assert (out[0].best_start_s, out[0].best_end_s) == (5, 30)


def test_a_best_part_under_a_second_is_ignored():
    out = normalise_segments([seg(0, 20, best_start_s=5, best_end_s=5.4)], 0, 20)
    assert out[0].best_start_s is None


def test_a_setup_segment_has_no_best_part():
    out = normalise_segments([seg(0, 6, "setup", best_start_s=1, best_end_s=4), seg(6, 20)], 0, 20)
    assert out[0].best_start_s is None


def test_nothing_usable_keeps_the_whole_file_flagged_not_lost():
    out = normalise_segments([seg(0, 20, "setup")], 0, 20)
    assert len(out) == 1 and out[0].kind == "usable" and out[0].unsure
    assert normalise_segments([], 0, 20)[0].unsure


def test_too_many_usable_segments_keeps_the_longest():
    proposed = [seg(i * 5, i * 5 + 5) for i in range(6)] + [seg(30, 60)]
    out = normalise_segments(proposed, 0, 60, max_usable=3)
    usable = [s for s in out if s.kind == "usable"]
    assert len(usable) == 3
    assert max(s.duration_s for s in usable) == 30  # the long one survived
    assert sum(1 for s in out if s.kind == "skipped") == 4


def test_segments_round_trip_through_json():
    original = Segment(1.0, 9.5, "usable", "a man walks", best_start_s=2.0, best_end_s=6.0, index=2)
    assert Segment.from_dict(original.to_dict()) == original


def test_only_usable_segments_become_shots_numbered_in_time_order():
    segments = [Segment(0, 5, "setup"), Segment(5, 20, "usable"), Segment(20, 25, "dead"), Segment(25, 60, "usable")]
    usable = number_usable(segments)
    assert [(s.index, s.start_s) for s in usable] == [(0, 5), (1, 25)]
    assert primary_segment_index(usable) == 1  # the longer one names the file


# -- how a stretch is looked at -------------------------------------------------------


def test_a_short_clip_is_one_call_a_long_one_is_read_a_minute_at_a_time():
    assert len(plan_windows(0, 20)) == 1
    windows = plan_windows(0, 300)
    assert len(windows) == 5 and windows[0].start_s == 0 and windows[-1].end_s == 300
    assert all(w.frames <= 12 for w in windows)


def test_a_sliver_at_the_end_does_not_cost_a_call():
    windows = plan_windows(0, 65)
    assert len(windows) == 1 or windows[-1].end_s - windows[-1].start_s >= 15


def test_frame_count_grows_with_the_length_of_the_shot():
    assert frame_count_for(3) == 3
    assert frame_count_for(20) == 5
    assert frame_count_for(400) == 8


def test_the_prompt_says_which_part_of_the_file_is_in_view():
    context = SegmentContext(source_filename="a.mp4", file_duration_s=300, window_start_s=60, window_end_s=120)
    prompt = build_segment_prompt(context, [65.0, 75.0])
    assert "60.0s to 120.0s" in prompt and "not the start" in prompt and "not the end" in prompt


def test_combining_windows_settles_the_seams():
    left = normalise_segments([seg(0, 60)], 0, 60)
    right = normalise_segments([seg(60, 120)], 60, 120)
    out = combine_windows([left, right])
    assert out[0].start_s == 0 and out[-1].end_s == 120


# -- end to end against real frames ----------------------------------------------------


@pytest.fixture()
def long_clip(tmp_path):
    """A 24-second test pattern, enough to be worth looking inside."""
    def make(name: str):
        path = tmp_path / name
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
             "testsrc2=size=320x240:rate=15:duration=24", "-pix_fmt", "yuv420p", "-y", str(path)],
            check=True,
        )
        return path
    return make


async def _plan(workspace, path, tmp_path):
    return await plan_segments(
        MockVisionProvider(), path, filename=path.name, start_s=0, end_s=24, file_duration_s=24,
        width=320, height=240, config=workspace, work_dir=tmp_path / "work",
    )


async def test_the_setup_at_the_start_is_left_out(workspace, long_clip, tmp_path):
    plan = await _plan(workspace, long_clip("beach_setup_01.mp4"), tmp_path)
    kinds = [(s.kind, round(s.start_s, 1), round(s.end_s, 1)) for s in plan.segments]
    assert kinds[0][0] == "setup" and kinds[1][0] == "usable"
    usable = number_usable(plan.segments)
    assert len(usable) == 1 and usable[0].start_s > 4  # the shot starts after the setup


async def test_two_scenes_in_one_take_become_two_shots(workspace, long_clip, tmp_path):
    plan = await _plan(workspace, long_clip("walk_scenes_01.mp4"), tmp_path)
    usable = number_usable(plan.segments)
    assert len(usable) == 2
    assert usable[0].end_s == pytest.approx(usable[1].start_s)
    assert all(s.best_start_s is not None for s in usable)


async def test_a_plain_clip_is_one_usable_segment_with_a_best_part(workspace, long_clip, tmp_path):
    plan = await _plan(workspace, long_clip("plain_01.mp4"), tmp_path)
    usable = number_usable(plan.segments)
    assert len(usable) == 1 and (usable[0].start_s, usable[0].end_s) == (0, 24)
    assert 0 < usable[0].best_start_s < usable[0].best_end_s < 24
    assert not list((tmp_path / "work").glob("*.jpg")), "frames used for looking are cleaned up"


async def test_a_provider_that_cannot_look_falls_back_to_the_whole_file(workspace, long_clip, tmp_path):
    from broll.analysis.providers.base import ProviderError

    class Blind(MockVisionProvider):
        async def segment(self, *a, **k):
            raise ProviderError("no support")

    path = long_clip("plain_02.mp4")
    plan = await plan_segments(
        Blind(), path, filename=path.name, start_s=0, end_s=24, file_duration_s=24, width=320,
        height=240, config=workspace, work_dir=tmp_path / "work",
    )
    assert [(s.kind, s.start_s, s.end_s) for s in plan.segments] == [("usable", 0, 24)]
    assert plan.fallback_reason == "no support"


async def test_a_transient_error_is_raised_so_the_queue_retries(workspace, long_clip, tmp_path):
    from broll.analysis.providers.base import TransientProviderError

    class Busy(MockVisionProvider):
        async def segment(self, *a, **k):
            raise TransientProviderError("503")

    path = long_clip("plain_03.mp4")
    with pytest.raises(TransientProviderError):
        await plan_segments(
            Busy(), path, filename=path.name, start_s=0, end_s=24, file_duration_s=24, width=320,
            height=240, config=workspace, work_dir=tmp_path / "work",
        )
