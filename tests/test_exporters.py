"""Golden-file tests for the NLE exports.

Regenerate deliberately (and read the diff) with:
    BROLL_REGENERATE_GOLDEN=1 pytest tests/test_exporters.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from broll.transcript.exporters import csv_export, edl, fcp7xml
from broll.transcript.exporters.base import (
    build_timeline,
    frames_to_timecode,
    ntsc_timebase,
    seconds_to_frames,
)
from tests.timeline_fixture import build_config, build_matches

GOLDEN = Path(__file__).parent / "fixtures" / "golden"
REGENERATE = os.environ.get("BROLL_REGENERATE_GOLDEN") == "1"


def assert_golden(name: str, produced: str) -> None:
    path = GOLDEN / name
    if REGENERATE or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(produced, encoding="utf-8")
        if not REGENERATE:
            pytest.fail(f"golden {name} was missing; it has been written - review it")
        return
    assert produced == path.read_text(encoding="utf-8"), f"{name} drifted from its golden file"


@pytest.fixture()
def timeline():
    return build_timeline(build_matches(), build_config(), name="Golden Timeline")


# -- frame maths -----------------------------------------------------------


@pytest.mark.parametrize(
    "fps,expected",
    [(25.0, (25, False)), (24.0, (24, False)), (23.976, (24, True)),
     (29.97, (30, True)), (30.0, (30, False)), (59.94, (60, True))],
)
def test_ntsc_timebase(fps, expected):
    assert ntsc_timebase(fps) == expected


@pytest.mark.parametrize(
    "frames,fps,expected",
    [(0, 25, "00:00:00:00"), (25, 25, "00:00:01:00"), (38, 25, "00:00:01:13"),
     (1500, 25, "00:01:00:00"), (90000, 25, "01:00:00:00")],
)
def test_frames_to_timecode(frames, fps, expected):
    assert frames_to_timecode(frames, fps) == expected


def test_seconds_to_frames_rounds():
    assert seconds_to_frames(1.5, 25) == 38
    assert seconds_to_frames(8.0, 29.97) == 240


# -- timeline construction -------------------------------------------------


def test_sequence_fps_is_the_modal_rate_and_mixed_rates_are_flagged(timeline):
    assert timeline.fps == 25.0
    assert any("Mixed frame rates" in w for w in timeline.warnings)


def test_config_fps_overrides_the_modal_rate():
    config = build_config()
    config.transcript.sequence_fps = 24.0
    timeline = build_timeline(build_matches(), config)
    assert timeline.fps == 24.0


def test_clips_start_at_their_beat_and_are_trimmed_to_it(timeline):
    first, second = timeline.items[0], timeline.items[1]
    assert first.start_frame == 0 and first.end_frame == 150      # 0-6s at 25fps
    assert second.start_frame == 150 and second.end_frame == 350  # 6-14s
    assert first.source_in_frame == 50                            # shot starts at 2.0s


def test_a_short_clip_leaves_a_gap_rather_than_stretching(timeline):
    third = timeline.items[2]
    assert third.gap_frames == 125          # 8s beat, 3s clip, 5s short
    assert third.duration_frames == 75      # and the clip is not stretched
    assert any("left as a gap" in w for w in timeline.warnings)


def test_source_timecodes_use_the_source_timebase(timeline):
    ntsc_item = timeline.items[1]
    assert ntsc_item.source_fps == 29.97
    # 8 seconds of sequence time is 240 frames at 29.97, not 200.
    assert ntsc_item.source_out_frame - ntsc_item.source_in_frame == 240


def test_beats_with_no_good_match_become_gaps_not_clips(timeline):
    assert len(timeline.items) == 3
    assert len(timeline.gaps) == 1
    assert timeline.gaps[0].missing_footage == "A close-up of a hand signing a contract."


def test_media_resolves_through_the_drive_mount(timeline):
    assert timeline.items[0].media_path.startswith("/Users/editor/Google Drive/My Drive/")
    assert not timeline.items[0].offline


def test_without_a_mount_the_export_still_happens_but_warns_loudly():
    config = build_config(mount=None)
    timeline = build_timeline(build_matches(), config)
    assert timeline.items, "the export must still be produced"
    assert any("drive_local_mount_path is not set" in w for w in timeline.warnings)
    assert any("import offline" in w for w in timeline.warnings)
    assert all(item.offline for item in timeline.items)


# -- golden files ----------------------------------------------------------


def test_fcp7_xml_golden(timeline):
    assert_golden("timeline.xml", fcp7xml.build(timeline))


def test_edl_golden(timeline):
    assert_golden("timeline.edl", edl.build(timeline))


def test_csv_golden(timeline):
    assert_golden("timeline.csv", csv_export.build(build_matches(), timeline))


def test_fcp7_xml_is_well_formed_and_shaped_right(timeline):
    from xml.etree import ElementTree as ET

    root = ET.fromstring(fcp7xml.build(timeline))
    assert root.tag == "xmeml" and root.get("version") == "5"
    clips = root.findall(".//track/clipitem")
    assert len(clips) == 3
    assert [c.findtext("start") for c in clips] == ["0", "150", "350"]

    # The 29.97 source keeps its own rate on the file element.
    file_rates = [(f.findtext("rate/timebase"), f.findtext("rate/ntsc"))
                  for f in root.findall(".//file") if f.find("rate") is not None]
    assert ("30", "TRUE") in file_rates
    assert all(f.findtext("pathurl", "").startswith("file://localhost/")
               for f in root.findall(".//file") if f.find("pathurl") is not None)


def test_edl_is_cmx3600_shaped(timeline):
    lines = edl.build(timeline).splitlines()
    assert lines[0] == "TITLE: Golden Timeline"
    assert lines[1] == "FCM: NON-DROP FRAME"
    events = [line for line in lines if line[:3].isdigit()]
    assert len(events) == 3
    for event in events:
        assert "V     C" in event
        assert len(event.split()[-1].split(":")) == 4


def test_files_are_written_to_disk(timeline, tmp_path):
    xml_path = fcp7xml.write(timeline, tmp_path / "out" / "seq.xml")
    edl_path = edl.write(timeline, tmp_path / "out" / "seq.edl")
    csv_path = csv_export.write(build_matches(), timeline, tmp_path / "out" / "seq.csv")
    for path in (xml_path, edl_path, csv_path):
        assert path.exists() and path.stat().st_size > 0
