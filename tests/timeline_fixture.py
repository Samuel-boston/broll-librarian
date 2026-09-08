"""A deterministic set of matched beats, shared by the exporter golden tests."""

from __future__ import annotations

from broll.config import WorkspaceConfig
from broll.db.models import Shot, Source
from broll.transcript.matcher import BeatMatch, Suggestion
from broll.transcript.parser import Beat


def _source(source_id: str, filename: str, fps: float, drive_path: str) -> Source:
    return Source(
        id=source_id,
        workspace_id="test",
        content_hash=source_id,
        original_filename=filename,
        origin="local",
        origin_path=f"/footage/{filename}",
        drive_file_id=f"drive-{source_id}",
        drive_web_link=f"https://drive.google.com/file/d/{source_id}/view",
        drive_path=drive_path,
        duration_s=30.0,
        width=1920,
        height=1080,
        fps=fps,
        codec="h264",
        filesize_bytes=1024,
        status="indexed",
    )


def _shot(shot_id: str, source_id: str, start_s: float, duration_s: float, **facets) -> Shot:
    return Shot(
        id=shot_id,
        workspace_id="test",
        source_id=source_id,
        shot_index=0,
        is_primary=True,
        start_s=start_s,
        end_s=start_s + duration_s,
        duration_s=duration_s,
        caption=facets.pop("caption", "A clip."),
        status="indexed",
        **facets,
    )


def build_matches() -> list[BeatMatch]:
    """Three placed beats and one gap.

    Deliberately awkward: clip 2 was shot at 29.97 while the rest are 25, and
    clip 3 is shorter than the beat it has to cover.
    """
    beach_source = _source("src-beach", "beach_meditation.mp4", 25.0,
                           "_Library/2026-09/beach_meditating_dawn_wide_aaaaaaaa.mp4")
    office_source = _source("src-office", "office_meeting.mov", 29.97,
                            "_Library/2026-09/office_brainstorming_indoor_medium_bbbbbbbb.mov")
    coffee_source = _source("src-coffee", "coffee pour.mp4", 25.0,
                            "_Library/2026-09/cafe_pouring_indoor_close_up_cccccccc.mp4")

    beach_shot = _shot("src-beach-0", "src-beach", start_s=2.0, duration_s=12.0,
                       caption="A lone figure meditates on an empty beach at first light.",
                       shot_type="wide", camera_movement="static", setting="beach",
                       time_of_day="dawn", mood=["calm", "serene"], pace="slow")
    office_shot = _shot("src-office-0", "src-office", start_s=0.0, duration_s=9.0,
                        caption="Four colleagues cluster around a whiteboard.",
                        shot_type="medium", camera_movement="handheld", setting="open-plan office",
                        time_of_day="indoor_artificial", mood=["professional"], pace="moderate")
    coffee_shot = _shot("src-coffee-0", "src-coffee", start_s=1.5, duration_s=3.0,
                        caption="Hands tip a kettle over a filter, steam curling off.",
                        shot_type="extreme_close_up", camera_movement="static", setting="cafe",
                        time_of_day="indoor_artificial", mood=["cosy"], pace="slow")

    return [
        BeatMatch(
            beat=Beat(index=0, start_s=0.0, end_s=6.0,
                      text="Most of us start the day already behind.", words=8),
            suggestions=[
                Suggestion(shot=beach_shot, source=beach_source,
                           reason="Calm and slow: it sets the tone before the problem lands.",
                           confidence=0.88, score=1.0),
            ],
        ),
        BeatMatch(
            beat=Beat(index=1, start_s=6.0, end_s=14.0,
                      text="So we built something that keeps up with your team.", words=10),
            suggestions=[
                Suggestion(shot=office_shot, source=office_source,
                           reason="Shows the team working, without a literal product shot.",
                           confidence=0.81, score=1.0),
            ],
        ),
        BeatMatch(
            beat=Beat(index=2, start_s=14.0, end_s=22.0,
                      text="It starts before your first coffee and it never gets in the way.",
                      words=13),
            suggestions=[
                Suggestion(shot=coffee_shot, source=coffee_source,
                           reason="A small human detail to land the line.",
                           confidence=0.74, score=1.0),
            ],
        ),
        BeatMatch(
            beat=Beat(index=3, start_s=22.0, end_s=28.0,
                      text="And when the contract is signed, everyone already knows.", words=9),
            suggestions=[],
            no_good_match=True,
            missing_footage="A close-up of a hand signing a contract.",
        ),
    ]


def build_config(mount: str | None = "/Users/editor/Google Drive/My Drive") -> WorkspaceConfig:
    config = WorkspaceConfig(id="test", name="Golden")
    config.drive_local_mount_path = mount
    return config
