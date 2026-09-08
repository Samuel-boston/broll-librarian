"""Taxonomy is pure, and it decides what the client's Drive looks like.

Table-driven, because the interesting behaviour is all in the thresholds.
"""

from __future__ import annotations

import pytest

from broll.config import TaxonomyConfig
from broll.db.models import ShotFacets
from broll.drive.taxonomy import (
    FolderPath,
    folder_label,
    library_folder,
    plan_filename,
    plan_paths,
    plan_shortcuts,
    primary_shot,
    review_folder,
    shortcut_name,
)


def facets(**overrides) -> ShotFacets:
    base = dict(
        shot_id="shot-1",
        source_id="src-1",
        is_primary=True,
        start_s=0.0,
        subjects=["person"],
        action="meditating",
        setting="beach",
        mood=["calm"],
        usable_for=["establishing shot"],
        shot_type="wide",
        camera_movement="static",
        time_of_day="golden_hour",
        colour_profile="warm",
    )
    base.update(overrides)
    return ShotFacets(**base)


def counts(**pairs) -> dict[tuple[str, str], int]:
    """counts(setting__beach=9) -> {('setting', 'beach'): 9}"""
    return {tuple(k.split("__", 1)): v for k, v in pairs.items()}


def paths(result) -> set[str]:
    return {str(p) for p in result}


# -- labels and filenames --------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("golden_hour", "Golden Hour"),
        ("open-plan office", "Open-Plan Office"),
        ("b-roll under narration", "B-Roll Under Narration"),
        ("wide", "Wide"),
        ("drone_fly_over", "Drone Fly Over"),
        ("person's hands", "Person's Hands"),
    ],
)
def test_folder_label(value, expected):
    assert folder_label(value) == expected


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, "beach_meditating_golden_hour_wide_a1b2c3d4.mp4"),
        ({"time_of_day": "unknown"}, "beach_meditating_wide_a1b2c3d4.mp4"),
        ({"action": None}, "beach_golden_hour_wide_a1b2c3d4.mp4"),
        ({"setting": "open-plan office"}, "open_plan_office_meditating_golden_hour_wide_a1b2c3d4.mp4"),
        ({"setting": "café"}, "cafe_meditating_golden_hour_wide_a1b2c3d4.mp4"),
        (
            {"setting": None, "action": None, "time_of_day": None, "shot_type": None},
            "clip_a1b2c3d4.mp4",
        ),
    ],
)
def test_plan_filename(kwargs, expected):
    assert plan_filename(facets(**kwargs), "a1b2c3d4e5f6", ".mp4") == expected


def test_filename_is_truncated_but_keeps_the_hash_and_extension():
    long = facets(setting="a" * 200, action="b" * 200)
    name = plan_filename(long, "a1b2c3d4", ".mov", TaxonomyConfig(filename_max_length=40))
    assert len(name) <= 40
    assert name.endswith("_a1b2c3d4.mov")
    assert not name.startswith("_") and "__" not in name


def test_filenames_are_unique_per_content_hash():
    a = plan_filename(facets(), "aaaaaaaa11", ".mp4")
    b = plan_filename(facets(), "bbbbbbbb22", ".mp4")
    assert a != b and a[:-13] == b[:-13]


def test_shortcut_name_carries_the_timecode_for_non_primary_shots():
    name = "beach_meditating_golden_hour_wide_a1b2c3d4.mp4"
    assert shortcut_name(name, facets(is_primary=True)) == name
    assert shortcut_name(name, facets(is_primary=False, start_s=42.4)) == (
        "beach_meditating_golden_hour_wide_a1b2c3d4_at_0m42s.mp4"
    )
    assert shortcut_name(name, facets(is_primary=False, start_s=125.0)).endswith("_at_2m05s.mp4")


# -- folder planning -------------------------------------------------------


def test_a_clip_appears_under_every_facet_it_has():
    result = paths(plan_paths([facets()], counts()))
    assert result == {
        "By Subject/Person",
        "By Action/Meditating",
        "By Setting/Beach",
        "By Mood/Calm",
        "By Shot Type/Wide",
        "By Camera Movement/Static",
        "By Time of Day/Golden Hour",
        "By Colour/Warm",
        "By Use/Establishing Shot",
    }


def test_uninformative_values_get_no_folder():
    result = paths(plan_paths([facets(time_of_day="unknown", action="none")], counts()))
    assert not any(p.startswith("By Time of Day") for p in result)
    assert not any(p.startswith("By Action") for p in result)


def test_third_level_only_when_enough_clips_share_the_combination():
    config = TaxonomyConfig(min_clips_for_subfolder=5)
    shot = facets()

    below = plan_paths([shot], counts(), config, pair_counts={("action", "meditating", "beach"): 4})
    assert "By Action/Meditating" in paths(below)
    assert "By Action/Meditating/Beach" not in paths(below)

    at_threshold = plan_paths(
        [shot], counts(), config, pair_counts={("action", "meditating", "beach"): 5}
    )
    assert "By Action/Meditating/Beach" in paths(at_threshold)
    assert "By Action/Meditating" not in paths(at_threshold)


def test_third_level_picks_the_most_common_co_occurring_value():
    config = TaxonomyConfig(min_clips_for_subfolder=3)
    shot = facets(subjects=["person"], setting="beach")
    plans = plan_paths(
        [shot], counts(), config,
        pair_counts={
            ("setting", "beach", "golden_hour"): 9,
            ("subjects", "person", "beach"): 4,
        },
    )
    assert "By Setting/Beach/Golden Hour" in paths(plans)
    assert "By Subject/Person/Beach" in paths(plans)


def test_promotion_happens_as_the_library_grows():
    """Below the threshold a clip sits at level two; later it is promoted."""
    config = TaxonomyConfig(min_clips_for_subfolder=5)
    shot = facets()
    small = paths(plan_paths([shot], counts(), config, {("setting", "beach", "golden_hour"): 2}))
    grown = paths(plan_paths([shot], counts(), config, {("setting", "beach", "golden_hour"): 12}))
    assert "By Setting/Beach" in small and "By Setting/Beach/Golden Hour" not in small
    assert "By Setting/Beach/Golden Hour" in grown


def test_long_tail_is_grouped_under_other():
    config = TaxonomyConfig(max_folders_per_level=2)
    # Three settings exist; only the two most common get their own folder.
    facet_counts = {
        ("setting", "beach"): 30,
        ("setting", "office"): 20,
        ("setting", "cave"): 1,
    }
    common = paths(plan_paths([facets(setting="beach")], facet_counts, config))
    rare = paths(plan_paths([facets(setting="cave")], facet_counts, config))
    assert "By Setting/Beach" in common
    assert "By Setting/Other" in rare
    assert "By Setting/Cave" not in rare


def test_needs_review_shots_are_routed_for_triage():
    assert "_Needs Review" in paths(plan_paths([facets(needs_review=True)], counts()))
    assert "_Needs Review" not in paths(plan_paths([facets(needs_review=False)], counts()))


def test_library_and_review_folders():
    assert str(library_folder("2026-09")) == "_Library/2026-09"
    assert str(review_folder()) == "_Needs Review"
    assert str(FolderPath(("a", "b"))) == "a/b"


# -- multi-shot union ------------------------------------------------------


def test_multi_shot_source_gets_the_union_of_its_shots_facets():
    beach = facets(shot_id="s0", is_primary=True, setting="beach", action="meditating")
    office = facets(
        shot_id="s1", is_primary=False, start_s=42.0, setting="office",
        action="typing", subjects=["office worker"], mood=["professional"],
        time_of_day="indoor_artificial", colour_profile="neutral",
    )
    result = paths(plan_paths([beach, office], counts()))
    assert {"By Setting/Beach", "By Setting/Office"} <= result
    assert {"By Action/Meditating", "By Action/Typing"} <= result


def test_shortcuts_name_each_shot_and_are_deduped():
    beach = facets(shot_id="s0", is_primary=True, setting="beach")
    office = facets(
        shot_id="s1", is_primary=False, start_s=42.0, setting="office",
        action="typing", subjects=["office worker"], mood=["professional"],
    )
    filename = "beach_meditating_golden_hour_wide_a1b2c3d4.mp4"
    plans = plan_shortcuts([beach, office], filename, counts())
    by_path = {p.path for p in plans}

    assert f"By Setting/Beach/{filename}" in by_path
    assert (
        "By Setting/Office/beach_meditating_golden_hour_wide_a1b2c3d4_at_0m42s.mp4"
        in by_path
    )
    assert len(by_path) == len(plans), "the same shortcut must not be planned twice"


def test_shared_facets_across_shots_produce_one_shortcut_per_shot_not_per_facet():
    """Both shots are 'wide', so By Shot Type/Wide holds two named shortcuts."""
    a = facets(shot_id="s0", is_primary=True)
    b = facets(shot_id="s1", is_primary=False, start_s=10.0)
    plans = plan_shortcuts([a, b], "clip_a1b2c3d4.mp4", counts())
    wide = [p for p in plans if str(p.folder) == "By Shot Type/Wide"]
    assert {p.name for p in wide} == {
        "clip_a1b2c3d4.mp4",
        "clip_a1b2c3d4_at_0m10s.mp4",
    }


def test_primary_shot_selection():
    a = facets(shot_id="s0", is_primary=False)
    b = facets(shot_id="s1", is_primary=True)
    assert primary_shot([a, b]).shot_id == "s1"
    assert primary_shot([a]).shot_id == "s0"
    assert primary_shot([]) is None
