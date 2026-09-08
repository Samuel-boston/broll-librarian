"""The analysis contract. If this drifts, every downstream feature degrades."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from broll.analysis.schema import (
    AnalysisResult,
    CameraMove,
    ColourProfile,
    Pace,
    PeopleCount,
    ShotContext,
    ShotType,
    TimeOfDay,
    find_oov,
    in_vocabulary,
    normalise_term,
)

BASE = {
    "caption": "A lone figure meditates on an empty beach as the sun comes up.",
    "subjects": ["person", "beach", "ocean"],
    "action": "meditating",
    "setting": "beach",
    "setting_detail": "wide empty sand with low surf",
    "shot_type": "wide",
    "camera_movement": "static",
    "time_of_day": "dawn",
    "mood": ["calm", "peaceful", "serene"],
    "colour_profile": "warm",
    "people_count": "one",
    "has_recognisable_faces": False,
    "has_text_on_screen": False,
    "pace": "slow",
    "tags": ["meditation", "mindfulness", "wellness", "calm", "sunrise"],
    "usable_for": ["establishing shot", "mood setter"],
    "quality_flags": [],
    "confidence": 0.86,
}


def test_valid_result_is_in_vocabulary():
    result = AnalysisResult.model_validate(BASE)
    assert in_vocabulary(result)
    assert result.shot_type is ShotType.wide
    assert result.people_count is PeopleCount.one


@pytest.mark.parametrize(
    "field,given,expected",
    [
        ("shot_type", "Wide Shot", ShotType.wide),
        ("shot_type", "extreme close-up", ShotType.extreme_close_up),
        ("shot_type", "drone", ShotType.aerial),
        ("camera_movement", "slow push in", CameraMove.push_in),
        ("camera_movement", "locked off", CameraMove.static),
        ("camera_movement", "hand-held", CameraMove.handheld),
        ("time_of_day", "sunset", TimeOfDay.golden_hour),
        ("time_of_day", "indoors", TimeOfDay.indoor_artificial),
        ("colour_profile", "black and white", ColourProfile.monochrome),
        ("colour_profile", "moody", ColourProfile.dark_moody),
        ("people_count", "couple", PeopleCount.two),
        ("people_count", 1, PeopleCount.one),
        ("pace", "normal", Pace.moderate),
    ],
)
def test_enum_aliases_are_absorbed(field, given, expected):
    result = AnalysisResult.model_validate({**BASE, field: given})
    assert getattr(result, field) is expected


def test_unmappable_enum_is_a_validation_error():
    """An unmappable enum must fail so the analyzer retries with the error."""
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate({**BASE, "shot_type": "banana"})


def test_mood_capped_at_three_and_tags_at_fifteen():
    result = AnalysisResult.model_validate(
        {**BASE, "mood": ["calm", "peaceful", "serene", "quiet", "still"],
         "tags": [f"tag{i}" for i in range(30)]}
    )
    assert len(result.mood) == 3
    assert len(result.tags) == 15


def test_lists_are_normalised_and_deduped():
    result = AnalysisResult.model_validate(
        {**BASE, "subjects": ["  Person ", "person", "OCEAN."]}
    )
    assert result.subjects == ["person", "ocean"]


def test_confidence_is_clamped():
    assert AnalysisResult.model_validate({**BASE, "confidence": 4.2}).confidence == 1.0
    assert AnalysisResult.model_validate({**BASE, "confidence": -1}).confidence == 0.0
    assert AnalysisResult.model_validate({**BASE, "confidence": "nonsense"}).confidence == 0.5


def test_quality_flags_are_snake_cased():
    result = AnalysisResult.model_validate({**BASE, "quality_flags": ["Out Of Focus", "shaky"]})
    assert result.quality_flags == ["out_of_focus", "shaky"]


def test_out_of_vocabulary_terms_are_kept_and_reported():
    result = AnalysisResult.model_validate(
        {**BASE, "subjects": ["person", "hydrofoil"], "action": "foiling"}
    )
    assert "hydrofoil" in result.subjects  # kept on the row
    oov = dict((term, field) for field, term in find_oov(result))
    assert oov["hydrofoil"] == "subjects"
    assert oov["foiling"] == "action"


def test_vocabulary_overrides_suppress_candidates():
    result = AnalysisResult.model_validate({**BASE, "subjects": ["person", "hydrofoil"]})
    assert find_oov(result, {"subjects": ["hydrofoil"]}) == []


def test_open_tags_are_never_out_of_vocabulary():
    result = AnalysisResult.model_validate({**BASE, "tags": ["hydrofoiling", "kitesurf"]})
    assert find_oov(result) == []


def test_embedding_text_leads_with_the_caption():
    result = AnalysisResult.model_validate(BASE)
    text = result.embedding_text()
    assert text.startswith(BASE["caption"])
    for term in ("meditating", "beach", "mindfulness", "wide", "slow pace"):
        assert term in text


def test_normalise_term():
    assert normalise_term("  Golden  Hour. ") == "golden hour"


def test_shot_context_describes_position():
    single = ShotContext(source_filename="a.mp4", duration_s=4.0, width=1920, height=1080)
    assert "single continuous shot" in single.describe()
    multi = ShotContext(
        source_filename="a.mp4", duration_s=4.0, width=1920, height=1080,
        shot_index=1, shot_count=3, start_s=12.5,
    )
    assert "shot 2 of 3" in multi.describe()
    assert "12.5s" in multi.describe()
