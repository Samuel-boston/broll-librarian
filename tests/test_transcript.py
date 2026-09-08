"""Transcript parsing, beat segmentation and matching."""

from __future__ import annotations

import pytest

from broll.config import WorkspaceConfig
from broll.search.filters import SearchFilters
from broll.search.query import SearchEngine
from broll.transcript.matcher import (
    RerankChoice,
    RerankResult,
    TranscriptMatcher,
    gaps,
)
from broll.transcript.parser import (
    TranscriptError,
    parse,
    parse_and_segment,
    parse_file,
    parse_plain_text,
    parse_srt,
    parse_vtt,
    segment,
)

SRT = """1
00:00:00,000 --> 00:00:06,500
Most of us start the day already behind.

2
00:00:06,500 --> 00:00:14,000
So we built something different. It gets out of the way.

3
00:00:14,000 --> 00:00:21,000
<i>And it keeps up with you, wherever the work happens.</i>
"""

VTT = """WEBVTT

NOTE recorded 2026-09-01

00:00:00.000 --> 00:00:05.000
NARRATOR: The morning starts quietly.

00:00:05.000 --> 00:00:12.000
Then the day arrives all at once.
"""


def test_parse_srt_timecodes_and_text():
    cues = parse_srt(SRT)
    assert len(cues) == 3
    assert cues[0].start_s == 0.0 and cues[0].end_s == 6.5
    assert cues[2].text == "And it keeps up with you, wherever the work happens."


def test_parse_vtt_strips_headers_speakers_and_notes():
    cues = parse_vtt(VTT)
    assert len(cues) == 2
    assert cues[0].text == "The morning starts quietly."
    assert cues[1].start_s == 5.0


def test_parse_detects_format_without_a_filename():
    assert parse(SRT)[1] is False
    assert parse(VTT)[1] is False
    assert parse("Just some prose about the product.")[1] is True


def test_plain_text_timings_follow_words_per_minute():
    text = " ".join(["word"] * 150) + "."
    cues = parse_plain_text(text, words_per_minute=150)
    assert len(cues) == 1
    assert 59.0 < cues[0].end_s < 61.0  # 150 words at 150 wpm is a minute


def test_parse_file_missing(tmp_path):
    with pytest.raises(TranscriptError):
        parse_file(tmp_path / "nope.srt")


def test_beats_land_inside_the_target_window():
    beats = parse_and_segment(SRT, "a.srt", beat_min_s=3.0, beat_max_s=15.0)
    assert beats
    assert all(3.0 <= b.duration_s <= 15.0 for b in beats), [b.duration_s for b in beats]
    assert beats[0].start_s == 0.0
    assert all(b.end_s > b.start_s for b in beats)


def test_long_cues_are_divided():
    from broll.transcript.parser import Cue

    beats = segment([Cue(0.0, 60.0, " ".join(["word"] * 200))], beat_max_s=15.0)
    assert len(beats) >= 4
    assert all(b.duration_s <= 15.5 for b in beats)


def test_short_cues_are_merged():
    from broll.transcript.parser import Cue

    cues = [Cue(0.0, 1.0, "One."), Cue(1.0, 2.0, "Two."), Cue(2.0, 3.5, "Three.")]
    beats = segment(cues, beat_min_s=3.0, beat_max_s=15.0)
    assert len(beats) == 1
    assert beats[0].text == "One. Two. Three."


def test_beats_are_contiguous_and_ordered():
    beats = parse_and_segment(SRT, "a.srt")
    for earlier, later in zip(beats, beats[1:]):
        assert later.start_s >= earlier.end_s - 0.001
        assert later.index == earlier.index + 1


# -- matching --------------------------------------------------------------


class StubTextProvider:
    """Replays a scripted rerank, so matching is tested without a live model."""

    name = "stub"

    def __init__(self, results: list[RerankResult]):
        self.results = list(results)
        self.prompts: list[str] = []

    async def complete(self, prompt: str, schema):
        self.prompts.append(prompt)
        return self.results.pop(0) if self.results else RerankResult(choices=[])

    def estimate_cost(self, prompt: str) -> float:
        return 0.0


@pytest.fixture()
def matcher_setup(store, workspace):
    from tests.test_search import seed_library

    seed_library(store)
    return workspace, SearchEngine(store, embedder=None)


async def test_matcher_falls_back_to_search_order_without_a_text_provider(matcher_setup):
    workspace, engine = matcher_setup
    beats = parse_and_segment("Someone pours a coffee slowly.", "a.txt")
    matches = await TranscriptMatcher(workspace, engine).match(beats)
    assert matches[0].suggestions
    assert matches[0].suggestions[0].reason == "Ranked by hybrid search."


async def test_reranker_choices_are_honoured(matcher_setup):
    workspace, engine = matcher_setup
    provider = StubTextProvider([
        RerankResult(choices=[RerankChoice(candidate=2, reason="Calmer than the literal shot.",
                                           confidence=0.9)])
    ])
    beats = parse_and_segment("Slowing down at the end of the day.", "a.txt")
    matches = await TranscriptMatcher(workspace, engine, provider).match(beats)

    assert matches[0].suggestions[0].reason == "Calmer than the literal shot."
    assert matches[0].suggestions[0].confidence == 0.9
    assert "CANDIDATES:" in provider.prompts[0]
    assert "literal match is often the wrong answer" in provider.prompts[0]


async def test_no_good_match_is_reported_as_a_gap(matcher_setup):
    workspace, engine = matcher_setup
    provider = StubTextProvider([
        RerankResult(choices=[], no_good_match=True,
                     missing_footage="A close-up of a hand signing a contract.")
    ])
    beats = parse_and_segment("The contract is finally signed.", "a.txt")
    matches = await TranscriptMatcher(workspace, engine, provider).match(beats)

    assert matches[0].no_good_match
    assert not matches[0].suggestions
    assert gaps(matches) == matches


async def test_a_reranker_outage_does_not_lose_the_timeline(matcher_setup):
    workspace, engine = matcher_setup

    class Broken:
        name = "broken"

        async def complete(self, prompt, schema):
            raise RuntimeError("provider down")

        def estimate_cost(self, prompt):
            return 0.0

    beats = parse_and_segment("Someone pours a coffee slowly.", "a.txt")
    matches = await TranscriptMatcher(workspace, engine, Broken()).match(beats)
    assert matches[0].suggestions, "search order should still stand in"


async def test_variety_penalises_reusing_the_same_shot(matcher_setup):
    workspace, engine = matcher_setup
    workspace.transcript.suggestions_per_beat = 1
    text = ("Coffee is poured. Coffee is poured again. Coffee is poured once more. "
            "And coffee is poured a fourth time.")
    beats = parse_and_segment(text, "a.txt", beat_min_s=1.0, beat_max_s=6.0)
    matches = await TranscriptMatcher(workspace, engine).match(beats)

    chosen = [m.chosen.shot.id for m in matches if m.chosen]
    assert len(chosen) >= 2
    assert len(set(chosen)) > 1, "the same clip was used for every beat"


async def test_alternatives_are_kept_so_a_suggestion_can_be_swapped(matcher_setup):
    workspace, engine = matcher_setup
    beats = parse_and_segment("Someone pours a coffee slowly.", "a.txt")
    matches = await TranscriptMatcher(workspace, engine).match(beats)
    match = matches[0]

    assert len(match.alternatives) >= len(match.suggestions)
    other = next(s for s in match.alternatives if s.shot.id != match.chosen.shot.id)

    assert match.choose(other.shot.id) is True
    assert match.chosen.shot.id == other.shot.id
    assert match.choose("not-a-shot") is False
