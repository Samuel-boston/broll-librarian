"""Beats -> ranked clip suggestions.

Retrieval is the same hybrid search the search screen uses; a text model then
reranks the shortlist, because the best B-roll for a line of narration is
usually not its most literal illustration. Narration about "slowing down" wants
a calm, slow-paced visual, not necessarily a clock.

Two things the reranker is explicitly told to do: return "no good match" rather
than forcing a bad one, and explain each choice in one line. The gaps are
surfaced to the user as a list of footage they should go shoot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ..config import WorkspaceConfig
from ..db.models import Shot, Source
from ..search.filters import SearchFilters
from ..search.query import SearchEngine, SearchResult
from .parser import Beat

log = logging.getLogger(__name__)

# How much a shot's score is cut for each time it has already been used.
REPEAT_PENALTY = 0.45


class RerankChoice(BaseModel):
    candidate: int = Field(description="1-based index of the candidate you are choosing.")
    reason: str = Field(description="One line: why this shot suits this narration.")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class RerankResult(BaseModel):
    choices: list[RerankChoice] = Field(default_factory=list)
    no_good_match: bool = Field(
        default=False,
        description="True when none of the candidates genuinely suits this beat.",
    )
    missing_footage: str | None = Field(
        default=None,
        description="If nothing fits, one line describing the shot that would.",
    )


RERANK_SYSTEM = """\
You are a video editor choosing B-roll to cut under a line of narration.

You are given the narration and a numbered list of candidate clips, each with a \
caption and its facets. Choose the best {count} in order.

What matters:
- The literal match is often the wrong answer. Narration about "slowing down" \
wants a calm, slow-paced visual, not necessarily a clock. Narration about \
"growth" rarely wants a chart.
- Prefer clips whose mood and pace match the tone of the line.
- Prefer clips long enough to cover the beat, but do not reject a good shot \
just for being short.
- Give one line of reason per choice, addressed to the editor.
- If none of the candidates genuinely works, set no_good_match and say in \
missing_footage what should be shot instead. An honest gap is more useful than \
a forced match."""


@dataclass
class Suggestion:
    shot: Shot
    source: Source
    reason: str = ""
    confidence: float = 0.0
    score: float = 0.0
    reused: bool = False

    @property
    def covers(self) -> float:
        return self.shot.duration_s

    @property
    def drive_link(self) -> str | None:
        if not self.source.drive_web_link:
            return None
        if self.shot.start_s <= 0.5:
            return self.source.drive_web_link
        return f"{self.source.drive_web_link}#t={self.shot.start_s:.0f}"


@dataclass
class BeatMatch:
    beat: Beat
    suggestions: list[Suggestion] = field(default_factory=list)
    # Everything the reranker ranked, so the UI can offer a swap without
    # re-running retrieval.
    alternatives: list[Suggestion] = field(default_factory=list)
    no_good_match: bool = False
    missing_footage: str | None = None

    def choose(self, shot_id: str) -> bool:
        """Promote an alternative to the top. Returns whether anything moved."""
        pool = self.alternatives or self.suggestions
        picked = next((s for s in pool if s.shot.id == shot_id), None)
        if picked is None:
            return False
        rest = [s for s in self.suggestions if s.shot.id != shot_id]
        self.suggestions = [picked, *rest][: max(1, len(self.suggestions))]
        return True

    @property
    def chosen(self) -> Suggestion | None:
        return self.suggestions[0] if self.suggestions else None

    @property
    def short_by_s(self) -> float:
        """How much of the beat the chosen clip cannot cover."""
        if not self.chosen:
            return self.beat.duration_s
        return max(0.0, self.beat.duration_s - self.chosen.shot.duration_s)


class TranscriptMatcher:
    def __init__(
        self,
        config: WorkspaceConfig,
        engine: SearchEngine,
        text_provider=None,
        filters: SearchFilters | None = None,
    ):
        self.config = config
        self.engine = engine
        self.text_provider = text_provider
        self.filters = filters or SearchFilters(exclude_flagged=True)

    async def match(self, beats: list[Beat]) -> list[BeatMatch]:
        usage: dict[str, int] = {}
        matches: list[BeatMatch] = []
        for beat in beats:
            matches.append(await self._match_beat(beat, usage))
        return matches

    async def _match_beat(self, beat: Beat, usage: dict[str, int]) -> BeatMatch:
        # Loose on purpose: the reranker judges relevance itself and is told to
        # return "no good match" - strict search would starve it on abstract
        # narration, which rarely shares words with a caption.
        candidates = self.engine.search(
            beat.text, self.filters, self.config.transcript.candidates_per_beat, strict=False
        )
        if not candidates:
            return BeatMatch(
                beat=beat,
                no_good_match=True,
                missing_footage="Nothing in the library matched this beat at all.",
            )

        reranked = await self._rerank(beat, candidates)
        ordered = self._enforce_variety(reranked, usage)
        wanted = self.config.transcript.suggestions_per_beat
        chosen = ordered[:wanted]

        for suggestion in chosen[:1]:
            usage[suggestion.shot.id] = usage.get(suggestion.shot.id, 0) + 1

        return BeatMatch(
            beat=beat,
            suggestions=chosen,
            alternatives=ordered,
            no_good_match=not chosen,
            missing_footage=None if chosen else "No candidate suited this beat.",
        )

    async def _rerank(self, beat: Beat, candidates: list[SearchResult]) -> list[Suggestion]:
        fallback = [
            Suggestion(
                shot=result.shot,
                source=result.source,
                reason="Ranked by hybrid search.",
                confidence=0.4,
                score=1.0 / (index + 1),
            )
            for index, result in enumerate(candidates)
        ]
        if self.text_provider is None:
            return fallback

        prompt = _rerank_prompt(beat, candidates, self.config.transcript.suggestions_per_beat)
        try:
            result = await self.text_provider.complete(prompt, RerankResult)
        except Exception as exc:  # a reranker outage must not lose the timeline
            log.warning("rerank failed for beat %d, using search order: %s", beat.index, exc)
            return fallback

        if result.no_good_match and not result.choices:
            return []

        suggestions: list[Suggestion] = []
        for position, choice in enumerate(result.choices):
            index = choice.candidate - 1
            if not (0 <= index < len(candidates)):
                continue
            candidate = candidates[index]
            suggestions.append(
                Suggestion(
                    shot=candidate.shot,
                    source=candidate.source,
                    reason=choice.reason.strip(),
                    confidence=choice.confidence,
                    score=1.0 / (position + 1),
                )
            )
        return suggestions or fallback

    def _enforce_variety(
        self, suggestions: list[Suggestion], usage: dict[str, int]
    ) -> list[Suggestion]:
        """Don't reuse a shot across the timeline unless nothing else fits."""
        scored: list[Suggestion] = []
        for suggestion in suggestions:
            used = usage.get(suggestion.shot.id, 0)
            suggestion.reused = used > 0
            suggestion.score *= REPEAT_PENALTY ** used
            scored.append(suggestion)
        return sorted(scored, key=lambda s: -s.score)


def _rerank_prompt(beat: Beat, candidates: list[SearchResult], count: int) -> str:
    lines = [
        RERANK_SYSTEM.format(count=count),
        "",
        f"NARRATION (beat {beat.index + 1}, {beat.duration_s:.1f}s):",
        beat.text,
        "",
        "CANDIDATES:",
    ]
    for index, result in enumerate(candidates, start=1):
        shot = result.shot
        facets = ", ".join(
            value for value in (
                shot.shot_type, shot.camera_movement, shot.setting,
                shot.time_of_day, shot.pace and f"{shot.pace} pace",
            ) if value
        )
        mood = ", ".join(shot.mood)
        lines.append(
            f"{index}. {shot.caption or 'no caption'} "
            f"[{facets}{'; mood: ' + mood if mood else ''}; "
            f"{shot.duration_s:.1f}s]"
        )
    return "\n".join(lines)


def gaps(matches: list[BeatMatch]) -> list[BeatMatch]:
    """Beats with nothing suitable - the shot list for the next shoot day."""
    return [m for m in matches if m.no_good_match]
