"""Hybrid search: FTS5 keyword + vector KNN, fused with reciprocal rank fusion,
then gated for relevance.

Neither method alone is good enough. Keyword search misses "calm morning by the
water" against a caption that says "serene sunrise beach"; vector search misses
an exact term the editor knows is in the tags.

Details that are easy to get wrong and are handled here:

* **FTS5 escaping.** A raw query like ``person's wide-shot`` is a syntax error,
  not a search. Every token is quoted before it reaches FTS5.
* **Filler words.** "I need a shot of him meditating" is a request, not six
  search terms: "a" and "of" match every caption. Only meaningful words reach
  the keyword index.
* **Vector filtering.** A vec0 KNN needs a ``k`` and cannot pre-filter against a
  join on shots, so the vector side over-fetches (``limit x 10``) and the filter
  predicates are applied afterwards. The FTS5 side applies them as normal SQL.
* **Relevance.** Ranking alone always returns *something*: a vector search
  finds the nearest clips however far away they are. So after fusion a clip
  must earn its place - see ``SearchEngine._relevant``. Loosely related clips
  are counted, not lost: ``strict=False`` (the "show them" link) returns them.

Fusion is RRF with k=60. Normalising and adding raw scores is worse and fiddlier.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..analysis.embedder import Embedder
from ..db.models import Shot, Source
from ..db.store import Store
from .filters import SearchFilters
from .spelling import suggest

log = logging.getLogger(__name__)

RRF_K = 60
VECTOR_OVERFETCH = 10

# The relevance gate. A clip is kept if enough of the query is literally in it, or if it
# means the same thing and sits near the best match. "Means the same thing" is a cosine
# similarity, and how big a similarity must be to mean anything depends entirely on the
# embedding model: MiniLM scores unrelated clips near 0.05, Gemini's near 0.5. So the floor
# is not a fixed number. It is measured, per query, against how the query scores on a sample
# of the library itself: a clip has to stand clear of that background (BACKGROUND_Z standard
# deviations) to count. Only a library too small to sample falls back to the fixed numbers,
# which are per embedder (CALIBRATION).
#
# Fixed fallback, calibrated against real client footage: a clearly relevant clip scores
# 0.33-0.71 cosine on MiniLM, and the closest *irrelevant* clip trails the best match by 0.13
# or more.
CALIBRATION: dict[str, tuple[float, float]] = {
    # embedder name -> (floor, gap)
    "local": (0.25, 0.12),
    "gemini": (0.62, 0.07),
    "openai": (0.30, 0.10),
}
DEFAULT_CALIBRATION = (0.25, 0.12)
VECTOR_FLOOR = 0.25       # kept for callers/tests that read it: the MiniLM floor
VECTOR_GAP = 0.12
BACKGROUND_MIN = 60       # shots needed before the background can be measured
BACKGROUND_SAMPLE = 400   # shots sampled to measure it
BACKGROUND_Z = 3.0        # standard deviations above the background to count as on topic
GAP_Z = 1.5               # ...and how close to the best match, in the same units
MIN_GAP = 0.04
KEYWORD_COVERAGE = 0.5    # half the (weighted) meaningful words present: keep
KEYWORD_PARTIAL = 0.25    # a quarter present: keep only if also on topic
NEAR_MISS_GAP = 0.18      # a hidden clip is a "near match" only if this close
NEAR_MISS_FLOOR = 0.20    # ...and related in its own right, not just "least bad"
# A word found only in the concept column (a theme, a feeling, a folder name) is an interpretation,
# so it counts for less than one found in the description of what is on screen.
CONCEPT_ONLY_WEIGHT = 0.7
# How much more the picture's own words matter than the concept words when ranking.
FTS_WEIGHTS = (1.0, 0.6)
# A clip whose shot type / camera movement is what the query asked for ("close up of ...") rises a
# little. It is a nudge, never a filter: the model's shot-type label can be wrong.
STRUCTURE_BOOST = 0.004

_TOKEN = re.compile(r"[\w']+", re.UNICODE)

# Words that make a query a request rather than a description.
STOPWORDS = frozenset(
    """
    a an the and or but nor so of to in on at by for from with without into onto
    about over under off out up as than then there here this that these those
    i me my mine we us our you your he him his she her they them their it its
    is are was were be been being am do does did have has had can could would
    should will shall may might must need needs needed want wants wanted like
    looking look find show give get got please just some any something anything
    shot shots clip clips footage video videos broll b roll b-roll one ones
    which what who whom where when while how
    """.split()
)


def meaningful_tokens(text: str) -> list[str]:
    """The words of a query worth searching for, lower-case, in order."""
    out: list[str] = []
    for raw in _TOKEN.findall(text or ""):
        token = raw.lower().strip("'")
        if token and token not in STOPWORDS and token not in out:
            out.append(token)
    return out


@dataclass(frozen=True)
class Structure:
    """A phrase that asks for a kind of shot rather than a subject."""

    pattern: re.Pattern
    field: str
    values: tuple[str, ...]
    #: Take the phrase out of the words searched for. Off for words that also describe content.
    strip: bool = True


def _s(pattern: str, field: str, values: tuple[str, ...], strip: bool = True) -> Structure:
    return Structure(re.compile(pattern, re.I), field, values, strip)


# Shot type, camera movement and pace are not in the searchable text (see broll.searchtext), because
# as words they sit on most clips. Said as a *request* they still mean something, so they are read
# here and become a nudge on the ranking.
STRUCTURES: tuple[Structure, ...] = (
    _s(r"\b(?:extreme[- ])?close[- ]?ups?\b", "shot_type", ("close_up", "extreme_close_up", "macro")),
    _s(r"\bmacro\b", "shot_type", ("macro", "extreme_close_up"), strip=False),
    _s(r"\b(?:wide|establishing)(?:[- ]angle)?[- ]shots?\b|\bwide[- ]angle\b",
       "shot_type", ("wide", "extreme_wide")),
    _s(r"\b(?:aerial|drone|bird'?s[- ]eye)\b", "shot_type", ("aerial", "top_down"), strip=False),
    _s(r"\b(?:overhead|top[- ]down)\b", "shot_type", ("top_down",), strip=False),
    _s(r"\bover[- ]the[- ]shoulder\b", "shot_type", ("over_the_shoulder",)),
    _s(r"\bpov\b|\bpoint of view\b", "shot_type", ("pov",)),
    _s(r"\bhand[- ]?held\b", "camera_movement", ("handheld",)),
    _s(r"\b(?:locked[- ]off|tripod)\b|\bstatic shot\b", "camera_movement", ("static",)),
    _s(r"\b(?:slow )?push[- ]in\b|\bdolly in\b", "camera_movement", ("push_in",)),
    _s(r"\bpull[- ]out\b|\bdolly out\b", "camera_movement", ("pull_out",)),
    _s(r"\b(?:orbit|orbiting)\b", "camera_movement", ("orbit",), strip=False),
)


def parse_structure(text: str) -> tuple[str, dict[str, set[str]]]:
    """Split a query into (what is left to search for, the kinds of shot it asked for)."""
    wanted: dict[str, set[str]] = {}
    for structure in STRUCTURES:
        if structure.pattern.search(text or ""):
            wanted.setdefault(structure.field, set()).update(structure.values)
            if structure.strip:
                text = structure.pattern.sub(" ", text)
    return re.sub(r"\s+", " ", text or "").strip(), wanted


def _quote(token: str) -> str:
    return '"' + token.replace('"', '""') + '"'


def escape_fts_query(text: str) -> str:
    """Turn arbitrary user text into a safe FTS5 MATCH expression.

    Tokens are extracted and quoted individually, so apostrophes, hyphens and
    FTS operators (AND, OR, NOT, NEAR, *, ^, :) are all literal text and can
    never raise a syntax error. Tokens are OR-ed; the relevance gate, not the
    MATCH expression, decides what is close enough.
    """
    tokens = _TOKEN.findall(text or "")
    return " OR ".join(_quote(token) for token in tokens if token)


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else list(vector)


@dataclass
class SearchResult:
    shot: Shot
    source: Source
    score: float
    keyword_rank: int | None = None
    vector_rank: int | None = None
    matched: list[str] = field(default_factory=list)
    similarity: float | None = None
    # Set by an annotated rank-only search: "match" would pass the strict
    # relevance gate, "near" is a near miss, "weak" is neither.
    relevance: str | None = None
    coverage: float | None = None

    @property
    def timecode(self) -> str:
        minutes, seconds = divmod(int(self.shot.start_s), 60)
        return f"{minutes}m{seconds:02d}s"

    @property
    def drive_link(self) -> str | None:
        """Open the clip in Drive near the right moment."""
        if not self.source.drive_web_link:
            return None
        if self.shot.start_s <= 0.5:
            return self.source.drive_web_link
        return f"{self.source.drive_web_link}#t={self.shot.start_s:.0f}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_id": self.shot.id,
            "source_id": self.source.id,
            "filename": self.source.original_filename,
            "caption": self.shot.caption,
            "score": round(self.score, 6),
            "similarity": None if self.similarity is None else round(self.similarity, 4),
            "keyword_rank": self.keyword_rank,
            "vector_rank": self.vector_rank,
            "start_s": self.shot.start_s,
            "end_s": self.shot.end_s,
            "duration_s": self.shot.duration_s,
            "timecode": self.timecode,
            "shot_type": self.shot.shot_type,
            "camera_movement": self.shot.camera_movement,
            "setting": self.shot.setting,
            "action": self.shot.action,
            "mood": self.shot.mood,
            "people_count": self.shot.people_count,
            "quality_flags": self.shot.quality_flags,
            "tags": self.shot.tags,
            "emotions": self.shot.emotions,
            "category": self.shot.category,
            "secondary_categories": self.shot.secondary_categories,
            "featured_person": self.shot.featured_person,
            "top_pick": self.shot.top_pick,
            "status": self.shot.status,
            "thumbnail_path": self.shot.thumbnail_path,
            "drive_link": self.drive_link,
        }


class SearchEngine:
    def __init__(
        self,
        store: Store,
        embedder: Embedder | None = None,
        featured_person: str | None = None,
    ):
        self.store = store
        self.embedder = embedder
        # How many near matches the last strict search held back.
        self.hidden_count = 0
        # In a client's library their name is on nearly every clip, so as a
        # search word it matches everything: "clip of Adam meditating" pulled
        # in "Adam yawns in bed". It is a filter, not a search term.
        parts = (featured_person or "").split()
        terms = sorted({" ".join(parts), *[p for p in parts if len(p) >= 3]}, key=len, reverse=True)
        self._name_pattern = (
            re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")(?:'s)?\b", re.I)
            if parts else None
        )
        self._name_terms = [t.lower() for t in terms] if parts else []
        self._vocab: Counter | None = None
        # What the last search did to the query, for "showing results for...".
        self.corrections: list[tuple[str, str]] = []
        self.corrected_query = ""

    # -- typos ----------------------------------------------------------------

    def _vocabulary(self) -> Counter:
        """Words the library knows, with how many clips use each."""
        if self._vocab is None:
            words = self.store.vocabulary()
            for term in self._name_terms:  # so "Adma" can still become "Adam"
                for word in term.split():
                    words[word] += 1
            self._vocab = Counter({
                w: n for w, n in words.items()
                if len(w) >= 3 and w not in STOPWORDS and not w.isdigit()
            })
        return self._vocab

    def _document_frequency(self, token: str) -> int:
        """Clips containing this word in any stemmed form."""
        return int(self.store.conn.execute(
            """SELECT COUNT(*) FROM shots_fts JOIN shots s ON s.rowid = shots_fts.rowid
               WHERE shots_fts MATCH ? AND s.workspace_id = ?""",
            (_quote(token), self.store.workspace_id),
        ).fetchone()[0])

    def correct_query(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        """Fix words the library has never seen that are a keystroke or two
        from one it has. Returns (corrected text, [(typed, corrected), ...])."""
        vocabulary = self._vocabulary()
        if not vocabulary:
            return text, []
        corrections: list[tuple[str, str]] = []

        def fix(match: re.Match) -> str:
            typed = match.group(0)
            word = typed.lower()
            suffix = ""
            if word.endswith("'s"):
                word, suffix = word[:-2], "'s"
            if not word or word in STOPWORDS or word in vocabulary:
                return typed
            if self._document_frequency(word):
                return typed  # a stemmed form is in the library ("meditate")
            fixed = suggest(word, vocabulary)
            if not fixed:
                return typed
            corrections.append((word, fixed))
            return fixed + suffix

        return _TOKEN.sub(fix, text), corrections

    def split_person(self, query: str) -> tuple[str, bool]:
        """Take the featured person's name out of a query: (what is left, mentioned)."""
        if self._name_pattern is None:
            return query, False
        rest, count = self._name_pattern.subn(" ", query)
        return re.sub(r"\s+", " ", rest).strip(), count > 0

    # -- the two sides ------------------------------------------------------

    def keyword_search(
        self, query: str, filters: SearchFilters, limit: int
    ) -> list[tuple[str, float]]:
        return self._keyword_candidates(meaningful_tokens(query), filters, limit)

    def _keyword_candidates(
        self, tokens: list[str], filters: SearchFilters, limit: int
    ) -> list[tuple[str, float]]:
        if not tokens:
            return []
        clauses, params = filters.predicates()
        where = "".join(f" AND {clause}" for clause in clauses)
        sql = f"""
            SELECT s.id AS shot_id, bm25(shots_fts, {FTS_WEIGHTS[0]}, {FTS_WEIGHTS[1]}) AS rank
            FROM shots_fts
            JOIN shots s ON s.rowid = shots_fts.rowid
            JOIN sources src ON src.id = s.source_id
            WHERE shots_fts MATCH ? AND s.workspace_id = ?{where}
            ORDER BY rank
            LIMIT ?
        """
        match = " OR ".join(_quote(t) for t in tokens)
        rows = self.store.conn.execute(
            sql, (match, self.store.workspace_id, *params, limit)
        ).fetchall()
        return [(r["shot_id"], float(r["rank"])) for r in rows]

    def _embed(self, query: str) -> list[float] | None:
        if self.embedder is None:
            return None
        try:
            return self.embedder.embed_query(query)
        except Exception as exc:  # a missing model must not break keyword search
            log.warning("embedding the query failed, keyword-only results: %s", exc)
            return None

    def vector_search(
        self, query: str, filters: SearchFilters, limit: int
    ) -> list[tuple[str, float]]:
        vector = self._embed(query)
        return self._vector_candidates(vector, filters, limit) if vector else []

    def _vector_candidates(
        self, vector: list[float], filters: SearchFilters, limit: int
    ) -> list[tuple[str, float]]:
        # Over-fetch, then post-filter: the KNN cannot see the filter predicates.
        candidates = self.store.vectors.search(vector, limit * VECTOR_OVERFETCH)
        if not candidates:
            return []
        allowed = self._allowed_ids([shot_id for shot_id, _ in candidates], filters)
        return [(sid, dist) for sid, dist in candidates if sid in allowed][:limit]

    def _allowed_ids(self, shot_ids: list[str], filters: SearchFilters) -> set[str]:
        clauses, params = filters.predicates()
        where = "".join(f" AND {clause}" for clause in clauses)
        placeholders = ", ".join("?" for _ in shot_ids)
        rows = self.store.conn.execute(
            f"""SELECT s.id FROM shots s JOIN sources src ON src.id = s.source_id
                WHERE s.workspace_id = ? AND s.id IN ({placeholders}){where}""",
            (self.store.workspace_id, *shot_ids, *params),
        ).fetchall()
        return {r["id"] for r in rows}

    # -- relevance ----------------------------------------------------------

    def _coverage(self, tokens: list[str], shot_ids: list[str]) -> dict[str, float]:
        """How much of the query each shot literally contains, weighted by rarity.

        A word on most of the library says little about any one clip; a word
        on 2% of it says a lot. So each word counts by its inverse document
        frequency. A word *nothing* contains is left out entirely: it cannot
        tell clips apart - but it is still something the clip was asked to be
        and is not, so it weighs as much as the rarest possible word. That way
        "nervous system regulation" still finds a clip tagged "nervous system"
        (2 of 3 specific words), while "a mariachi band playing trumpets at a
        wedding" does not return a child playing in a park (1 of 5).

        A word that is in the description of what is on screen counts in full; one found only
        among the concepts (a theme, a feeling, a folder name) counts CONCEPT_ONLY_WEIGHT, so a
        clip cannot earn a place on interpretation alone.
        """
        if not tokens or not shot_ids:
            return {}
        total = max(1, self.store.count_shots())
        placeholders = ", ".join("?" for _ in shot_ids)
        weights: dict[str, float] = {}
        credit: dict[str, dict[str, float]] = {sid: {} for sid in shot_ids}
        for token in tokens:
            frequency = self._document_frequency(token)
            weights[token] = math.log(1 + total / max(frequency, 1))
            if not frequency:
                continue
            for column, value in (("", CONCEPT_ONLY_WEIGHT), ("search_text:", 1.0)):
                rows = self.store.conn.execute(
                    f"""SELECT s.id FROM shots_fts JOIN shots s ON s.rowid = shots_fts.rowid
                        WHERE shots_fts MATCH ? AND s.workspace_id = ? AND s.id IN ({placeholders})""",
                    (column + _quote(token), self.store.workspace_id, *shot_ids),
                ).fetchall()
                for row in rows:
                    credit[row["id"]][token] = value  # the second pass overwrites with the full credit
        weight = sum(weights.values())
        if not weight:
            return {}
        return {
            sid: sum(weights[t] * c for t, c in earned.items()) / weight
            for sid, earned in credit.items()
        }

    def _similarities(self, query_vector: list[float], shot_ids: list[str]) -> dict[str, float]:
        query = _unit(query_vector)
        return {
            sid: sum(a * b for a, b in zip(_unit(vector), query))
            for sid, vector in self.store.vectors.get_many(shot_ids).items()
        }

    def _thresholds(self, query_vector: list[float]) -> tuple[float, float, float, float]:
        """(floor, gap, near-miss floor, near-miss gap) for this query against this library.

        Measured when the library is big enough to sample: the floor is BACKGROUND_Z standard
        deviations above how this query scores on a spread of the library's own clips, whatever
        the embedding model. Otherwise the fixed numbers for the model in use.
        """
        sample_ids = self.store.sample_shot_ids(BACKGROUND_SAMPLE)
        if len(sample_ids) >= BACKGROUND_MIN:
            sims = list(self._similarities(query_vector, sample_ids).values())
            if len(sims) >= BACKGROUND_MIN:
                mean = sum(sims) / len(sims)
                std = math.sqrt(sum((x - mean) ** 2 for x in sims) / len(sims))
                floor = mean + BACKGROUND_Z * std
                gap = max(MIN_GAP, GAP_Z * std)
                return floor, gap, floor - std, gap * 1.5
        name = getattr(self.embedder, "name", "local")
        floor, gap = CALIBRATION.get(name, DEFAULT_CALIBRATION)
        return floor, gap, floor - 0.05, gap + 0.06

    def _relevant(
        self,
        candidates: list[str],
        tokens: list[str],
        keyword_hits: set[str],
        query_vector: list[float] | None,
    ) -> tuple[set[str], set[str], dict[str, float], dict[str, float]]:
        """Which candidates are genuinely about the query - and which nearly were.

        Kept: most of what was asked for is literally in it; or a good part is
        and it is on topic; or it means the same thing and sits close to the
        best match. The literal route matters: "nervous system regulation"
        scores only 0.11 semantically against a clip *tagged* "nervous system".
        A near match failed all three but was close; only those are offered
        behind "show them" - never the whole library.
        """
        coverage = self._coverage(tokens, [c for c in candidates if c in keyword_hits])
        sims = self._similarities(query_vector, candidates) if query_vector else {}
        best = max(sims.values(), default=None)
        floor, gap, near_floor, near_gap = (
            self._thresholds(query_vector) if query_vector else (0.0, 0.0, 0.0, 0.0)
        )

        keep: set[str] = set()
        near: set[str] = set()
        for sid in candidates:
            cov = coverage.get(sid, 0.0)
            sim = sims.get(sid)
            if sim is None:
                # Nothing to judge meaning by (no embedder, or not embedded
                # yet): the weighted literal match has to carry it alone.
                (keep if cov >= KEYWORD_PARTIAL else near if cov > 0 else set()).add(sid)
            elif cov >= KEYWORD_COVERAGE:
                keep.add(sid)
            elif cov >= KEYWORD_PARTIAL and sim >= floor:
                keep.add(sid)
            elif best is not None and sim >= floor and sim >= best - gap:
                keep.add(sid)
            elif cov >= KEYWORD_PARTIAL or (
                best is not None and sim >= near_floor and sim >= best - near_gap
            ):
                # Close, but not close enough. When even the best match is
                # irrelevant, "near the best" means nothing - hence the floor.
                near.add(sid)
        return keep, near, sims, coverage

    # -- fusion -------------------------------------------------------------

    def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        limit: int = 20,
        strict: bool = True,
        person_filter: bool = True,
        rank_only: bool = False,
        correct: bool = True,
        annotate: bool = False,
    ) -> list[SearchResult]:
        """Search.

        ``strict`` (the default) returns only relevant clips; ``strict=False``
        adds the near matches behind the "show them" link. ``rank_only`` skips
        the relevance gate and returns the top of the ranking whatever it is -
        for the transcript matcher, whose reranker judges relevance itself and
        needs candidates even for narration as abstract as "most of us start
        the day already behind". ``annotate`` (with ``rank_only``) still drops
        nothing, but labels each result with what the gate would have said, so
        an agent choosing for itself knows whether anything was really about
        the line - and can fall back rather than force a clip. ``person_filter=False``
        still takes the client's name out of the query but does not require them
        in shot - narration says "Adam" over footage with no Adam in it.
        ``correct=False`` searches for exactly what was typed ("search instead
        for...")."""
        filters = filters or SearchFilters()
        self.hidden_count = 0
        self.corrections = []
        self.corrected_query = query or ""

        if not (query or "").strip():
            return self._browse(filters, limit)

        if correct:
            query, self.corrections = self.correct_query(query)
            self.corrected_query = query
        text, mentioned = self.split_person(query)
        if mentioned and person_filter:
            filters = filters.model_copy(update={"featured_person": True})
        text, wanted = parse_structure(text)
        tokens = meaningful_tokens(text)
        if mentioned and person_filter and not tokens and not wanted:
            return self._browse(filters, limit)  # "Adam" alone: all of Adam's clips
        if wanted and not tokens:
            # "close up", on its own, is a kind of shot and nothing else: show those.
            return self._browse(
                filters.model_copy(update={k: sorted(v) for k, v in wanted.items()}), limit
            )

        keyword = self._keyword_candidates(tokens, filters, limit * 2)
        # The meaning side gets the natural phrasing, minus only the name:
        # measured, stripping the rest of the filler blurs single-word queries.
        query_vector = self._embed(text) if text else None
        vector = self._vector_candidates(query_vector, filters, limit * 2) if query_vector else []

        scores: dict[str, float] = {}
        keyword_rank: dict[str, int] = {}
        vector_rank: dict[str, int] = {}
        for rank, (shot_id, _) in enumerate(keyword, start=1):
            scores[shot_id] = scores.get(shot_id, 0.0) + 1.0 / (RRF_K + rank)
            keyword_rank[shot_id] = rank
        for rank, (shot_id, _) in enumerate(vector, start=1):
            scores[shot_id] = scores.get(shot_id, 0.0) + 1.0 / (RRF_K + rank)
            vector_rank[shot_id] = rank

        sims: dict[str, float] = {}
        coverage: dict[str, float] = {}
        labels: dict[str, str] = {}
        if scores and (not rank_only or annotate):
            keep, near, sims, coverage = self._relevant(
                list(scores), tokens, set(keyword_rank), query_vector
            )
            if rank_only:
                labels = {
                    sid: "match" if sid in keep else "near" if sid in near else "weak"
                    for sid in scores
                }
            else:
                shown = keep if strict else keep | near
                self.hidden_count = len(near) if strict else 0
                scores = {sid: value for sid, value in scores.items() if sid in shown}

        if wanted and scores:
            for shot_id, bonus in self._structure_bonus(list(scores), wanted).items():
                scores[shot_id] += bonus

        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        results: list[SearchResult] = []
        for shot_id, score in ordered:
            hydrated = self._hydrate(shot_id)
            if hydrated is None:
                continue
            shot, source = hydrated
            matched = [
                name for name, present in
                (("keyword", shot_id in keyword_rank), ("vector", shot_id in vector_rank))
                if present
            ]
            results.append(
                SearchResult(
                    shot=shot,
                    source=source,
                    score=score,
                    keyword_rank=keyword_rank.get(shot_id),
                    vector_rank=vector_rank.get(shot_id),
                    matched=matched,
                    similarity=sims.get(shot_id),
                    relevance=labels.get(shot_id),
                    coverage=coverage.get(shot_id, 0.0) if labels else None,
                )
            )
        return results

    def _structure_bonus(self, shot_ids: list[str], wanted: dict[str, set[str]]) -> dict[str, float]:
        """A nudge for clips whose shot type or camera movement is what the query asked for."""
        placeholders = ", ".join("?" for _ in shot_ids)
        rows = self.store.conn.execute(
            f"""SELECT id, shot_type, camera_movement FROM shots
                WHERE workspace_id = ? AND id IN ({placeholders})""",
            (self.store.workspace_id, *shot_ids),
        ).fetchall()
        bonus: dict[str, float] = {}
        for row in rows:
            hits = sum(1 for field, values in wanted.items() if row[field] in values)
            if hits:
                bonus[row["id"]] = STRUCTURE_BOOST * hits
        return bonus

    def browse(self, filters: SearchFilters, limit: int, offset: int = 0) -> list[SearchResult]:
        """Clips matching the filters, newest first - no query involved."""
        return self._browse(filters, limit, offset)

    def _browse(self, filters: SearchFilters, limit: int, offset: int = 0) -> list[SearchResult]:
        """An empty query with filters is a browse, ordered newest first."""
        clauses, params = filters.predicates()
        where = "".join(f" AND {clause}" for clause in clauses)
        rows = self.store.conn.execute(
            f"""SELECT s.id FROM shots s JOIN sources src ON src.id = s.source_id
                WHERE s.workspace_id = ?{where}
                ORDER BY src.created_at DESC, s.shot_index LIMIT ? OFFSET ?""",
            (self.store.workspace_id, *params, limit, max(0, offset)),
        ).fetchall()
        results = []
        for row in rows:
            hydrated = self._hydrate(row["id"])
            if hydrated:
                shot, source = hydrated
                results.append(SearchResult(shot=shot, source=source, score=0.0))
        return results

    def _hydrate(self, shot_id: str) -> tuple[Shot, Source] | None:
        shot = self.store.get_shot(shot_id)
        if shot is None:
            return None
        source = self.store.get_source(shot.source_id)
        if source is None:
            return None
        return shot, source
