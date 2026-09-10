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
from dataclasses import dataclass, field
from typing import Any

from ..analysis.embedder import Embedder
from ..db.models import Shot, Source
from ..db.store import Store
from .filters import SearchFilters

log = logging.getLogger(__name__)

RRF_K = 60
VECTOR_OVERFETCH = 10

# The relevance gate. Calibrated for all-MiniLM-L6-v2 against real client
# footage and the fixture library: a clearly relevant clip scores 0.33-0.71
# cosine, and the closest *irrelevant* clip trails the best match by 0.13 or
# more. Re-calibrate if the embedder changes - other models score differently.
VECTOR_FLOOR = 0.25       # below this a clip is not about the query at all
VECTOR_GAP = 0.12         # ...and it must sit close to the best match
KEYWORD_COVERAGE = 0.5    # half the meaningful words literally present: keep
KEYWORD_SIM_FLOOR = 0.20  # fewer present: keep only if also in the right area

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
    def __init__(self, store: Store, embedder: Embedder | None = None):
        self.store = store
        self.embedder = embedder
        # How many candidates the last strict search judged not relevant.
        self.hidden_count = 0

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
            SELECT s.id AS shot_id, bm25(shots_fts) AS rank
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
            return self.embedder.embed_one(query)
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
        """Fraction of the query's meaningful words each shot literally contains."""
        if not tokens or not shot_ids:
            return {}
        hits = dict.fromkeys(shot_ids, 0)
        placeholders = ", ".join("?" for _ in shot_ids)
        for token in tokens:
            rows = self.store.conn.execute(
                f"""SELECT s.id FROM shots_fts JOIN shots s ON s.rowid = shots_fts.rowid
                    WHERE shots_fts MATCH ? AND s.workspace_id = ? AND s.id IN ({placeholders})""",
                (_quote(token), self.store.workspace_id, *shot_ids),
            ).fetchall()
            for row in rows:
                hits[row["id"]] += 1
        return {sid: count / len(tokens) for sid, count in hits.items()}

    def _similarities(self, query_vector: list[float], shot_ids: list[str]) -> dict[str, float]:
        query = _unit(query_vector)
        return {
            sid: sum(a * b for a, b in zip(_unit(vector), query))
            for sid, vector in self.store.vectors.get_many(shot_ids).items()
        }

    def _relevant(
        self,
        candidates: list[str],
        tokens: list[str],
        keyword_hits: set[str],
        query_vector: list[float] | None,
    ) -> tuple[set[str], dict[str, float]]:
        """Which candidates are genuinely about the query.

        A clip is kept if most of what was asked for is literally in it; or
        some of it is, and it means roughly the same thing; or it means the same
        thing and sits close to the best match. The literal route matters:
        "nervous system regulation" scores only 0.11 semantically against a
        meditation clip *tagged* "nervous system", and must still find it.
        """
        coverage = self._coverage(tokens, [c for c in candidates if c in keyword_hits])
        sims = self._similarities(query_vector, candidates) if query_vector else {}
        best = max(sims.values(), default=None)

        keep: set[str] = set()
        for sid in candidates:
            cov = coverage.get(sid, 0.0)
            sim = sims.get(sid)
            if sim is None:
                # Nothing to judge meaning by (no embedder, or not embedded
                # yet): a literal hit on a meaningful word is enough.
                if cov > 0:
                    keep.add(sid)
            elif cov >= KEYWORD_COVERAGE:
                keep.add(sid)
            elif cov > 0 and sim >= KEYWORD_SIM_FLOOR:
                keep.add(sid)
            elif best is not None and sim >= VECTOR_FLOOR and sim >= best - VECTOR_GAP:
                keep.add(sid)
        return keep, sims

    # -- fusion -------------------------------------------------------------

    def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        limit: int = 20,
        strict: bool = True,
    ) -> list[SearchResult]:
        filters = filters or SearchFilters()
        self.hidden_count = 0

        if not (query or "").strip():
            return self._browse(filters, limit)

        tokens = meaningful_tokens(query)
        keyword = self._keyword_candidates(tokens, filters, limit * 2)
        query_vector = self._embed(query)
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
        if strict and scores:
            keep, sims = self._relevant(list(scores), tokens, set(keyword_rank), query_vector)
            self.hidden_count = len(scores) - len(keep)
            scores = {sid: value for sid, value in scores.items() if sid in keep}

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
                )
            )
        return results

    def _browse(self, filters: SearchFilters, limit: int) -> list[SearchResult]:
        """An empty query with filters is a browse, ordered newest first."""
        clauses, params = filters.predicates()
        where = "".join(f" AND {clause}" for clause in clauses)
        rows = self.store.conn.execute(
            f"""SELECT s.id FROM shots s JOIN sources src ON src.id = s.source_id
                WHERE s.workspace_id = ?{where}
                ORDER BY src.created_at DESC, s.shot_index LIMIT ?""",
            (self.store.workspace_id, *params, limit),
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
