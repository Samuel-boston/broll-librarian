"""Hybrid search: FTS5 keyword + vector KNN, fused with reciprocal rank fusion.

Neither method alone is good enough. Keyword search misses "calm morning by the
water" against a caption that says "serene sunrise beach"; vector search misses
an exact term the editor knows is in the tags.

Two details that are easy to get wrong and are handled here:

* **FTS5 escaping.** A raw query like ``person's wide-shot`` is a syntax error,
  not a search. Every token is quoted before it reaches FTS5.
* **Vector filtering.** A vec0 KNN needs a ``k`` and cannot pre-filter against a
  join on shots, so the vector side over-fetches (``limit x 10``) and the filter
  predicates are applied afterwards. The FTS5 side applies them as normal SQL.

Fusion is RRF with k=60. Normalising and adding raw scores is worse and fiddlier.
"""

from __future__ import annotations

import logging
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

_TOKEN = re.compile(r"[\w']+", re.UNICODE)


@dataclass
class SearchResult:
    shot: Shot
    source: Source
    score: float
    keyword_rank: int | None = None
    vector_rank: int | None = None
    matched: list[str] = field(default_factory=list)

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


def escape_fts_query(text: str) -> str:
    """Turn arbitrary user text into a safe FTS5 MATCH expression.

    Tokens are extracted and quoted individually, so apostrophes, hyphens and
    FTS operators (AND, OR, NOT, NEAR, *, ^, :) are all literal text and can
    never raise a syntax error. Tokens are OR-ed: for natural language, bm25
    ranking over OR beats an AND that returns nothing.
    """
    tokens = _TOKEN.findall(text or "")
    quoted = [f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens if token]
    return " OR ".join(quoted)


class SearchEngine:
    def __init__(self, store: Store, embedder: Embedder | None = None):
        self.store = store
        self.embedder = embedder

    # -- the two sides ------------------------------------------------------

    def keyword_search(
        self, query: str, filters: SearchFilters, limit: int
    ) -> list[tuple[str, float]]:
        match = escape_fts_query(query)
        if not match:
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
        rows = self.store.conn.execute(
            sql, (match, self.store.workspace_id, *params, limit)
        ).fetchall()
        return [(r["shot_id"], float(r["rank"])) for r in rows]

    def vector_search(
        self, query: str, filters: SearchFilters, limit: int
    ) -> list[tuple[str, float]]:
        if self.embedder is None:
            return []
        try:
            vector = self.embedder.embed_one(query)
        except Exception as exc:  # a missing model must not break keyword search
            log.warning("embedding the query failed, keyword-only results: %s", exc)
            return []

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

    # -- fusion -------------------------------------------------------------

    def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        limit: int = 20,
    ) -> list[SearchResult]:
        filters = filters or SearchFilters()

        if not (query or "").strip():
            return self._browse(filters, limit)

        keyword = self.keyword_search(query, filters, limit * 2)
        vector = self.vector_search(query, filters, limit * 2)

        scores: dict[str, float] = {}
        keyword_rank: dict[str, int] = {}
        vector_rank: dict[str, int] = {}

        for rank, (shot_id, _) in enumerate(keyword, start=1):
            scores[shot_id] = scores.get(shot_id, 0.0) + 1.0 / (RRF_K + rank)
            keyword_rank[shot_id] = rank
        for rank, (shot_id, _) in enumerate(vector, start=1):
            scores[shot_id] = scores.get(shot_id, 0.0) + 1.0 / (RRF_K + rank)
            vector_rank[shot_id] = rank

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
