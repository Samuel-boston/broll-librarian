"""Building a shot's search columns from its database row.

Used by the store (every time a shot changes) and by the migration that introduced the two-column
index. The words themselves are chosen in broll.searchtext.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from typing import Any

from .. import searchtext


def _list(row: sqlite3.Row, column: str) -> list[str]:
    # A column added by a later migration is not there yet while an earlier one rewrites the text.
    if column not in row.keys():
        return []
    return json.loads(row[column] or "[]")


def tags_for(conn: sqlite3.Connection, workspace_id: str, shot_id: str) -> list[str]:
    rows = conn.execute(
        """SELECT t.name FROM shot_tags st
           JOIN tags t ON t.id = st.tag_id
           WHERE st.workspace_id = ? AND st.shot_id = ? ORDER BY t.name""",
        (workspace_id, shot_id),
    ).fetchall()
    return [r["name"] for r in rows]


def _categories(row: sqlite3.Row) -> list[str | None]:
    return [row["category"], *_list(row, "secondary_categories_json")]


def search_columns(row: sqlite3.Row, tags: list[str]) -> tuple[str, str]:
    """(primary text, concept text) for one shot row."""
    primary = searchtext.primary_terms(
        caption=row["caption"],
        action=row["action"],
        setting=row["setting"],
        setting_detail=row["setting_detail"],
        subjects=_list(row, "subjects_json"),
        tags=tags,
        time_of_day=row["time_of_day"],
    )
    concept = searchtext.concept_terms(
        themes=_list(row, "themes_json"),
        mood=_list(row, "mood_json"),
        emotions=_list(row, "emotions_json"),
        categories=_categories(row),
        phrases=_list(row, "phrases_json"),
    )
    return searchtext.join_terms(primary), searchtext.join_terms(concept)


def embedding_for(row: sqlite3.Row, tags: list[str], folder_notes: Mapping[str, str] | None = None) -> str:
    return searchtext.embedding_text(
        caption=row["caption"],
        action=row["action"],
        setting=row["setting"],
        setting_detail=row["setting_detail"],
        subjects=_list(row, "subjects_json"),
        tags=tags,
        time_of_day=row["time_of_day"],
        themes=_list(row, "themes_json"),
        mood=_list(row, "mood_json"),
        emotions=_list(row, "emotions_json"),
        categories=_categories(row),
        folder_notes=folder_notes,
        phrases=_list(row, "phrases_json"),
    )


def rewrite_all_search_text(conn: sqlite3.Connection) -> int:
    """Recompute both search columns for every shot. Returns how many were rewritten."""
    previous = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        rows: list[Any] = conn.execute("SELECT * FROM shots").fetchall()
        for row in rows:
            tags = tags_for(conn, row["workspace_id"], row["id"])
            primary, concept = search_columns(row, tags)
            conn.execute(
                "UPDATE shots SET search_text = ?, concept_text = ? WHERE id = ?",
                (primary, concept, row["id"]),
            )
        return len(rows)
    finally:
        conn.row_factory = previous
