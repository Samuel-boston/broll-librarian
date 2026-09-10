"""Search filters, compiled to SQL predicates.

Filters are applied as ordinary SQL predicates on the FTS5 side. On the vector
side they cannot be: sqlite-vec's KNN needs a ``k`` constraint and cannot
pre-filter against a join, so that side over-fetches and post-filters. See
query.py.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# Tables are aliased s (shots) and src (sources) in every query below.
QUALITY_FLAG_SQL = (
    "EXISTS (SELECT 1 FROM json_each(s.quality_flags_json) q WHERE q.value = ?)"
)


class SearchFilters(BaseModel):
    duration_min_s: float | None = None
    duration_max_s: float | None = None
    shot_type: list[str] = Field(default_factory=list)
    camera_movement: list[str] = Field(default_factory=list)
    time_of_day: list[str] = Field(default_factory=list)
    colour_profile: list[str] = Field(default_factory=list)
    people_count: list[str] = Field(default_factory=list)
    pace: list[str] = Field(default_factory=list)
    setting: list[str] = Field(default_factory=list)
    action: list[str] = Field(default_factory=list)
    subjects: list[str] = Field(default_factory=list)
    emotions: list[str] = Field(default_factory=list)
    category: list[str] = Field(default_factory=list)   # a folder, or a parent of folders
    mood: list[str] = Field(default_factory=list)
    usable_for: list[str] = Field(default_factory=list)
    has_faces: bool | None = None
    has_text_on_screen: bool | None = None
    featured_person: bool | None = None
    top_pick: bool | None = None
    min_width: int | None = None
    min_height: int | None = None
    quality_flags: list[str] = Field(default_factory=list)      # must have all of these
    exclude_quality_flags: list[str] = Field(default_factory=list)
    exclude_flagged: bool = False                                # any flag at all
    added_after: str | None = None
    added_before: str | None = None
    status: list[str] = Field(default_factory=lambda: ["indexed", "needs_review"])

    def is_empty(self) -> bool:
        return not self.predicates()[0]

    def predicates(self) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        def in_clause(column: str, values: list[str]) -> None:
            if values:
                clauses.append(f"{column} IN ({', '.join('?' for _ in values)})")
                params.extend(values)

        def json_any(column: str, values: list[str]) -> None:
            if values:
                placeholders = ", ".join("?" for _ in values)
                clauses.append(
                    f"EXISTS (SELECT 1 FROM json_each(s.{column}) j "
                    f"WHERE j.value IN ({placeholders}))"
                )
                params.extend(values)

        if self.duration_min_s is not None:
            clauses.append("s.duration_s >= ?")
            params.append(self.duration_min_s)
        if self.duration_max_s is not None:
            clauses.append("s.duration_s <= ?")
            params.append(self.duration_max_s)

        in_clause("s.shot_type", self.shot_type)
        in_clause("s.camera_movement", self.camera_movement)
        in_clause("s.time_of_day", self.time_of_day)
        in_clause("s.colour_profile", self.colour_profile)
        in_clause("s.people_count", self.people_count)
        in_clause("s.pace", self.pace)
        in_clause("s.setting", self.setting)
        in_clause("s.action", self.action)
        in_clause("s.status", self.status)

        json_any("mood_json", self.mood)
        json_any("usable_for_json", self.usable_for)
        json_any("subjects_json", self.subjects)
        json_any("emotions_json", self.emotions)
        if self.category:
            # Picking "01_Nervous System Practices" should find everything under it.
            clauses.append(
                "(" + " OR ".join("(s.category = ? OR s.category LIKE ?)" for _ in self.category) + ")"
            )
            for value in self.category:
                params.extend([value, value + "/%"])

        if self.has_faces is not None:
            clauses.append("s.has_recognisable_faces = ?")
            params.append(int(self.has_faces))
        if self.has_text_on_screen is not None:
            clauses.append("s.has_text_on_screen = ?")
            params.append(int(self.has_text_on_screen))
        if self.featured_person is not None:
            clauses.append("s.featured_person = ?")
            params.append(int(self.featured_person))
        if self.top_pick is not None:
            clauses.append("s.top_pick = ?")
            params.append(int(self.top_pick))

        if self.min_width is not None:
            clauses.append("src.width >= ?")
            params.append(self.min_width)
        if self.min_height is not None:
            clauses.append("src.height >= ?")
            params.append(self.min_height)

        for flag in self.quality_flags:
            clauses.append(QUALITY_FLAG_SQL)
            params.append(flag)
        for flag in self.exclude_quality_flags:
            clauses.append(f"NOT {QUALITY_FLAG_SQL}")
            params.append(flag)
        if self.exclude_flagged:
            clauses.append("json_array_length(s.quality_flags_json) = 0")

        if self.added_after:
            clauses.append("src.created_at >= ?")
            params.append(self.added_after)
        if self.added_before:
            clauses.append("src.created_at <= ?")
            params.append(self.added_before)

        return clauses, params
