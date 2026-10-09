"""Pydantic row models. These mirror the tables in schema.sql."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..analysis.schema import AnalysisResult

SourceStatus = Literal["pending", "analysing", "indexed", "failed", "needs_review"]
ShotStatus = Literal["pending", "indexed", "failed", "needs_review"]
Origin = Literal["upload", "local", "drive"]
MediaKind = Literal["video", "image"]


class Workspace(BaseModel):
    id: str
    name: str
    drive_root_folder_id: str | None = None
    provider: str = "gemini"
    db_path: str
    created_at: str | None = None


class Source(BaseModel):
    id: str
    workspace_id: str
    content_hash: str
    original_filename: str
    origin: Origin
    media_kind: MediaKind = "video"
    origin_path: str | None = None
    drive_file_id: str | None = None
    drive_web_link: str | None = None
    drive_path: str | None = None
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    codec: str | None = None
    filesize_bytes: int | None = None
    created_at: str | None = None
    indexed_at: str | None = None
    analysis_version: str | None = None
    status: SourceStatus = "pending"
    error_message: str | None = None
    #: The plan for this file: every stretch the model found, including the ones left out (setup,
    #: dead air). Stored so a run killed half way resumes with the same plan.
    segments_json: str | None = None

    @property
    def segments(self) -> list[dict[str, Any]]:
        try:
            value = json.loads(self.segments_json or "[]")
        except ValueError:
            return []
        return value if isinstance(value, list) else []


class Shot(BaseModel):
    id: str
    workspace_id: str
    source_id: str
    shot_index: int = 0
    is_primary: bool = True
    start_s: float = 0.0
    end_s: float = 0.0
    duration_s: float = 0.0
    thumbnail_path: str | None = None

    caption: str | None = None
    confidence: float | None = None

    action: str | None = None
    setting: str | None = None
    setting_detail: str | None = None
    shot_type: str | None = None
    camera_movement: str | None = None
    time_of_day: str | None = None
    colour_profile: str | None = None
    people_count: str | None = None
    has_recognisable_faces: bool | None = None
    has_text_on_screen: bool | None = None
    pace: str | None = None

    subjects: list[str] = Field(default_factory=list)
    mood: list[str] = Field(default_factory=list)
    usable_for: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    emotions: list[str] = Field(default_factory=list)
    #: Literal things the model said it could see; the tags and caption are built from these.
    observations: list[str] = Field(default_factory=list)
    #: Client themes the footage clearly shows (a closed list, so never free-form noise).
    themes: list[str] = Field(default_factory=list)
    category: str | None = None
    category_confidence: float | None = None
    secondary_categories: list[str] = Field(default_factory=list)
    featured_person: bool = False
    top_pick: bool = False
    #: The strongest stretch inside this shot, when the model picked one (seconds in the file).
    best_start_s: float | None = None
    best_end_s: float | None = None
    #: Why this shot is waiting for a person (empty when nothing is wrong).
    review_reasons: list[str] = Field(default_factory=list)

    status: ShotStatus = "pending"
    error_message: str | None = None
    analysis_version: str | None = None
    analysed_at: str | None = None
    search_text: str = ""
    raw_analysis: dict[str, Any] | None = None

    @classmethod
    def from_row(cls, row: Any, tags: list[str] | None = None) -> "Shot":
        d = dict(row)
        for src, dst in (
            ("subjects_json", "subjects"),
            ("mood_json", "mood"),
            ("usable_for_json", "usable_for"),
            ("quality_flags_json", "quality_flags"),
            ("emotions_json", "emotions"),
            ("secondary_categories_json", "secondary_categories"),
            ("observations_json", "observations"),
            ("themes_json", "themes"),
            ("review_reasons_json", "review_reasons"),
        ):
            d[dst] = json.loads(d.pop(src, "[]") or "[]")
        raw = d.pop("raw_analysis_json", None)
        d["raw_analysis"] = json.loads(raw) if raw else None
        d["is_primary"] = bool(d.get("is_primary"))
        d["featured_person"] = bool(d.get("featured_person") or 0)
        d["top_pick"] = bool(d.get("top_pick") or 0)
        for key in ("has_recognisable_faces", "has_text_on_screen"):
            if d.get(key) is not None:
                d[key] = bool(d[key])
        d.pop("rowid", None)
        d.pop("concept_text", None)
        d["tags"] = tags or []
        return cls.model_validate(d)

    def apply_analysis(self, result: AnalysisResult, analysis_version: str) -> "Shot":
        self.caption = result.caption
        self.confidence = result.confidence
        self.action = result.action
        self.setting = result.setting
        self.setting_detail = result.setting_detail
        self.shot_type = result.shot_type.value
        self.camera_movement = result.camera_movement.value
        self.time_of_day = result.time_of_day.value
        self.colour_profile = result.colour_profile.value
        self.people_count = result.people_count.value
        self.has_recognisable_faces = result.has_recognisable_faces
        self.has_text_on_screen = result.has_text_on_screen
        self.pace = result.pace.value
        self.subjects = result.subjects
        self.mood = result.mood
        self.usable_for = result.usable_for
        self.quality_flags = result.quality_flags
        self.tags = result.tags
        self.emotions = result.emotions
        self.observations = result.observations
        self.themes = result.themes
        self.category = result.category
        self.category_confidence = result.category_confidence
        self.secondary_categories = result.secondary_categories
        self.featured_person = result.featured_person_in_shot
        # top_pick is an editor's choice, so a re-analysis never touches it.
        self.raw_analysis = result.model_dump(mode="json")
        self.analysis_version = analysis_version
        self.analysed_at = datetime.now(UTC).isoformat(timespec="seconds")
        return self


#: Reasons that mean "the folder it was given may be wrong". A shaky clip is flagged for review too,
#: but it is still in the right folder, so it is not among these.
ROUTING_REASONS = frozenset({
    "low_confidence", "low_category_confidence", "category_unmatched", "analysis_failed",
    "no_usable_part",
})


class ShotFacets(BaseModel):
    """The subset of a shot the taxonomy needs. Keeps taxonomy.py pure."""

    shot_id: str
    source_id: str
    is_primary: bool = True
    start_s: float = 0.0
    subjects: list[str] = Field(default_factory=list)
    action: str | None = None
    setting: str | None = None
    mood: list[str] = Field(default_factory=list)
    usable_for: list[str] = Field(default_factory=list)
    shot_type: str | None = None
    camera_movement: str | None = None
    time_of_day: str | None = None
    colour_profile: str | None = None
    needs_review: bool = False
    emotions: list[str] = Field(default_factory=list)
    category: str | None = None
    secondary_categories: list[str] = Field(default_factory=list)
    top_pick: bool = False
    #: The folder this shot was given is in doubt; see ROUTING_REASONS.
    misfile_risk: bool = False

    @classmethod
    def from_shot(cls, shot: Shot) -> "ShotFacets":
        return cls(
            shot_id=shot.id,
            source_id=shot.source_id,
            is_primary=shot.is_primary,
            start_s=shot.start_s,
            subjects=shot.subjects,
            action=shot.action,
            setting=shot.setting,
            mood=shot.mood,
            usable_for=shot.usable_for,
            shot_type=shot.shot_type,
            camera_movement=shot.camera_movement,
            time_of_day=shot.time_of_day,
            colour_profile=shot.colour_profile,
            needs_review=shot.status == "needs_review",
            emotions=shot.emotions,
            category=shot.category,
            secondary_categories=shot.secondary_categories,
            top_pick=shot.top_pick,
            misfile_risk=bool(ROUTING_REASONS.intersection(shot.review_reasons)),
        )


class Job(BaseModel):
    id: str
    workspace_id: str
    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)
    status: str = "queued"
    attempts: int = 0
    last_error: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    not_before: str | None = None
    cost_estimate_usd: float = 0.0
    claimed_by: str | None = None

    @classmethod
    def from_row(cls, row: Any) -> "Job":
        d = dict(row)
        d["payload"] = json.loads(d.pop("payload_json", "{}") or "{}")
        return cls.model_validate(d)
