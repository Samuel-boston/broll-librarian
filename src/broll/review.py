"""Human corrections to a shot's analysis.

Corrections are the most valuable data in the system: they are ground truth. So
they are stored, and everything derived from the shot - search_text, tags, the
embedding - is recomputed on save.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .analysis.embedder import Embedder
from .analysis.schema import (
    AnalysisResult,
    CameraMove,
    ColourProfile,
    Pace,
    PeopleCount,
    ShotType,
    TimeOfDay,
    normalise_term,
)
from .config import WorkspaceConfig
from .db.models import Shot
from .db.store import Store

log = logging.getLogger(__name__)

SCALAR_FIELDS = {
    "caption": None,
    "action": None,
    "setting": None,
    "setting_detail": None,
    "shot_type": ShotType,
    "camera_movement": CameraMove,
    "time_of_day": TimeOfDay,
    "colour_profile": ColourProfile,
    "people_count": PeopleCount,
    "pace": Pace,
    "category": None,
}
LIST_FIELDS = ("subjects", "mood", "emotions", "usable_for", "quality_flags", "tags", "themes")
BOOL_FIELDS = ("has_recognisable_faces", "has_text_on_screen", "top_pick", "featured_person")

ENUM_OPTIONS = {
    "shot_type": [m.value for m in ShotType],
    "camera_movement": [m.value for m in CameraMove],
    "time_of_day": [m.value for m in TimeOfDay],
    "colour_profile": [m.value for m in ColourProfile],
    "people_count": [m.value for m in PeopleCount],
    "pace": [m.value for m in Pace],
}


class CorrectionError(ValueError):
    pass


REASON_TEXT = {
    "category_unmatched": "the folder the model named isn't one of the client's folders",
    "analysis_failed": "the model could not describe it",
    "no_usable_part": "nothing in this file looked usable, so the whole of it was kept for you to judge",
}


def parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [part for part in value.replace("\n", ",").split(",")]
    return [term for term in (normalise_term(v) for v in value if isinstance(v, str)) if term]


def apply_correction(
    config: WorkspaceConfig,
    store: Store,
    shot_id: str,
    updates: dict[str, Any],
    embedder: Embedder | None = None,
    status: str = "indexed",
) -> Shot:
    """Write an operator's edits to a shot and recompute everything derived."""
    shot = store.get_shot(shot_id)
    if shot is None:
        raise CorrectionError(f"No shot {shot_id!r} in workspace {config.id!r}.")

    for field, enum_cls in SCALAR_FIELDS.items():
        if field not in updates:
            continue
        value = updates[field]
        value = value.strip() if isinstance(value, str) else value
        if enum_cls is not None and value:
            allowed = {m.value for m in enum_cls}
            if value not in allowed:
                raise CorrectionError(
                    f"{field}={value!r} is not one of: {', '.join(sorted(allowed))}"
                )
        setattr(shot, field, value or None)

    for field in LIST_FIELDS:
        if field in updates:
            setattr(shot, field, parse_list(updates[field]))

    for field in BOOL_FIELDS:
        if field in updates:
            setattr(shot, field, _as_bool(updates[field]))

    shot.status = status
    shot.error_message = None
    if status == "indexed":
        shot.review_reasons = []  # a person has looked: nothing is in doubt any more
    if shot.raw_analysis is not None:
        shot.raw_analysis = {**shot.raw_analysis, "corrected_by_operator": True}

    store.update_shot(shot)
    text = store.recompute_search_text(shot.id)
    store.recompute_source_status(shot.source_id)

    if embedder is not None and text:
        try:
            store.vectors.upsert(shot.id, embedder.embed_documents([text])[0])
        except Exception as exc:  # a correction must land even if embedding fails
            log.warning("re-embedding %s after correction failed: %s", shot.id, exc)

    return store.get_shot(shot.id)  # type: ignore[return-value]


def as_analysis(shot: Shot) -> AnalysisResult | None:
    """The shot's current facets as an AnalysisResult, for re-export or diffing."""
    if not shot.caption:
        return None
    try:
        return AnalysisResult.model_validate(
            {
                "caption": shot.caption,
                "subjects": shot.subjects,
                "action": shot.action,
                "setting": shot.setting or "abstract",
                "setting_detail": shot.setting_detail,
                "shot_type": shot.shot_type or "unknown",
                "camera_movement": shot.camera_movement or "unknown",
                "time_of_day": shot.time_of_day or "unknown",
                "mood": shot.mood,
                "colour_profile": shot.colour_profile or "neutral",
                "people_count": shot.people_count or "none",
                "has_recognisable_faces": bool(shot.has_recognisable_faces),
                "has_text_on_screen": bool(shot.has_text_on_screen),
                "pace": shot.pace or "moderate",
                "tags": shot.tags,
                "usable_for": shot.usable_for,
                "quality_flags": shot.quality_flags,
                "confidence": shot.confidence or 0.5,
            }
        )
    except Exception:
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def review_queue(
    store: Store, limit: int = 200, below_confidence: float = 0.7
) -> list[dict[str, Any]]:
    """Shots needing attention, newest first, with why."""
    rows = store.conn.execute(
        """SELECT s.*, src.original_filename, src.duration_s AS source_duration
           FROM shots s JOIN sources src ON src.id = s.source_id
           WHERE s.workspace_id = ? AND s.status = 'needs_review'
           ORDER BY src.created_at DESC, s.shot_index
           LIMIT ?""",
        (store.workspace_id, limit),
    ).fetchall()

    queue = []
    for row in rows:
        shot = Shot.from_row(row, store.shot_tags(row["id"]))
        reasons = []
        if row["error_message"]:
            reasons.append(row["error_message"])
        for code in shot.review_reasons:
            if code == "quality_defect":
                reasons.append("quality flags: " + ", ".join(shot.quality_flags))
            elif code == "low_confidence" and shot.confidence is not None:
                reasons.append(f"low confidence ({shot.confidence:.2f})")
            elif code == "low_category_confidence" and shot.category_confidence is not None:
                reasons.append(f"unsure of the folder ({shot.category_confidence:.2f})")
            elif code in REASON_TEXT:
                reasons.append(REASON_TEXT[code])
        if not shot.review_reasons:  # shots flagged before reasons were recorded
            if shot.quality_flags:
                reasons.append("quality flags: " + ", ".join(shot.quality_flags))
            if shot.confidence is not None and shot.confidence < below_confidence:
                reasons.append(f"low confidence ({shot.confidence:.2f})")
        queue.append(
            {
                "shot": shot,
                "filename": row["original_filename"],
                "reasons": reasons or ["flagged for review"],
            }
        )
    return queue


# --------------------------------------------------------------------------
# New folders the model suggested
# --------------------------------------------------------------------------


def approve_folder_proposal(
    config: WorkspaceConfig,
    store: Store,
    proposal_id: int,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    """Create the suggested folder and move the clips that suggested it into it.

    A clip a person has already corrected keeps the folder they chose. Returns what happened:
    {"path", "created", "moved": [shot ids], "sources": [source ids to re-file in Drive]}.
    """
    proposal = store.get_folder_proposal(proposal_id)
    if proposal is None or proposal["status"] != "open":
        raise CorrectionError("That folder suggestion is no longer open.")
    path = proposal["path"]
    parent, _, name = path.rpartition("/")
    if config.taxonomy.find_node(path) is None:
        if config.taxonomy.find_node(parent) is None:
            raise CorrectionError(f"The parent folder {parent!r} no longer exists.")
        config.taxonomy.add_folder(parent, name, proposal.get("note") or "Added from a suggestion.")
        config.save()
        created = True
    else:
        created = False

    moved: list[str] = []
    sources: set[str] = set()
    for shot_id in proposal["shot_ids"]:
        shot = store.get_shot(shot_id)
        if shot is None or _is_corrected(shot):
            continue
        shot.category = path
        shot.secondary_categories = [c for c in shot.secondary_categories if c != path]
        store.update_shot(shot)
        text = store.recompute_search_text(shot.id)
        if embedder is not None and text:
            try:
                store.vectors.upsert(shot.id, embedder.embed_documents([text])[0])
            except Exception as exc:  # noqa: BLE001 - moving the clip matters more than its vector
                log.warning("re-embedding %s failed: %s", shot.id, exc)
        moved.append(shot.id)
        sources.add(shot.source_id)
    store.set_folder_proposal_status(proposal_id, "approved")
    return {"path": path, "created": created, "moved": moved, "sources": sorted(sources)}


def dismiss_folder_proposal(store: Store, proposal_id: int) -> bool:
    proposal = store.get_folder_proposal(proposal_id)
    if proposal is None or proposal["status"] != "open":
        return False
    store.set_folder_proposal_status(proposal_id, "dismissed")
    return True


def _is_corrected(shot: Shot) -> bool:
    return bool((shot.raw_analysis or {}).get("corrected_by_operator"))
