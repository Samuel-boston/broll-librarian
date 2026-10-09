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
from .config import CategoryNode, WorkspaceConfig
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
    shot.raw_analysis = {**(shot.raw_analysis or {}), "corrected_by_operator": True}

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
        # Approving the folder is a person saying where this clip belongs: the doubt about its
        # folder is over. Without this it would stay filed in the review folder.
        shot.review_reasons = [
            r for r in shot.review_reasons if r not in ("low_category_confidence", "category_unmatched")
        ]
        if shot.status == "needs_review" and not shot.review_reasons and not shot.error_message:
            shot.status = "indexed"
        store.update_shot(shot)
        store.recompute_source_status(shot.source_id)
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


# --------------------------------------------------------------------------
# Changing the client's folder tree
# --------------------------------------------------------------------------


def _under(path: str | None, prefix: str) -> bool:
    return bool(path) and (path == prefix or path.startswith(prefix + "/"))


def _swap_prefix(path: str, old: str, new: str) -> str:
    return new + path[len(old):]


def rename_folder(
    config: WorkspaceConfig,
    store: Store,
    path: str,
    new_name: str,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    """Rename a folder in the client's tree, and carry every clip filed in or under it along.

    Clips remember their folder by path, so a rename that only touched the tree would leave them
    pointing at a folder that no longer exists. Also updates folder suggestions and the search text of
    the clips that moved. In Drive nothing is renamed: the next `broll organise` files each clip under the
    new name, and the old, now empty, folder is left for a person to delete.

    Returns {"old", "new", "clips"}.
    """
    node = config.taxonomy.find_node(path)
    if node is None:
        raise CorrectionError(f"There is no folder {path!r}.")
    new_name = new_name.strip().strip("/")
    if not new_name or "/" in new_name:
        raise CorrectionError("A folder name can't be empty or contain a slash.")
    parent, _, _old_name = path.rpartition("/")
    new_path = f"{parent}/{new_name}" if parent else new_name
    if new_path == path:
        return {"old": path, "new": new_path, "clips": 0}
    if config.taxonomy.find_node(new_path) is not None:
        raise CorrectionError(f"There is already a folder {new_path!r}.")

    node.name = new_name
    config.save()

    moved = 0
    rows = store.conn.execute(
        "SELECT id, category, secondary_categories_json FROM shots WHERE workspace_id = ?",
        (store.workspace_id,),
    ).fetchall()
    for row in rows:
        category = row["category"]
        secondary = json.loads(row["secondary_categories_json"] or "[]")
        if not (_under(category, path) or any(_under(c, path) for c in secondary)):
            continue
        if _under(category, path):
            category = _swap_prefix(category, path, new_path)
        secondary = [_swap_prefix(c, path, new_path) if _under(c, path) else c for c in secondary]
        store.set_shot_fields(row["id"], category=category, secondary_categories_json=json.dumps(secondary))
        text = store.recompute_search_text(row["id"])
        if embedder is not None and text:
            try:
                store.vectors.upsert(row["id"], embedder.embed_documents([text])[0])
            except Exception as exc:  # noqa: BLE001 - the rename matters more than the vector
                log.warning("re-embedding %s after a folder rename failed: %s", row["id"], exc)
        moved += 1
    for proposal in store.conn.execute(
        "SELECT id, path FROM folder_proposals WHERE workspace_id = ?", (store.workspace_id,)
    ).fetchall():
        if _under(proposal["path"], path):
            store.conn.execute(
                "UPDATE folder_proposals SET path = ? WHERE id = ?",
                (_swap_prefix(proposal["path"], path, new_path), proposal["id"]),
            )
    return {"old": path, "new": new_path, "clips": moved}


def add_folder(config: WorkspaceConfig, parent: str, name: str, note: str = "") -> str:
    """Add a folder under `parent` ("" for the top level). Returns its path."""
    name = name.strip().strip("/")
    if not name or "/" in name:
        raise CorrectionError("A folder name can't be empty or contain a slash.")
    if parent:
        if config.taxonomy.find_node(parent) is None:
            raise CorrectionError(f"There is no folder {parent!r} to put it in.")
        path = f"{parent}/{name}"
    else:
        path = name
    if config.taxonomy.find_node(path) is not None:
        raise CorrectionError(f"There is already a folder {path!r}.")
    if parent:
        config.taxonomy.add_folder(parent, name, note)
    else:
        config.taxonomy.tree.append(CategoryNode(name=name, description=note))
    config.save()
    return path


def set_folder_note(config: WorkspaceConfig, path: str, note: str) -> None:
    """Change what a folder is for. The model reads this note when it files a clip."""
    node = config.taxonomy.find_node(path)
    if node is None:
        raise CorrectionError(f"There is no folder {path!r}.")
    node.description = note.strip()
    config.save()


def remove_folder(
    config: WorkspaceConfig,
    store: Store,
    path: str,
    move_to: str | None = None,
) -> dict[str, Any]:
    """Take a folder out of the client's tree.

    Clips filed in it go to `move_to` when one is given. Otherwise they are left without a folder and
    flagged for review, so a person (or `broll reanalyse`) files them again: a clip is never left
    pointing at a folder that no longer exists. Nothing is deleted in Drive; the next `broll organise`
    re-files the clips, and the old folder is left for a person to delete.

    Refuses a folder with folders inside it: remove or move those first.
    """
    node = config.taxonomy.find_node(path)
    if node is None:
        raise CorrectionError(f"There is no folder {path!r}.")
    if node.children:
        raise CorrectionError(f"{path!r} has folders inside it. Remove those first.")
    if move_to is not None:
        target = config.taxonomy.find_node(move_to)
        if target is None or not target.accepts_clips() or move_to == path:
            raise CorrectionError(f"{move_to!r} is not a folder clips can be filed in.")

    parent_path, _, name = path.rpartition("/")
    siblings = config.taxonomy.find_node(parent_path).children if parent_path else config.taxonomy.tree
    siblings[:] = [n for n in siblings if n.name != name]
    config.save()

    moved = 0
    for row in store.conn.execute(
        "SELECT id, category, secondary_categories_json FROM shots WHERE workspace_id = ?",
        (store.workspace_id,),
    ).fetchall():
        secondary = json.loads(row["secondary_categories_json"] or "[]")
        if row["category"] != path and path not in secondary:
            continue
        shot = store.get_shot(row["id"])
        secondary = [c for c in secondary if c != path]
        if row["category"] == path:
            shot.category = move_to
            if move_to is None:
                shot.review_reasons = [*shot.review_reasons, "category_unmatched"]
                shot.status = "needs_review"
        elif move_to and move_to != shot.category and move_to not in secondary and len(secondary) < 2:
            secondary.append(move_to)
        shot.secondary_categories = secondary
        store.update_shot(shot)
        store.recompute_source_status(shot.source_id)
        moved += 1
    for proposal in store.list_folder_proposals("open"):
        if _under(proposal["path"], path):
            store.set_folder_proposal_status(proposal["id"], "dismissed")
    return {"removed": path, "clips": moved, "moved_to": move_to}
