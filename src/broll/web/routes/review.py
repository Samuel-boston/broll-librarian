"""Review screen: the needs_review queue, at shot level.

A multi-shot file shows exactly which shot needs attention. Corrections are
ground truth, so saving one recomputes the shot's search text and embedding.
"""

from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse

import asyncio
import logging

from ...review import (
    ENUM_OPTIONS,
    CorrectionError,
    apply_correction,
    approve_folder_proposal,
    dismiss_folder_proposal,
    review_queue,
)
from ..app import client_state, templates

log = logging.getLogger(__name__)

router = APIRouter()

EDITABLE = (
    "caption", "setting", "setting_detail", "action", "shot_type", "camera_movement",
    "time_of_day", "colour_profile", "people_count", "pace", "subjects", "mood",
    "emotions", "usable_for", "quality_flags", "tags", "themes", "body_language", "search_phrases",
    "category", "top_pick",
)


def _categories(config) -> list[str]:
    if config.taxonomy.mode != "tree":
        return []
    return [path for path, _ in config.taxonomy.category_leaves()]


def _context(request: Request, state, store, message: str | None = None) -> dict:
    context = {
        "request": request,
        "workspace": state.config,
        "queue": review_queue(store, below_confidence=state.config.ingest.review_below_confidence),
        "options": ENUM_OPTIONS,
        "categories": _categories(state.config),
        "indexed": store.count_shots("indexed"),
        "proposals": store.list_folder_proposals("open"),
    }
    if message:
        context["message"] = message
    return context


async def _refile(state, source_ids: list[str]) -> None:
    """Move these files to their new folders in Drive, if Drive is connected. Best effort."""
    organise = state.drive_organise
    if organise is None:
        return
    for source_id in source_ids:
        try:
            await asyncio.to_thread(organise, source_id)
        except Exception as exc:  # noqa: BLE001 - the correction is saved either way
            log.warning("could not re-file %s in Drive: %s", source_id, exc)


@router.get("/review", response_class=HTMLResponse)
async def review_page(request: Request):
    state = client_state(request)
    store = state.store()
    try:
        context = _context(request, state, store)
    finally:
        store.close()
    return templates.TemplateResponse(request=request, name="review.html", context=context)


@router.post("/review/folders/{proposal_id}/approve", response_class=HTMLResponse)
async def approve_folder(request: Request, proposal_id: int):
    state = client_state(request)
    store = state.store()
    try:
        try:
            done = approve_folder_proposal(state.config, store, proposal_id, state.embedder)
        except CorrectionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await _refile(state, done["sources"])
        context = _context(
            request, state, store,
            message=f"Created {done['path']} and moved {len(done['moved'])} clip(s) into it.",
        )
    finally:
        store.close()
    return templates.TemplateResponse(
        request=request, name="partials/review_queue.html", context=context
    )


@router.post("/review/folders/{proposal_id}/dismiss", response_class=HTMLResponse)
async def dismiss_folder(request: Request, proposal_id: int):
    state = client_state(request)
    store = state.store()
    try:
        dismissed = dismiss_folder_proposal(store, proposal_id)
        context = _context(
            request, state, store,
            message="Suggestion dismissed." if dismissed else "That suggestion was already closed.",
        )
    finally:
        store.close()
    return templates.TemplateResponse(
        request=request, name="partials/review_queue.html", context=context
    )


@router.post("/review/{shot_id}", response_class=HTMLResponse)
async def save_correction(request: Request, shot_id: str, status: str = Form("indexed")):
    state = client_state(request)
    form = await request.form()
    # getlist()[-1]: a checkbox posts a hidden "false" then, if ticked, "true".
    updates = {field: form.getlist(field)[-1] for field in EDITABLE if field in form}

    store = state.store()
    try:
        shot = apply_correction(state.config, store, shot_id, updates, state.embedder, status)
        # A clip confirmed out of the review folder goes to its real folder.
        await _refile(state, [shot.source_id])
        context = _context(
            request, state, store, message=f"Saved {shot_id}. Search text and embedding recomputed."
        )
    except CorrectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        store.close()

    return templates.TemplateResponse(
        request=request, name="partials/review_queue.html", context=context
    )
