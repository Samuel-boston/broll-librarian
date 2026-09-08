"""Review screen: the needs_review queue, at shot level.

A multi-shot file shows exactly which shot needs attention. Corrections are
ground truth, so saving one recomputes the shot's search text and embedding.
"""

from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse

from ...review import ENUM_OPTIONS, CorrectionError, apply_correction, review_queue
from ..app import templates

router = APIRouter()

EDITABLE = (
    "caption", "setting", "setting_detail", "action", "shot_type", "camera_movement",
    "time_of_day", "colour_profile", "people_count", "pace", "subjects", "mood",
    "usable_for", "quality_flags", "tags",
)


@router.get("/review", response_class=HTMLResponse)
async def review_page(request: Request):
    state = request.app.state.broll
    store = state.store()
    try:
        context = {
            "request": request,
            "workspace": state.config,
            "queue": review_queue(store),
            "options": ENUM_OPTIONS,
            "indexed": store.count_shots("indexed"),
        }
    finally:
        store.close()
    return templates.TemplateResponse(request=request, name="review.html", context=context)


@router.post("/review/{shot_id}", response_class=HTMLResponse)
async def save_correction(request: Request, shot_id: str, status: str = Form("indexed")):
    state = request.app.state.broll
    form = await request.form()
    updates = {field: form[field] for field in EDITABLE if field in form}

    store = state.store()
    try:
        apply_correction(state.config, store, shot_id, updates, state.embedder, status)
        context = {
            "request": request,
            "workspace": state.config,
            "queue": review_queue(store),
            "options": ENUM_OPTIONS,
            "indexed": store.count_shots("indexed"),
            "message": f"Saved {shot_id}. Search text and embedding recomputed.",
        }
    except CorrectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        store.close()

    return templates.TemplateResponse(
        request=request, name="partials/review_queue.html", context=context
    )
