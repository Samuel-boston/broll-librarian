"""Files the library did not index, and why. See broll.attention."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ... import attention
from ...ingest.limits import GB, human_duration
from ..app import client_state, templates

log = logging.getLogger(__name__)
router = APIRouter()


def _items(store) -> list[dict]:
    out = []
    for item in store.list_attention("open"):
        item["heading"] = attention.heading(item["kind"])
        item["advice"] = attention.advice(item["kind"])
        item["size_text"] = f"{item['size_bytes'] / GB:.1f} GB" if item["size_bytes"] else ""
        item["duration_text"] = human_duration(item["duration_s"]) if item["duration_s"] else ""
        # A file that is too big for the disk cannot be forced through, however much we'd like to.
        item["can_force"] = item["kind"] in ("too_long", "analysis_failed", "download_incomplete", "unreadable")
        out.append(item)
    return out


def _context(request: Request, state, store, message: str | None = None, error: str | None = None) -> dict:
    items = _items(store)
    groups: dict[str, list[dict]] = {}
    for item in items:
        groups.setdefault(item["heading"], []).append(item)
    return {
        "request": request,
        "workspace": state.config,
        "groups": groups,
        "total": len(items),
        "message": message,
        "error": error,
        "limits": state.config.ingest,
        "limit_text": human_duration(state.config.ingest.max_duration_s) if state.config.ingest.max_duration_s else "",
    }


@router.get("/attention", response_class=HTMLResponse)
async def attention_page(request: Request):
    state = client_state(request)
    store = state.store()
    try:
        context = _context(request, state, store)
    finally:
        store.close()
    return templates.TemplateResponse(request=request, name="attention.html", context=context)


@router.get("/attention/badge", response_class=HTMLResponse)
async def attention_badge(request: Request):
    """The nav link, with a count when something is waiting."""
    state = client_state(request)
    store = state.store()
    try:
        total = sum(store.attention_counts().values())
    finally:
        store.close()
    badge = f' <span class="tag tag-warn">{total}</span>' if total else ""
    return HTMLResponse(f"Needs attention{badge}")


@router.post("/attention/{item_id}/index", response_class=HTMLResponse)
async def index_anyway(request: Request, item_id: int):
    state = client_state(request)
    store = state.store()
    try:
        item = store.get_attention(item_id)
        if item is None:
            context = _context(request, state, store, error="That file is no longer on the list.")
        else:
            attention.requeue(state.config, store, item_id, forced=True)
            context = _context(
                request, state, store,
                message=f"Queued {item['filename']}. It will be indexed with the length limit lifted.",
            )
    finally:
        store.close()
    return templates.TemplateResponse(request=request, name="partials/attention_list.html", context=context)


@router.post("/attention/{item_id}/dismiss", response_class=HTMLResponse)
async def dismiss_one(request: Request, item_id: int):
    state = client_state(request)
    store = state.store()
    try:
        attention.dismiss(store, item_id, state.config)
        context = _context(request, state, store, message="Dismissed. It won't be queued again.")
    finally:
        store.close()
    return templates.TemplateResponse(request=request, name="partials/attention_list.html", context=context)


@router.post("/attention/sync", response_class=HTMLResponse)
async def sync_to_drive(request: Request):
    """Put a shortcut to each listed file in a "_Needs Attention" folder in Drive."""
    state = client_state(request)

    def work() -> int:
        from ...drive.session import DriveSession

        # The app's own session when there is one, so this takes turns with the worker's Drive calls.
        session = getattr(state.drive_organise, "__self__", None) or DriveSession(state.config)
        return session.mirror_attention()

    store = state.store()
    try:
        try:
            created = await asyncio.to_thread(work)
            context = _context(
                request, state, store,
                message=(f"Added {created} shortcut(s) to the \"{attention.DRIVE_FOLDER}\" folder in Drive."
                         if created else "Drive already has a shortcut for everything listed."),
            )
        except Exception as exc:  # noqa: BLE001 - say what happened, don't crash the page
            context = _context(request, state, store, error=f"Couldn't update Drive: {exc}")
    finally:
        store.close()
    return templates.TemplateResponse(request=request, name="partials/attention_list.html", context=context)
