"""Library screen: browse the footage by its folders, feelings and recency.

The home screen is two doors, Videos and Images, and nothing else. Behind each door is the same folder
tree, holding only that kind: a clip and a photograph are never listed together. Search finds a clip
you can describe; browsing is for seeing what there is, so the client's own folder structure is the
main thing here.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from ...library_admin import clear_library, remove_source, running_jobs
from ...search.browse import children, folder_label, summarise_folders
from ...search.filters import SearchFilters
from ...search.query import SearchEngine
from ... import kinds
from ..app import client_state, templates, url_with

router = APIRouter()

PAGE = 60
#: Quick views inside one kind. Which kind is not part of them: the page already is one kind.
VIEWS = {
    "top-picks": ("★ Top Picks", {"top_pick": True}),
    "featured": (None, {"featured_person": True}),
    "unsorted": ("Unsorted", {"uncategorised": True}),
    "review": ("Needs review", {"status": ["needs_review"]}),
}
#: Links from before videos and images were kept apart. They landed on a mixed list; now they land on a side.
_LEGACY_VIEWS = {"photos": "image", "videos": "video"}


def _mosaic(rows: list[dict], limit: int = 4) -> list[str]:
    return [r["id"] for r in rows if r["has_thumbnail"]][:limit]


@router.get("/library", response_class=HTMLResponse)
async def library(
    request: Request,
    kind: str = "",
    path: str = "",
    emotion: str = "",
    view: str = "",
    offset: int = 0,
):
    state = client_state(request)
    config = state.config
    store = state.store()
    kind = kinds.clean(kind)
    try:
        if not kind:
            if path or emotion or view:
                # An old link. Send it to a side, rather than to the mixed list it used to show.
                wanted = _LEGACY_VIEWS.get(view) or "video"
                return RedirectResponse(
                    url_with(request, kind=wanted, view=None if view in _LEGACY_VIEWS else (view or None)),
                    status_code=303,
                )
            rows = store.browse_rows()
            tiles = []
            for side in kinds.KINDS:
                mine = [r for r in rows if r["media_kind"] == side]
                tiles.append({"kind": side, "label": kinds.LABELS[side], "count": len(mine),
                              "thumbnails": _mosaic(mine)})
            context = {"request": request, "workspace": config, "kind": "", "tiles": tiles,
                       "empty": not rows}
            return templates.TemplateResponse(request=request, name="library.html", context=context)

        taxonomy = config.taxonomy
        featured_name = (config.client.featured_person or "").split()[0] or None \
            if config.client.featured_person else None
        tree = taxonomy.mode == "tree"
        folder_paths = ["/".join(p) for p in taxonomy.tree_folders()] if tree else []
        hidden_folders = {taxonomy.guide_folder, taxonomy.top_picks_folder} - {None}
        rows = store.browse_rows(kind)
        summaries = summarise_folders(rows, folder_paths) if tree else {}
        counts = store.facet_counts(kind)
        others = kinds.other(kind)
        other_rows = store.browse_rows(others)

        # What to list: one folder, one feeling, one quick view - or the front of this side.
        heading = None
        filters: SearchFilters | None = None
        if path:
            filters = SearchFilters(category=[path], media_kind=[kind])
        elif emotion:
            filters = SearchFilters(emotions=[emotion], media_kind=[kind])
            heading = f"Feeling {emotion}"
        elif view in VIEWS:
            title, fields = VIEWS[view]
            filters = SearchFilters(**fields, media_kind=[kind])
            heading = title or f"{featured_name or 'Featured person'} in shot"

        engine = SearchEngine(store, state.embedder)
        if filters is not None:
            clips = engine.browse(filters, PAGE + 1, offset)
            more = len(clips) > PAGE
            clips = clips[:PAGE]
        else:
            clips = engine.browse(SearchFilters(media_kind=[kind]), 12)
            more = False

        level = [summaries[p] for p in children(path, folder_paths) if p not in hidden_folders]
        crumbs: list[tuple[str, str]] = []
        walked: list[str] = []
        for part in path.split("/") if path else []:
            walked.append(part)
            crumbs.append(("/".join(walked), folder_label(part)[1]))
        current = summaries.get(path)

        # The same place on the other side, so a person can hop across without losing their spot.
        if path and tree:
            there = summarise_folders(other_rows, folder_paths).get(path)
            other_count = there.count if there else 0
        elif not (path or emotion or view):
            other_count = len(other_rows)
        else:
            other_count = None
        here_count = current.count if (path and current) else (len(rows) if not (path or emotion or view) else None)

        context = {
            "request": request,
            "workspace": config,
            "kind": kind,
            "kind_label": kinds.LABELS[kind],
            "other": others,
            "other_label": kinds.LABELS[others],
            "other_count": other_count,
            "here_count": here_count,
            "noun": kinds.SINGULAR[kind],
            "tree": tree,
            "path": path,
            "crumbs": crumbs,
            "current": current,
            "folders": level,
            "emotion": emotion,
            "view": view,
            "heading": heading,
            "clips": clips,
            "offset": offset,
            "more": more,
            "page": PAGE,
            "featured_name": featured_name,
            "emotions": sorted(
                ((v, n) for (f, v), n in counts.items() if f == "emotions"),
                key=lambda kv: (-kv[1], kv[0]),
            ),
            "settings": sorted(
                ((v, n) for (f, v), n in counts.items() if f == "setting"),
                key=lambda kv: (-kv[1], kv[0]),
            ),
            "quick": {
                "total": len(rows),
                "top_picks": sum(1 for r in rows if r["top_pick"]),
                "featured": sum(1 for r in rows if r["featured"]),
                "unsorted": sum(1 for r in rows if not r["category"]),
                "review": store.count_shots("needs_review", media_kind=kind),
            },
        }
    finally:
        store.close()
    return templates.TemplateResponse(request=request, name="library.html", context=context)


def _alert(text: str) -> Response:
    """Nothing on the page changes; the browser shows the message."""
    return Response(status_code=200, headers={"HX-Reswap": "none", "HX-Trigger": json.dumps({"broll-alert": text})})


@router.post("/sources/{source_id}/delete", response_class=HTMLResponse)
async def delete_source(request: Request, source_id: str):
    """Remove one file, and every shot from it, from the library. Drive is not touched."""
    state = client_state(request)

    def work():
        # The database connection belongs to one thread, so it is opened where it is used.
        store = state.store()
        try:
            return remove_source(state.config, store, source_id)
        finally:
            store.close()

    result = await asyncio.to_thread(work)
    if result is None:
        raise HTTPException(status_code=404, detail="That file is no longer in the library.")
    headers = {}
    if result.dashboard_error:
        headers["HX-Trigger"] = json.dumps({"broll-alert": f"Removed, but the dashboard wasn't updated: {result.dashboard_error}"})
    # The card is swapped for nothing, so it disappears.
    return HTMLResponse("", headers=headers)


@router.post("/library/delete-all")
async def delete_everything(request: Request):
    """Empty the whole library. The browser asks for the word DELETE first; Drive is not touched."""
    if request.headers.get("hx-prompt", "").strip() != "DELETE":
        return _alert("Nothing was deleted. You have to type DELETE exactly.")
    state = client_state(request)

    def work():
        store = state.store()
        try:
            if running_jobs(store):
                return None
            return clear_library(state.config, store)
        finally:
            store.close()

    result = await asyncio.to_thread(work)
    if result is None:
        return _alert("A file is being indexed right now. Let it finish, or clear the queue first, then try again.")
    note = f" The dashboard wasn't updated: {result.dashboard_error}" if result.dashboard_error else ""
    return Response(
        status_code=200,
        headers={
            "HX-Redirect": "/library",
            "HX-Trigger": json.dumps({"broll-alert": f"Deleted {result.sources} file(s) and {result.shots} shot(s) from the library. Drive was not touched.{note}"}),
        },
    )
