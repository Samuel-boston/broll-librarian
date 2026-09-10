"""Library screen: browse the footage by its folders, feelings and recency.

The home screen. Search finds a clip you can describe; browsing is for seeing
what there is - so the client's own folder structure is the main thing here,
not six arbitrary thumbnails.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ...search.browse import children, folder_label, summarise_folders
from ...search.filters import SearchFilters
from ...search.query import SearchEngine
from ..app import templates

router = APIRouter()

PAGE = 60
VIEWS = {
    "top-picks": ("★ Top Picks", {"top_pick": True}),
    "featured": (None, {"featured_person": True}),
    "unsorted": ("Unsorted", {"uncategorised": True}),
    "review": ("Needs review", {"status": ["needs_review"]}),
}


@router.get("/library", response_class=HTMLResponse)
async def library(
    request: Request,
    path: str = "",
    emotion: str = "",
    view: str = "",
    offset: int = 0,
):
    state = request.app.state.broll
    config = state.config
    taxonomy = config.taxonomy
    featured_name = (config.client.featured_person or "").split()[0] or None \
        if config.client.featured_person else None

    store = state.store()
    try:
        tree = taxonomy.mode == "tree"
        folder_paths = ["/".join(p) for p in taxonomy.tree_folders()] if tree else []
        hidden_folders = {taxonomy.guide_folder, taxonomy.top_picks_folder} - {None}
        rows = store.browse_rows()
        summaries = summarise_folders(rows, folder_paths) if tree else {}
        counts = store.facet_counts()

        # What to list: one folder, one feeling, one quick view - or the home page.
        heading = None
        filters: SearchFilters | None = None
        if path:
            filters = SearchFilters(category=[path])
        elif emotion:
            filters = SearchFilters(emotions=[emotion])
            heading = f"Feeling {emotion}"
        elif view in VIEWS:
            title, fields = VIEWS[view]
            filters = SearchFilters(**fields)
            heading = title or f"{featured_name or 'Featured person'} in shot"

        engine = SearchEngine(store, state.embedder)
        if filters is not None:
            clips = engine.browse(filters, PAGE + 1, offset)
            more = len(clips) > PAGE
            clips = clips[:PAGE]
        else:
            clips = engine.browse(SearchFilters(), 12)
            more = False

        level = [summaries[p] for p in children(path, folder_paths) if p not in hidden_folders]
        crumbs: list[tuple[str, str]] = []
        walked: list[str] = []
        for part in path.split("/") if path else []:
            walked.append(part)
            crumbs.append(("/".join(walked), folder_label(part)[1]))
        current = summaries.get(path)

        context = {
            "request": request,
            "workspace": config,
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
                "review": store.count_shots("needs_review"),
            },
        }
    finally:
        store.close()
    return templates.TemplateResponse(request=request, name="library.html", context=context)
