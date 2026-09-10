"""Search screen: a query box, a filter sidebar and a thumbnail grid."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from ...analysis.schema import (
    MOODS,
    SETTINGS,
    USABLE_FOR,
    CameraMove,
    PeopleCount,
    ShotType,
    TimeOfDay,
)
from ...ingest.pipeline import drive_organise_callable
from ...search.filters import SearchFilters
from ...search.query import SearchEngine
from ..app import templates

log = logging.getLogger(__name__)
router = APIRouter()


def _opt_float(value: str | None) -> float | None:
    """An untouched number input posts "", which must mean "no filter"."""
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _available(values: list[str], counts: dict, facet: str) -> list[tuple[str, int]]:
    """Only offer filters the library can actually satisfy, commonest first."""
    present = [(v, counts.get((facet, v), 0)) for v in values]
    return sorted([p for p in present if p[1] > 0], key=lambda kv: (-kv[1], kv[0]))


def _present(counts: dict, facet: str) -> list[tuple[str, int]]:
    """Every value the library holds for a facet, vocabulary or not."""
    values = [(v, n) for (f, v), n in counts.items() if f == facet]
    return sorted(values, key=lambda kv: (-kv[1], kv[0]))


def _featured_name(config) -> str | None:
    person = config.client.featured_person
    return person.split()[0] if person else None


@router.get("/", response_class=HTMLResponse)
async def home() -> RedirectResponse:
    return RedirectResponse("/search")


@router.get("/search", response_class=HTMLResponse)
async def search_page(
    request: Request,
    q: str = "",
    shot_type: list[str] = Query(default=[]),
    camera_movement: list[str] = Query(default=[]),
    mood: list[str] = Query(default=[]),
    emotion: list[str] = Query(default=[]),
    category: list[str] = Query(default=[]),
    setting: list[str] = Query(default=[]),
    people: list[str] = Query(default=[]),
    time_of_day: list[str] = Query(default=[]),
    usable_for: list[str] = Query(default=[]),
    min_duration: str | None = None,
    max_duration: str | None = None,
    clean: bool = False,
    featured: bool = False,
    top_picks: bool = False,
    limit: int = 48,
):
    state = request.app.state.broll
    config = state.config
    store = state.store()
    try:
        engine = SearchEngine(store, state.embedder)
        filters = SearchFilters(
            shot_type=shot_type, camera_movement=camera_movement, mood=mood,
            emotions=emotion, category=category, setting=setting,
            people_count=people, time_of_day=time_of_day, usable_for=usable_for,
            duration_min_s=_opt_float(min_duration),
            duration_max_s=_opt_float(max_duration),
            exclude_flagged=clean,
            featured_person=True if featured else None,
            top_pick=True if top_picks else None,
        )
        results = engine.search(q, filters, limit)
        counts = store.facet_counts()
        context = {
            "request": request,
            "workspace": config,
            "query": q,
            "results": results,
            "featured_name": _featured_name(config),
            "selected": {
                "shot_type": shot_type, "camera_movement": camera_movement,
                "mood": mood, "emotion": emotion, "category": category,
                "setting": setting, "people": people, "time_of_day": time_of_day,
                "usable_for": usable_for, "clean": clean, "featured": featured,
                "top_picks": top_picks,
                "min_duration": _opt_float(min_duration),
                "max_duration": _opt_float(max_duration),
            },
            "facets": {
                "emotion": _present(counts, "emotions"),
                "category": sorted(_present(counts, "category")),
                "shot_type": _available([m.value for m in ShotType], counts, "shot_type"),
                "camera_movement": _available(
                    [m.value for m in CameraMove], counts, "camera_movement"),
                "time_of_day": _available([m.value for m in TimeOfDay], counts, "time_of_day"),
                "people": _available([m.value for m in PeopleCount], counts, "people_count"),
                "mood": _available(list(MOODS), counts, "mood"),
                "setting": _available(list(SETTINGS), counts, "setting"),
                "usable_for": _available(list(USABLE_FOR), counts, "usable_for"),
            },
            "total_shots": store.count_shots(),
        }
    finally:
        store.close()

    template = "partials/results.html" if request.headers.get("hx-request") else "search.html"
    return templates.TemplateResponse(request=request, name=template, context=context)


@router.post("/shots/{shot_id}/top-pick", response_class=HTMLResponse)
async def toggle_top_pick(request: Request, shot_id: str):
    """Star or unstar a shot, and update the Top Picks shortcut in Drive."""
    state = request.app.state.broll
    store = state.store()
    try:
        shot = store.get_shot(shot_id)
        if shot is None:
            raise HTTPException(status_code=404, detail="No such shot.")
        store.set_shot_fields(shot_id, top_pick=int(not shot.top_pick))
        shot = store.get_shot(shot_id)
    finally:
        store.close()

    organise = drive_organise_callable(state.config)
    if organise is not None:
        try:
            await asyncio.to_thread(organise, shot.source_id)
        except Exception as exc:  # the star must stick even if Drive is unreachable
            log.warning("could not update Top Picks in Drive: %s", exc)

    return templates.TemplateResponse(
        request=request, name="partials/star.html", context={"request": request, "shot": shot}
    )
