"""Search screen: a query box, a filter sidebar and a thumbnail grid."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
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
from ...search.filters import SearchFilters
from ...search.query import SearchEngine
from ..app import templates

router = APIRouter()


def _filters(
    shot_type: list[str],
    camera_movement: list[str],
    mood: list[str],
    setting: list[str],
    people: list[str],
    time_of_day: list[str],
    usable_for: list[str],
    min_duration: float | None,
    max_duration: float | None,
    clean: bool,
) -> SearchFilters:
    return SearchFilters(
        shot_type=shot_type, camera_movement=camera_movement, mood=mood,
        setting=setting, people_count=people, time_of_day=time_of_day,
        usable_for=usable_for, duration_min_s=min_duration,
        duration_max_s=max_duration, exclude_flagged=clean,
    )


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
    setting: list[str] = Query(default=[]),
    people: list[str] = Query(default=[]),
    time_of_day: list[str] = Query(default=[]),
    usable_for: list[str] = Query(default=[]),
    # These arrive from the form as strings, and an untouched number input
    # sends "" - which would 422 if it were typed as `float | None`.
    min_duration: str | None = None,
    max_duration: str | None = None,
    clean: bool = False,
    limit: int = 48,
):
    state = request.app.state.broll
    store = state.store()
    try:
        engine = SearchEngine(store, state.embedder)
        filters = _filters(shot_type, camera_movement, mood, setting, people,
                           time_of_day, usable_for, _opt_float(min_duration),
                           _opt_float(max_duration), clean)
        results = engine.search(q, filters, limit)
        counts = store.facet_counts()
        context = {
            "request": request,
            "workspace": state.config,
            "query": q,
            "results": results,
            "selected": {
                "shot_type": shot_type, "camera_movement": camera_movement,
                "mood": mood, "setting": setting, "people": people,
                "time_of_day": time_of_day, "usable_for": usable_for,
                "clean": clean, "min_duration": _opt_float(min_duration),
                "max_duration": _opt_float(max_duration),
            },
            "facets": {
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


def _opt_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _available(values: list[str], counts: dict, facet: str) -> list[tuple[str, int]]:
    """Only offer filters the library can actually satisfy, commonest first."""
    present = [(v, counts.get((facet, v), 0)) for v in values]
    return sorted([p for p in present if p[1] > 0], key=lambda kv: (-kv[1], kv[0]))
