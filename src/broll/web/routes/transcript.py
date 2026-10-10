"""Transcript screen: paste a transcript, see suggestions per beat, export."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response

from ... import kinds
from ...analysis.providers.registry import get_text_provider
from ...search.filters import SearchFilters
from ...search.query import SearchEngine
from ...transcript.exporters import csv_export, edl, fcp7xml
from ...transcript.exporters.base import Timeline, build_timeline
from ...transcript.matcher import BeatMatch, TranscriptMatcher
from ...transcript.parser import join_beats, parse_and_segment, split_beat
from ..app import client_state, templates

router = APIRouter()

EXPORT_TYPES = {
    "xml": ("application/xml", "xml"),
    "edl": ("text/plain", "edl"),
    "csv": ("text/csv", "csv"),
}


@dataclass
class TranscriptRun:
    """Held in memory for the life of the process - a run is cheap to redo."""

    id: str
    name: str
    matches: list[BeatMatch] = field(default_factory=list)
    media: str = "both"
    # What it takes to match a beat again after a person splits or joins it.
    config: object = None
    filters: object = None
    rerank: bool = False

    def timeline(self, config) -> Timeline:
        return build_timeline(self.matches, config, name=self.name)


@router.get("/transcript", response_class=HTMLResponse)
async def transcript_page(request: Request):
    state = client_state(request)
    return templates.TemplateResponse(
        request=request,
        name="transcript.html",
        context={"request": request, "workspace": state.config, "run": None},
    )


@router.post("/transcript", response_class=HTMLResponse)
async def match_transcript(
    request: Request,
    text: str = Form(default=""),
    filename: str = Form(default="transcript.txt"),
    rerank: bool = Form(default=True),
    media: list[str] = Form(default=[]),
    cut: str = Form(default="phrase"),
    min_s: float | None = Form(default=None),
    max_s: float | None = Form(default=None),
    per_line: int | None = Form(default=None),
    upload: UploadFile | None = File(default=None),
):
    state = client_state(request)
    if upload is not None and upload.filename:
        text = (await upload.read()).decode("utf-8", errors="replace")
        filename = upload.filename
    if not text.strip():
        raise HTTPException(status_code=400, detail="Paste a transcript or upload a file.")

    config = state.config
    if per_line and 1 <= per_line <= 10 and per_line != config.transcript.suggestions_per_beat:
        config = config.model_copy(deep=True)  # this run only: the saved setting is untouched
        config.transcript.suggestions_per_beat = per_line
    beats = parse_and_segment(
        text, filename, config.transcript.words_per_minute,
        min_s if min_s and min_s > 0 else config.transcript.beat_min_s,
        max_s if max_s and max_s > 0 else config.transcript.beat_max_s,
        granularity="phrase" if cut == "phrase" else "sentence",
    )

    store = state.store()
    try:
        engine = SearchEngine(store, state.embedder, featured_person=config.client.featured_person)
        text_provider = None
        if rerank:
            try:
                text_provider = get_text_provider(config)
            except Exception:  # a missing key must not break the screen
                text_provider = None
        # Videos and images both by default; untick one to cut from the other alone.
        sides = [k for k in kinds.KINDS if k in {kinds.clean(m) for m in media}]
        media = sides[0] if len(sides) == 1 else "both"
        wanted = SearchFilters(exclude_flagged=True, media_kind=sides if len(sides) == 1 else [])
        matches = await TranscriptMatcher(config, engine, text_provider, wanted).match(beats)
    finally:
        store.close()

    run = TranscriptRun(id=uuid.uuid4().hex[:12], name=filename.rsplit(".", 1)[0], matches=matches, media=media,
                         config=config, filters=wanted, rerank=bool(text_provider))
    state.runs[run.id] = run

    return templates.TemplateResponse(
        request=request,
        name="partials/beats.html",
        context=_run_context(request, state, run),
    )


async def _rematch(request: Request, state, run: TranscriptRun, new_beats: list, keep: dict[int, BeatMatch]):
    """Rebuild a run's list after a split or join: only the new beats are matched again."""
    from ...transcript.matcher import TranscriptMatcher

    usage: dict[str, int] = {}
    for match in keep.values():
        if match.chosen:
            usage[match.chosen.shot.id] = usage.get(match.chosen.shot.id, 0) + 1
    store = state.store()
    try:
        engine = SearchEngine(store, state.embedder, featured_person=run.config.client.featured_person)
        provider = None
        if run.rerank:
            try:
                provider = get_text_provider(run.config)
            except Exception:  # noqa: BLE001
                provider = None
        matcher = TranscriptMatcher(run.config, engine, provider, run.filters)
        matches = []
        for number, beat in enumerate(new_beats):
            old = keep.get(id(beat))
            beat.index = number
            matches.append(old if old else await matcher.match_one(beat, usage))
            if old is None and matches[-1].chosen:
                usage[matches[-1].chosen.shot.id] = usage.get(matches[-1].chosen.shot.id, 0) + 1
    finally:
        store.close()
    run.matches = matches


@router.post("/transcript/{run_id}/split", response_class=HTMLResponse)
async def split_one(request: Request, run_id: str, beat: int = Form(...), start: int = Form(...),
                    end: int = Form(...)):
    """Make the selected words of a beat a beat of their own (and the rest before and after, theirs)."""
    state = client_state(request)
    run = _get_run(state, run_id)
    current = next((m for m in run.matches if m.beat.index == beat), None)
    if current is None:
        raise HTTPException(status_code=404, detail="No such beat.")
    pieces = split_beat(current.beat, start, end)
    if len(pieces) < 2:
        raise HTTPException(status_code=400, detail="Select part of the line, not all of it.")
    new_beats, keep = [], {}
    for match in run.matches:
        if match is current:
            new_beats.extend(pieces)
        else:
            new_beats.append(match.beat)
            keep[id(match.beat)] = match
    await _rematch(request, state, run, new_beats, keep)
    return templates.TemplateResponse(request=request, name="partials/beats.html",
                                      context=_run_context(request, state, run))


@router.post("/transcript/{run_id}/join", response_class=HTMLResponse)
async def join_one(request: Request, run_id: str, beat: int = Form(...)):
    """Put a beat and the one after it back together as one B-roll."""
    state = client_state(request)
    run = _get_run(state, run_id)
    spot = next((i for i, m in enumerate(run.matches) if m.beat.index == beat), None)
    if spot is None or spot + 1 >= len(run.matches):
        raise HTTPException(status_code=400, detail="There is no beat after this one.")
    joined = join_beats(run.matches[spot].beat, run.matches[spot + 1].beat)
    new_beats, keep = [], {}
    for i, match in enumerate(run.matches):
        if i == spot:
            new_beats.append(joined)
        elif i != spot + 1:
            new_beats.append(match.beat)
            keep[id(match.beat)] = match
    await _rematch(request, state, run, new_beats, keep)
    return templates.TemplateResponse(request=request, name="partials/beats.html",
                                      context=_run_context(request, state, run))


CARDS = 12  # how many options a person's own search lays out


def _beat_response(request: Request, state, run: TranscriptRun, match: BeatMatch):
    """Just the one line that changed, plus the totals: the page stays where it is."""
    context = _run_context(request, state, run)
    context["match"] = match
    return templates.TemplateResponse(request=request, name="partials/beat_update.html", context=context)


@router.post("/transcript/{run_id}/search", response_class=HTMLResponse)
async def search_for_beat(request: Request, run_id: str, beat: int = Form(...), q: str = Form("")):
    """A person's own search for one line, when the suggestions were no good."""
    from ...transcript.matcher import Suggestion

    state = client_state(request)
    run = _get_run(state, run_id)
    match = next((m for m in run.matches if m.beat.index == beat), None)
    if match is None:
        raise HTTPException(status_code=404, detail="No such beat.")
    q = q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Type what you want to find.")
    store = state.store()
    try:
        engine = SearchEngine(store, state.embedder, featured_person=run.config.client.featured_person)
        found = engine.search(q, run.filters, CARDS)
        near = False
        if len(found) < CARDS // 2:  # a person hunting wants options: add near matches below the sure ones
            seen = {r.shot.id for r in found}
            extra = [r for r in engine.search(q, run.filters, CARDS, strict=False) if r.shot.id not in seen]
            found, near = [*found, *extra][:CARDS], bool(extra)
        suggestions = [Suggestion(shot=r.shot, source=r.source, reason=f"Your search: {q}",
                                  confidence=0.5, score=r.score) for r in found]
    finally:
        store.close()
    match.query = q
    if suggestions:
        n = run.config.transcript.suggestions_per_beat
        match.suggestions, match.alternatives = suggestions[:n], suggestions
        match.no_good_match, match.missing_footage, match.carousel = False, None, True
        match.note = "Includes near matches." if near else None
    else:
        match.note = f"Nothing found for “{q}”. Try other words."
    return _beat_response(request, state, run, match)


@router.post("/transcript/{run_id}/swap", response_class=HTMLResponse)
async def swap_suggestion(request: Request, run_id: str, beat: int = Form(...),
                          shot_id: str = Form(...)):
    state = client_state(request)
    run = _get_run(state, run_id)
    match = next((m for m in run.matches if m.beat.index == beat), None)
    if match is None or not match.choose(shot_id):
        raise HTTPException(status_code=404, detail="No such beat or candidate.")
    return _beat_response(request, state, run, match)


@router.get("/transcript/{run_id}/export/{fmt}")
async def export(request: Request, run_id: str, fmt: str):
    state = client_state(request)
    run = _get_run(state, run_id)
    if fmt not in EXPORT_TYPES:
        raise HTTPException(status_code=404, detail=f"Unknown format {fmt!r}.")

    timeline = run.timeline(state.config)
    if fmt == "xml":
        body = fcp7xml.build(timeline)
    elif fmt == "edl":
        body = edl.build(timeline)
    else:
        body = csv_export.build(run.matches, timeline)

    media_type, extension = EXPORT_TYPES[fmt]
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{run.name}.{extension}"'},
    )


def _get_run(state, run_id: str) -> TranscriptRun:
    run = state.runs.get(run_id)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail="That run has expired - runs live in memory. Match the transcript again.",
        )
    return run


def _run_context(request: Request, state, run: TranscriptRun) -> dict:
    timeline = run.timeline(state.config)
    return {
        "request": request,
        "workspace": state.config,
        "run": run,
        "matches": run.matches,
        "timeline": timeline,
        "gaps": timeline.gaps,
    }
