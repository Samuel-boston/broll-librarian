"""Transcript screen: paste a transcript, see suggestions per beat, export."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response

from ...analysis.providers.registry import get_text_provider
from ...search.query import SearchEngine
from ...transcript.exporters import csv_export, edl, fcp7xml
from ...transcript.exporters.base import Timeline, build_timeline
from ...transcript.matcher import BeatMatch, TranscriptMatcher
from ...transcript.parser import parse_and_segment
from ..app import templates

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

    def timeline(self, config) -> Timeline:
        return build_timeline(self.matches, config, name=self.name)


@router.get("/transcript", response_class=HTMLResponse)
async def transcript_page(request: Request):
    state = request.app.state.broll
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
    upload: UploadFile | None = File(default=None),
):
    state = request.app.state.broll
    if upload is not None and upload.filename:
        text = (await upload.read()).decode("utf-8", errors="replace")
        filename = upload.filename
    if not text.strip():
        raise HTTPException(status_code=400, detail="Paste a transcript or upload a file.")

    config = state.config
    beats = parse_and_segment(
        text, filename, config.transcript.words_per_minute,
        config.transcript.beat_min_s, config.transcript.beat_max_s,
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
        matches = await TranscriptMatcher(config, engine, text_provider).match(beats)
    finally:
        store.close()

    run = TranscriptRun(id=uuid.uuid4().hex[:12], name=filename.rsplit(".", 1)[0], matches=matches)
    state.runs[run.id] = run

    return templates.TemplateResponse(
        request=request,
        name="partials/beats.html",
        context=_run_context(request, state, run),
    )


@router.post("/transcript/{run_id}/swap", response_class=HTMLResponse)
async def swap_suggestion(request: Request, run_id: str, beat: int = Form(...),
                          shot_id: str = Form(...)):
    state = request.app.state.broll
    run = _get_run(state, run_id)
    match = next((m for m in run.matches if m.beat.index == beat), None)
    if match is None or not match.choose(shot_id):
        raise HTTPException(status_code=404, detail="No such beat or candidate.")
    return templates.TemplateResponse(
        request=request,
        name="partials/beats.html",
        context=_run_context(request, state, run),
    )


@router.get("/transcript/{run_id}/export/{fmt}")
async def export(request: Request, run_id: str, fmt: str):
    state = request.app.state.broll
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
