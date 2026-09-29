"""A JSON API, for an agent rather than a person.

The web screens are for browsing; this is what an editing agent calls when it
is cutting a video and needs to know what footage exists. Everything here is
read-mostly and cheap: no vision calls, no Drive writes, nothing that costs
money except an explicitly requested rerank.

The shortlist endpoint is the important one. It does the mechanical half of
B-roll selection - segment the script, search each beat, drop what is too short
or already used - and hands back candidates with their descriptions. The
judgement half, which clip and for how long, belongs to the agent reading it.

Auth: set BROLL_API_TOKEN and send it as `Authorization: Bearer <token>` or
`X-Broll-Token`. With no token set, only requests from the loopback address are
served, which is the normal case for an agent on the same machine.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ...db.store import Store
from ...ingest.scanner import media_kind, scan_local
from ...jobs.queue import enqueue_files, queue_stats
from ...search.filters import SearchFilters
from ...search.query import SearchEngine
from ...transcript.parser import parse_and_segment
from ..app import client_state

router = APIRouter(prefix="/api", tags=["api"])

# "testclient" is what Starlette's in-process test client reports; a real
# server takes this from the socket, so it can never arrive over a network.
LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}


def _authorise(request: Request) -> None:
    token = os.environ.get("BROLL_API_TOKEN", "").strip()
    if token:
        header = request.headers.get("authorization", "")
        supplied = header[7:].strip() if header.lower().startswith("bearer ") else ""
        supplied = supplied or request.headers.get("x-broll-token", "").strip()
        if supplied != token:
            raise HTTPException(status_code=401, detail="Bad or missing API token.")
        return
    host = request.client.host if request.client else ""
    if host not in LOOPBACK:
        raise HTTPException(
            status_code=403,
            detail="Set BROLL_API_TOKEN to use the API from another machine.",
        )


def _studio(request: Request):
    return request.app.state.studio


def _client_for(request: Request, client: str | None):
    """The named client, the one in the cookie, or the only one there is."""
    studio = _studio(request)
    if client and client not in studio.clients:
        raise HTTPException(
            status_code=404,
            detail=f"No client {client!r}. Known: {', '.join(studio.order)}",
        )
    return studio.state(client) if client else client_state(request)


def _shot_json(result: Any, client_id: str) -> dict:
    shot, source = result.shot, result.source
    return {
        "client": client_id,
        "shot_id": shot.id,
        "source_id": source.id,
        "filename": source.original_filename,
        "media": source.media_kind,
        "caption": shot.caption,
        "start_s": round(shot.start_s, 2),
        "duration_s": round(shot.duration_s, 2),
        "timecode": result.timecode,
        "folder": shot.category,
        "also_fits": shot.secondary_categories,
        "emotions": shot.emotions,
        "mood": shot.mood,
        "tags": shot.tags,
        "action": shot.action,
        "setting": shot.setting,
        "shot_type": shot.shot_type,
        "camera_movement": shot.camera_movement,
        "time_of_day": shot.time_of_day,
        "pace": shot.pace,
        "people": shot.people_count,
        "featured_person": shot.featured_person,
        "top_pick": shot.top_pick,
        "quality_flags": shot.quality_flags,
        "status": shot.status,
        "score": round(result.score, 4),
        "drive_link": result.drive_link,
        "drive_path": source.drive_path,
        "local_path": source.origin_path,
        "thumbnail_url": f"/thumbnails/{shot.id}.jpg" if shot.thumbnail_path else None,
    }


def _filters(
    media: str | None = None,
    emotion: list[str] | None = None,
    folder: list[str] | None = None,
    min_duration: float | None = None,
    top_picks: bool = False,
    featured: bool | None = None,
) -> SearchFilters:
    return SearchFilters(
        media_kind=[media] if media in ("video", "image") else [],
        emotions=emotion or [],
        category=folder or [],
        duration_min_s=min_duration,
        top_pick=True if top_picks else None,
        featured_person=featured,
        exclude_flagged=True,
    )


# -- what exists ------------------------------------------------------------


@router.get("/clients")
async def clients(request: Request) -> dict:
    """Every client library this server holds, with how much is in each."""
    _authorise(request)
    studio = _studio(request)
    out = []
    for client_id in studio.order:
        state = studio.clients[client_id]
        store = state.store()
        try:
            stats = queue_stats(store)
            out.append({
                "id": client_id,
                "name": state.config.name,
                "featured_person": state.config.client.featured_person,
                "clips": len(store.list_sources(limit=1_000_000)),
                "shots": store.count_shots(),
                "needs_review": store.count_shots("needs_review"),
                "queued": stats.outstanding,
                "failed": stats.failed,
                "folders": ["/".join(parts) for parts in state.config.taxonomy.tree_folders()],
                "drive_root": state.config.drive_root_folder_name,
            })
        finally:
            store.close()
    return {"clients": out, "default": studio.default_id}


@router.get("/status")
async def status(request: Request, client: str | None = None) -> dict:
    """Queue and index state for one client - what an agent polls after ingesting."""
    _authorise(request)
    state = _client_for(request, client)
    store = state.store()
    try:
        stats = queue_stats(store)
        return {
            "client": state.config.id,
            "queued": stats.queued,
            "running": stats.running,
            "done": stats.done,
            "failed": stats.failed,
            "shots": store.count_shots(),
            "needs_review": store.count_shots("needs_review"),
            "worker_running": bool(state.worker),
        }
    finally:
        store.close()


# -- finding footage --------------------------------------------------------


@router.get("/search")
async def search(
    request: Request,
    q: str = "",
    client: str | None = None,
    limit: int = 20,
    media: str | None = None,
    emotion: list[str] | None = None,
    folder: list[str] | None = None,
    min_duration: float | None = None,
    top_picks: bool = False,
    loose: bool = False,
    all_clients: bool = False,
) -> dict:
    """Search one client, or every client at once.

    `all_clients` is for "have I shot this before, for anyone?" - the answer
    names the client each result belongs to.
    """
    _authorise(request)
    studio = _studio(request)
    targets = (
        [studio.clients[cid] for cid in studio.order] if all_clients
        else [_client_for(request, client)]
    )
    filters = _filters(media, emotion, folder, min_duration, top_picks)

    results: list[dict] = []
    for state in targets:
        store = state.store()
        try:
            engine = SearchEngine(store, state.embedder,
                                  featured_person=state.config.client.featured_person)
            found = engine.search(q, filters, limit, strict=not loose)
            results.extend(_shot_json(r, state.config.id) for r in found)
        finally:
            store.close()

    results.sort(key=lambda r: -r["score"])
    return {"query": q, "count": len(results[:limit]), "results": results[:limit]}


@router.get("/clip/{shot_id}")
async def clip(request: Request, shot_id: str) -> dict:
    """One shot in full, looked up across every client."""
    _authorise(request)
    studio = _studio(request)
    for client_id in studio.order:
        state = studio.clients[client_id]
        store = state.store()
        try:
            engine = SearchEngine(store, state.embedder)
            found = [r for r in engine.browse(SearchFilters(), 1_000_000) if r.shot.id == shot_id]
            if found:
                return _shot_json(found[0], client_id)
        finally:
            store.close()
    raise HTTPException(status_code=404, detail=f"No shot {shot_id!r} in any client.")


# -- choosing footage for a script -----------------------------------------


class ShortlistRequest(BaseModel):
    text: str = Field(description="The script or transcript. SRT, VTT or plain text.")
    client: str | None = None
    filename: str = "script.txt"
    candidates_per_beat: int = 8
    min_duration_s: float | None = None
    media: str | None = None
    exclude_shots: list[str] = Field(
        default_factory=list,
        description="Shot ids already used in this video, or used too recently.",
    )


@router.post("/shortlist")
async def shortlist(request: Request, body: ShortlistRequest) -> dict:
    """Candidates per line of script, for an agent to choose between.

    Costs nothing: this is the search half of transcript matching, with the
    model rerank left out on purpose. The caller does the judgement, which is
    the half worth doing well.
    """
    _authorise(request)
    state = _client_for(request, body.client)
    config = state.config
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="No script text.")

    beats = parse_and_segment(
        body.text, body.filename, config.transcript.words_per_minute,
        config.transcript.beat_min_s, config.transcript.beat_max_s,
    )
    filters = _filters(body.media, min_duration=body.min_duration_s)
    excluded = set(body.exclude_shots)

    store = state.store()
    try:
        engine = SearchEngine(store, state.embedder,
                              featured_person=config.client.featured_person)
        out = []
        for beat in beats:
            found = engine.search(
                beat.text, filters, body.candidates_per_beat + len(excluded),
                rank_only=True, person_filter=False,
            )
            candidates = [
                _shot_json(r, config.id) for r in found if r.shot.id not in excluded
            ][:body.candidates_per_beat]
            out.append({
                "beat": beat.index,
                "timecode": beat.timecode,
                "start_s": round(beat.start_s, 2),
                "end_s": round(beat.end_s, 2),
                "duration_s": round(beat.duration_s, 2),
                "line": beat.text,
                "candidates": candidates,
            })
    finally:
        store.close()

    return {
        "client": config.id,
        "beats": len(out),
        "featured_person": config.client.featured_person,
        "shortlist": out,
    }


class SuggestRequest(ShortlistRequest):
    rerank: bool = Field(default=True, description="Let the text model choose. Costs one request per line.")


@router.post("/suggest")
async def suggest(request: Request, body: SuggestRequest) -> dict:
    """Full transcript matching, including the model's pick per line and why.

    Use this when nothing is going to read the shortlist. It spends one text
    request per line of script.
    """
    _authorise(request)
    from ...analysis.providers.registry import get_text_provider
    from ...transcript.matcher import TranscriptMatcher

    state = _client_for(request, body.client)
    config = state.config
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="No script text.")

    beats = parse_and_segment(
        body.text, body.filename, config.transcript.words_per_minute,
        config.transcript.beat_min_s, config.transcript.beat_max_s,
    )
    store = state.store()
    try:
        engine = SearchEngine(store, state.embedder,
                              featured_person=config.client.featured_person)
        provider = None
        if body.rerank:
            try:
                provider = get_text_provider(config)
            except Exception:  # a missing key must not fail the call
                provider = None
        matches = await TranscriptMatcher(config, engine, provider).match(beats)

        out = []
        for match in matches:
            chosen = match.chosen
            out.append({
                "beat": match.beat.index,
                "timecode": match.beat.timecode,
                "line": match.beat.text,
                "duration_s": round(match.beat.duration_s, 2),
                "chosen": _shot_json(chosen, config.id) | {"why": chosen.reason} if chosen else None,
                "alternatives": [
                    _shot_json(s, config.id) | {"why": s.reason}
                    for s in match.alternatives[1:4]
                ],
                "no_good_match": match.no_good_match,
                "missing_footage": match.missing_footage,
            })
    finally:
        store.close()

    return {
        "client": config.id,
        "reranked": bool(provider),
        "beats": out,
        "gaps": [b["missing_footage"] for b in out if b["no_good_match"]],
    }


# -- adding footage ---------------------------------------------------------


class IngestRequest(BaseModel):
    paths: list[str] = Field(description="Files or folders on this machine.")
    client: str | None = None
    force: bool = False


@router.post("/ingest")
async def ingest(request: Request, body: IngestRequest) -> dict:
    """Queue footage for a client. The worker in this process picks it up."""
    _authorise(request)
    state = _client_for(request, body.client)

    discovered = []
    missing = []
    for raw in body.paths:
        path = Path(raw).expanduser()
        if not path.exists():
            missing.append(str(path))
            continue
        if path.is_dir():
            discovered.extend(scan_local(path))
        elif media_kind(path) is not None:
            from ...ingest.scanner import DiscoveredFile

            discovered.append(DiscoveredFile(
                origin="local", path=path.resolve(), filename=path.name,
                origin_path=str(path.resolve()),
            ))
        else:
            missing.append(f"{path} (not a video or photo)")

    store: Store = state.store()
    try:
        jobs = enqueue_files(store, discovered, force=body.force)
        stats = queue_stats(store)
    finally:
        store.close()

    return {
        "client": state.config.id,
        "queued": len(jobs),
        "already_queued": len(discovered) - len(jobs),
        "skipped": missing,
        "outstanding": stats.outstanding,
    }
