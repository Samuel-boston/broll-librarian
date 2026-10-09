"""A JSON API, for an agent rather than a person.

The web screens are for browsing; this is what an editing agent calls when it
is cutting a video and needs to know what footage exists. Everything here is
cheap: no vision calls, no Drive writes, nothing that costs money except an
explicitly requested rerank.

The shortlist endpoint is the important one. It does the mechanical half of
B-roll selection - segment the script (or take the caller's own beats), search
each beat, drop what is too short or already used - and hands back candidates
with their descriptions and a verdict on whether each is really about the
line. The judgement half, which clip and for how long, belongs to the agent
reading it.

Once the agent has chosen, /api/fetch turns a shot into a local file it can
render (cut down to the shot when asked), and /api/usage remembers which video
it went into, so the next shortlist does not offer it again.

Auth: set BROLL_API_TOKEN and send it as `Authorization: Bearer <token>` or
`X-Broll-Token`. With no token set, only requests from the loopback address are
served, which is the normal case for an agent on the same machine.
"""

from __future__ import annotations

import asyncio
import hmac
import os
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from ...db.store import Store, canonical_time
from ...ingest.scanner import DiscoveredFile, media_kind, scan_local
from ...jobs.queue import enqueue_files, queue_stats
from ...search.filters import SearchFilters
from ...search.query import SearchEngine, SearchResult
from ...transcript.parser import parse_and_segment
from ..app import client_state

router = APIRouter(prefix="/api", tags=["api"])

# Bumped whenever an endpoint is added or a shape changes, so a caller can tell
# a server started before an update (it keeps the code it started with) from
# one that has what it needs. 2: fetch, usage, health, queue, upload, beats.
API_VERSION = 2

# "testclient" is what Starlette's in-process test client reports; a real
# server takes this from the socket, so it can never arrive over a network.
LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}

RELEVANCE_ORDER = {"match": 0, "near": 1, "weak": 2, None: 3}


def _authorise(request: Request) -> None:
    token = os.environ.get("BROLL_API_TOKEN", "").strip()
    if token:
        header = request.headers.get("authorization", "")
        supplied = header[7:].strip() if header.lower().startswith("bearer ") else ""
        supplied = supplied or request.headers.get("x-broll-token", "").strip()
        if not hmac.compare_digest(supplied.encode(), token.encode()):
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


def _named_client(request: Request, client: str | None):
    """Like _client_for, but a write must not land in whichever client a cookie
    happens to name: with more than one library, the caller has to say."""
    studio = _studio(request)
    if not client and len(studio.order) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"Name the client: one of {', '.join(studio.order)}.",
        )
    return _client_for(request, client or studio.default_id)


def _owner_of(request: Request, shot_id: str, client: str | None):
    """The client whose library holds this shot. Shot ids are unique across clients."""
    studio = _studio(request)
    states = [_client_for(request, client)] if client else [studio.clients[c] for c in studio.order]
    for state in states:
        store = state.store()
        try:
            if store.get_shot(shot_id) is not None:
                return state
        finally:
            store.close()
    where = f"client {client!r}" if client else "any client"
    raise HTTPException(status_code=404, detail=f"No shot {shot_id!r} in {where}.")


def _shot_json(result: Any, client_id: str, usage: dict[str, list[dict]] | None = None) -> dict:
    shot, source = result.shot, result.source
    out = {
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
        "end_s": round(shot.end_s, 2),
        "width": source.width,
        "height": source.height,
        "fps": source.fps,
    }
    if getattr(result, "similarity", None) is not None:
        out["similarity"] = round(result.similarity, 4)
    if getattr(result, "relevance", None) is not None:
        out["relevance"] = result.relevance
        out["keyword_coverage"] = round(result.coverage or 0.0, 3)
    if usage is not None:
        used = usage.get(shot.id, [])
        out["used_in"] = used[:5]
        out["use_count"] = len(used)
    return out


def _filters(
    media: str | None = None,
    emotion: list[str] | None = None,
    folder: list[str] | None = None,
    min_duration: float | None = None,
    top_picks: bool = False,
    featured: bool | None = None,
    exclude: set[str] | None = None,
) -> SearchFilters:
    return SearchFilters(
        media_kind=[media] if media in ("video", "image") else [],
        emotions=emotion or [],
        category=folder or [],
        duration_min_s=min_duration,
        top_pick=True if top_picks else None,
        featured_person=featured,
        exclude_flagged=True,
        exclude_shot_ids=sorted(exclude or ()),
    )


def _recently_used(store: Store, within_days: int | None, project: str | None) -> set[str]:
    """Shots cut into another video in the last `within_days` days."""
    if not within_days or within_days <= 0:
        return set()
    since = datetime.now(UTC) - timedelta(days=within_days)
    return store.shots_used_since(canonical_time(since), other_than_project=project)


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


class ClientRequest(BaseModel):
    name: str = Field(min_length=1, description="The client's name, as the calling app knows it.")
    id: str | None = Field(default=None, description="Optional library id; defaults to a slug of the name.")


@router.post("/clients")
async def create_client(request: Request, body: ClientRequest) -> dict:
    """A library for this client, created if it does not exist yet.

    Idempotent: asking again for the same client (same name, ignoring case and
    spacing, or the same id) returns the library it already has, with
    `created: false`. A new library is served at once, empty, and copies the
    provider, embedder and rate cap of this server's first library, because
    they share one key and one embedding model.
    """
    _authorise(request)
    from ...clients import create_library
    from ...config import load_workspace_config, slugify_id

    if body.id is not None and (not body.id or slugify_id(body.id) != body.id):
        raise HTTPException(
            status_code=422,
            detail=f"id {body.id!r} must be lower-case letters, digits and dashes.",
        )
    studio = _studio(request)
    like = studio.clients[studio.default_id].config
    try:
        library_id, name, created = await asyncio.to_thread(create_library, body.name, body.id, like)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if library_id not in studio.clients:
        await studio.add_client(load_workspace_config(library_id))
    return {"id": library_id, "name": name, "created": created}


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


@router.get("/queue")
async def queue(request: Request, client: str | None = None, limit: int = 40) -> dict:
    """The files waiting, being indexed and recently finished, for a queue panel."""
    _authorise(request)
    state = _client_for(request, client)
    store = state.store()
    try:
        stats = queue_stats(store)
        rows = store.recent_jobs(min(limit, 500))
        return {
            "client": state.config.id,
            "queued": stats.queued,
            "running": stats.running,
            "done": stats.done,
            "failed": stats.failed,
            "cost_usd": round(store.total_cost(), 4),
            "jobs": [
                {
                    "id": r["id"],
                    "filename": r["filename"],
                    "status": r["status"],
                    "attempts": r["attempts"],
                    "error": r["last_error"],
                    "cost_usd": round(r["cost_estimate_usd"] or 0.0, 5),
                    "created_at": r["created_at"],
                    "started_at": r["started_at"],
                    "finished_at": r["finished_at"],
                    "retry_after": r["not_before"] if r["status"] == "queued" else None,
                }
                for r in rows
            ],
        }
    finally:
        store.close()


# -- is it working ----------------------------------------------------------

# Checking a Drive login can mean a round trip to Google to refresh it, so the
# answer is kept for a few minutes; health is polled, and must stay quick.
DRIVE_CHECK_TTL_S = 300.0
_drive_checks: dict[str, tuple[float, bool]] = {}


def _drive_connected(config, refresh: bool = False) -> bool:
    if not config.drive_token_path.exists():
        return False
    cached = _drive_checks.get(config.id)
    if cached and not refresh and time.monotonic() - cached[0] < DRIVE_CHECK_TTL_S:
        return cached[1]
    try:
        from ...drive.auth import load_credentials

        credentials = load_credentials(config)
        connected = bool(credentials and credentials.valid)
    except Exception:  # noqa: BLE001 - an expired or revoked login is an answer, not an error
        connected = False
    _drive_checks[config.id] = (time.monotonic(), connected)
    return connected


def _worker_alive(state) -> bool:
    task = state.worker_task
    return bool(task is not None and not task.done())


def _provider_waits(errors: list[str]) -> tuple[int, bool]:
    """(files waiting out a provider refusal, whether it is a daily quota).

    A free Gemini key allows about 20 requests a day. Past that, every request
    is refused until tomorrow; the queue keeps the files and retries them on
    its own, so this is a state to report, not an error to raise.
    """
    from ...analysis.providers.base import TransientProviderError, classify_error

    refused = [e for e in errors if classify_error(e) is TransientProviderError]
    daily = any(
        marker in e.lower().replace(" ", "")
        for e in refused for marker in ("perday", "requestsperday", "dailylimit")
    )
    return len(refused), daily


@router.get("/health")
async def health(request: Request, refresh: bool = False) -> dict:
    """Is the library usable, client by client. Booleans only: never a key or a token."""
    _authorise(request)
    from importlib.metadata import PackageNotFoundError, version

    from ...analysis.prompt import PROMPT_VERSION
    from ...db.migrations import SCHEMA_VERSION
    from ...sync.dashboard import is_connected

    try:
        package_version = version("broll-librarian")
    except PackageNotFoundError:  # running from a checkout that was never installed
        package_version = "unknown"

    studio = _studio(request)
    clients_out: list[dict] = []
    problems: list[str] = []
    for client_id in studio.order:
        state = studio.clients[client_id]
        config = state.config
        store = state.store()
        try:
            stats = queue_stats(store)
            shots = store.count_shots()
            needs_review = store.count_shots("needs_review")
            waiting, daily_quota = _provider_waits(store.waiting_errors())
        finally:
            store.close()
        provider = config.provider.vision
        key_present = provider == "mock" or bool(config.api_key(provider))
        token = config.drive_token_path.exists()
        drive_ok = await asyncio.to_thread(_drive_connected, config, refresh) if token else False
        mount = config.drive_local_mount_path
        entry = {
            "id": client_id,
            "name": config.name,
            "shots": shots,
            "needs_review": needs_review,
            "queued": stats.queued,
            "running": stats.running,
            "failed": stats.failed,
            "worker_alive": _worker_alive(state),
            "provider": provider,
            "provider_key": key_present,
            "provider_waiting": waiting,
            "provider_daily_quota_used": daily_quota,
            "rate_cap_per_minute": config.ingest.requests_per_minute,
            "drive_token": token,
            "drive_connected": drive_ok,
            "drive_mount": bool(mount and Path(mount).expanduser().is_dir()),
            "dashboard_connected": is_connected(config),
        }
        clients_out.append(entry)
        if not key_present:
            problems.append(f"{client_id}: no API key for {provider} - new footage cannot be analysed.")
        if daily_quota:
            problems.append(
                f"{client_id}: the {provider} key's daily quota is used up (a free Gemini key allows "
                f"about 20 requests a day); {waiting} file(s) wait and retry on their own."
            )
        elif waiting:
            problems.append(
                f"{client_id}: {provider} is refusing requests for now (rate limit or overload); "
                f"{waiting} file(s) will retry on their own."
            )
        if not token:
            problems.append(f"{client_id}: Drive is not connected - run `broll drive login -w {client_id}`.")
        elif not drive_ok:
            problems.append(
                f"{client_id}: the Drive login no longer works (expired or revoked) - "
                f"run `broll drive login -w {client_id}`."
            )
        if state.run_worker and not entry["worker_alive"]:
            problems.append(f"{client_id}: the ingest worker is not running - restart the server.")
        if stats.failed:
            problems.append(f"{client_id}: {stats.failed} file(s) failed to index - `broll retry -w {client_id}`.")

    workers = [c["worker_alive"] for c in clients_out]
    return {
        "ok": True,
        "service": "broll-librarian",
        "version": package_version,
        "api": API_VERSION,
        "schema": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "worker_alive": bool(workers) and all(workers),
        "clients": clients_out,
        "problems": problems,
    }


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
    exclude_used_in_project: str | None = None,
    exclude_used_within_days: int | None = None,
    project: str | None = None,
) -> dict:
    """Search one client, or every client at once.

    `all_clients` is for "have I shot this before, for anyone?" - the answer
    names the client each result belongs to. `exclude_used_in_project` drops
    shots already cut into that video; `exclude_used_within_days` drops shots
    cut into any other video that recently (`project`, or the video named by
    `exclude_used_in_project`, never counts against itself).
    """
    _authorise(request)
    studio = _studio(request)
    targets = (
        [studio.clients[cid] for cid in studio.order] if all_clients
        else [_client_for(request, client)]
    )
    own = project or exclude_used_in_project

    results: list[dict] = []
    for state in targets:
        store = state.store()
        try:
            excluded = _recently_used(store, exclude_used_within_days, own)
            if exclude_used_in_project:
                excluded |= store.shots_used_in_project(exclude_used_in_project)
            filters = _filters(media, emotion, folder, min_duration, top_picks, exclude=excluded)
            engine = SearchEngine(store, state.embedder,
                                  featured_person=state.config.client.featured_person)
            found = engine.search(q, filters, limit, strict=not loose)
            usage = store.usage_for(r.shot.id for r in found)
            results.extend(_shot_json(r, state.config.id, usage) for r in found)
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
            shot = store.get_shot(shot_id)
            source = store.get_source(shot.source_id) if shot else None
            if shot and source:
                return _shot_json(SearchResult(shot=shot, source=source, score=0.0), client_id,
                                  store.usage_for([shot_id]))
        finally:
            store.close()
    raise HTTPException(status_code=404, detail=f"No shot {shot_id!r} in any client.")


# -- choosing footage for a script -----------------------------------------


class BeatRequest(BaseModel):
    id: str | None = Field(
        default=None,
        description="The beat's id in the caller's storyboard. A shot recorded against this "
                    "beat stays on offer to it when the video's other shots are excluded.",
    )
    queries: list[str] = Field(
        description="What the viewer should see, one idea per string, most important first. "
                    "Each is searched on its own, so every idea gets candidates.",
    )
    line: str = Field(default="", description="The spoken line, returned as context.")
    duration_s: float | None = None
    min_duration_s: float | None = Field(
        default=None, description="Shots shorter than this are not offered for this beat.",
    )


class ShortlistRequest(BaseModel):
    text: str = Field(default="", description="The script or transcript. SRT, VTT or plain text.")
    beats: list[BeatRequest] | None = Field(
        default=None,
        description="Beats already cut, e.g. from a storyboard. Used as given instead of "
                    "segmenting `text`.",
    )
    client: str | None = None
    filename: str = "script.txt"
    candidates_per_beat: int = 8
    max_per_source: int = Field(
        default=3, description="With beats: at most this many shots of one file per beat.",
    )
    min_duration_s: float | None = None
    media: str | None = None
    exclude_shots: list[str] = Field(
        default_factory=list,
        description="Shot ids already used in this video, or used too recently.",
    )
    exclude_used_in_project: str | None = Field(
        default=None, description="Drop shots already cut into this video (see /api/usage).",
    )
    exclude_used_within_days: int | None = Field(
        default=None, description="Drop shots cut into any other video in the last N days.",
    )
    project: str | None = Field(
        default=None,
        description="The video this shortlist is for. Its own earlier choices never count "
                    "against it as recent use. Defaults to exclude_used_in_project.",
    )


def _candidates_for_beat(
    engine: SearchEngine, queries: list[str], filters: SearchFilters, limit: int,
    max_per_source: int,
) -> list[tuple[SearchResult, str, list[str]]]:
    """Search each idea, then merge: every idea represented, best verdict first.

    Candidates are taken round-robin by rank across the queries, so a chained
    beat gets something for each idea rather than the twelve best matches for
    whichever idea shares the most words with the library. Then they are
    ordered by verdict (match, near, weak) and similarity.
    """
    ranked = [
        engine.search(q, filters, limit, rank_only=True, person_filter=False,
                      annotate=True, correct=False)
        for q in queries
    ]
    best: dict[str, tuple[tuple, SearchResult, int]] = {}
    found_by: dict[str, list[int]] = {}
    for qi, results in enumerate(ranked):
        for rank, result in enumerate(results):
            sid = result.shot.id
            found_by.setdefault(sid, []).append(qi)
            key = (RELEVANCE_ORDER.get(result.relevance, 3), -(result.similarity or -1.0), rank)
            if sid not in best or key < best[sid][0]:
                best[sid] = (key, result, qi)

    picked: list[str] = []
    per_source: Counter = Counter()
    for k in range(max((len(r) for r in ranked), default=0)):
        for results in ranked:
            if k >= len(results) or len(picked) >= limit:
                continue
            result = results[k]
            if result.shot.id in picked:
                continue
            if max_per_source and per_source[result.source.id] >= max_per_source:
                continue
            per_source[result.source.id] += 1
            picked.append(result.shot.id)
    picked.sort(key=lambda sid: best[sid][0])
    return [
        (best[sid][1], queries[best[sid][2]], [queries[i] for i in dict.fromkeys(found_by[sid])])
        for sid in picked
    ]


def _no_match_reason(library_shots: int, candidates: list[dict], has_queries: bool) -> str | None:
    """Why a beat has nothing worth cutting to, in words the caller can pass on.

    An empty library is a normal state - most clients start with one - so it
    is said plainly rather than looking like a failure.
    """
    if not library_shots:
        return "The library is empty: nothing has been indexed for this client yet."
    if not has_queries:
        return "The beat asked for nothing (no queries)."
    if not candidates:
        return ("Nothing in the library passes this beat's filters (length, media, quality "
                "flags, or already used).")
    if not any(c.get("relevance") == "match" for c in candidates):
        return "Nothing in the library is really about this; the candidates are only the closest."
    return None


@router.post("/shortlist")
async def shortlist(request: Request, body: ShortlistRequest) -> dict:
    """Candidates per line of script, or per beat of a storyboard, to choose between.

    Costs nothing: this is the search half of transcript matching, with the
    model rerank left out on purpose. The caller does the judgement, which is
    the half worth doing well. Each candidate says whether it is really about
    the line ("relevance": match, near or weak), so the caller can use a beat's
    fallback instead of forcing a clip that is merely the least bad.
    """
    _authorise(request)
    state = _client_for(request, body.client)
    config = state.config
    if not body.beats and not body.text.strip():
        raise HTTPException(status_code=400, detail="No script text (or beats).")

    project = body.project or body.exclude_used_in_project
    given = set(body.exclude_shots)
    store = state.store()
    try:
        library_shots = store.count_shots("indexed") + store.count_shots("needs_review")
        engine = SearchEngine(store, state.embedder,
                              featured_person=config.client.featured_person)
        recent = _recently_used(store, body.exclude_used_within_days, project)
        in_project = (
            store.shots_used_in_project(body.exclude_used_in_project)
            if body.exclude_used_in_project else set()
        )
        out: list[dict] = []
        if body.beats:
            for index, beat in enumerate(body.beats):
                queries = list(dict.fromkeys(q.strip() for q in beat.queries if q and q.strip()))
                excluded = given | recent
                if body.exclude_used_in_project:
                    excluded |= store.shots_used_in_project(body.exclude_used_in_project,
                                                            except_beat=beat.id)
                min_duration = beat.min_duration_s if beat.min_duration_s is not None else body.min_duration_s
                filters = _filters(body.media, min_duration=min_duration, exclude=excluded)
                found = _candidates_for_beat(engine, queries, filters, body.candidates_per_beat,
                                             body.max_per_source) if queries and library_shots else []
                usage = store.usage_for(r.shot.id for r, _, _ in found)
                candidates = [
                    _shot_json(r, config.id, usage) | {"query": query, "queries": matched}
                    for r, query, matched in found
                ]
                out.append({
                    "beat": beat.id if beat.id is not None else index,
                    "line": beat.line,
                    "duration_s": beat.duration_s,
                    "queries": queries,
                    "matches": sum(1 for c in candidates if c.get("relevance") == "match"),
                    "reason": _no_match_reason(library_shots, candidates, bool(queries)),
                    "candidates": candidates,
                })
        else:
            beats = parse_and_segment(
                body.text, body.filename, config.transcript.words_per_minute,
                config.transcript.beat_min_s, config.transcript.beat_max_s,
            )
            filters = _filters(body.media, min_duration=body.min_duration_s,
                               exclude=given | recent | in_project)
            for beat in beats:
                found = engine.search(
                    beat.text, filters, body.candidates_per_beat,
                    rank_only=True, person_filter=False, annotate=True,
                ) if library_shots else []
                usage = store.usage_for(r.shot.id for r in found)
                candidates = [_shot_json(r, config.id, usage) for r in found]
                out.append({
                    "beat": beat.index,
                    "timecode": beat.timecode,
                    "start_s": round(beat.start_s, 2),
                    "end_s": round(beat.end_s, 2),
                    "duration_s": round(beat.duration_s, 2),
                    "line": beat.text,
                    "matches": sum(1 for c in candidates if c.get("relevance") == "match"),
                    "reason": _no_match_reason(library_shots, candidates, True),
                    "candidates": candidates,
                })
    finally:
        store.close()

    return {
        "client": config.id,
        "library_shots": library_shots,
        "beats": len(out),
        "featured_person": config.client.featured_person,
        "excluded": {
            "given": len(given),
            "used_in_project": len(in_project),
            "used_recently": len(recent),
        },
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


# -- after choosing: the file, and the memory -------------------------------


class FetchRequest(BaseModel):
    shot_id: str
    client: str | None = Field(default=None, description="Optional: shot ids are unique across clients.")
    dest_dir: str | None = Field(
        default=None,
        description="Absolute folder for a download, a trimmed cut, or (with copy) a copy. "
                    "Without it, downloads and cuts go to the workspace temp directory.",
    )
    trim: bool = Field(default=False, description="Cut just the shot, plus handles, out of the file.")
    handles_s: float = Field(default=0.5, ge=0.0, le=10.0)
    copy_file: bool = Field(
        default=False, alias="copy",
        description="Put a copy in dest_dir even when the original is already on this machine.",
    )

    model_config = {"populate_by_name": True}


@router.post("/fetch")
async def fetch(request: Request, body: FetchRequest) -> dict:
    """A local file for a shot: the original, the Drive for Desktop copy, or a download.

    With `trim`, only the shot (plus handles) is cut out, and the answer says
    where the shot sits in the returned file (`in_s`, `out_s`, `offset_s`).
    Nothing in Drive is changed.
    """
    _authorise(request)
    from ...fetch import FetchError, fetch_shot

    state = _owner_of(request, body.shot_id, body.client)
    dest = Path(body.dest_dir).expanduser() if body.dest_dir else None
    if dest is not None and not dest.is_absolute():
        raise HTTPException(status_code=422, detail=f"dest_dir must be an absolute path, got {body.dest_dir!r}.")

    def work():
        store = state.store()  # opened in this thread: a sqlite connection may not cross threads
        try:
            return fetch_shot(state.config, store, body.shot_id, dest_dir=dest, trim=body.trim,
                              handles_s=body.handles_s, copy=body.copy_file)
        finally:
            store.close()

    try:
        result = await asyncio.to_thread(work)
    except FetchError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    return result.to_dict()


class UsageRequest(BaseModel):
    project: str = Field(min_length=1, description="The video the shots are cut into.")
    shot_ids: list[str]
    client: str | None = None
    used_at: str | None = Field(default=None, description="ISO 8601; defaults to now.")
    beats: dict[str, str] = Field(
        default_factory=dict,
        description="shot id -> the beat it covers, so re-choosing that beat still offers it.",
    )
    replace: bool = Field(
        default=False,
        description="Make this the video's whole list: shots it no longer uses stop counting.",
    )


@router.post("/usage")
async def record_usage(request: Request, body: UsageRequest) -> dict:
    """Remember which shots a video uses, so later shortlists can leave them out."""
    _authorise(request)
    state = _named_client(request, body.client)
    try:
        when = canonical_time(body.used_at)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"used_at {body.used_at!r} is not an ISO 8601 date/time.")
    project = body.project.strip()
    if not project:
        raise HTTPException(status_code=422, detail="project is empty.")
    store = state.store()
    try:
        result = store.record_usage(body.shot_ids, project, when, body.beats, body.replace)
    finally:
        store.close()
    return {
        "client": state.config.id,
        "project": project,
        "used_at": when,
        "recorded": len(result["recorded"]),
        "unknown": result["unknown"],
        "released": result["released"],
    }


@router.get("/usage")
async def list_usage(
    request: Request,
    client: str | None = None,
    project: str | None = None,
    shot_id: str | None = None,
    limit: int = 200,
) -> dict:
    """Which videos use which shots: by video, by shot, or everything recent."""
    _authorise(request)
    state = _client_for(request, client)
    store = state.store()
    try:
        rows = store.list_usage(project=project, shot_id=shot_id, limit=min(limit, 5000))
    finally:
        store.close()
    return {"client": state.config.id, "project": project, "count": len(rows), "usage": rows}


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


@router.post("/upload")
async def upload(request: Request, client: str | None = None,
                 files: list[UploadFile] = File(default=[])) -> dict:
    """Queue files sent in the request body (a drag-and-drop proxied by another app).

    They are staged in the client's workspace, indexed, filed into Drive, and
    the staged copy is removed once Drive has it - the same as the Upload screen.
    """
    _authorise(request)
    from .ingest import _unique_path

    state = _named_client(request, client)
    accepted: list[DiscoveredFile] = []
    rejected: list[str] = []
    for upload_file in files:
        name = Path(upload_file.filename or "clip").name
        if media_kind(name) is None:
            rejected.append(name)
            continue
        target = _unique_path(state.config.staging_dir, name)
        with target.open("wb") as handle:
            while chunk := await upload_file.read(1024 * 1024):
                handle.write(chunk)
        accepted.append(DiscoveredFile(origin="upload", path=target, filename=name,
                                       origin_path=str(target)))
    store = state.store()
    try:
        jobs = enqueue_files(store, accepted)
        stats = queue_stats(store)
    finally:
        store.close()
    return {
        "client": state.config.id,
        "queued": len(jobs),
        "files": [{"filename": j.payload.get("filename"), "job_id": j.id} for j in jobs],
        "skipped": rejected,
        "outstanding": stats.outstanding,
    }
