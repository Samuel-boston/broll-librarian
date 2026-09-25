"""Ingest screen: drag-and-drop uploads, a Drive folder, and the live queue."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse

from ...ingest.scanner import DiscoveredFile, media_kind, scan_local
from ...jobs.queue import enqueue_files, queue_stats
from ...library_admin import cancel_job, clear_queue
from ..app import templates

router = APIRouter()


@router.get("/ingest", response_class=HTMLResponse)
async def ingest_page(request: Request):
    state = request.app.state.broll
    store = state.store()
    try:
        context = _queue_context(request, store, state)
    finally:
        store.close()
    return templates.TemplateResponse(request=request, name="ingest.html", context=context)


@router.post("/ingest/retry", response_class=HTMLResponse)
async def retry_failed(request: Request):
    """Requeue everything that failed - usually a run of provider 503s."""
    state = request.app.state.broll
    store = state.store()
    try:
        requeued = store.requeue_failed_jobs()
        context = _queue_context(request, store, state)
        context["message"] = (
            f"Requeued {requeued} failed job(s)." if requeued else "Nothing had failed."
        )
    finally:
        store.close()
    return templates.TemplateResponse(request=request, name="partials/queue.html", context=context)


@router.post("/ingest/jobs/{job_id}/cancel", response_class=HTMLResponse)
async def cancel_one(request: Request, job_id: str):
    """Take one waiting file out of the queue."""
    state = request.app.state.broll
    store = state.store()
    try:
        result = cancel_job(state.config, store, job_id)
        context = _queue_context(request, store, state)
        if result.cancelled:
            context["message"] = f"Removed {result.names[0]} from the queue."
        elif result.still_running:
            context["error"] = "That one is already being indexed, so it will finish. Only waiting files can be removed."
        else:
            context["message"] = "That file was already out of the queue."
    finally:
        store.close()
    return templates.TemplateResponse(request=request, name="partials/queue.html", context=context)


@router.post("/ingest/clear", response_class=HTMLResponse)
async def clear_the_queue(request: Request):
    """Empty the queue of everything still waiting. Files already being indexed finish."""
    state = request.app.state.broll
    store = state.store()
    try:
        result = clear_queue(state.config, store)
        context = _queue_context(request, store, state)
        parts = [f"Removed {result.cancelled} file(s) from the queue." if result.cancelled else "Nothing was waiting."]
        if result.still_running:
            parts.append(f"{result.still_running} already being indexed will finish.")
        context["message"] = " ".join(parts)
    finally:
        store.close()
    return templates.TemplateResponse(request=request, name="partials/queue.html", context=context)


@router.get("/ingest/queue", response_class=HTMLResponse)
async def queue_partial(request: Request):
    """Polled by HTMX every couple of seconds while anything is outstanding."""
    state = request.app.state.broll
    store = state.store()
    try:
        context = _queue_context(request, store, state)
    finally:
        store.close()
    return templates.TemplateResponse(request=request, name="partials/queue.html", context=context)


@router.post("/ingest/upload", response_class=HTMLResponse)
async def upload(request: Request, files: list[UploadFile] = File(default=[])):
    state = request.app.state.broll
    store = state.store()
    accepted, rejected = [], []
    try:
        for upload_file in files:
            name = Path(upload_file.filename or "clip").name
            if media_kind(name) is None:
                rejected.append(name)
                continue
            target = _unique_path(state.config.staging_dir, name)
            with target.open("wb") as handle:
                while chunk := await upload_file.read(1024 * 1024):
                    handle.write(chunk)
            accepted.append(
                DiscoveredFile(origin="upload", path=target, filename=name,
                               origin_path=str(target))
            )
        enqueue_files(store, accepted)
        context = _queue_context(request, store, state)
        context["message"] = f"Queued {len(accepted)} file(s)."
        if rejected:
            context["message"] += f" Skipped {len(rejected)} non-video file(s)."
    finally:
        store.close()
    return templates.TemplateResponse(request=request, name="partials/queue.html", context=context)


@router.post("/ingest/path", response_class=HTMLResponse)
async def ingest_path(request: Request, path: str = Form(...)):
    """Point the tool at a local folder. Files there are never moved or deleted."""
    state = request.app.state.broll
    store = state.store()
    try:
        target = Path(path).expanduser()
        if not target.exists():
            context = _queue_context(request, store, state)
            context["error"] = f"{target} does not exist."
            return templates.TemplateResponse(request=request, name="partials/queue.html", context=context)
        found = scan_local(target)
        enqueue_files(store, found)
        context = _queue_context(request, store, state)
        context["message"] = f"Queued {len(found)} file(s) from {target}."
    finally:
        store.close()
    return templates.TemplateResponse(request=request, name="partials/queue.html", context=context)


def _unique_path(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    counter = 1
    while target.exists():
        target = directory / f"{Path(name).stem}_{counter}{Path(name).suffix}"
        counter += 1
    return target


def _queue_context(request: Request, store, state) -> dict:
    stats = queue_stats(store)
    recent = store.conn.execute(
        """SELECT id, kind, status, attempts, last_error, cost_estimate_usd,
                  json_extract(payload_json, '$.filename') AS filename,
                  created_at, started_at, finished_at
           FROM jobs WHERE workspace_id = ? AND status != 'cancelled'
           ORDER BY CASE status WHEN 'running' THEN 0 WHEN 'queued' THEN 1 ELSE 2 END,
                    created_at DESC
           LIMIT 40""",
        (store.workspace_id,),
    ).fetchall()
    average = store.conn.execute(
        "SELECT AVG(cost_estimate_usd) FROM jobs WHERE workspace_id = ?"
        " AND status = 'done' AND cost_estimate_usd > 0",
        (store.workspace_id,),
    ).fetchone()[0]
    per_file = float(average or 0.0)

    return {
        "request": request,
        "workspace": state.config,
        "stats": stats,
        "jobs": [dict(r) for r in recent],
        "cost_so_far": store.total_cost(),
        "cost_remaining": per_file * stats.outstanding,
        "shots": store.count_shots(),
        "needs_review": store.count_shots("needs_review"),
        "worker_running": state.worker is not None,
        "poll": stats.outstanding > 0,
    }
