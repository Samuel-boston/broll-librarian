"""Ingest screen: drag-and-drop uploads, a Drive folder, and the live queue."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse

from ... import kinds
from ...ingest.scanner import DiscoveredFile, media_kind, scan_local
from ...jobs.queue import KIND_INDEX_SOURCE, enqueue_files, queue_stats
from ...library_admin import cancel_job, clear_queue
from ..app import client_state, templates

router = APIRouter()


@router.get("/ingest", response_class=HTMLResponse)
async def ingest_page(request: Request):
    state = client_state(request)
    store = state.store()
    try:
        context = _queue_context(request, store, state)
    finally:
        store.close()
    return templates.TemplateResponse(request=request, name="ingest.html", context=context)


@router.post("/ingest/retry", response_class=HTMLResponse)
async def retry_failed(request: Request):
    """Requeue everything that failed - usually a run of provider 503s."""
    state = client_state(request)
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
    state = client_state(request)
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
    state = client_state(request)
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
    state = client_state(request)
    store = state.store()
    try:
        context = _queue_context(request, store, state)
    finally:
        store.close()
    return templates.TemplateResponse(request=request, name="partials/queue.html", context=context)


@router.post("/ingest/upload", response_class=HTMLResponse)
async def upload(request: Request, files: list[UploadFile] = File(default=[])):
    state = client_state(request)
    store = state.store()
    accepted, rejected, rejected_full = [], [], []
    try:
        for upload_file in files:
            name = Path(upload_file.filename or "clip").name
            if media_kind(name) is None:
                rejected.append(name)
                continue
            if not _has_room(state.config):
                rejected_full.append(f"{name} (the server is almost out of disk)")
                continue
            target = _unique_path(state.config.staging_dir, name)
            try:
                with target.open("wb") as handle:
                    written = 0
                    while chunk := await upload_file.read(1024 * 1024):
                        handle.write(chunk)
                        written += len(chunk)
                        # Every 256 MB: is there still room, with the spare the server needs to stay up?
                        if written % (256 * 1024 * 1024) < len(chunk) and not _has_room(state.config):
                            raise OSError("the server is almost out of disk")
            except OSError as exc:
                target.unlink(missing_ok=True)
                rejected_full.append(f"{name} ({exc})")
                continue
            accepted.append(
                DiscoveredFile(origin="upload", path=target, filename=name,
                               origin_path=str(target))
            )
        enqueue_files(store, accepted)
        context = _queue_context(request, store, state)
        context["message"] = f"Queued {len(accepted)} file(s)."
        if rejected:
            context["message"] += f" Skipped {len(rejected)} file(s) that are neither a video nor an image."
        if rejected_full:
            context["error"] = "Not uploaded, the disk is too full: " + "; ".join(rejected_full)
    finally:
        store.close()
    return templates.TemplateResponse(request=request, name="partials/queue.html", context=context)


@router.post("/ingest/path", response_class=HTMLResponse)
async def ingest_path(request: Request, path: str = Form(...)):
    """Point the tool at a local folder. Files there are never moved or deleted."""
    state = client_state(request)
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


def _has_room(config) -> bool:
    """Is there still more than the disk headroom free? Uploads stop before they fill the server."""
    import shutil

    try:
        config.staging_dir.mkdir(parents=True, exist_ok=True)
        return shutil.disk_usage(config.staging_dir).free > config.ingest.disk_headroom_gb * 1_000_000_000
    except OSError:
        return True


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

    # The same queue, counted once for videos and once for images.
    by_side = {k: {"queued": 0, "running": 0, "done": 0, "failed": 0} for k in kinds.KINDS}
    for row in store.conn.execute(
        """SELECT status, json_extract(payload_json, '$.filename') AS filename FROM jobs
           WHERE workspace_id = ? AND kind = ? AND status IN ('queued', 'running', 'done', 'failed')""",
        (store.workspace_id, KIND_INDEX_SOURCE),
    ):
        by_side[kinds.of_file(row["filename"])][row["status"]] += 1
    jobs = [{**dict(r), "media": kinds.of_file(r["filename"])} for r in recent]

    return {
        "request": request,
        "workspace": state.config,
        "stats": stats,
        "jobs": jobs,
        "by_side": [{"key": k, "label": kinds.LABELS[k], **by_side[k],
                     "indexed": store.count_shots(media_kind=k),
                     "review": store.count_shots("needs_review", media_kind=k)} for k in kinds.KINDS],
        "cost_so_far": store.total_cost(),
        "cost_remaining": per_file * stats.outstanding,
        "shots": store.count_shots(),
        "needs_review": store.count_shots("needs_review"),
        "worker_running": state.worker is not None,
        "poll": stats.outstanding > 0,
    }
