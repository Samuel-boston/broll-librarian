"""Settings screen.

Non-secret settings are written to the workspace config.yaml. API keys are
written to the .env file under BROLL_HOME with 0600 permissions and are never
rendered back to the page - only whether one is set.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from ...analysis.embedder import local_embeddings_available
from ...analysis.schema import VOCABULARIES
from ...config import PROVIDER_KEY_ENV, broll_home, write_env_var
from ...analysis.providers.registry import VISION_PROVIDERS
from ...sync.dashboard import is_connected
from ..app import templates

router = APIRouter()


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, message: str | None = None):
    return templates.TemplateResponse(
        request=request, name="settings.html",
        context=_context(request, message=message),
    )


@router.post("/settings", response_class=HTMLResponse)
async def save_settings(
    request: Request,
    provider: str = Form(...),
    model: str = Form(""),
    drive_root_folder_id: str = Form(""),
    drive_local_mount_path: str = Form(""),
    min_clips_for_subfolder: int = Form(5),
    max_folders_per_level: int = Form(40),
    words_per_minute: int = Form(150),
    concurrency: int = Form(4),
    requests_per_minute: float = Form(0.0),
    review_below_confidence: float = Form(0.7),
):
    state = request.app.state.broll
    config = state.config
    if provider in VISION_PROVIDERS:
        config.provider.vision = provider
    config.provider.vision_model = model.strip() or None
    config.drive_root_folder_id = drive_root_folder_id.strip() or None
    config.drive_local_mount_path = drive_local_mount_path.strip() or None
    config.taxonomy.min_clips_for_subfolder = max(1, min_clips_for_subfolder)
    config.taxonomy.max_folders_per_level = max(1, max_folders_per_level)
    config.transcript.words_per_minute = max(30, words_per_minute)
    config.ingest.concurrency = max(1, concurrency)
    config.ingest.requests_per_minute = max(0.0, requests_per_minute)
    config.ingest.review_below_confidence = min(1.0, max(0.0, review_below_confidence))
    config.save()

    return templates.TemplateResponse(
        request=request, name="partials/settings_form.html",
        context=_context(request, message=f"Saved to {config.config_path}."),
    )


@router.post("/settings/key", response_class=HTMLResponse)
async def save_key(request: Request, provider: str = Form(...), api_key: str = Form(...)):
    """Write a provider key to BROLL_HOME/.env. It is never rendered back."""
    variables = PROVIDER_KEY_ENV.get(provider, ())
    if not variables:
        return templates.TemplateResponse(
            request=request, name="partials/settings_form.html",
            context=_context(request, message=f"{provider} needs no API key."),
        )

    name = variables[0]
    value = api_key.strip()
    if not value:
        return templates.TemplateResponse(
            request=request, name="partials/settings_form.html",
            context=_context(request, message="No key given - nothing was written."),
        )

    _write_env(name, value)
    os.environ[name] = value
    return templates.TemplateResponse(
        request=request, name="partials/settings_form.html",
        context=_context(request, message=f"{name} saved to {broll_home() / '.env'}."),
    )


@router.post("/settings/dashboard", response_class=HTMLResponse)
async def save_dashboard(request: Request, supabase_url: str = Form(""), service_key: str = Form("")):
    """Connect to the Content Ops dashboard's Supabase and sync straight away."""
    import asyncio

    from ...sync.dashboard import KEY_ENV, DashboardSync, DashboardSyncError

    state = request.app.state.broll
    config = state.config
    url = supabase_url.strip().rstrip("/")
    stored_url = (config.dashboard.supabase_url or "").rstrip("/")
    # The saved key is only ever reused for the address it was saved with. A different address
    # needs the key typed again, so the saved one can't be sent somewhere new.
    key = service_key.strip() or (os.environ.get(KEY_ENV, "") if url == stored_url else "")
    from urllib.parse import urlparse

    parsed = urlparse(url)
    local = parsed.hostname in ("localhost", "127.0.0.1")
    if url and not (parsed.scheme == "https" or (parsed.scheme == "http" and local)):
        return templates.TemplateResponse(
            request=request, name="partials/settings_form.html",
            context=_context(request, message="The Project URL must start with https://."),
        )
    if not url or not key:
        return templates.TemplateResponse(
            request=request, name="partials/settings_form.html",
            context=_context(request, message="Both the Project URL and the service_role key are needed."),
        )

    previous = (config.dashboard.enabled, config.dashboard.supabase_url, os.environ.get(KEY_ENV))
    config.dashboard.enabled = True
    config.dashboard.supabase_url = url
    os.environ[KEY_ENV] = key
    sync = DashboardSync(config)
    try:
        await asyncio.to_thread(sync.check)
    except DashboardSyncError as exc:
        config.dashboard.enabled, config.dashboard.supabase_url = previous[0], previous[1]
        if previous[2] is None:
            os.environ.pop(KEY_ENV, None)
        else:
            os.environ[KEY_ENV] = previous[2]
        return templates.TemplateResponse(
            request=request, name="partials/settings_form.html",
            context=_context(request, message=f"Not connected: {exc}"),
        )

    config.save()
    write_env_var(KEY_ENV, key)
    try:
        result = await asyncio.to_thread(sync.run)
        message = f"Connected. {result.summary()}"
        state.sync_status = {"state": "ok", "message": result.summary(), "at": None}
    except DashboardSyncError as exc:
        message = f"Connected, but the first sync failed: {exc}"
    return templates.TemplateResponse(
        request=request, name="partials/settings_form.html",
        context=_context(request, message=message),
    )


@router.post("/settings/vocab", response_class=HTMLResponse)
async def promote_term(request: Request, field: str = Form(...), term: str = Form(...)):
    state = request.app.state.broll
    if field not in VOCABULARIES:
        return templates.TemplateResponse(
            request=request, name="partials/settings_form.html",
            context=_context(request, message=f"Unknown vocabulary {field!r}."),
        )

    existing = state.config.vocabulary_overrides.setdefault(field, [])
    if term not in existing:
        existing.append(term)
    state.config.save()

    store = state.store()
    try:
        store.promote_vocabulary_candidate(field, term)
    finally:
        store.close()

    return templates.TemplateResponse(
        request=request, name="partials/settings_form.html",
        context=_context(request, message=f"Promoted {term!r} into {field}."),
    )


def _write_env(name: str, value: str) -> Path:
    return write_env_var(name, value)


def _context(request: Request, message: str | None = None) -> dict:
    state = request.app.state.broll
    config = state.config
    store = state.store()
    try:
        candidates = store.vocabulary_candidates(min_count=2)[:30]
        vector_backend = store.vectors.backend
        vectors = store.vectors.count()
    finally:
        store.close()

    keys = {
        name: bool(config.api_key(name))
        for name in VISION_PROVIDERS
        if PROVIDER_KEY_ENV.get(name)
    }
    return {
        "request": request,
        "workspace": config,
        "message": message,
        "providers": VISION_PROVIDERS,
        "keys": keys,
        "key_env": {n: (PROVIDER_KEY_ENV.get(n) or ("-",))[0] for n in VISION_PROVIDERS},
        "drive_connected": config.drive_token_path.exists(),
        "embeddings_local": local_embeddings_available(),
        "vector_backend": vector_backend,
        "vectors": vectors,
        "candidates": candidates,
        "vocabularies": sorted(VOCABULARIES),
        "env_path": broll_home() / ".env",
        "dashboard": {**state.sync_status, "connected": is_connected(config)},
    }
