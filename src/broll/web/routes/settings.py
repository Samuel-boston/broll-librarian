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
from ...config import PROVIDER_KEY_ENV, broll_home
from ...analysis.providers.registry import VISION_PROVIDERS
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
    path = broll_home() / ".env"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text().splitlines() if path.exists() else []
    replaced = False
    for index, line in enumerate(lines):
        if line.split("=", 1)[0].strip() == name:
            lines[index] = f"{name}={value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{name}={value}")
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o600)
    return path


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
    }
