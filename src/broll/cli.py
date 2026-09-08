"""Typer entrypoint.

The CLI is the real interface; the web UI is a client of it. Every action the
UI can perform must also exist as a command here.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer

from . import config as cfg
from .analysis.analyzer import Analyzer
from .analysis.prompt import PROMPT_VERSION
from .analysis.providers.registry import VISION_PROVIDERS, check_credentials
from .analysis.schema import ShotContext, find_oov
from .db.models import Shot, Source, Workspace
from .db.store import Registry, Store, new_id
from .ingest.frames import best_frame, save_thumbnail
from .ingest.hashing import content_hash
from .ingest.probe import NotAVideoError, probe

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="B-Roll Librarian - turn unsorted footage into a searchable, organised library.",
)

WORKSPACE_ENV = "BROLL_WORKSPACE"
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mts", ".mxf", ".wmv"}


def _echo(message: str, err: bool = False) -> None:
    typer.echo(message, err=err)


def _fail(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def resolve_workspace(workspace: str | None) -> cfg.WorkspaceConfig:
    """Explicit flag, then $BROLL_WORKSPACE, then the only workspace there is."""
    cfg.load_env()
    registry = Registry()
    try:
        name = workspace or os.environ.get(WORKSPACE_ENV)
        if name is None:
            only = registry.default_workspace()
            if only is None:
                count = len(registry.list())
                if count == 0:
                    _fail("No workspaces yet. Run: broll init --name \"My Library\"")
                _fail(
                    f"{count} workspaces exist - choose one with --workspace, "
                    f"or set {WORKSPACE_ENV}."
                )
            name = only.id
        if registry.get(name) is None:
            _fail(f"No workspace {name!r}. Run `broll workspaces` to list them.")
    finally:
        registry.close()
    return cfg.load_workspace_config(name)


# --------------------------------------------------------------------------
# Workspace management
# --------------------------------------------------------------------------


@app.command()
def init(
    name: str = typer.Option(..., "--name", "-n", help="Human-readable workspace name."),
    workspace_id: Optional[str] = typer.Option(None, "--id", help="Defaults to a slug of the name."),
    provider: str = typer.Option("gemini", "--provider", "-p", help=f"One of {', '.join(VISION_PROVIDERS)}."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Provider model override."),
    drive_folder: Optional[str] = typer.Option(None, "--drive-folder", help="Existing Drive root folder ID."),
) -> None:
    """Create a workspace: its config, its database, and a registry entry."""
    cfg.load_env()
    if provider not in VISION_PROVIDERS:
        _fail(f"Unknown provider {provider!r}. Choose one of {', '.join(VISION_PROVIDERS)}.")

    ws_id = workspace_id or cfg.slugify_id(name)
    registry = Registry()
    if registry.get(ws_id) is not None:
        registry.close()
        _fail(f"Workspace {ws_id!r} already exists.")

    workspace_config = cfg.WorkspaceConfig(id=ws_id, name=name)
    workspace_config.provider.vision = provider
    workspace_config.provider.vision_model = model
    workspace_config.drive_root_folder_id = drive_folder
    workspace_config.save()

    Store.for_config(workspace_config).close()  # creates and migrates library.db
    registry.create(
        Workspace(
            id=ws_id,
            name=name,
            provider=provider,
            db_path=str(workspace_config.db_path),
            drive_root_folder_id=drive_folder,
        )
    )
    registry.close()

    _echo(f"Created workspace {ws_id!r}")
    _echo(f"  config: {workspace_config.config_path}")
    _echo(f"  database: {workspace_config.db_path}")
    if not workspace_config.api_key(provider) and provider != "mock":
        variables = " or ".join(cfg.PROVIDER_KEY_ENV.get(provider, ()))
        typer.secho(
            f"  note: {variables} is not set yet - analysis will fail until it is.",
            fg=typer.colors.YELLOW,
        )


@app.command()
def workspaces() -> None:
    """List workspaces."""
    registry = Registry()
    rows = registry.list()
    registry.close()
    if not rows:
        _echo("No workspaces. Run: broll init --name \"My Library\"")
        return
    for row in rows:
        _echo(f"{row.id:<24} {row.name:<28} provider={row.provider:<10} db={row.db_path}")


@app.command()
def doctor(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
) -> None:
    """Check system dependencies and credentials."""
    cfg.load_env()
    ok = True

    try:
        cfg.check_ffmpeg()
        _echo(f"[ok]   ffmpeg: {cfg.ffmpeg_version()}")
    except cfg.DependencyError as exc:
        ok = False
        typer.secho(f"[fail] {exc}", fg=typer.colors.RED)

    import sqlite3

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        _echo(f"[ok]   sqlite {sqlite3.sqlite_version} with FTS5")
    except sqlite3.OperationalError:
        ok = False
        typer.secho("[fail] this Python's sqlite3 has no FTS5 - keyword search will not work.",
                    fg=typer.colors.RED)
    if hasattr(conn, "enable_load_extension"):
        _echo("[ok]   sqlite extension loading available (sqlite-vec can be used)")
    else:
        typer.secho(
            "[warn] this Python's sqlite3 cannot load extensions, so sqlite-vec is "
            "unavailable. Vector search needs a Python built with "
            "--enable-loadable-sqlite-extensions.",
            fg=typer.colors.YELLOW,
        )
    conn.close()

    from .analysis.embedder import local_embeddings_available

    if local_embeddings_available():
        _echo("[ok]   sentence-transformers installed (local embeddings)")
    else:
        typer.secho(
            "[warn] sentence-transformers not installed - install "
            "'broll-librarian[embeddings-local]' before M2 search.",
            fg=typer.colors.YELLOW,
        )

    try:
        workspace_config = resolve_workspace(workspace)
    except typer.Exit:
        _echo("[warn] no workspace selected - skipping provider checks.")
        raise typer.Exit(code=0 if ok else 1)

    _echo(f"[ok]   workspace {workspace_config.id!r} at {workspace_config.dir}")
    provider = workspace_config.provider.vision
    if workspace_config.api_key(provider) or provider == "mock":
        _echo(f"[ok]   provider {provider} ({workspace_config.provider.resolved_vision_model()})")
    else:
        ok = False
        variables = " or ".join(cfg.PROVIDER_KEY_ENV.get(provider, ()))
        typer.secho(f"[fail] no API key for {provider}: set {variables}", fg=typer.colors.RED)

    raise typer.Exit(code=0 if ok else 1)


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def _single_shot_context(path: Path, meta) -> ShotContext:
    return ShotContext(
        source_filename=path.name,
        duration_s=meta.duration_s,
        width=meta.width,
        height=meta.height,
        fps=meta.fps,
        shot_index=0,
        shot_count=1,
        start_s=0.0,
        end_s=meta.duration_s,
    )


@app.command()
def analyse(
    clip: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Override the workspace provider."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    keep_frames: bool = typer.Option(False, "--keep-frames", help="Leave extracted frames on disk."),
    pretty: bool = typer.Option(True, "--pretty/--compact"),
) -> None:
    """Analyse one video and print the AnalysisResult as JSON. Writes nothing."""
    workspace_config = resolve_workspace(workspace)
    if provider:
        workspace_config.provider.vision = provider
        workspace_config.provider.vision_model = model
    elif model:
        workspace_config.provider.vision_model = model

    try:
        check_credentials(workspace_config)
    except Exception as exc:
        _fail(str(exc))

    try:
        meta = probe(clip)
    except (NotAVideoError, cfg.DependencyError) as exc:
        _fail(str(exc))

    workspace_config.ensure_dirs()
    work_dir = workspace_config.temp_dir / f"analyse-{new_id()[:8]}"
    analyzer = Analyzer(workspace_config)
    context = _single_shot_context(clip, meta)

    outcome = asyncio.run(analyzer.analyse_shot(clip, context, work_dir))

    if outcome.result is None:
        _fail(f"Analysis failed: {outcome.error}")

    payload = {
        "file": str(clip),
        "provider": analyzer.provider.name,
        "model": getattr(analyzer.provider, "model", None),
        "analysis_version": PROMPT_VERSION,
        "estimated_cost_usd": round(outcome.cost_usd, 6),
        "frames_used": len(outcome.frames),
        "status": outcome.status,
        "out_of_vocabulary": [{"field": f, "term": t} for f, t in outcome.oov],
        "probe": meta.model_dump(),
        "analysis": outcome.result.model_dump(mode="json"),
    }
    _echo(json.dumps(payload, indent=2 if pretty else None))

    if not keep_frames:
        for frame in outcome.frames:
            frame.unlink(missing_ok=True)
        if work_dir.exists() and not any(work_dir.iterdir()):
            work_dir.rmdir()
    else:
        _echo(f"frames kept in {work_dir}", err=True)


@app.command()
def index(
    path: Path = typer.Argument(..., exists=True, readable=True),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    force: bool = typer.Option(False, "--force", help="Re-index even if the content hash is known."),
) -> None:
    """Index a local video into the workspace database.

    M1 treats each file as a single shot; shot detection and the job queue land
    in M2, at which point this command gains folder and Drive-folder sources.
    """
    workspace_config = resolve_workspace(workspace)
    try:
        check_credentials(workspace_config)
    except Exception as exc:
        _fail(str(exc))

    files = _collect_videos(path)
    if not files:
        _fail(f"No video files found at {path}")
    if len(files) > 1:
        _echo(
            f"{len(files)} videos found. M1 indexes them one at a time, serially; "
            "the concurrent job queue arrives in M2."
        )

    workspace_config.ensure_dirs()
    store = Store.for_config(workspace_config)
    analyzer = Analyzer(workspace_config)
    total_cost = 0.0

    try:
        for video in files:
            total_cost += _index_one(store, workspace_config, analyzer, video, force)
    finally:
        store.close()

    _echo(f"Estimated cost so far: ${total_cost:.4f}")


def _collect_videos(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    )


def _index_one(store, workspace_config, analyzer, video: Path, force: bool) -> float:
    digest = content_hash(video)
    existing = store.source_by_hash(digest)
    if existing and not force:
        _echo(f"skip  {video.name} - already indexed as {existing.id} ({existing.status})")
        return 0.0

    try:
        meta = probe(video)
    except NotAVideoError as exc:
        _echo(f"skip  {video.name} - {exc}")
        return 0.0

    source = existing or store.insert_source(
        Source(
            id=new_id(),
            workspace_id=workspace_config.id,
            content_hash=digest,
            original_filename=video.name,
            origin="local",
            origin_path=str(video.resolve()),
            duration_s=meta.duration_s,
            width=meta.width,
            height=meta.height,
            fps=meta.fps,
            codec=meta.codec,
            filesize_bytes=meta.filesize_bytes,
            status="analysing",
        )
    )
    store.update_source(source.id, status="analysing", analysis_version=PROMPT_VERSION)

    context = _single_shot_context(video, meta)
    work_dir = workspace_config.temp_dir / f"index-{source.id[:8]}"
    outcome = asyncio.run(analyzer.analyse_shot(video, context, work_dir))

    shot = store.get_shot(f"{source.id}-0") or Shot(
        id=f"{source.id}-0",
        workspace_id=workspace_config.id,
        source_id=source.id,
        shot_index=0,
        is_primary=True,
        start_s=0.0,
        end_s=meta.duration_s,
        duration_s=meta.duration_s,
    )
    is_new = store.get_shot(shot.id) is None

    if outcome.result is not None:
        shot.apply_analysis(outcome.result, PROMPT_VERSION)
        shot.status = outcome.status
        shot.error_message = None
    else:
        shot.status = "needs_review"
        shot.error_message = outcome.error

    keyframe = best_frame(outcome.frames)
    if keyframe:
        thumbnail = workspace_config.thumbnails_dir / f"{shot.id}.jpg"
        save_thumbnail(keyframe, thumbnail, workspace_config.ingest.thumbnail_max_edge)
        shot.thumbnail_path = str(thumbnail)

    store.insert_shot(shot) if is_new else store.update_shot(shot)
    if outcome.oov:
        store.record_vocabulary_candidates(outcome.oov)
    status = store.recompute_source_status(source.id)

    for frame in outcome.frames:
        frame.unlink(missing_ok=True)
    if work_dir.exists() and not any(work_dir.iterdir()):
        work_dir.rmdir()

    caption = (outcome.result.caption if outcome.result else outcome.error) or ""
    _echo(f"{status:<12} {video.name} - {caption}")
    return outcome.cost_usd


@app.command(name="show")
def show_source(
    source_id: str = typer.Argument(...),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
) -> None:
    """Print a stored source and its shots as JSON."""
    workspace_config = resolve_workspace(workspace)
    store = Store.for_config(workspace_config)
    try:
        source = store.get_source(source_id)
        if source is None:
            _fail(f"No source {source_id!r} in workspace {workspace_config.id!r}.")
        payload = {
            "source": source.model_dump(mode="json"),
            "shots": [s.model_dump(mode="json") for s in store.shots_for_source(source_id)],
        }
    finally:
        store.close()
    _echo(json.dumps(payload, indent=2))


@app.command()
def status(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
) -> None:
    """Library and queue counts."""
    workspace_config = resolve_workspace(workspace)
    store = Store.for_config(workspace_config)
    try:
        sources = store.conn.execute(
            "SELECT status, COUNT(*) AS n FROM sources WHERE workspace_id = ? GROUP BY status",
            (workspace_config.id,),
        ).fetchall()
        _echo(f"workspace: {workspace_config.id} ({workspace_config.name})")
        _echo("sources:   " + (", ".join(f"{r['status']}={r['n']}" for r in sources) or "none"))
        _echo(f"shots:     {store.count_shots()} total, "
              f"{store.count_shots('needs_review')} need review")
        jobs = store.job_counts()
        _echo("jobs:      " + (", ".join(f"{k}={v}" for k, v in jobs.items()) or "none"))
        _echo(f"cost:      ${store.total_cost():.4f} estimated so far")
        candidates = store.vocabulary_candidates(min_count=2)
        if candidates:
            top = ", ".join(f"{c['term']}({c['count']})" for c in candidates[:8])
            _echo(f"vocab candidates: {top}")
    finally:
        store.close()


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


if __name__ == "__main__":
    app()
