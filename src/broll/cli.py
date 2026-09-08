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
from .analysis.embedder import get_embedder
from .analysis.providers.registry import VISION_PROVIDERS, check_credentials
from .analysis.schema import ShotContext, find_oov
from .db.models import Shot, Source, Workspace
from .db.store import Registry, Store, new_id
from .ingest.frames import best_frame, save_thumbnail
from .ingest.hashing import content_hash
from .ingest.pipeline import IngestPipeline
from .ingest.probe import NotAVideoError, probe
from .ingest.scanner import scan_local
from .jobs.queue import enqueue_files, queue_stats
from .jobs.worker import Worker
from .search.filters import SearchFilters
from .drive.organizer import Organizer
from .search.query import SearchEngine

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
def index(
    path: Optional[Path] = typer.Argument(None, exists=True, readable=True,
                                          help="A video file or a folder of them."),
    drive_folder: Optional[str] = typer.Option(None, "--drive-folder",
                                               help="Index footage already in a Drive folder (M3)."),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    force: bool = typer.Option(False, "--force", help="Re-analyse even if the content hash is known."),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Work the queue now, or just enqueue."),
    concurrency: Optional[int] = typer.Option(None, "--concurrency", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plan and touch nothing."),
    organise_after: bool = typer.Option(False, "--organise",
                                        help="File the results into Drive when indexing finishes."),
) -> None:
    """Queue footage for indexing and, unless --no-wait, work the queue."""
    workspace_config = resolve_workspace(workspace)
    if (path is None) == (drive_folder is None):
        _fail("Give either a path or --drive-folder.")

    if drive_folder:
        from .ingest.scanner import scan_drive_folder

        files = scan_drive_folder(_drive_client(workspace_config), drive_folder)
    else:
        files = scan_local(path)

    if not files:
        _fail(f"No video files found at {path}")

    if dry_run:
        _echo(f"Would queue {len(files)} file(s) into workspace {workspace_config.id!r}:")
        for discovered in files:
            _echo(f"  {discovered.origin:<7} {discovered.origin_path or discovered.filename}")
        _echo("Nothing was written. Drop --dry-run to run it.")
        return

    try:
        check_credentials(workspace_config)
    except Exception as exc:
        _fail(str(exc))

    workspace_config.ensure_dirs()
    store = Store.for_config(workspace_config)
    try:
        jobs = enqueue_files(store, files, force=force)
        skipped = len(files) - len(jobs)
        _echo(f"Queued {len(jobs)} file(s)" + (f", {skipped} already queued" if skipped else ""))
        if not wait:
            _echo("Run `broll work` to process the queue.")
            return
        _run_worker(workspace_config, store, concurrency)
        if organise_after:
            client = _drive_client(workspace_config)
            _report_organise(Organizer(workspace_config, store, client).reorganise(), False)
    finally:
        store.close()


@app.command()
def work(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    concurrency: Optional[int] = typer.Option(None, "--concurrency", "-c"),
    follow: bool = typer.Option(False, "--follow", "-f",
                                help="Keep running and wait for new jobs."),
) -> None:
    """Work the job queue. Safe to kill and restart - it resumes cleanly."""
    workspace_config = resolve_workspace(workspace)
    try:
        check_credentials(workspace_config)
    except Exception as exc:
        _fail(str(exc))
    store = Store.for_config(workspace_config)
    try:
        _run_worker(workspace_config, store, concurrency, drain=not follow)
    finally:
        store.close()


def _run_worker(workspace_config, store, concurrency: int | None, drain: bool = True) -> None:
    embedder = _load_embedder(workspace_config)
    if embedder is not None:
        # Load the model once here rather than racing to load it in four
        # worker threads at the same time.
        embedder.warm_up()
    pipeline = IngestPipeline(workspace_config, store, embedder=embedder)

    def on_progress(event: str, job, result) -> None:
        name = job.payload.get("filename", job.id[:8])
        if event == "started":
            _echo(f"  ... {name}")
        elif event == "finished" and result is not None:
            if result.deduped:
                _echo(f"  dup {name} - already indexed as {result.source_id}")
            else:
                _echo(
                    f"  {result.status:<12} {name} "
                    f"({result.shots_analysed} shot(s), ${result.cost_usd:.4f})"
                )
        elif event == "retrying":
            _echo(f"  retry {name}")
        elif event == "failed":
            typer.secho(f"  FAILED {name}", fg=typer.colors.RED)

    worker = Worker(workspace_config, store, pipeline, concurrency, on_progress)
    stats = asyncio.run(worker.run(drain=drain))
    _echo(
        f"Done: {stats.done} file(s), {stats.shots} shot(s), {stats.failed} failed, "
        f"{stats.deduped} deduped, ${stats.cost_usd:.4f} estimated, "
        f"{stats.elapsed_s:.1f}s elapsed"
    )


def _load_embedder(workspace_config, required: bool = False):
    try:
        return get_embedder(workspace_config)
    except Exception as exc:
        if required:
            _fail(str(exc))
        typer.secho(f"  note: embeddings unavailable ({exc}). "
                    "Shots are indexed for keyword search only; run `broll reembed` "
                    "once embeddings work.", fg=typer.colors.YELLOW, err=True)
        return None


@app.command()
def search(
    query: str = typer.Argument("", help="Natural language, e.g. 'calm beach wide shot golden hour'."),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    limit: int = typer.Option(20, "--limit", "-n"),
    as_json: bool = typer.Option(False, "--json"),
    shot_type: list[str] = typer.Option([], "--shot-type"),
    camera_movement: list[str] = typer.Option([], "--camera-movement"),
    mood: list[str] = typer.Option([], "--mood"),
    setting: list[str] = typer.Option([], "--setting"),
    people: list[str] = typer.Option([], "--people", help="none|one|two|small_group|crowd"),
    time_of_day: list[str] = typer.Option([], "--time-of-day"),
    usable_for: list[str] = typer.Option([], "--usable-for"),
    min_duration: Optional[float] = typer.Option(None, "--min-duration"),
    max_duration: Optional[float] = typer.Option(None, "--max-duration"),
    min_width: Optional[int] = typer.Option(None, "--min-width"),
    faces: Optional[bool] = typer.Option(None, "--faces/--no-faces"),
    clean: bool = typer.Option(False, "--clean", help="Exclude anything with a quality flag."),
) -> None:
    """Search the library. Hybrid keyword + vector, fused with RRF."""
    workspace_config = resolve_workspace(workspace)
    store = Store.for_config(workspace_config)
    try:
        embedder = _load_embedder(workspace_config)
        engine = SearchEngine(store, embedder)
        filters = SearchFilters(
            shot_type=list(shot_type), camera_movement=list(camera_movement),
            mood=list(mood), setting=list(setting), people_count=list(people),
            time_of_day=list(time_of_day), usable_for=list(usable_for),
            duration_min_s=min_duration, duration_max_s=max_duration,
            min_width=min_width, has_faces=faces, exclude_flagged=clean,
        )
        results = engine.search(query, filters, limit)
    finally:
        store.close()

    if as_json:
        _echo(json.dumps([r.to_dict() for r in results], indent=2))
        return
    if not results:
        _echo("No matches.")
        return
    for position, result in enumerate(results, start=1):
        link = result.drive_link or result.shot.thumbnail_path or ""
        _echo(
            f"{position:>2}. [{result.score:.4f} {'+'.join(result.matched) or 'browse'}] "
            f"{result.source.original_filename} @{result.timecode} "
            f"({result.shot.duration_s:.1f}s, {result.shot.shot_type})"
        )
        _echo(f"    {result.shot.caption}")
        if link:
            _echo(f"    {link}")


@app.command()
def reembed(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    """Drop and rebuild the vector index. Needed after changing the embedder."""
    workspace_config = resolve_workspace(workspace)
    store = Store.for_config(workspace_config)
    try:
        embedder = _load_embedder(workspace_config, required=True)
        shots = store.list_shots(limit=1_000_000)
        indexable = [s for s in shots if s.search_text]
        if not yes:
            typer.confirm(
                f"Rebuild embeddings for {len(indexable)} shot(s) using "
                f"{embedder.signature}?",
                abort=True,
            )
        store.vectors.rebuild(embedder.dimensions)
        batch_size = 64
        for start in range(0, len(indexable), batch_size):
            batch = indexable[start:start + batch_size]
            texts = [store.recompute_search_text(s.id) for s in batch]
            vectors = embedder.embed(texts)
            for shot, vector in zip(batch, vectors):
                store.vectors.upsert(shot.id, vector)
            _echo(f"  embedded {min(start + batch_size, len(indexable))}/{len(indexable)}")
        _echo(f"Rebuilt {store.vectors.count()} vector(s) via the "
              f"{store.vectors.backend} backend.")
    finally:
        store.close()


@app.command()
def status(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Queue state, library counts and cost."""
    workspace_config = resolve_workspace(workspace)
    store = Store.for_config(workspace_config)
    try:
        stats = queue_stats(store)
        sources = {
            r["status"]: r["n"]
            for r in store.conn.execute(
                "SELECT status, COUNT(*) AS n FROM sources WHERE workspace_id = ? GROUP BY status",
                (workspace_config.id,),
            ).fetchall()
        }
        spent = store.total_cost()
        done_cost = store.conn.execute(
            "SELECT AVG(cost_estimate_usd) FROM jobs"
            " WHERE workspace_id = ? AND status = 'done' AND cost_estimate_usd > 0",
            (workspace_config.id,),
        ).fetchone()[0]
        per_file = float(done_cost) if done_cost else 0.0
        remaining = per_file * stats.outstanding

        payload = {
            "workspace": workspace_config.id,
            "jobs": stats.__dict__,
            "sources": sources,
            "shots": store.count_shots(),
            "needs_review": store.count_shots("needs_review"),
            "vectors": store.vectors.count(),
            "vector_backend": store.vectors.backend,
            "cost_usd_so_far": round(spent, 4),
            "cost_usd_estimated_remaining": round(remaining, 4),
        }
        if as_json:
            _echo(json.dumps(payload, indent=2))
            return

        _echo(f"workspace: {workspace_config.id} ({workspace_config.name})")
        _echo(f"jobs:      queued={stats.queued} running={stats.running} "
              f"done={stats.done} failed={stats.failed}")
        _echo("sources:   " + (", ".join(f"{k}={v}" for k, v in sources.items()) or "none"))
        _echo(f"shots:     {payload['shots']} total, {payload['needs_review']} need review")
        _echo(f"vectors:   {payload['vectors']} ({payload['vector_backend']} backend)")
        _echo(f"cost:      ${spent:.4f} spent, ~${remaining:.4f} remaining "
              f"({stats.outstanding} file(s) outstanding)")
        candidates = store.vocabulary_candidates(min_count=2)
        if candidates:
            _echo("vocab candidates: " +
                  ", ".join(f"{c['term']}({c['count']})" for c in candidates[:8]))
    finally:
        store.close()


drive_app = typer.Typer(no_args_is_help=True, help="Google Drive connection.")
app.add_typer(drive_app, name="drive")


def _drive_client(workspace_config):
    from .drive.auth import DriveAuthError, load_credentials
    from .drive.client import DriveClient

    try:
        credentials = load_credentials(workspace_config)
    except DriveAuthError as exc:
        _fail(str(exc))
    if credentials is None:
        _fail(
            f"Drive is not connected for workspace {workspace_config.id!r}. "
            "Run `broll drive login` (see the README walkthrough for creating "
            "the Google Cloud project first)."
        )
    return DriveClient(credentials)


@drive_app.command("login")
def drive_login(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    port: int = typer.Option(0, "--port", help="Local callback port. 0 picks one."),
) -> None:
    """Authorise this workspace against your Google account."""
    workspace_config = resolve_workspace(workspace)
    from .drive.auth import DriveAuthError, login

    try:
        login(workspace_config, port=port)
    except DriveAuthError as exc:
        _fail(str(exc))
    _echo(f"Connected. Token stored at {workspace_config.drive_token_path}")
    typer.secho(
        "  note: Google expires refresh tokens for apps in testing mode after "
        "7 days - re-run this when it stops working.",
        fg=typer.colors.YELLOW,
    )


@drive_app.command("logout")
def drive_logout(workspace: Optional[str] = typer.Option(None, "--workspace", "-w")) -> None:
    """Forget this workspace's Drive token."""
    workspace_config = resolve_workspace(workspace)
    from .drive.auth import logout

    _echo("Token removed." if logout(workspace_config) else "No token stored.")


@drive_app.command("status")
def drive_status(workspace: Optional[str] = typer.Option(None, "--workspace", "-w")) -> None:
    """Show the Drive connection and root folder."""
    workspace_config = resolve_workspace(workspace)
    from .drive.auth import load_credentials

    connected = workspace_config.drive_token_path.exists()
    _echo(f"workspace:   {workspace_config.id}")
    _echo(f"token:       {'present' if connected else 'missing'} "
          f"({workspace_config.drive_token_path})")
    _echo(f"root folder: {workspace_config.drive_root_folder_id or workspace_config.drive_root_folder_name + ' (by name)'}")
    _echo(f"local mount: {workspace_config.drive_local_mount_path or 'not set - NLE exports will import offline'}")
    if not connected:
        return
    try:
        credentials = load_credentials(workspace_config)
    except Exception as exc:
        typer.secho(f"token invalid: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    _echo("credentials: " + ("valid" if credentials and credentials.valid else "refreshable"))


@app.command()
def organise(
    source_id: Optional[str] = typer.Argument(None, help="One source; omit for the whole library."),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plan and touch nothing."),
) -> None:
    """Upload, rename and file footage into the Drive tree. Idempotent."""
    workspace_config = resolve_workspace(workspace)
    store = Store.for_config(workspace_config)
    try:
        client = _drive_client(workspace_config)
        organizer = Organizer(workspace_config, store, client, dry_run=dry_run)
        report = (
            organizer.organise_source(source_id) if source_id else organizer.reorganise()
        )
        _report_organise(report, dry_run)
    finally:
        store.close()


@app.command()
def reorganise(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Rebuild the whole shortcut tree from the database.

    The escape hatch when the taxonomy changes: the database is the source of
    truth and Drive is a rendering of it.
    """
    workspace_config = resolve_workspace(workspace)
    store = Store.for_config(workspace_config)
    try:
        client = _drive_client(workspace_config)
        report = Organizer(workspace_config, store, client, dry_run=dry_run).reorganise()
        _report_organise(report, dry_run)
    finally:
        store.close()


def _report_organise(report, dry_run: bool) -> None:
    for action in report.actions:
        _echo(f"  {action}")
    for error in report.errors:
        typer.secho(f"  ! {error}", fg=typer.colors.YELLOW)
    _echo(("Would apply: " if dry_run else "Applied: ") + report.summary())
    if dry_run and report.actions:
        _echo("Nothing was written. Drop --dry-run to apply it.")


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


if __name__ == "__main__":
    app()
