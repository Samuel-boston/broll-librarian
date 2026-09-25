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
from .analysis.schema import VOCABULARIES, ShotContext, find_oov
from .db.models import Shot, Source, Workspace
from .db.store import Registry, Store, new_id
from .ingest.frames import best_frame, save_thumbnail
from .ingest.hashing import content_hash
from .ingest.pipeline import IngestPipeline, drive_organise_callable
from .ingest.probe import NotAVideoError, probe
from .ingest.scanner import DiscoveredFile, scan_local
from .jobs.queue import enqueue_files, queue_stats
from .jobs.worker import Worker
from .review import CorrectionError, apply_correction, review_queue
from .search.filters import SearchFilters
from .analysis.providers.registry import get_text_provider
from .drive.organizer import Organizer
from .search.query import SearchEngine
from .transcript.exporters import csv_export, edl, fcp7xml
from .transcript.exporters.base import build_timeline
from .transcript.matcher import TranscriptMatcher
from .transcript.parser import parse_and_segment

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="B-Roll Librarian - turn unsorted footage into a searchable, organised library.",
)

WORKSPACE_ENV = "BROLL_WORKSPACE"


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
        _echo("[ok]   sqlite can load extensions - vector search will use sqlite-vec")
    else:
        _echo(
            "[ok]   sqlite cannot load extensions here, so vector search uses the "
            "exact numpy backend instead of sqlite-vec (same results; fine up to "
            "tens of thousands of shots)"
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

    from .sync.dashboard import DashboardSync, DashboardSyncError, is_connected

    if is_connected(workspace_config):
        try:
            DashboardSync(workspace_config).check()
            _echo("[ok]   dashboard connected - the Footage index updates automatically")
        except DashboardSyncError as exc:
            ok = False
            typer.secho(f"[fail] dashboard: {exc}", fg=typer.colors.RED)
    else:
        _echo("[info] dashboard not connected (optional): run `broll connect-dashboard`")

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
        media_kind=meta.media_kind,
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


@app.command("connect-dashboard")
def connect_dashboard(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    url: Optional[str] = typer.Option(None, "--url", help="The dashboard's Supabase Project URL."),
    key: Optional[str] = typer.Option(
        None, "--key", help="The service_role key. Leave out to be asked (input is hidden)."),
) -> None:
    """Connect this library to the Content Ops dashboard, then sync it.

    Once connected, the running app keeps the dashboard's Footage index current
    on its own. Nothing else needs to be run.
    """
    from .sync.dashboard import KEY_ENV, URL_ENV, DashboardSync, DashboardSyncError

    workspace_config = resolve_workspace(workspace)
    url = (url or os.environ.get(URL_ENV) or workspace_config.dashboard.supabase_url
           or typer.prompt("Dashboard Supabase Project URL (https://xxxx.supabase.co)")).strip().rstrip("/")
    key = (key or os.environ.get(KEY_ENV)
           or typer.prompt("Dashboard service_role key (hidden)", hide_input=True)).strip()

    workspace_config.dashboard.enabled = True
    workspace_config.dashboard.supabase_url = url
    os.environ[KEY_ENV] = key
    sync = DashboardSync(workspace_config)
    try:
        sync.check()
    except DashboardSyncError as exc:
        _fail(f"Not connected: {exc}")

    workspace_config.save()
    cfg.write_env_var(KEY_ENV, key)
    _echo("Connected. First sync...")
    try:
        _echo(sync.run().summary())
    except DashboardSyncError as exc:
        _fail(f"Connected, but the first sync failed: {exc}")
    _echo("From now on the running app keeps the dashboard up to date by itself.")


@app.command()
def sync(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    force: bool = typer.Option(False, "--force", help="Send everything again."),
    no_prune: bool = typer.Option(False, "--no-prune",
                                  help="Keep dashboard rows for shots the library no longer has."),
) -> None:
    """Push the index to the dashboard now (the running app does this automatically)."""
    from .sync.dashboard import DashboardSync, DashboardSyncError

    workspace_config = resolve_workspace(workspace)
    try:
        result = DashboardSync(workspace_config).run(prune=not no_prune, force=force)
    except DashboardSyncError as exc:
        _fail(str(exc))
    _echo(result.summary())
    for message in result.errors:
        typer.secho(f"  {message}", fg=typer.colors.YELLOW)


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
    organise = drive_organise_callable(workspace_config)
    pipeline = IngestPipeline(workspace_config, store, embedder=embedder, organise=organise)
    if organise:
        _echo("  Drive connected - each clip is filed into Drive as soon as it is indexed.")

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
                    + (" -> Drive" if result.organised else "")
                )
                for message in result.messages:
                    if "Drive" in message:
                        typer.secho(f"      {message}", fg=typer.colors.YELLOW)
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
    _sync_dashboard_quietly(workspace_config)


def _sync_dashboard_quietly(workspace_config) -> None:
    """Push the index to the dashboard when connected. A failure never fails the run."""
    from .sync.dashboard import DashboardSync, DashboardSyncError, is_connected

    if not is_connected(workspace_config):
        return
    try:
        _echo(DashboardSync(workspace_config).run().summary())
    except Exception as exc:  # noqa: BLE001 - syncing is a courtesy; it must never fail a finished run
        typer.secho(f"  Dashboard sync failed: {exc}", fg=typer.colors.YELLOW)


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
    emotion: list[str] = typer.Option([], "--emotion", help="e.g. --emotion calm --emotion grounded"),
    category: list[str] = typer.Option([], "--folder", help="A folder, or a parent folder, in tree mode."),
    featured: bool = typer.Option(False, "--featured", help="Only shots of the client's featured person."),
    top_picks: bool = typer.Option(False, "--top-picks", help="Only starred shots."),
    loose: bool = typer.Option(False, "--loose", help="Also show near matches."),
    exact: bool = typer.Option(False, "--exact", help="Don't correct spelling."),
) -> None:
    """Search the library. Hybrid keyword + vector, fused with RRF."""
    workspace_config = resolve_workspace(workspace)
    store = Store.for_config(workspace_config)
    try:
        embedder = _load_embedder(workspace_config)
        engine = SearchEngine(store, embedder, featured_person=workspace_config.client.featured_person)
        filters = SearchFilters(
            shot_type=list(shot_type), camera_movement=list(camera_movement),
            mood=list(mood), setting=list(setting), people_count=list(people),
            time_of_day=list(time_of_day), usable_for=list(usable_for),
            duration_min_s=min_duration, duration_max_s=max_duration,
            min_width=min_width, has_faces=faces, exclude_flagged=clean,
            emotions=list(emotion), category=list(category),
            featured_person=True if featured else None,
            top_pick=True if top_picks else None,
        )
        results = engine.search(query, filters, limit, strict=not loose, correct=not exact)
    finally:
        store.close()

    if as_json:
        _echo(json.dumps([r.to_dict() for r in results], indent=2))
        return
    hidden = engine.hidden_count
    if engine.corrections:
        fixes = ", ".join(f"{typed} -> {fixed}" for typed, fixed in engine.corrections)
        _echo(f"Showing results for: {engine.corrected_query}   ({fixes}; --exact to search as typed)")
    if not results:
        _echo("No matches." + (f" {hidden} near match(es) hidden - add --loose." if hidden else ""))
        return
    for position, result in enumerate(results, start=1):
        link = result.drive_link or result.shot.thumbnail_path or ""
        _echo(
            f"{position:>2}. [{result.score:.4f} {'+'.join(result.matched) or 'browse'}] "
            f"{result.source.original_filename} @{result.timecode} "
            f"({result.shot.duration_s:.1f}s, {result.shot.shot_type})"
        )
        _echo(f"    {'★ ' if result.shot.top_pick else ''}{result.shot.caption}")
        if result.shot.category:
            _echo(f"    folder:   {result.shot.category.replace('/', ' > ')}")
        if result.shot.emotions:
            _echo(f"    emotions: {', '.join(result.shot.emotions)}")
        if result.shot.mood:
            _echo(f"    mood:     {', '.join(result.shot.mood)}")
        if result.shot.tags:
            _echo(f"    tags:     {', '.join(result.shot.tags)}")
        if link:
            _echo(f"    {link}")
    if hidden:
        _echo(f"\n({hidden} near match(es) hidden - add --loose to see them.)")


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


@app.command()
def serve(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    worker: bool = typer.Option(True, "--worker/--no-worker",
                                help="Run the ingest worker inside the web process."),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Serve the web UI (and, by default, the ingest worker) on one process."""
    workspace_config = resolve_workspace(workspace)
    try:
        import uvicorn
    except ImportError:
        _fail("The web UI needs the extra: pip install 'broll-librarian[web]'")

    from .web.app import create_app

    _echo(f"Serving {workspace_config.name!r} on http://{host}:{port}")
    if not worker:
        _echo("Worker disabled - run `broll work --follow` separately.")
    uvicorn.run(create_app(workspace_config, run_worker=worker),
                host=host, port=port, reload=reload)


@app.command()
def transcript(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True,
                                help="An .srt, .vtt or plain-text transcript."),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    out: Optional[Path] = typer.Option(None, "--out", "-o",
                                       help="Directory for the exports. Omit to only print."),
    formats: str = typer.Option("fcp7,edl,csv", "--format", "-f",
                                help="Comma-separated: fcp7, edl, csv."),
    name: Optional[str] = typer.Option(None, "--name", help="Sequence name."),
    no_rerank: bool = typer.Option(False, "--no-rerank",
                                   help="Skip the model rerank and use search order."),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Match a transcript to B-roll and export a timeline."""
    workspace_config = resolve_workspace(workspace)
    store = Store.for_config(workspace_config)
    try:
        beats = parse_and_segment(
            path.read_text(encoding="utf-8", errors="replace"),
            path.name,
            workspace_config.transcript.words_per_minute,
            workspace_config.transcript.beat_min_s,
            workspace_config.transcript.beat_max_s,
        )
        if not beats:
            _fail(f"No narration found in {path.name}.")

        embedder = _load_embedder(workspace_config)
        engine = SearchEngine(store, embedder, featured_person=workspace_config.client.featured_person)
        text_provider = None
        if not no_rerank:
            try:
                text_provider = get_text_provider(workspace_config)
            except Exception as exc:
                typer.secho(f"  note: rerank unavailable ({exc}); using search order.",
                            fg=typer.colors.YELLOW, err=True)

        matcher = TranscriptMatcher(workspace_config, engine, text_provider)
        matches = asyncio.run(matcher.match(beats))
        timeline = build_timeline(matches, workspace_config,
                                  name=name or path.stem.replace("_", " ").title())
    finally:
        store.close()

    if as_json:
        _echo(json.dumps(_transcript_payload(matches, timeline), indent=2))
    else:
        _print_transcript(matches, timeline)

    if out:
        written = _write_exports(matches, timeline, out, formats, path.stem)
        for target in written:
            _echo(f"wrote {target}")


def _print_transcript(matches, timeline) -> None:
    for match in matches:
        beat = match.beat
        header = f"[{beat.timecode}] {beat.text}"
        _echo(header)
        if match.no_good_match:
            typer.secho(f"    no good match - {match.missing_footage or 'nothing suitable'}",
                        fg=typer.colors.YELLOW)
            continue
        for rank, suggestion in enumerate(match.suggestions, start=1):
            marker = "->" if rank == 1 else "  "
            _echo(f"  {marker} {suggestion.source.original_filename} "
                  f"@{suggestion.shot.start_s:.1f}s ({suggestion.shot.duration_s:.1f}s)"
                  f"{' [reused]' if suggestion.reused else ''}")
            if suggestion.reason:
                _echo(f"       {suggestion.reason}")
        _echo("")

    _echo(f"sequence: {timeline.fps} fps, {len(timeline.items)} clip(s), "
          f"{len(timeline.gaps)} gap(s)")
    for warning in timeline.warnings:
        typer.secho(f"  ! {warning}", fg=typer.colors.YELLOW)
    if timeline.gaps:
        _echo("\nFootage you should go and shoot:")
        for gap in timeline.gaps:
            _echo(f"  [{gap.beat.timecode}] {gap.missing_footage or gap.beat.text}")


def _transcript_payload(matches, timeline) -> dict:
    return {
        "sequence": {
            "fps": timeline.fps,
            "clips": len(timeline.items),
            "gaps": len(timeline.gaps),
            "warnings": timeline.warnings,
        },
        "beats": [
            {
                "index": m.beat.index,
                "start_s": m.beat.start_s,
                "end_s": m.beat.end_s,
                "text": m.beat.text,
                "no_good_match": m.no_good_match,
                "missing_footage": m.missing_footage,
                "suggestions": [
                    {
                        "filename": s.source.original_filename,
                        "shot_id": s.shot.id,
                        "start_s": s.shot.start_s,
                        "duration_s": s.shot.duration_s,
                        "caption": s.shot.caption,
                        "reason": s.reason,
                        "confidence": s.confidence,
                        "reused": s.reused,
                        "drive_link": s.drive_link,
                    }
                    for s in m.suggestions
                ],
            }
            for m in matches
        ],
    }


def _write_exports(matches, timeline, out: Path, formats: str, stem: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    wanted = {f.strip().lower() for f in formats.split(",") if f.strip()}
    unknown = wanted - {"fcp7", "fcp7xml", "xml", "edl", "csv"}
    if unknown:
        _fail(f"Unknown export format(s): {', '.join(sorted(unknown))}")

    written: list[Path] = []
    if wanted & {"fcp7", "fcp7xml", "xml"}:
        written.append(fcp7xml.write(timeline, out / f"{stem}.xml"))
    if "edl" in wanted:
        written.append(edl.write(timeline, out / f"{stem}.edl"))
    if "csv" in wanted:
        written.append(csv_export.write(matches, timeline, out / f"{stem}.csv"))
    return written


@app.command()
def review(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    limit: int = typer.Option(50, "--limit", "-n"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """List shots that need a human look."""
    workspace_config = resolve_workspace(workspace)
    store = Store.for_config(workspace_config)
    try:
        queue = review_queue(store, limit, workspace_config.ingest.review_below_confidence)
    finally:
        store.close()

    if as_json:
        _echo(json.dumps(
            [
                {
                    "shot_id": entry["shot"].id,
                    "filename": entry["filename"],
                    "start_s": entry["shot"].start_s,
                    "caption": entry["shot"].caption,
                    "reasons": entry["reasons"],
                }
                for entry in queue
            ],
            indent=2,
        ))
        return

    if not queue:
        _echo("Nothing needs review.")
        return
    for entry in queue:
        shot = entry["shot"]
        _echo(f"{shot.id}  {entry['filename']} @{shot.start_s:.1f}s")
        _echo(f"    {shot.caption or '(no caption)'}")
        _echo(f"    why: {'; '.join(entry['reasons'])}")
    _echo(f"\n{len(queue)} shot(s) awaiting review. "
          "Correct one with: broll fix <shot_id> --set setting=beach")


@app.command()
def fix(
    shot_id: str = typer.Argument(...),
    set_: list[str] = typer.Option([], "--set", "-s",
                                   help="field=value, repeatable. Lists take commas."),
    status: str = typer.Option("indexed", "--status",
                               help="Status to leave the shot in."),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
) -> None:
    """Correct a shot's facets by hand. Recomputes search text and embedding."""
    workspace_config = resolve_workspace(workspace)
    updates: dict[str, str] = {}
    for item in set_:
        if "=" not in item:
            _fail(f"--set expects field=value, got {item!r}")
        field, value = item.split("=", 1)
        updates[field.strip()] = value
    if not updates:
        _fail("Nothing to change. Use --set field=value.")

    store = Store.for_config(workspace_config)
    try:
        embedder = _load_embedder(workspace_config)
        shot = apply_correction(workspace_config, store, shot_id, updates, embedder, status)
    except CorrectionError as exc:
        store.close()
        _fail(str(exc))
    else:
        _echo(f"{shot.id}: {shot.caption}")
        _echo(f"  status={shot.status} setting={shot.setting} action={shot.action} "
              f"shot_type={shot.shot_type}")
        _echo("  search text and embedding recomputed.")
    finally:
        store.close()


@app.command()
def vocab(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    min_count: int = typer.Option(2, "--min-count", help="Only show terms seen this often."),
    promote: list[str] = typer.Option([], "--promote",
                                      help="field=term, repeatable. Adds it to the vocabulary."),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Review and promote out-of-vocabulary terms the model keeps returning."""
    workspace_config = resolve_workspace(workspace)
    store = Store.for_config(workspace_config)
    try:
        for item in promote:
            if "=" not in item:
                _fail(f"--promote expects field=term, got {item!r}")
            field, term = (part.strip() for part in item.split("=", 1))
            if field not in VOCABULARIES:
                _fail(f"Unknown vocabulary {field!r}. One of: {', '.join(VOCABULARIES)}")
            existing = workspace_config.vocabulary_overrides.setdefault(field, [])
            if term not in existing:
                existing.append(term)
            store.promote_vocabulary_candidate(field, term)
            _echo(f"promoted {field}: {term}")
        if promote:
            workspace_config.save()
            _echo(f"saved to {workspace_config.config_path}")

        candidates = store.vocabulary_candidates(min_count=min_count)
    finally:
        store.close()

    if as_json:
        _echo(json.dumps(candidates, indent=2))
        return
    if not candidates:
        _echo(f"No out-of-vocabulary terms seen {min_count}+ times.")
        return
    for candidate in candidates:
        _echo(f"{candidate['count']:>4}x  {candidate['field']:<14} {candidate['term']}")
    _echo("\nPromote one with: broll vocab --promote subjects=hydrofoil")


@app.command()
def costs(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    project: int = typer.Option(0, "--project", help="Project the cost of indexing N more clips."),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """What indexing has cost so far, from real usage - not a brochure figure."""
    workspace_config = resolve_workspace(workspace)
    store = Store.for_config(workspace_config)
    try:
        total = store.total_cost()
        done = store.conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(cost_estimate_usd), 0) AS spent"
            " FROM jobs WHERE workspace_id = ? AND status = 'done'",
            (workspace_config.id,),
        ).fetchone()
        shots = store.count_shots()
        by_day = store.conn.execute(
            """SELECT date(finished_at) AS day, COUNT(*) AS files,
                      SUM(cost_estimate_usd) AS spent
               FROM jobs WHERE workspace_id = ? AND finished_at IS NOT NULL
               GROUP BY day ORDER BY day DESC LIMIT 14""",
            (workspace_config.id,),
        ).fetchall()
    finally:
        store.close()

    files = done["n"] or 0
    per_file = (done["spent"] / files) if files else 0.0
    per_shot = (total / shots) if shots else 0.0
    payload = {
        "workspace": workspace_config.id,
        "provider": workspace_config.provider.vision,
        "model": workspace_config.provider.resolved_vision_model(),
        "files_indexed": files,
        "shots_indexed": shots,
        "total_usd": round(total, 4),
        "per_file_usd": round(per_file, 5),
        "per_shot_usd": round(per_shot, 5),
        "by_day": [dict(r) for r in by_day],
    }
    if project:
        payload["projection"] = {
            "clips": project,
            "usd": round(per_file * project, 2) if per_file else None,
        }

    if as_json:
        _echo(json.dumps(payload, indent=2))
        return

    _echo(f"provider:  {payload['provider']} ({payload['model']})")
    _echo(f"indexed:   {files} file(s), {shots} shot(s)")
    _echo(f"spent:     ${total:.4f}")
    _echo(f"per file:  ${per_file:.5f}")
    _echo(f"per shot:  ${per_shot:.5f}")
    if by_day:
        _echo("recent:")
        for row in by_day:
            _echo(f"  {row['day']}  {row['files']:>4} file(s)  ${row['spent'] or 0:.4f}")
    if project:
        if per_file:
            _echo(f"\nIndexing {project} more clips would cost about "
                  f"${per_file * project:.2f} at this workspace's measured rate.")
        else:
            _echo("\nNo measured cost yet - index some clips first.")


@app.command()
def reanalyse(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    source_id: Optional[str] = typer.Option(None, "--source", help="Just this source."),
    stale: bool = typer.Option(True, "--stale/--all",
                               help="Only rows analysed with an older prompt version."),
    wait: bool = typer.Option(True, "--wait/--no-wait"),
    overwrite_corrections: bool = typer.Option(
        False, "--overwrite-corrections",
        help="Also re-analyse shots a human has corrected. Off by default.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Re-run analysis, e.g. after the prompt improved. Never automatic."""
    workspace_config = resolve_workspace(workspace)
    store = Store.for_config(workspace_config)
    try:
        sources = store.list_sources(limit=1_000_000)
        if source_id:
            sources = [s for s in sources if s.id == source_id]
            if not sources:
                _fail(f"No source {source_id!r}.")
        elif stale:
            sources = [s for s in sources if s.analysis_version != PROMPT_VERSION]

        targets = [s for s in sources if s.origin_path and Path(s.origin_path).exists()]
        missing = len(sources) - len(targets)

        if not targets:
            _echo(f"Nothing to re-analyse (prompt version {PROMPT_VERSION})."
                  + (f" {missing} source(s) have no local file." if missing else ""))
            return

        _echo(f"{len(targets)} source(s) to re-analyse at prompt version {PROMPT_VERSION}."
              + (f" Skipping {missing} with no local file." if missing else ""))
        if dry_run:
            for source in targets[:20]:
                _echo(f"  {source.original_filename} (was {source.analysis_version})")
            if len(targets) > 20:
                _echo(f"  ... and {len(targets) - 20} more")
            _echo("Nothing was written. Drop --dry-run to run it.")
            return

        files = [
            DiscoveredFile(origin="local", path=Path(s.origin_path), filename=s.original_filename,
                           origin_path=s.origin_path)
            for s in targets
        ]
        jobs = enqueue_files(store, files, force=True,
                             overwrite_corrections=overwrite_corrections)
        _echo(f"Queued {len(jobs)} job(s)."
              + ("" if overwrite_corrections else " Operator-corrected shots are kept."))
        if wait:
            _run_worker(workspace_config, store, None)
    finally:
        store.close()


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


@app.command()
def retry(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    wait: bool = typer.Option(True, "--wait/--no-wait"),
    concurrency: Optional[int] = typer.Option(None, "--concurrency", "-c"),
) -> None:
    """Put failed jobs back in the queue - e.g. after a run of provider 503s."""
    workspace_config = resolve_workspace(workspace)
    store = Store.for_config(workspace_config)
    try:
        requeued = store.requeue_failed_jobs()
        if not requeued:
            _echo("No failed jobs to retry.")
            return
        _echo(f"Requeued {requeued} job(s).")
    finally:
        store.close()

    if not wait:
        return
    try:
        check_credentials(workspace_config)
    except Exception as exc:
        _fail(str(exc))
    store = Store.for_config(workspace_config)
    try:
        _run_worker(workspace_config, store, concurrency, drain=True)
    finally:
        store.close()


@app.command()
def reset(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    yes: bool = typer.Option(False, "--yes", help="Required: this cannot be undone."),
) -> None:
    """Empty the local library: every source, shot, job, vector and thumbnail.

    Drive is not touched. Files already filed there stay where they are, so
    clear them by hand first if you want a blank Drive too.
    """
    workspace_config = resolve_workspace(workspace)
    if not yes:
        _fail("This deletes the whole local index for this workspace. Re-run with --yes.")

    from .library_admin import clear_library, running_jobs

    store = Store.for_config(workspace_config)
    try:
        if running_jobs(store):
            _fail("A file is being indexed right now. Let it finish (or stop the app), then try again.")
        result = clear_library(workspace_config, store)
    finally:
        store.close()

    _echo(
        f"Cleared {result.sources} source(s), {result.shots} shot(s), "
        f"{result.jobs} job(s), {result.thumbnails} thumbnail(s), {result.staged} staged file(s)."
    )
    if result.dashboard_removed:
        _echo(f"Removed {result.dashboard_removed} shot(s) from the dashboard.")
    if result.dashboard_error:
        typer.secho(f"The dashboard was not updated: {result.dashboard_error}", fg=typer.colors.YELLOW)
    _echo("Drive was not touched.")


@app.command()
def remove(
    source_id: str = typer.Argument(..., help="The file's id (see `broll show`)."),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    yes: bool = typer.Option(False, "--yes", help="Required: this cannot be undone."),
) -> None:
    """Remove one file, and all its shots, from the library. Drive is not touched."""
    from .library_admin import remove_source

    workspace_config = resolve_workspace(workspace)
    if not yes:
        _fail("This removes the file and its shots from the library. Re-run with --yes.")
    store = Store.for_config(workspace_config)
    try:
        result = remove_source(workspace_config, store, source_id)
    finally:
        store.close()
    if result is None:
        _fail(f"No file with id {source_id!r} in this workspace.")
    _echo(f"Removed {result.filename} ({result.shots} shot(s)). Drive was not touched.")
    if result.dashboard_error:
        typer.secho(f"The dashboard was not updated: {result.dashboard_error}", fg=typer.colors.YELLOW)


@app.command()
def cancel(
    job_id: Optional[str] = typer.Argument(None, help="The job's id. Leave out with --all."),
    all_waiting: bool = typer.Option(False, "--all", help="Clear the whole queue."),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
) -> None:
    """Take one job, or every waiting job, out of the queue. A job already running finishes."""
    from .library_admin import cancel_job, clear_queue

    workspace_config = resolve_workspace(workspace)
    if not job_id and not all_waiting:
        _fail("Give a job id, or --all to clear the queue.")
    store = Store.for_config(workspace_config)
    try:
        result = clear_queue(workspace_config, store) if all_waiting else cancel_job(workspace_config, store, job_id)
    finally:
        store.close()
    _echo(f"Removed {result.cancelled} job(s) from the queue.")
    if result.still_running:
        _echo(f"{result.still_running} job(s) already running will finish.")


if __name__ == "__main__":
    app()
