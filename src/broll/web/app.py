"""FastAPI app: server-rendered Jinja2 + HTMX, no build step, no npm.

A single deployable process. The web UI is a client of the same code the CLI
calls - every action here has a CLI equivalent, and the queue it feeds is the
same SQLite queue `broll work` drains.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..analysis.embedder import get_embedder
from ..config import WorkspaceConfig
from ..db.store import Store
from ..ingest.pipeline import IngestPipeline, drive_organise_callable
from ..jobs.worker import Worker

log = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


def url_with(request, **updates) -> str:
    """This page's address with some query parameters changed, as a *relative* link ("/search?q=...").

    Never absolute: behind the HTTPS reverse proxy the app sees plain http, and a link built from
    request.url pointed at http:// on an https:// page. The browser blocks that silently, so tabs and
    "show more" buttons did nothing. A parameter set to None is dropped; a list replaces every value.
    """
    from urllib.parse import urlencode

    items = [(k, v) for k, v in request.query_params.multi_items() if k not in updates]
    for key, value in updates.items():
        if value is None:
            continue
        for item in value if isinstance(value, (list, tuple)) else [value]:
            items.append((key, str(item)))
    return request.url.path + (f"?{urlencode(items)}" if items else "")


templates.env.globals["url_with"] = url_with
# Lets the layout show a Sign out button only when the shared-password gate is on.
templates.env.globals["access_enabled"] = lambda: bool(os.environ.get("BROLL_ACCESS_PASSWORD"))


class AppState:
    """Everything the routes need, built once at startup."""

    def __init__(self, config: WorkspaceConfig, run_worker: bool = True, gentle: "Gentle | None" = None):
        self.config = config
        self.run_worker = run_worker
        self.gentle = gentle or Gentle()
        self.embedder = None
        self.worker: Worker | None = None
        self.worker_task: asyncio.Task | None = None
        self.worker_store: Store | None = None
        self.drive_organise = None  # shared by the worker and the Top Picks star
        self.sync_task: asyncio.Task | None = None
        self.watch_task: asyncio.Task | None = None
        self.backup_task: asyncio.Task | None = None
        self.sync_status: dict[str, object] = {"state": "off", "message": "", "at": None}
        # Transcript runs live in memory: a run is cheap to redo, and
        # persisting a whole timeline would be a schema for one screen.
        self.runs: dict[str, object] = {}

    def store(self) -> Store:
        """A fresh connection per request. SQLite connections are cheap."""
        return Store.for_config(self.config)

    async def start_worker(self, embedder=None) -> None:
        # The embedding model is ~400 MB resident, so a studio serving eight
        # clients loads it once and hands the same one to every client.
        if embedder is not None:
            self.embedder = embedder
        if not self.run_worker:
            return
        if self.embedder is None:
            try:
                self.embedder = get_embedder(self.config)
                if not self.gentle.lazy_model:
                    await asyncio.to_thread(self.embedder.warm_up)
            except Exception as exc:  # keyword search still works without embeddings
                log.warning("embeddings unavailable: %s", exc)
                self.embedder = None

        self.worker_store = self.store()
        self.drive_organise = drive_organise_callable(self.config)
        pipeline = IngestPipeline(
            self.config, self.worker_store, embedder=self.embedder,
            organise=self.drive_organise,
        )
        self.worker = Worker(self.config, self.worker_store, pipeline,
                             gate=self.gentle.gate, idle_poll_s=self.gentle.idle_poll_s)
        self.worker_task = asyncio.create_task(self.worker.run(drain=False))

    def start_watching(self) -> None:
        """Watch this client's drop folders, if it has any."""
        from ..ingest.watcher import watch_loop, watched_dirs

        if self.watch_task is not None or not self.run_worker:
            return
        if not watched_dirs(self.config, create=True):
            return
        self.watch_task = asyncio.create_task(watch_loop(self.config, self.store))

    def start_dashboard_sync(self) -> None:
        """Keep the dashboard's Footage index current, for as long as the app runs."""
        if self.sync_task is None:
            self.sync_task = asyncio.create_task(self._sync_loop())

    def start_backups(self) -> None:
        """A hosted install keeps a few recent copies of the database. Off unless asked for."""
        from ..backup import backups_enabled

        if self.backup_task is None and backups_enabled(self.config):
            self.backup_task = asyncio.create_task(self._backup_loop())

    async def _backup_loop(self) -> None:
        from ..backup import backup_database, list_backups

        interval = max(1, self.config.backup.interval_h) * 3600
        while True:
            try:
                recent = list_backups(self.config)
                fresh = recent and (datetime.now(UTC).timestamp() - recent[0].stat().st_mtime) < interval
                if not fresh:
                    await asyncio.to_thread(backup_database, self.config)
            except Exception as exc:  # noqa: BLE001 - a backup problem must not stop the app
                log.warning("database backup failed: %s", exc)
            await asyncio.sleep(3600)

    async def _sync_loop(self) -> None:
        from datetime import UTC, datetime

        from ..sync.dashboard import DashboardSync, DashboardSyncError, is_connected

        last_error = None
        while True:
            interval = max(15, self.config.dashboard.interval_s)
            if is_connected(self.config):
                try:
                    result = await asyncio.to_thread(DashboardSync(self.config).run)
                    self.sync_status = {"state": "ok", "message": result.summary(),
                                        "at": datetime.now(UTC).isoformat(timespec="seconds")}
                    last_error = None
                except DashboardSyncError as exc:
                    self.sync_status = {"state": "error", "message": str(exc),
                                        "at": datetime.now(UTC).isoformat(timespec="seconds")}
                    if str(exc) != last_error:  # say it once, not every minute
                        log.warning("dashboard sync: %s", exc)
                        last_error = str(exc)
                except Exception as exc:  # noqa: BLE001 - a sync problem must not stop the app
                    self.sync_status = {"state": "error", "message": f"{type(exc).__name__}: {exc}",
                                        "at": datetime.now(UTC).isoformat(timespec="seconds")}
                    log.warning("dashboard sync failed: %s", exc)
            else:
                self.sync_status = {"state": "off", "message": "", "at": None}
            await asyncio.sleep(interval)

    async def stop_worker(self) -> None:
        if self.watch_task:
            self.watch_task.cancel()
        if self.sync_task:
            self.sync_task.cancel()
        if self.backup_task:
            self.backup_task.cancel()
        if self.worker:
            self.worker.stop()
        if self.worker_task:
            try:
                await asyncio.wait_for(self.worker_task, timeout=30)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self.worker_task.cancel()
        if self.worker_store:
            self.worker_store.close()


class Gentle:
    """How hard a server works in the background. The defaults are the old behaviour.

    A Mac that runs the library all day for Editing Joe wants it quiet: at most
    `max_jobs` files indexed at once across every client (None: each library's
    own concurrency), the embedding model loaded when first needed rather than
    at start, and an idle worker checking the queue every `idle_poll_s`.
    """

    def __init__(self, max_jobs: int | None = None, lazy_model: bool = False, idle_poll_s: float = 0.5):
        self.max_jobs = max_jobs
        self.lazy_model = lazy_model
        self.idle_poll_s = idle_poll_s
        self._gate: asyncio.Semaphore | None = None

    @property
    def gate(self) -> asyncio.Semaphore | None:
        # Made on first use, inside the running event loop that will share it.
        if self.max_jobs and self._gate is None:
            self._gate = asyncio.Semaphore(self.max_jobs)
        return self._gate


class Studio:
    """Every client library this server serves, and which one a request means.

    One process, one embedding model, one worker per client. A client is a
    workspace: its own database, its own Drive tree, its own folder structure.
    """

    def __init__(self, configs: list[WorkspaceConfig], run_worker: bool = True,
                 gentle: Gentle | None = None):
        if not configs:
            raise ValueError("A studio needs at least one client library.")
        self.gentle = gentle or Gentle()
        self.order = [config.id for config in configs]
        self.clients: dict[str, AppState] = {
            config.id: AppState(config, run_worker=run_worker, gentle=self.gentle) for config in configs
        }
        self.embedder = None

    @property
    def default_id(self) -> str:
        return self.order[0]

    def state(self, client_id: str | None) -> AppState:
        """The runtime for this client, or the default when the id is unknown."""
        return self.clients.get(client_id or "", self.clients[self.default_id])

    def summaries(self) -> list[dict[str, str]]:
        return [
            {"id": cid, "name": self.clients[cid].config.name} for cid in self.order
        ]

    async def start(self) -> None:
        first = self.clients[self.default_id]
        if first.run_worker:
            try:
                self.embedder = get_embedder(first.config)
                if not self.gentle.lazy_model:
                    await asyncio.to_thread(self.embedder.warm_up)
            except Exception as exc:  # keyword search still works without embeddings
                log.warning("embeddings unavailable: %s", exc)
                self.embedder = None
        for state in self.clients.values():
            shared = self.embedder if state.config.embedder == first.config.embedder else None
            await state.start_worker(embedder=shared)
            state.start_watching()
            state.start_dashboard_sync()
            state.start_backups()

    async def add_client(self, config: WorkspaceConfig) -> AppState:
        """Serve a library created while the server is running, straight away:
        its worker, drop folder and dashboard sync start as they would at boot."""
        if config.id in self.clients:
            return self.clients[config.id]
        config.ensure_dirs()
        first = self.clients[self.default_id]
        state = AppState(config, run_worker=first.run_worker, gentle=self.gentle)
        self.clients[config.id] = state
        self.order.append(config.id)
        shared = self.embedder if config.embedder == first.config.embedder else None
        await state.start_worker(embedder=shared)
        state.start_watching()
        state.start_dashboard_sync()
        state.start_backups()
        return state

    async def stop(self) -> None:
        for state in self.clients.values():
            await state.stop_worker()


def client_state(request: Request) -> AppState:
    """The client library this request is about.

    Resolved once per request by the middleware below, from ?client= or the
    cookie that remembers the last one used.
    """
    return request.app.state.studio.state(getattr(request.state, "client_id", None))


CLIENT_COOKIE = "broll_client"


def create_app(config: WorkspaceConfig, run_worker: bool = True) -> FastAPI:
    """One client library. The same app as a studio of one."""
    return create_studio_app([config], run_worker=run_worker)


def create_studio_app(configs: list[WorkspaceConfig], run_worker: bool = True,
                      gentle: Gentle | None = None) -> FastAPI:
    for config in configs:
        config.ensure_dirs()
    studio = Studio(configs, run_worker=run_worker, gentle=gentle)
    config = configs[0]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await studio.start()
        try:
            yield
        finally:
            await studio.stop()

    title = (
        f"B-Roll Librarian - {config.name}" if len(configs) == 1
        else f"B-Roll Librarian - {len(configs)} clients"
    )
    app = FastAPI(title=title, lifespan=lifespan)
    app.state.studio = studio
    # Kept so anything holding the old handle still reaches a working client.
    app.state.broll = studio.clients[studio.default_id]

    @app.middleware("http")
    async def pick_client(request: Request, call_next):
        """Which client's library this request is about.

        ?client= wins and is remembered in a cookie, so every link and form on
        the page after it stays inside that client.
        """
        chosen = request.query_params.get("client") or request.cookies.get(CLIENT_COOKIE)
        if chosen not in studio.clients:
            chosen = studio.default_id
        request.state.client_id = chosen
        request.state.client = studio.clients[chosen].config
        request.state.clients = studio.summaries()
        response = await call_next(request)
        if request.query_params.get("client") in studio.clients:
            response.set_cookie(CLIENT_COOKIE, chosen, max_age=60 * 60 * 24 * 365,
                                httponly=True, samesite="lax")
        return response

    @app.middleware("http")
    async def same_origin_only(request, call_next):
        """The app has no login, so a web page you happen to visit must not be able to drive it.

        A browser always says where a cross-site form post came from (Origin, or Referer as a
        fallback); if that isn't this app's own address, the request is refused. Requests with
        neither header (curl, the CLI, tests) are not browser cross-site posts and pass.
        """
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            from urllib.parse import urlparse

            source = request.headers.get("origin") or request.headers.get("referer")
            if source and urlparse(source).netloc != request.headers.get("host", ""):
                from fastapi.responses import PlainTextResponse

                return PlainTextResponse("Cross-site request refused.", status_code=403)
        return await call_next(request)

    from . import access

    access.install(app)  # /healthz, and the optional shared-password login

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    @app.get("/thumbnails/{name}")
    async def thumbnail(request: Request, name: str):
        """Thumbnails live per client, so the active client is looked in first.

        Shot ids are unique across clients, so a thumbnail referenced while
        another client is active still resolves rather than 404ing.
        """
        from fastapi.responses import FileResponse, Response

        if "/" in name or "\\" in name or name.startswith("."):
            return Response(status_code=404)
        active = client_state(request).config
        for candidate in (active, *(s.config for s in studio.clients.values())):
            path = candidate.thumbnails_dir / name
            if path.is_file():
                return FileResponse(path, headers={"Cache-Control": "private, max-age=86400"})
        return Response(status_code=404)

    from .routes import (
        api, attention, drive_connect, ingest, library, review, search, settings, transcript,
    )

    app.include_router(library.router)
    app.include_router(search.router)
    app.include_router(ingest.router)
    app.include_router(transcript.router)
    app.include_router(review.router)
    app.include_router(attention.router)
    app.include_router(settings.router)
    app.include_router(drive_connect.router)
    app.include_router(api.router)

    @app.exception_handler(404)
    async def not_found(request: Request, exc):  # noqa: ANN001
        # An agent calling the API needs the reason, not a page of HTML.
        if request.url.path.startswith("/api/"):
            from fastapi.responses import JSONResponse

            detail = getattr(exc, "detail", "Not found.")
            return JSONResponse({"detail": detail}, status_code=404)
        return HTMLResponse(
            templates.get_template("404.html").render({"request": request}), status_code=404
        )

    return app
