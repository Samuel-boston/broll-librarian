"""FastAPI app: server-rendered Jinja2 + HTMX, no build step, no npm.

A single deployable process. The web UI is a client of the same code the CLI
calls - every action here has a CLI equivalent, and the queue it feeds is the
same SQLite queue `broll work` drains.
"""

from __future__ import annotations

import asyncio
import logging
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


class AppState:
    """Everything the routes need, built once at startup."""

    def __init__(self, config: WorkspaceConfig, run_worker: bool = True):
        self.config = config
        self.run_worker = run_worker
        self.embedder = None
        self.worker: Worker | None = None
        self.worker_task: asyncio.Task | None = None
        self.worker_store: Store | None = None
        self.drive_organise = None  # shared by the worker and the Top Picks star
        # Transcript runs live in memory: a run is cheap to redo, and
        # persisting a whole timeline would be a schema for one screen.
        self.runs: dict[str, object] = {}

    def store(self) -> Store:
        """A fresh connection per request. SQLite connections are cheap."""
        return Store.for_config(self.config)

    async def start_worker(self) -> None:
        if not self.run_worker:
            return
        try:
            self.embedder = get_embedder(self.config)
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
        self.worker = Worker(self.config, self.worker_store, pipeline)
        self.worker_task = asyncio.create_task(self.worker.run(drain=False))

    async def stop_worker(self) -> None:
        if self.worker:
            self.worker.stop()
        if self.worker_task:
            try:
                await asyncio.wait_for(self.worker_task, timeout=30)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self.worker_task.cancel()
        if self.worker_store:
            self.worker_store.close()


def create_app(config: WorkspaceConfig, run_worker: bool = True) -> FastAPI:
    config.ensure_dirs()
    state = AppState(config, run_worker=run_worker)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await state.start_worker()
        try:
            yield
        finally:
            await state.stop_worker()

    app = FastAPI(title=f"B-Roll Librarian - {config.name}", lifespan=lifespan)
    app.state.broll = state

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.mount(
        "/thumbnails",
        StaticFiles(directory=str(config.thumbnails_dir)),
        name="thumbnails",
    )

    from .routes import ingest, review, search, settings, transcript

    app.include_router(search.router)
    app.include_router(ingest.router)
    app.include_router(transcript.router)
    app.include_router(review.router)
    app.include_router(settings.router)

    @app.exception_handler(404)
    async def not_found(request: Request, exc):  # noqa: ANN001
        return HTMLResponse(
            templates.get_template("404.html").render({"request": request}), status_code=404
        )

    return app
