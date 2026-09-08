"""Bounded async worker pool over the SQLite job queue.

Concurrency defaults to 4. Failures get exponential backoff and three attempts
before a job is marked failed; a failed job never blocks the rest of the run.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

from ..config import WorkspaceConfig
from ..db.models import Job
from ..db.store import Store
from ..ingest.pipeline import IngestPipeline, IngestResult
from ..ingest.scanner import DiscoveredFile
from .queue import KIND_INDEX_SOURCE, MAX_ATTEMPTS, backoff_delay

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, Job, IngestResult | None], None]


@dataclass
class WorkerStats:
    done: int = 0
    failed: int = 0
    retried: int = 0
    deduped: int = 0
    shots: int = 0
    cost_usd: float = 0.0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at


class Worker:
    def __init__(
        self,
        config: WorkspaceConfig,
        store: Store,
        pipeline: IngestPipeline,
        concurrency: int | None = None,
        on_progress: ProgressFn | None = None,
    ):
        self.config = config
        self.store = store
        self.pipeline = pipeline
        self.concurrency = concurrency or config.ingest.concurrency
        self.on_progress = on_progress
        self.stats = WorkerStats()
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run(self, drain: bool = True, poll_interval: float = 0.5) -> WorkerStats:
        """Work the queue until it is empty (drain) or until stopped."""
        requeued = self.store.reset_stale_jobs()
        if requeued:
            log.info("requeued %d job(s) left running by a previous worker", requeued)

        semaphore = asyncio.Semaphore(self.concurrency)
        tasks: set[asyncio.Task] = set()

        while not self._stop.is_set():
            job = self.store.claim_job()
            if job is None:
                if tasks:
                    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    tasks = {t for t in tasks if not t.done()}
                    continue
                if drain:
                    break
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=poll_interval)
                except asyncio.TimeoutError:
                    pass
                continue

            task = asyncio.create_task(self._run_job(job, semaphore))
            tasks.add(task)
            task.add_done_callback(tasks.discard)

            if len(tasks) >= self.concurrency:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = {t for t in tasks if not t.done()}

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return self.stats

    async def _run_job(self, job: Job, semaphore: asyncio.Semaphore) -> None:
        async with semaphore:
            self._notify("started", job, None)
            try:
                result = await self._dispatch(job)
            except Exception as exc:  # noqa: BLE001 - the queue decides what is fatal
                self._handle_failure(job, f"{type(exc).__name__}: {exc}")
                return

            if result.error and result.status == "failed":
                self._handle_failure(job, result.error)
                return

            self.store.finish_job(job.id, "done", cost=result.cost_usd)
            self.stats.done += 1
            self.stats.shots += result.shots_analysed
            self.stats.cost_usd += result.cost_usd
            if result.deduped:
                self.stats.deduped += 1
            self._notify("finished", job, result)

    async def _dispatch(self, job: Job) -> IngestResult:
        if job.kind != KIND_INDEX_SOURCE:
            raise ValueError(f"Unknown job kind {job.kind!r}")
        payload = job.payload
        discovered = DiscoveredFile(
            origin=payload.get("origin", "local"),
            path=Path(payload["path"]) if payload.get("path") else None,
            filename=payload.get("filename", ""),
            drive_file_id=payload.get("drive_file_id"),
            origin_path=payload.get("origin_path"),
        )
        return await self.pipeline.ingest(discovered, force=bool(payload.get("force")))

    def _handle_failure(self, job: Job, error: str) -> None:
        if job.attempts < MAX_ATTEMPTS:
            delay = backoff_delay(job.attempts)
            self.store.retry_job(job.id, error, delay)
            self.stats.retried += 1
            log.warning("job %s failed (attempt %d), retrying in %.0fs: %s",
                        job.id[:8], job.attempts, delay, error)
            self._notify("retrying", job, None)
            return
        self.store.finish_job(job.id, "failed", error=error)
        self.stats.failed += 1
        log.error("job %s failed permanently: %s", job.id[:8], error)
        self._notify("failed", job, None)

    def _notify(self, event: str, job: Job, result: IngestResult | None) -> None:
        if self.on_progress:
            self.on_progress(event, job, result)
