"""Bounded async worker pool over the SQLite job queue.

Concurrency defaults to 4. Failures get exponential backoff and three attempts
before a job is marked failed; a failed job never blocks the rest of the run.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

from .. import attention
from ..config import WorkspaceConfig
from ..db.models import Job
from ..db.store import Store
from ..ingest.pipeline import IngestPipeline, IngestResult
from ..ingest.scanner import DiscoveredFile
from ..analysis.providers.base import TransientProviderError, classify_error, is_environment_error
from ..analysis.usage import UsageLedger, UsageTally, current_tally
from .queue import KIND_INDEX_SOURCE, backoff_delay, max_attempts

log = logging.getLogger(__name__)

#: How long to wait before trying again after a problem with the setup (sign-in, API key, daily quota).
ENVIRONMENT_PAUSE_S = 300

#: How often the spend caps are looked at again. A cap is checked before a file is claimed, and the queue
#: is polled twice a second, so the ledger is read at most this often.
CAP_CHECK_EVERY_S = 5.0
#: After a cap stops the worker it looks again this often, in case the cap was raised in the config.
CAP_RECHECK_S = 60.0

ProgressFn = Callable[[str, Job, IngestResult | None], None]


@dataclass
class WorkerStats:
    done: int = 0
    failed: int = 0
    retried: int = 0
    deduped: int = 0
    skipped: int = 0
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
        gate: asyncio.Semaphore | None = None,
        idle_poll_s: float = 0.5,
    ):
        self.config = config
        self.store = store
        self.pipeline = pipeline
        self.concurrency = concurrency or config.ingest.concurrency
        self.on_progress = on_progress
        # Shared by every client's worker in one server, to cap how many files
        # the whole machine indexes at once (a laptop running it all day).
        self.gate = gate
        self.idle_poll_s = idle_poll_s
        self.stats = WorkerStats()
        self._stop = asyncio.Event()
        # Set when every file is failing for the same reason outside the file (see is_environment_error).
        self.paused_until = 0.0
        self.pause_reason: str | None = None
        # What the model has really cost (see analysis/usage.py), and the moment this run began: the queue
        # going from empty to not empty. The caps are checked against it.
        self.ledger = UsageLedger.for_config(config)
        self.run_started_at: datetime | None = None
        self._cap_checked_at = 0.0
        self._cap_reason: str | None = None

    def stop(self) -> None:
        self._stop.set()

    def spend_cap_reached(self) -> str | None:
        """Why no more files should be started - the real spend passed a cap - or None.

        Two caps, both in dollars as Google billed them: `max_run_usd` since this run began and
        `max_day_usd` over the last 24 hours (the one that also holds across a restart). A file that is
        already running finishes; none is claimed after the cap is passed.
        """
        now = time.monotonic()
        if now - self._cap_checked_at < CAP_CHECK_EVERY_S:
            return self._cap_reason
        self._cap_checked_at = now
        ingest = self.config.ingest
        reason = None
        try:
            if ingest.max_run_usd and self.run_started_at is not None:
                spent = self.ledger.spent(self.run_started_at)
                if spent >= ingest.max_run_usd:
                    reason = (
                        f"This run has cost ${spent:.2f}, which is at its cap of ${ingest.max_run_usd:.2f} "
                        "(ingest.max_run_usd)."
                    )
            if reason is None and ingest.max_day_usd:
                spent = self.ledger.spent_last(24)
                if spent >= ingest.max_day_usd:
                    reason = (
                        f"The last 24 hours have cost ${spent:.2f}, which is at the daily cap of "
                        f"${ingest.max_day_usd:.2f} (ingest.max_day_usd)."
                    )
        except Exception as exc:  # noqa: BLE001 - a broken ledger must not stop the queue silently either way
            log.warning("could not read the spend ledger: %s", exc)
            return self._cap_reason
        self._cap_reason = reason
        return reason

    def _reload_caps(self) -> None:
        """Pick up a cap raised in config.yaml while the worker waits, without a restart."""
        try:
            from ..config import load_workspace_config

            fresh = load_workspace_config(self.config.id).ingest
            self.config.ingest.max_run_usd = fresh.max_run_usd
            self.config.ingest.max_day_usd = fresh.max_day_usd
        except Exception:  # noqa: BLE001 - keep the caps already known
            pass

    async def run(self, drain: bool = True, poll_interval: float | None = None) -> WorkerStats:
        """Work the queue until it is empty (drain) or until stopped."""
        poll_interval = self.idle_poll_s if poll_interval is None else poll_interval
        try:
            from ..drive.fetcher import sweep_partial_downloads

            swept = sweep_partial_downloads(self.config)
            if swept:
                log.info("removed %d unfinished download(s) left by a previous run", swept)
        except Exception as exc:  # noqa: BLE001 - tidying up must never stop the worker
            log.debug("could not sweep partial downloads: %s", exc)
        requeued = self.store.reset_stale_jobs()
        if requeued:
            log.info("requeued %d job(s) left running by a previous worker", requeued)

        semaphore = asyncio.Semaphore(self.concurrency)
        tasks: set[asyncio.Task] = set()

        while not self._stop.is_set():
            if time.monotonic() < self.paused_until:
                if drain:
                    break  # a one-off command should end, saying why, rather than wait for a person
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=min(poll_interval * 10, 5.0))
                except asyncio.TimeoutError:
                    pass
                continue
            capped = self.spend_cap_reached()
            if capped:
                if self.pause_reason != capped:
                    log.error("indexing stopped by a spend cap: %s", capped)
                self.pause_reason = capped
                self.paused_until = time.monotonic() + CAP_RECHECK_S
                self._cap_checked_at = 0.0
                self._reload_caps()
                continue
            if self.pause_reason and time.monotonic() >= self.paused_until:
                self.pause_reason = None
            # Take the machine-wide slot before claiming, so a job waiting for
            # its turn stays "queued" instead of looking like it is running. The
            # wait is in short steps so a stop is noticed, and a worker told to
            # stop while it waited never claims another job.
            if self.gate is not None:
                try:
                    await asyncio.wait_for(self.gate.acquire(), timeout=poll_interval)
                except asyncio.TimeoutError:
                    continue
                if self._stop.is_set():
                    self.gate.release()
                    break
            job = self.store.claim_job()
            if job is None:
                if self.gate is not None:
                    self.gate.release()
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

            if self.run_started_at is None:
                self.run_started_at = datetime.now(UTC)  # the queue was empty and is not any more
            task = asyncio.create_task(self._run_job(job, semaphore))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
            if self.gate is not None:
                task.add_done_callback(lambda _done: self.gate.release())

            if len(tasks) >= self.concurrency:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = {t for t in tasks if not t.done()}

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return self.stats

    async def _run_job(self, job: Job, semaphore: asyncio.Semaphore) -> None:
        async with semaphore:
            self._notify("started", job, None)
            # Every model call this job makes adds itself to this tally (a context variable, and each job
            # is its own task), so a job's cost is what Google billed - including for an attempt that fails.
            tally = UsageTally()
            current_tally.set(tally)
            try:
                result = await self._dispatch(job)
            except asyncio.CancelledError:
                # Stopped mid-job (shutdown, Ctrl-C): hand it back rather than
                # leaving it "running" under an owner that is about to vanish.
                self.store.release_job(job.id, "worker stopped mid-job", cost=tally.cost_usd)
                raise
            except Exception as exc:  # noqa: BLE001 - the queue decides what is fatal
                self._handle_failure(
                    job, f"{type(exc).__name__}: {exc}",
                    transient=isinstance(exc, TransientProviderError), cost=tally.cost_usd,
                )
                await self._mirror_attention()
                return

            if tally.calls:
                result.cost_usd = tally.cost_usd  # what was billed, not what a formula guessed

            if result.error and result.status == "failed":
                self._handle_failure(
                    job, result.error,
                    transient=classify_error(result.error) is TransientProviderError,
                    cost=tally.cost_usd,
                )
                return

            if result.status == "skipped":
                # Not a failure: it is on the "Needs attention" list, with the reason.
                self.store.finish_job(job.id, "done", error="; ".join(result.messages) or None,
                                      cost=tally.cost_usd)
                self.stats.skipped += 1
                self._notify("finished", job, result)
                await self._mirror_attention()
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
        # Only passed when asked for, so a pipeline that predates it (or a test double) still works.
        extra = {"allow_long": True} if payload.get("allow_long") else {}
        return await self.pipeline.ingest(
            self._discovered(job),
            force=bool(payload.get("force")),
            overwrite_corrections=bool(payload.get("overwrite_corrections")),
            **extra,
        )

    @staticmethod
    def _discovered(job: Job) -> DiscoveredFile:
        payload = job.payload
        return DiscoveredFile(
            origin=payload.get("origin", "local"),
            path=Path(payload["path"]) if payload.get("path") else None,
            filename=payload.get("filename", ""),
            drive_file_id=payload.get("drive_file_id"),
            origin_path=payload.get("origin_path"),
            size_bytes=payload.get("size_bytes"),
            duration_s=payload.get("duration_s"),
            link=payload.get("link"),
        )

    async def _mirror_attention(self) -> None:
        """Put a shortcut to each listed file in Drive's "_Needs Attention" folder, when Drive is
        connected and filing is on. A courtesy: a failure here never touches the job."""
        session = getattr(getattr(self.pipeline, "_organise", None), "__self__", None)
        if session is None or not hasattr(session, "mirror_attention"):
            return
        if not self.config.ingest.auto_organise:
            return
        try:
            await asyncio.to_thread(session.mirror_attention)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not mirror the needs-attention list to Drive: %s", exc)

    def _handle_failure(self, job: Job, error: str, transient: bool = False, cost: float = 0.0) -> None:
        if is_environment_error(error):
            # Nothing is wrong with this file. Hand it back untouched, mark nothing failed, and stop
            # claiming work for a while: the next try may be after the problem is fixed.
            self.store.release_job(job.id, error, cost=cost, delay_s=ENVIRONMENT_PAUSE_S * 2)
            self.pause_reason = error[:300]
            self.paused_until = time.monotonic() + ENVIRONMENT_PAUSE_S
            log.error("indexing paused for %d s: %s", ENVIRONMENT_PAUSE_S, error[:300])
            self._notify("paused", job, None)
            return
        limit = max_attempts(transient)
        spent = float(getattr(job, "cost_estimate_usd", 0.0) or 0.0) + cost
        cap = self.config.ingest.max_job_usd
        if cap and job.attempts >= 2 and spent >= cap and job.attempts < limit:
            # A hopeless file must not be paid for again and again: stop here, and say why.
            error = f"{error} (given up after {job.attempts} attempts that cost ${spent:.2f}, the per-file cap)"
            limit = job.attempts
        if job.attempts < limit:
            delay = backoff_delay(job.attempts, transient)
            self.store.retry_job(job.id, error, delay, cost=cost)
            self.stats.retried += 1
            log.warning("job %s failed (attempt %d of %d), retrying in %.0fs: %s",
                        job.id[:8], job.attempts, limit, delay, error)
            self._notify("retrying", job, None)
            return
        self.store.finish_job(job.id, "failed", error=error, cost=cost)
        self.stats.failed += 1
        log.error("job %s failed permanently: %s", job.id[:8], error)
        try:
            # Listed with a link, not just left as a red line in the queue.
            attention.flag_file(
                self.store, self._discovered(job), attention.classify_failure(error), error
            )
        except Exception as exc:  # noqa: BLE001 - the list is a courtesy
            log.warning("could not list %s as needing attention: %s", job.id[:8], exc)
        self._notify("failed", job, None)

    def _notify(self, event: str, job: Job, result: IngestResult | None) -> None:
        if self.on_progress:
            self.on_progress(event, job, result)
