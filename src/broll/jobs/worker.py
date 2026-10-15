"""Bounded async worker pool over the SQLite job queue.

Concurrency defaults to 4. Failures get exponential backoff and three attempts
before a job is marked failed; a failed job never blocks the rest of the run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections import deque
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
from ..analysis.providers.base import (
    TransientProviderError, classify_error, is_drive_quota_error, is_environment_error,
)
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

# Drive's download quota is a recovering allowance, not a wall: part of the requests get through while the
# rest are refused, on the same file seconds apart. So a refused file is held back on its own and the queue
# carries on, rather than the whole worker stopping for minutes.
#: A refused file waits this long before its next try, doubling each time it is refused again (with jitter)...
QUOTA_RETRY_BASE_S = 180.0
#: ...up to this.
QUOTA_RETRY_MAX_S = 3600.0
#: This many different files refused in a row, with none getting through for QUOTA_QUIET_S, is a real lockout:
#: stop for a while. (A file counts as through when it finishes, which is a minute after its download passed,
#: so refusals alone pile up fast even while one request in six is getting through: measured 15 Oct.)
QUOTA_STREAK_PAUSE = 12
QUOTA_QUIET_S = 300.0
QUOTA_STREAK_PAUSE_S = 600.0
#: For this long after Drive last refused a read, files are started on a schedule that follows how much of
#: what is asked is getting through (the last QUOTA_RATE_WINDOW_S, once there are QUOTA_RATE_MIN_EVENTS of them):
#: * under QUOTA_SLOW_BELOW getting through: one file every 20-30 s, one at a time (2-3 requests a minute);
#: * over QUOTA_FAST_ABOVE: one file every 3-6 s, two at a time at most;
#: * in between, whichever it was doing.
QUOTA_PRESSURE_S = 1800.0
QUOTA_PACE_S = (3.0, 6.0)
QUOTA_SLOW_PACE_S = (20.0, 30.0)
QUOTA_PRESSURE_CONCURRENCY = 2
QUOTA_SLOW_BELOW = 0.15
QUOTA_FAST_ABOVE = 0.30
QUOTA_RATE_WINDOW_S = 600.0
QUOTA_RATE_MIN_EVENTS = 8
#: Under QUOTA_REST_BELOW getting through over QUOTA_REST_WINDOW_S (with at least QUOTA_REST_MIN_EVENTS asked) the
#: worker rests for QUOTA_REST_S, asking Drive for half a megabyte of one waiting file every QUOTA_PROBE_EVERY_S and
#: nothing else. Two probes in a row that pass end the rest early; it then resumes at the slow pace.
QUOTA_REST_BELOW = 0.08
QUOTA_REST_WINDOW_S = 1800.0
QUOTA_REST_MIN_EVENTS = 30
QUOTA_REST_S = 2700.0
QUOTA_PROBE_EVERY_S = 300.0
#: How often the success / refusal ratio is written to the log.
DRIVE_RATIO_EVERY_S = 600.0


def quota_retry_delay(hits: int, rng: random.Random | None = None) -> float:
    """How long a file refused `hits` times waits: 3 min, 6, 12... up to an hour, give or take 30%."""
    base = min(QUOTA_RETRY_BASE_S * 2 ** max(0, hits - 1), QUOTA_RETRY_MAX_S)
    return base * (rng or random).uniform(0.7, 1.3)


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
        # Drive's download quota (see QUOTA_* above): how often each waiting file has been refused, the
        # different files refused since anything last got through, when Drive last refused, and the tally
        # for the ratio line in the log.
        self._quota_hits: dict[str, int] = {}
        self._quota_streak: set[str] = set()
        self._last_through_at = time.monotonic()
        self._last_quota_at: float | None = None
        self._next_start_at = 0.0
        self._drive_ok = 0
        self._drive_blocked = 0
        self._ratio_logged_at = time.monotonic()
        self._finished_at: deque[float] = deque()
        # What Drive did with each file asked of it: (when, True if it got through). The pass rates that
        # decide the pace and the rest are read off this.
        self._events: deque[tuple[float, bool]] = deque()
        self._events_since = time.monotonic()
        self._slow = False
        self._resting_until = 0.0
        self._next_probe_at = 0.0
        self._probes_passed = 0

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

    # -- Drive's download quota ----------------------------------------------------------------------------

    def quota_pressure(self) -> bool:
        """Has Drive refused a read in the last half hour? Then files are started slowly and few at a time."""
        return self._last_quota_at is not None and time.monotonic() - self._last_quota_at < QUOTA_PRESSURE_S

    def _pass_rate(self, window_s: float, min_events: int) -> float | None:
        """How much of what was asked of Drive in the last `window_s` got through (None: too few to say)."""
        now = time.monotonic()
        while self._events and now - self._events[0][0] > QUOTA_REST_WINDOW_S:
            self._events.popleft()
        recent = [ok for at, ok in self._events if now - at <= window_s]
        if len(recent) < min_events:
            return None
        return sum(recent) / len(recent)

    def _pace(self) -> tuple[int, tuple[float, float]] | None:
        """How many files may be in flight and how far apart they start, while Drive is refusing; None if it is not."""
        if not self.quota_pressure():
            return None
        rate = self._pass_rate(QUOTA_RATE_WINDOW_S, QUOTA_RATE_MIN_EVENTS)
        if rate is not None:
            if rate < QUOTA_SLOW_BELOW:
                self._slow = True
            elif rate > QUOTA_FAST_ABOVE:
                self._slow = False
        if self._slow:
            return 1, QUOTA_SLOW_PACE_S
        return min(self.concurrency, QUOTA_PRESSURE_CONCURRENCY), QUOTA_PACE_S

    def _maybe_rest(self) -> None:
        """Start a long rest when next to nothing has got through for half an hour."""
        now = time.monotonic()
        if now < self._resting_until or now - self._events_since < 0.8 * QUOTA_REST_WINDOW_S:
            return
        rate = self._pass_rate(QUOTA_REST_WINDOW_S, QUOTA_REST_MIN_EVENTS)
        if rate is None or rate >= QUOTA_REST_BELOW:
            return
        self._resting_until = now + QUOTA_REST_S
        self._next_probe_at = now + QUOTA_PROBE_EVERY_S
        self._probes_passed = 0
        self._events.clear()
        self._events_since = now
        self._slow = True
        self.pause_reason = (
            f"Drive let {rate:.0%} through over {QUOTA_REST_WINDOW_S / 60:.0f} min: resting {QUOTA_REST_S / 60:.0f} min"
        )
        log.error("indexing resting: %s; asking Drive for half a megabyte every %d min", self.pause_reason,
                  QUOTA_PROBE_EVERY_S // 60)

    async def _rest_step(self) -> None:
        """One step of a rest: a probe when it is due, then a short wait. Ends the rest when Drive recovers."""
        now = time.monotonic()
        if now >= self._next_probe_at:
            self._next_probe_at = now + QUOTA_PROBE_EVERY_S
            result = await self._probe_drive()
            self._probes_passed = self._probes_passed + 1 if result else 0
            log.warning("Drive probe while resting: %s", {True: "half a megabyte came through",
                        False: "refused (download quota)", None: "could not tell"}[result])
            if self._probes_passed >= 2:
                self._resting_until = 0.0
                self.pause_reason = None
                self._events.clear()
                self._events_since = time.monotonic()
                log.warning("Drive is letting reads through again: resuming at the slow pace")
        wait = max(0.05, min(1.0, self._next_probe_at - time.monotonic()))
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=wait)
        except asyncio.TimeoutError:
            pass

    async def _probe_drive(self) -> bool | None:
        """Ask Drive for half a megabyte of one waiting file. Costs no model money and no attempt."""
        from ..ingest.remote import RemoteDriveVideo

        row = self.store.conn.execute(
            "SELECT payload_json FROM jobs WHERE workspace_id = ? AND status = 'queued'"
            " AND json_extract(payload_json, '$.drive_file_id') IS NOT NULL ORDER BY RANDOM() LIMIT 1",
            (self.store.workspace_id,),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload_json"])
        try:
            client = await asyncio.to_thread(self.pipeline._drive_client)
            video = RemoteDriveVideo(client, payload["drive_file_id"], payload.get("filename", ""),
                                     payload.get("size_bytes"))
            return await asyncio.to_thread(video.can_read)
        except Exception as exc:  # noqa: BLE001 - a probe is only a question
            log.warning("could not probe Drive: %s", str(exc)[:200])
            return None

    def _note_through(self, job: Job) -> None:
        """A file got its bytes from Drive: whatever refusals came before, it is not a lockout."""
        self._quota_hits.pop(job.id, None)
        self._quota_streak.clear()
        self._drive_ok += 1
        now = self._last_through_at = time.monotonic()
        self._events.append((now, True))
        self._finished_at.append(now)
        while self._finished_at and now - self._finished_at[0] > 3600:
            self._finished_at.popleft()

    def _hold_for_quota(self, job: Job, error: str, cost: float) -> None:
        """Drive refused this file. Hand it back untouched, make it wait its turn, and carry on with the next.

        No attempt is spent: nothing is wrong with the file. Only a run of different files all refused with
        none getting through means the allowance is truly gone, and then the worker rests.
        """
        hits = self._quota_hits[job.id] = self._quota_hits.get(job.id, 0) + 1
        delay = quota_retry_delay(hits)
        self.store.release_job(job.id, error, cost=cost, delay_s=delay)
        self._last_quota_at = time.monotonic()
        self._drive_blocked += 1
        self._events.append((self._last_quota_at, False))
        self._quota_streak.add(job.id)
        log.warning("Drive refused %s (download quota), asking again in %.0f s (refused %d time(s))",
                    self._filename(job), delay, hits)
        self._notify("retrying", job, None)
        if (
            len(self._quota_streak) >= QUOTA_STREAK_PAUSE
            and time.monotonic() - self._last_through_at >= QUOTA_QUIET_S
        ):
            self._quota_streak.clear()
            self.pause_reason = f"Drive refused {QUOTA_STREAK_PAUSE} different files in a row: {error[:200]}"
            self.paused_until = time.monotonic() + QUOTA_STREAK_PAUSE_S
            log.error("indexing paused for %d s: %s", QUOTA_STREAK_PAUSE_S, self.pause_reason)
            self._notify("paused", job, None)

    @staticmethod
    def _filename(job: Job) -> str:
        return str((job.payload or {}).get("filename") or job.id[:8])

    def _log_drive_ratio(self, force: bool = False) -> None:
        """One line every ten minutes saying how much of what was asked of Drive got through."""
        now = time.monotonic()
        if not force and now - self._ratio_logged_at < DRIVE_RATIO_EVERY_S:
            return
        self._ratio_logged_at = now
        total = self._drive_ok + self._drive_blocked
        if not total and not self.quota_pressure():
            return
        while self._finished_at and now - self._finished_at[0] > 3600:
            self._finished_at.popleft()
        through = f"{100 * self._drive_ok / total:.0f}%" if total else "n/a"
        rate30 = self._pass_rate(QUOTA_REST_WINDOW_S, 1)
        mode = "resting" if now < self._resting_until else (
            "off" if not self.quota_pressure() else "slow" if self._slow else "fast")
        log.warning(
            "Drive download quota, last %d min: %d file(s) through, %d refused (%s through); "
            "%d file(s) done in the last hour; last 30 min %s through; pacing %s",
            DRIVE_RATIO_EVERY_S // 60, self._drive_ok, self._drive_blocked, through,
            len(self._finished_at), f"{rate30:.0%}" if rate30 is not None else "n/a", mode,
        )
        self._drive_ok = self._drive_blocked = 0

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
            self._log_drive_ratio()
            self._maybe_rest()
            if time.monotonic() < self._resting_until:
                if drain:
                    break  # a one-off command ends, saying why, rather than rest for 45 minutes
                await self._rest_step()
                continue
            # While Drive has been refusing reads, start files slowly and keep few in flight: the allowance
            # lets requests through at a rate, and hammering it only gets more of them refused. How slowly
            # follows how much of what is asked is getting through.
            limit = self.concurrency
            pace = self._pace()
            if pace is not None:
                limit = pace[0]
                wait = self._next_start_at - time.monotonic()
                if wait > 0:
                    try:
                        await asyncio.wait_for(self._stop.wait(), timeout=min(wait, 1.0))
                    except asyncio.TimeoutError:
                        pass
                    continue
                if len(tasks) >= limit:
                    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    tasks = {t for t in tasks if not t.done()}
                    continue
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
            pace = self._pace()
            if pace is not None:
                self._next_start_at = time.monotonic() + random.uniform(*pace[1])

            if len(tasks) >= limit:
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
                self._note_through(job)
                self.stats.skipped += 1
                self._notify("finished", job, result)
                await self._mirror_attention()
                return
            self.store.finish_job(job.id, "done", cost=result.cost_usd)
            self._note_through(job)
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
        if is_drive_quota_error(error):
            self._hold_for_quota(job, error, cost)
            return
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
