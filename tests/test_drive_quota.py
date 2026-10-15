"""Drive's download quota is a recovering allowance, not a wall.

Measured on the live account on 15 Oct 2026, on the same files within seconds: a 1 KB ranged read passed, a
512 KB one was refused, a full download passed, a 1 MB range was refused; 20 photo downloads six seconds apart
went XXXXXXXXXXOXXOOOXOXO (refused until the allowance began to recover, then about six in ten through). A
worker that stops for five minutes at the first refusal makes no progress while half the requests would pass,
so a refused file is held back on its own, files are started slowly while refusals are about, and a ratio line
shows the allowance coming back.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time

import pytest

import broll.jobs.worker as worker_module
from broll.analysis.providers.base import is_drive_quota_error, is_environment_error
from broll.ingest.pipeline import IngestPipeline, IngestResult
from broll.jobs.queue import KIND_INDEX_SOURCE, queue_stats
from broll.jobs.worker import (
    DRIVE_RATIO_EVERY_S,
    QUOTA_PRESSURE_S,
    QUOTA_RETRY_BASE_S,
    QUOTA_RETRY_MAX_S,
    QUOTA_STREAK_PAUSE,
    Worker,
    quota_retry_delay,
)

QUOTA = ("RuntimeError: Drive's download quota for this account is used up (downloadQuotaExceeded)")
FFPROBE_403 = (
    "ffprobe could not read IMG_1831.MOV: https://www.googleapis.com/drive/v3/files/"
    "1DE7P1TmhAERPy8vp3hT98yPaCyn4jga1?alt=media&supportsAllDrives=true: "
    "Server returned 403 Forbidden (access denied)"
)


def _queue(store, tmp_path, n):
    for i in range(n):
        clip = tmp_path / f"{i:02d}.mp4"
        clip.write_bytes(b"x")
        store.enqueue(KIND_INDEX_SOURCE, {"origin": "local", "path": str(clip), "filename": clip.name,
                                          "origin_path": str(clip)})


def _claim_all(store, n):
    jobs = [store.claim_job() for _ in range(n)]
    assert all(jobs)
    return jobs


@pytest.fixture()
def no_pacing(monkeypatch):
    monkeypatch.setattr(worker_module, "QUOTA_PACE_S", (0.0, 0.0))


def test_the_quota_is_recognised_in_every_shape_it_comes_in():
    for message in (QUOTA, FFPROBE_403,
                    'HttpError: <HttpError 403 when requesting https://www.googleapis.com/drive/v3/files/x?alt=media '
                    'returned "The download quota for this file has been exceeded.">'):
        assert is_drive_quota_error(message) and is_environment_error(message)
    # A dead sign-in is a setup problem for the whole queue, not Drive's allowance.
    assert is_environment_error("invalid_grant: Token has been expired or revoked.")
    assert not is_drive_quota_error("invalid_grant: Token has been expired or revoked.")
    assert not is_drive_quota_error("ffprobe could not read a.mov: moov atom not found")
    assert not is_drive_quota_error("TransientProviderError: gemini request failed: 503 UNAVAILABLE")


def test_a_refused_file_waits_longer_each_time_it_is_refused_up_to_an_hour():
    rng = random.Random(1)
    delays = [quota_retry_delay(n, rng) for n in range(1, 9)]
    for n, delay in enumerate(delays, start=1):
        nominal = min(QUOTA_RETRY_BASE_S * 2 ** (n - 1), QUOTA_RETRY_MAX_S)
        assert 0.7 * nominal <= delay <= 1.3 * nominal
    assert delays[0] < 300 and delays[-1] > 0.7 * 3600, "from about three minutes to about an hour"
    assert len({round(quota_retry_delay(3, rng)) for _ in range(20)}) > 10, "with jitter, so files do not return in step"


async def test_one_refused_file_does_not_stop_the_files_behind_it(workspace, store, tmp_path, no_pacing):
    _queue(store, tmp_path, 6)
    tried: list[str] = []

    class Drive(IngestPipeline):
        async def ingest(self, discovered, force=False, overwrite_corrections=False, **kw):
            tried.append(discovered.filename)
            if discovered.filename in ("01.mp4", "03.mp4"):
                raise RuntimeError(QUOTA)
            return IngestResult(status="indexed", filename=discovered.filename)

    worker = Worker(workspace, store, pipeline=Drive(workspace, store), concurrency=1)
    await worker.run(drain=True)

    assert sorted(tried) == [f"{i:02d}.mp4" for i in range(6)], "every file was tried, in one go"
    stats = queue_stats(store)
    assert stats.done == 4 and stats.queued == 2 and stats.failed == 0
    assert worker.pause_reason is None and worker.paused_until == 0.0, "the worker never stopped"
    assert store.conn.execute("SELECT MAX(attempts) FROM jobs WHERE status = 'queued'").fetchone()[0] == 0, \
        "the refused files spent no attempt"


def test_refusals_carry_over_between_tries_and_a_success_wipes_the_slate(workspace, store, tmp_path):
    _queue(store, tmp_path, 2)
    a, b = _claim_all(store, 2)
    worker = Worker(workspace, store, pipeline=None)

    def wait_of(job_id):
        return store.conn.execute(
            "SELECT (julianday(not_before) - julianday('now')) * 86400 AS s FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()["s"]

    waits = []
    for _ in range(4):
        worker._handle_failure(a, QUOTA)
        waits.append(wait_of(a.id))
    held = store.get_job(a.id)
    assert held.status == "queued" and held.attempts == 0
    assert 100 < waits[0] < 250, "about three minutes the first time"
    assert waits[3] > 2.5 * waits[0], "and longer each time the same file is refused"
    assert worker._quota_hits[a.id] == 4

    worker._note_through(b)
    assert worker._quota_streak == set() and b.id not in worker._quota_hits


def test_a_real_lockout_still_stops_the_worker_but_only_after_many_different_files(workspace, store, tmp_path):
    _queue(store, tmp_path, QUOTA_STREAK_PAUSE + 2)
    jobs = _claim_all(store, QUOTA_STREAK_PAUSE + 2)
    worker = Worker(workspace, store, pipeline=None)

    worker._last_through_at -= 400   # nothing has got through for a while
    for job in jobs[: QUOTA_STREAK_PAUSE - 1]:
        worker._handle_failure(job, QUOTA)
    assert worker.paused_until == 0.0, "eleven refusals in a row are not yet a lockout"
    worker._handle_failure(jobs[QUOTA_STREAK_PAUSE - 1], FFPROBE_403)
    assert worker.paused_until > time.monotonic() + 500, "the twelfth different file is"
    assert "12 different files" in (worker.pause_reason or "")

    # A file that gets through in between starts the count again.
    worker2 = Worker(workspace, store, pipeline=None)
    worker2._last_through_at -= 400
    for job in jobs[: QUOTA_STREAK_PAUSE - 1]:
        worker2._handle_failure(job, QUOTA)
    worker2._note_through(jobs[-1])
    worker2._handle_failure(jobs[QUOTA_STREAK_PAUSE - 1], QUOTA)
    assert worker2.paused_until == 0.0


def test_many_refusals_while_files_are_still_getting_through_are_not_a_lockout(workspace, store, tmp_path):
    """Measured on 15 Oct: about one request in six passed, and a file counts as through only when it finishes,
    so twelve refusals in a row came well within a minute of a success. That must not stop the worker."""
    _queue(store, tmp_path, QUOTA_STREAK_PAUSE + 3)
    jobs = _claim_all(store, QUOTA_STREAK_PAUSE + 3)
    worker = Worker(workspace, store, pipeline=None)
    worker._last_through_at = time.monotonic() - 60     # a file got through a minute ago
    for job in jobs[:QUOTA_STREAK_PAUSE + 2]:
        worker._handle_failure(job, QUOTA)
    assert worker.paused_until == 0.0 and worker.pause_reason is None
    worker._last_through_at = time.monotonic() - 400    # ...and then nothing for over five minutes
    worker._handle_failure(jobs[-1], QUOTA)
    assert worker.paused_until > time.monotonic() + 500


def test_the_same_file_refused_again_and_again_is_not_a_lockout(workspace, store, tmp_path):
    _queue(store, tmp_path, 1)
    (job,) = _claim_all(store, 1)
    worker = Worker(workspace, store, pipeline=None)
    for _ in range(QUOTA_STREAK_PAUSE * 2):
        worker._handle_failure(job, QUOTA)
    assert worker.paused_until == 0.0


def test_a_dead_sign_in_still_pauses_everything_at_once(workspace, store, tmp_path):
    _queue(store, tmp_path, 1)
    (job,) = _claim_all(store, 1)
    worker = Worker(workspace, store, pipeline=None)
    worker._handle_failure(job, "RuntimeError: invalid_grant: Token has been expired or revoked.")
    assert worker.paused_until > time.monotonic() + 200 and worker._last_quota_at is None


def test_pacing_switches_on_with_a_refusal_and_off_after_half_an_hour(workspace, store):
    worker = Worker(workspace, store, pipeline=None)
    assert not worker.quota_pressure()
    worker._last_quota_at = time.monotonic()
    assert worker.quota_pressure()
    worker._last_quota_at = time.monotonic() - QUOTA_PRESSURE_S - 1
    assert not worker.quota_pressure()


async def test_files_are_started_a_few_seconds_apart_while_drive_is_refusing(workspace, store, tmp_path, monkeypatch):
    monkeypatch.setattr(worker_module, "QUOTA_PACE_S", (0.3, 0.3))
    _queue(store, tmp_path, 4)
    starts: list[float] = []
    in_flight = {"now": 0, "most": 0}

    class Slow(IngestPipeline):
        async def ingest(self, discovered, force=False, overwrite_corrections=False, **kw):
            starts.append(time.monotonic())
            in_flight["now"] += 1
            in_flight["most"] = max(in_flight["most"], in_flight["now"])
            await asyncio.sleep(0.5)
            in_flight["now"] -= 1
            return IngestResult(status="indexed", filename=discovered.filename)

    worker = Worker(workspace, store, pipeline=Slow(workspace, store), concurrency=4)
    worker._last_quota_at = time.monotonic()   # Drive has just refused something
    await worker.run(drain=True)

    assert len(starts) == 4
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert all(g >= 0.28 for g in gaps), gaps
    assert in_flight["most"] <= 2, "at most two files in flight while it lasts"


async def test_without_refusals_nothing_is_slowed_down(workspace, store, tmp_path, monkeypatch):
    monkeypatch.setattr(worker_module, "QUOTA_PACE_S", (5.0, 5.0))
    _queue(store, tmp_path, 4)
    in_flight = {"now": 0, "most": 0}

    class Quick(IngestPipeline):
        async def ingest(self, discovered, force=False, overwrite_corrections=False, **kw):
            in_flight["now"] += 1
            in_flight["most"] = max(in_flight["most"], in_flight["now"])
            await asyncio.sleep(0.2)
            in_flight["now"] -= 1
            return IngestResult(status="indexed", filename=discovered.filename)

    worker = Worker(workspace, store, pipeline=Quick(workspace, store), concurrency=4)
    started = time.monotonic()
    await worker.run(drain=True)
    assert time.monotonic() - started < 3, "no pacing when Drive has not refused anything"
    assert in_flight["most"] >= 3


def test_the_ratio_line_says_how_much_got_through(workspace, store, tmp_path, caplog):
    _queue(store, tmp_path, 4)
    jobs = _claim_all(store, 4)
    worker = Worker(workspace, store, pipeline=None)
    worker._note_through(jobs[0])
    worker._note_through(jobs[1])
    worker._note_through(jobs[2])
    worker._handle_failure(jobs[3], QUOTA)

    with caplog.at_level(logging.WARNING, logger="broll.jobs.worker"):
        worker._log_drive_ratio()
        assert not [r for r in caplog.records if "Drive download quota" in r.message], "not before ten minutes are up"
        worker._ratio_logged_at -= DRIVE_RATIO_EVERY_S
        worker._log_drive_ratio()
    [line] = [r.message for r in caplog.records if "Drive download quota" in r.message]
    assert "3 file(s) through, 1 refused (75% through)" in line
    assert "3 file(s) done in the last hour" in line and "pacing on" in line
    assert (worker._drive_ok, worker._drive_blocked) == (0, 0), "the tally starts again for the next ten minutes"


def test_a_quiet_ten_minutes_writes_nothing(workspace, store, caplog):
    worker = Worker(workspace, store, pipeline=None)
    worker._ratio_logged_at -= DRIVE_RATIO_EVERY_S
    with caplog.at_level(logging.WARNING, logger="broll.jobs.worker"):
        worker._log_drive_ratio()
    assert not caplog.records


# -- the one-megabyte question --------------------------------------------------------------------------------


class _BlocksBigReads:
    """Passes a tiny ranged read and refuses a big one, as Drive did on 15 Oct."""

    def __init__(self, limit: int):
        import http.server
        import re
        import threading

        outer = self
        outer.sizes = []
        body = (b'{"error": {"code": 403, "errors": [{"reason": "downloadQuotaExceeded"}], '
                b'"message": "The download quota for this file has been exceeded."}}')

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                a, b = re.match(r"bytes=(\d+)-(\d+)", self.headers.get("Range", "bytes=0-0")).groups()
                size = int(b) - int(a) + 1
                outer.sizes.append(size)
                if size > limit:
                    self.send_response(403)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(206)
                self.send_header("Content-Length", str(size))
                self.send_header("Content-Range", f"bytes {a}-{b}/10000000")
                self.end_headers()
                self.wfile.write(b"x" * size)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/clip.mp4"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def test_a_refusal_that_only_hits_big_reads_is_still_found():
    from broll.ingest.remote import RemoteDriveVideo

    class Token:
        def access_token(self):
            return "tok"

    server = _BlocksBigReads(limit=4096)
    try:
        remote = RemoteDriveVideo(Token(), "id", "a.mov", url=server.url)
        message = remote.quota_error()
        assert message and is_drive_quota_error(message), "a one-byte question would have been answered"
        assert max(server.sizes) >= 512 * 1024, "so the question asks for a real piece"
    finally:
        server.close()
