"""What the model costs, measured from what it bills.

The first full run was budgeted at ~$33 from a formula and cost about five times that: the formula knew
nothing about the model's thinking tokens (billed as output), about image tokens (1,120 a frame, not 300)
or about the 7,000-token prompt sent with every frame. These tests pin the replacement: thinking off by
default, the prompt in an order that can be cached, and every call booked from the usage the response
reports - failed and timed-out calls included - with caps that stop the worker.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from broll.analysis.providers.base import (
    Pricing,
    ProviderError,
    TransientProviderError,
    is_environment_error,
)
from broll.analysis.prompt import build_user_prompt, shot_prompt, static_prompt
from broll.analysis.schema import AnalysisResult, ShotContext
from broll.analysis.usage import Usage, UsageLedger, UsageTally, current_tally, report
from broll.jobs.queue import KIND_INDEX_SOURCE, queue_stats

pytest.importorskip("google.genai")
from broll.analysis.providers import gemini  # noqa: E402


FLASH = gemini.PRICING["gemini-3.6-flash"]


# -- the arithmetic ----------------------------------------------------------------------------------------


def test_thinking_is_billed_as_output_and_cached_tokens_at_a_tenth():
    # The call that was measured on the real model: 10,188 tokens in, 361 out, 1,345 of thinking.
    default = FLASH.usage_cost(10_188, 361, thought_tokens=1_345)
    assert default == pytest.approx(10_188 * 0.75 / 1e6 + (361 + 1_345) * 3.75 / 1e6)
    assert default == pytest.approx(0.0140, abs=0.0001)

    thinking_off = FLASH.usage_cost(10_188, 573)
    assert thinking_off < default * 0.75

    cached = FLASH.usage_cost(10_232, 574, cached_tokens=8_465)
    assert cached == pytest.approx(0.0041, abs=0.0002)
    assert cached < thinking_off * 0.5


def test_a_cached_token_is_a_tenth_of_an_input_token_unless_the_table_says_otherwise():
    plain = Pricing(input_per_m=1.0, output_per_m=1.0, tokens_per_image=1)
    assert plain.usage_cost(1_000_000, 0, cached_tokens=1_000_000) == pytest.approx(0.1)
    explicit = Pricing(input_per_m=1.0, output_per_m=1.0, cached_per_m=0.25, tokens_per_image=1)
    assert explicit.usage_cost(1_000_000, 0, cached_tokens=1_000_000) == pytest.approx(0.25)
    # More cached than sent is a bad report, not a negative bill.
    assert plain.usage_cost(100, 0, cached_tokens=500) >= 0


def test_the_price_table_has_the_cheaper_tiers_and_a_cache_rate():
    for name in ("gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"):
        assert name in gemini.PRICING and gemini.PRICING[name].cached_per_m
    assert gemini.PRICING["gemini-3.5-flash-lite"].output_per_m < FLASH.output_per_m


# -- thinking and picture size -------------------------------------------------------------------------------


def test_thinking_is_turned_down_in_the_words_each_model_family_understands():
    from google.genai import types

    cfg = gemini.thinking_config(types, "gemini-3.6-flash", "minimal")
    assert cfg.thinking_level == types.ThinkingLevel.MINIMAL
    assert gemini.thinking_config(types, "gemini-3.6-flash", "low").thinking_level == types.ThinkingLevel.LOW
    # A 2.5 model takes a budget; Pro cannot go to zero.
    assert gemini.thinking_config(types, "gemini-2.5-flash", "minimal").thinking_budget == 0
    assert gemini.thinking_config(types, "gemini-2.5-pro", "minimal").thinking_budget >= 128
    # A model with no "minimal" is given the lowest it has.
    assert gemini.thinking_config(types, "gemini-3.8-flash", "minimal").thinking_level == types.ThinkingLevel.LOW
    # "default" leaves it to the model (which thinks at "medium").
    assert gemini.thinking_config(types, "gemini-3.6-flash", "default") is None
    with pytest.raises(ValueError):
        gemini.thinking_config(types, "gemini-3.6-flash", "bogus")


def test_picture_size_maps_to_the_sdk_enum_and_default_leaves_it_alone():
    from google.genai import types

    assert gemini.media_resolution_for(types, "low") == types.MediaResolution.MEDIA_RESOLUTION_LOW
    assert gemini.media_resolution_for(types, "default") is None
    assert gemini.media_resolution_for(types, None) is None


def test_the_shipped_defaults_think_as_little_as_possible(workspace):
    assert workspace.provider.thinking == "minimal"
    assert workspace.provider.segment_media_resolution == "low"
    assert workspace.provider.media_resolution == "low"
    assert workspace.provider.photo_media_resolution == "default"
    assert workspace.provider.cache_prompt is True


# -- the prompt, in an order that can be cached ------------------------------------------------------------------


def _context(**kw) -> ShotContext:
    base = dict(
        source_filename="a.jpg", duration_s=0, width=100, height=100, media_kind="image",
        client_context="CLIENT CONTEXT - Adam", emotion_vocab=["calm", "tense"],
        category_options=["Mood/Reflective — sad", "Mood/Struggle"],
    )
    base.update(kw)
    return ShotContext(**base)


def test_the_two_halves_of_the_prompt_make_the_old_prompt():
    ctx = _context()
    assert build_user_prompt(ctx) == static_prompt(ctx) + "\n\n" + shot_prompt(ctx)


def test_the_static_half_does_not_change_from_one_shot_to_the_next():
    a = _context(source_filename="a.jpg", frame_times=[0.0])
    b = _context(source_filename="b.mov", media_kind="video", duration_s=12, frame_times=[1.0, 5.0, 9.0])
    assert static_prompt(a) == static_prompt(b)
    assert shot_prompt(a) != shot_prompt(b)
    assert "a.jpg" not in static_prompt(a)
    assert "a.jpg" in shot_prompt(a)


# -- one call, as the provider makes and books it ------------------------------------------------------------


def _usage_meta(prompt=10_000, cached=0, out=600, thoughts=0, image=1_080):
    return SimpleNamespace(
        prompt_token_count=prompt, cached_content_token_count=cached, candidates_token_count=out,
        thoughts_token_count=thoughts,
        prompt_tokens_details=[SimpleNamespace(modality=SimpleNamespace(value="IMAGE"), token_count=image)],
    )


def _answer(**usage) -> SimpleNamespace:
    parsed = AnalysisResult(
        caption="A man sits on a bed.", setting="bedroom", shot_type="medium", camera_movement="static",
        time_of_day="day", colour_profile="neutral", people_count="one", pace="still",
    )
    return SimpleNamespace(parsed=parsed, text=None, usage_metadata=_usage_meta(**usage))


class _FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.cache_calls: list[dict] = []
        self.aio = SimpleNamespace(
            models=SimpleNamespace(generate_content=self._generate),
            caches=SimpleNamespace(create=self._create_cache),
        )

    async def _generate(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def _create_cache(self, **kwargs):
        self.cache_calls.append(kwargs)
        return SimpleNamespace(name="cachedContents/abc")


@pytest.fixture()
def booked(monkeypatch):
    """A Gemini provider wired to a fake client, with everything it books collected in a list."""
    sink: list[Usage] = []

    def make(replies, **kw):
        client = _FakeClient(replies)
        monkeypatch.setattr(gemini, "_client", lambda key: client)
        gemini._CACHES._items.clear()
        gemini._CACHES._failed_until.clear()
        kw.setdefault("cache_prompt", False)  # on by default in production; the tests that want it say so
        provider = gemini.GeminiVisionProvider(model="gemini-3.6-flash", api_key="k", usage_sink=sink.append, **kw)
        return provider, client

    return make, sink


def _frame(tmp_path, name="f.jpg"):
    path = tmp_path / name
    path.write_bytes(b"\xff\xd8jpeg")
    return path


async def test_a_call_turns_thinking_down_and_puts_the_prompt_before_the_picture(booked, tmp_path):
    make, sink = booked
    provider, client = make([_answer()])
    await provider.analyse([_frame(tmp_path)], _context())

    call = client.calls[0]
    config = call["config"]
    assert config.thinking_config.thinking_level.name == "MINIMAL"
    assert config.system_instruction
    parts = call["contents"][0].parts
    assert "CONTROLLED VOCABULARIES" in parts[0].text, "the part that never changes comes first"
    assert "SHOT METADATA" in parts[1].text
    assert parts[-1].inline_data is not None, "the picture comes last"


async def test_every_call_is_booked_from_the_usage_the_response_reports(booked, tmp_path):
    make, sink = booked
    provider, _ = make([_answer(prompt=10_232, cached=8_465, out=574, thoughts=0, image=1_100)])
    await provider.analyse([_frame(tmp_path)], _context())

    (usage,) = sink
    assert (usage.kind, usage.ok, usage.estimated) == ("vision", True, False)
    assert usage.input_tokens == 10_232 and usage.cached_tokens == 8_465
    assert usage.output_tokens == 574 and usage.thought_tokens == 0 and usage.image_tokens == 1_100
    assert usage.cost_usd == pytest.approx(FLASH.usage_cost(10_232, 574, 8_465))


async def test_an_answer_that_fails_validation_is_still_billed(booked, tmp_path):
    """Google charged for it; only the retry's answer is used. Both must reach the ledger."""
    make, sink = booked
    bad = SimpleNamespace(parsed=None, text='{"caption": 7}', usage_metadata=_usage_meta(out=300))
    provider, _ = make([bad, _answer()])
    with pytest.raises(Exception):
        await provider.analyse([_frame(tmp_path)], _context())
    await provider.analyse([_frame(tmp_path)], _context(), "fix it")

    assert [u.ok for u in sink] == [False, True]
    assert sum(u.cost_usd for u in sink) > FLASH.usage_cost(10_000, 600)


async def test_a_timeout_is_booked_as_an_estimate_but_a_503_is_not(booked, tmp_path):
    make, sink = booked
    provider, _ = make([RuntimeError("The read operation timed out"), RuntimeError("503 UNAVAILABLE")])
    for _ in range(2):
        with pytest.raises(TransientProviderError):
            await provider.analyse([_frame(tmp_path)], _context())

    assert len(sink) == 1, "a refused request is not billed; one nobody answered may well have been"
    assert sink[0].estimated and not sink[0].ok and sink[0].cost_usd > 0


async def test_calls_add_up_in_the_tally_of_the_job_that_made_them(booked, tmp_path):
    make, sink = booked
    provider, _ = make([_answer(), _answer()])
    tally = UsageTally()
    current_tally.set(tally)
    await provider.analyse([_frame(tmp_path)], _context())
    await provider.analyse([_frame(tmp_path)], _context())
    assert tally.calls == 2
    assert tally.cost_usd == pytest.approx(sum(u.cost_usd for u in sink))


async def test_the_segmentation_call_looks_at_small_pictures_and_thinks_little(booked, tmp_path):
    from broll.analysis.segmentation import SegmentationResult, SegmentContext

    make, sink = booked
    reply = SimpleNamespace(
        parsed=SegmentationResult(segments=[]), text=None, usage_metadata=_usage_meta(prompt=4_000, image=3_360),
    )
    provider, client = make([reply])
    await provider.segment(
        [_frame(tmp_path, "a.jpg"), _frame(tmp_path, "b.jpg")], [0.0, 3.0],
        SegmentContext(source_filename="a.mov", file_duration_s=6),
    )
    config = client.calls[0]["config"]
    assert config.thinking_config.thinking_level.name == "MINIMAL"
    assert config.media_resolution.name == "MEDIA_RESOLUTION_LOW"
    assert sink[0].kind == "segment"


async def test_a_photo_keeps_full_detail_and_a_video_shots_frames_go_small(booked, tmp_path):
    """One photograph is all the evidence there is (a tattoo, a pendant); four to eight frames of a shot are not."""
    make, _ = booked
    provider, client = make([_answer(), _answer()], media_resolution="low", photo_media_resolution="default")
    await provider.analyse([_frame(tmp_path)], _context(media_kind="image"))
    assert client.calls[0]["config"].media_resolution is None, "the default is left to the model"

    await provider.analyse([_frame(tmp_path)], _context(media_kind="video", duration_s=8.0))
    assert client.calls[1]["config"].media_resolution.name == "MEDIA_RESOLUTION_LOW"

    provider, client = make([_answer()], media_resolution="default", photo_media_resolution="low")
    await provider.analyse([_frame(tmp_path)], _context(media_kind="image"))
    assert client.calls[0]["config"].media_resolution.name == "MEDIA_RESOLUTION_LOW"


async def test_text_calls_are_booked_too(monkeypatch):
    from pydantic import BaseModel

    class Out(BaseModel):
        ok: bool = True

    sink: list[Usage] = []
    client = _FakeClient([SimpleNamespace(parsed=Out(), text=None, usage_metadata=_usage_meta(prompt=900, out=40, image=0))])
    monkeypatch.setattr(gemini, "_client", lambda key: client)
    provider = gemini.GeminiTextProvider(model="gemini-3.6-flash", api_key="k", usage_sink=sink.append)
    await provider.complete("hello", Out)
    assert sink[0].kind == "text" and client.calls[0]["config"].thinking_config is not None


# -- the explicit cache --------------------------------------------------------------------------------------


async def test_with_the_cache_on_the_prompt_is_sent_once_and_read_after_that(booked, tmp_path):
    make, sink = booked
    provider, client = make([_answer(cached=9_000), _answer(cached=9_000)], cache_prompt=True)
    for _ in range(2):
        await provider.analyse([_frame(tmp_path)], _context())

    assert len(client.cache_calls) == 1, "one cache serves every shot"
    first = client.calls[0]
    assert first["config"].cached_content == "cachedContents/abc"
    assert first["config"].system_instruction is None, "the system prompt lives in the cache"
    text = "".join(p.text or "" for p in first["contents"][0].parts)
    assert "CONTROLLED VOCABULARIES" not in text, "and so does the vocabulary: it is not sent again"
    assert "SHOT METADATA" in text


async def test_a_cache_that_cannot_be_made_is_not_a_reason_to_fail_a_file(booked, tmp_path, monkeypatch):
    make, sink = booked
    provider, client = make([_answer()], cache_prompt=True)

    async def refuse(**kwargs):
        raise RuntimeError("INVALID_ARGUMENT: cached content is too small")

    client.aio.caches.create = refuse
    await provider.analyse([_frame(tmp_path)], _context())
    assert client.calls[0]["config"].cached_content is None
    assert "CONTROLLED VOCABULARIES" in client.calls[0]["contents"][0].parts[0].text


# -- the ledger ----------------------------------------------------------------------------------------------


def test_the_ledger_adds_up_what_was_billed(workspace):
    ledger = UsageLedger.for_config(workspace)
    ledger.record(Usage(model="m", kind="vision", input_tokens=1000, cached_tokens=500, output_tokens=100,
                        cost_usd=0.01))
    ledger.record(Usage(model="m", kind="segment", input_tokens=2000, output_tokens=50, thought_tokens=900,
                        cost_usd=0.02, ok=False))
    assert ledger.spent() == pytest.approx(0.03)
    assert ledger.spent_last(1) == pytest.approx(0.03)
    assert ledger.spent(datetime.now(UTC) + timedelta(hours=1)) == 0.0

    summary = ledger.summary()
    assert summary["calls"] == 2 and summary["failed_calls"] == 1
    assert summary["failed_usd"] == pytest.approx(0.02)
    assert summary["cached_tokens"] == 500 and summary["thought_tokens"] == 900
    assert {r["kind"] for r in summary["by_kind"]} == {"vision", "segment"}


def test_a_library_made_before_the_ledger_gets_the_table_on_migration(tmp_path):
    from broll.db.migrations import SCHEMA_VERSION, migrate
    from broll.db.store import connect

    conn = connect(tmp_path / "old.db")
    migrate(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert conn.execute("SELECT COUNT(*) FROM api_usage").fetchone()[0] == 0


def test_a_ledger_that_cannot_be_written_never_fails_the_call():
    def broken(_usage):
        raise OSError("disk full")

    tally = UsageTally()
    current_tally.set(tally)
    report(Usage(model="m", kind="vision", cost_usd=0.5), broken)
    assert tally.cost_usd == 0.5


# -- the worker: what a job cost, and when to stop ----------------------------------------------------------------


def _queue_files(store, tmp_path, n):
    for i in range(n):
        clip = tmp_path / f"{i}.mp4"
        clip.write_bytes(b"x")
        store.enqueue(KIND_INDEX_SOURCE, {"origin": "local", "path": str(clip), "filename": clip.name,
                                          "origin_path": str(clip)})


async def test_a_job_costs_what_its_calls_billed_even_when_it_then_fails(workspace, store, tmp_path):
    from broll.ingest.pipeline import IngestPipeline
    from broll.jobs.worker import Worker

    _queue_files(store, tmp_path, 1)

    class PaysThenFails(IngestPipeline):
        async def ingest(self, discovered, force=False, overwrite_corrections=False, **kw):
            report(Usage(model="m", kind="vision", cost_usd=0.0123), None)
            raise TransientProviderError("gemini request failed: 503 UNAVAILABLE")

    worker = Worker(workspace, store, pipeline=PaysThenFails(workspace, store), concurrency=1)
    await worker.run(drain=True)

    cost = store.conn.execute("SELECT cost_estimate_usd FROM jobs").fetchone()[0]
    assert cost == pytest.approx(0.0123), "the attempt that failed was paid for, and is counted"


async def test_a_finished_job_carries_the_billed_cost_not_the_estimate(workspace, store, tmp_path):
    from broll.ingest.pipeline import IngestPipeline, IngestResult
    from broll.jobs.worker import Worker

    _queue_files(store, tmp_path, 1)

    class Pays(IngestPipeline):
        async def ingest(self, discovered, force=False, overwrite_corrections=False, **kw):
            report(Usage(model="m", kind="vision", cost_usd=0.004), None)
            report(Usage(model="m", kind="segment", cost_usd=0.002), None)
            return IngestResult(status="indexed", filename=discovered.filename, cost_usd=0.0001)

    worker = Worker(workspace, store, pipeline=Pays(workspace, store), concurrency=1)
    stats = await worker.run(drain=True)
    assert stats.cost_usd == pytest.approx(0.006)
    assert store.conn.execute("SELECT cost_estimate_usd FROM jobs").fetchone()[0] == pytest.approx(0.006)


async def test_the_worker_stops_claiming_files_once_a_run_has_cost_its_cap(workspace, store, tmp_path):
    from broll.ingest.pipeline import IngestPipeline, IngestResult
    from broll.jobs.worker import Worker

    workspace.ingest.max_run_usd = 0.05
    workspace.ingest.max_day_usd = 0
    _queue_files(store, tmp_path, 6)
    ledger = UsageLedger.for_config(workspace)
    done: list[str] = []

    class Spender(IngestPipeline):
        async def ingest(self, discovered, force=False, overwrite_corrections=False, **kw):
            done.append(discovered.filename)
            usage = Usage(model="m", kind="vision", cost_usd=0.03)
            ledger.record(usage)
            report(usage, None)
            return IngestResult(status="indexed", filename=discovered.filename)

    worker = Worker(workspace, store, pipeline=Spender(workspace, store), concurrency=1)
    worker._cap_checked_at = -1e9
    import broll.jobs.worker as worker_module

    worker_module.CAP_CHECK_EVERY_S = 0.0
    try:
        await asyncio.wait_for(worker.run(drain=True), timeout=20)
    finally:
        worker_module.CAP_CHECK_EVERY_S = 5.0

    assert len(done) == 2, "$0.03 + $0.03 passes the $0.05 cap: nothing starts after the second file"
    assert "cap" in (worker.pause_reason or "")
    stats = queue_stats(store)
    assert stats.queued == 4 and stats.failed == 0, "the rest wait; none is failed or lost"


async def test_the_daily_cap_holds_across_a_restart(workspace, store, tmp_path):
    from broll.ingest.pipeline import IngestPipeline
    from broll.jobs.worker import Worker

    workspace.ingest.max_run_usd = 0
    workspace.ingest.max_day_usd = 1.0
    UsageLedger.for_config(workspace).record(Usage(model="m", kind="vision", cost_usd=1.5))
    _queue_files(store, tmp_path, 3)
    tried: list[str] = []

    class Never(IngestPipeline):
        async def ingest(self, discovered, force=False, overwrite_corrections=False, **kw):
            tried.append(discovered.filename)
            raise AssertionError("must not be started")

    worker = Worker(workspace, store, pipeline=Never(workspace, store), concurrency=1)
    assert "daily cap" in (worker.spend_cap_reached() or "")
    await asyncio.wait_for(worker.run(drain=True), timeout=20)
    assert tried == []
    assert queue_stats(store).queued == 3


def test_a_cap_of_zero_means_no_cap(workspace, store):
    from broll.jobs.worker import Worker

    workspace.ingest.max_run_usd = 0
    workspace.ingest.max_day_usd = 0
    UsageLedger.for_config(workspace).record(Usage(model="m", kind="vision", cost_usd=999))
    assert Worker(workspace, store, pipeline=None).spend_cap_reached() is None


async def test_drives_download_quota_holds_the_file_back_instead_of_failing_it(workspace, store, tmp_path):
    """614 files were marked "couldn't be analysed" for this. It costs no model money and clears itself, and
    it lets some requests through while it refuses others: so the file that was refused waits on its own and
    the queue carries on."""
    from broll.ingest.pipeline import IngestPipeline
    from broll.jobs.worker import Worker

    message = (
        'HttpError: <HttpError 403 when requesting https://www.googleapis.com/drive/v3/files/x?alt=media '
        'returned "The download quota for this file has been exceeded.". Details: '
        "\"[{'reason': 'downloadQuotaExceeded'}]\">"
    )
    assert is_environment_error(message)
    assert is_environment_error("Drive's download quota for this account is used up (downloadQuotaExceeded)")

    _queue_files(store, tmp_path, 4)
    tried: list[str] = []

    class Quota(IngestPipeline):
        async def ingest(self, discovered, force=False, overwrite_corrections=False, **kw):
            tried.append(discovered.filename)
            raise RuntimeError(message)

    worker = Worker(workspace, store, pipeline=Quota(workspace, store), concurrency=1)
    worker._next_start_at = 0.0
    import broll.jobs.worker as worker_module

    original = worker_module.QUOTA_PACE_S
    worker_module.QUOTA_PACE_S = (0.0, 0.0)
    try:
        await worker.run(drain=True)
    finally:
        worker_module.QUOTA_PACE_S = original
    assert len(tried) == 4, "each file was tried once: a refusal does not stop the queue"
    stats = queue_stats(store)
    assert stats.failed == 0 and stats.queued == 4
    assert store.conn.execute("SELECT MAX(attempts) FROM jobs").fetchone()[0] == 0, "no attempt is burned"
    assert store.conn.execute("SELECT COUNT(*) FROM jobs WHERE not_before > datetime('now')").fetchone()[0] == 4, \
        "every refused file waits before it is asked for again"
    assert store.list_attention("open") == []
    assert worker.pause_reason is None, "and the worker did not stop"


# -- what the CLI shows ----------------------------------------------------------------------------------------


def test_status_reports_the_real_spend_and_the_old_estimate_apart(broll_home):
    from typer.testing import CliRunner

    from broll.cli import app
    from broll.config import load_workspace_config
    from broll.db.store import Store

    runner = CliRunner()
    assert runner.invoke(app, ["init", "--name", "Fixtures", "--provider", "mock"]).exit_code == 0
    workspace = load_workspace_config("fixtures")
    store = Store.for_config(workspace)
    try:
        job = store.enqueue(KIND_INDEX_SOURCE, {"filename": "old.mp4"})
        store.finish_job(job.id, "done", cost=13.46)  # what the formula said, before the ledger existed
        store.conn.execute("UPDATE jobs SET created_at = '2026-10-10 12:00:00'")
    finally:
        store.close()
    ledger = UsageLedger.for_config(workspace)
    ledger.record(Usage(model="m", kind="vision", input_tokens=100, cost_usd=0.5))
    ledger.record(Usage(model="m", kind="vision", input_tokens=100, cost_usd=0.25, ok=False))

    result = runner.invoke(app, ["status", "--workspace", workspace.id])
    assert result.exit_code == 0, result.output
    assert "cost:      $0.7500 spent" in result.output, "the first figure is the real one: the guard reads it"
    assert "2 call(s)" in result.output and "1 failed" in result.output
    assert "$13.4600 by the old formula" in result.output

    costs = runner.invoke(app, ["costs", "--workspace", workspace.id])
    assert costs.exit_code == 0, costs.output
    assert "$0.7500 over 2 model call(s)" in costs.output


async def test_a_file_that_keeps_costing_money_is_given_up_on(workspace, store, tmp_path):
    """A hopeless file must not be paid for again and again: after two attempts that together cost
    more than the per-file cap, the third is not made."""
    from broll.ingest.pipeline import IngestPipeline
    from broll.jobs.worker import Worker

    workspace.ingest.max_job_usd = 0.10
    _queue_files(store, tmp_path, 1)

    class PaysThenBreaks(IngestPipeline):
        async def ingest(self, discovered, force=False, overwrite_corrections=False, **kw):
            report(Usage(model="m", kind="vision", cost_usd=0.06), None)
            raise RuntimeError("could not make sense of this file")

    worker = Worker(workspace, store, pipeline=PaysThenBreaks(workspace, store), concurrency=1)
    await worker.run(drain=True)
    assert queue_stats(store).queued == 1, "the first failure is retried: $0.06 is under the cap"
    store.conn.execute("UPDATE jobs SET not_before = NULL")
    await worker.run(drain=True)

    stats = queue_stats(store)
    assert stats.failed == 1 and stats.queued == 0
    row = store.conn.execute("SELECT attempts, last_error, cost_estimate_usd FROM jobs").fetchone()
    assert row["attempts"] == 2 and "given up" in row["last_error"]
    assert row["cost_estimate_usd"] == pytest.approx(0.12)


async def test_a_cheap_file_still_gets_all_its_attempts(workspace, store, tmp_path):
    from broll.ingest.pipeline import IngestPipeline
    from broll.jobs.queue import MAX_ATTEMPTS
    from broll.jobs.worker import Worker

    _queue_files(store, tmp_path, 1)

    class Breaks(IngestPipeline):
        async def ingest(self, discovered, force=False, overwrite_corrections=False, **kw):
            report(Usage(model="m", kind="vision", cost_usd=0.001), None)
            raise RuntimeError("could not make sense of this file")

    worker = Worker(workspace, store, pipeline=Breaks(workspace, store), concurrency=1)
    for _ in range(MAX_ATTEMPTS):
        store.conn.execute("UPDATE jobs SET not_before = NULL")
        await worker.run(drain=True)
    assert store.conn.execute("SELECT attempts FROM jobs").fetchone()[0] == MAX_ATTEMPTS
    assert queue_stats(store).failed == 1


def test_the_provider_is_built_from_the_workspace_settings_and_reports_to_its_ledger(workspace):
    from broll.analysis.providers.registry import get_text_provider, get_vision_provider

    workspace.provider.vision = "gemini"
    workspace.provider.thinking = "low"
    workspace.provider.media_resolution = "medium"
    workspace.provider.segment_media_resolution = "low"
    workspace.provider.photo_media_resolution = "high"
    workspace.provider.cache_prompt = True

    vision = get_vision_provider(workspace)
    assert (vision.thinking, vision.media_resolution, vision.segment_media_resolution) == ("low", "medium", "low")
    assert vision.photo_media_resolution == "high"
    assert vision.cache_prompt is True

    vision.usage_sink(Usage(model="m", kind="vision", cost_usd=0.01))
    assert UsageLedger.for_config(workspace).spent() == pytest.approx(0.01)

    text = get_text_provider(workspace)
    assert text.thinking == "low", "the text calls (refile, the script matcher) think as little as the vision ones"
    text.usage_sink(Usage(model="m", kind="text", cost_usd=0.02))
    assert UsageLedger.for_config(workspace).spent() == pytest.approx(0.03)


def test_costs_shows_the_cached_share_and_the_calls_that_gave_no_answer(broll_home):
    import json as jsonlib

    from typer.testing import CliRunner

    from broll.cli import app
    from broll.config import load_workspace_config

    runner = CliRunner()
    assert runner.invoke(app, ["init", "--name", "Fixtures", "--provider", "mock"]).exit_code == 0
    workspace = load_workspace_config("fixtures")
    ledger = UsageLedger.for_config(workspace)
    ledger.record(Usage(model="m", kind="vision", input_tokens=10_000, cached_tokens=9_000, output_tokens=600,
                        cost_usd=0.004))
    ledger.record(Usage(model="m", kind="segment", input_tokens=3_000, output_tokens=100, thought_tokens=500,
                        cost_usd=0.003, ok=False))

    out = runner.invoke(app, ["costs", "--workspace", workspace.id, "--json"])
    assert out.exit_code == 0, out.output
    data = jsonlib.loads(out.output)
    assert data["model_calls"] == 2 and data["failed_calls"] == 1
    assert data["cached_share"] == pytest.approx(9_000 / 13_000, abs=0.001)
    assert data["thought_tokens"] == 500
    assert data["total_usd"] == pytest.approx(0.007)
    assert {row["kind"] for row in data["by_kind"]} == {"vision", "segment"}


async def test_a_refused_drive_read_holds_that_file_back_and_the_rest_carry_on(workspace, store, tmp_path):
    """The 403 the first retry of the library run hit on streamed videos. It must hand the file back without
    spending an attempt, make it wait, and let the files behind it through (Drive refuses some reads and
    passes others, on the same file seconds apart)."""
    from broll.ingest.pipeline import IngestPipeline, IngestResult
    from broll.jobs.worker import Worker
    from tests.test_intake import REAL_FFPROBE_403

    _queue_files(store, tmp_path, 3)
    tried: list[str] = []

    class Drive(IngestPipeline):
        async def ingest(self, discovered, force=False, overwrite_corrections=False, **kw):
            tried.append(discovered.filename)
            if len(tried) == 1:
                return IngestResult(status="failed", filename=discovered.filename, error=REAL_FFPROBE_403)
            return IngestResult(status="indexed", filename=discovered.filename)

    import broll.jobs.worker as worker_module

    original = worker_module.QUOTA_PACE_S
    worker_module.QUOTA_PACE_S = (0.0, 0.0)
    worker = Worker(workspace, store, pipeline=Drive(workspace, store), concurrency=1)
    try:
        await worker.run(drain=True)
    finally:
        worker_module.QUOTA_PACE_S = original

    assert len(tried) == 3 and len(set(tried)) == 3, "the other two files went through after the refusal"
    stats = queue_stats(store)
    assert stats.failed == 0 and stats.queued == 1 and stats.done == 2
    row = store.conn.execute(
        "SELECT attempts, not_before > datetime('now') AS waiting, json_extract(payload_json, '$.filename') AS f "
        "FROM jobs WHERE status = 'queued'"
    ).fetchone()
    assert row["attempts"] == 0, "no attempt was spent"
    assert row["waiting"] and row["f"] == tried[0], "the refused file waits its turn"
    assert store.list_attention("open") == [], "and is not listed as one that could not be read"


