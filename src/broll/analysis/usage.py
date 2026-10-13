"""What the model actually billed, call by call.

The first full run was budgeted from a formula (`Pricing.cost`: so many tokens per image, so many for
the prompt, 350 out) and cost about five times what that formula said, because the formula did not know
the model *thinks* before it answers and bills those thoughts as output. Every Gemini response says
what it used (`usage_metadata`); this module turns that into dollars and keeps it:

* a provider reports each call to a *sink* (`UsageLedger.record`), which writes one row to the
  workspace database. That table, not the formula, is what `broll status`, the spend guard and the
  per-run cap read;
* a per-job *tally* (a context variable, so concurrent jobs do not mix) adds the same calls up, so a
  job's cost is what its calls billed, including the calls of an attempt that then failed.
"""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)


@dataclass
class Usage:
    """One model call, as billed."""

    model: str
    kind: str  # vision | segment | text | embed
    input_tokens: int = 0  # everything sent, cached tokens included
    cached_tokens: int = 0  # the part of input_tokens that was read from a cache
    output_tokens: int = 0  # the answer
    thought_tokens: int = 0  # thinking, billed as output
    image_tokens: int = 0  # the part of input_tokens that was pictures
    cost_usd: float = 0.0
    #: False when the call produced no usable answer (it still cost money if Google answered).
    ok: bool = True
    #: True when no usage came back (a timeout) and the figure is a guess from what was sent.
    estimated: bool = False
    note: str = ""

    @property
    def billed_output_tokens(self) -> int:
        return self.output_tokens + self.thought_tokens


UsageSink = Callable[[Usage], None]


@dataclass
class UsageTally:
    """The calls of one job, added up."""

    calls: int = 0
    failed_calls: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0

    def add(self, usage: Usage) -> None:
        self.calls += 1
        if not usage.ok:
            self.failed_calls += 1
        self.cost_usd += usage.cost_usd
        self.input_tokens += usage.input_tokens
        self.cached_tokens += usage.cached_tokens
        self.output_tokens += usage.output_tokens
        self.thought_tokens += usage.thought_tokens


current_tally: contextvars.ContextVar[UsageTally | None] = contextvars.ContextVar(
    "broll_usage_tally", default=None
)


def report(usage: Usage, sink: UsageSink | None) -> None:
    """Hand one call to the job's tally and to the ledger. Never raises: bookkeeping must not fail a file."""
    tally = current_tally.get()
    if tally is not None:
        tally.add(usage)
    if sink is not None:
        try:
            sink(usage)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not record model usage: %s", exc)


# --------------------------------------------------------------------------
# The ledger
# --------------------------------------------------------------------------

LEDGER_DDL = """
CREATE TABLE IF NOT EXISTS api_usage (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id   TEXT NOT NULL,
    at             TEXT NOT NULL,
    kind           TEXT NOT NULL,
    model          TEXT NOT NULL,
    input_tokens   INTEGER NOT NULL DEFAULT 0,
    cached_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens  INTEGER NOT NULL DEFAULT 0,
    thought_tokens INTEGER NOT NULL DEFAULT 0,
    image_tokens   INTEGER NOT NULL DEFAULT 0,
    cost_usd       REAL NOT NULL DEFAULT 0,
    ok             INTEGER NOT NULL DEFAULT 1,
    estimated      INTEGER NOT NULL DEFAULT 0,
    note           TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_api_usage_at ON api_usage (workspace_id, at);
"""


def _stamp(when: datetime | None = None) -> str:
    return (when or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="seconds")


class UsageLedger:
    """The `api_usage` table of one workspace.

    Every write opens its own short connection: a call is reported from the event loop, from a worker
    thread or from the CLI, and a few thousand one-row inserts are nothing next to the calls themselves.
    """

    def __init__(self, workspace_id: str, path: Path):
        self.workspace_id = workspace_id
        self.path = path

    @classmethod
    def for_config(cls, config) -> "UsageLedger":
        return cls(config.id, config.db_path)

    def _connect(self):
        from ..db.store import connect

        conn = connect(self.path)
        conn.executescript(LEDGER_DDL)  # a database made before the ledger existed still works
        return conn

    def record(self, usage: Usage) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO api_usage (workspace_id, at, kind, model, input_tokens, cached_tokens,
                       output_tokens, thought_tokens, image_tokens, cost_usd, ok, estimated, note)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.workspace_id, _stamp(), usage.kind, usage.model, usage.input_tokens,
                    usage.cached_tokens, usage.output_tokens, usage.thought_tokens, usage.image_tokens,
                    usage.cost_usd, int(usage.ok), int(usage.estimated), usage.note[:200],
                ),
            )
        finally:
            conn.close()

    def spent(self, since: datetime | str | None = None) -> float:
        """Dollars billed since a moment (all of it when `since` is None)."""
        conn = self._connect()
        try:
            if since is None:
                row = conn.execute(
                    "SELECT COALESCE(SUM(cost_usd), 0) FROM api_usage WHERE workspace_id = ?",
                    (self.workspace_id,),
                ).fetchone()
            else:
                stamp = since if isinstance(since, str) else _stamp(since)
                row = conn.execute(
                    "SELECT COALESCE(SUM(cost_usd), 0) FROM api_usage WHERE workspace_id = ? AND at >= ?",
                    (self.workspace_id, stamp),
                ).fetchone()
            return float(row[0])
        finally:
            conn.close()

    def spent_last(self, hours: float) -> float:
        return self.spent(datetime.now(UTC) - timedelta(hours=hours))

    def summary(self) -> dict:
        """Totals for `broll status` / `broll costs`: calls, tokens, dollars, the cached share."""
        conn = self._connect()
        try:
            row = conn.execute(
                """SELECT COUNT(*) AS calls, COALESCE(SUM(cost_usd), 0) AS usd,
                          COALESCE(SUM(input_tokens), 0) AS input_tokens,
                          COALESCE(SUM(cached_tokens), 0) AS cached_tokens,
                          COALESCE(SUM(output_tokens), 0) AS output_tokens,
                          COALESCE(SUM(thought_tokens), 0) AS thought_tokens,
                          COALESCE(SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END), 0) AS failed_calls,
                          COALESCE(SUM(CASE WHEN ok = 0 THEN cost_usd ELSE 0 END), 0) AS failed_usd,
                          MIN(at) AS since
                   FROM api_usage WHERE workspace_id = ?""",
                (self.workspace_id,),
            ).fetchone()
            by_kind = conn.execute(
                """SELECT kind, COUNT(*) AS calls, COALESCE(SUM(cost_usd), 0) AS usd
                   FROM api_usage WHERE workspace_id = ? GROUP BY kind ORDER BY usd DESC""",
                (self.workspace_id,),
            ).fetchall()
            out = dict(row)
            out["by_kind"] = [dict(r) for r in by_kind]
            return out
        finally:
            conn.close()


# --------------------------------------------------------------------------
# What the screens and the CLI show
# --------------------------------------------------------------------------


def spend_report(store, config) -> dict:
    """What the model has billed (the ledger), against the caps, for `broll status` and the spend guard."""
    ledger = UsageLedger.for_config(config)
    summary = ledger.summary()
    day = ledger.spent_last(24)
    since = summary["since"]
    done = 0
    if since:
        done = store.conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE workspace_id = ? AND status = 'done'"
            " AND cost_estimate_usd > 0 AND datetime(finished_at) >= datetime(?)",
            (config.id, since),
        ).fetchone()[0]
    ledger_job_cost = store.conn.execute(
        "SELECT COALESCE(SUM(cost_estimate_usd), 0) FROM jobs WHERE workspace_id = ?"
        " AND (? IS NULL OR datetime(created_at) < datetime(?))",
        (config.id, since, since),
    ).fetchone()[0]
    ingest = config.ingest
    blocked = ""
    if ingest.max_day_usd and day >= ingest.max_day_usd:
        blocked = (f"the last 24 hours cost ${day:.2f}, at the daily cap of ${ingest.max_day_usd:.2f} "
                   "(ingest.max_day_usd).")
    return {
        "usd": float(summary["usd"]),
        "calls": summary["calls"],
        "failed_calls": summary["failed_calls"],
        "failed_usd": float(summary["failed_usd"]),
        "since": since,
        "day_usd": day,
        "per_file_usd": float(summary["usd"]) / done if done else 0.0,
        "legacy_usd": float(ledger_job_cost),
        "caps": {"max_run_usd": ingest.max_run_usd, "max_day_usd": ingest.max_day_usd, "blocked": blocked},
    }
