"""Re-decide which folder each clip belongs in, from what the library already knows about it.

Adding or splitting a folder leaves existing clips where they were: a clip keeps the folder the vision
model gave it at analysis time. Re-running the vision model on every file fixes that but costs the full
analysis again and, for Drive footage, a re-download. This module asks a *text* model instead, in batches,
with the stored description of each clip (caption, observations, action, tags and so on) and the folder
list as it is now. No video is read, and nothing is written to Drive.

Stability rules, because a model can flip a borderline clip between two folders from one run to the next:
a clip moves only when the new folder differs from the current one, and when the model is unsure of its
first choice but names the folder the clip is already in as the runner-up, the clip stays where it is.
A clip a person corrected by hand is never moved unless the caller asks.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .analysis.analyzer import match_category
from .analysis.prompt import render_client_context
from .analysis.providers.base import ProviderError, TextProvider, TransientProviderError
from .config import WorkspaceConfig
from .db.models import ROUTING_REASONS, Shot
from .db.store import Store

log = logging.getLogger(__name__)

BATCH_SIZE = 20
CONCURRENCY = 4
TRANSIENT_ATTEMPTS = 5
TRANSIENT_WAIT_S = 4.0

#: The two review reasons that are about the folder alone. A refile replaces exactly these.
FOLDER_REASONS = frozenset({"low_category_confidence", "category_unmatched"})
assert FOLDER_REASONS <= ROUTING_REASONS

# Same wording rules the vision prompt gives the model (analysis/prompt.py CATEGORY_HEADER).
RULES = (
    "Set `category` to the single best-fit folder, copying the path exactly as written before the "
    '" — ". Prefer the most specific folder that fits and follow each folder\'s note. '
    "File a clip by what it is *for*, not by what is in the frame: if the point of the shot is how the "
    "person feels or the state they are in (low, stressed, lost in thought, calm, proud) and they are not "
    "busy doing something else for most of it, use the folder for that feeling. File by an activity only "
    "when doing that activity is what the shot shows for most of its length. An object in frame never "
    "decides the folder. Use secondary_categories for up to 2 other folders the clip also clearly belongs "
    "in, and leave it empty otherwise. Set category_confidence to how sure you are of `category` alone: "
    "below 0.6 when two folders fit about equally well or none fits well."
)


class RefileItem(BaseModel):
    clip: int
    category: str | None = None
    secondary_categories: list[str] = Field(default_factory=list)
    category_confidence: float = Field(ge=0.0, le=1.0)


class RefileBatch(BaseModel):
    items: list[RefileItem]


@dataclass
class Move:
    shot_id: str
    source_id: str
    before: str | None
    after: str | None
    secondary: list[str]
    confidence: float | None
    reasons: list[str]
    unmatched: bool = False


@dataclass
class RefileReport:
    considered: int = 0
    skipped_corrected: int = 0
    unchanged: int = 0
    kept_on_tie: int = 0
    failed: int = 0
    moves: list[Move] = field(default_factory=list)
    cost_usd: float = 0.0
    estimated_cost_usd: float = 0.0
    calls: int = 0
    applied: bool = False

    def matrix(self) -> list[tuple[str, str, int]]:
        """(from folder, to folder, how many clips), biggest first."""
        counts = Counter((m.before or "(no folder)", m.after or "(no folder)") for m in self.moves)
        return sorted(((a, b, n) for (a, b), n in counts.items()), key=lambda r: (-r[2], r[0], r[1]))


def is_corrected(shot: Shot) -> bool:
    return bool((shot.raw_analysis or {}).get("corrected_by_operator"))


def _under(path: str | None, prefix: str) -> bool:
    return bool(path) and (path == prefix or path.startswith(prefix + "/"))


def select_shots(
    store: Store,
    folders: list[str] | None = None,
    statuses: list[str] | None = None,
    source_ids: list[str] | None = None,
) -> list[Shot]:
    """The clips a refile looks at. Failed and pending clips have nothing to go on, so they are left out."""
    statuses = statuses or ["indexed", "needs_review"]
    marks = ",".join("?" for _ in statuses)
    rows = store.conn.execute(
        f"SELECT id FROM shots WHERE workspace_id = ? AND status IN ({marks}) ORDER BY source_id, shot_index",
        (store.workspace_id, *statuses),
    ).fetchall()
    chosen: list[Shot] = []
    wanted_sources = set(source_ids or [])
    for row in rows:
        shot = store.get_shot(row["id"])
        if shot is None or not shot.caption:
            continue
        if wanted_sources and shot.source_id not in wanted_sources:
            continue
        if folders and not any(_under(shot.category, f) for f in folders):
            continue
        chosen.append(shot)
    return chosen


def _clip_block(index: int, shot: Shot, media_kind: str) -> str:
    lines = [f"CLIP {index} ({'photograph' if media_kind == 'image' else 'video clip'})"]

    def add(label: str, value: Any) -> None:
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value if v)
        if value:
            lines.append(f"  {label}: {value}")

    add("caption", shot.caption)
    add("action", shot.action if shot.action != "none" else "")
    add("observations", shot.observations)
    add("body language", shot.body_language)
    add("setting", ", ".join(x for x in (shot.setting, shot.setting_detail) if x))
    add("subjects", shot.subjects)
    add("tags", shot.tags)
    add("themes", shot.themes)
    add("emotions", shot.emotions)
    add("search phrases", shot.search_phrases)
    return "\n".join(lines)


def build_prompt(
    config: WorkspaceConfig,
    options: list[str],
    batch: list[tuple[int, Shot, str]],
    retry_error: str | None = None,
) -> str:
    parts = [
        "You are a video editor's assistant. Each clip below was already described by looking at its "
        "footage. Decide which of the client's folders each one is filed in, from that description alone.",
    ]
    context = render_client_context(config.client)
    if context:
        parts += ["", context]
    parts += ["", "FOLDERS - this client files footage into their own folders.", RULES, ""]
    parts += [f"- {line}" for line in options]
    parts += ["", "CLIPS", ""]
    parts += [_clip_block(i, shot, kind) for i, shot, kind in batch]
    parts += [
        "",
        "Return one item per clip, with `clip` set to the clip's number, in any order. Use only folders "
        "from the list above, written exactly. Give every clip an item.",
    ]
    if retry_error:
        parts += ["", "YOUR PREVIOUS ANSWER FAILED VALIDATION. Fix exactly this and return all items again:",
                  retry_error]
    return "\n".join(parts)


def _decide(
    config: WorkspaceConfig, leaves: list[str], shot: Shot, item: RefileItem,
) -> tuple[Move | None, str]:
    """Turn the model's answer for one clip into a move, or the reason to leave the clip alone.

    Returns (move or None, outcome) where outcome is 'move', 'unchanged' or 'tie'.
    """
    threshold = config.ingest.review_below_category_confidence
    category = match_category(item.category, leaves)
    secondary: list[str] = []
    for candidate in item.secondary_categories:
        matched = match_category(candidate, leaves)
        if matched and matched != category and matched not in secondary:
            secondary.append(matched)
    secondary = secondary[:2]
    current_valid = shot.category in leaves

    if category is None:
        if current_valid:
            return None, "unchanged"  # the model gave nothing usable: keep what is there
        move = Move(shot.id, shot.source_id, shot.category, None, [], None,
                    ["category_unmatched"], unmatched=True)
        return (move, "move") if shot.category is not None else (None, "unchanged")

    if category == shot.category:
        return None, "unchanged"
    if current_valid and (item.category_confidence < threshold) and shot.category in secondary:
        return None, "tie"

    reasons: list[str] = []
    if item.category_confidence < threshold:
        reasons.append("low_category_confidence")
    return Move(shot.id, shot.source_id, shot.category, category,
                secondary,
                item.category_confidence, reasons), "move"


async def _ask(
    provider: TextProvider, config: WorkspaceConfig, options: list[str],
    batch: list[tuple[int, Shot, str]], report: RefileReport,
) -> dict[int, RefileItem] | None:
    """One batch, one retry on a bad answer. None when the batch could not be answered."""
    wanted = {i for i, _, _ in batch}
    retry_error: str | None = None
    for attempt in (1, 2):
        prompt = build_prompt(config, options, batch, retry_error)
        reply: RefileBatch | None = None
        for wait in range(1, TRANSIENT_ATTEMPTS + 1):
            try:
                report.cost_usd += provider.estimate_cost(prompt)
                report.calls += 1
                reply = await provider.complete(prompt, RefileBatch)  # type: ignore[assignment]
                break
            except TransientProviderError as exc:
                if wait == TRANSIENT_ATTEMPTS:
                    log.warning("refile batch still failing: %s", exc)
                    return None
                await asyncio.sleep(TRANSIENT_WAIT_S * wait)
            except ValidationError as exc:
                retry_error = "Schema validation errors:\n" + "\n".join(
                    f"- {'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:6])
                break
            except ProviderError as exc:
                retry_error = str(exc)
                break
        if reply is not None:
            got = {item.clip: item for item in reply.items if item.clip in wanted}
            missing = sorted(wanted - set(got))
            if not missing:
                return got
            retry_error = f"These clips have no item: {', '.join(str(m) for m in missing)}."
            if attempt == 2:
                return got or None  # keep what did come back; the rest stay where they are
    return None


async def _refile(
    config: WorkspaceConfig, store: Store, provider: TextProvider, shots: list[Shot],
    include_corrected: bool,
) -> RefileReport:
    report = RefileReport()
    leaves_notes = config.taxonomy.category_leaves() if config.taxonomy.mode == "tree" else []
    if not leaves_notes:
        raise ValueError("This workspace has no folder tree to file clips into.")
    leaves = [p for p, _ in leaves_notes]
    options = [f"{p} — {n}" if n else p for p, n in leaves_notes]

    media: dict[str, str] = {}
    todo: list[Shot] = []
    for shot in shots:
        if is_corrected(shot) and not include_corrected:
            report.skipped_corrected += 1
            continue
        if shot.source_id not in media:
            source = store.get_source(shot.source_id)
            media[shot.source_id] = source.media_kind if source else "video"
        todo.append(shot)
    report.considered = len(todo)

    batches = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    report.estimated_cost_usd = sum(
        provider.estimate_cost(build_prompt(
            config, options, [(i, s, media[s.source_id]) for i, s in enumerate(b, 1)]))
        for b in batches)

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def run(batch_shots: list[Shot]) -> None:
        numbered = [(i, s, media[s.source_id]) for i, s in enumerate(batch_shots, 1)]
        async with semaphore:
            answers = await _ask(provider, config, options, numbered, report)
        for i, shot, _ in numbered:
            item = (answers or {}).get(i)
            if item is None:
                report.failed += 1
                continue
            move, outcome = _decide(config, leaves, shot, item)
            if move is not None:
                report.moves.append(move)
            elif outcome == "tie":
                report.kept_on_tie += 1
            else:
                report.unchanged += 1

    await asyncio.gather(*(run(b) for b in batches))
    report.moves.sort(key=lambda m: (m.before or "", m.after or "", m.shot_id))
    return report


def plan(
    config: WorkspaceConfig, store: Store, provider: TextProvider, shots: list[Shot],
    include_corrected: bool = False,
) -> RefileReport:
    """Ask the model where each clip belongs. Writes nothing."""
    return asyncio.run(_refile(config, store, provider, shots, include_corrected))


def apply(store: Store, report: RefileReport, embedder: Any | None = None) -> int:
    """Write the moves, rebuild search text for those clips and re-embed only them."""
    changed = 0
    touched_sources: set[str] = set()
    for move in report.moves:
        shot = store.get_shot(move.shot_id)
        if shot is None:
            continue
        kept = [r for r in shot.review_reasons if r not in FOLDER_REASONS]
        reasons = list(dict.fromkeys([*kept, *move.reasons]))
        raw = dict(shot.raw_analysis or {})
        if raw:
            raw.update(category=move.after, secondary_categories=move.secondary,
                       category_confidence=move.confidence)
        store.set_shot_fields(
            shot.id,
            category=move.after,
            secondary_categories_json=json.dumps(move.secondary),
            category_confidence=move.confidence,
            review_reasons_json=json.dumps(reasons),
            status="needs_review" if reasons else "indexed",
            **({"raw_analysis_json": json.dumps(raw)} if raw else {}),
        )
        text = store.recompute_search_text(shot.id)
        if embedder is not None and text:
            try:
                store.vectors.upsert(shot.id, embedder.embed_documents([text])[0])
            except Exception as exc:  # noqa: BLE001 - the new folder matters more than the vector
                log.warning("re-embedding %s after a refile failed: %s", shot.id, exc)
        touched_sources.add(shot.source_id)
        changed += 1
    for source_id in touched_sources:
        store.recompute_source_status(source_id)
    report.applied = True
    return changed
