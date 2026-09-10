"""Orchestrates one shot: frames -> provider -> validated result.

On validation failure the provider is asked again once, with the error appended
to the prompt. If it fails a second time the shot is marked ``needs_review`` and
the rest of the source carries on - one bad shot must never fail a whole file.
A *transient* provider error is different: it propagates, so the job queue
retries the source with backoff instead of parking a shot for a human.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from ..config import WorkspaceConfig
from ..ingest.frames import extract_frames
from .prompt import PROMPT_VERSION, render_client_context
from .providers.base import ProviderError, TransientProviderError, VisionProvider
from .providers.registry import get_vision_provider
from .schema import DEFECT_FLAGS, EMOTIONS, AnalysisResult, ShotContext, find_oov

log = logging.getLogger(__name__)


@dataclass
class AnalysisOutcome:
    context: ShotContext
    result: AnalysisResult | None = None
    status: str = "indexed"  # indexed | needs_review
    error: str | None = None
    oov: list[tuple[str, str]] = field(default_factory=list)
    cost_usd: float = 0.0
    frames: list[Path] = field(default_factory=list)
    analysis_version: str = PROMPT_VERSION

    @property
    def ok(self) -> bool:
        return self.result is not None


def _squash(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def match_category(value: str | None, leaves: list[str]) -> str | None:
    """Map what the model wrote onto a real folder, tolerating small slips."""
    if not value:
        return None
    if value in leaves:
        return value
    target = _squash(value)
    for leaf in leaves:
        if _squash(leaf) == target:
            return leaf
    # Just the last segment ("Meditation & Stillness"), when it is unambiguous.
    tail = _squash(value.split("/")[-1])
    hits = [leaf for leaf in leaves if _squash(leaf.split("/")[-1]) == tail]
    return hits[0] if len(hits) == 1 else None


def _clean_folder_name(name: str) -> str:
    """Title Case, "&" for "and", no slashes - matching how the tree is written."""
    words = re.sub(r"\s+", " ", name.replace("/", " ")).strip().split(" ")
    words = ["&" if w.lower() == "and" else w[:1].upper() + w[1:] for w in words if w]
    return " ".join(words)[:40].strip()


def _near(a: str, b: str) -> bool:
    """Same folder in all but spelling: identical, plural, or one inside the other."""
    a, b = _squash(a), _squash(b)
    if not a or not b:
        return False
    return a == b or a.rstrip("s") == b.rstrip("s") or (len(a) > 3 and len(b) > 3 and (a in b or b in a))


class Analyzer:
    def __init__(self, config: WorkspaceConfig, provider: VisionProvider | None = None):
        self.config = config
        self.provider = provider or get_vision_provider(config)

        self.new_folders: list[str] = []
        self._load_tree()
        self.emotion_vocab = list(config.client.emotions) or list(EMOTIONS)
        self.client_context = render_client_context(config.client)

        overrides = {k: list(v) for k, v in config.vocabulary_overrides.items()}
        overrides.setdefault("emotions", []).extend(config.client.emotions)
        self.vocab_overrides = overrides

    def _load_tree(self) -> None:
        tree = self.config.taxonomy.category_leaves() if self.config.taxonomy.mode == "tree" else []
        self.leaves = [path for path, _ in tree]
        self.category_options = [f"{path} — {note}" if note else path for path, note in tree]

    def _create_folder(self, proposal: str, note: str | None) -> str | None:
        """Add the folder the model proposed, if it is sound; return its path.

        Guard rails, because an unchecked model invents "Jet Ski", "Jetski" and
        "Jet Skiing" on three consecutive clips: the folder must sit under an
        existing one, and a near-match to an existing sibling reuses it.
        """
        taxonomy = self.config.taxonomy
        parent_raw, _, name_raw = proposal.strip().strip("/").rpartition("/")
        name = _clean_folder_name(name_raw)
        parent = match_category(parent_raw, taxonomy.parent_folders()) if parent_raw else None
        if not parent or not name:
            return None
        node = taxonomy.find_node(parent)
        for child in node.children:
            if _near(child.name, name):
                return f"{parent}/{child.name}"
        path = taxonomy.add_folder(parent, name, (note or "").strip() or "Added automatically.")
        self.config.save()  # persisted, so the next clip is offered this folder
        self._load_tree()
        self.new_folders.append(path)
        log.info("created a new folder: %s", path)
        return path

    # -- frames -------------------------------------------------------------

    def extract(self, video: Path, context: ShotContext, work_dir: Path) -> list[Path]:
        return extract_frames(
            video,
            work_dir,
            start_s=context.start_s,
            duration_s=context.duration_s,
            count=self.config.ingest.frames_per_shot,
            max_edge=self.config.ingest.frame_max_edge,
            prefix=f"shot{context.shot_index:03d}",
        )

    # -- analysis -----------------------------------------------------------

    def _prepare(self, context: ShotContext) -> ShotContext:
        return context.model_copy(
            update={
                "client_context": self.client_context,
                "emotion_vocab": self.emotion_vocab,
                "category_options": self.category_options,
            }
        )

    def _normalise(self, result: AnalysisResult) -> list[tuple[str, str]]:
        """Snap categories onto real folders; drop fields this workspace lacks."""
        unmatched: list[tuple[str, str]] = []
        if self.leaves:
            raw = result.category
            result.category = match_category(raw, self.leaves)
            if raw and result.category is None:
                unmatched.append(("category", raw))
            secondary: list[str] = []
            for candidate in result.secondary_categories:
                matched = match_category(candidate, self.leaves)
                if matched and matched != result.category and matched not in secondary:
                    secondary.append(matched)
            result.secondary_categories = secondary[:2]
            if result.new_category and self.config.taxonomy.allow_new_folders:
                created = self._create_folder(result.new_category, result.new_category_note)
                if created:
                    result.category = created
                    result.secondary_categories = [
                        c for c in result.secondary_categories if c != created
                    ]
                    unmatched = [u for u in unmatched if u[0] != "category"]
        else:
            result.category = None
            result.secondary_categories = []
        if not self.config.client.featured_person:
            result.featured_person_in_shot = False
        return unmatched

    async def analyse_frames(
        self, frames: list[Path], context: ShotContext
    ) -> AnalysisOutcome:
        context = self._prepare(context)
        outcome = AnalysisOutcome(context=context, frames=frames)
        if not frames:
            outcome.status = "needs_review"
            outcome.error = "no usable frames could be extracted"
            return outcome

        outcome.cost_usd = self.provider.estimate_cost(frames)
        retry_error: str | None = None

        for attempt in (1, 2):
            try:
                result = await self.provider.analyse(frames, context, retry_error)
            except ValidationError as exc:
                retry_error = _validation_summary(exc)
            except TransientProviderError:
                # Not ours to solve inside one job: the queue retries this
                # source with backoff rather than flagging it for a human.
                raise
            except ProviderError as exc:
                retry_error = str(exc)
            else:
                unmatched = self._normalise(result)
                outcome.result = result
                outcome.oov = find_oov(result, self.vocab_overrides) + unmatched
                defects = DEFECT_FLAGS.intersection(result.quality_flags)
                if result.confidence < 0.35 or defects:
                    outcome.status = "needs_review"
                return outcome

            if attempt == 1:
                outcome.cost_usd += self.provider.estimate_cost(frames)
                log.warning(
                    "analysis attempt 1 failed for %s shot %d: %s",
                    context.source_filename, context.shot_index, retry_error,
                )

        outcome.status = "needs_review"
        outcome.error = retry_error
        return outcome

    async def analyse_shot(
        self, video: Path, context: ShotContext, work_dir: Path
    ) -> AnalysisOutcome:
        frames = self.extract(video, context, work_dir)
        return await self.analyse_frames(frames, context)


def _validation_summary(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors()[:6]:
        location = ".".join(str(p) for p in error["loc"])
        lines.append(f"- {location}: {error['msg']}")
    return "Schema validation errors:\n" + "\n".join(lines)
