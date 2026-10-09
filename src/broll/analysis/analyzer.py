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
from ..ingest.frames import Frame, extract_frames_timed, extract_still, frame_count_for, wants_fast_seek
from .limiter import RateLimiter
from .prompt import PROMPT_VERSION, render_client_context
from .providers.base import ProviderError, TransientProviderError, VisionProvider
from .providers.registry import get_vision_provider
from .schema import (
    DEFECT_FLAGS,
    EMOTIONS,
    MOODS,
    TAG_STOPLIST,
    AnalysisResult,
    CameraMove,
    Pace,
    ShotContext,
    find_oov,
    normalise_term,
)

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
    #: Why a person should look at this shot (empty when nothing is in doubt).
    review_reasons: list[str] = field(default_factory=list)
    #: A new folder the model suggested: (path, note). Held for approval, not created.
    proposal: tuple[str, str] | None = None

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


def _stem(word: str) -> str:
    return word[:5] if len(word) > 5 else word


def _mentioned(term: str, text: str) -> bool:
    """Whether every word of `term` (loosely stemmed) appears in `text`."""
    have = {_stem(w) for w in re.findall(r"[a-z0-9']+", text.lower())}
    words = re.findall(r"[a-z0-9']+", term.lower())
    return bool(words) and all(_stem(w) in have for w in words)


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
        self.limiter = RateLimiter(config.ingest.requests_per_minute)

        self.new_folders: list[str] = []
        try:
            self._config_stamp = config.config_path.stat().st_mtime
        except OSError:
            self._config_stamp = None
        self._load_tree()
        # The featured person's name, in every form a tag might take. It is
        # recorded as featured_person_in_shot; as a tag it is on nearly every
        # clip in their library, so it only adds noise - on the card, and in
        # search, where it matched half of "clip of Adam meditating".
        parts = (config.client.featured_person or "").lower().split()
        self.name_terms = ({" ".join(parts), *[p for p in parts if len(p) >= 3]}
                           if parts else set())
        self.emotion_vocab = list(config.client.emotions) or list(EMOTIONS)
        self.client_context = render_client_context(config.client)
        # Themes are a closed list: the model may only pick from it. Compared in normalised form.
        self.theme_options = list(config.client.themes)
        self._themes = {normalise_term(t): normalise_term(t) for t in self.theme_options}
        # Words that name a feeling, a look or an idea. As a *visible* tag they are interpretation.
        self._concept_words = (
            set(MOODS) | set(EMOTIONS) | {normalise_term(e) for e in config.client.emotions}
            | set(self._themes)
        )

        overrides = {k: list(v) for k, v in config.vocabulary_overrides.items()}
        overrides.setdefault("emotions", []).extend(config.client.emotions)
        self.vocab_overrides = overrides

    def _load_tree(self) -> None:
        tree = self.config.taxonomy.category_leaves() if self.config.taxonomy.mode == "tree" else []
        self.leaves = [path for path, _ in tree]
        self.category_options = [f"{path} — {note}" if note else path for path, note in tree]

    def _refresh_tree(self) -> None:
        """Offer the folders as they are now. A folder approved while this runs (on the Review page,
        or with `broll folders approve` from another process) is available to the very next clip."""
        try:
            stamp = self.config.config_path.stat().st_mtime
        except OSError:
            stamp = None
        if stamp is not None and stamp != self._config_stamp:
            from ..config import load_workspace_config

            try:
                self.config.taxonomy.tree = load_workspace_config(self.config.id).taxonomy.tree
            except Exception:  # noqa: BLE001 - keep working with the folders already known
                log.warning("could not re-read the folder list from %s", self.config.config_path)
        self._config_stamp = stamp
        self._load_tree()

    def _resolve_proposal(self, proposal: str, note: str | None):
        """Sort a suggested folder into: an existing near-match, or a genuinely new one.

        Guard rails, because an unchecked model invents "Jet Ski", "Jetski" and "Jet Skiing" on
        three consecutive clips: the folder must sit under an existing one, and a near-match to an
        existing sibling reuses it. Returns (existing path, None) or (None, (new path, note)),
        or (None, None) when the suggestion is unusable.
        """
        taxonomy = self.config.taxonomy
        parent_raw, _, name_raw = proposal.strip().strip("/").rpartition("/")
        name = _clean_folder_name(name_raw)
        parent = match_category(parent_raw, taxonomy.parent_folders()) if parent_raw else None
        if not parent or not name:
            return None, None
        node = taxonomy.find_node(parent)
        for child in node.children:
            if _near(child.name, name):
                return f"{parent}/{child.name}", None
        return None, (f"{parent}/{name}", (note or "").strip() or "Added automatically.")

    def _create_folder(self, path: str, note: str) -> str:
        """Add a folder to the client's tree now (only when auto-creation is on)."""
        taxonomy = self.config.taxonomy
        parent, _, name = path.rpartition("/")
        created = taxonomy.add_folder(parent, name, note)
        self.config.save()  # persisted, so the next clip is offered this folder
        self._load_tree()
        self.new_folders.append(created)
        log.info("created a new folder: %s", created)
        return created

    # -- frames -------------------------------------------------------------

    def extract_timed(self, video, context: ShotContext, work_dir: Path) -> list[Frame]:
        # A photograph has one frame, and sending it three times would only
        # triple the bill. ffmpeg does the decode either way, HEIC included.
        if context.media_kind == "image":
            paths = extract_still(
                video, work_dir,
                max_edge=self.config.ingest.frame_max_edge,
                prefix=f"shot{context.shot_index:03d}",
            )
            return [Frame(p, 0.0) for p in paths]
        ingest = self.config.ingest
        return extract_frames_timed(
            video,
            work_dir,
            start_s=context.start_s,
            duration_s=context.duration_s,
            count=frame_count_for(
                context.duration_s, ingest.frames_per_shot, ingest.max_frames_per_shot,
                ingest.frame_every_s,
            ),
            max_edge=ingest.frame_max_edge,
            prefix=f"shot{context.shot_index:03d}",
            fast=wants_fast_seek(context.width, context.height),
        )

    def extract(self, video, context: ShotContext, work_dir: Path) -> list[Path]:
        return [f.path for f in self.extract_timed(video, context, work_dir)]

    # -- analysis -----------------------------------------------------------

    def _prepare(self, context: ShotContext) -> ShotContext:
        self._refresh_tree()
        return context.model_copy(
            update={
                "client_context": self.client_context,
                "emotion_vocab": self.emotion_vocab,
                "category_options": self.category_options,
                "theme_options": self.theme_options,
            }
        )

    def _normalise(self, result: AnalysisResult) -> tuple[list[tuple[str, str]], tuple[str, str] | None]:
        """Snap categories onto real folders, hold the model to its own evidence, and drop fields this
        workspace lacks. Returns (unmatched terms, a folder proposal)."""
        unmatched: list[tuple[str, str]] = []
        proposal: tuple[str, str] | None = None
        taxonomy = self.config.taxonomy
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
            if result.new_category and taxonomy.allow_new_folders:
                existing, new = self._resolve_proposal(result.new_category, result.new_category_note)
                if existing and existing in self.leaves:
                    result.category = existing
                    unmatched = [u for u in unmatched if u[0] != "category"]
                elif new and taxonomy.auto_create_folders:
                    result.category = self._create_folder(*new)
                    unmatched = [u for u in unmatched if u[0] != "category"]
                elif new:
                    proposal = new  # held for a person to approve; the clip stays in the closest folder
            result.secondary_categories = [c for c in result.secondary_categories if c != result.category]
        else:
            result.category = None
            result.category_confidence = None
            result.secondary_categories = []
        if not self.config.client.featured_person:
            result.featured_person_in_shot = False
        self._hold_to_the_evidence(result)
        return unmatched, proposal

    def _hold_to_the_evidence(self, result: AnalysisResult) -> None:
        """Keep tags to what is visible and themes to the client's list.

        The prompt asks for this; this makes it so. A feeling or an idea in `tags` is
        interpretation, and one on every calm clip is how a library starts returning half of itself
        for a search that should find a few clips.
        """
        seen_text = " ".join(
            [result.caption, *result.observations, *result.subjects, result.action or "",
             result.setting, result.setting_detail or ""]
        )
        themes = [t for t in dict.fromkeys(result.themes) if t in self._themes]
        tags: list[str] = []
        for tag in result.tags:
            if tag in TAG_STOPLIST or tag in self.name_terms or tag in tags:
                continue
            if tag in self._concept_words and not _mentioned(tag, seen_text):
                # A theme the model put in the wrong field still says something: keep it as a theme.
                if tag in self._themes and tag not in themes:
                    themes.append(tag)
                continue
            tags.append(tag)
        result.tags = tags
        result.themes = themes[:4]

    def _review_reasons(self, result: AnalysisResult, unmatched: list[tuple[str, str]]) -> list[str]:
        ingest = self.config.ingest
        reasons: list[str] = []
        if result.confidence < ingest.review_below_confidence:
            reasons.append("low_confidence")
        if (
            self.leaves and result.category_confidence is not None
            and result.category_confidence < ingest.review_below_category_confidence
        ):
            reasons.append("low_category_confidence")
        if any(field == "category" for field, _ in unmatched):
            reasons.append("category_unmatched")
        if DEFECT_FLAGS.intersection(result.quality_flags):
            reasons.append("quality_defect")
        return reasons

    async def analyse_frames(
        self, frames: list[Path], context: ShotContext
    ) -> AnalysisOutcome:
        context = self._prepare(context)
        outcome = AnalysisOutcome(context=context, frames=frames)
        if not frames:
            outcome.status = "needs_review"
            outcome.review_reasons = ["analysis_failed"]
            outcome.error = "no usable frames could be extracted"
            return outcome

        outcome.cost_usd = self.provider.estimate_cost(frames)
        retry_error: str | None = None

        for attempt in (1, 2):
            try:
                waited = await self.limiter.acquire()
                if waited > 1:
                    log.debug("held %.1fs for the request cap", waited)
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
                unmatched, proposal = self._normalise(result)
                if context.media_kind == "image":
                    # Not the model's to judge: a photograph cannot move, and a
                    # model that says it pans left is describing an illusion.
                    result.camera_movement = CameraMove.static
                    result.pace = Pace.still
                outcome.result = result
                outcome.proposal = proposal
                outcome.oov = find_oov(result, self.vocab_overrides) + unmatched
                outcome.review_reasons = self._review_reasons(result, unmatched)
                if outcome.review_reasons:
                    outcome.status = "needs_review"
                return outcome

            if attempt == 1:
                outcome.cost_usd += self.provider.estimate_cost(frames)
                log.warning(
                    "analysis attempt 1 failed for %s shot %d: %s",
                    context.source_filename, context.shot_index, retry_error,
                )

        outcome.status = "needs_review"
        outcome.review_reasons = ["analysis_failed"]
        outcome.error = retry_error
        return outcome

    async def analyse_shot(
        self, video: Path, context: ShotContext, work_dir: Path
    ) -> AnalysisOutcome:
        frames = self.extract_timed(video, context, work_dir)
        context = context.model_copy(update={"frame_times": [round(f.t, 2) for f in frames]})
        return await self.analyse_frames([f.path for f in frames], context)


def _validation_summary(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors()[:6]:
        location = ".".join(str(p) for p in error["loc"])
        lines.append(f"- {location}: {error['msg']}")
    return "Schema validation errors:\n" + "\n".join(lines)
