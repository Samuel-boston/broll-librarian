"""Orchestrates one shot: frames -> provider -> validated result.

On validation failure the provider is asked again once, with the error appended
to the prompt. If it fails a second time the shot is marked ``needs_review`` and
the rest of the source carries on - one bad shot must never fail a whole file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from ..config import WorkspaceConfig
from ..ingest.frames import extract_frames
from .prompt import PROMPT_VERSION
from .providers.base import ProviderError, TransientProviderError, VisionProvider
from .providers.registry import get_vision_provider
from .schema import DEFECT_FLAGS, AnalysisResult, ShotContext, find_oov

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


class Analyzer:
    def __init__(self, config: WorkspaceConfig, provider: VisionProvider | None = None):
        self.config = config
        self.provider = provider or get_vision_provider(config)

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

    async def analyse_frames(
        self, frames: list[Path], context: ShotContext
    ) -> AnalysisOutcome:
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
                # Not our problem to solve inside one job: let the queue retry
                # this source with backoff rather than flagging it for a human.
                raise
            except ProviderError as exc:
                retry_error = str(exc)
            else:
                outcome.result = result
                outcome.oov = find_oov(result, self.config.vocabulary_overrides)
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
