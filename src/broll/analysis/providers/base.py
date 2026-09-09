"""Provider interfaces.

Two interfaces, deliberately separate: the vision interface is the wrong shape
for the transcript reranker, and reusing it produces awkward code.

Nothing outside this package may import a provider module directly. Everything
goes through ``registry.get_vision_provider`` / ``registry.get_text_provider``.
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel

from ..schema import AnalysisResult, ShotContext


class ProviderError(RuntimeError):
    """Anything the provider could not do: auth, transport, or bad output."""


class TransientProviderError(ProviderError):
    """The provider was momentarily unavailable - worth retrying later.

    A 503 while Google is overloaded is not a reason to mark a shot as needing
    human review; it is a reason to try again in a minute.
    """


TRANSIENT_MARKERS = (
    "503", "502", "504", "500", "429", "unavailable", "resource_exhausted",
    "overloaded", "high demand", "timeout", "timed out", "deadline",
    "connection", "temporarily",
)


def classify_error(message: str) -> type[ProviderError]:
    lowered = (message or "").lower()
    return (
        TransientProviderError
        if any(marker in lowered for marker in TRANSIENT_MARKERS)
        else ProviderError
    )


class MissingDependencyError(ProviderError):
    pass


class MissingCredentialsError(ProviderError):
    pass


class Pricing(BaseModel):
    """USD per 1M tokens. Public list prices - override in config if yours differ."""

    input_per_m: float
    output_per_m: float
    tokens_per_image: int
    prompt_tokens: int = 2400  # the shared prompt, dominated by the vocabularies
    output_tokens: int = 350

    def cost(self, image_count: int) -> float:
        input_tokens = self.prompt_tokens + tokens_per_image_total(self, image_count)
        return (
            input_tokens / 1_000_000 * self.input_per_m
            + self.output_tokens / 1_000_000 * self.output_per_m
        )

    def text_cost(self, prompt_tokens: int, output_tokens: int | None = None) -> float:
        out = self.output_tokens if output_tokens is None else output_tokens
        return (
            prompt_tokens / 1_000_000 * self.input_per_m
            + out / 1_000_000 * self.output_per_m
        )


def tokens_per_image_total(pricing: Pricing, image_count: int) -> int:
    return pricing.tokens_per_image * image_count


def encode_image(path: Path) -> tuple[str, str]:
    """Return (media_type, base64 data) for a frame."""
    suffix = path.suffix.lower()
    media_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
    }.get(suffix, "image/jpeg")
    return media_type, base64.standard_b64encode(path.read_bytes()).decode()


def estimate_prompt_tokens(text: str) -> int:
    """Rough token count for cost estimation only. ~4 characters per token."""
    return max(1, len(text) // 4)


class VisionProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def analyse(
        self,
        frames: list[Path],
        context: ShotContext,
        retry_error: str | None = None,
    ) -> AnalysisResult: ...

    @abstractmethod
    def estimate_cost(self, frames: list[Path]) -> float: ...


class TextProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def complete(self, prompt: str, schema: type[BaseModel]) -> BaseModel: ...

    @abstractmethod
    def estimate_cost(self, prompt: str) -> float: ...
