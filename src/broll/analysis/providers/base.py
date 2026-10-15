"""Provider interfaces.

Two interfaces, deliberately separate: the vision interface is the wrong shape
for the transcript reranker, and reusing it produces awkward code.

Nothing outside this package may import a provider module directly. Everything
goes through ``registry.get_vision_provider`` / ``registry.get_text_provider``.
"""

from __future__ import annotations

import base64
import re
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel

from ..schema import AnalysisResult, ShotContext
from ..segmentation import SegmentationResult, SegmentContext


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


#: Not about this file but about the setup: a dead Google sign-in, a bad or blocked API key, a spent daily
#: quota. Every file will fail the same way until a person fixes it, so retrying each one three times in a
#: few seconds only burns the queue. The worker waits instead, and no file is marked failed for it.
ENVIRONMENT_MARKERS = (
    "invalid_grant", "expired or revoked", "reauthenticate", "api key not valid", "api_key_invalid",
    "unauthenticated", "invalid authentication credentials", "billing", "per day", "perday",
    "permission_denied", "drive is not connected",
    # Drive's per-account download quota ("downloadQuotaExceeded"): every file fails the same way until it
    # resets, a day later. 614 files were marked "couldn't be analysed" for it before this was a pause.
    "downloadquotaexceeded", "download quota",
)


#: A Drive file read in place (a video past `ingest.stream_above_gb`) is opened by ffprobe / ffmpeg, which
#: report a refused read as "<drive url>: Server returned 403 Forbidden (access denied)" with none of the words
#: above. On a Drive media URL that 403 is the download quota (or the same wall in another guise): every file
#: meets it until it lifts, so it is told apart from a bad file by the address it names.
DRIVE_READ_REFUSED = re.compile(r"googleapis\.com/drive/.*\b403\b")


def is_drive_quota_error(message: str) -> bool:
    """Drive refusing a read because of its download quota - in any of the shapes it comes in.

    Unlike a dead sign-in or a bad key, this one is not all-or-nothing: Drive's allowance recovers a
    little at a time and lets some requests through while it refuses others, even on the same file seconds
    apart. So the worker holds back the one file, not the whole queue (see Worker._hold_for_quota).
    """
    lowered = (message or "").lower()
    return "downloadquotaexceeded" in lowered or "download quota" in lowered or bool(DRIVE_READ_REFUSED.search(lowered))


def is_environment_error(message: str) -> bool:
    lowered = (message or "").lower()
    return any(marker in lowered for marker in ENVIRONMENT_MARKERS) or bool(DRIVE_READ_REFUSED.search(lowered))


def classify_error(message: str) -> type[ProviderError]:
    lowered = (message or "").lower()
    return (
        TransientProviderError
        if any(marker in lowered for marker in TRANSIENT_MARKERS) or is_environment_error(lowered)
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
    #: What a token read from a cache costs. Unset means a tenth of the input price (Google's rate).
    cached_per_m: float | None = None
    #: What keeping a token in a context cache costs for an hour (per 1M tokens).
    cache_storage_per_m_hour: float = 0.0
    # What a call sends besides its pictures: the system prompt, the vocabularies, the client brief and the
    # folder tree. Measured on Adam Kunder's library (75 folders with notes): about 9,100 tokens. A library
    # with a short tree is nearer 3,500. The answer is about 600 tokens of JSON (more if the model thinks).
    prompt_tokens: int = 9000
    output_tokens: int = 650

    def usage_cost(
        self, input_tokens: int, output_tokens: int, cached_tokens: int = 0, thought_tokens: int = 0
    ) -> float:
        """What one call really cost, from the token counts the provider reported.

        `input_tokens` includes the cached ones (that is how Gemini reports it); thinking is billed as
        output, which is what the old formula left out.
        """
        cached = min(max(cached_tokens, 0), max(input_tokens, 0))
        cached_rate = self.cached_per_m if self.cached_per_m is not None else self.input_per_m * 0.1
        return (
            (input_tokens - cached) / 1_000_000 * self.input_per_m
            + cached / 1_000_000 * cached_rate
            + (output_tokens + thought_tokens) / 1_000_000 * self.output_per_m
        )

    def cost(self, image_count: int) -> float:
        input_tokens = self.prompt_tokens + tokens_per_image_total(self, image_count)
        return (
            input_tokens / 1_000_000 * self.input_per_m
            + self.output_tokens / 1_000_000 * self.output_per_m
        )

    def segment_cost(self, image_count: int) -> float:
        """Looking through a clip: a short prompt, many small frames, a short answer."""
        input_tokens = 900 + tokens_per_image_total(self, image_count)
        return (
            input_tokens / 1_000_000 * self.input_per_m
            + 250 / 1_000_000 * self.output_per_m
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
    #: Where the real token usage of each call goes (see analysis/usage.py). Set by the registry.
    usage_sink = None

    @abstractmethod
    async def analyse(
        self,
        frames: list[Path],
        context: ShotContext,
        retry_error: str | None = None,
    ) -> AnalysisResult: ...

    @abstractmethod
    def estimate_cost(self, frames: list[Path]) -> float: ...

    async def segment(
        self,
        frames: list[Path],
        times: list[float],
        context: SegmentContext,
        retry_error: str | None = None,
    ) -> SegmentationResult:
        """Split a stretch of a clip into usable / setup / dead segments. See segmentation.py."""
        raise ProviderError(f"{self.name} cannot look through a clip for its usable parts yet")

    def estimate_segment_cost(self, frame_count: int) -> float:
        return 0.0


class TextProvider(ABC):
    name: str = "base"
    usage_sink = None

    @abstractmethod
    async def complete(self, prompt: str, schema: type[BaseModel]) -> BaseModel: ...

    @abstractmethod
    def estimate_cost(self, prompt: str) -> float: ...
