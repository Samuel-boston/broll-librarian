"""Anthropic provider (Claude).

Uses the official ``anthropic`` SDK and its structured-output support, so the
response is validated against AnalysisResult by the SDK rather than parsed out
of free text.

Model default is ``claude-opus-5``. Anthropic returns the richest structured
descriptions of the three providers and costs the most per image; for a bulk
index of thousands of shots, switch the model (or the provider) in config -
``provider.vision_model: claude-haiku-4-5`` is a one-line change.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from ...config import DEFAULT_MODELS
from ..prompt import SYSTEM_PROMPT, build_user_prompt
from ..schema import AnalysisResult, ShotContext
from .base import (
    MissingCredentialsError,
    MissingDependencyError,
    Pricing,
    ProviderError,
    TextProvider,
    VisionProvider,
    encode_image,
    estimate_prompt_tokens,
)

# USD per 1M tokens (Anthropic list prices). Image tokens are roughly
# (width x height) / 750; a 768x432 frame is ~440.
PRICING: dict[str, Pricing] = {
    "claude-opus-5": Pricing(input_per_m=5.00, output_per_m=25.00, tokens_per_image=450),
    "claude-sonnet-5": Pricing(input_per_m=2.00, output_per_m=10.00, tokens_per_image=450),
    "claude-haiku-4-5": Pricing(input_per_m=1.00, output_per_m=5.00, tokens_per_image=450),
}
DEFAULT_PRICING = Pricing(input_per_m=5.00, output_per_m=25.00, tokens_per_image=450)

MAX_TOKENS = 4096


def _client(api_key: str | None):
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on the install extra
        raise MissingDependencyError(
            "The Anthropic provider needs the SDK: pip install 'broll-librarian[anthropic]'"
        ) from exc
    if not api_key:
        raise MissingCredentialsError("ANTHROPIC_API_KEY is not set.")
    return anthropic.AsyncAnthropic(api_key=api_key)


class AnthropicVisionProvider(VisionProvider):
    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or DEFAULT_MODELS["anthropic"]
        self.api_key = api_key
        self.pricing = PRICING.get(self.model, DEFAULT_PRICING)

    async def analyse(
        self,
        frames: list[Path],
        context: ShotContext,
        retry_error: str | None = None,
    ) -> AnalysisResult:
        client = _client(self.api_key)
        content: list[dict] = []
        for frame in frames:
            media_type, data = encode_image(frame)
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": data},
                }
            )
        content.append({"type": "text", "text": build_user_prompt(context, retry_error)})

        try:
            response = await client.messages.parse(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
                output_format=AnalysisResult,
            )
        except Exception as exc:  # SDK raises typed errors; the caller retries
            raise ProviderError(f"anthropic request failed: {exc}") from exc

        if response.stop_reason == "refusal":
            raise ProviderError("anthropic declined to analyse this shot")
        parsed = response.parsed_output
        if parsed is None:
            raise ProviderError("anthropic returned no structured output")
        return parsed

    def estimate_cost(self, frames: list[Path]) -> float:
        return self.pricing.cost(len(frames))


class AnthropicTextProvider(TextProvider):
    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or DEFAULT_MODELS["anthropic"]
        self.api_key = api_key
        self.pricing = PRICING.get(self.model, DEFAULT_PRICING)

    async def complete(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        client = _client(self.api_key)
        try:
            response = await client.messages.parse(
                model=self.model,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
                output_format=schema,
            )
        except Exception as exc:
            raise ProviderError(f"anthropic request failed: {exc}") from exc
        if response.parsed_output is None:
            raise ProviderError("anthropic returned no structured output")
        return response.parsed_output

    def estimate_cost(self, prompt: str) -> float:
        return self.pricing.text_cost(estimate_prompt_tokens(prompt))
