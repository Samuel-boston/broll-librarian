"""OpenAI provider.

Uses the Responses API's structured-output parse helper so AnalysisResult is
enforced as a strict JSON schema rather than parsed out of free text.
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

# USD per 1M tokens.
PRICING: dict[str, Pricing] = {
    "gpt-4.1-mini": Pricing(input_per_m=0.40, output_per_m=1.60, tokens_per_image=800),
    "gpt-4.1": Pricing(input_per_m=2.00, output_per_m=8.00, tokens_per_image=800),
    "gpt-4.1-nano": Pricing(input_per_m=0.10, output_per_m=0.40, tokens_per_image=800),
}
DEFAULT_PRICING = Pricing(input_per_m=0.40, output_per_m=1.60, tokens_per_image=800)


def _client(api_key: str | None):
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover - depends on the install extra
        raise MissingDependencyError(
            "The OpenAI provider needs the SDK: pip install 'broll-librarian[openai]'"
        ) from exc
    if not api_key:
        raise MissingCredentialsError("OPENAI_API_KEY is not set.")
    return AsyncOpenAI(api_key=api_key)


class OpenAIVisionProvider(VisionProvider):
    name = "openai"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or DEFAULT_MODELS["openai"]
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
                {"type": "input_image", "image_url": f"data:{media_type};base64,{data}"}
            )
        content.append({"type": "input_text", "text": build_user_prompt(context, retry_error)})

        try:
            response = await client.responses.parse(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                input=[{"role": "user", "content": content}],
                text_format=AnalysisResult,
            )
        except Exception as exc:
            raise ProviderError(f"openai request failed: {exc}") from exc

        parsed = response.output_parsed
        if parsed is None:
            raise ProviderError("openai returned no structured output")
        return parsed

    def estimate_cost(self, frames: list[Path]) -> float:
        return self.pricing.cost(len(frames))


class OpenAITextProvider(TextProvider):
    name = "openai"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or DEFAULT_MODELS["openai"]
        self.api_key = api_key
        self.pricing = PRICING.get(self.model, DEFAULT_PRICING)

    async def complete(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        client = _client(self.api_key)
        try:
            response = await client.responses.parse(
                model=self.model,
                input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                text_format=schema,
            )
        except Exception as exc:
            raise ProviderError(f"openai request failed: {exc}") from exc
        if response.output_parsed is None:
            raise ProviderError("openai returned no structured output")
        return response.output_parsed

    def estimate_cost(self, prompt: str) -> float:
        return self.pricing.text_cost(estimate_prompt_tokens(prompt))
