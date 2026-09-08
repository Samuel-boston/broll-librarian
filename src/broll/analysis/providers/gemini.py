"""Google Gemini provider.

The default for bulk indexing: the lowest cost per image of the three, which is
what dominates the bill when analysing thousands of shots at 3 frames each.
Uses the google-genai SDK's native JSON-schema mode with AnalysisResult as the
response schema.
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
    estimate_prompt_tokens,
)

# USD per 1M tokens. Gemini bills an image at 258 tokens per 768x768 tile.
PRICING: dict[str, Pricing] = {
    "gemini-2.5-flash": Pricing(input_per_m=0.30, output_per_m=2.50, tokens_per_image=300),
    "gemini-2.5-flash-lite": Pricing(input_per_m=0.10, output_per_m=0.40, tokens_per_image=300),
    "gemini-2.5-pro": Pricing(input_per_m=1.25, output_per_m=10.00, tokens_per_image=300),
}
DEFAULT_PRICING = Pricing(input_per_m=0.30, output_per_m=2.50, tokens_per_image=300)


def _sdk():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - depends on the install extra
        raise MissingDependencyError(
            "The Gemini provider needs the SDK: pip install 'broll-librarian[gemini]'"
        ) from exc
    return genai, types


def _client(api_key: str | None):
    genai, _ = _sdk()
    if not api_key:
        raise MissingCredentialsError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")
    return genai.Client(api_key=api_key)


class GeminiVisionProvider(VisionProvider):
    name = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or DEFAULT_MODELS["gemini"]
        self.api_key = api_key
        self.pricing = PRICING.get(self.model, DEFAULT_PRICING)

    async def analyse(
        self,
        frames: list[Path],
        context: ShotContext,
        retry_error: str | None = None,
    ) -> AnalysisResult:
        _, types = _sdk()
        client = _client(self.api_key)

        parts = [
            types.Part.from_bytes(data=frame.read_bytes(), mime_type="image/jpeg")
            for frame in frames
        ]
        parts.append(types.Part.from_text(text=build_user_prompt(context, retry_error)))

        try:
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=AnalysisResult,
                    temperature=0.2,
                ),
            )
        except Exception as exc:
            raise ProviderError(f"gemini request failed: {exc}") from exc

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, AnalysisResult):
            return parsed
        if parsed is not None:
            return AnalysisResult.model_validate(parsed)
        if response.text:
            return AnalysisResult.model_validate_json(response.text)
        raise ProviderError("gemini returned no structured output")

    def estimate_cost(self, frames: list[Path]) -> float:
        return self.pricing.cost(len(frames))


class GeminiTextProvider(TextProvider):
    name = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or DEFAULT_MODELS["gemini"]
        self.api_key = api_key
        self.pricing = PRICING.get(self.model, DEFAULT_PRICING)

    async def complete(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        _, types = _sdk()
        client = _client(self.api_key)
        try:
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.2,
                ),
            )
        except Exception as exc:
            raise ProviderError(f"gemini request failed: {exc}") from exc
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            return parsed if isinstance(parsed, schema) else schema.model_validate(parsed)
        if response.text:
            return schema.model_validate_json(response.text)
        raise ProviderError("gemini returned no structured output")

    def estimate_cost(self, prompt: str) -> float:
        return self.pricing.text_cost(estimate_prompt_tokens(prompt))
