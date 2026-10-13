"""Google Gemini provider.

The default for bulk indexing: the lowest cost per image of the three, which is
what dominates the bill when analysing thousands of shots at 3 frames each.
Uses the google-genai SDK's native JSON-schema mode with AnalysisResult as the
response schema.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from pathlib import Path

from pydantic import BaseModel

from ...config import DEFAULT_MODELS
from ..prompt import SYSTEM_PROMPT, shot_prompt, static_prompt
from ..schema import AnalysisResult, ShotContext
from ..segmentation import (
    SEGMENT_SYSTEM_PROMPT,
    SegmentationResult,
    SegmentContext,
    build_segment_prompt,
)
from ..usage import Usage, UsageSink, report
from .base import (
    MissingCredentialsError,
    MissingDependencyError,
    Pricing,
    ProviderError,
    classify_error,
    TextProvider,
    VisionProvider,
    estimate_prompt_tokens,
)

# USD per 1M tokens, from Google's price list (checked 13 Oct 2026 against the first bill, which matched to
# the cent). A cached token costs a tenth of an input token. The 3.x Flash prices are the promotional rate
# that runs to 31 Dec 2026; they double on 1 Jan 2027, so re-check this table then.
# tokens_per_image is what a frame costs at the DEFAULT media resolution (1,120 for a Gemini 3 model);
# at `low` it is a quarter of that. It is used only for estimates: the cost that is booked comes from the
# token counts in each response.
PRICING: dict[str, Pricing] = {
    "gemini-3.8-flash": Pricing(input_per_m=0.75, output_per_m=3.75, cached_per_m=0.075, cache_storage_per_m_hour=0.50, tokens_per_image=1120),
    "gemini-3.7-flash": Pricing(input_per_m=0.75, output_per_m=3.75, cached_per_m=0.075, cache_storage_per_m_hour=0.50, tokens_per_image=1120),
    "gemini-3.6-flash": Pricing(input_per_m=0.75, output_per_m=3.75, cached_per_m=0.075, cache_storage_per_m_hour=0.50, tokens_per_image=1120),
    "gemini-3.5-flash": Pricing(input_per_m=1.50, output_per_m=9.00, cached_per_m=0.15, tokens_per_image=1120),
    "gemini-3.5-flash-lite": Pricing(input_per_m=0.30, output_per_m=2.50, cached_per_m=0.03, cache_storage_per_m_hour=1.00, tokens_per_image=1120),
    "gemini-3.1-flash-lite": Pricing(input_per_m=0.25, output_per_m=1.50, cached_per_m=0.025, cache_storage_per_m_hour=1.00, tokens_per_image=1120),
    # Still priced here for existing users; new API projects get a 404.
    "gemini-2.5-flash": Pricing(input_per_m=0.30, output_per_m=2.50, cached_per_m=0.03, tokens_per_image=300),
    "gemini-2.5-flash-lite": Pricing(input_per_m=0.10, output_per_m=0.40, cached_per_m=0.01, tokens_per_image=300),
    "gemini-2.5-pro": Pricing(input_per_m=1.25, output_per_m=10.00, cached_per_m=0.125, tokens_per_image=300),
}
DEFAULT_PRICING = Pricing(input_per_m=0.75, output_per_m=3.75, cached_per_m=0.075, tokens_per_image=1120)


log = logging.getLogger(__name__)


def _sdk():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - depends on the install extra
        raise MissingDependencyError(
            "The Gemini provider needs the SDK: pip install 'broll-librarian[gemini]'"
        ) from exc
    return genai, types


# Caught live on a free key: a request was sent, Google never answered, and with
# no timeout the worker waited on it for good - which, with the always-on server
# indexing one file at a time, stops all indexing. A vision request takes seconds;
# after this long it is abandoned as "timed out", which the queue retries later.
REQUEST_TIMEOUT_MS = 180_000


def _client(api_key: str | None):
    genai, types = _sdk()
    if not api_key:
        raise MissingCredentialsError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))




# Thinking: Gemini 3 takes a named level, a 2.5 model a token budget. The words in config are the same for both.
_LEVELS = {"minimal": "MINIMAL", "low": "LOW", "medium": "MEDIUM", "high": "HIGH"}
_BUDGETS = {"minimal": 0, "low": 512, "medium": 2048, "high": 8192}
# Models that refuse a level the others accept: the lowest they take, by name prefix.
_LOWEST_LEVEL = {"gemini-3.8-flash": "low"}
_RESOLUTIONS = {
    "low": "MEDIA_RESOLUTION_LOW", "medium": "MEDIA_RESOLUTION_MEDIUM", "high": "MEDIA_RESOLUTION_HIGH",
}


def thinking_config(types, model: str, level: str | None):
    """The `thinking_config` for a call, or None to leave the model's own default (which is "medium")."""
    level = (level or "default").strip().lower()
    if level in ("", "default", "auto"):
        return None
    if level not in _LEVELS:
        raise ValueError(f"unknown thinking level {level!r}: use minimal, low, medium, high or default")
    if model.startswith("gemini-2.5"):
        budget = _BUDGETS[level]
        if "pro" in model:
            budget = max(budget, 128)  # 2.5 Pro cannot think less than this
        return types.ThinkingConfig(thinking_budget=budget)
    for prefix, lowest in _LOWEST_LEVEL.items():
        if model.startswith(prefix) and level == "minimal":
            level = lowest
    return types.ThinkingConfig(thinking_level=_LEVELS[level])


def media_resolution_for(types, value: str | None):
    key = (value or "default").strip().lower()
    name = _RESOLUTIONS.get(key)
    return getattr(types.MediaResolution, name) if name else None


def _tokens(response) -> dict:
    """The token counts of a response, whatever this SDK version calls them."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return {}

    def number(*names: str) -> int:
        for name in names:
            value = getattr(meta, name, None)
            if value:
                return int(value)
        return 0

    image = 0
    for detail in getattr(meta, "prompt_tokens_details", None) or []:
        if str(getattr(detail.modality, "value", detail.modality)).upper() == "IMAGE":
            image += int(detail.token_count or 0)
    return {
        "input": number("prompt_token_count"),
        "cached": number("cached_content_token_count"),
        "output": number("candidates_token_count", "response_token_count"),
        "thoughts": number("thoughts_token_count"),
        "image": image,
    }


class _PromptCaches:
    """Google context caches for the part of the prompt that never changes.

    One cache per (key, model, text): created on first use, shared by every worker, replaced a little
    before it expires. A cache costs $0.50 per million tokens per hour to keep (about half a cent an hour
    for this prompt) and makes its tokens a tenth of the price on every call that reads it. If creating
    one fails for any reason, the call simply goes ahead with the prompt inline: a cache is an
    optimisation and must never be the reason a file fails.
    """

    TTL_S = 3600
    RENEW_BEFORE_S = 300
    RETRY_AFTER_FAILURE_S = 600

    def __init__(self) -> None:
        self._items: dict[str, tuple[str, float]] = {}
        self._failed_until: dict[str, float] = {}
        self._locks: dict[int, asyncio.Lock] = {}

    def _lock(self) -> asyncio.Lock:
        loop_id = id(asyncio.get_running_loop())
        return self._locks.setdefault(loop_id, asyncio.Lock())

    async def get(
        self, client, types, model: str, system: str, text: str, key_extra: str = "", on_created=None
    ) -> str | None:
        key = hashlib.sha256(f"{key_extra}|{model}|{system}|{text}".encode()).hexdigest()
        now = time.monotonic()
        item = self._items.get(key)
        if item and item[1] - now > self.RENEW_BEFORE_S:
            return item[0]
        if self._failed_until.get(key, 0) > now:
            return None
        async with self._lock():
            item = self._items.get(key)
            if item and item[1] - time.monotonic() > self.RENEW_BEFORE_S:
                return item[0]
            try:
                created = await client.aio.caches.create(
                    model=model,
                    config=types.CreateCachedContentConfig(
                        system_instruction=system,
                        contents=[types.Content(role="user", parts=[types.Part.from_text(text=text)])],
                        ttl=f"{self.TTL_S}s",
                        display_name="broll-static-prompt",
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("could not create a prompt cache, sending the prompt inline: %s", str(exc)[:200])
                self._failed_until[key] = time.monotonic() + self.RETRY_AFTER_FAILURE_S
                return None
            self._items[key] = (created.name, time.monotonic() + self.TTL_S)
            if on_created is not None:
                tokens = getattr(getattr(created, "usage_metadata", None), "total_token_count", None)
                on_created(int(tokens or 0))
            return created.name

    def forget(self, name: str) -> None:
        for key, (cached_name, _) in list(self._items.items()):
            if cached_name == name:
                del self._items[key]


_CACHES = _PromptCaches()


class _GeminiBase:
    """What the vision and text providers share: one place that makes a call and books what it cost."""

    model: str
    api_key: str | None
    pricing: Pricing
    usage_sink: UsageSink | None = None
    thinking: str = "minimal"

    def _book(self, response, kind: str, *, ok: bool, note: str = "") -> None:
        counts = _tokens(response)
        if not counts:
            return
        cost = self.pricing.usage_cost(
            counts["input"], counts["output"], counts["cached"], counts["thoughts"]
        )
        report(
            Usage(
                model=self.model, kind=kind, input_tokens=counts["input"], cached_tokens=counts["cached"],
                output_tokens=counts["output"], thought_tokens=counts["thoughts"],
                image_tokens=counts["image"], cost_usd=cost, ok=ok, note=note,
            ),
            self.usage_sink,
        )

    def _book_cache(self, tokens: int) -> None:
        """Keeping a prompt in a context cache for its hour costs a little; book it when it is made."""
        cost = tokens / 1_000_000 * self.pricing.cache_storage_per_m_hour * (_PromptCaches.TTL_S / 3600)
        report(
            Usage(model=self.model, kind="cache", input_tokens=tokens, cost_usd=cost,
                  estimated=True, note="storage for the prompt cache, one hour"),
            self.usage_sink,
        )

    def _book_lost_call(self, kind: str, error: str, sent_tokens: int) -> None:
        """A request that timed out may still have been answered, and billed, on Google's side.

        Nothing came back to read the usage from, so what was sent is booked as input, marked as an
        estimate. (A 429 or 503 is refused before it is billed and costs nothing.)
        """
        lowered = error.lower()
        if not any(word in lowered for word in ("timeout", "timed out", "deadline")):
            return
        report(
            Usage(
                model=self.model, kind=kind, input_tokens=sent_tokens, ok=False, estimated=True,
                cost_usd=self.pricing.usage_cost(sent_tokens, 0), note="no answer: " + error[:120],
            ),
            self.usage_sink,
        )

    async def _generate(self, client, types, contents, config_kwargs: dict, kind: str, sent_tokens: int):
        """One request. Returns the response; the caller books it once it knows whether the answer was usable."""
        thinking = thinking_config(types, self.model, self.thinking)
        if thinking is not None:
            config_kwargs["thinking_config"] = thinking
        try:
            return await client.aio.models.generate_content(
                model=self.model, contents=contents, config=types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as exc:
            self._book_lost_call(kind, str(exc), sent_tokens)
            raise classify_error(str(exc))(f"gemini request failed: {exc}") from exc


def _parse(response, schema: type[BaseModel]):
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, schema):
        return parsed
    if parsed is not None:
        return schema.model_validate(parsed)
    if response.text:
        return schema.model_validate_json(response.text)
    raise ProviderError("gemini returned no structured output")


class GeminiVisionProvider(_GeminiBase, VisionProvider):
    name = "gemini"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        *,
        thinking: str = "minimal",
        media_resolution: str = "low",
        segment_media_resolution: str = "low",
        photo_media_resolution: str = "default",
        cache_prompt: bool = True,
        usage_sink: UsageSink | None = None,
    ):
        self.model = model or DEFAULT_MODELS["gemini"]
        self.api_key = api_key
        self.pricing = PRICING.get(self.model, DEFAULT_PRICING)
        self.thinking = thinking
        self.media_resolution = media_resolution
        self.segment_media_resolution = segment_media_resolution
        self.photo_media_resolution = photo_media_resolution
        self.cache_prompt = cache_prompt
        self.usage_sink = usage_sink

    async def analyse(
        self,
        frames: list[Path],
        context: ShotContext,
        retry_error: str | None = None,
    ) -> AnalysisResult:
        _, types = _sdk()
        client = _client(self.api_key)

        # The prompt is the same for every shot of a workspace and the frames are different for each, so
        # the prompt goes first and the frames last: Gemini reuses (and discounts) a repeated prefix, and
        # with the frames first there was never a prefix to reuse.
        static = static_prompt(context)
        images = [
            types.Part.from_bytes(data=frame.read_bytes(), mime_type="image/jpeg") for frame in frames
        ]
        config_kwargs: dict = dict(
            response_mime_type="application/json", response_schema=AnalysisResult, temperature=0.2,
        )
        # A photograph is one picture carrying all the evidence; a video shot is several, each worth less.
        wanted = self.photo_media_resolution if context.media_kind == "image" else self.media_resolution
        resolution = media_resolution_for(types, wanted)
        if resolution is not None:
            config_kwargs["media_resolution"] = resolution

        sent = estimate_prompt_tokens(static) + self.pricing.tokens_per_image * len(frames)
        shot = types.Part.from_text(text=shot_prompt(context, retry_error))

        response = None
        for attempt in (1, 2):
            cache_name = None
            if self.cache_prompt:
                cache_name = await _CACHES.get(
                    client, types, self.model, SYSTEM_PROMPT, static, self.api_key or "", self._book_cache
                )
            call_config = dict(config_kwargs)
            if cache_name:
                call_config["cached_content"] = cache_name
                parts = [shot, *images]
            else:
                call_config["system_instruction"] = SYSTEM_PROMPT
                parts = [types.Part.from_text(text=static), shot, *images]
            try:
                response = await self._generate(
                    client, types, [types.Content(role="user", parts=parts)], call_config, "vision", sent
                )
                break
            except ProviderError as exc:
                if cache_name and "cache" in str(exc).lower():
                    # It expired or was deleted under us: make a new one and go again, once.
                    _CACHES.forget(cache_name)
                    if attempt == 1:
                        continue
                raise
        try:
            result = _parse(response, AnalysisResult)
        except Exception:
            self._book(response, "vision", ok=False, note="answer did not validate")
            raise
        self._book(response, "vision", ok=True)
        return result

    def estimate_cost(self, frames: list[Path]) -> float:
        return self.pricing.cost(len(frames))

    async def segment(
        self,
        frames: list[Path],
        times: list[float],
        context: SegmentContext,
        retry_error: str | None = None,
    ) -> SegmentationResult:
        _, types = _sdk()
        client = _client(self.api_key)

        parts = []
        for frame, t in zip(frames, times):
            parts.append(types.Part.from_text(text=f"Frame at {t:.1f}s:"))
            parts.append(types.Part.from_bytes(data=frame.read_bytes(), mime_type="image/jpeg"))
        parts.append(types.Part.from_text(text=build_segment_prompt(context, times, retry_error)))
        config_kwargs: dict = dict(
            system_instruction=SEGMENT_SYSTEM_PROMPT, response_mime_type="application/json",
            response_schema=SegmentationResult, temperature=0.1,
        )
        resolution = media_resolution_for(types, self.segment_media_resolution)
        if resolution is not None:
            config_kwargs["media_resolution"] = resolution
        sent = 900 + self.pricing.tokens_per_image * len(frames)

        response = await self._generate(
            client, types, [types.Content(role="user", parts=parts)], config_kwargs, "segment", sent
        )
        try:
            result = _parse(response, SegmentationResult)
        except Exception:
            self._book(response, "segment", ok=False, note="answer did not validate")
            raise
        self._book(response, "segment", ok=True)
        return result

    def estimate_segment_cost(self, frame_count: int) -> float:
        return self.pricing.segment_cost(frame_count)


class GeminiTextProvider(_GeminiBase, TextProvider):
    name = "gemini"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        *,
        thinking: str = "minimal",
        usage_sink: UsageSink | None = None,
    ):
        self.model = model or DEFAULT_MODELS["gemini"]
        self.api_key = api_key
        self.pricing = PRICING.get(self.model, DEFAULT_PRICING)
        self.thinking = thinking
        self.usage_sink = usage_sink

    async def complete(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        _, types = _sdk()
        client = _client(self.api_key)
        config_kwargs: dict = dict(
            response_mime_type="application/json", response_schema=schema, temperature=0.2,
        )
        response = await self._generate(
            client, types, prompt, config_kwargs, "text", estimate_prompt_tokens(prompt)
        )
        try:
            result = _parse(response, schema)
        except Exception:
            self._book(response, "text", ok=False, note="answer did not validate")
            raise
        self._book(response, "text", ok=True)
        return result

    def estimate_cost(self, prompt: str) -> float:
        return self.pricing.text_cost(estimate_prompt_tokens(prompt))
