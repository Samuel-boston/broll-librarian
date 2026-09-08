"""The only place that knows which provider modules exist."""

from __future__ import annotations

from ...config import WorkspaceConfig
from .base import MissingCredentialsError, TextProvider, VisionProvider

VISION_PROVIDERS = ("gemini", "anthropic", "openai", "mock")
TEXT_PROVIDERS = ("gemini", "anthropic", "openai", "mock")


def get_vision_provider(config: WorkspaceConfig) -> VisionProvider:
    name = config.provider.vision
    model = config.provider.resolved_vision_model()
    key = config.api_key(name)

    if name == "gemini":
        from .gemini import GeminiVisionProvider

        return GeminiVisionProvider(model=model, api_key=key)
    if name == "anthropic":
        from .anthropic import AnthropicVisionProvider

        return AnthropicVisionProvider(model=model, api_key=key)
    if name == "openai":
        from .openai import OpenAIVisionProvider

        return OpenAIVisionProvider(model=model, api_key=key)
    if name == "mock":
        from .mock import MockVisionProvider

        return MockVisionProvider(model=model)
    raise ValueError(f"Unknown vision provider {name!r}. Choose one of {VISION_PROVIDERS}.")


def get_text_provider(config: WorkspaceConfig) -> TextProvider:
    name = config.provider.resolved_text_provider()
    model = config.provider.resolved_text_model()
    key = config.api_key(name)

    if name == "gemini":
        from .gemini import GeminiTextProvider

        return GeminiTextProvider(model=model, api_key=key)
    if name == "anthropic":
        from .anthropic import AnthropicTextProvider

        return AnthropicTextProvider(model=model, api_key=key)
    if name == "openai":
        from .openai import OpenAITextProvider

        return OpenAITextProvider(model=model, api_key=key)
    if name == "mock":
        from .mock import MockTextProvider

        return MockTextProvider(model=model)
    raise ValueError(f"Unknown text provider {name!r}. Choose one of {TEXT_PROVIDERS}.")


def check_credentials(config: WorkspaceConfig) -> None:
    """Fail early and clearly rather than at the first API call."""
    name = config.provider.vision
    if name == "mock":
        return
    if not config.api_key(name):
        from ...config import PROVIDER_KEY_ENV

        variables = " or ".join(PROVIDER_KEY_ENV.get(name, ()))
        raise MissingCredentialsError(
            f"No API key for provider {name!r}. Set {variables} in your environment "
            f"or in {config.dir / '.env'}."
        )
