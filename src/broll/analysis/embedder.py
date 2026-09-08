"""Text embeddings.

Local by default: search quality then does not depend on which vision provider
the client chose, costs nothing per clip, and re-embedding the whole library
after a prompt change is free.

The vector table's dimension is fixed when it is created, so swapping to a model
with a different dimension requires ``broll reembed``, which drops and rebuilds
it.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod

from ..config import EmbedderConfig, WorkspaceConfig

log = logging.getLogger(__name__)

PROVIDER_EMBEDDING_MODELS = {
    "gemini": ("gemini-embedding-001", 768),
    "openai": ("text-embedding-3-small", 1536),
}


class Embedder(ABC):
    name: str
    model: str
    dimensions: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def warm_up(self) -> None:
        """Load the model once, up front, before any worker thread needs it."""
        self.embed(["warm up"])

    @property
    def signature(self) -> str:
        """Stored alongside the vectors so a model change is detectable."""
        return f"{self.name}:{self.model}:{self.dimensions}"


class LocalEmbedder(Embedder):
    """sentence-transformers, all-MiniLM-L6-v2 by default (384 dimensions)."""

    name = "local"

    def __init__(self, config: EmbedderConfig):
        self.config = config
        self.model = config.model
        self.dimensions = config.dimensions
        self._encoder = None
        # The pipeline embeds from worker threads; loading torch weights
        # concurrently is not safe, and encoding concurrently is not faster.
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            return self._load_locked()

    def _load_locked(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Local embeddings need the extra: "
                    "pip install 'broll-librarian[embeddings-local]' (it pulls "
                    "torch, ~2GB), or set embedder.kind to gemini or openai."
                ) from exc
            self._encoder = SentenceTransformer(self.config.model)
            getter = getattr(self._encoder, "get_embedding_dimension", None) or (
                self._encoder.get_sentence_embedding_dimension
            )
            self.dimensions = getter()
        return self._encoder

    def embed(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            encoder = self._load_locked()
            vectors = encoder.encode(texts, normalize_embeddings=True)
        return [[float(x) for x in v] for v in vectors]


class GeminiEmbedder(Embedder):
    name = "gemini"

    def __init__(self, config: EmbedderConfig, api_key: str | None):
        model, dimensions = PROVIDER_EMBEDDING_MODELS["gemini"]
        self.model = config.model if config.model.startswith("gemini-") else model
        self.dimensions = config.dimensions if config.dimensions != 384 else dimensions
        self.api_key = api_key

    def embed(self, texts: list[str]) -> list[list[float]]:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        result = client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=self.dimensions),
        )
        return [list(e.values) for e in result.embeddings]


class OpenAIEmbedder(Embedder):
    name = "openai"

    def __init__(self, config: EmbedderConfig, api_key: str | None):
        model, dimensions = PROVIDER_EMBEDDING_MODELS["openai"]
        self.model = config.model if config.model.startswith("text-embedding") else model
        self.dimensions = config.dimensions if config.dimensions != 384 else dimensions
        self.api_key = api_key

    def embed(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.embeddings.create(
            model=self.model, input=texts, dimensions=self.dimensions
        )
        return [list(item.embedding) for item in response.data]


def local_embeddings_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def get_embedder(config: WorkspaceConfig) -> Embedder:
    """Resolve the configured embedder, falling back with a clear explanation."""
    kind = config.embedder.kind

    if kind == "local":
        if local_embeddings_available():
            return LocalEmbedder(config.embedder)
        fallback = config.provider.vision
        if fallback not in PROVIDER_EMBEDDING_MODELS:
            raise RuntimeError(
                "sentence-transformers is not installed, and the configured "
                f"provider {fallback!r} has no embeddings API to fall back to. "
                "Install 'broll-librarian[embeddings-local]', or set "
                "embedder.kind to gemini or openai and supply that key."
            )
        log.warning(
            "sentence-transformers is not installed - falling back to %s embeddings. "
            "Install 'broll-librarian[embeddings-local]' to keep embeddings local "
            "and free.",
            fallback,
        )
        kind = fallback

    if kind == "gemini":
        return GeminiEmbedder(config.embedder, config.api_key("gemini"))
    if kind == "openai":
        return OpenAIEmbedder(config.embedder, config.api_key("openai"))
    if kind == "anthropic":
        raise RuntimeError(
            "Anthropic has no embeddings API. Keep embedder.kind: local "
            "(install 'broll-librarian[embeddings-local]'), or set it to gemini "
            "or openai with that provider's key. The vision provider is "
            "unaffected - it stays Anthropic."
        )
    raise ValueError(f"Unknown embedder kind {kind!r}.")
