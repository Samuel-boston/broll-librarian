"""Text embeddings.

Local by default: search quality then does not depend on which vision provider
the client chose, costs nothing per clip, and re-embedding the whole library
after a prompt change is free.

The vector table's dimension is fixed when it is created, so swapping to a
model with a different dimension requires ``broll reembed``, which drops and
rebuilds it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import EmbedderConfig, WorkspaceConfig


class Embedder(ABC):
    name: str
    dimensions: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class LocalEmbedder(Embedder):
    """sentence-transformers, all-MiniLM-L6-v2 by default (384 dimensions)."""

    name = "local"

    def __init__(self, config: EmbedderConfig):
        self.config = config
        self.dimensions = config.dimensions
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Local embeddings need the extra: "
                    "pip install 'broll-librarian[embeddings-local]' "
                    "(it pulls torch, ~2GB). Alternatively set embedder.kind: provider."
                ) from exc
            self._model = SentenceTransformer(self.config.model)
            self.dimensions = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]


def get_embedder(config: WorkspaceConfig) -> Embedder:
    if config.embedder.kind == "local":
        return LocalEmbedder(config.embedder)
    raise ValueError(
        f"Unknown embedder kind {config.embedder.kind!r}. "
        "Only 'local' is implemented; provider embeddings arrive with M2."
    )


def local_embeddings_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True
