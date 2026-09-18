"""Embedding backends.

Pluggable for the same reason the engines are: the choice has cost and
dependency consequences that should be configuration, not architecture.

| `EMBEDDING_PROVIDER` | Cost | Dependency | Use for |
|---|---|---|---|
| `openai` | ~15.4M tokens for the full 544k dataset | already installed | real runs |
| `hash`   | free, offline, deterministic | none | tests, smoke runs, CI |

`hash` is not a real embedding model and will not retrieve sensibly. It
exists so the whole index pipeline -- chunking, manifest, build, count,
search -- can be exercised with no API key, no network, and no model
download, which is what keeps the test suite honest and fast.

A local sentence-transformers backend would slot in here as a third option.
It is not implemented because it pulls torch into the image, and this repo's
`docker compose up` currently needs nothing but Docker.
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct

from app.config import Settings

logger = logging.getLogger(__name__)

#: Batch size for the embedding API. Large enough that 544k rows is a few
#: thousand requests rather than half a million, small enough to stay well
#: inside per-request token limits at ~28 tokens per chunk.
BATCH_SIZE = 256


class HashEmbedder:
    """Deterministic pseudo-embeddings from a hash. Offline and free.

    Produces stable, normalized vectors so similarity search runs and returns
    *something*, but the geometry is meaningless -- similar clues are not
    near each other. Never use this for a real index.
    """

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions
        self.id = f"hash:{dimensions}"

    def is_available(self) -> str | None:
        return None

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        # Expand a digest to the requested width, then L2-normalize so cosine
        # distance behaves the way the store expects.
        raw = b""
        counter = 0
        while len(raw) < self.dimensions * 4:
            raw += hashlib.sha256(f"{counter}:{text}".encode()).digest()
            counter += 1
        values = [
            struct.unpack("<I", raw[i * 4 : i * 4 + 4])[0] / 2**32 - 0.5
            for i in range(self.dimensions)
        ]
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]


class OpenAIEmbedder:
    """Embeddings via the OpenAI API, reusing the key the engines already use."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        self.id = f"openai:{self.model}:{self.dimensions}"
        self._client: object | None = None

    def is_available(self) -> str | None:
        if self._settings.openai_api_key is None:
            return "OPENAI_API_KEY is not set (needed to build the index)"
        return None

    def _get_client(self):  # noqa: ANN202 - openai's client type
        if self._client is None:
            import openai  # noqa: PLC0415 - already a hard dependency

            key = self._settings.openai_api_key
            self._client = openai.OpenAI(
                api_key=key.get_secret_value() if key else None,
                timeout=self._settings.request_timeout_seconds,
                # Indexing is a long batch job; unlike the request path, the
                # SDK retrying here is a feature rather than a conflict with
                # the harness's own schedule.
                max_retries=3,
            )
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        response = client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
        )
        # Sort by index: the API documents order preservation, but relying on
        # it silently would misalign every vector with its chunk if it ever
        # changed, and that failure is invisible.
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]


def build_embedder(settings: Settings) -> HashEmbedder | OpenAIEmbedder:
    provider = settings.embedding_provider
    if provider == "hash":
        return HashEmbedder(settings.embedding_dimensions)
    if provider == "openai":
        return OpenAIEmbedder(settings)
    raise ValueError(f"unknown embedding provider {provider!r}; known: ['openai', 'hash']")
