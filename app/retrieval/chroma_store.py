"""ChromaDB adapter -- the only module in this repo that imports chromadb.

Embedded persistent mode: a directory on disk, no server, no extra container.
That directory is gitignored and must stay that way. Chroma stores the
document text alongside each vector, so committing it would commit the clue
text verbatim and undo `db7cbf1 "Ship without clue data for public release"`
in a binary blob nobody would spot in review.

Embeddings are computed by our own `Embedder` and passed in explicitly rather
than letting Chroma pick a default embedding function. Two reasons: the
manifest has to know exactly which model produced the vectors, and Chroma's
default would silently download an ONNX model on first use.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.retrieval.base import Embedder

logger = logging.getLogger(__name__)

#: Chroma rejects metadata values that are not str/int/float/bool/None.
_SCALARS = (str, int, float, bool)


def _clean_metadata(metadata: dict[str, object]) -> dict[str, Any]:
    return {k: v for k, v in metadata.items() if isinstance(v, _SCALARS)}


class ChromaClueStore:
    """A persistent Chroma collection of clue chunks."""

    def __init__(self, directory: Path, collection: str, embedder: Embedder) -> None:
        self._directory = directory
        self._collection_name = collection
        self._embedder = embedder
        self._collection: Any | None = None
        self._broken: str | None = None

    # -- lifecycle ---------------------------------------------------------
    def _connect(self, *, create: bool) -> Any | None:
        if self._collection is not None or self._broken:
            return self._collection
        try:
            import chromadb  # noqa: PLC0415 - optional dependency
        except ImportError:
            self._broken = "chromadb is not installed (pip install 'jeopardy2[rag]')"
            return None

        try:
            client = chromadb.PersistentClient(path=str(self._directory))
            if create:
                self._collection = client.get_or_create_collection(
                    name=self._collection_name,
                    # Cosine rather than Chroma's default L2: our embedders
                    # return normalized vectors, and cosine is what the
                    # providers' similarity guidance assumes.
                    metadata={"hnsw:space": "cosine"},
                )
            else:
                self._collection = client.get_collection(name=self._collection_name)
        except Exception as exc:  # noqa: BLE001 - absent collection is normal
            self._broken = f"{type(exc).__name__}: {exc}"
            return None
        return self._collection

    def is_available(self) -> str | None:
        if self._connect(create=False) is None:
            return self._broken or "index not built"
        return None

    # -- writing -----------------------------------------------------------
    def add(self, ids: list[str], texts: list[str], metadatas: list[dict[str, object]]) -> None:
        collection = self._connect(create=True)
        if collection is None:
            raise RuntimeError(self._broken or "could not open the Chroma collection")
        vectors = self._embedder.embed(texts)
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=vectors,
            metadatas=[_clean_metadata(m) for m in metadatas],
        )

    def reset(self) -> None:
        """Drop the collection so a rebuild does not merge with stale vectors."""
        try:
            import chromadb  # noqa: PLC0415
        except ImportError:
            return
        try:
            client = chromadb.PersistentClient(path=str(self._directory))
            client.delete_collection(name=self._collection_name)
        except Exception:  # noqa: BLE001 - not existing yet is the common case
            logger.debug("no existing collection %r to drop", self._collection_name)
        self._collection = None
        self._broken = None

    # -- reading -----------------------------------------------------------
    def count(self) -> int:
        collection = self._connect(create=False)
        return 0 if collection is None else int(collection.count())

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        collection = self._connect(create=False)
        if collection is None:
            return []
        vector = self._embedder.embed([query])[0]
        result = collection.query(query_embeddings=[vector], n_results=limit)

        # Chroma returns parallel lists-of-lists, one inner list per query.
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]
        return [
            {
                "id": ids[i] if i < len(ids) else None,
                "text": documents[i] if i < len(documents) else "",
                "distance": distances[i] if i < len(distances) else None,
                **(metadatas[i] or {} if i < len(metadatas) else {}),
            }
            for i in range(len(documents))
        ]
