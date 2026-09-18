"""Clue retrieval: chunking, embedding, and the vector index.

Plumbing only at this stage. Nothing here is wired into the prompt yet --
`open_store` gives you a searchable index and `/jeopardy2/config` reports its
state, but the agent still answers from general knowledge.

Optional throughout. With `RETRIEVAL_ENABLED=false` (the default), no clue
data present, or chromadb not installed, every entry point returns an
`UnavailableStore` that says which of those it was.
"""

from app.retrieval.base import ClueStore, Embedder, IndexManifest, UnavailableStore
from app.retrieval.chunks import CHUNK_SCHEMES, Chunk, read_chunks, render
from app.retrieval.index import IndexState, build, inspect, open_store

__all__ = [
    "CHUNK_SCHEMES",
    "Chunk",
    "ClueStore",
    "Embedder",
    "IndexManifest",
    "IndexState",
    "UnavailableStore",
    "build",
    "inspect",
    "open_store",
    "read_chunks",
    "render",
]
