"""Index state machine: decide, build, open.

This is where the three states from `base.py` are actually resolved. Two
deliberate choices worth stating:

**Builds happen at startup or from `make index`, never inside a request.**
Two uvicorn workers lazily building on first request would duplicate the work
and write to Chroma's SQLite concurrently, which is not what it is for. A
build inside a request would also compete with `REQUEST_TIMEOUT_SECONDS` and
pollute the harness's trace timings.

**A stale index rebuilds rather than being silently reused.** See the note on
`IndexManifest`: an embedder swap at the same dimensionality produces
plausible-looking garbage with no error at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import Settings
from app.retrieval.base import ClueStore, IndexManifest, UnavailableStore, describe_source
from app.retrieval.chroma_store import ChromaClueStore
from app.retrieval.chunks import CHUNK_SCHEME_VERSION, read_chunks
from app.retrieval.embedders import BATCH_SIZE, build_embedder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexState:
    """What `resolve` decided and why, for logging and /jeopardy2/config."""

    status: str  # ready | needs_build | stale | no_dataset | unavailable
    reason: str
    changes: list[str] | None = None


def _expected_manifest(settings: Settings, embedder_id: str, dimensions: int) -> IndexManifest:
    size, mtime = describe_source(settings.dataset_path)
    return IndexManifest(
        source_path=str(settings.dataset_path),
        source_size=size,
        source_mtime_ns=mtime,
        row_count=0,  # filled in after a build; excluded from comparisons
        embedder_id=embedder_id,
        dimensions=dimensions,
        chunk_scheme=settings.chunk_scheme,
        chunk_scheme_version=CHUNK_SCHEME_VERSION,
    )


def inspect(settings: Settings) -> IndexState:
    """Decide what state the index is in, without building anything."""
    directory = settings.index_path
    existing = IndexManifest.read(directory)

    if not settings.dataset_path.exists():
        if existing is not None:
            # An index built earlier still works even if the source TSV has
            # since moved; there is nothing to rebuild *from*, but nothing
            # broken either.
            return IndexState("ready", f"index present; source {settings.dataset_path} absent")
        return IndexState(
            "no_dataset",
            f"no clue data at {settings.dataset_path}. None ships with this repo: "
            f"download it and run `make sample`, or set DATASET_PATH. "
            f"See data/README.md.",
        )

    embedder = build_embedder(settings)
    if (reason := embedder.is_available()) is not None:
        return IndexState("unavailable", reason)

    expected = _expected_manifest(settings, embedder.id, embedder.dimensions)
    if existing is None:
        return IndexState("needs_build", f"no index at {directory}")

    changes = expected.differences(existing)
    if changes:
        return IndexState("stale", "index inputs changed", changes)
    return IndexState("ready", f"{existing.row_count} chunks")


def build(settings: Settings, *, force: bool = False, limit: int | None = None) -> IndexState:
    """Build or rebuild the index. Safe to call when it is already current."""
    state = inspect(settings)
    if state.status in {"no_dataset", "unavailable"}:
        logger.warning("cannot build index: %s", state.reason)
        return state
    if state.status == "ready" and not force:
        logger.info("index already current (%s)", state.reason)
        return state
    if state.changes:
        logger.info("rebuilding index; inputs changed: %s", "; ".join(state.changes))

    embedder = build_embedder(settings)
    store = ChromaClueStore(settings.index_path, settings.collection_name, embedder)
    # Drop first: upserting into a collection built with a different scheme
    # would leave orphaned vectors from rows that no longer render the same.
    store.reset()

    total = 0
    batch_ids: list[str] = []
    batch_texts: list[str] = []
    batch_meta: list[dict[str, object]] = []

    for chunk in read_chunks(settings.dataset_path, scheme=settings.chunk_scheme, limit=limit):
        batch_ids.append(chunk.id)
        batch_texts.append(chunk.text)
        batch_meta.append(chunk.metadata)
        if len(batch_ids) >= BATCH_SIZE:
            store.add(batch_ids, batch_texts, batch_meta)
            total += len(batch_ids)
            logger.info("indexed %d chunks", total)
            batch_ids, batch_texts, batch_meta = [], [], []

    if batch_ids:
        store.add(batch_ids, batch_texts, batch_meta)
        total += len(batch_ids)

    manifest = _expected_manifest(settings, embedder.id, embedder.dimensions)
    manifest = IndexManifest(**{**manifest.__dict__, "row_count": total})
    manifest.write(settings.index_path)
    logger.info("index built: %d chunks at %s", total, settings.index_path)
    return IndexState("ready", f"{total} chunks")


def open_store(settings: Settings) -> ClueStore:
    """Open the index for querying, or an `UnavailableStore` explaining why not.

    Never builds. Startup decides whether to build; a query only ever reads.
    """
    if not settings.retrieval_enabled:
        return UnavailableStore("RETRIEVAL_ENABLED is false")

    state = inspect(settings)
    if state.status in {"no_dataset", "unavailable"}:
        return UnavailableStore(state.reason)
    if state.status == "needs_build":
        return UnavailableStore(f"{state.reason}; run `make index`")
    if state.status == "stale":
        return UnavailableStore(
            f"index is stale ({'; '.join(state.changes or [])}); run `make index`"
        )

    embedder = build_embedder(settings)
    store = ChromaClueStore(settings.index_path, settings.collection_name, embedder)
    if (reason := store.is_available()) is not None:
        return UnavailableStore(reason)
    return store
