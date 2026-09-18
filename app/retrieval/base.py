"""Retrieval protocols and the index manifest.

Same shape as `app/engines` and `app/obs`: a protocol, a null implementation,
and `is_available() -> str | None` so an unconfigured or unbuildable index is
something the service reports rather than crashes on.

The three states this package has to handle, in the order you hit them:

1. index present and its fingerprint matches  -> load, near-instant
2. index absent, dataset present              -> build
3. index absent, dataset absent               -> unavailable, with the reason

State 3 is the default for a fresh clone, because `data/` ships empty on
purpose (the source dataset asks not to be redistributed in a public-facing
product). The professor's first run is state 3, and the app has to stay up
and answer general questions anyway -- exactly as it does today with no API
keys configured.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "index_manifest.json"


class Embedder(Protocol):
    """Turns text into vectors. The only thing the store needs from a model."""

    #: Stable identifier recorded in the manifest, e.g. "openai:text-embedding-3-small".
    id: str
    #: Vector width. A mismatch against an existing index is a hard rebuild.
    dimensions: int

    def is_available(self) -> str | None: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class IndexManifest:
    """What an index was built from, written next to it.

    Staleness has two failure modes and only one is loud. Changing embedding
    *dimensions* raises at query time and you fix it in a minute. Changing to
    a different model with the *same* dimensions returns plausible-looking
    garbage with no error anywhere -- that is the one that costs an evening,
    and it is why `embedder_id` is compared and not just `dimensions`.
    """

    source_path: str
    source_size: int
    source_mtime_ns: int
    row_count: int
    embedder_id: str
    dimensions: int
    chunk_scheme: str
    chunk_scheme_version: int

    def write(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / MANIFEST_FILENAME).write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def read(cls, directory: Path) -> IndexManifest | None:
        path = directory / MANIFEST_FILENAME
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(**data)
        except (json.JSONDecodeError, TypeError) as exc:
            # A manifest we cannot read means we cannot trust the index.
            logger.warning("unreadable index manifest at %s (%s); will rebuild", path, exc)
            return None

    def differences(self, other: IndexManifest) -> list[str]:
        """Field-by-field mismatches, named, for a log line worth reading."""
        return [
            f"{name}: {getattr(other, name)!r} -> {getattr(self, name)!r}"
            for name in asdict(self)
            # row_count is an output of building, not an input to it: comparing
            # it would make every index permanently stale against itself.
            if name != "row_count" and getattr(self, name) != getattr(other, name)
        ]


def describe_source(path: Path) -> tuple[int, int]:
    """(size, mtime_ns) for the dataset, the cheap half of the fingerprint.

    Deliberately not a content hash: the full TSV is ~80MB and hashing it on
    every startup would cost more than it saves. Size plus mtime catches the
    realistic cases -- a regenerated sample, a swap from sample to full
    dataset -- and `make index --force` covers the rest.
    """
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


class ClueStore(Protocol):
    """A built, queryable index of clue chunks."""

    def is_available(self) -> str | None: ...

    def count(self) -> int: ...

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]: ...


class UnavailableStore:
    """Stands in when there is no index, carrying the reason why.

    Returns empty results rather than raising: a missing index must degrade
    the answer, not fail the request.
    """

    def __init__(self, reason: str) -> None:
        self._reason = reason

    def is_available(self) -> str | None:
        return self._reason

    def count(self) -> int:
        return 0

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        return []
