#!/usr/bin/env python3
"""Build the clue vector index.

Reads the TSV at `DATASET_PATH`, renders one chunk per clue, embeds them, and
writes a Chroma store to `INDEX_PATH` with a manifest recording exactly what
it was built from.

No clue data ships with this repository, so a fresh clone has nothing to
index. That is a supported state: this script says so and exits 0, rather
than failing a build someone did not ask for.

Usage:
    python scripts/build_index.py                  # build if needed
    python scripts/build_index.py --force          # rebuild regardless
    python scripts/build_index.py --status         # report, change nothing
    python scripts/build_index.py --limit 500      # a cheap partial index
    EMBEDDING_PROVIDER=hash python scripts/build_index.py   # offline, free

Costs real money with `EMBEDDING_PROVIDER=openai`: roughly 28 tokens per
clue, so ~112k tokens for the 4,001-row sample and ~15.4M for the full
544,111-row dataset.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.retrieval import build, inspect, open_store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild even if current.")
    parser.add_argument("--status", action="store_true", help="Report state and exit.")
    parser.add_argument("--limit", type=int, default=None, help="Index only the first N clues.")
    parser.add_argument("--query", default=None, help="Run one search against the built index.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    settings = Settings()

    print(f"dataset:   {settings.dataset_path}")
    print(f"index:     {settings.index_path}")
    print(f"scheme:    {settings.chunk_scheme}")
    print(f"embedder:  {settings.embedding_provider} ({settings.embedding_dimensions}d)\n")

    state = inspect(settings)
    print(f"state:     {state.status} -- {state.reason}")
    for change in state.changes or []:
        print(f"  changed: {change}")

    if args.status:
        return 0

    if state.status == "no_dataset":
        # Not an error. The repo ships without clue data on purpose.
        print("\nNothing to index. The app runs fine without it.")
        return 0

    result = build(settings, force=args.force, limit=args.limit)
    print(f"\nresult:    {result.status} -- {result.reason}")
    if result.status != "ready":
        return 1

    if args.query:
        store = open_store(settings)
        if (reason := store.is_available()) is not None:
            print(f"\ncannot query: {reason}")
            return 1
        print(f"\ntop matches for {args.query!r}:\n")
        for hit in store.search(args.query, limit=5):
            distance = hit.get("distance")
            shown = f"{distance:.4f}" if isinstance(distance, float) else "n/a"
            print(f"  [{shown}] {hit.get('category')}: {hit.get('clue_text')}")
            print(f"           -> {hit.get('correct_response')}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
