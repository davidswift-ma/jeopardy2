#!/usr/bin/env python3
"""Build a local sample of the Jeopardy clue dataset from your own download.

The full dataset is ~80MB for the combined TSV (496MB for the whole download)
and its source asks that it not be used in a public-facing product, so this
repository carries **no** clue data at all -- not even a sample. The output
of this script is gitignored and stays on your machine.

Sampling is a deterministic stride (every Nth row), not a random shuffle, so
the result is reproducible and spreads evenly across all 42 seasons instead of
clustering in whichever years happen to get picked.

Usage:
    python scripts/make_sample.py [--source PATH] [--out PATH] [--rows N]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

DEFAULT_SOURCE = Path.home() / "jeopardy/jeopardy_clue_dataset/combined_season1-42.tsv"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data/jeopardy_sample.tsv"
DEFAULT_ROWS = 4000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    args = parser.parse_args()

    if not args.source.exists():
        print(f"error: source not found: {args.source}", file=sys.stderr)
        print(
            "Download it from https://github.com/jwolle1/jeopardy_clue_dataset/releases",
            file=sys.stderr,
        )
        return 1

    with args.source.open(newline="", encoding="utf-8") as fh:
        total = sum(1 for _ in fh) - 1  # minus the header
    stride = max(1, total // args.rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with (
        args.source.open(newline="", encoding="utf-8") as src,
        args.out.open("w", newline="", encoding="utf-8") as dst,
    ):
        reader = csv.DictReader(src, delimiter="\t")
        if reader.fieldnames is None:
            print("error: source has no header row", file=sys.stderr)
            return 1
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()
        for i, row in enumerate(reader):
            if i % stride == 0:
                writer.writerow(row)
                written += 1

    size_mb = args.out.stat().st_size / 1_048_576
    print(
        f"wrote {written:,} of {total:,} clues (every {stride}th) to {args.out} [{size_mb:.1f} MB]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
