#!/usr/bin/env python3
"""Export the clue TSV to SQLite, for the MCP server to query.

Why SQLite and not the vector index: these clues are *structured* -- round,
category, dollar value, air date. "Which categories came up most in the 90s?"
and "all $2000 opera clues" are SQL questions, and answering them by
embedding similarity would be the wrong tool badly applied. The vector index
handles "clues like this one"; this handles everything with a WHERE clause.

The output is gitignored. It is the clue data in another container, and the
same redistribution reasoning applies as for data/chroma/.

Usage:
    python scripts/export_clues_db.py
    python scripts/export_clues_db.py --limit 1000
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.retrieval.chunks import read_chunks  # noqa: E402

SCHEMA = """
CREATE TABLE IF NOT EXISTS clues (
    id                TEXT PRIMARY KEY,
    clue_text         TEXT NOT NULL,
    correct_response  TEXT NOT NULL,
    category          TEXT,
    round             TEXT,
    clue_value        INTEGER,
    air_date          TEXT
);
CREATE INDEX IF NOT EXISTS idx_clues_category ON clues(category);
CREATE INDEX IF NOT EXISTS idx_clues_value    ON clues(clue_value);
CREATE INDEX IF NOT EXISTS idx_clues_airdate  ON clues(air_date);
"""


def export(settings: Settings, limit: int | None = None) -> int:
    if not settings.dataset_path.exists():
        print(
            f"no clue data at {settings.dataset_path}. None ships with this repo: "
            f"download it and run `make sample`. See data/README.md.",
            file=sys.stderr,
        )
        return -1

    settings.clues_db_path.parent.mkdir(parents=True, exist_ok=True)
    # Rebuild from scratch: a partial export merged into a previous one would
    # silently answer "how many clues are there" wrongly.
    settings.clues_db_path.unlink(missing_ok=True)

    conn = sqlite3.connect(settings.clues_db_path)
    try:
        conn.executescript(SCHEMA)
        rows = 0
        batch = []
        for chunk in read_chunks(settings.dataset_path, limit=limit):
            batch.append(
                (
                    chunk.id,
                    chunk.clue_text,
                    chunk.correct_response,
                    chunk.category,
                    chunk.round,
                    chunk.clue_value,
                    chunk.air_date,
                )
            )
            if len(batch) >= 1000:
                conn.executemany("INSERT OR REPLACE INTO clues VALUES (?,?,?,?,?,?,?)", batch)
                rows += len(batch)
                batch = []
        if batch:
            conn.executemany("INSERT OR REPLACE INTO clues VALUES (?,?,?,?,?,?,?)", batch)
            rows += len(batch)
        conn.commit()
        return rows
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    settings = Settings()
    rows = export(settings, args.limit)
    if rows < 0:
        # Not an error: a fresh clone has no clue data by design.
        print("Nothing to export.")
        return 0
    size_mb = settings.clues_db_path.stat().st_size / 1_048_576
    print(f"wrote {rows:,} clues to {settings.clues_db_path} [{size_mb:.1f} MB]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
