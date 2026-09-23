#!/usr/bin/env python3
"""An MCP server exposing the clue dataset over SQL.

Run standalone (it speaks MCP over stdio, so it is normally launched as a
subprocess by the agent, not by you):

    python -m agents.clue_mcp_server

Why write one instead of using the Supabase MCP server from the course demo:
the clue data is local, gitignored, and must not be uploaded to a hosted
database -- the source asks that it not be redistributed in a public-facing
product. A local MCP server keeps the data on disk and still demonstrates the
thing MCP is for: the agent discovers these tools at runtime and writes its
own queries, rather than calling Python functions we imported.

Safety: `query_clues` takes model-authored SQL, so it runs read-only. The
connection is opened in SQLite's immutable mode and statements are rejected
unless they are a single SELECT. A model that decides to DROP TABLE gets an
error string back, not a dropped table.
"""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from app.config import Settings  # noqa: E402

server = MCPServer(
    name="jeopardy-clues",
    instructions=(
        "SQL access to a Jeopardy clue archive. Table `clues` has columns: "
        "id, clue_text (what was read to contestants), correct_response "
        "(what the contestant had to say), category, round, clue_value, "
        "air_date (YYYY-MM-DD). Use query_clues for aggregate and filtered "
        "questions."
    ),
)

#: Only a single SELECT (or a CTE leading to one) is allowed through.
_ALLOWED_START = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|replace)\b",
    re.IGNORECASE,
)
MAX_ROWS = 50


def _settings() -> Settings:
    return Settings()


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open read-only. `immutable=1` also prevents the WAL/journal writes that
    a plain file:...?mode=ro connection can still make."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@server.tool(
    name="query_clues",
    description=(
        "Run a read-only SQL SELECT against the `clues` table and return rows. "
        "Columns: id, clue_text, correct_response, category, round, "
        "clue_value, air_date. Only SELECT/WITH is permitted. At most "
        f"{MAX_ROWS} rows are returned."
    ),
)
def query_clues(sql: str) -> dict[str, Any]:
    """Execute a read-only SELECT against the clue archive."""
    settings = _settings()
    if not settings.clues_db_path.exists():
        return {
            "error": (
                f"no clue database at {settings.clues_db_path}. "
                f"Run `make clues-db` (needs the dataset; see data/README.md)."
            )
        }
    if not _ALLOWED_START.match(sql):
        return {"error": "only SELECT or WITH statements are allowed"}
    if _FORBIDDEN.search(sql):
        return {"error": "statement contains a forbidden keyword; this tool is read-only"}
    if ";" in sql.rstrip().rstrip(";"):
        return {"error": "only a single statement is allowed"}

    try:
        conn = _connect(settings.clues_db_path)
    except sqlite3.Error as exc:
        return {"error": f"could not open the clue database: {exc}"}
    try:
        cursor = conn.execute(sql)
        rows = [dict(r) for r in cursor.fetchmany(MAX_ROWS)]
        truncated = cursor.fetchone() is not None
        return {"rows": rows, "row_count": len(rows), "truncated": truncated}
    except sqlite3.Error as exc:
        # Handed back as data, not raised: the model can read the message and
        # correct its own SQL, which is the whole point of giving it SQL.
        return {"error": f"SQL error: {exc}"}
    finally:
        conn.close()


@server.tool(
    name="describe_clues",
    description="Report the clue table's schema, row count, and date range.",
)
def describe_clues() -> dict[str, Any]:
    """Schema and summary statistics for the clue archive."""
    settings = _settings()
    if not settings.clues_db_path.exists():
        return {"error": f"no clue database at {settings.clues_db_path}. Run `make clues-db`."}
    conn = _connect(settings.clues_db_path)
    try:
        columns = [dict(r) for r in conn.execute("PRAGMA table_info(clues)")]
        stats = conn.execute(
            "SELECT COUNT(*) AS clues, COUNT(DISTINCT category) AS categories, "
            "MIN(air_date) AS first_aired, MAX(air_date) AS last_aired FROM clues"
        ).fetchone()
        return {
            "table": "clues",
            "columns": [{"name": c["name"], "type": c["type"]} for c in columns],
            **dict(stats),
            "note": (
                "clue_text is what was read to contestants; correct_response "
                "is what they had to answer. The source dataset names these "
                "backwards; they are corrected here."
            ),
        }
    finally:
        conn.close()


if __name__ == "__main__":
    server.run(transport="stdio")
