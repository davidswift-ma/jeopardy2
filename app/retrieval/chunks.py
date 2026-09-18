"""Turning dataset rows into indexable chunks.

**The column trap.** In this dataset the column names are inverted from
intuition: `answer` holds the clue read to contestants, and `question` holds
the correct response. The very first row is `answer = "River mentioned most
often in the Bible"`, `question = "the Jordan"`.

That inversion is renamed away *here*, at the file boundary, and never
propagates further. Everything downstream sees `clue_text` and
`correct_response`. Passing `answer`/`question` into a prompt and hoping the
model works it out is exactly the bug this naming prevents.

Chunk schemes are versioned for the same reason prompts are: changing how a
row becomes text invalidates every embedding built under the old scheme, and
the manifest needs to be able to notice.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

#: Dataset column names, kept in one place so the inversion is stated once.
_COL_CLUE = "answer"  # yes, really: this column holds the clue
_COL_RESPONSE = "question"  # and this one holds the response

#: Round numbers as they appear in the data.
_ROUND_NAMES = {"1": "Jeopardy", "2": "Double Jeopardy", "3": "Final Jeopardy"}

#: Bump when the text produced by `render` changes in any scheme. The manifest
#: compares this, so a stale index rebuilds instead of silently mixing schemes.
CHUNK_SCHEME_VERSION = 1

CHUNK_SCHEMES = ("qa-glued", "clue-only")


@dataclass(frozen=True)
class Chunk:
    """One indexable unit. For this dataset, one clue.

    A Jeopardy clue averages 28 tokens and the longest in a 4,001-row sample
    is 93, so there is nothing to split: the row *is* the chunk. Splitting
    would only risk severing a clue from its response.
    """

    id: str
    text: str
    clue_text: str
    correct_response: str
    category: str
    round: str
    clue_value: int | None
    air_date: str
    metadata: dict[str, object] = field(default_factory=dict)


def _stable_id(clue_text: str, correct_response: str, air_date: str) -> str:
    """Content-derived ID, so re-indexing the same row overwrites rather than duplicates.

    Row position would not survive regenerating a sample with a different
    stride, which is a thing `make sample` does routinely.
    """
    payload = f"{air_date}|{clue_text}|{correct_response}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def render(
    *,
    clue_text: str,
    correct_response: str,
    category: str,
    scheme: str,
) -> str:
    """Produce the text that actually gets embedded.

    `qa-glued` follows the brief of gluing each clue/response pair into one
    chunk. Be aware of what that does to evaluation: the correct response is
    *inside* the embedded text, so retrieving with clue text as the query
    scores well partly because the answer was indexed. `clue-only` is the
    honest control for measuring retrieval quality; the response is still on
    the chunk as metadata either way.
    """
    if scheme == "qa-glued":
        return f"Category: {category}\nClue: {clue_text}\nResponse: {correct_response}"
    if scheme == "clue-only":
        return f"Category: {category}\nClue: {clue_text}"
    raise ValueError(f"unknown chunk scheme {scheme!r}; known: {list(CHUNK_SCHEMES)}")


def _parse_value(raw: str) -> int | None:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def read_chunks(
    path: Path, *, scheme: str = "qa-glued", limit: int | None = None
) -> Iterator[Chunk]:
    """Stream chunks from the clue TSV.

    A generator rather than a list: the full dataset is 544,111 rows, and
    materializing all of them before embedding would hold the entire corpus
    in memory for no reason.
    """
    if scheme not in CHUNK_SCHEMES:
        raise ValueError(f"unknown chunk scheme {scheme!r}; known: {list(CHUNK_SCHEMES)}")

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        missing = {_COL_CLUE, _COL_RESPONSE} - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{path} is missing column(s) {sorted(missing)}; "
                f"expected the jwolle1 clue TSV, found {reader.fieldnames}"
            )

        emitted = 0
        for row in reader:
            clue_text = (row.get(_COL_CLUE) or "").strip()
            correct_response = (row.get(_COL_RESPONSE) or "").strip()
            # A row missing either half cannot be retrieved usefully and would
            # embed to near-noise. Skip rather than index garbage.
            if not clue_text or not correct_response:
                continue

            category = (row.get("category") or "").strip()
            round_raw = (row.get("round") or "").strip()
            air_date = (row.get("air_date") or "").strip()
            yield Chunk(
                id=_stable_id(clue_text, correct_response, air_date),
                text=render(
                    clue_text=clue_text,
                    correct_response=correct_response,
                    category=category,
                    scheme=scheme,
                ),
                clue_text=clue_text,
                correct_response=correct_response,
                category=category,
                round=_ROUND_NAMES.get(round_raw, round_raw),
                clue_value=_parse_value(row.get("clue_value") or ""),
                air_date=air_date,
                metadata={
                    # Flat and primitive-valued: vector stores generally
                    # reject nested metadata, and these are the columns worth
                    # filtering on (a SQL-shaped question, not a semantic one).
                    "clue_text": clue_text,
                    "correct_response": correct_response,
                    "category": category,
                    "round": _ROUND_NAMES.get(round_raw, round_raw),
                    "clue_value": _parse_value(row.get("clue_value") or "") or 0,
                    "air_date": air_date,
                },
            )
            emitted += 1
            if limit is not None and emitted >= limit:
                return
