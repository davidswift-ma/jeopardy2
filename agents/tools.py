"""Local tools for the ADK specialists.

ADK builds each tool's schema by reflecting on the function: type hints
become the parameter schema and **the docstring becomes the description the
model reads when deciding whether to call it**. So the docstrings here are
interface, not commentary -- vague wording degrades tool selection the same
way a vague `description` degrades routing.

Every tool returns a plain dict. ADK needs JSON-serializable returns, and a
dict with an `error` key is preferable to raising: the model can read the
message and recover, where an exception just ends the turn.
"""

from __future__ import annotations

import logging

from app.config import get_settings
from app.retrieval import open_store

logger = logging.getLogger(__name__)


def search_clues(query: str) -> dict:
    """Find Jeopardy clues semantically similar to a description or topic.

    Use this for fuzzy questions like "clues about Norse mythology" or
    "clues similar to this one", where the wording of the question will not
    literally appear in the clue. For counting, filtering by dollar value,
    category or date, use the SQL tools instead.
    """
    settings = get_settings()
    store = open_store(settings)
    if (reason := store.is_available()) is not None:
        # Reported rather than raised, so the router can still answer with a
        # caveat instead of the turn dying.
        return {"error": f"clue search is unavailable: {reason}", "results": []}

    hits = store.search(query, limit=5)
    return {
        "query": query,
        "result_count": len(hits),
        "results": [
            {
                "clue_text": h.get("clue_text"),
                "correct_response": h.get("correct_response"),
                "category": h.get("category"),
                "round": h.get("round"),
                "clue_value": h.get("clue_value"),
                "air_date": h.get("air_date"),
                "similarity_distance": h.get("distance"),
            }
            for h in hits
        ],
    }


def check_clue_index_status() -> dict:
    """Report whether the clue archive is available, and why not if it is not.

    Call this when a clue search returns nothing, so the user can be told
    what is actually missing rather than "no results found".
    """
    from app import retrieval  # noqa: PLC0415 - avoids importing chromadb at module load

    settings = get_settings()
    state = retrieval.inspect(settings) if settings.retrieval_enabled else None
    if state is None:
        return {"available": False, "reason": "RETRIEVAL_ENABLED is false"}
    return {
        "available": state.status == "ready",
        "status": state.status,
        "reason": state.reason,
        "chunk_scheme": settings.chunk_scheme,
    }
