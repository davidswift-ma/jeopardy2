"""Output-quality detectors.

Lifted out of `scripts/probe_answer_quality.py` so the offline eval and the
live request path score answers with *the same* function. Two detectors that
drift apart is how you end up with a probe reporting 0% while production
quietly corrupts output.

History worth keeping: the first version of this detector counted only line
breaks and reported 30%, under-reporting the real 87% by roughly 3x, because
line breaks were one artifact form out of five. Inspect the bytes (`repr()`),
not how the rendered text looks.

Known limitation, inherited and still true: `escape_artifacts` flags *any*
non-ASCII character. Genuine Claude corruption and OpenAI's perfectly good
curly apostrophes both land in the same bucket. The counts alone cannot tell
them apart -- read the text.
"""

from __future__ import annotations

import re
from collections import Counter

from app.schemas import Answer

#: Phrases a model uses when it notices its own output is broken. Observed in
#: practice: one run flagged its own garbled answer in `caveats` and supplied
#: a clean rewrite, with self-reported confidence dropping to 0.6.
_SELF_FLAG_PHRASES = ("formatting error", "cleaner version", "garbled", "repetition")


def repeated_ngrams(text: str, n: int = 4) -> int:
    """How many n-grams appear more than once."""
    words = text.lower().split()
    if len(words) < n:
        return 0
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return sum(count - 1 for count in grams.values() if count > 1)


def escape_artifacts(text: str) -> list[str]:
    """Detect mis-escaped output, the actual failure mode observed.

    Opus 5 mis-escapes non-ASCII punctuation inside structured-output JSON.
    One run produced five distinct corruptions of a single em dash, so
    counting newlines alone under-reported the rate by roughly 3x.
    """
    found = []
    if "\\u" in text:
        # A literal backslash-u sequence: the model double-escaped.
        found.append("literal-\\u-escape")
    if "\n" in text or "\r" in text:
        found.append("line-break")
    if "\\n" in text or "\\t" in text:
        found.append("literal-\\n-text")
    # A quote or dash-word stranded mid-sentence, e.g. 'examples " like text'
    # or the observed '\ndash like text'.
    if re.search(r'\w\s+["\']\s+\w', text):
        found.append("stray-quote-midsentence")
    if re.search(r"\bdash\b", text) and "-" not in text:
        found.append("the-word-dash")
    # Any non-ASCII char at all: we now ask for ASCII only, so this is a
    # violation even when it renders correctly (an em dash that survives is
    # still one that could have been mangled).
    non_ascii = sorted({c for c in text if ord(c) > 127})
    if non_ascii:
        found.append("non-ascii:" + "".join(non_ascii))
    return found


def diagnose(answer: Answer) -> dict[str, object]:
    text = answer.answer
    self_flagged = any(phrase in c.lower() for c in answer.caveats for phrase in _SELF_FLAG_PHRASES)
    artifacts = escape_artifacts(text)
    return {
        "artifacts": artifacts,
        "newlines": text.count("\n"),
        "repeated_4grams": repeated_ngrams(text),
        "chars": len(text),
        "confidence": answer.confidence,
        "self_flagged": self_flagged,
        "text": text,
    }


def is_degenerate(d: dict[str, object]) -> bool:
    """Any escape artifact, repetition, or self-flagged failure counts."""
    return bool(d["artifacts"] or int(d["repeated_4grams"]) > 0 or d["self_flagged"])
