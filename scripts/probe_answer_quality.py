#!/usr/bin/env python3
"""Measure how often a provider corrupts the `answer` field.

Background: claude-opus-5 mis-escapes non-ASCII punctuation inside
structured-output JSON. A single em dash was observed surfacing five
different ways -- as the literal text "\\u2014", a line break, the word
"dash", a stray quote, or (once in eight) correctly. Measured rate was 7/8
corrupted before the prompt forbade non-ASCII punctuation, and 0/12 after.

The point of this script is that the fix was *measured*, not guessed. Run it
before and after any prompt change that touches formatting.

Lesson from its own history: the first version of the detector counted only
line breaks and reported 30%, under-reporting the real 87% threefold,
because line breaks were one artifact form out of five. Inspect bytes
(`repr()`), not how the rendered text looks.

Usage:
    python scripts/probe_answer_quality.py --trials 10
    python scripts/probe_answer_quality.py --trials 10 --provider anthropic
    python scripts/probe_answer_quality.py --trials 10 --variant tightened

Costs real money: one API call per trial.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.engines import build_engine  # noqa: E402
from app.engines.base import EngineError  # noqa: E402
from app.schemas import Answer  # noqa: E402

QUESTION = "Explain AI in one or two sentences that my grandfather could understand."


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
    One run produced FIVE distinct corruptions of a single em dash, so
    counting newlines alone (as an earlier version of this script did)
    under-reported the rate by roughly 3x.
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
    self_flagged = any(
        phrase in c.lower()
        for c in answer.caveats
        for phrase in ("formatting error", "cleaner version", "garbled", "repetition")
    )
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


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument(
        "--variant",
        default="baseline",
        choices=["baseline", "tightened"],
        help="'tightened' adds explicit length/format guidance to the request.",
    )
    args = parser.parse_args()

    settings = Settings()
    engine = build_engine(args.provider, settings)
    if (reason := engine.is_available()) is not None:
        print(f"error: {reason}", file=sys.stderr)
        return 1

    question = args.question
    if args.variant == "tightened":
        question = (
            f"{args.question}\n\n"
            "Format: one or two complete sentences of plain prose. "
            "No line breaks, no bullet points, and do not offer alternative "
            "phrasings -- commit to a single wording."
        )

    print(
        f"provider={args.provider} model={engine.model} "
        f"variant={args.variant} trials={args.trials}\n"
    )

    results: list[dict[str, object]] = []
    errors = 0
    for i in range(1, args.trials + 1):
        try:
            answer = await engine.answer(question)
        except EngineError as exc:
            errors += 1
            print(f"  {i:2}. ERROR {exc.error_type}: {exc.message[:80]}")
            continue
        d = diagnose(answer)
        results.append(d)
        bad = is_degenerate(d)
        arts = ", ".join(d["artifacts"]) or "-"
        print(
            f"  {i:2}. {'BAD ' if bad else 'ok  '} conf={d['confidence']} "
            f"chars={d['chars']:3}  artifacts: {arts}"
            f"{'  SELF-FLAGGED' if d['self_flagged'] else ''}"
        )

    if not results:
        print("\nno successful trials")
        return 1

    bad = [d for d in results if is_degenerate(d)]
    print(f"\n{'=' * 68}")
    print(
        f"degenerate: {len(bad)}/{len(results)} "
        f"({100 * len(bad) / len(results):.0f}%)   errors: {errors}"
    )
    print(f"median length: {statistics.median(int(d['chars']) for d in results):.0f} chars")
    print(f"median confidence: {statistics.median(float(d['confidence']) for d in results):.2f}")

    if bad:
        print("\n--- worst example ---")
        worst = max(bad, key=lambda d: (int(d["newlines"]), int(d["repeated_4grams"])))
        print(worst["text"])
    else:
        print("\n--- sample answer ---")
        print(results[0]["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
