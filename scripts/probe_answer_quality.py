#!/usr/bin/env python3
"""Measure how often a provider corrupts the `answer` field.

Background: claude-opus-5 mis-escapes non-ASCII punctuation inside
structured-output JSON. A single em dash was observed surfacing five
different ways -- as the literal text "\\u2014", a line break, the word
"dash", a stray quote, or (once in eight) correctly. Measured rate was 7/8
corrupted before the prompt forbade non-ASCII punctuation, and 0/12 after.

The point of this script is that the fix was *measured*, not guessed. Run it
before and after any prompt change that touches formatting.

The two system prompts that produced those numbers are now named variants in
`app/prompts.py`, so the comparison is a flag rather than a hand-edit:

    # the control -- expect a high artifact rate on claude-opus-5
    python scripts/probe_answer_quality.py --trials 8 --system-prompt no-ascii-rule

    # production -- expect none
    python scripts/probe_answer_quality.py --trials 12 --system-prompt ascii-guard

    # or run both and print the comparison
    python scripts/probe_answer_quality.py --trials 8 --compare

With `LANGFUSE_ENABLED=true` and keys set, every trial is also sent as a
trace with `answer_clean` / `self_reported_confidence` scores and the prompt
digest in metadata. Without it the script behaves exactly as before, so the
measurement never depends on a service being up.

Lesson from its own history: the first version of the detector counted only
line breaks and reported 30%, under-reporting the real 87% threefold,
because line breaks were one artifact form out of five. Inspect bytes
(`repr()`), not how the rendered text looks. The detector now lives in
`app/quality.py` so the live request path scores answers identically.

Costs real money: one API call per trial (two per trial with --compare).
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.engines import build_engine  # noqa: E402
from app.engines.base import EngineError  # noqa: E402
from app.obs import build_tracer  # noqa: E402
from app.prompts import get_prompt, prompt_names  # noqa: E402
from app.quality import diagnose, is_degenerate  # noqa: E402

QUESTION = "Explain AI in one or two sentences that my grandfather could understand."

TIGHTENED_SUFFIX = (
    "\n\nFormat: one or two complete sentences of plain prose. "
    "No line breaks, no bullet points, and do not offer alternative "
    "phrasings -- commit to a single wording."
)


async def run_variant(
    *,
    provider: str,
    system_prompt: str,
    question: str,
    trials: int,
    settings: Settings,
    tracer: object,
    quiet: bool = False,
) -> dict[str, object] | None:
    """Run `trials` calls against one prompt variant and summarize."""
    engine = build_engine(provider, settings, prompt_name=system_prompt)
    if (reason := engine.is_available()) is not None:
        print(f"error: {reason}", file=sys.stderr)
        return None

    prompt = get_prompt(system_prompt)
    if not quiet:
        print(
            f"provider={provider} model={engine.model} "
            f"prompt={prompt.name}@{prompt.digest} trials={trials}\n"
        )

    results: list[dict[str, object]] = []
    errors = 0
    for i in range(1, trials + 1):
        trace = tracer.trace(  # type: ignore[attr-defined]
            "probe.answer_quality",
            input={"question": question},
            metadata={
                "prompt_variant": prompt.name,
                "prompt_digest": prompt.digest,
                "provider": provider,
                "model": engine.model,
                "trial": i,
            },
            tags=["probe", f"prompt:{prompt.name}", f"provider:{provider}"],
        )
        try:
            answer = await engine.answer(question)
        except EngineError as exc:
            errors += 1
            trace.score("served", 0.0)
            trace.end(metadata={"error_type": exc.error_type, "message": exc.message})
            print(f"  {i:2}. ERROR {exc.error_type}: {exc.message[:80]}")
            continue

        d = diagnose(answer)
        results.append(d)
        bad = is_degenerate(d)

        trace.generation(
            "answer",
            model=engine.model,
            input={"question": question, "system_prompt_digest": prompt.digest},
            output=answer.model_dump(),
            metadata={"artifacts": d["artifacts"]},
        )
        trace.score("served", 1.0)
        trace.score(
            "answer_clean",
            0.0 if bad else 1.0,
            comment=", ".join(d["artifacts"]) or None,  # type: ignore[arg-type]
        )
        trace.score("self_reported_confidence", float(answer.confidence))
        trace.end(output=answer.model_dump())

        arts = ", ".join(d["artifacts"]) or "-"  # type: ignore[arg-type]
        print(
            f"  {i:2}. {'BAD ' if bad else 'ok  '} conf={d['confidence']} "
            f"chars={d['chars']:3}  artifacts: {arts}"
            f"{'  SELF-FLAGGED' if d['self_flagged'] else ''}"
        )

    if not results:
        print("\nno successful trials")
        return None

    bad_runs = [d for d in results if is_degenerate(d)]
    return {
        "prompt": prompt.name,
        "digest": prompt.digest,
        "model": engine.model,
        "results": results,
        "bad": bad_runs,
        "errors": errors,
    }


def print_summary(summary: dict[str, object]) -> None:
    results = summary["results"]
    bad = summary["bad"]
    assert isinstance(results, list) and isinstance(bad, list)
    print(f"\n{'=' * 68}")
    print(
        f"degenerate: {len(bad)}/{len(results)} "
        f"({100 * len(bad) / len(results):.0f}%)   errors: {summary['errors']}"
    )
    print(f"median length: {statistics.median(int(d['chars']) for d in results):.0f} chars")
    print(f"median confidence: {statistics.median(float(d['confidence']) for d in results):.2f}")

    if bad:
        print("\n--- worst example ---")
        worst = max(bad, key=lambda d: (int(d["newlines"]), int(d["repeated_4grams"])))
        print(repr(worst["text"]))
    else:
        print("\n--- sample answer ---")
        print(results[0]["text"])


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument(
        "--system-prompt",
        default=None,
        choices=prompt_names(),
        help="Named variant from app/prompts.py. Defaults to PROMPT_VARIANT.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Run no-ascii-rule then ascii-guard and print both rates. This is "
            "the 7/8 vs 0/12 measurement; it costs 2x --trials API calls."
        ),
    )
    parser.add_argument(
        "--variant",
        default="baseline",
        choices=["baseline", "tightened"],
        help=(
            "Modifies the *user* message, not the system prompt. 'tightened' "
            "adds explicit length/format guidance to the request."
        ),
    )
    args = parser.parse_args()

    settings = Settings()
    tracer = build_tracer(settings)
    if (reason := tracer.is_available()) is not None:
        print(f"(tracing off: {reason})\n")

    question = args.question
    if args.variant == "tightened":
        question = f"{args.question}{TIGHTENED_SUFFIX}"

    try:
        if args.compare:
            summaries = []
            for name in ("no-ascii-rule", "ascii-guard"):
                print(f"\n### {name} -- {get_prompt(name).notes}\n")
                s = await run_variant(
                    provider=args.provider,
                    system_prompt=name,
                    question=question,
                    trials=args.trials,
                    settings=settings,
                    tracer=tracer,
                )
                if s is None:
                    return 1
                print_summary(s)
                summaries.append(s)

            print(f"\n{'=' * 68}\nCOMPARISON ({args.provider})\n")
            for s in summaries:
                results, bad = s["results"], s["bad"]
                assert isinstance(results, list) and isinstance(bad, list)
                print(
                    f"  {str(s['prompt']):16} {len(bad)}/{len(results)} degenerate "
                    f"({100 * len(bad) / len(results):3.0f}%)   digest={s['digest']}"
                )
            print(
                "\nExpect a large gap on claude-opus-5 and none on gpt-5.5. "
                "The asymmetry is the finding: the rule is load-bearing for "
                "Claude and cosmetic for OpenAI."
            )
            return 0

        summary = await run_variant(
            provider=args.provider,
            system_prompt=args.system_prompt or settings.prompt_variant,
            question=question,
            trials=args.trials,
            settings=settings,
            tracer=tracer,
        )
        if summary is None:
            return 1
        print_summary(summary)
        return 0
    finally:
        # Buffered events are lost if the process exits first.
        tracer.flush()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
