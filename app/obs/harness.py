"""Tracing wrapper around `run_agent`.

`app/harness/orchestrator.py` is deliberately not modified by any of this.
It already yields a complete description of what happened -- a `ProgressEvent`
stream plus a final `EngineTrace` -- so observability can be a *consumer* of
that generator rather than an edit to it. The harness stays about provider
resilience; this module stays about reporting.

Two shapes it gets right that a naive integration gets wrong:

1. **One trace per request, attempts nested inside it.** Emitting a
   generation per provider call would turn a request that retries three times
   and then fails over into six top-level generations, and every quality
   metric downstream would double-count the failures.
2. **Retrieval, when it arrives, is traced above the provider loop.** Not
   here yet, but the seam is the same one: a span opened before the first
   attempt, never inside the retry loop.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence

from app.config import Settings
from app.engines import Engine
from app.harness import run_agent
from app.obs.base import NullTrace, Trace, Tracer
from app.quality import diagnose, is_degenerate
from app.schemas import AgentResponse, AttemptOutcome, ProgressEvent

logger = logging.getLogger(__name__)


async def traced_run_agent(
    question: str,
    settings: Settings,
    tracer: Tracer,
    *,
    engines: Sequence[Engine] | None = None,
    tags: list[str] | None = None,
) -> AsyncIterator[ProgressEvent | AgentResponse]:
    """`run_agent`, with one Langfuse trace per call.

    Yields exactly what `run_agent` yields, in the same order, so both routes
    can swap this in without changing their consumption logic.
    """
    # Guarded because this is the one telemetry call outside the tracer's own
    # `never_raises` methods: a backend that throws on trace creation would
    # otherwise fail the request before the harness ever ran.
    try:
        trace: Trace = tracer.trace(
            "jeopardy2.ask",
            input={"question": question},
            metadata={
                "prompt_variant": settings.prompt_variant,
                "provider_order": settings.provider_order,
                "max_attempts": settings.max_attempts,
                "classify_errors": settings.classify_errors,
            },
            tags=tags,
        )
    except Exception:  # noqa: BLE001 - telemetry must not break the request
        logger.warning("could not open trace; continuing untraced", exc_info=True)
        trace = NullTrace()

    final: AgentResponse | None = None
    seen_events: list[str] = []
    try:
        async for item in run_agent(question, settings, engines):
            if isinstance(item, AgentResponse):
                final = item
            else:
                seen_events.append(item.event)
            yield item
    finally:
        # A `finally` because the SSE route abandons this generator when the
        # client disconnects. Without it those requests would leave a trace
        # open forever and silently skew every duration statistic.
        #
        # Guarded for the same reason as trace creation, and doubly so here:
        # an exception raised inside a `finally` would replace whatever the
        # caller was already handling, including a legitimate CancelledError.
        try:
            _record(trace, final, seen_events, settings)
        except Exception:  # noqa: BLE001 - telemetry must not break the request
            logger.warning("could not record trace", exc_info=True)


def _record(
    trace: Trace,
    final: AgentResponse | None,
    seen_events: list[str],
    settings: Settings,
) -> None:
    if final is None:
        # Abandoned mid-flight: record what we saw so the trace is not a lie.
        trace.end(output=None, metadata={"aborted": True, "events": seen_events})
        return

    # One span per attempt, built from the EngineTrace rather than replayed
    # from events: the trace carries durations, outcomes and error types that
    # the progress events intentionally do not.
    for attempt in final.trace.attempts:
        trace.span(
            f"attempt:{attempt.provider}#{attempt.attempt}",
            input={"provider": attempt.provider, "model": attempt.model},
            output={"outcome": attempt.outcome.value},
            metadata={
                "duration_ms": attempt.duration_ms,
                "slept_before_ms": attempt.slept_before_ms,
                "error_type": attempt.error_type,
                "error_message": attempt.error_message,
                "retryable": attempt.outcome is AttemptOutcome.TRANSIENT_ERROR,
            },
        )

    if final.answer is not None:
        trace.generation(
            "answer",
            model=final.trace.served_by_model or "unknown",
            input={"question": final.question, "prompt_variant": settings.prompt_variant},
            output=final.answer.model_dump(),
            metadata={
                "provider": final.trace.served_by_provider,
                "used_fallback": final.trace.used_fallback,
            },
        )

    _score(trace, final)
    trace.end(
        output=final.answer.model_dump() if final.answer else None,
        metadata={
            "status": final.status,
            "served_by_provider": final.trace.served_by_provider,
            "served_by_model": final.trace.served_by_model,
            "used_fallback": final.trace.used_fallback,
            "total_ms": final.trace.total_ms,
            "attempts": len(final.trace.attempts),
        },
    )


def _score(trace: Trace, final: AgentResponse) -> None:
    """Attach the scores a prompt-quality review actually reads.

    `answer_clean` is the same detector the offline probe uses, so live
    traffic and eval runs produce a comparable number. That is the point of
    scoring in production: the probe tells you what a prompt does on one
    fixed question, this tells you what it does on real ones.
    """
    trace.score("served", 0.0 if final.status == "degraded" else 1.0)
    trace.score("used_fallback", 1.0 if final.trace.used_fallback else 0.0)
    trace.score("attempts", float(len(final.trace.attempts)))

    if final.answer is None:
        return
    d = diagnose(final.answer)
    artifacts = d["artifacts"]
    trace.score(
        "answer_clean",
        0.0 if is_degenerate(d) else 1.0,
        comment=", ".join(artifacts) if isinstance(artifacts, list) and artifacts else None,
    )
    trace.score("self_reported_confidence", float(final.answer.confidence))
