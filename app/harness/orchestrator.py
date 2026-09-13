"""Provider chain with per-provider retry, backoff, and failover.

Shape: `run_agent` is an async generator that yields `ProgressEvent`s as work
happens and finishes by yielding exactly one `AgentResponse`. Both endpoints
consume the same generator -- the streaming one forwards every event, the
synchronous one drains and keeps the final response. One implementation, so
the two routes cannot drift apart in their retry behaviour.

Failure policy: this generator does not raise for provider failures. When
every engine is exhausted it yields a `degraded` response, which is why the
service can promise never to return a 5xx.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Sequence

from app.config import Settings
from app.engines import Engine, EngineError, build_engine
from app.schemas import (
    AgentResponse,
    Answer,
    AttemptOutcome,
    EngineAttempt,
    EngineTrace,
    ProgressEvent,
)

logger = logging.getLogger(__name__)


async def run_agent(
    question: str,
    settings: Settings,
    engines: Sequence[Engine] | None = None,
) -> AsyncIterator[ProgressEvent | AgentResponse]:
    """Answer `question`, trying each configured provider in order.

    Yields progress events throughout and exactly one `AgentResponse` last.
    """
    chain: Sequence[Engine] = engines or [
        build_engine(name, settings) for name in settings.provider_order
    ]
    trace = EngineTrace()
    started = time.perf_counter()
    primary_name = chain[0].name if chain else None

    yield ProgressEvent(
        event="started",
        message=f"Answering via {' → '.join(e.name for e in chain)}",
    )

    for position, engine in enumerate(chain):
        if position > 0:
            yield ProgressEvent(
                event="falling_back",
                message=f"Falling back to {engine.name} ({engine.model})",
                provider=engine.name,
                model=engine.model,
            )

        unavailable = engine.is_available()
        if unavailable:
            # Skip rather than spend the full backoff schedule rediscovering a
            # missing API key on every attempt.
            trace.attempts.append(
                EngineAttempt(
                    provider=engine.name,
                    model=engine.model,
                    attempt=1,
                    outcome=AttemptOutcome.SKIPPED,
                    duration_ms=0,
                    error_type="Unavailable",
                    error_message=unavailable,
                )
            )
            yield ProgressEvent(
                event="provider_skipped",
                message=f"Skipping {engine.name}: {unavailable}",
                provider=engine.name,
            )
            continue

        answer: Answer | None = None

        for attempt in range(1, settings.max_attempts + 1):
            slept_ms = 0
            if attempt > 1:
                delay = settings.backoff_for(attempt)
                if delay > 0:
                    yield ProgressEvent(
                        event="waiting",
                        message=f"Retrying {engine.name} in {delay:.0f}s",
                        provider=engine.name,
                        attempt=attempt,
                        retry_in_seconds=delay,
                    )
                    await asyncio.sleep(delay)
                    slept_ms = int(delay * 1000)

            yield ProgressEvent(
                event="attempt_started",
                message=f"{engine.name} attempt {attempt}/{settings.max_attempts}",
                provider=engine.name,
                model=engine.model,
                attempt=attempt,
            )

            attempt_started = time.perf_counter()
            try:
                answer = await engine.answer(question)
            except EngineError as exc:
                duration_ms = _elapsed_ms(attempt_started)
                outcome = (
                    AttemptOutcome.TRANSIENT_ERROR
                    if exc.retryable
                    else AttemptOutcome.PERMANENT_ERROR
                )
                trace.attempts.append(
                    EngineAttempt(
                        provider=engine.name,
                        model=engine.model,
                        attempt=attempt,
                        outcome=outcome,
                        duration_ms=duration_ms,
                        error_type=exc.error_type,
                        error_message=exc.message,
                        slept_before_ms=slept_ms,
                    )
                )
                logger.warning(
                    "engine attempt failed: provider=%s attempt=%d type=%s retryable=%s",
                    engine.name,
                    attempt,
                    exc.error_type,
                    exc.retryable,
                )
                yield ProgressEvent(
                    event="attempt_failed",
                    message=f"{engine.name} attempt {attempt} failed: {exc.error_type}",
                    provider=engine.name,
                    model=engine.model,
                    attempt=attempt,
                )

                # Classification is what keeps a typo'd API key from costing
                # 30 seconds of pointless sleeping before failover.
                if settings.classify_errors and not exc.retryable:
                    break
                continue
            except asyncio.CancelledError:
                # Client disconnected or shutdown: never swallow this.
                raise
            except Exception as exc:  # noqa: BLE001 - adapter bug, not a provider error
                duration_ms = _elapsed_ms(attempt_started)
                trace.attempts.append(
                    EngineAttempt(
                        provider=engine.name,
                        model=engine.model,
                        attempt=attempt,
                        outcome=AttemptOutcome.PERMANENT_ERROR,
                        duration_ms=duration_ms,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        slept_before_ms=slept_ms,
                    )
                )
                logger.exception("unexpected error from engine %s", engine.name)
                yield ProgressEvent(
                    event="attempt_failed",
                    message=f"{engine.name} raised {type(exc).__name__}",
                    provider=engine.name,
                    attempt=attempt,
                )
                break

            # Success.
            trace.attempts.append(
                EngineAttempt(
                    provider=engine.name,
                    model=engine.model,
                    attempt=attempt,
                    outcome=AttemptOutcome.SUCCESS,
                    duration_ms=_elapsed_ms(attempt_started),
                    slept_before_ms=slept_ms,
                )
            )
            break

        if answer is not None:
            trace.served_by_provider = engine.name
            trace.served_by_model = engine.model
            trace.used_fallback = engine.name != primary_name
            trace.total_ms = _elapsed_ms(started)
            yield ProgressEvent(
                event="completed",
                message=f"Answered by {engine.name} ({engine.model})",
                provider=engine.name,
                model=engine.model,
            )
            yield AgentResponse(
                status="ok",
                question=question,
                answer=answer,
                trace=trace,
            )
            return

    # Every provider exhausted.
    trace.total_ms = _elapsed_ms(started)
    message = _degraded_message(trace)
    logger.error("all providers failed: %s", message)
    yield ProgressEvent(event="failed", message=message)
    yield AgentResponse(
        status="degraded",
        question=question,
        answer=None,
        message=message,
        trace=trace,
    )


def _elapsed_ms(since: float) -> int:
    return int((time.perf_counter() - since) * 1000)


def _degraded_message(trace: EngineTrace) -> str:
    """A one-line explanation the UI can show without parsing the trace."""
    if not trace.attempts:
        return "No providers are configured."

    if all(a.outcome is AttemptOutcome.SKIPPED for a in trace.attempts):
        reasons = ", ".join(f"{a.provider} ({a.error_message})" for a in trace.attempts)
        return f"No provider was usable: {reasons}. Set an API key in .env."

    last = trace.attempts[-1]
    # Order of first appearance, not sorted -- the chain order is the
    # informative part when you're reading why a request failed.
    providers = list(dict.fromkeys(a.provider for a in trace.attempts))
    return (
        f"All providers failed after {len(trace.attempts)} attempt(s) across "
        f"{' → '.join(providers)}. Last error: {last.error_type} - {last.error_message}"
    )
