"""Harness behaviour: retry counts, classification, failover, degradation."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.harness import run_agent
from app.schemas import AgentResponse, AttemptOutcome, ProgressEvent
from tests.conftest import FakeEngine, ok_answer, permanent, transient


async def drain(question, settings, engines):
    """Run the agent and split the stream into events and the final response."""
    events: list[ProgressEvent] = []
    final: AgentResponse | None = None
    async for item in run_agent(question, settings, engines):
        if isinstance(item, AgentResponse):
            final = item
        else:
            events.append(item)
    assert final is not None, "run_agent must always yield a final AgentResponse"
    return events, final


async def test_primary_success_never_touches_fallback(fast_settings):
    primary = FakeEngine("openai", script=[ok_answer("yes")])
    fallback = FakeEngine("anthropic", script=[ok_answer("should not run")])

    _, res = await drain("q", fast_settings, [primary, fallback])

    assert res.status == "ok"
    assert res.answer.answer == "yes"
    assert res.trace.served_by_provider == "openai"
    assert res.trace.used_fallback is False
    assert fallback.calls == 0


async def test_retries_transient_then_succeeds_on_same_provider(fast_settings):
    primary = FakeEngine(
        "openai", script=[transient("openai"), transient("openai"), ok_answer("third time")]
    )
    fallback = FakeEngine("anthropic", script=[ok_answer("unused")])

    _, res = await drain("q", fast_settings, [primary, fallback])

    assert res.status == "ok"
    assert res.answer.answer == "third time"
    assert primary.calls == 3
    assert fallback.calls == 0
    assert res.trace.used_fallback is False
    outcomes = [a.outcome for a in res.trace.attempts]
    assert outcomes == [
        AttemptOutcome.TRANSIENT_ERROR,
        AttemptOutcome.TRANSIENT_ERROR,
        AttemptOutcome.SUCCESS,
    ]


async def test_exhausts_attempts_then_falls_back(fast_settings):
    primary = FakeEngine("openai", script=[transient("openai")])
    fallback = FakeEngine("anthropic", script=[ok_answer("claude here")])

    _, res = await drain("q", fast_settings, [primary, fallback])

    assert res.status == "ok"
    assert res.answer.answer == "claude here"
    # Exactly max_attempts on the primary -- not the SDK's retries on top.
    assert primary.calls == fast_settings.max_attempts == 3
    assert fallback.calls == 1
    assert res.trace.served_by_provider == "anthropic"
    assert res.trace.used_fallback is True


async def test_permanent_error_fails_over_without_retrying(fast_settings):
    """A bad API key must not burn the backoff schedule."""
    primary = FakeEngine("openai", script=[permanent("openai")])
    fallback = FakeEngine("anthropic", script=[ok_answer("rescued")])

    _, res = await drain("q", fast_settings, [primary, fallback])

    assert res.status == "ok"
    assert primary.calls == 1, "permanent errors should not be retried"
    assert res.trace.used_fallback is True
    assert res.trace.attempts[0].outcome == AttemptOutcome.PERMANENT_ERROR


async def test_classification_off_retries_permanent_errors(fast_settings):
    """With classify_errors disabled, every error burns the full schedule."""
    settings = fast_settings.model_copy(update={"classify_errors": False})
    primary = FakeEngine("openai", script=[permanent("openai")])
    fallback = FakeEngine("anthropic", script=[ok_answer("rescued")])

    _, res = await drain("q", settings, [primary, fallback])

    assert primary.calls == 3
    assert res.status == "ok"


async def test_all_providers_fail_returns_degraded_not_exception(fast_settings):
    primary = FakeEngine("openai", script=[transient("openai")])
    fallback = FakeEngine("anthropic", script=[transient("anthropic")])

    _, res = await drain("q", fast_settings, [primary, fallback])

    assert res.status == "degraded"
    assert res.answer is None
    assert res.message
    assert primary.calls == 3
    assert fallback.calls == 3
    assert len(res.trace.attempts) == 6


async def test_missing_key_skips_provider_without_attempts(fast_settings):
    primary = FakeEngine("openai", unavailable="OPENAI_API_KEY is not set")
    fallback = FakeEngine("anthropic", script=[ok_answer("claude only")])

    events, res = await drain("q", fast_settings, [primary, fallback])

    assert res.status == "ok"
    assert primary.calls == 0
    assert res.trace.attempts[0].outcome == AttemptOutcome.SKIPPED
    # A provider that was skipped still counts as falling back.
    assert res.trace.used_fallback is True
    assert any(e.event == "provider_skipped" for e in events)


async def test_no_usable_provider_explains_itself(fast_settings):
    primary = FakeEngine("openai", unavailable="OPENAI_API_KEY is not set")
    fallback = FakeEngine("anthropic", unavailable="ANTHROPIC_API_KEY is not set")

    _, res = await drain("q", fast_settings, [primary, fallback])

    assert res.status == "degraded"
    assert "API key" in res.message


async def test_unexpected_exception_is_contained(fast_settings):
    """An adapter bug should fail over, not propagate out of the harness."""
    primary = FakeEngine("openai", script=[ValueError("bug in adapter")])
    fallback = FakeEngine("anthropic", script=[ok_answer("still works")])

    _, res = await drain("q", fast_settings, [primary, fallback])

    assert res.status == "ok"
    assert res.answer.answer == "still works"
    assert res.trace.attempts[0].error_type == "ValueError"
    # Unexpected errors are not retried -- we don't know that they're transient.
    assert primary.calls == 1


async def test_progress_events_narrate_the_failover(fast_settings):
    primary = FakeEngine("openai", script=[transient("openai")])
    fallback = FakeEngine("anthropic", script=[ok_answer("hi")])

    events, _ = await drain("q", fast_settings, [primary, fallback])
    kinds = [e.event for e in events]

    assert kinds[0] == "started"
    assert kinds.count("attempt_started") == 4  # 3 primary + 1 fallback
    assert kinds.count("attempt_failed") == 3
    assert kinds.count("waiting") == 2  # backoff only between attempts
    assert "falling_back" in kinds
    assert kinds[-1] == "completed"


async def test_waiting_events_report_configured_backoff(fast_settings):
    settings = fast_settings.model_copy(update={"backoff_seconds": [0.001, 0.002]})
    primary = FakeEngine("openai", script=[transient("openai")])
    fallback = FakeEngine("anthropic", script=[ok_answer("hi")])

    events, _ = await drain("q", settings, [primary, fallback])
    waits = [e.retry_in_seconds for e in events if e.event == "waiting"]

    # Two gaps for three attempts, taken in schedule order.
    assert waits == [0.001, 0.002]


async def test_single_provider_chain_still_degrades_cleanly(fast_settings):
    settings = fast_settings.model_copy(update={"provider_order": ["openai"]})
    only = FakeEngine("openai", script=[transient("openai")])

    _, res = await drain("q", settings, [only])

    assert res.status == "degraded"
    assert only.calls == 3


# --------------------------------------------------------------------------
# Backoff schedule arithmetic
# --------------------------------------------------------------------------
def test_backoff_schedule_matches_spec():
    """3 attempts with gaps of 10s then 20s -- 30s worst case per provider."""
    s = Settings(max_attempts=3, backoff_seconds=[10, 20])

    assert s.backoff_for(1) == 0.0, "first attempt is immediate"
    assert s.backoff_for(2) == 10.0
    assert s.backoff_for(3) == 20.0
    assert sum(s.backoff_for(i) for i in range(1, 4)) == 30.0


def test_backoff_repeats_last_value_when_attempts_exceed_schedule():
    s = Settings(max_attempts=5, backoff_seconds=[10, 20])

    assert s.backoff_for(4) == 20.0
    assert s.backoff_for(5) == 20.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backoff_seconds": []},
        {"backoff_seconds": [-1]},
        {"max_attempts": 0},
        {"provider_order": []},
        {"provider_order": ["openai", "gemini"]},
    ],
)
def test_invalid_config_is_rejected_at_startup(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)
