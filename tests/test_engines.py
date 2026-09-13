"""Provider-adapter error classification.

These cases are built from real error bodies observed against live accounts,
not invented shapes -- the quota case in particular was found only by running
the deployment against an OpenAI account with no credits.
"""

from __future__ import annotations

import httpx
import openai
import pytest

from app.engines.openai_engine import OpenAIEngine, _is_quota_exhaustion
from app.schemas import FaultTarget


def _rate_limit_error(body: dict) -> openai.RateLimitError:
    """Build a real RateLimitError carrying `body` as its JSON response."""
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(429, json=body, request=request)
    return openai.RateLimitError("rate limited", response=response, body=body.get("error"))


# The exact payload OpenAI returned for an unfunded account.
QUOTA_BODY = {
    "error": {
        "message": (
            "You have no credits remaining. Add credits to continue using the "
            "API at https://platform.openai.com/settings/organization/billing/."
        ),
        "type": "insufficient_quota",
        "param": None,
        "code": "credit_balance_exhausted",
    }
}

# A genuine rate limit, which *is* worth retrying.
RATE_BODY = {
    "error": {
        "message": "Rate limit reached for gpt-5.5 in organization org-x.",
        "type": "requests",
        "param": None,
        "code": "rate_limit_exceeded",
    }
}


def test_quota_exhaustion_is_detected():
    assert _is_quota_exhaustion(_rate_limit_error(QUOTA_BODY)) is True


def test_genuine_rate_limit_is_not_quota_exhaustion():
    assert _is_quota_exhaustion(_rate_limit_error(RATE_BODY)) is False


def test_empty_body_is_not_treated_as_quota():
    """A 429 with no parsable body stays retryable -- fail open, not closed."""
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(429, text="", request=request)
    exc = openai.RateLimitError("rate limited", response=response, body=None)

    assert _is_quota_exhaustion(exc) is False


@pytest.mark.parametrize(
    "code",
    ["insufficient_quota", "credit_balance_exhausted", "billing_hard_limit_reached"],
)
def test_all_billing_codes_are_permanent(code):
    body = {"error": {"message": "no funds", "type": code, "code": code}}
    assert _is_quota_exhaustion(_rate_limit_error(body)) is True


async def test_quota_error_surfaces_as_permanent_engine_error(fast_settings):
    """End result: the harness fails over at once instead of sleeping 30s."""
    engine = OpenAIEngine(fast_settings)

    class _Boom:
        async def parse(self, **_kwargs):
            raise _rate_limit_error(QUOTA_BODY)

    class _Client:
        responses = _Boom()

    engine._client = _Client()  # bypass the real constructor

    from app.engines.base import EngineError

    with pytest.raises(EngineError) as caught:
        await engine.answer("hi")

    assert caught.value.retryable is False, "quota exhaustion must not be retried"
    assert caught.value.error_type == "QuotaExhausted"
    assert "billing" in caught.value.message


async def test_rate_limit_error_stays_retryable(fast_settings):
    engine = OpenAIEngine(fast_settings)

    class _Boom:
        async def parse(self, **_kwargs):
            raise _rate_limit_error(RATE_BODY)

    class _Client:
        responses = _Boom()

    engine._client = _Client()

    from app.engines.base import EngineError

    with pytest.raises(EngineError) as caught:
        await engine.answer("hi")

    assert caught.value.retryable is True
    assert caught.value.error_type == "RateLimitError"


def test_engine_is_unavailable_without_a_key(fast_settings):
    engine = OpenAIEngine(fast_settings.model_copy(update={"openai_api_key": None}))
    assert engine.is_available() == "OPENAI_API_KEY is not set"
    assert OpenAIEngine(fast_settings).is_available() is None


async def test_injected_fault_raises_before_any_network_call(fast_settings):
    """The fault hook must fire without constructing a client."""
    from app.engines.base import EngineError
    from app.engines.faults import forced_fault

    engine = OpenAIEngine(fast_settings)
    with forced_fault(FaultTarget.OPENAI), pytest.raises(EngineError) as caught:
        await engine.answer("hi")

    assert caught.value.error_type == "InjectedFault"
    assert caught.value.retryable is True
    assert engine._client is None, "no client should have been constructed"
