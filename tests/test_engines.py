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


# --------------------------------------------------------------------------
# Prompt invariants
# --------------------------------------------------------------------------
def test_system_prompt_forbids_non_ascii_punctuation():
    """This rule is load-bearing and was expensive to find -- don't drop it.

    Measured against claude-opus-5 with structured output: without an
    explicit ASCII-punctuation rule, ~87% of responses (7 of 8) mis-escaped
    an em dash inside the JSON string, surfacing as a literal "\\u2014", a
    line break, the word "dash", or a stray quote mid-sentence. With the
    rule, 0 of 12 showed any artifact. Re-measure with
    scripts/probe_answer_quality.py before changing this.
    """
    from app.engines.base import SYSTEM_PROMPT

    lowered = SYSTEM_PROMPT.lower()
    assert "ascii" in lowered
    assert "em dash" in lowered or "em dashes" in lowered
    assert "line break" in lowered


# --------------------------------------------------------------------------
# Cross-provider schema compatibility
# --------------------------------------------------------------------------
def test_answer_schema_has_no_unsupported_keywords():
    """The wire schema must stay portable across both providers.

    Anthropic's structured-output endpoint rejects `minimum`/`maximum` on a
    number outright:

        400 output_config.format.schema: For 'number' type, properties
            maximum, minimum are not supported

    Anthropic's messages.parse() silently strips them, so the Anthropic path
    survived by accident rather than by design.

    Since confirmed against the live APIs: OpenAI ACCEPTS minimum/maximum, so
    the OpenAI path was never at risk from this. The guard stays because the
    Anthropic path's survival depends on undocumented SDK stripping behaviour
    that could change in any release, and because one portable schema is
    simpler to reason about than two provider-specific ones.

    Enforce the range with a validator (no schema keywords), not Field bounds.
    """
    from app.schemas import Answer

    schema = Answer.model_json_schema()
    number_fields = {
        name: spec for name, spec in schema["properties"].items() if spec.get("type") == "number"
    }
    assert number_fields, "expected at least one number field to guard"

    banned = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}
    for name, spec in number_fields.items():
        offending = banned & set(spec)
        assert not offending, (
            f"{name} emits {sorted(offending)} into the wire schema; use a "
            f"field_validator instead (see this test's docstring)"
        )

    # Both providers' strict modes require these two properties.
    assert schema.get("additionalProperties") is False
    assert set(schema["required"]) == set(schema["properties"])


def test_confidence_range_is_still_enforced():
    """Dropping Field bounds must not drop the constraint."""
    from app.schemas import Answer

    assert Answer(answer="x", confidence=0.9, caveats=[]).confidence == 0.9
    assert Answer(answer="x", confidence=1.05, caveats=[]).confidence == 1.0
    assert Answer(answer="x", confidence=-0.2, caveats=[]).confidence == 0.0


def test_openai_strict_conversion_produces_a_clean_schema():
    """Guard the actual payload the OpenAI SDK builds, not just Pydantic's."""
    import json

    from openai.lib._pydantic import to_strict_json_schema

    from app.schemas import Answer

    serialized = json.dumps(to_strict_json_schema(Answer))
    for keyword in ('"minimum"', '"maximum"', '"multipleOf"'):
        assert keyword not in serialized, f"{keyword} would be sent to OpenAI"
