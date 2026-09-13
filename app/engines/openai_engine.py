"""OpenAI engine -- the primary provider.

Uses the Responses API's `parse` helper, which enforces our Pydantic schema
server-side rather than asking for JSON and hoping.
"""

from __future__ import annotations

import logging

import openai

from app.config import Settings
from app.engines.base import SYSTEM_PROMPT, EngineError
from app.engines.faults import current_fault
from app.schemas import Answer, FaultTarget

logger = logging.getLogger(__name__)

PROVIDER = "openai"

#: Retryable: waiting might genuinely help.
_TRANSIENT = (
    openai.RateLimitError,  # 429
    openai.InternalServerError,  # 5xx
    openai.APITimeoutError,
    openai.APIConnectionError,  # DNS, refused, dropped
)

#: Non-retryable: the request itself is wrong, so three tries waste 30 seconds.
_PERMANENT = (
    openai.AuthenticationError,  # 401 - bad or missing key
    openai.PermissionDeniedError,  # 403
    openai.NotFoundError,  # 404 - unknown model ID
    openai.BadRequestError,  # 400 - malformed request or schema
    openai.UnprocessableEntityError,  # 422
)

#: OpenAI overloads HTTP 429 for two unrelated conditions: genuine rate
#: limiting (waiting helps) and an exhausted credit balance (waiting never
#: helps -- someone has to add money). Treating both as transient means an
#: unfunded account burns the entire backoff schedule to rediscover the same
#: permanent fact on every attempt. Found by running this against a real
#: account with no credits.
_QUOTA_CODES = frozenset(
    {
        "insufficient_quota",
        "credit_balance_exhausted",
        "billing_hard_limit_reached",
        "account_deactivated",
    }
)


def _is_quota_exhaustion(exc: openai.RateLimitError) -> bool:
    """True when a 429 means "out of money", not "slow down"."""
    # `.code` is the machine-readable error code; the error `type` lives in
    # the response body, which the SDK does not surface as an attribute.
    if getattr(exc, "code", None) in _QUOTA_CODES:
        return True
    body = getattr(exc, "response", None)
    try:
        payload = body.json() if body is not None else {}
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        if error.get("type") in _QUOTA_CODES or error.get("code") in _QUOTA_CODES:
            return True
    except Exception:  # noqa: BLE001 - body may be empty or not JSON
        pass
    return False


class OpenAIEngine:
    name = PROVIDER

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.model = settings.openai_model
        self._client: openai.AsyncOpenAI | None = None

    def is_available(self) -> str | None:
        if self._settings.openai_api_key is None:
            return "OPENAI_API_KEY is not set"
        return None

    def _get_client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            key = self._settings.openai_api_key
            self._client = openai.AsyncOpenAI(
                api_key=key.get_secret_value() if key else None,
                timeout=self._settings.request_timeout_seconds,
                # The harness owns retries. Leaving the SDK's default of 2 in
                # place would silently turn our 3 attempts into 9 and stretch
                # the backoff schedule we advertise.
                max_retries=0,
            )
        return self._client

    async def answer(self, question: str) -> Answer:
        self._maybe_inject_fault()

        client = self._get_client()
        try:
            response = await client.responses.parse(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                input=question,
                text_format=Answer,
                # Generous on purpose. On the GPT-5 family reasoning tokens
                # count against this budget, so a tight ceiling can exhaust it
                # before any answer is emitted -- which surfaces here as
                # output_parsed being None, i.e. an unexplained failover rather
                # than an obvious truncation. You are billed for tokens used,
                # not the ceiling, so headroom is free.
                max_output_tokens=16000,
            )
        except openai.RateLimitError as exc:
            # Must precede _TRANSIENT: a 429 is only worth retrying when it is
            # rate limiting rather than an exhausted balance.
            if _is_quota_exhaustion(exc):
                raise EngineError(
                    f"OpenAI quota exhausted -- add credits at "
                    f"https://platform.openai.com/settings/organization/billing/ "
                    f"({exc})",
                    provider=PROVIDER,
                    retryable=False,
                    error_type="QuotaExhausted",
                    status_code=exc.status_code,
                ) from exc
            raise self._wrap(exc, retryable=True) from exc
        except _TRANSIENT as exc:
            raise self._wrap(exc, retryable=True) from exc
        except _PERMANENT as exc:
            raise self._wrap(exc, retryable=False) from exc
        except openai.APIStatusError as exc:
            # Anything unmapped: treat >=500 as worth another try.
            raise self._wrap(exc, retryable=exc.status_code >= 500) from exc

        parsed = response.output_parsed
        if parsed is None:
            # A refusal or truncation. Retrying the identical prompt will get
            # the identical result, so classify it permanent and let the
            # harness fail over to the other provider instead of sleeping.
            raise EngineError(
                "model returned no parsable structured output (refusal or truncation)",
                provider=PROVIDER,
                retryable=False,
                error_type="EmptyStructuredOutput",
            )
        return parsed

    def _wrap(self, exc: Exception, *, retryable: bool) -> EngineError:
        status = getattr(exc, "status_code", None)
        return EngineError(
            str(exc),
            provider=PROVIDER,
            retryable=retryable,
            error_type=type(exc).__name__,
            status_code=status,
        )

    def _maybe_inject_fault(self) -> None:
        """Raise on demand so the fallback chain can be demonstrated."""
        target = current_fault()
        if target is None:
            return
        if target in (FaultTarget.OPENAI, FaultTarget.BOTH):
            raise EngineError(
                "injected transient fault (force_fail)",
                provider=PROVIDER,
                retryable=True,
                error_type="InjectedFault",
            )
        if target is FaultTarget.OPENAI_PERMANENT:
            raise EngineError(
                "injected permanent fault (force_fail)",
                provider=PROVIDER,
                retryable=False,
                error_type="InjectedFault",
            )
