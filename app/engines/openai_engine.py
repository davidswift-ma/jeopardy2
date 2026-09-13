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
                max_output_tokens=4096,
            )
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
