"""Anthropic (Claude) engine -- the fallback provider.

Uses `messages.parse`, the Anthropic SDK's structured-output helper, with the
same `Answer` schema the OpenAI engine uses. That shared schema is what makes
the two interchangeable: a client cannot tell which one served a request
except by reading the trace.
"""

from __future__ import annotations

import logging

import anthropic

from app.config import Settings
from app.engines.base import EngineError
from app.engines.faults import current_fault
from app.prompts import get_prompt
from app.schemas import Answer, FaultTarget

logger = logging.getLogger(__name__)

PROVIDER = "anthropic"

_TRANSIENT = (
    anthropic.RateLimitError,  # 429
    anthropic.InternalServerError,  # 5xx, includes 529 overloaded
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
)

_PERMANENT = (
    anthropic.AuthenticationError,  # 401
    anthropic.PermissionDeniedError,  # 403
    anthropic.NotFoundError,  # 404 - unknown model ID
    anthropic.BadRequestError,  # 400
    anthropic.UnprocessableEntityError,  # 422
)


class ClaudeEngine:
    name = PROVIDER

    def __init__(self, settings: Settings, prompt_name: str | None = None) -> None:
        self._settings = settings
        self.model = settings.anthropic_model
        self._client: anthropic.AsyncAnthropic | None = None
        prompt = get_prompt(prompt_name or settings.prompt_variant)
        self.prompt_name = prompt.name
        self.prompt_digest = prompt.digest
        self.system_prompt = prompt.text

    def is_available(self) -> str | None:
        if self._settings.anthropic_api_key is None:
            return "ANTHROPIC_API_KEY is not set"
        return None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            key = self._settings.anthropic_api_key
            self._client = anthropic.AsyncAnthropic(
                api_key=key.get_secret_value() if key else None,
                timeout=self._settings.request_timeout_seconds,
                # See the note in the OpenAI engine: the harness owns retries,
                # so the SDK's built-in retries are switched off.
                max_retries=0,
            )
        return self._client

    async def answer(self, question: str) -> Answer:
        self._maybe_inject_fault()

        client = self._get_client()
        try:
            response = await client.messages.parse(
                model=self.model,
                max_tokens=4096,
                system=self.system_prompt,
                messages=[{"role": "user", "content": question}],
                output_format=Answer,
            )
        except _TRANSIENT as exc:
            raise self._wrap(exc, retryable=True) from exc
        except _PERMANENT as exc:
            raise self._wrap(exc, retryable=False) from exc
        except anthropic.APIStatusError as exc:
            raise self._wrap(exc, retryable=exc.status_code >= 500) from exc

        # A safety decline arrives as HTTP 200 with stop_reason "refusal", not
        # as a raised exception -- checking this before reading content is the
        # difference between a clean failover and an AttributeError.
        if response.stop_reason == "refusal":
            detail = getattr(response, "stop_details", None)
            category = getattr(detail, "category", None) or "unspecified"
            raise EngineError(
                f"model declined the request (category: {category})",
                provider=PROVIDER,
                retryable=False,
                error_type="Refusal",
            )

        parsed = response.parsed_output
        if parsed is None:
            raise EngineError(
                "model returned no parsable structured output",
                provider=PROVIDER,
                retryable=False,
                error_type="EmptyStructuredOutput",
            )
        return parsed

    def _wrap(self, exc: Exception, *, retryable: bool) -> EngineError:
        return EngineError(
            str(exc),
            provider=PROVIDER,
            retryable=retryable,
            error_type=type(exc).__name__,
            status_code=getattr(exc, "status_code", None),
        )

    def _maybe_inject_fault(self) -> None:
        target = current_fault()
        if target in (FaultTarget.ANTHROPIC, FaultTarget.BOTH):
            raise EngineError(
                "injected transient fault (force_fail)",
                provider=PROVIDER,
                retryable=True,
                error_type="InjectedFault",
            )
