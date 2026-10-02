"""Gemini engine -- a third provider, added for the public deployment.

The public URL runs on its own capped Gemini key rather than the OpenAI and
Anthropic keys used in development: a stranger can only spend what that one
key is allowed to. Same `Answer` schema as the other two engines, enforced by
Gemini's structured-output mode.
"""

from __future__ import annotations

import logging

import httpx
from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from app.config import Settings
from app.engines.base import EngineError
from app.prompts import get_prompt
from app.schemas import Answer

#: Built once: the schema is static and every call sends it.
_ANSWER_SCHEMA = Answer.model_json_schema()

logger = logging.getLogger(__name__)

PROVIDER = "gemini"

#: HTTP statuses where waiting might genuinely help. Everything else in the
#: 4xx range (bad key, unknown model, malformed schema) is permanent.
_TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


class GeminiEngine:
    name = PROVIDER

    def __init__(self, settings: Settings, prompt_name: str | None = None) -> None:
        self._settings = settings
        self.model = settings.gemini_model
        self._client: genai.Client | None = None
        prompt = get_prompt(prompt_name or settings.prompt_variant)
        self.prompt_name = prompt.name
        self.prompt_digest = prompt.digest
        self.system_prompt = prompt.text

    def is_available(self) -> str | None:
        if self._settings.google_api_key is None:
            return "GOOGLE_API_KEY is not set"
        return None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            key = self._settings.google_api_key
            self._client = genai.Client(
                api_key=key.get_secret_value() if key else None,
                # Timeout is in milliseconds, unlike the other two SDKs. The
                # SDK wraps every call in its own retry layer, pinned to one
                # attempt here for the same reason the other engines set
                # max_retries=0: the harness owns the retry schedule.
                http_options=types.HttpOptions(
                    timeout=int(self._settings.request_timeout_seconds * 1000),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        return self._client

    async def answer(self, question: str) -> Answer:
        client = self._get_client()
        try:
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=question,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    response_mime_type="application/json",
                    # response_json_schema, not response_schema: the latter
                    # is Gemini's own schema dialect and rejects the
                    # `additionalProperties: false` that OpenAI and Anthropic
                    # strict modes require --
                    #   400 Unknown name "additional_properties" at
                    #   'generation_config.response_schema'
                    # This takes plain JSON Schema, so all three providers
                    # share one schema, as the module docstring promises.
                    response_json_schema=_ANSWER_SCHEMA,
                ),
            )
        except errors.APIError as exc:
            # 429 covers both rate limiting and an exhausted prepaid balance,
            # and unlike OpenAI the body does not reliably say which. Treated
            # as transient; the daily request cap is what bounds spend.
            raise self._wrap(exc, retryable=exc.code in _TRANSIENT_STATUSES) from exc
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise self._wrap(exc, retryable=True) from exc

        text = response.text
        if not text:
            # A safety block or truncation. Same reasoning as the OpenAI
            # engine: the identical prompt gets the identical result.
            raise EngineError(
                "model returned no structured output (blocked or truncated)",
                provider=PROVIDER,
                retryable=False,
                error_type="EmptyStructuredOutput",
            )
        try:
            return Answer.model_validate_json(text)
        except ValidationError as exc:
            raise EngineError(
                f"structured output did not match the Answer schema: {exc}",
                provider=PROVIDER,
                retryable=False,
                error_type="SchemaViolation",
            ) from exc

    def _wrap(self, exc: Exception, *, retryable: bool) -> EngineError:
        return EngineError(
            str(exc),
            provider=PROVIDER,
            retryable=retryable,
            error_type=type(exc).__name__,
            status_code=getattr(exc, "code", None),
        )
