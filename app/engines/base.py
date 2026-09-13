"""Engine protocol and the error type the harness classifies on.

The harness never sees a provider-specific exception. Each adapter catches its
own SDK's typed exceptions and re-raises `EngineError` with `retryable` already
decided, which keeps the retry logic provider-agnostic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.config import Settings
from app.schemas import Answer

SYSTEM_PROMPT = (
    "You are a careful question-answering assistant.\n"
    "Answer the user's question directly and concisely.\n"
    "Report genuine uncertainty in `confidence` rather than overstating it, and "
    "put any assumptions or ambiguities in `caveats`. If the question is "
    "ambiguous, answer the most likely reading and say so in `caveats`.\n"
    "If you do not know, say so plainly in `answer` and set a low confidence "
    "rather than inventing detail."
)


class EngineError(Exception):
    """A provider call failed.

    `retryable` is the harness's whole decision surface: True means "the same
    request might work in 10 seconds" (rate limit, 5xx, timeout, connection
    drop); False means "waiting cannot help" (bad key, unknown model,
    malformed request, schema violation).
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        retryable: bool,
        error_type: str,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.retryable = retryable
        self.error_type = error_type
        self.status_code = status_code

    def __str__(self) -> str:  # pragma: no cover - diagnostic only
        code = f" [{self.status_code}]" if self.status_code is not None else ""
        kind = "retryable" if self.retryable else "permanent"
        return f"{self.provider}: {self.error_type}{code} ({kind}): {self.message}"


@runtime_checkable
class Engine(Protocol):
    """One LLM provider, reduced to the single operation the harness needs."""

    #: Stable identifier used in config, traces, and fault injection.
    name: str
    #: The model ID this engine will call, surfaced for telemetry.
    model: str

    def is_available(self) -> str | None:
        """Return None if usable, else a human-readable reason it is not.

        Used to skip a provider whose API key is missing instead of spending
        the full retry schedule discovering the same thing three times.
        """
        ...

    async def answer(self, question: str) -> Answer:
        """Produce a validated `Answer` or raise `EngineError`."""
        ...


def build_engine(provider: str, settings: Settings) -> Engine:
    """Construct an engine by name. Imports are local to keep startup cheap."""
    if provider == "openai":
        from app.engines.openai_engine import OpenAIEngine

        return OpenAIEngine(settings)
    if provider == "anthropic":
        from app.engines.claude_engine import ClaudeEngine

        return ClaudeEngine(settings)
    raise ValueError(f"unknown provider: {provider}")
