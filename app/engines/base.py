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
    "rather than inventing detail.\n"
    # Measured, not precautionary: without this rule ~87% of Opus 5 responses
    # (7/8) mis-escaped an em dash inside the structured-output JSON, landing
    # as a literal "\\u2014", a newline, the word "dash", or a stray quote in
    # the middle of a sentence. With it, 0/12. Restricting punctuation to ASCII
    # removes the escaping problem at the source.
    #
    # The bug is Anthropic-specific. gpt-5.5 emits correct curly apostrophes
    # (U+2019) and never corrupted anything in 8 trials without this rule, so
    # for the OpenAI path the rule is cosmetic -- it just standardizes
    # apostrophes so output reads the same whichever engine served it. Do not
    # remove it on the grounds that "OpenAI is fine"; Claude is not.
    # See scripts/probe_answer_quality.py.
    "Write using only plain ASCII punctuation. Do not use em dashes, en "
    "dashes, curly quotes, ellipsis characters, or any other non-ASCII "
    "symbol. Use commas, periods, semicolons, or parentheses instead. "
    "Never put a line break inside a field value."
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
