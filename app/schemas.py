"""Pydantic contracts.

Two distinct families live here, and the split matters:

* `Answer` is the *model-facing* schema. Both providers are asked to emit
  exactly this shape via their structured-output APIs, so every field must be
  required with no defaults -- OpenAI's strict JSON-schema mode rejects
  optional properties, and a schema that works on one provider but not the
  other would defeat the point of having a fallback.

* Everything else is *client-facing* telemetry we construct ourselves, where
  defaults are fine.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------
# Model-facing schema
# --------------------------------------------------------------------------
class Answer(BaseModel):
    """The structured answer we require from whichever engine serves a request.

    Keep this small and provider-neutral. Fields are all required by design;
    see the module docstring.
    """

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="The answer to the user's question, in plain prose.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Self-reported confidence from 0.0 to 1.0.",
    )
    caveats: list[str] = Field(
        description=(
            "Assumptions, ambiguities, or limitations affecting the answer. "
            "Empty list if there are none."
        )
    )


# --------------------------------------------------------------------------
# Request
# --------------------------------------------------------------------------
class FaultTarget(StrEnum):
    """Which provider to deliberately break, for demonstrating the fallback.

    `openai_permanent` raises a non-retryable error, so with error
    classification on you can watch the harness skip the backoff entirely and
    fail straight over to Claude.
    """

    OPENAI = "openai"
    OPENAI_PERMANENT = "openai_permanent"
    ANTHROPIC = "anthropic"
    BOTH = "both"


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)
    force_fail: FaultTarget | None = Field(
        default=None,
        description=(
            "Test hook: force the named provider to fail. Ignored unless "
            "FAULT_INJECTION_ENABLED is true."
        ),
    )


# --------------------------------------------------------------------------
# Telemetry
# --------------------------------------------------------------------------
class AttemptOutcome(StrEnum):
    SUCCESS = "success"
    TRANSIENT_ERROR = "transient_error"
    PERMANENT_ERROR = "permanent_error"
    SKIPPED = "skipped"


class EngineAttempt(BaseModel):
    """One call to one provider. Every try lands here, successes included."""

    provider: str
    model: str
    attempt: int = Field(ge=1, description="1-based attempt number within this provider.")
    outcome: AttemptOutcome
    duration_ms: int
    error_type: str | None = None
    error_message: str | None = None
    slept_before_ms: int = Field(
        default=0,
        description="Backoff actually waited before this attempt.",
    )


class EngineTrace(BaseModel):
    """The whole story of how a request got answered (or didn't)."""

    attempts: list[EngineAttempt] = Field(default_factory=list)
    served_by_provider: str | None = None
    served_by_model: str | None = None
    used_fallback: bool = False
    total_ms: int = 0

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)


class AgentResponse(BaseModel):
    """The endpoint's return value: a validated object, never a bare string.

    `status` is `ok` when an engine produced a valid `Answer`, `degraded` when
    every configured engine failed. A degraded response is still a 200 with
    this same shape -- the service does not emit 5xx (see README).
    """

    status: Literal["ok", "degraded"]
    question: str
    answer: Answer | None = None
    message: str | None = Field(
        default=None,
        description="Human-readable explanation, set when status is 'degraded'.",
    )
    trace: EngineTrace
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# --------------------------------------------------------------------------
# Streaming events (SSE)
# --------------------------------------------------------------------------
class ProgressEvent(BaseModel):
    """A harness progress notification pushed to the browser mid-request.

    These exist because the backoff schedule can hold a request open for a
    minute or more; without them the UI has nothing to show but a spinner.
    """

    event: Literal[
        "started",
        "attempt_started",
        "attempt_failed",
        "waiting",
        "falling_back",
        "provider_skipped",
        "completed",
        "failed",
    ]
    message: str
    provider: str | None = None
    model: str | None = None
    attempt: int | None = None
    retry_in_seconds: float | None = None
