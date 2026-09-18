"""Shared fixtures.

Tests must never sleep the real 10s/20s schedule, so `fast_settings` keeps the
attempt counts and classification logic identical but shrinks the backoff to
milliseconds. The code path under test is the same one production runs; only
the numbers differ.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.engines.base import EngineError
from app.schemas import Answer


@pytest.fixture
def fast_settings() -> Settings:
    return Settings(
        openai_api_key="test-openai-key",
        anthropic_api_key="test-anthropic-key",
        max_attempts=3,
        backoff_seconds=[0.001, 0.002],
        provider_order=["openai", "anthropic"],
        classify_errors=True,
        fault_injection_enabled=True,
    )


class FakeEngine:
    """A scriptable engine.

    `script` is consumed one entry per call: an `Answer` is returned, an
    `EngineError` is raised, and any other exception is raised as-is (to
    exercise the unexpected-error path).
    """

    def __init__(
        self,
        name: str,
        model: str = "fake-model",
        script: list[object] | None = None,
        unavailable: str | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.script: list[object] = script or []
        self._unavailable = unavailable
        self.calls = 0

    def is_available(self) -> str | None:
        return self._unavailable

    async def answer(self, question: str) -> Answer:
        self.calls += 1
        step = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(step, Exception):
            raise step
        assert isinstance(step, Answer)
        return step


class RecordingTrace:
    """Captures what a tracer was asked to record, for assertions."""

    def __init__(self, name: str, input: object, metadata: dict | None, tags: list | None):
        self.name = name
        self.input = input
        self.metadata = metadata or {}
        self.tags = tags or []
        self.spans: list[dict] = []
        self.generations: list[dict] = []
        self.scores: dict[str, float] = {}
        self.score_comments: dict[str, str | None] = {}
        self.ended = False
        self.end_output: object = None
        self.end_metadata: dict = {}

    def span(self, name: str, **kw: object) -> None:
        self.spans.append({"name": name, **kw})

    def generation(self, name: str, **kw: object) -> None:
        self.generations.append({"name": name, **kw})

    def score(self, name: str, value: float, *, comment: str | None = None) -> None:
        self.scores[name] = value
        self.score_comments[name] = comment

    def end(self, *, output: object = None, metadata: dict | None = None) -> None:
        self.ended = True
        self.end_output = output
        self.end_metadata = metadata or {}


class RecordingTracer:
    """An in-memory tracer. Keeps the suite offline and assertions concrete."""

    def __init__(self) -> None:
        self.traces: list[RecordingTrace] = []
        self.flushed = 0

    def is_available(self) -> str | None:
        return None

    def trace(self, name: str, *, input=None, metadata=None, tags=None) -> RecordingTrace:
        t = RecordingTrace(name, input, metadata, tags)
        self.traces.append(t)
        return t

    def flush(self) -> None:
        self.flushed += 1


class ExplodingTracer:
    """Fails on every call, to prove telemetry cannot break a request."""

    def is_available(self) -> str | None:
        return None

    def trace(self, *a: object, **kw: object) -> object:
        raise RuntimeError("collector is down")

    def flush(self) -> None:
        raise RuntimeError("collector is down")


def ok_answer(text: str = "42") -> Answer:
    return Answer(answer=text, confidence=0.9, caveats=[])


def transient(provider: str, kind: str = "RateLimitError") -> EngineError:
    return EngineError(
        "rate limited", provider=provider, retryable=True, error_type=kind, status_code=429
    )


def permanent(provider: str, kind: str = "AuthenticationError") -> EngineError:
    return EngineError(
        "bad key", provider=provider, retryable=False, error_type=kind, status_code=401
    )
