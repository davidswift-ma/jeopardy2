"""Tracer protocol and the no-op implementation.

Why a protocol rather than calling Langfuse directly, exactly as with the
provider SDKs: the call sites should not know which observability backend is
attached, or whether one is attached at all. `app/obs/langfuse_tracer.py` is
the only module that imports the vendor SDK, so a breaking change there is a
one-file fix.

The hard rule in this package: **telemetry must never affect a response.** A
missing key, an unreachable collector, or a vendor SDK that renamed a method
all degrade to a no-op plus one log line. The service promises never to
return a 5xx (see README); an observability backend is not permitted to be
the thing that breaks that promise.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any, Protocol

logger = logging.getLogger(__name__)


def never_raises[F: Callable[..., Any]](func: F) -> F:
    """Swallow and log any exception from a telemetry call.

    Deliberately broad. The alternative -- letting an ingestion error
    propagate -- converts a monitoring outage into a user-visible failure,
    which is precisely backwards. Logged at warning with the traceback so the
    failure is still discoverable.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception:  # noqa: BLE001 - telemetry must not break the request
            logger.warning("telemetry call %s failed; continuing", func.__name__, exc_info=True)
            return None

    return wrapper  # type: ignore[return-value]


class Trace(Protocol):
    """One unit of observed work, typically one HTTP request or one eval trial."""

    def span(
        self,
        name: str,
        *,
        input: Any = None,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a completed sub-step (a retry attempt, a retrieval call)."""
        ...

    def generation(
        self,
        name: str,
        *,
        model: str,
        input: Any = None,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a model call specifically, so token/cost views can pick it up."""
        ...

    def score(self, name: str, value: float, *, comment: str | None = None) -> None:
        """Attach a numeric quality score to this trace."""
        ...

    def end(self, *, output: Any = None, metadata: dict[str, Any] | None = None) -> None:
        """Close the trace, optionally recording its final output."""
        ...


class Tracer(Protocol):
    def is_available(self) -> str | None:
        """None if usable, else a human-readable reason it is not.

        Same contract as `Engine.is_available` -- the app reports what is
        missing rather than crashing or pretending it is fine.
        """
        ...

    def trace(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> Trace: ...

    def flush(self) -> None:
        """Push buffered events. Call on shutdown and at the end of scripts."""
        ...


class NullTrace:
    """Accepts every call and does nothing."""

    def span(self, name: str, **kwargs: Any) -> None:
        return None

    def generation(self, name: str, **kwargs: Any) -> None:
        return None

    def score(self, name: str, value: float, **kwargs: Any) -> None:
        return None

    def end(self, **kwargs: Any) -> None:
        return None


class NullTracer:
    """The tracer used when observability is off, unconfigured, or broken.

    `reason` is surfaced by /jeopardy2/config so "no traces are appearing" is
    a question the app can answer about itself.
    """

    def __init__(self, reason: str = "tracing is disabled") -> None:
        self._reason = reason

    def is_available(self) -> str | None:
        return self._reason

    def trace(self, name: str, **kwargs: Any) -> Trace:
        return NullTrace()

    def flush(self) -> None:
        return None
