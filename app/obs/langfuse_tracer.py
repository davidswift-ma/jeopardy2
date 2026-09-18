"""Langfuse adapter -- the only module in this repo that imports the SDK.

A caution worth reading before editing. The Langfuse Python SDK changed its
tracing surface substantially between major versions: v2 exposes
`client.trace(...)` returning a stateful object with `.span()`,
`.generation()` and `.score()`, while v3 is OpenTelemetry-based and exposes
`client.start_span(...)` / `.start_generation(...)` with scores created off
the client or the span. Neither is wrong; they are just different.

Rather than pin one and break on the other, the adapter detects which surface
the installed client offers and normalizes both behind `app/obs/base.Trace`.
If a future version offers neither, `is_available()` says so in words instead
of raising -- and because every method here is wrapped in `never_raises`, an
SDK that renames something mid-trace degrades to missing telemetry rather
than a failed request.

**Verify this file against the SDK version you actually install.** It is
deliberately the one place that has to change.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import Settings
from app.obs.base import NullTracer, Trace, never_raises

logger = logging.getLogger(__name__)


class _LangfuseTrace:
    """Normalizes a v2 stateful trace or a v3 root span to one interface."""

    def __init__(self, client: Any, handle: Any, style: str) -> None:
        self._client = client
        self._handle = handle
        self._style = style

    @never_raises
    def span(
        self,
        name: str,
        *,
        input: Any = None,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._style == "v2":
            self._handle.span(name=name, input=input, output=output, metadata=metadata)
            return
        child = self._handle.start_span(name=name, input=input, metadata=metadata)
        child.update(output=output)
        child.end()

    @never_raises
    def generation(
        self,
        name: str,
        *,
        model: str,
        input: Any = None,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._style == "v2":
            self._handle.generation(
                name=name, model=model, input=input, output=output, metadata=metadata
            )
            return
        gen = self._handle.start_generation(name=name, model=model, input=input, metadata=metadata)
        gen.update(output=output)
        gen.end()

    @never_raises
    def score(self, name: str, value: float, *, comment: str | None = None) -> None:
        if self._style == "v2":
            self._handle.score(name=name, value=value, comment=comment)
            return
        # v3 moved scoring around between releases; try the span-level helper
        # first, then the client-level one. Both are no-ops if absent, because
        # a missing score is not worth failing a request over.
        scorer = getattr(self._handle, "score_trace", None) or getattr(self._handle, "score", None)
        if scorer is not None:
            scorer(name=name, value=value, comment=comment)
            return
        create = getattr(self._client, "create_score", None)
        if create is not None:
            create(name=name, value=value, comment=comment)
            return
        logger.debug("no scoring method on this Langfuse client; skipping %r", name)

    @never_raises
    def end(self, *, output: Any = None, metadata: dict[str, Any] | None = None) -> None:
        if self._style == "v2":
            self._handle.update(output=output, metadata=metadata)
            return
        self._handle.update(output=output, metadata=metadata)
        self._handle.end()


class LangfuseTracer:
    """Lazily-constructed Langfuse client behind the `Tracer` protocol."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any | None = None
        self._style: str | None = None
        self._broken: str | None = None

    def is_available(self) -> str | None:
        if not self._settings.langfuse_enabled:
            return "LANGFUSE_ENABLED is false"
        missing = [
            env
            for env, value in (
                ("LANGFUSE_PUBLIC_KEY", self._settings.langfuse_public_key),
                ("LANGFUSE_SECRET_KEY", self._settings.langfuse_secret_key),
            )
            if value is None
        ]
        if missing:
            return f"{' and '.join(missing)} not set"
        if self._broken:
            return self._broken
        return None

    def _get_client(self) -> Any | None:
        if self._client is not None or self._broken:
            return self._client
        try:
            from langfuse import Langfuse  # noqa: PLC0415 - optional dependency
        except ImportError:
            self._broken = "langfuse is not installed (pip install 'jeopardy2[obs]')"
            logger.warning("%s; tracing disabled", self._broken)
            return None

        pk = self._settings.langfuse_public_key
        sk = self._settings.langfuse_secret_key
        try:
            client = Langfuse(
                public_key=pk.get_secret_value() if pk else None,
                secret_key=sk.get_secret_value() if sk else None,
                host=self._settings.langfuse_host,
            )
        except Exception as exc:  # noqa: BLE001 - never let init break startup
            self._broken = f"Langfuse client init failed: {type(exc).__name__}: {exc}"
            logger.warning("%s; tracing disabled", self._broken)
            return None

        if hasattr(client, "trace"):
            self._style = "v2"
        elif hasattr(client, "start_span"):
            self._style = "v3"
        else:
            self._broken = (
                "installed langfuse client exposes neither .trace() (v2) nor "
                ".start_span() (v3); update app/obs/langfuse_tracer.py"
            )
            logger.warning("%s; tracing disabled", self._broken)
            return None

        logger.info(
            "Langfuse tracing enabled: host=%s sdk_surface=%s",
            self._settings.langfuse_host,
            self._style,
        )
        self._client = client
        return client

    def trace(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> Trace:
        client = self._get_client()
        if client is None:
            from app.obs.base import NullTrace  # noqa: PLC0415 - avoids a cycle at import

            return NullTrace()
        try:
            if self._style == "v2":
                handle = client.trace(name=name, input=input, metadata=metadata, tags=tags)
            else:
                handle = client.start_span(
                    name=name, input=input, metadata={**(metadata or {}), "tags": tags}
                )
        except Exception:  # noqa: BLE001 - degrade to no telemetry, not no answer
            logger.warning("could not start Langfuse trace %r", name, exc_info=True)
            from app.obs.base import NullTrace  # noqa: PLC0415

            return NullTrace()
        return _LangfuseTrace(client, handle, self._style or "v3")

    @never_raises
    def flush(self) -> None:
        if self._client is not None:
            self._client.flush()


def build_tracer(settings: Settings) -> Any:
    """Return a usable tracer, or a `NullTracer` carrying the reason why not."""
    tracer = LangfuseTracer(settings)
    reason = tracer.is_available()
    if reason is not None:
        return NullTracer(reason)
    return tracer
