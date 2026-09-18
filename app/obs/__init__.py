"""Observability: tracing, prompt versions, and quality scores.

Optional by construction. With `LANGFUSE_ENABLED=false` (the default), or no
keys, or the SDK not installed, `build_tracer` returns a `NullTracer` whose
`is_available()` explains which of those it was -- and the app behaves
exactly as it did before this package existed.
"""

from app.obs.base import NullTrace, NullTracer, Trace, Tracer
from app.obs.langfuse_tracer import build_tracer

__all__ = ["NullTrace", "NullTracer", "Trace", "Tracer", "build_tracer"]
