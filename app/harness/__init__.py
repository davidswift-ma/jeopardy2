"""The harness: retries, backoff, timeouts, fallback, and degradation.

"The model is the engine, the harness is the car."
"""

from app.harness.orchestrator import run_agent

__all__ = ["run_agent"]
