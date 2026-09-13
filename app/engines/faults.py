"""Deliberate fault injection, so the retry/fallback chain is demonstrable.

Prompt content cannot reliably make a provider fail -- a refusal comes back as
a normal 200 response and never reaches the retry path. So instead of hunting
for magic inputs, a request can name a provider to break via `force_fail`.

The target is held in a `ContextVar`, which is per-task under asyncio: two
concurrent requests with different `force_fail` values do not interfere.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from app.schemas import FaultTarget

_forced_fault: ContextVar[FaultTarget | None] = ContextVar("forced_fault", default=None)


@contextmanager
def forced_fault(target: FaultTarget | None) -> Iterator[None]:
    """Scope a forced-failure target to the current task."""
    token = _forced_fault.set(target)
    try:
        yield
    finally:
        _forced_fault.reset(token)


def current_fault() -> FaultTarget | None:
    return _forced_fault.get()
