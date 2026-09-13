"""Provider adapters. Each engine turns a question into an `Answer` or raises
an `EngineError` that the harness knows how to classify."""

from app.engines.base import Engine, EngineError, build_engine

__all__ = ["Engine", "EngineError", "build_engine"]
