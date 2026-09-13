"""FastAPI surface for the Jeopardy2 agent.

Routes
------
GET  /                      the browser UI
POST /jeopardy2             ask a question, get a validated AgentResponse
GET  /jeopardy2/stream      same, as Server-Sent Events with live progress
GET  /jeopardy2/config      effective configuration (secrets redacted)
GET  /health                liveness

Error policy: this service does not return 5xx. Provider failures come back as
a 200 with `status: "degraded"`, and an unexpected exception is caught by the
handler at the bottom of this module and reported in the same shape. Malformed
input is still a 422 from FastAPI's own validation -- that is a client error
with a precise message, and flattening it into a 200 would hide real bugs.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.engines.faults import forced_fault
from app.harness import run_agent
from app.schemas import (
    AgentResponse,
    AskRequest,
    EngineTrace,
    FaultTarget,
    ProgressEvent,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    configured = [
        name
        for name, key in (
            ("openai", settings.openai_api_key),
            ("anthropic", settings.anthropic_api_key),
        )
        if key is not None
    ]
    logger.info(
        "Jeopardy2 starting: provider_order=%s credentials_present=%s attempts=%d backoff=%s",
        settings.provider_order,
        configured or "none",
        settings.max_attempts,
        settings.backoff_seconds,
    )
    if not configured:
        logger.warning(
            "No provider API keys found. Copy .env.example to .env and add at "
            "least one key, or every request will come back degraded."
        )
    yield


app = FastAPI(
    title="Jeopardy2 Agent",
    version="0.1.0",
    description=(
        "A resilient agent harness: OpenAI primary, Claude fallback, "
        "structured Pydantic output, and no 5xx."
    ),
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Health and introspection
# --------------------------------------------------------------------------
@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/jeopardy2/config", tags=["ops"])
async def show_config() -> dict[str, object]:
    """The effective harness configuration, with credentials reduced to booleans."""
    s = get_settings()
    return {
        "provider_order": s.provider_order,
        "models": {"openai": s.openai_model, "anthropic": s.anthropic_model},
        "credentials_present": {
            "openai": s.openai_api_key is not None,
            "anthropic": s.anthropic_api_key is not None,
        },
        "retry": {
            "max_attempts_per_provider": s.max_attempts,
            "backoff_seconds": s.backoff_seconds,
            "classify_errors": s.classify_errors,
            "worst_case_backoff_seconds": sum(
                s.backoff_for(i) for i in range(1, s.max_attempts + 1)
            )
            * len(s.provider_order),
        },
        "request_timeout_seconds": s.request_timeout_seconds,
        "fault_injection_enabled": s.fault_injection_enabled,
        "dataset_path": str(s.dataset_path),
        "dataset_present": s.dataset_path.exists(),
    }


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------
@app.post("/jeopardy2", response_model=AgentResponse, tags=["agent"])
async def ask(payload: AskRequest) -> AgentResponse:
    """Ask a question and wait for the validated answer.

    Note this can legitimately take a while: with the default schedule a full
    failover sleeps 30s on each provider before giving up. Use
    `/jeopardy2/stream` if you want progress in the meantime.
    """
    settings = get_settings()
    target = _resolve_fault(payload.force_fail, settings)

    final: AgentResponse | None = None
    with forced_fault(target):
        async for item in run_agent(payload.question, settings):
            if isinstance(item, AgentResponse):
                final = item

    # run_agent always yields a final AgentResponse; this guards the contract
    # rather than expecting to fire.
    if final is None:  # pragma: no cover - defensive
        return AgentResponse(
            status="degraded",
            question=payload.question,
            message="Harness produced no response.",
            trace=EngineTrace(),
        )
    return final


@app.get("/jeopardy2/stream", tags=["agent"])
async def ask_stream(
    request: Request,
    question: Annotated[str, Query(min_length=1, max_length=4000)],
    force_fail: Annotated[FaultTarget | None, Query()] = None,
) -> StreamingResponse:
    """Same work as POST /jeopardy2, streamed as Server-Sent Events.

    Emits `progress` events while retrying and a single terminal `result`
    event carrying the full `AgentResponse`. A GET with query parameters (not
    a POST body) so the browser's native `EventSource` can consume it.
    """
    settings = get_settings()
    target = _resolve_fault(force_fail, settings)

    async def event_source() -> AsyncIterator[str]:
        try:
            with forced_fault(target):
                async for item in run_agent(question, settings):
                    if await request.is_disconnected():
                        logger.info("client disconnected; abandoning request")
                        return
                    if isinstance(item, ProgressEvent):
                        yield _sse("progress", item.model_dump_json())
                    else:
                        yield _sse("result", item.model_dump_json())
        except Exception:  # noqa: BLE001 - a stream must still terminate cleanly
            logger.exception("streaming request failed")
            yield _sse(
                "result",
                AgentResponse(
                    status="degraded",
                    question=question,
                    message="The harness hit an unexpected internal error.",
                    trace=EngineTrace(),
                ).model_dump_json(),
            )

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Stops nginx-style proxies from buffering the stream into silence.
            "X-Accel-Buffering": "no",
        },
    )


def _resolve_fault(target: FaultTarget | None, settings: Settings) -> FaultTarget | None:
    """Honour `force_fail` only when fault injection is switched on."""
    if target is None:
        return None
    if not settings.fault_injection_enabled:
        logger.info("ignoring force_fail=%s: fault injection disabled", target.value)
        return None
    logger.info("fault injection active: force_fail=%s", target.value)
    return target


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --------------------------------------------------------------------------
# Never return a 500
# --------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Convert any unhandled exception into a degraded 200.

    The full traceback goes to the log; the client gets the same envelope it
    would get from a provider failure so it only has one shape to parse.
    """
    logger.exception("unhandled exception on %s %s", request.method, request.url.path)
    body = AgentResponse(
        status="degraded",
        question="",
        message=f"Unexpected internal error ({type(exc).__name__}). See server logs.",
        trace=EngineTrace(),
    )
    return JSONResponse(status_code=200, content=json.loads(body.model_dump_json()))
