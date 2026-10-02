"""FastAPI surface for the Jeopardy2 agent.

Routes
------
GET  /                      the browser UI
POST /jeopardy2             ask a question, get a validated AgentResponse
GET  /jeopardy2/stream      same, as Server-Sent Events with live progress
GET  /jeopardy2/memory      what the agent remembers about a user_id
DELETE /jeopardy2/memory    forget one fact, or everything, for a user_id
GET  /jeopardy2/config      effective configuration (secrets redacted)
GET  /health                liveness

Memory: a request carrying `user_id` gets that user's remembered facts in
its prompt, and any new facts the answer proposes are stored for next time.
Without `user_id` the service is stateless. See app/memory.py.

Error policy: this service does not return 5xx. Provider failures come back as
a 200 with `status: "degraded"`, and an unexpected exception is caught by the
handler at the bottom of this module and reported in the same shape. Malformed
input is still a 422 from FastAPI's own validation -- that is a client error
with a precise message, and flattening it into a 200 would hide real bugs.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import retrieval
from app.config import Settings, get_settings
from app.engines.faults import forced_fault
from app.memory import MemoryStore, compose_prompt, get_memory_store
from app.obs import NullTracer, Tracer, build_tracer
from app.obs.harness import traced_run_agent
from app.prompts import get_prompt, prompt_names
from app.schemas import (
    USER_ID_PATTERN,
    AgentResponse,
    AskRequest,
    EngineTrace,
    FaultTarget,
    MemoryItem,
    MemoryReport,
    ProgressEvent,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

#: Built once at startup rather than per request: a tracer owns a background
#: flush thread and an HTTP connection pool, and constructing one per request
#: would leak both. Replaced in tests via `set_tracer`.
_tracer: Tracer = NullTracer("tracer not initialised")


def get_tracer() -> Tracer:
    return _tracer


def set_tracer(tracer: Tracer) -> None:
    """Swap the process-wide tracer. For tests and for lifespan startup."""
    global _tracer
    _tracer = tracer


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
            ("gemini", settings.google_api_key),
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

    # Index state is resolved once, at startup, never inside a request: two
    # workers lazily building on first request would duplicate the work and
    # write to Chroma's SQLite concurrently. `make index` is the build path;
    # startup only reports, so a slow build cannot delay readiness.
    if settings.retrieval_enabled:
        state = retrieval.inspect(settings)
        level = logging.INFO if state.status == "ready" else logging.WARNING
        logger.log(level, "Clue index: %s -- %s", state.status, state.reason)
        for change in state.changes or []:
            logger.warning("  index input changed: %s", change)
    else:
        logger.info("Clue retrieval disabled (RETRIEVAL_ENABLED is false)")

    set_tracer(build_tracer(settings))
    if (reason := get_tracer().is_available()) is not None:
        # Informational, not a warning: tracing off is the default and a
        # perfectly normal way to run this service.
        logger.info("Tracing disabled (%s); prompt_variant=%s", reason, settings.prompt_variant)
    try:
        yield
    finally:
        # Langfuse buffers events and flushes on a timer. Without this, the
        # last few traces of a short-lived container are lost on shutdown --
        # exactly the ones you were watching during a demo.
        get_tracer().flush()


app = FastAPI(
    title="Jeopardy2 Agent",
    version="0.1.0",
    description=(
        "A resilient agent harness: OpenAI, Claude, or Gemini in a configurable "
        "fallback chain, structured Pydantic output, cross-session memory, and no 5xx."
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
        "models": {
            "openai": s.openai_model,
            "anthropic": s.anthropic_model,
            "gemini": s.gemini_model,
        },
        "credentials_present": {
            "openai": s.openai_api_key is not None,
            "anthropic": s.anthropic_api_key is not None,
            "gemini": s.google_api_key is not None,
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
        "prompt": {
            "variant": s.prompt_variant,
            # The digest, not the text: a name can be reused after an edit, a
            # content hash cannot. This is what to compare when two eval runs
            # disagree.
            "digest": get_prompt(s.prompt_variant).digest,
            "available": prompt_names(),
        },
        "tracing": {
            "enabled": s.langfuse_enabled,
            "host": s.langfuse_host,
            # None when tracing is working; otherwise the reason it is not,
            # so "no traces are appearing" is a question the app can answer.
            "unavailable_reason": get_tracer().is_available(),
        },
        "dataset_path": str(s.dataset_path),
        "dataset_present": s.dataset_path.exists(),
        "retrieval": _retrieval_config(s),
        "memory": {
            "db_path": str(s.memory_db_path),
            "max_items_per_user": s.memory_max_items,
            "ttl_days": s.memory_ttl_days,
            "max_new_per_request": s.memory_max_new_per_request,
        },
        "limits": {
            "rate_limit_per_minute": s.rate_limit_per_minute,
            "daily_request_limit": s.daily_request_limit,
            "client_ip_header": s.client_ip_header,
        },
    }


def _retrieval_config(s: Settings) -> dict[str, object]:
    """Index state, so "why is retrieval not working" is self-answerable.

    Plumbing only at this stage: the index is not yet consulted when
    answering. `status` is one of ready / needs_build / stale / no_dataset /
    unavailable.
    """
    if not s.retrieval_enabled:
        return {"enabled": False, "status": "disabled", "reason": "RETRIEVAL_ENABLED is false"}
    state = retrieval.inspect(s)
    return {
        "enabled": True,
        "status": state.status,
        "reason": state.reason,
        "changes": state.changes,
        "index_path": str(s.index_path),
        "chunk_scheme": s.chunk_scheme,
        "embedding_provider": s.embedding_provider,
        "embedding_dimensions": s.embedding_dimensions,
        "wired_into_prompt": False,
    }


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------
@app.post("/jeopardy2", response_model=AgentResponse, tags=["agent"])
async def ask(payload: AskRequest, request: Request) -> AgentResponse:
    """Ask a question and wait for the validated answer.

    Note this can legitimately take a while: with the default schedule a full
    failover sleeps 30s on each provider before giving up. Use
    `/jeopardy2/stream` if you want progress in the meantime.
    """
    settings = get_settings()
    if (refusal := _over_limit(request, settings)) is not None:
        return _limited_response(payload.question, refusal)
    target = _resolve_fault(payload.force_fail, settings)

    final: AgentResponse | None = None
    with forced_fault(target):
        async for item in _answer_with_memory(
            payload.question, payload.user_id, settings, tags=["route:post"]
        ):
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
    user_id: Annotated[str | None, Query(pattern=USER_ID_PATTERN)] = None,
) -> StreamingResponse:
    """Same work as POST /jeopardy2, streamed as Server-Sent Events.

    Emits `progress` events while retrying and a single terminal `result`
    event carrying the full `AgentResponse`. A GET with query parameters (not
    a POST body) so the browser's native `EventSource` can consume it.
    """
    settings = get_settings()
    target = _resolve_fault(force_fail, settings)
    refusal = _over_limit(request, settings)

    async def event_source() -> AsyncIterator[str]:
        if refusal is not None:
            yield _sse("result", _limited_response(question, refusal).model_dump_json())
            return
        try:
            with forced_fault(target):
                async for item in _answer_with_memory(
                    question, user_id, settings, tags=["route:stream"]
                ):
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


async def _answer_with_memory(
    question: str,
    user_id: str | None,
    settings: Settings,
    *,
    tags: list[str],
) -> AsyncIterator[ProgressEvent | AgentResponse]:
    """`traced_run_agent`, with recall before and remembering after.

    Memory is best-effort in both directions: a broken memory file degrades
    to a stateless answer, never to a failed request.
    """
    store = _memory_store(settings) if user_id else None
    recalled: list[str] = []
    if store is not None and user_id is not None:
        try:
            recalled = store.recall(user_id)
        except Exception:  # noqa: BLE001 - memory must not break a request
            logger.exception("memory recall failed; answering without it")
            store = None

    prompt = compose_prompt(question, recalled) if recalled else None
    async for item in traced_run_agent(question, settings, get_tracer(), tags=tags, prompt=prompt):
        if isinstance(item, AgentResponse) and store is not None and user_id is not None:
            report = MemoryReport(recalled=recalled)
            if item.answer is not None and item.answer.remember:
                try:
                    report.saved, report.rejected = store.remember(
                        user_id,
                        item.answer.remember,
                        max_new=settings.memory_max_new_per_request,
                    )
                except Exception:  # noqa: BLE001 - the answer is still good
                    logger.exception("memory write failed; answer returned anyway")
            item.memory = report
        yield item


def _memory_store(settings: Settings) -> MemoryStore:
    return get_memory_store(
        settings.memory_db_path, settings.memory_max_items, settings.memory_ttl_days
    )


# --------------------------------------------------------------------------
# Public-traffic limits
# --------------------------------------------------------------------------
#: Request timestamps per client IP, for the per-minute limit. In-process on
#: purpose: a restart resetting a one-minute window costs nothing, and the
#: deployment runs a single machine. The daily budget, which a restart must
#: not reset, is counted in SQLite instead.
_recent: defaultdict[str, deque[float]] = defaultdict(deque)


def _client_ip(request: Request, settings: Settings) -> str:
    if settings.client_ip_header:
        forwarded = request.headers.get(settings.client_ip_header)
        if forwarded:
            return forwarded.strip()
    return request.client.host if request.client else "unknown"


def _over_limit(request: Request, settings: Settings) -> str | None:
    """Return why this request may not spend model credit, or None if it may."""
    if settings.rate_limit_per_minute > 0:
        now = time.monotonic()
        window = _recent[_client_ip(request, settings)]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= settings.rate_limit_per_minute:
            return "Too many questions from you in the last minute. Wait a moment and try again."
        window.append(now)
    if settings.daily_request_limit > 0:
        try:
            if not _memory_store(settings).count_request(settings.daily_request_limit):
                return "This public demo has reached its daily question limit. Try again tomorrow."
        except Exception:  # noqa: BLE001 - fail closed: no count, no spend
            logger.exception("daily usage counter unavailable")
            return "The demo's usage counter is unavailable, so it is not answering right now."
    return None


def _limited_response(question: str, message: str) -> AgentResponse:
    # Degraded-200, like every other refusal this service makes, so a client
    # has one envelope to parse.
    return AgentResponse(status="degraded", question=question, message=message, trace=EngineTrace())


# --------------------------------------------------------------------------
# Memory management
# --------------------------------------------------------------------------
UserId = Annotated[str, Query(pattern=USER_ID_PATTERN)]


@app.get("/jeopardy2/memory", response_model=list[MemoryItem], tags=["memory"])
async def list_memory(user_id: UserId) -> list[MemoryItem]:
    """Everything remembered for this user_id, oldest first."""
    return _memory_store(get_settings()).list(user_id)


@app.delete("/jeopardy2/memory", tags=["memory"])
async def forget_memory(
    user_id: UserId, memory_id: Annotated[int | None, Query()] = None
) -> dict[str, int]:
    """Forget one fact (`memory_id`) or, without it, everything for this user."""
    return {"deleted": _memory_store(get_settings()).forget(user_id, memory_id)}


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
