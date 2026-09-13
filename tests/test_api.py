"""Endpoint behaviour: response shape, SSE framing, and the no-5xx promise."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import main
from app.harness import orchestrator
from tests.conftest import FakeEngine, ok_answer, transient


@pytest.fixture
def client(monkeypatch, fast_settings):
    """A TestClient whose harness uses scriptable fake engines."""
    monkeypatch.setattr(main, "get_settings", lambda: fast_settings)

    def fake_build(provider: str, settings):
        if provider == "openai":
            return FakeEngine("openai", "gpt-fake", [ok_answer("Paris")])
        return FakeEngine("anthropic", "claude-fake", [ok_answer("Paris (claude)")])

    monkeypatch.setattr(orchestrator, "build_engine", fake_build)
    with TestClient(main.app) as c:
        yield c


def test_ask_returns_validated_object_not_a_string(client):
    r = client.post("/jeopardy2", json={"question": "Capital of France?"})

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # The answer is a nested object with typed fields, not a bare string.
    assert isinstance(body["answer"], dict)
    assert body["answer"]["answer"] == "Paris"
    assert 0.0 <= body["answer"]["confidence"] <= 1.0
    assert isinstance(body["answer"]["caveats"], list)
    assert body["trace"]["served_by_provider"] == "openai"
    assert body["trace"]["used_fallback"] is False


def test_trace_is_present_and_shaped(client):
    body = client.post("/jeopardy2", json={"question": "hi"}).json()
    attempt = body["trace"]["attempts"][0]

    assert attempt["provider"] == "openai"
    assert attempt["model"] == "gpt-fake"
    assert attempt["outcome"] == "success"
    assert attempt["attempt"] == 1
    assert "duration_ms" in attempt


def test_blank_question_is_rejected(client):
    assert client.post("/jeopardy2", json={"question": ""}).status_code == 422


def test_unknown_field_is_rejected(client):
    r = client.post("/jeopardy2", json={"question": "hi", "temperature": 0.5})
    assert r.status_code == 422


def test_forced_failure_drives_the_real_engine_hook(monkeypatch, fast_settings):
    """force_fail must drive the real OpenAI engine's injection hook, not a stub.

    Anthropic's key is cleared so the fallback is skipped as unavailable,
    keeping this test entirely offline: the OpenAI engine raises before it
    constructs a client, so no socket is opened either.
    """
    settings = fast_settings.model_copy(update={"anthropic_api_key": None})
    monkeypatch.setattr(main, "get_settings", lambda: settings)

    with TestClient(main.app) as c:
        r = c.post("/jeopardy2", json={"question": "hi", "force_fail": "openai"})

    body = r.json()
    assert r.status_code == 200
    assert body["status"] == "degraded"

    openai_attempts = [a for a in body["trace"]["attempts"] if a["provider"] == "openai"]
    assert len(openai_attempts) == 3, "injected transient fault should use every attempt"
    assert all(a["error_type"] == "InjectedFault" for a in openai_attempts)
    assert all(a["outcome"] == "transient_error" for a in openai_attempts)

    skipped = [a for a in body["trace"]["attempts"] if a["provider"] == "anthropic"]
    assert [a["outcome"] for a in skipped] == ["skipped"]


def test_forced_permanent_failure_skips_the_backoff(monkeypatch, fast_settings):
    """The permanent variant should cost exactly one attempt, not three."""
    settings = fast_settings.model_copy(update={"anthropic_api_key": None})
    monkeypatch.setattr(main, "get_settings", lambda: settings)

    with TestClient(main.app) as c:
        body = c.post(
            "/jeopardy2", json={"question": "hi", "force_fail": "openai_permanent"}
        ).json()

    openai_attempts = [a for a in body["trace"]["attempts"] if a["provider"] == "openai"]
    assert len(openai_attempts) == 1
    assert openai_attempts[0]["outcome"] == "permanent_error"


def test_fault_injection_ignored_when_disabled(monkeypatch, fast_settings):
    settings = fast_settings.model_copy(update={"fault_injection_enabled": False})
    monkeypatch.setattr(main, "get_settings", lambda: settings)

    def fake_build(provider, _settings):
        return FakeEngine(provider, f"{provider}-fake", [ok_answer("unaffected")])

    monkeypatch.setattr(orchestrator, "build_engine", fake_build)

    with TestClient(main.app) as c:
        body = c.post("/jeopardy2", json={"question": "hi", "force_fail": "both"}).json()

    assert body["status"] == "ok"
    assert body["answer"]["answer"] == "unaffected"


def test_never_returns_500_on_unexpected_error(monkeypatch, fast_settings):
    """A genuine bug still yields the standard envelope with a 200."""
    monkeypatch.setattr(main, "get_settings", lambda: fast_settings)

    def exploding_build(provider, settings):
        raise RuntimeError("catastrophic wiring failure")

    monkeypatch.setattr(orchestrator, "build_engine", exploding_build)

    with TestClient(main.app, raise_server_exceptions=False) as c:
        r = c.post("/jeopardy2", json={"question": "hi"})

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert "Unexpected internal error" in body["message"]


def test_degraded_response_keeps_the_same_envelope(monkeypatch, fast_settings):
    monkeypatch.setattr(main, "get_settings", lambda: fast_settings)
    monkeypatch.setattr(
        orchestrator,
        "build_engine",
        lambda p, s: FakeEngine(p, f"{p}-fake", [transient(p)]),
    )

    with TestClient(main.app) as c:
        body = c.post("/jeopardy2", json={"question": "hi"}).json()

    assert body["status"] == "degraded"
    assert body["answer"] is None
    assert body["message"]
    assert body["trace"]["served_by_provider"] is None
    assert len(body["trace"]["attempts"]) == 6


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------
def _parse_sse(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        name = data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        if name and data:
            out.append((name, json.loads(data)))
    return out


def test_stream_emits_progress_then_a_single_result(client):
    with client.stream("GET", "/jeopardy2/stream?question=hello") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse("".join(r.iter_text()))

    names = [n for n, _ in events]
    assert names.count("result") == 1
    assert names[-1] == "result", "the result event must terminate the stream"
    assert "progress" in names

    _, result = events[-1]
    assert result["status"] == "ok"
    assert result["answer"]["answer"] == "Paris"


def test_stream_narrates_retries_and_fallback(monkeypatch, fast_settings):
    monkeypatch.setattr(main, "get_settings", lambda: fast_settings)

    def fake_build(provider, settings):
        if provider == "openai":
            return FakeEngine("openai", "gpt-fake", [transient("openai")])
        return FakeEngine("anthropic", "claude-fake", [ok_answer("fallback answer")])

    monkeypatch.setattr(orchestrator, "build_engine", fake_build)

    with TestClient(main.app) as c:
        with c.stream("GET", "/jeopardy2/stream?question=hi") as r:
            events = _parse_sse("".join(r.iter_text()))

    progress = [d for n, d in events if n == "progress"]
    assert [p["event"] for p in progress].count("attempt_failed") == 3
    assert any(p["event"] == "falling_back" for p in progress)
    assert any("Retrying openai" in p["message"] for p in progress)

    _, result = events[-1]
    assert result["status"] == "ok"
    assert result["trace"]["used_fallback"] is True
    assert result["answer"]["answer"] == "fallback answer"


def test_stream_rejects_a_blank_question(client):
    assert client.get("/jeopardy2/stream?question=").status_code == 422


# --------------------------------------------------------------------------
# Ops endpoints
# --------------------------------------------------------------------------
def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_config_endpoint_redacts_secrets(client):
    body = client.get("/jeopardy2/config").json()
    serialized = json.dumps(body)

    assert body["credentials_present"] == {"openai": True, "anthropic": True}
    assert "test-openai-key" not in serialized
    assert "test-anthropic-key" not in serialized
    assert body["retry"]["max_attempts_per_provider"] == 3


def test_config_reports_worst_case_wait():
    """The documented 30s-per-provider figure comes from config, not a comment."""
    from app.config import Settings

    s = Settings(max_attempts=3, backoff_seconds=[10, 20], provider_order=["openai", "anthropic"])
    per_provider = sum(s.backoff_for(i) for i in range(1, s.max_attempts + 1))

    assert per_provider == 30.0
    assert per_provider * len(s.provider_order) == 60.0


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Jeopardy" in r.text


def test_openapi_documents_the_endpoint(client):
    spec = client.get("/openapi.json").json()
    assert "/jeopardy2" in spec["paths"]
    assert "/jeopardy2/stream" in spec["paths"]
