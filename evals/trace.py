"""One trace per agent run, stored as one JSONL line.

The record answers the four questions the course asks a trace to answer:

| Field | Question |
|---|---|
| `input` | what was asked |
| `tool_calls`, `observations`, `route` | what context and tools were used |
| `output`, `error` | what came back |
| `human_notes` | what a person thought of it |

Unlike `agents.trace_log.LoopLogger`, which compacts observations to one
readable line, this keeps them **whole**. The grounding check has to prove a
quoted clue came out of a tool result, and a result cut off at 300 characters
can't prove that. The routing log in `docs/runs/` already had that problem:
the Thames clue the agent quoted was past the cut-off, so the log alone
couldn't show whether the quote was real.

Because observations are whole, trace files contain clue text and are
gitignored for the same reason `data/` is. See `.gitignore`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.security import redact_secrets

#: ADK's delegation is itself a tool call; this is its name.
TRANSFER_TOOL = "transfer_to_agent"


def _plain(value: Any) -> Any:
    """Make a tool argument or result JSON-safe, with secrets redacted."""
    try:
        text = json.dumps(value, default=str)
    except (TypeError, ValueError):
        text = json.dumps(str(value))
    return json.loads(redact_secrets(text))


def instructions_digest(texts: list[str]) -> str:
    """A short fingerprint of every instruction in the system.

    Stored on each trace so a before/after comparison can show *which*
    prompts produced which numbers, the same job `app/prompts.py` digests do
    for the HTTP path.
    """
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode())
        h.update(b"\0")
    return h.hexdigest()[:12]


@dataclass
class Trace:
    case_id: str
    input: str
    run_label: str = ""
    trial: int = 1
    recorded_at: str = ""
    model: str = ""
    instructions_digest: str = ""
    route: list[str] = field(default_factory=list)
    delegated_to: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    output: str = ""
    error: dict[str, str] | None = None
    attempts: int = 1
    llm_calls: int = 0
    #: `thoughts` are billed as output but reported separately; Gemini 3.6
    #: Flash spends 60-170 of them even on a one-word reply.
    tokens: dict[str, int] = field(
        default_factory=lambda: {"prompt": 0, "output": 0, "thoughts": 0}
    )
    latency_s: float = 0.0
    human_notes: str = ""

    @property
    def key(self) -> str:
        return f"{self.case_id}#{self.trial}"

    def observation_text(self) -> str:
        """Every tool result in the run, as one searchable string."""
        return "\n".join(json.dumps(o.get("response"), default=str) for o in self.observations)

    def tools_used(self) -> list[str]:
        return [c["name"] for c in self.tool_calls if c["name"] != TRANSFER_TOOL]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trace:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class TraceBuilder:
    """Feed it ADK events in order; it fills in a `Trace`."""

    def __init__(self, trace: Trace) -> None:
        self.trace = trace

    def record(self, event: Any) -> None:
        if getattr(event, "partial", False) is True:
            return  # streaming chunk; the complete event follows
        t = self.trace
        author = getattr(event, "author", None) or "agent"
        if author != "user" and (not t.route or t.route[-1] != author):
            t.route.append(author)

        usage = getattr(event, "usage_metadata", None)
        if usage is not None:
            # One usage block per model response, so this counts LLM calls
            # as well as tokens. Relayed A2A events may carry none.
            t.llm_calls += 1
            t.tokens["prompt"] += getattr(usage, "prompt_token_count", 0) or 0
            t.tokens["output"] += getattr(usage, "candidates_token_count", 0) or 0
            t.tokens["thoughts"] = t.tokens.get("thoughts", 0) + (
                getattr(usage, "thoughts_token_count", 0) or 0
            )

        calls = event.get_function_calls() if hasattr(event, "get_function_calls") else []
        for call in calls or []:
            args = _plain(dict(call.args or {}))
            t.tool_calls.append({"agent": author, "name": call.name, "args": args})
            if call.name == TRANSFER_TOOL and t.delegated_to is None:
                t.delegated_to = args.get("agent_name")

        responses = (
            event.get_function_responses() if hasattr(event, "get_function_responses") else []
        )
        for response in responses or []:
            t.observations.append(
                {"agent": author, "name": response.name, "response": _plain(response.response)}
            )

        if code := getattr(event, "error_code", None):
            t.error = {
                "type": str(code),
                "message": redact_secrets(str(getattr(event, "error_message", "") or ""))[:300],
            }

        is_final = bool(event.is_final_response()) if hasattr(event, "is_final_response") else False
        content = getattr(event, "content", None)
        if is_final and content and getattr(content, "parts", None):
            text = "\n".join(
                p.text for p in content.parts if getattr(p, "text", None) and not _is_thought(p)
            ).strip()
            if text:
                # Last one wins: an A2A hop relays the remote agent's final
                # answer a second time as the router passes it back.
                t.output = redact_secrets(text)

    def fail(self, exc: BaseException) -> None:
        self.trace.error = {
            "type": type(exc).__name__,
            "message": redact_secrets(" ".join(str(exc).split()))[:300],
        }


def _is_thought(part: Any) -> bool:
    return bool(getattr(part, "thought", False))


# --------------------------------------------------------------------------
# JSONL
# --------------------------------------------------------------------------
def write_traces(path: Path, traces: list[Trace]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")


def append_trace(path: Path, trace: Trace) -> None:
    """Written one at a time, so a run that dies halfway keeps what it had."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(trace), ensure_ascii=False) + "\n")


def read_traces(path: Path) -> list[Trace]:
    with path.open(encoding="utf-8") as f:
        return [Trace.from_dict(json.loads(line)) for line in f if line.strip()]
