"""Map ADK's event stream onto Think / Act / Observe.

ADK does not emit events called "think", "act" and "observe" -- it emits a
flat stream of `Event` objects, and the loop structure is implied by what is
attached to each one. The mapping is:

| What the event carries | Loop phase |
|---|---|
| Text, and it is not the final response | **THINK** -- reasoning out loud |
| `get_function_calls()` is non-empty | **ACT** -- proposing a tool call |
| `get_function_responses()` is non-empty | **OBSERVE** -- the tool's real result |
| `is_final_response()` | **ANSWER** -- done |

One event can carry more than one of these (a model often thinks and calls a
tool in the same turn), so `phases_of` returns a list rather than a single
label, in the order they logically occur.

Nothing here prints secrets: only tool names, tool arguments and tool
results are shown, never configuration or credentials.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

#: Fixed width so the phase column lines up in a terminal log.
_WIDTH = 8


@dataclass(frozen=True)
class Phase:
    """One labelled step of the agent loop."""

    label: str  # THINK | ACT | OBSERVE | ANSWER
    author: str
    detail: str

    def render(self) -> str:
        return f"[{self.label:<{_WIDTH}}] {self.author}: {self.detail}"


def _short(value: Any, limit: int = 300) -> str:
    """Compact a tool argument or result to one readable line."""
    try:
        text = value if isinstance(value, str) else json.dumps(value, default=str)
    except (TypeError, ValueError):
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def phases_of(event: Any) -> list[Phase]:
    """Classify one ADK event into zero or more loop phases."""
    author = getattr(event, "author", None) or "agent"
    phases: list[Phase] = []

    calls = event.get_function_calls() if hasattr(event, "get_function_calls") else []
    responses = event.get_function_responses() if hasattr(event, "get_function_responses") else []

    is_final = bool(event.is_final_response()) if hasattr(event, "is_final_response") else False

    # Any text on a non-final event is the model reasoning or narrating.
    text = ""
    content = getattr(event, "content", None)
    if content and getattr(content, "parts", None):
        text = " ".join(p.text for p in content.parts if getattr(p, "text", None)).strip()

    if text and not is_final:
        phases.append(Phase("THINK", author, _short(text)))

    for call in calls:
        args = _short(dict(call.args or {}))
        phases.append(Phase("ACT", author, f"call {call.name}({args})"))

    for response in responses:
        phases.append(Phase("OBSERVE", author, f"{response.name} -> {_short(response.response)}"))

    if is_final and text:
        phases.append(Phase("ANSWER", author, _short(text, limit=600)))

    return phases


class LoopLogger:
    """Prints the Think/Act/Observe trace and remembers it for assertions."""

    def __init__(self, *, echo: bool = True) -> None:
        self.echo = echo
        self.phases: list[Phase] = []

    def record(self, event: Any) -> None:
        for phase in phases_of(event):
            # Drop an exact repeat of the previous phase. An A2A hop emits its
            # final response twice -- once from the remote agent and once as
            # the router relays it -- which read as "ANSWER -> ANSWER" and
            # made the loop summary look like two answers to one question.
            if self.phases and self.phases[-1] == phase:
                continue
            self.phases.append(phase)
            if self.echo:
                print("  " + phase.render())

    @property
    def labels(self) -> list[str]:
        return [p.label for p in self.phases]

    def tool_calls(self) -> list[str]:
        return [p.detail for p in self.phases if p.label == "ACT"]

    def proved_the_loop(self) -> bool:
        """True when a tool was proposed, ran, and an answer followed.

        This is the claim "it is an agent, not a workflow" reduced to
        something checkable: the model chose to call a tool, the tool
        returned a real observation, and the answer came after it.
        """
        labels = self.labels
        try:
            act = labels.index("ACT")
            observe = labels.index("OBSERVE", act)
            answer = labels.index("ANSWER", observe)
        except ValueError:
            return False
        return act < observe < answer

    def summary(self) -> str:
        order = " -> ".join(self.labels) or "(no events)"
        verdict = "LOOP PROVED" if self.proved_the_loop() else "loop NOT proved"
        return f"{order}\n{verdict}"
