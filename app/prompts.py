"""Versioned system prompts.

Prompt text is data here, not a literal buried in an engine. The whole claim
behind `scripts/probe_answer_quality.py` is that the ASCII-punctuation rule
was *measured* rather than guessed, and you cannot re-measure what you cannot
swap: reproducing the 7/8 corrupted result means running the prompt *without*
the rule, which until now meant hand-editing a constant and remembering to put
it back.

Each variant is addressable by name, carries its own notes, and hashes to a
short digest so a trace can record exactly which text produced an answer --
the same job Langfuse prompt versioning does, kept in-repo so the suite and
the probe script work with no external service.

`ascii-guard` is production. `no-ascii-rule` exists only to reproduce the
defect that justifies it. Do not make it the default.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# The behavioural core, identical across variants: what to answer and how to
# report uncertainty. Isolating it means a variant differs in exactly one
# dimension, which is the only way the measurement attributes a change in
# corruption rate to the rule rather than to incidental rewording.
_CORE = (
    "You are a careful question-answering assistant.\n"
    "Answer the user's question directly and concisely.\n"
    "Report genuine uncertainty in `confidence` rather than overstating it, and "
    "put any assumptions or ambiguities in `caveats`. If the question is "
    "ambiguous, answer the most likely reading and say so in `caveats`.\n"
    "If you do not know, say so plainly in `answer` and set a low confidence "
    "rather than inventing detail.\n"
)

# Measured, not precautionary: without this rule ~87% of Opus 5 responses
# (7/8) mis-escaped an em dash inside the structured-output JSON, landing as a
# literal "\\u2014", a newline, the word "dash", or a stray quote in the middle
# of a sentence. With it, 0/12. Restricting punctuation to ASCII removes the
# escaping problem at the source.
#
# The bug is Anthropic-specific. gpt-5.5 emits correct curly apostrophes
# (U+2019) and never corrupted anything in 8 trials without this rule, so for
# the OpenAI path the rule is cosmetic -- it just standardizes apostrophes so
# output reads the same whichever engine served it. Do not remove it on the
# grounds that "OpenAI is fine"; Claude is not.
_ASCII_RULE = (
    "Write using only plain ASCII punctuation. Do not use em dashes, en "
    "dashes, curly quotes, ellipsis characters, or any other non-ASCII "
    "symbol. Use commas, periods, semicolons, or parentheses instead. "
    "Never put a line break inside a field value."
)


@dataclass(frozen=True)
class PromptVersion:
    """One named system prompt, identified by content rather than by label.

    `digest` is what belongs in a trace: a name can be reused after an edit,
    a content hash cannot. If two eval runs disagree, comparing digests tells
    you whether you actually ran the same text.
    """

    name: str
    text: str
    notes: str

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:12]


#: The prompt the app ships with. Changing this default changes production.
DEFAULT_PROMPT_NAME = "ascii-guard"

_REGISTRY: dict[str, PromptVersion] = {
    "ascii-guard": PromptVersion(
        name="ascii-guard",
        text=_CORE + _ASCII_RULE,
        notes=(
            "Production. Adds the ASCII-punctuation rule that took Opus 5 "
            "structured-output corruption from 7/8 to 0/12."
        ),
    ),
    "no-ascii-rule": PromptVersion(
        name="no-ascii-rule",
        text=_CORE.rstrip("\n"),
        notes=(
            "The control, for reproducing the defect. Identical to "
            "ascii-guard minus the punctuation rule. Expect a high artifact "
            "rate on claude-opus-5 and a clean run on gpt-5.5; that asymmetry "
            "is the finding, not a bug in the probe."
        ),
    ),
}


def get_prompt(name: str) -> PromptVersion:
    """Look up a prompt variant, failing loudly on a typo.

    A silent fallback to the default would be the worst possible behaviour
    here: an eval run would report the production rate under the control's
    name and you would conclude the rule does nothing.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown prompt variant {name!r}; known: {sorted(_REGISTRY)}") from None


def prompt_names() -> list[str]:
    return sorted(_REGISTRY)
