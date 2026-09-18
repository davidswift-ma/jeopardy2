"""Prompt registry invariants.

The registry exists so the ASCII-rule measurement can be re-run without
hand-editing a constant. These tests protect the two properties that make
that measurement meaningful: the production prompt still carries the rule,
and the control differs from it in exactly one dimension.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.engines.base import SYSTEM_PROMPT
from app.prompts import DEFAULT_PROMPT_NAME, get_prompt, prompt_names


def test_default_is_the_ascii_guarded_prompt():
    assert DEFAULT_PROMPT_NAME == "ascii-guard"
    assert SYSTEM_PROMPT == get_prompt("ascii-guard").text


def test_production_prompt_forbids_non_ascii_punctuation():
    """Load-bearing and expensive to find -- see README. 7/8 -> 0/12."""
    lowered = get_prompt("ascii-guard").text.lower()
    assert "ascii" in lowered
    assert "em dash" in lowered or "em dashes" in lowered
    assert "line break" in lowered


def test_control_differs_only_by_the_rule():
    """The comparison is only valid if one dimension changed.

    If the control were also reworded, a change in corruption rate could not
    be attributed to the rule -- which is the entire claim being tested.
    """
    guarded = get_prompt("ascii-guard").text
    control = get_prompt("no-ascii-rule").text
    assert guarded.startswith(control)
    removed = guarded[len(control) :].lower()
    assert "ascii" in removed
    assert "ascii" not in control.lower()


def test_digests_differ_and_are_stable():
    a = get_prompt("ascii-guard")
    b = get_prompt("no-ascii-rule")
    assert a.digest != b.digest
    assert a.digest == get_prompt("ascii-guard").digest


def test_unknown_prompt_raises_rather_than_falling_back():
    """A silent default would report the production rate under the control's name."""
    with pytest.raises(ValueError, match="unknown prompt variant"):
        get_prompt("does-not-exist")


def test_settings_rejects_unknown_prompt_variant():
    with pytest.raises(ValueError, match="unknown prompt_variant"):
        Settings(prompt_variant="nope")


def test_settings_accepts_every_registered_variant():
    for name in prompt_names():
        assert Settings(prompt_variant=name).prompt_variant == name
