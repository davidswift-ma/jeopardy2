"""Multi-agent system structure, and the MCP server's read-only guarantee.

No Google API key and no network. Building an ADK `Agent` does not call the
model, so the graph's shape -- who routes to whom, who has which tools -- is
testable offline. The MCP server is exercised directly as Python functions;
its behaviour over the wire was verified separately.
"""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("google.adk", reason="needs the [adk] extra")

from agents.clue_mcp_server import describe_clues, query_clues  # noqa: E402
from agents.judge_agent import judge_response  # noqa: E402
from agents.system import build_router  # noqa: E402
from app.config import Settings  # noqa: E402


@pytest.fixture
def adk_settings(tmp_path):
    return Settings(
        google_api_key="test-key",
        retrieval_enabled=True,
        index_path=tmp_path / "chroma",
        clues_db_path=tmp_path / "clues.sqlite3",
    )


@pytest.fixture
def clues_db(tmp_path, monkeypatch):
    """A tiny real SQLite clue archive, wired into the MCP server's settings."""
    path = tmp_path / "clues.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE clues (id TEXT PRIMARY KEY, clue_text TEXT, correct_response TEXT,"
        " category TEXT, round TEXT, clue_value INTEGER, air_date TEXT);"
    )
    conn.executemany(
        "INSERT INTO clues VALUES (?,?,?,?,?,?,?)",
        [
            ("a", "River in the Bible", "the Jordan", "GEOGRAPHY", "Jeopardy", 100, "1984-09-10"),
            (
                "b",
                "Chinese sweet from America",
                "the fortune cookie",
                "FOOD",
                "Jeopardy",
                200,
                "1984-09-12",
            ),
        ],
    )
    conn.commit()
    conn.close()

    import agents.clue_mcp_server as mod

    monkeypatch.setattr(mod, "_settings", lambda: Settings(clues_db_path=path))
    return path


# --------------------------------------------------------------------------
# Router / specialist graph
# --------------------------------------------------------------------------
def test_router_has_no_tools_only_sub_agents(adk_settings):
    """A router is defined by delegating, not by doing."""
    router = build_router(adk_settings)
    assert not router.tools
    assert len(router.sub_agents) == 4


def test_all_four_specialists_are_present(adk_settings):
    """The homework asks for 3+ specialists; these are the four."""
    router = build_router(adk_settings)
    assert {a.name for a in router.sub_agents} == {
        "clue_search_agent",
        "clue_stats_agent",
        "general_agent",
        "judge_agent",
    }


def test_every_specialist_has_a_description(adk_settings):
    """Routing is decided on `description`. A missing one is unroutable."""
    router = build_router(adk_settings)
    for agent in router.sub_agents:
        assert agent.description, f"{agent.name} has no description"


def test_remote_judge_is_a_different_type_but_the_same_slot(adk_settings):
    """The A2A point: the router cannot tell local from remote."""
    from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

    router = build_router(adk_settings)
    by_name = {a.name: a for a in router.sub_agents}
    assert isinstance(by_name["judge_agent"], RemoteA2aAgent)
    assert not isinstance(by_name["general_agent"], RemoteA2aAgent)


def test_general_agent_has_no_tools(adk_settings):
    """It exists so non-Jeopardy questions never reach a clue specialist."""
    router = build_router(adk_settings)
    general = next(a for a in router.sub_agents if a.name == "general_agent")
    assert not general.tools


def test_subsets_can_be_built_without_mcp_or_a2a(adk_settings):
    """Routing must be demonstrable without a subprocess or a second server."""
    router = build_router(adk_settings, include_mcp=False, include_a2a=False)
    assert {a.name for a in router.sub_agents} == {"clue_search_agent", "general_agent"}


# --------------------------------------------------------------------------
# MCP server
# --------------------------------------------------------------------------
def test_query_returns_rows(clues_db):
    result = query_clues("SELECT correct_response FROM clues WHERE category = 'GEOGRAPHY'")
    assert result["rows"] == [{"correct_response": "the Jordan"}]
    assert result["truncated"] is False


def test_describe_reports_schema_and_counts(clues_db):
    result = describe_clues()
    assert result["clues"] == 2
    assert {c["name"] for c in result["columns"]} >= {"clue_text", "correct_response"}


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE clues",
        "DELETE FROM clues",
        "UPDATE clues SET clue_text = 'x'",
        "INSERT INTO clues VALUES ('c','x','y','z','1',1,'2000-01-01')",
        "PRAGMA table_info(clues)",
    ],
)
def test_write_statements_are_refused(clues_db, sql):
    """The model authors this SQL. It must not be able to mutate anything."""
    assert "error" in query_clues(sql)


def test_second_statement_is_refused(clues_db):
    assert "error" in query_clues("SELECT 1; DROP TABLE clues")


def test_the_table_survives_an_attack(clues_db):
    query_clues("DROP TABLE clues")
    assert describe_clues()["clues"] == 2


def test_bad_sql_returns_an_error_the_model_can_read(clues_db):
    """Handed back as data so the model can correct itself, not raised."""
    result = query_clues("SELECT * FROM does_not_exist")
    assert "no such table" in result["error"]


def test_missing_database_is_reported_not_raised(tmp_path, monkeypatch):
    import agents.clue_mcp_server as mod

    monkeypatch.setattr(mod, "_settings", lambda: Settings(clues_db_path=tmp_path / "absent.db"))
    assert "make clues-db" in query_clues("SELECT 1")["error"]


# --------------------------------------------------------------------------
# Judge tool
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "expected,given,match_type",
    [
        ("the Jordan", "Jordan", "exact"),  # article stripped both sides
        ("the Jordan", "the Jordan", "exact"),
        ("the Jordan", "Jordan River", "substring"),
        ("the Jordan", "the Nile", "different"),
        ("the Jordan", "", "empty"),
    ],
)
def test_judge_classifies_response_forms(expected, given, match_type):
    """Jeopardy is lenient on form, strict on substance."""
    result = judge_response("a clue", expected, given)
    assert result["match_type"] == match_type
