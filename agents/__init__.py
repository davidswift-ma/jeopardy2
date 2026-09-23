"""ADK multi-agent system: router, specialists, MCP, and A2A.

Separate from `app/` on purpose. ADK's `Runner` owns its own orchestration,
so this does not share the retry/backoff/failover machinery in
`app/harness/orchestrator.py`; the only thing the two systems share is
`app/retrieval`. See the README section "Multi-agent system".

Optional: nothing here is imported by the FastAPI app, and the dependencies
live in the `[adk]` extra.
"""
