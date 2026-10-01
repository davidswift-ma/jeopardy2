"""Evals for the ADK multi-agent system: record traces, check them, show them.

    make eval-record LABEL=baseline   # costs money: runs every case live
    make eval TRACES=evals/traces/baseline.jsonl
    make eval-dashboard

Recording and checking are separate on purpose. A trace file is recorded once
and can be re-checked for free as often as you like, which is what lets a new
check be run against *old* behaviour, and what makes "the same set" in a
before/after comparison literally the same set of inputs.
"""
