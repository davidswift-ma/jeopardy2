"""Streamlit page: run the eval suite on recorded traces, show pass/fail.

    make eval-dashboard        # http://localhost:8501

The page runs the real pytest suite (`evals/test_traces.py`) in a
subprocess and reads its JUnit report, so the numbers on this page are
pytest's numbers, not a second implementation of them that could drift.

Pick a second file to compare two recordings of the same cases, e.g. before
and after a prompt fix. Human notes typed here are saved into the trace
file's `human_notes` field.

Nothing on this page calls a model. Recording costs money and is done from
the command line (`make eval-record`) so it can't happen by accident.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.checks import CATEGORIES, CHECKS, load_cases  # noqa: E402
from evals.trace import read_traces, write_traces  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACES_DIR = REPO_ROOT / "evals" / "traces"
SUITE = REPO_ROOT / "evals" / "test_traces.py"
ORDER = [c.__name__ for c in CHECKS]


@st.cache_data(show_spinner=False)
def run_suite(path: str, mtime: float) -> list[dict]:
    """Run pytest on one trace file; one dict per test. `mtime` busts the cache."""
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "junit.xml"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(SUITE),
                "-q",
                "-p",
                "no:cacheprovider",
                f"--junitxml={report}",
            ],
            cwd=REPO_ROOT,
            env={**os.environ, "EVAL_TRACES": path},
            capture_output=True,
            check=False,  # failing checks are the point, not an error
        )
        if not report.exists():
            return []
        tests = []
        for tc in ET.parse(report).getroot().iter("testcase"):
            ident = tc.get("name", "").removeprefix("test_check[").removesuffix("]")
            if "::" not in ident:
                continue
            key, check = ident.rsplit("::", 1)
            failure = tc.find("failure")
            tests.append(
                {
                    "key": key,
                    "check": check,
                    "passed": failure is None,
                    "reason": (failure.get("message") or "").removeprefix("Failed: ")
                    if failure is not None
                    else "",
                }
            )
        return tests


def per_check(tests: list[dict]) -> dict[str, Counter]:
    out: dict[str, Counter] = defaultdict(Counter)
    for t in tests:
        out[t["check"]]["pass" if t["passed"] else "fail"] += 1
    return out


def rate(c: Counter | None) -> str:
    if not c:
        return "-"
    n = c["pass"] + c["fail"]
    return f"{c['pass']}/{n} ({100 * c['pass'] / n:.0f}%)"


# --------------------------------------------------------------------------
st.set_page_config(page_title="Jeopardy agent evals", layout="wide")
st.title("Jeopardy multi-agent evals")
st.caption(
    "Binary checks over recorded traces. Every failure has a one-line reason. "
    "Nothing here calls a model."
)

files = sorted(TRACES_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
if not files:
    st.warning("No trace files yet. Record some with `make eval-record LABEL=baseline`.")
    st.stop()

names = [f.stem for f in files]
with st.sidebar:
    primary = st.selectbox("Trace file", names, index=len(names) - 1)
    compare = st.selectbox("Compare against (optional)", ["(none)"] + names, index=0)
    st.markdown("---")
    st.markdown("**Record new traces** (costs money):")
    st.code("make eval-record LABEL=<name>", language="bash")

path = TRACES_DIR / f"{primary}.jsonl"
compare_path = None if compare == "(none)" else TRACES_DIR / f"{compare}.jsonl"

# The suite is free and takes about a second, so it runs on load. The
# button is for after a trace file changed under a running page.
if st.button("Re-run suite"):
    run_suite.clear()

with st.spinner("Running pytest..."):
    tests = run_suite(str(path), path.stat().st_mtime)
    baseline = run_suite(str(compare_path), compare_path.stat().st_mtime) if compare_path else None

if not tests:
    st.error("The suite produced no results. Run `make eval` in a terminal to see why.")
    st.stop()

passed = sum(t["passed"] for t in tests)
failed = len(tests) - passed
traces = read_traces(path)
clean = len(traces) - len({t["key"] for t in tests if not t["passed"]})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Checks passed", passed)
c2.metric("Checks failed", failed)
c3.metric("Pass rate", f"{100 * passed / len(tests):.0f}%")
c4.metric("Traces with no failure", f"{clean}/{len(traces)}")
if baseline is not None:
    b_pass = sum(t["passed"] for t in baseline)
    st.caption(
        f"Compared with **{compare}**: {b_pass}/{len(baseline)} checks passed "
        f"({100 * b_pass / len(baseline):.0f}%)."
    )

st.subheader("By failure category")
mine = per_check(tests)
theirs = per_check(baseline) if baseline is not None else {}
table = []
for check in ORDER:
    if check not in mine and check not in theirs:
        continue
    row = {
        "category": CATEGORIES[check],
        "check": check,
        "pass": mine.get(check, Counter())["pass"],
        "fail": mine.get(check, Counter())["fail"],
        primary: rate(mine.get(check)),
    }
    if baseline is not None:
        row[compare] = rate(theirs.get(check))
    table.append(row)
st.dataframe(table, hide_index=True, width="stretch")

st.subheader(f"Failures ({failed})")
fails = [t for t in tests if not t["passed"]]
if not fails:
    st.success("Every check passed.")
else:
    st.dataframe(
        [{"trace": t["key"], "check": t["check"], "reason": t["reason"]} for t in fails],
        hide_index=True,
        width="stretch",
    )

st.subheader("Inspect a trace")
cases = load_cases()
by_key = {t.key: t for t in traces}
failing_keys = sorted({t["key"] for t in fails})
only_failing = st.checkbox("Only traces with a failure", value=bool(failing_keys))
keys = failing_keys if only_failing else sorted(by_key)
if keys:
    key = st.selectbox("Trace", keys)
    tr = by_key[key]
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Input:** {tr.input}")
        st.markdown(
            f"**Expected route:** {' or '.join(cases.get(tr.case_id, {}).get('route', []))}  \n"
            f"**Route taken:** {' -> '.join(tr.route) or '(none)'}  \n"
            f"**Tools:** {', '.join(tr.tools_used()) or '(none)'}  \n"
            f"**LLM calls:** {tr.llm_calls}  **Latency:** {tr.latency_s}s"
        )
        for t in tests:
            if t["key"] == key:
                icon = "PASS" if t["passed"] else "FAIL"
                st.markdown(f"`{icon}` **{t['check']}** {t['reason']}")
    with right:
        st.markdown("**Output**")
        st.text(tr.output or "(no answer)")
        if tr.error:
            st.error(f"{tr.error['type']}: {tr.error['message']}")
    with st.expander("Tool observations (raw)"):
        st.code(json.dumps(tr.observations, indent=2)[:20000], language="json")

    notes = st.text_area("Human notes", value=tr.human_notes, key=f"notes-{primary}-{key}")
    if st.button("Save notes"):
        tr.human_notes = notes
        write_traces(path, [by_key[k] for k in by_key])
        st.success(f"Saved to {path.name}.")
