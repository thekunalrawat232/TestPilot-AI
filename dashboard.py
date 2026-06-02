"""Streamlit dashboard for the AI Test Agent pipeline.

Launch with:
    streamlit run dashboard.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Test Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
LIVE_STATUS = ROOT / "generated" / "reports" / "live_status.json"
CHECKPOINTS_DIR = ROOT / "generated" / "reports" / "checkpoints"

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False
if "process" not in st.session_state:
    st.session_state.process = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NODE_ORDER = [
    "requirement_and_design",
    "code_generator",
    "execution",
    "debug",
    "finalise",
]

NODE_LABELS = {
    "requirement_and_design": "Requirement & Design",
    "code_generator": "Code Generator",
    "execution": "Execution",
    "debug": "Debug Loop",
    "finalise": "Finalise",
}

STATUS_COLORS = {
    "pending": "#6c757d",
    "running": "#0d6efd",
    "done": "#198754",
    "failed": "#dc3545",
}

STATUS_ICONS = {
    "pending": "⏳",
    "running": "⚙️",
    "done": "✅",
    "failed": "❌",
}

SEVERITY_COLORS = {
    "critical": "#dc3545",
    "high": "#fd7e14",
    "medium": "#ffc107",
    "low": "#0dcaf0",
}


def _read_status() -> dict:
    try:
        if LIVE_STATUS.exists():
            return json.loads(LIVE_STATUS.read_text())
    except Exception:
        pass
    return {}


def _list_checkpoints() -> list[str]:
    if not CHECKPOINTS_DIR.exists():
        return []
    runs: set[str] = set()
    for f in CHECKPOINTS_DIR.glob("*.json"):
        parts = f.stem.split("_", 1)
        if parts:
            runs.add(parts[0])
    return sorted(runs, reverse=True)


def _launch_pipeline(requirement: str, retries: int = 1) -> None:
    cmd = [sys.executable, str(ROOT / "main.py"), "--retries", str(retries), requirement]
    st.session_state.process = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    st.session_state.pipeline_running = True


def _resume_pipeline(run_id: str = "latest") -> None:
    cmd = [sys.executable, str(ROOT / "main.py"), "--resume", run_id]
    st.session_state.process = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    st.session_state.pipeline_running = True


def _poll_process() -> bool:
    """Return True if process is still running."""
    proc = st.session_state.get("process")
    if proc is None:
        return False
    return proc.poll() is None


def _node_card(col, name: str, status: str) -> None:
    label = NODE_LABELS.get(name, name)
    icon = STATUS_ICONS.get(status, "⏳")
    color = STATUS_COLORS.get(status, "#6c757d")
    col.markdown(
        f"""
        <div style="border:2px solid {color}; border-radius:8px; padding:12px; text-align:center; background:#0e1117;">
            <div style="font-size:1.6rem;">{icon}</div>
            <div style="font-weight:600; color:{color}; font-size:0.9rem; margin-top:4px;">{label}</div>
            <div style="font-size:0.75rem; color:{color}; text-transform:uppercase; letter-spacing:0.05em;">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🤖 AI Test Agent")
    st.markdown("---")

    requirement = st.text_area(
        "Feature requirement",
        placeholder="As a user I can log in with email and password",
        height=100,
        key="requirement_input",
    )

    retries = st.slider("Max debug retries", min_value=0, max_value=5, value=1)

    col_run, col_resume = st.columns(2)

    with col_run:
        if st.button("▶ Run", use_container_width=True, type="primary",
                     disabled=st.session_state.pipeline_running):
            if requirement.strip():
                _launch_pipeline(requirement.strip(), retries)
                st.rerun()
            else:
                st.warning("Enter a requirement first.")

    with col_resume:
        if st.button("↺ Resume", use_container_width=True,
                     disabled=st.session_state.pipeline_running):
            _resume_pipeline("latest")
            st.rerun()

    if st.session_state.pipeline_running:
        if st.button("⏹ Stop", use_container_width=True, type="secondary"):
            proc = st.session_state.get("process")
            if proc:
                proc.terminate()
            st.session_state.pipeline_running = False
            st.rerun()

    st.markdown("---")
    st.subheader("Recent runs")
    checkpoints = _list_checkpoints()
    if checkpoints:
        for run in checkpoints[:8]:
            if st.button(run, key=f"resume_{run}", use_container_width=True,
                         disabled=st.session_state.pipeline_running):
                _resume_pipeline(run)
                st.rerun()
    else:
        st.caption("No checkpoints yet.")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

status = _read_status()

# Check if process finished
if st.session_state.pipeline_running and not _poll_process():
    st.session_state.pipeline_running = False

# ---------------------------------------------------------------------------
# Header strip
# ---------------------------------------------------------------------------

pipeline_status = status.get("pipeline_status", "idle")

if st.session_state.pipeline_running:
    st.info("⚙️  Pipeline is running…", icon=None)
elif pipeline_status == "passed":
    st.success("✅  Pipeline finished — all tests passed!")
elif pipeline_status == "failed":
    st.error("❌  Pipeline finished — failures detected.")
elif status:
    st.warning("⚠️  Pipeline stopped.")
else:
    st.info("Enter a requirement in the sidebar and click **Run** to start.")

# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

if status:
    run_id = status.get("run_id", "—")
    started_at = status.get("started_at", "—")
    requirement_text = status.get("requirement", "—")

    meta_cols = st.columns(3)
    meta_cols[0].metric("Run ID", run_id)
    meta_cols[1].metric("Started", started_at[:19].replace("T", " ") if started_at != "—" else "—")
    meta_cols[2].metric("Status", pipeline_status.upper())

    st.markdown(f"**Requirement:** {requirement_text}")
    st.markdown("---")

# ---------------------------------------------------------------------------
# Pipeline node progress
# ---------------------------------------------------------------------------

st.subheader("Pipeline Progress")

node_statuses = status.get("node_statuses", {n: "pending" for n in NODE_ORDER})
node_cols = st.columns(len(NODE_ORDER))
for i, name in enumerate(NODE_ORDER):
    ns = node_statuses.get(name, "pending")
    _node_card(node_cols[i], name, ns)

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

summary = status.get("summary", {})
if summary:
    st.markdown("---")
    st.subheader("Metrics")
    m_cols = st.columns(5)
    m_cols[0].metric("Test Suites", summary.get("test_suites", 0))
    m_cols[1].metric("Test Cases", summary.get("test_cases", 0))
    m_cols[2].metric("Generated Files", summary.get("generated_files", 0))
    m_cols[3].metric("Failed Files", len(summary.get("failed_files", [])))
    retries_used = summary.get("retry_count", 0)
    max_r = summary.get("max_retries", 1)
    m_cols[4].metric("Retries Used", f"{retries_used} / {max_r}")

# ---------------------------------------------------------------------------
# Tabs: Test Plan | Execution Results | Bug Reports
# ---------------------------------------------------------------------------

if status:
    st.markdown("---")
    tab_plan, tab_exec, tab_bugs = st.tabs(["📋 Test Plan", "🧪 Execution Results", "🐛 Bug Reports"])

    # -------- Test Plan --------
    with tab_plan:
        test_plan = status.get("test_plan", {})
        if not test_plan:
            st.caption("No test plan generated yet.")
        else:
            feature = test_plan.get("feature_name", "")
            if feature:
                st.markdown(f"**Feature:** {feature}")

            for suite in test_plan.get("test_suites", []):
                with st.expander(f"Suite: {suite.get('suite_name', 'Unnamed')} — {suite.get('framework', '')}", expanded=True):
                    cases = suite.get("test_cases", [])
                    if cases:
                        rows = []
                        for tc in cases:
                            rows.append({
                                "ID": tc.get("id", ""),
                                "Name": tc.get("name", ""),
                                "Type": tc.get("type", ""),
                                "Priority": tc.get("priority", ""),
                                "Description": tc.get("description", ""),
                            })
                        st.dataframe(rows, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No test cases.")

    # -------- Execution Results --------
    with tab_exec:
        exec_result = status.get("execution_result", {})
        if not exec_result:
            st.caption("No execution results yet.")
        else:
            all_passed = exec_result.get("all_passed", False)
            failed_files = exec_result.get("failed_files", [])
            raw_outputs = exec_result.get("raw_outputs", {})

            if all_passed:
                st.success("All test files passed!")
            else:
                st.error(f"Failed files: {', '.join(failed_files) if failed_files else 'none'}")

            for fname, output in raw_outputs.items():
                is_failed = fname in failed_files
                label = f"❌ {fname}" if is_failed else f"✅ {fname}"
                with st.expander(label, expanded=is_failed):
                    st.code(output, language="text")

    # -------- Bug Reports --------
    with tab_bugs:
        bug_reports = summary.get("bug_reports", [])
        if not bug_reports:
            st.caption("No bugs found yet.")
        else:
            st.markdown(f"**{len(bug_reports)} bug(s) identified:**")
            for br in bug_reports:
                severity = br.get("severity", "low").lower()
                color = SEVERITY_COLORS.get(severity, "#6c757d")
                title = br.get("title", "Untitled")
                with st.expander(f"🐛 {title} — {severity.upper()}", expanded=True):
                    st.markdown(
                        f'<span style="background:{color}; color:#fff; padding:2px 8px; '
                        f'border-radius:4px; font-size:0.8rem;">{severity.upper()}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Expected:** {br.get('expected', '—')}")
                    st.markdown(f"**Actual:** {br.get('actual', '—')}")
                    steps = br.get("steps_to_reproduce", [])
                    if steps:
                        st.markdown("**Steps to reproduce:**")
                        for i, step in enumerate(steps, 1):
                            st.markdown(f"{i}. {step}")
                    evidence = br.get("evidence", "")
                    if evidence:
                        with st.expander("Evidence"):
                            st.code(evidence, language="text")

# ---------------------------------------------------------------------------
# Auto-refresh while running
# ---------------------------------------------------------------------------

if st.session_state.pipeline_running:
    time.sleep(2)
    st.rerun()
