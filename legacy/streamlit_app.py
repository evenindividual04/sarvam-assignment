"""
Streamlit entry point for the Deep Research Agent.
Runs env validation at startup. Thread+queue async bridge (no nest_asyncio).
"""
from __future__ import annotations

import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔬",
    layout="wide",
)

# ── Startup validation (must run before anything else) ───────────────────
from utils.env_check import validate
from utils.logging_config import setup_logging

setup_logging()
try:
    validate()
except EnvironmentError as e:
    st.error(f"**Configuration error:** {e}")
    st.stop()

from agent.memory import init_db, save_circuit_event
from agent.orchestrator import ResearchOrchestrator
from utils.async_bridge import run_agent_sync
from utils.circuit_breaker import get_breaker
from utils.failure_policy import register_all_breakers

# Wire breakers (idempotent under Streamlit reruns).
register_all_breakers()
get_breaker().set_event_persister(save_circuit_event)

import asyncio

# ── DB init (once per process) ────────────────────────────────────────────
@st.cache_resource
def _init_db_once():
    import threading
    def _run():
        import asyncio
        asyncio.run(init_db())
    t = threading.Thread(target=_run)
    t.start()
    t.join()
    return True

_init_db_once()

# ── Shared agent instance (one per app process) ───────────────────────────
@st.cache_resource
def _get_agent() -> ResearchOrchestrator:
    return ResearchOrchestrator()

agent = _get_agent()

# ── Session state ─────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Session")
    entered_id = st.text_input(
        "Session ID",
        value=st.session_state.session_id,
        help="Paste a previous session ID to continue a prior research thread.",
    )
    if st.button("Load Session"):
        st.session_state.session_id = entered_id.strip()
        st.session_state.messages = []
        import asyncio
        import threading
        from agent.memory import get_session_turns
        
        def _get_turns():
            return asyncio.run(get_session_turns(st.session_state.session_id))
        
        # Need to run in a fresh loop via thread
        class ReturnThread(threading.Thread):
            def __init__(self):
                super().__init__()
                self.result = None
            def run(self):
                self.result = _get_turns()
                
        t = ReturnThread()
        t.start()
        t.join()
        
        for turn in t.result:
            st.session_state.messages.append({"role": "user", "content": turn.query})
            if turn.response:
                st.session_state.messages.append({"role": "assistant", "content": turn.response})
        st.rerun()
    if st.button("New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()
    if st.button("Run Evaluation"):
        import subprocess
        with st.spinner("Running eval/eval_runner.py..."):
            result = subprocess.run(["python", "eval/eval_runner.py"], capture_output=True, text=True)
            st.code(result.stdout)

    st.divider()
    st.caption(f"**Session ID:** `{st.session_state.session_id}`")
    st.caption("**Models:** Parallel AI → Gemini 2.5 Flash → Groq Llama 3.3 70B")

    st.divider()
    with st.expander("About & Assumptions"):
        st.markdown("""
**Deep Research Agent** — Sarvam AI FDSE Assignment

**Key assumptions:**
- Parallel AI as primary search (16K free quota)
- Gemini 2.5 Flash for synthesis (1.5K req/day free)
- Groq Llama 3.3 70B for planning & conflict detection
- GitHub Models GPT-4o-mini for eval judging
- Citations: `[Title — domain](URL)` format in final output
- Session context via SQLite + FTS5 retrieval
- No LangChain / LangGraph / CrewAI / LlamaIndex
""")

# ── Main chat area ─────────────────────────────────────────────────────────
st.title("🔬 Deep Research Agent")
st.caption("Multi-source research with conflict detection and citation grounding.")

# Replay message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a research question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Status container for intermediate steps (collapses on completion)
        status = st.status("Researching…", expanded=True)
        answer_placeholder = st.empty()
        full_answer = ""
        turn_meta: dict = {}

        for event in run_agent_sync(prompt, st.session_state.session_id, agent):
            if event.step == "planning":
                if event.data and "queries" in event.data:
                    queries = event.data["queries"]
                    strategy = event.data.get("strategy", "")
                    if strategy:
                        status.write(f"**Strategy:** {strategy}")
                    status.write(f"**Plan:** Searching for:\n" + "\n".join(f"- {q}" for q in queries))
                else:
                    status.write("**Planning** search strategy…")

            elif event.step == "searching":
                status.write("🔍 **Searching the web…**")

            elif event.step == "fetching":
                status.write("📥 **Fetching sources…**")

            elif event.step == "selecting":
                status.write("⚙️ **Selecting relevant context…**")

            elif event.step == "generating":
                if event.data and isinstance(event.data, str):
                    full_answer += event.data
                    answer_placeholder.markdown(full_answer + "▌")
                else:
                    status.write("✍️ **Generating answer with citations…**")

            elif event.step == "done":
                turn_meta = event.data or {}
                status.update(label="Research complete", state="complete", expanded=False)
                answer_placeholder.markdown(turn_meta.get("answer", full_answer))

            elif event.step == "error":
                status.update(label="Error", state="error")
                st.error(f"Agent error: {event.data}")

        final_answer = turn_meta.get("answer", full_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})

        # ── Trace Inspector (collapsed by default) ────────────────────────
        with st.expander("🔍 Trace Inspector", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.caption("Context sent to LLM")
                st.code(turn_meta.get("context_xml", "(none)"), language="xml")
            with col2:
                st.caption("Planner strategy")
                st.code(turn_meta.get("planning_strategy", "(none)"))
                st.caption("Context selector")
                st.code(turn_meta.get("selection_strategy", "heuristic"))
                st.caption("Token usage")
                st.metric("Prompt tokens", turn_meta.get("prompt_tokens", 0))
                st.metric("Completion tokens", turn_meta.get("completion_tokens", 0))
                st.caption("Stage latency (ms)")
                st.code(
                    "\\n".join([
                        f"planning: {turn_meta.get('planning_ms', 0)}",
                        f"search: {turn_meta.get('search_ms', 0)}",
                        f"fetch: {turn_meta.get('fetch_ms', 0)}",
                        f"select: {turn_meta.get('select_ms', 0)}",
                        f"synthesize: {turn_meta.get('synthesize_ms', 0)}",
                    ])
                )
                ci_score = turn_meta.get("citation_integrity_score", 1.0)
                st.caption("Citation integrity")
                st.progress(float(ci_score))
                st.caption(f"Score: {ci_score:.2f}")
                st.caption("Sources fetched")
                for url in (turn_meta.get("urls") or []):
                    st.markdown(f"- [{url}]({url})")
