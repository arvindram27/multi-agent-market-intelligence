"""Streamlit demo for the market intelligence workflow.

Live run:
    streamlit run app.py

Deterministic fixture run (no API key, used for screenshots):
    PowerShell:  $env:DEMO_MODE=1; streamlit run app.py
    bash:        DEMO_MODE=1 streamlit run app.py
"""
import os
import uuid

import streamlit as st
from langgraph.types import Command

from ui import render
from ui.session import DEMO_MODE, init_session, load_fixture

DEFAULT_RSS = "https://feeds.bbci.co.uk/news/business/rss.xml"
DEFAULT_COMPETITOR = "https://www.google.com/about/"

st.set_page_config(
    page_title="Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

graph, config = init_session()

with st.sidebar:
    st.title("Market Intelligence")
    st.caption("LangGraph · LangChain · OpenAI")

    if DEMO_MODE:
        st.info("Fixture mode — deterministic snapshot, no API calls.")
    else:
        rss_url = st.text_input("RSS feed", DEFAULT_RSS)
        competitor_url = st.text_input("Competitor page", DEFAULT_COMPETITOR)
        num_articles = st.slider("Articles to analyze", 1, 5, 3)
        run = st.button("Run analysis", type="primary", width="stretch")
        if st.button("Reset", width="stretch"):
            for key in ("state", "thread_id"):
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    if config:
        st.caption(f"Thread: `{config['configurable']['thread_id'][:12]}`")

    st.divider()
    tracing = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    st.caption(f"LangSmith tracing: {'on' if tracing else 'off'}")
    st.markdown("[Open LangSmith](https://smith.langchain.com/)")

st.title("Market Intelligence")
st.caption(
    "News → trends + sentiment → competitor profile → human approval, "
    "orchestrated by a checkpointed LangGraph workflow."
)

if DEMO_MODE:
    if "state" not in st.session_state:
        st.session_state.state = load_fixture()
    st.warning(
        "Fixture snapshot captured 2026-10-21 — deterministic demo data, not a live run.",
        icon="🧪",
    )
elif run:
    st.session_state.thread_id = uuid.uuid4().hex
    try:
        st.session_state.state = render.run_and_stream(
            graph,
            {"configurable": {"thread_id": st.session_state.thread_id}},
            {
                "rss_url": rss_url,
                "competitor_url": competitor_url,
                "num_articles": num_articles,
            },
        )
    except Exception as exc:  # surface a friendly message instead of a stack trace
        st.error(f"Run failed: {exc}")

state = st.session_state.get("state")


def on_decision(approved: bool) -> None:
    """Approve or reject the pending competitor profile, then rerun the page."""
    if DEMO_MODE:
        state["approved"] = approved
        state["status"] = "complete" if approved else "rejected"
        state["pending"] = False
        st.rerun()

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    graph.invoke(Command(resume={"approved": approved}), config)
    st.session_state.state = render.view_from_snapshot(graph.get_state(config))
    st.rerun()


state = st.session_state.get("state")
if state:
    st.caption(f"Status: **{state.get('status', 'running')}**")

tabs = st.tabs(["News", "Trends", "Sentiment", "Competitors"])

if state is None:
    st.info("Run an analysis to populate the tabs.")
else:
    render.render_tabs(tabs, state, on_decision)
