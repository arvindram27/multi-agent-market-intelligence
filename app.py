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

from market_intelligence.persistence import persistence_configured

DEFAULT_RSS = "https://feeds.bbci.co.uk/news/business/rss.xml"

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
        if not persistence_configured():
            st.warning(
                "Persistence is not configured (DATABASE_URL is unset). "
                "Approved runs cannot be saved.",
                icon="⚠️",
            )
        rss_url = st.text_input("RSS feed", DEFAULT_RSS)
        subject_name = st.text_input(
            "Subject name", "", placeholder="Your company"
        )
        subject_page = st.text_input(
            "Subject page", "", placeholder="https://example.com/product"
        )
        subject_page_2 = st.text_input(
            "Subject page 2 (optional)",
            "",
            placeholder="https://example.com/pricing",
        )
        competitor_name = st.text_input(
            "Competitor name", "", placeholder="Competitor"
        )
        competitor_page = st.text_input(
            "Competitor page", "", placeholder="https://competitor.example/product"
        )
        competitor_page_2 = st.text_input(
            "Competitor page 2 (optional)",
            "",
            placeholder="https://competitor.example/pricing",
        )
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
    "News excerpts → topics + news tone → subject-vs-competitor comparison → "
    "human approval, orchestrated by a checkpointed LangGraph workflow."
)

if DEMO_MODE:
    if "state" not in st.session_state:
        st.session_state.state = load_fixture()
    st.warning(
        "Fixture snapshot captured 2026-10-21 — deterministic demo data, not a live run.",
        icon="🧪",
    )
elif run:
    subject_pages = [
        page.strip() for page in (subject_page, subject_page_2) if page.strip()
    ]
    competitor_pages = [
        page.strip()
        for page in (competitor_page, competitor_page_2)
        if page.strip()
    ]
    validation_error = None
    if not rss_url.strip():
        validation_error = "RSS feed is required."
    elif not subject_name.strip() or not competitor_name.strip():
        validation_error = "Subject and competitor names are required."
    elif not subject_pages or not competitor_pages:
        validation_error = "Subject and competitor each need at least one page."
    elif len(set(subject_pages)) != len(subject_pages):
        validation_error = "Subject page URLs must be unique."
    elif len(set(competitor_pages)) != len(competitor_pages):
        validation_error = "Competitor page URLs must be unique."
    if validation_error:
        st.error(validation_error)
    else:
        st.session_state.thread_id = uuid.uuid4().hex
        try:
            st.session_state.state = render.run_and_stream(
                graph,
                {"configurable": {"thread_id": st.session_state.thread_id}},
                {
                    "rss_url": rss_url,
                    "subject_name": subject_name,
                    "subject_pages": subject_pages,
                    "competitor_name": competitor_name,
                    "competitor_pages": competitor_pages,
                    "num_articles": num_articles,
                },
            )
        except Exception as exc:  # surface a friendly message instead of a stack trace
            st.error(f"Run failed: {exc}")

state = st.session_state.get("state")


def on_decision(approved: bool) -> None:
    """Approve or reject the profile. Runs as a button callback, before the rerun."""
    if DEMO_MODE:
        state["approved"] = approved
        state["status"] = "complete" if approved else "rejected"
        state["pending"] = False
        return

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    try:
        graph.invoke(Command(resume={"approved": approved}), config)
        st.session_state.state = render.view_from_snapshot(graph.get_state(config))
        st.session_state.decision_error = None
    except Exception as exc:  # keep the pending run visible if resume fails
        st.session_state.decision_error = str(exc)


decision_error = st.session_state.pop("decision_error", None)
if decision_error:
    st.error(f"Approval failed: {decision_error}")

if state:
    status = state.get("status", "running")
    st.caption(f"Status: **{status}**")
    for warning in state.get("warnings", []):
        st.warning(warning)
    if status == "persist_failed":
        detail = state.get("error") or ""
        st.error(
            "Persistence failed. The detail below says what is known about "
            "this attempt. Another writer may have stored a matching "
            "snapshot; check the stored runs before repeating anything. "
            f"{detail}".strip()
        )
    elif status == "persist_unknown":
        detail = state.get("error") or ""
        st.warning(
            "Persistence outcome unknown - the snapshot may or may not have "
            "been committed. Re-persisting the unchanged snapshot (same run id "
            "and content hash) is expected to be refused by the database's "
            "unique rule, which is not a guarantee for this database. Check "
            "the stored runs first. Changed content is stored as a new "
            f"version. {detail}".strip(),
            icon="⚠️",
        )
    elif state.get("error"):
        st.error(f"Decision not accepted: {state['error']}")

    if status == "complete" and state.get("approved") is True:
        st.success("Competitor profile approved — run complete.")
    elif status == "rejected" or state.get("approved") is False:
        st.error("Competitor profile rejected — nothing was persisted.")

tabs = st.tabs(["News", "Topics", "News tone", "Competitors"])

if state is None:
    st.info("Run an analysis to populate the tabs.")
else:
    render.render_tabs(tabs, state, on_decision)
