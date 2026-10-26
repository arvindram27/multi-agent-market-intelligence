"""Session wiring: one graph + checkpointer + thread id per Streamlit session."""
import json
import os
import sqlite3
import uuid
from pathlib import Path

import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver

from market_intelligence.agents.schemas import (
    CompetitorProfile,
    NewsSummary,
    NewsTone,
    TopicReport,
)
from market_intelligence.graph import build_graph, checkpoint_serde
from market_intelligence.persistence import default_profile_writer

DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"
FIXTURE_PATH = Path(__file__).resolve().parent.parent / "data" / "demo_run.json"


def init_session():
    """Build the graph once per session and return it with its run config."""
    if DEMO_MODE:
        return None, None

    if "graph" not in st.session_state:
        conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
        st.session_state.checkpoint_conn = conn
        st.session_state.graph = build_graph(
            checkpointer=SqliteSaver(conn, serde=checkpoint_serde()),
            writer=default_profile_writer(),
        )

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid.uuid4().hex

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    return st.session_state.graph, config


def load_fixture(path: Path = FIXTURE_PATH) -> dict:
    """Load the dated demo snapshot into the same view shape as a live run."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    state = payload["state"]
    return {
        "summaries": [NewsSummary(**item) for item in state["summaries"]],
        "topics": [TopicReport(**item) for item in state["topics"]],
        "tones": [NewsTone(**item) for item in state["tones"]],
        "competitor": CompetitorProfile(**state["competitor"]),
        "status": state.get("status", "awaiting_approval"),
        "approved": state.get("approved"),
        "warnings": state.get("warnings", []),
        "pending": state.get("status") == "awaiting_approval",
        "fixture_date": payload.get("captured_at"),
    }
