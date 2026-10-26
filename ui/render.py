"""Streaming loop and tab rendering for the demo."""
import streamlit as st

from ui.components import competitors, news, tone, topics

NODE_LABELS = {
    "start_run": "Run start",
    "fetch_news": "News agent",
    "identify_topics": "Topic agent",
    "analyze_news_tone": "News tone agent",
    "research_competitor": "Competitor agent",
    "review_profile": "Human review",
    "persist": "Persist",
}


def _describe(node: str, data: dict) -> str:
    if node == "fetch_news":
        return f"{len(data.get('summaries', []))} articles summarized"
    if node == "identify_topics":
        return f"{len(data.get('topics', []))} topics extracted"
    if node == "analyze_news_tone":
        return f"{len(data.get('tones', []))} tone results"
    if node == "research_competitor":
        profile = data.get("competitor")
        return f"profile built for {profile.company}" if profile else "profile built"
    if node == "persist":
        if data.get("persisted_run_id"):
            return "saved"
        return "outcome unknown" if data.get("status") == "persist_unknown" else "failed"
    return "done"


def view_from_snapshot(snapshot) -> dict:
    """Normalize a LangGraph snapshot into the dict the components render."""
    values = snapshot.values or {}
    return {
        "summaries": values.get("summaries", []),
        "topics": values.get("topics", []),
        "tones": values.get("tones", []),
        "competitor": values.get("competitor"),
        "status": values.get("status", "running"),
        "approved": values.get("approved"),
        "error": values.get("error"),
        "persisted_run_id": values.get("persisted_run_id"),
        "warnings": values.get("warnings", []),
        "pending": "review_profile" in (snapshot.next or ()),
    }


def run_and_stream(graph, config: dict, inputs: dict) -> dict:
    """Drive the synchronous graph stream and return the resulting view."""
    with st.status("Running agents…", expanded=True) as status:
        for update in graph.stream(inputs, config, stream_mode="updates"):
            for node, data in update.items():
                if node in ("__interrupt__", "start_run"):
                    continue
                st.write(f"**{NODE_LABELS.get(node, node)}** — {_describe(node, data)}")
        snapshot = graph.get_state(config)
        pending = "review_profile" in (snapshot.next or ())
        view = view_from_snapshot(snapshot)
        failed = view["status"] == "persist_failed"
        unknown = view["status"] == "persist_unknown"
        status.update(
            label=(
                "Paused for approval"
                if pending
                else "Persistence outcome unknown"
                if unknown
                else "Persistence failed"
                if failed
                else "Run complete"
            ),
            state="error" if (failed or unknown) else "complete",
        )
    return view


def render_tabs(tabs, state: dict, on_decision) -> None:
    with tabs[0]:
        news.render(state.get("summaries", []))
    with tabs[1]:
        topics.render(state.get("topics", []))
    with tabs[2]:
        tone.render(state.get("tones", []))
    with tabs[3]:
        competitors.render(
            state.get("competitor"),
            pending=state.get("pending", False),
            on_decision=on_decision,
        )
