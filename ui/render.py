"""Streaming loop and tab rendering for the demo."""
import streamlit as st

from ui.components import competitors, news, sentiment, trends

NODE_LABELS = {
    "fetch_news": "News agent",
    "identify_trends": "Trend agent",
    "analyze_sentiment": "Sentiment agent",
    "research_competitor": "Competitor agent",
    "review_profile": "Human review",
    "persist": "Persist",
}


def _describe(node: str, data: dict) -> str:
    if node == "fetch_news":
        return f"{len(data.get('summaries', []))} articles summarized"
    if node == "identify_trends":
        return f"{len(data.get('trends', []))} trends identified"
    if node == "analyze_sentiment":
        return f"{len(data.get('sentiments', []))} sentiment results"
    if node == "research_competitor":
        profile = data.get("competitor")
        return f"profile built for {profile.company}" if profile else "profile built"
    return "done"


def view_from_snapshot(snapshot) -> dict:
    """Normalize a LangGraph snapshot into the dict the components render."""
    values = snapshot.values or {}
    return {
        "summaries": values.get("summaries", []),
        "trends": values.get("trends", []),
        "sentiments": values.get("sentiments", []),
        "competitor": values.get("competitor"),
        "status": values.get("status", "running"),
        "approved": values.get("approved"),
        "pending": "review_profile" in (snapshot.next or ()),
    }


def run_and_stream(graph, config: dict, inputs: dict) -> dict:
    """Drive the synchronous graph stream and return the resulting view."""
    with st.status("Running agents…", expanded=True) as status:
        for update in graph.stream(inputs, config, stream_mode="updates"):
            for node, data in update.items():
                if node == "__interrupt__":
                    continue
                st.write(f"**{NODE_LABELS.get(node, node)}** — {_describe(node, data)}")
        snapshot = graph.get_state(config)
        pending = "review_profile" in (snapshot.next or ())
        status.update(
            label="Paused for approval" if pending else "Run complete",
            state="complete",
        )
    return view_from_snapshot(snapshot)


def render_tabs(tabs, state: dict, on_decision) -> None:
    with tabs[0]:
        news.render(state.get("summaries", []))
    with tabs[1]:
        trends.render(state.get("trends", []))
    with tabs[2]:
        sentiment.render(state.get("sentiments", []))
    with tabs[3]:
        competitors.render(
            state.get("competitor"),
            pending=state.get("pending", False),
            on_decision=on_decision,
        )
