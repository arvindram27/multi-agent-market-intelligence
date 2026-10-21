"""Topics tab: model-confidence chart and source-linked evidence per topic."""
import altair as alt
import streamlit as st


def _source_line(evidence) -> str:
    if evidence.source_url and evidence.source_url.startswith(("http://", "https://")):
        return f"[{evidence.source_title}]({evidence.source_url})"
    return evidence.source_title


def render(topics) -> None:
    st.subheader("Topics in this batch")
    if not topics:
        st.info("No topics yet - run an analysis from the sidebar.")
        return

    first = topics[0]
    if first.window_start and first.window_end:
        window = f"{first.window_start} → {first.window_end}"
    else:
        window = "publication times missing from feed"
    st.caption(
        f"Derived from {first.source_count} RSS excerpt(s); publication window: "
        f"{window}. This is topic extraction from one snapshot, not a time-series "
        "trend claim."
    )

    st.caption("Model-assessed confidence (LLM self-rating, not a calibrated probability)")
    chart = (
        alt.Chart(
            {
                "values": [
                    {"topic": topic.topic_name, "confidence": topic.model_confidence}
                    for topic in topics
                ]
            }
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "confidence:Q",
                scale=alt.Scale(domain=[0, 1]),
                title="Model-assessed confidence",
            ),
            y=alt.Y(
                "topic:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelLimit=280),
            ),
            tooltip=[
                alt.Tooltip("topic:N", title="Topic"),
                alt.Tooltip("confidence:Q", format=".2f"),
            ],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, width="stretch")
    st.dataframe(
        [
            {
                "Topic": topic.topic_name,
                "Model confidence": round(topic.model_confidence, 2),
                "Keywords": ", ".join(topic.keywords),
                "Evidence spans": len(topic.evidence),
            }
            for topic in topics
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Model confidence is the LLM's own rating. It is not calibrated and should "
        "not be read as a probability that the topic is true or important."
    )

    for topic in topics:
        with st.expander(f"{topic.topic_name} - cited evidence"):
            if not topic.evidence:
                st.warning("No cited evidence. Treat this topic as unsupported.")
                continue
            for evidence in topic.evidence:
                status = "✅ verified in excerpt" if evidence.verified else "⚠️ unverified quote"
                st.markdown(f"> {evidence.quote}")
                st.caption(
                    f"{_source_line(evidence)} · "
                    f"published: {evidence.published_at or 'missing'} · {status}"
                )
