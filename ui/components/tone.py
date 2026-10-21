"""News tone tab: metrics and per-summary tone with source links."""
import streamlit as st


def render(tones) -> None:
    st.subheader("News tone")
    if not tones:
        st.info("No tone results yet - run an analysis from the sidebar.")
        return

    st.caption(
        "Model-assessed tone of the model-written news summaries. This is not "
        "consumer sentiment, a survey, or a measured population opinion."
    )
    average = sum(item.score for item in tones) / len(tones)
    columns = st.columns(3)
    columns[0].metric(
        "Average model score",
        f"{average:+.2f}",
        help="Range -1 (negative) to +1 (positive); an LLM rating, not a calibrated measure.",
    )
    columns[1].metric("Positive summaries", sum(item.overall == "positive" for item in tones))
    columns[2].metric("Negative summaries", sum(item.overall == "negative" for item in tones))

    st.caption("Model score by summary")
    st.bar_chart(
        {item.headline[:45]: item.score for item in tones},
        height=260,
    )

    rows = []
    for item in tones:
        rows.append(
            {
                "Headline": item.headline,
                "Overall tone": item.overall,
                "Model score": round(item.score, 2),
                "Aspects": ", ".join(
                    f"{aspect.aspect} ({aspect.tone})" for aspect in item.aspects
                ),
                "Article ID": item.article_id[:8],
                "Published": item.published_at or "missing",
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(
        "Article ID prefixes match the full IDs and source URLs on the News tab. "
        "Missing publication times are shown explicitly."
    )
