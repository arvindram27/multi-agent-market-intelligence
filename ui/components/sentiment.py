"""Sentiment tab: headline metrics, per-article scores, and aspect breakdown."""
import streamlit as st


def render(sentiments) -> None:
    st.subheader("Consumer sentiment")
    if not sentiments:
        st.info("No sentiment results yet — run an analysis from the sidebar.")
        return

    average = sum(item.score for item in sentiments) / len(sentiments)
    columns = st.columns(3)
    columns[0].metric("Average score", f"{average:+.2f}", help="Range −1 (negative) to +1 (positive)")
    columns[1].metric("Positive", sum(item.overall == "positive" for item in sentiments))
    columns[2].metric("Negative", sum(item.overall == "negative" for item in sentiments))

    st.caption("Score by article")
    st.bar_chart(
        {f"Article {index + 1}": item.score for index, item in enumerate(sentiments)},
        height=260,
    )

    st.dataframe(
        [
            {
                "Article": f"Article {index + 1}",
                "Overall": item.overall,
                "Score": round(item.score, 2),
                "Aspects": ", ".join(
                    f"{aspect.aspect} ({aspect.sentiment})" for aspect in item.aspects
                ),
            }
            for index, item in enumerate(sentiments)
        ],
        width="stretch",
        hide_index=True,
    )
