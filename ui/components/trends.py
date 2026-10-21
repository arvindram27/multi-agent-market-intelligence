"""Trends tab: confidence chart, table, and evidence per trend."""
import altair as alt
import streamlit as st


def render(trends) -> None:
    st.subheader("Market trends")
    if not trends:
        st.info("No trends yet — run an analysis from the sidebar.")
        return

    st.caption("Confidence by trend")
    chart = (
        alt.Chart(
            {
                "values": [
                    {"trend": trend.trend_name, "confidence": trend.confidence}
                    for trend in trends
                ]
            }
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "confidence:Q",
                scale=alt.Scale(domain=[0, 1]),
                title="Confidence",
            ),
            y=alt.Y(
                "trend:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelLimit=280),
            ),
            tooltip=[
                alt.Tooltip("trend:N", title="Trend"),
                alt.Tooltip("confidence:Q", format=".2f"),
            ],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, width="stretch")
    st.dataframe(
        [
            {
                "Trend": trend.trend_name,
                "Confidence": round(trend.confidence, 2),
                "Keywords": ", ".join(trend.keywords),
            }
            for trend in trends
        ],
        width="stretch",
        hide_index=True,
    )

    for trend in trends:
        with st.expander(f"{trend.trend_name} — evidence"):
            for evidence in trend.evidence:
                st.markdown(f"- {evidence}")
