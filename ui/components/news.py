"""News tab: one expandable card per summarized article."""
import streamlit as st

BADGE = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}


def render(summaries) -> None:
    st.subheader("News summaries")
    if not summaries:
        st.info("No articles yet — run an analysis from the sidebar.")
        return

    st.caption(f"{len(summaries)} articles processed")
    for index, item in enumerate(summaries):
        with st.expander(item.headline, expanded=index == 0):
            st.write(item.summary)
            if item.key_facts:
                st.markdown("**Key facts**")
                for fact in item.key_facts:
                    st.markdown(f"- {fact}")
            st.caption(f"{BADGE.get(item.sentiment_hint, '⚪')} Sentiment hint: {item.sentiment_hint}")
