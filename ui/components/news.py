"""News tab: one card per RSS excerpt, with captured source provenance."""
import streamlit as st

BADGE = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}


def _source_link(item) -> None:
    if item.source_url and item.source_url.startswith(("http://", "https://")):
        st.markdown(f"[Open original article]({item.source_url})")
    elif item.source_url:
        st.caption(f"Source URL: {item.source_url}")
    else:
        st.caption("No article URL was present in the feed.")


def render(summaries) -> None:
    st.subheader("News summaries (RSS excerpts)")
    if not summaries:
        st.info("No articles yet - run an analysis from the sidebar.")
        return

    st.caption(
        f"{len(summaries)} RSS excerpts processed. These are feed excerpts, not "
        "full-article fetches; the source link and timestamps come from the feed."
    )
    for index, item in enumerate(summaries):
        with st.expander(item.headline, expanded=index == 0):
            st.write(item.summary)
            if item.key_facts:
                st.markdown("**Key facts**")
                for fact in item.key_facts:
                    st.markdown(f"- {fact}")

            st.markdown("**Source record**")
            st.caption(f"Feed title: {item.source_title}")
            _source_link(item)
            st.caption(
                f"Article ID: `{item.article_id}` · "
                f"Published: {item.published_at or 'missing from feed'} · "
                f"Retrieved: {item.retrieved_at}"
            )
            st.caption(
                f"Content hash: `{item.content_hash[:16]}…` · "
                f"Content kind: {item.content_kind}"
            )
            with st.expander("Captured RSS excerpt"):
                st.write(item.excerpt)

            st.caption(
                f"{BADGE.get(item.tone_hint, '⚪')} Model tone hint for this excerpt: "
                f"{item.tone_hint} (not consumer sentiment)"
            )
