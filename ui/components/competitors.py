"""Competitors tab: profile details and the human-in-the-loop approval gate."""
import streamlit as st


def render(competitor, pending: bool, on_decision) -> None:
    st.subheader("Competitor profile")
    if competitor is None:
        st.info("No competitor profile yet — run an analysis from the sidebar.")
        return

    st.markdown(f"### {competitor.company}")
    st.write(competitor.positioning)

    left, right = st.columns(2)
    with left:
        st.markdown("**Products**")
        if competitor.products:
            for product in competitor.products:
                st.markdown(f"- {product}")
        else:
            st.write("—")
        st.markdown("**Pricing model**")
        st.write(competitor.pricing_model or "—")
        st.markdown("**Funding stage**")
        st.write(competitor.funding_stage or "—")

    with right:
        st.markdown("**Strengths**")
        for strength in competitor.strengths:
            st.markdown(f"- {strength}")
        st.markdown("**Weaknesses**")
        for weakness in competitor.weaknesses:
            st.markdown(f"- {weakness}")

    if pending:
        st.warning("Human approval required before this profile is persisted.")
        approve, reject = st.columns(2)
        approve.button(
            "Approve profile",
            type="primary",
            width="stretch",
            on_click=on_decision,
            args=(True,),
        )
        reject.button(
            "Reject",
            width="stretch",
            on_click=on_decision,
            args=(False,),
        )

    if competitor.sources:
        st.markdown("**Sources**")
        for source in competitor.sources:
            st.markdown(f"- {source}")

    with st.expander("Raw profile (JSON)"):
        st.json(competitor.model_dump())
