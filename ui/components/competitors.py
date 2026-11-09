"""Competitors tab: side-by-side comparison, evidence and the approval gate."""
import streamlit as st

FIELD_LABELS = (
    ("product_focus", "Product focus"),
    ("pricing_model", "Pricing model"),
    ("positioning", "Positioning"),
)


def _approval_gate(pending: bool, on_decision) -> None:
    if not pending:
        return
    st.warning("Human approval required before this comparison is persisted.")
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


def _render_field(side, field_name: str, label: str, side_label: str) -> None:
    field = getattr(side, field_name)
    pages = {page.source_id: page for page in side.pages}
    st.markdown(f"**{label}**")
    if field.value is None:
        st.caption("not supported by retrieved sources")
        return
    st.write(field.value)
    for evidence in field.evidence:
        page = pages.get(evidence.source_id)
        st.markdown(f"> {evidence.quote}")
        source_label = f"{side_label} {evidence.source_id}"
        if page is not None and page.url.startswith(("http://", "https://")):
            st.markdown(f"[{source_label}]({page.url})")
        else:
            st.markdown(source_label)
        if page is not None:
            st.caption(
                f"Retrieved: {page.retrieved_at} · "
                f"Content hash: `{page.content_hash[:16]}…`"
            )


def _render_side(side, side_label: str) -> None:
    st.markdown(f"### {side.name}")
    for field_name, label in FIELD_LABELS:
        _render_field(side, field_name, label, side_label)
    st.markdown("**Retrieved sources**")
    for page in side.pages:
        st.markdown(f"- `{side_label} {page.source_id}`: {page.url}")
    st.caption(
        "Citations are checked against the capped page text sent to the model. "
        "Existence of the quote is not proof that it supports the value."
    )


def render(
    comparison,
    competitor,
    pending: bool,
    on_decision,
    warnings=(),
) -> None:
    st.subheader("Competitor comparison")
    if comparison is None:
        if competitor is not None:
            st.info(
                "Legacy run (pre-comparison): this checkpoint holds the old "
                "single-company profile. Re-run to produce a comparison."
            )
            st.markdown(f"### {competitor.company}")
            st.write(competitor.positioning)
            _approval_gate(pending, on_decision)
            with st.expander("Raw profile (JSON)"):
                st.json(competitor.model_dump())
            return
        st.info("No comparison yet - run an analysis from the sidebar.")
        return

    st.caption(
        "Citation check only: each quote is verified, after whitespace "
        "normalization, to appear in the exact capped page text sent to the "
        "model. This does not prove the quote supports the value - judge that "
        "before approval. Empty fields are explicitly unsupported."
    )
    if warnings:
        with st.expander(f"Warnings ({len(warnings)})"):
            for warning in warnings:
                st.warning(warning)

    subject, competitor_side = st.columns(2)
    with subject:
        _render_side(comparison.subject, "Subject")
    with competitor_side:
        _render_side(comparison.competitor, "Competitor")

    _approval_gate(pending, on_decision)
    with st.expander("Raw comparison (JSON)"):
        st.json(comparison.model_dump())
