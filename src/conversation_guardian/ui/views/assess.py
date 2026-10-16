"""
Assess Conversation view for ConversationGuardian Moderator Dashboard.

Provides interactive submission of ordered dialogue messages for checkpoints n in {1, 3, 5},
generates predictions using the validation-selected models, persists the assessment to MongoDB,
and displays results with heuristic signal summaries and the uncalibrated score notice.
"""

from typing import List
import streamlit as st

from conversation_guardian.storage import StorageWorkflowService
from conversation_guardian.ui.components import (
    render_alert_banner,
    render_uncalibrated_disclaimer,
    render_signals_table,
)

SAMPLE_ESCALATING = [
    "Please provide reliable citations for that claim before editing the article.",
    "I already provided the sources in the previous section if you bothered to read.",
    "Those citations do not support your assertion and misquote the author.",
    "You are being deliberately unhelpful and disruptive to this project.",
    "Stop making bad-faith personal attacks and adhere to our community guidelines.",
]

SAMPLE_CONSTRUCTIVE = [
    "Thank you for reviewing the recent revision to this section.",
    "I checked the secondary references and suggest rewording the second paragraph.",
    "Good point; I will update the phrasing to reflect the consensus on the talk page.",
    "Looks much better now. I appreciate the collaborative improvement.",
    "Glad we could reach consensus. I will mark the ticket as resolved.",
]


def render_assess_view(workflow: StorageWorkflowService):
    st.subheader("Assess Conversation Dialogue")
    st.markdown(
        "Submit an ordered conversational sequence at an observation checkpoint ($n = 1, 3, 5$) "
        "to forecast personal attack emergence within the next three dialogue turns ($H = 3$)."
    )

    # Initialize preset session state if needed
    if "messages_input" not in st.session_state:
        st.session_state["messages_input"] = list(SAMPLE_ESCALATING)
    if "conv_id_input" not in st.session_state:
        st.session_state["conv_id_input"] = "conv_interactive_demo_01"

    # Presets bar
    col_preset1, col_preset2, col_preset3 = st.columns([1, 1, 2])
    with col_preset1:
        if st.button("Load Escalating Preset", use_container_width=True):
            st.session_state["messages_input"] = list(SAMPLE_ESCALATING)
            st.session_state["conv_id_input"] = "conv_escalating_demo"
            st.rerun()

    with col_preset2:
        if st.button("Load Constructive Preset", use_container_width=True):
            st.session_state["messages_input"] = list(SAMPLE_CONSTRUCTIVE)
            st.session_state["conv_id_input"] = "conv_constructive_demo"
            st.rerun()

    # Form inputs
    col_cid, col_ckpt = st.columns([2, 1])
    with col_cid:
        conversation_id = st.text_input(
            "Conversation Identifier",
            value=st.session_state.get("conv_id_input", "conv_interactive_demo_01"),
            help="Stable identifier for the discussion thread being evaluated.",
        )
    with col_ckpt:
        checkpoint = st.selectbox(
            "Observation Checkpoint (n)",
            options=[1, 3, 5],
            index=1,
            help="Number of observed dialogue turns to evaluate.",
        )

    st.markdown(f"**Enter Observed Messages (Turn 1 to {checkpoint}):**")

    # Dynamic message inputs strictly matching selected checkpoint
    entered_messages: List[str] = []
    current_defaults = st.session_state.get("messages_input", SAMPLE_ESCALATING)

    for i in range(checkpoint):
        default_val = current_defaults[i] if i < len(current_defaults) else ""
        msg = st.text_area(
            f"Turn {i + 1}",
            value=default_val,
            height=70,
            key=f"turn_msg_{i+1}",
        )
        entered_messages.append(msg)

    # Submit button
    if st.button("Run Risk Assessment", type="primary"):
        # Validate inputs
        if not conversation_id.strip():
            st.error("Conversation Identifier cannot be empty.")
            return

        empty_indices = [idx + 1 for idx, m in enumerate(entered_messages) if not m.strip()]
        if empty_indices:
            st.error(f"Message fields for turns {empty_indices} are empty. All {checkpoint} observed turns are required.")
            return

        with st.spinner(f"Evaluating {checkpoint} turns through validation-selected model and persisting to database..."):
            try:
                asmt = workflow.assess_and_persist(
                    conversation_id=conversation_id.strip(),
                    messages=entered_messages,
                    source="dashboard_interactive",
                )
                st.session_state["latest_assessment"] = asmt
                st.session_state["selected_assessment_id"] = asmt["assessment_id"]
                st.success(f"Assessment complete and recorded in MongoDB: ID '{asmt['assessment_id']}'")
            except Exception as exc:
                st.error(f"Inference or persistence failed: {str(exc)}")
                return

    # Render latest result if available
    latest = st.session_state.get("latest_assessment")
    if latest:
        st.markdown("---")
        st.markdown("### Assessment Result")

        render_alert_banner(
            alert=latest["alert"],
            risk_score=latest["risk_score"],
            threshold=latest["decision_threshold"],
            checkpoint=latest["checkpoint_n_obs"],
        )

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Risk Score", f"{latest['risk_score']:.4f}")
        col_m2.metric("Decision Threshold", f"{latest['decision_threshold']:.4f}")
        col_m3.metric("Selected Model", latest.get("model_family", "-"))
        col_m4.metric("Forecast Horizon", f"H = {latest.get('horizon_h', 3)} turns")

        st.markdown("**Heuristic Conversational Tone Signals (Observed Turns):**")
        render_signals_table(latest.get("signals_summary", []))

        render_uncalibrated_disclaimer()

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("Review This Assessment in Moderator Panel"):
            st.session_state["active_tab"] = "Moderator Review"
            st.session_state["selected_assessment_id"] = latest["assessment_id"]
            st.rerun()
