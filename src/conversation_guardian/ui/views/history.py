"""
Assessment History view for ConversationGuardian Moderator Dashboard.

Enables searching or selecting conversations, inspecting chronological assessment records,
viewing detailed heuristic signals, and transferring assessments to the moderator review panel.
"""

from typing import List, Dict, Any
import streamlit as st
import pandas as pd

from conversation_guardian.storage import ConversationGuardianRepository
from conversation_guardian.ui.components import render_signals_table, render_alert_banner


def render_history_view(repo: ConversationGuardianRepository):
    st.subheader("Conversation Assessment History")
    st.markdown("Inspect the chronological sequence of risk evaluations generated for any monitored discussion thread.")

    # 1. Fetch recent conversations for quick selection
    recent_convs = []
    try:
        recent_convs = repo.list_recent_conversations(limit=25)
    except Exception:
        pass

    conv_options = [c["conversation_id"] for c in recent_convs]

    col_select, col_custom = st.columns([1, 1])
    with col_select:
        selected_from_list = st.selectbox(
            "Recent Monitored Conversations",
            options=["-- Select from database --"] + conv_options,
            index=1 if conv_options else 0,
        )

    with col_custom:
        custom_id = st.text_input(
            "Or Search by Specific Conversation ID",
            value=selected_from_list if selected_from_list != "-- Select from database --" else "",
            placeholder="e.g. demo_conv_audit_2026_01",
        )

    target_conv_id = custom_id.strip() if custom_id.strip() else (
        selected_from_list if selected_from_list != "-- Select from database --" else ""
    )

    if not target_conv_id:
        st.info("Select a recent conversation or enter a conversation ID above to inspect its assessment history.")
        return

    # 2. Fetch conversation metadata and historical assessments
    try:
        conv_meta = repo.get_conversation(target_conv_id)
        history = repo.get_assessment_history(target_conv_id)
    except Exception as exc:
        st.error(f"Error querying history for conversation '{target_conv_id}': {str(exc)}")
        return

    if not conv_meta and not history:
        st.warning(f"No records found in database for conversation identifier '{target_conv_id}'.")
        return

    # Render Conversation Metadata
    if conv_meta:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Conversation ID", conv_meta.get("conversation_id", "-"))
        col_m2.metric("Overall Review Status", conv_meta.get("review_status", "-"))
        col_m3.metric("Source Channel", conv_meta.get("source", "-"))
        col_m4.metric("Total Assessments", len(history))

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    if not history:
        st.info(f"Conversation '{target_conv_id}' exists but has no risk assessments recorded yet.")
        return

    st.markdown(f"**Chronological Risk Assessment Records ({len(history)} total):**")

    # Table formatting
    rows = []
    for h in history:
        created_str = h["created_at"].strftime("%Y-%m-%d %H:%M:%S UTC") if hasattr(h["created_at"], "strftime") else str(h["created_at"])
        rows.append({
            "Assessment ID": h["assessment_id"],
            "Observed Checkpoint": f"n = {h['checkpoint_n_obs']} turns",
            "Created Timestamp": created_str,
            "Risk Score": f"{h['risk_score']:.4f}",
            "Threshold": f"{h['decision_threshold']:.4f}",
            "Alert Decision": "⚠️ ALERT" if h["alert"] else "✓ Safe",
            "Model Family": h.get("model_family", "-"),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # 3. Assessment Detail Inspector
    st.markdown("---")
    st.markdown("### Inspect Assessment Details")

    asmt_id_list = [h["assessment_id"] for h in history]
    selected_asmt_id = st.selectbox(
        "Select Assessment ID to View Signal Breakdown",
        options=asmt_id_list,
        index=len(asmt_id_list) - 1,  # Default to latest
    )

    selected_doc = next((h for h in history if h["assessment_id"] == selected_asmt_id), None)
    if selected_doc:
        render_alert_banner(
            alert=selected_doc["alert"],
            risk_score=selected_doc["risk_score"],
            threshold=selected_doc["decision_threshold"],
            checkpoint=selected_doc["checkpoint_n_obs"],
        )

        st.markdown("**Observed Dialogue Turns Provided to Classifier:**")
        for idx, text in enumerate(selected_doc.get("observed_messages", []), 1):
            st.markdown(f"> **Turn {idx}:** {text}")

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("**Turn-by-Turn Heuristic Signal Summary:**")
        render_signals_table(selected_doc.get("signals_summary", []))

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("Open This Assessment in Moderator Review", key="btn_history_to_review"):
            st.session_state["active_tab"] = "Moderator Review"
            st.session_state["selected_assessment_id"] = selected_doc["assessment_id"]
            st.rerun()
