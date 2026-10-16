"""
Moderator Review and Audit View for ConversationGuardian Streamlit Dashboard.

Allows moderators to:
1. Inspect an assessment context (conversation, checkpoint, score, messages).
2. View current active review status and decision.
3. Submit review actions (transitions status and decision, recording immutable audit events).
4. Inspect the append-only review event history.
"""

from typing import Optional, Dict, Any, List
import streamlit as st
import pandas as pd

from conversation_guardian.storage import (
    ConversationGuardianRepository,
    ReviewStatus,
    ModeratorDecision,
    ValidationError,
    RepositoryError,
)
from conversation_guardian.ui.components import render_alert_banner


def render_review_view(repo: ConversationGuardianRepository):
    st.subheader("Moderator Decision & Audit Trail")
    st.markdown("Evaluate flagged assessments, execute moderation decisions, and review immutable audit event logs.")

    # 1. Select assessment
    recent_asmts = []
    try:
        recent_asmts = repo.list_recent_assessments(limit=30)
    except Exception:
        pass

    asmt_options = [
        f"{a['assessment_id']} | Conv: {a['conversation_id']} | n={a['checkpoint_n_obs']} | {'⚠️ ALERT' if a['alert'] else 'Safe'}"
        for a in recent_asmts
    ]

    preselected_id = st.session_state.get("selected_assessment_id", "")
    default_idx = 0
    if preselected_id:
        for idx, opt in enumerate(asmt_options):
            if opt.startswith(preselected_id):
                default_idx = idx + 1
                break

    col_select, col_manual = st.columns([2, 1])
    with col_select:
        selected_opt = st.selectbox(
            "Recent Assessments Queue",
            options=["-- Select an assessment --"] + asmt_options,
            index=default_idx,
        )
    with col_manual:
        manual_id = st.text_input(
            "Or Enter Assessment ID",
            value=preselected_id if preselected_id else "",
            placeholder="e.g. asmt_ea475d2806dc",
        )

    target_asmt_id = ""
    if manual_id.strip():
        target_asmt_id = manual_id.strip()
    elif selected_opt != "-- Select an assessment --":
        target_asmt_id = selected_opt.split(" | ")[0].strip()

    if not target_asmt_id:
        st.info("Select an assessment from the queue or enter an Assessment ID to open the review panel.")
        return

    # 2. Fetch assessment document
    try:
        asmt = repo.get_assessment(target_asmt_id)
    except Exception as exc:
        st.error(f"Error fetching assessment '{target_asmt_id}': {str(exc)}")
        return

    if not asmt:
        st.warning(f"Assessment '{target_asmt_id}' was not found in the database.")
        return

    # Render Assessment Context Card
    st.markdown("---")
    st.markdown(f"### Assessment Context: `{asmt['assessment_id']}`")

    render_alert_banner(
        alert=asmt["alert"],
        risk_score=asmt["risk_score"],
        threshold=asmt["decision_threshold"],
        checkpoint=asmt["checkpoint_n_obs"],
    )

    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    col_c1.metric("Conversation ID", asmt.get("conversation_id", "-"))
    col_c2.metric("Checkpoint turns", f"n = {asmt.get('checkpoint_n_obs', '-')}")
    col_c3.metric("Model Architecture", asmt.get("model_family", "-"))
    col_c4.metric("Risk Score", f"{asmt.get('risk_score', 0.0):.4f}")

    with st.expander("View Observed Dialogue Turns for This Assessment", expanded=False):
        for idx, turn_txt in enumerate(asmt.get("observed_messages", []), 1):
            st.markdown(f"**Turn {idx}:** {turn_txt}")

    # 3. Current Review Status
    st.markdown("---")
    st.markdown("### Current Active Review Record")

    current_review = None
    try:
        current_review = repo.get_review(target_asmt_id)
    except Exception as exc:
        st.error(f"Error checking existing review: {str(exc)}")

    if current_review:
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("Current Status", current_review.get("review_status", "-"))
        col_r2.metric("Current Decision", current_review.get("decision", "-"))
        col_r3.metric("Last Reviewer", current_review.get("reviewer_id", "-"))
        last_updated = current_review.get("updated_at")
        updated_str = last_updated.strftime("%Y-%m-%d %H:%M UTC") if hasattr(last_updated, "strftime") else str(last_updated)
        col_r4.metric("Last Updated", updated_str)

        if current_review.get("comment"):
            st.info(f"**Active Moderator Comment:** {current_review['comment']}")
    else:
        st.markdown(
            "> *No moderator review is currently active for this assessment. Complete the form below to file an initial review.*"
        )

    # 4. Action Form
    st.markdown("### Submit Moderator Decision")
    with st.form("moderator_action_form"):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            reviewer_id = st.text_input(
                "Reviewer / Actor ID",
                value=current_review.get("reviewer_id", "mod_lead_01") if current_review else "mod_lead_01",
            )
        with col_f2:
            status_options = [s.value for s in ReviewStatus]
            default_status_idx = 0
            if current_review and current_review.get("review_status") in status_options:
                default_status_idx = status_options.index(current_review["review_status"])
            else:
                default_status_idx = status_options.index("IN_REVIEW")

            new_status = st.selectbox(
                "Target Review Status",
                options=status_options,
                index=default_status_idx,
            )
        with col_f3:
            decision_options = [d.value for d in ModeratorDecision]
            default_dec_idx = 0
            if current_review and current_review.get("decision") in decision_options:
                default_dec_idx = decision_options.index(current_review["decision"])
            elif asmt["alert"]:
                default_dec_idx = decision_options.index("WARN_USER")
            else:
                default_dec_idx = decision_options.index("NO_ACTION")

            new_decision = st.selectbox(
                "Moderator Action Decision",
                options=decision_options,
                index=default_dec_idx,
            )

        comment = st.text_area(
            "Review Notes & Decision Justification",
            value="",
            placeholder="Explain the rationale for this status update or disciplinary action...",
            height=80,
        )

        submitted = st.form_submit_button("Record Moderator Action", type="primary")

    if submitted:
        if not reviewer_id.strip():
            st.error("Reviewer ID cannot be empty.")
        else:
            try:
                updated_rev = repo.create_or_update_moderator_review(
                    assessment_id=target_asmt_id,
                    conversation_id=asmt["conversation_id"],
                    review_status=ReviewStatus(new_status),
                    decision=ModeratorDecision(new_decision),
                    reviewer_id=reviewer_id.strip(),
                    comment=comment.strip(),
                )
                st.success(
                    f"Successfully recorded review action: Status -> {updated_rev['review_status']}, "
                    f"Decision -> {updated_rev['decision']}. Audit event appended."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to record moderator action: {str(exc)}")

    # 5. Append-only Audit Event History
    st.markdown("---")
    st.markdown("### Immutable Review Event Audit History")
    st.markdown(
        "<span style='color: #64748B; font-size: 0.85rem;'>"
        "The <code>review_events</code> collection is append-only. Past actions cannot be overwritten or deleted."
        "</span>",
        unsafe_allow_html=True,
    )

    try:
        events = repo.get_review_history(assessment_id=target_asmt_id)
    except Exception as exc:
        st.error(f"Error querying audit events: {str(exc)}")
        events = []

    if not events:
        st.info("No historical review events recorded for this assessment yet.")
    else:
        event_rows = []
        for ev in events:
            time_str = ev["event_time"].strftime("%Y-%m-%d %H:%M:%S UTC") if hasattr(ev["event_time"], "strftime") else str(ev["event_time"])
            prev = ev.get("previous_status") or "(Initial)"
            transition = f"{prev} → {ev['new_status']}"
            event_rows.append({
                "Event ID": ev["event_id"],
                "Timestamp": time_str,
                "Action": ev["action"],
                "State Transition": transition,
                "Decision": ev["decision"],
                "Actor ID": ev["actor_id"],
                "Comment": ev.get("comment", ""),
            })

        df_events = pd.DataFrame(event_rows)
        st.dataframe(df_events, use_container_width=True, hide_index=True)
