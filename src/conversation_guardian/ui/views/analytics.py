"""
Analytics View for ConversationGuardian Moderator Dashboard.

Provides database-driven aggregations:
- Alert distribution across observation checkpoints
- Moderator review distribution by status
- Alert rate by checkpoint calculation
"""

import streamlit as st
import pandas as pd
import altair as alt

from conversation_guardian.storage import ConversationGuardianRepository


def render_analytics_view(repo: ConversationGuardianRepository):
    st.subheader("Database Analytics & Triage Distributions")
    st.markdown(
        "Quantitative distribution of risk evaluations and moderation decisions stored in MongoDB. "
        "All figures reflect recorded application database entries."
    )

    try:
        alerts_by_ckpt = repo.aggregate_alerts_by_checkpoint()
        reviews_by_status = repo.aggregate_reviews_by_status()
        recent_asmts = repo.list_recent_assessments(limit=100)
    except Exception as exc:
        st.error(f"Error fetching analytics data: {str(exc)}")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Personal-Attack Alerts by Checkpoint")
        ckpt_counts = {1: {"total": 0, "alerts": 0}, 3: {"total": 0, "alerts": 0}, 5: {"total": 0, "alerts": 0}}
        for a in recent_asmts:
            c = a.get("checkpoint_n_obs")
            if c in ckpt_counts:
                ckpt_counts[c]["total"] += 1
                if a.get("alert"):
                    ckpt_counts[c]["alerts"] += 1

        rows = []
        for c in (1, 3, 5):
            tot = ckpt_counts[c]["total"]
            alt_cnt = ckpt_counts[c]["alerts"]
            rate = (alt_cnt / tot * 100) if tot > 0 else 0.0
            rows.append({
                "Checkpoint": f"n = {c}",
                "Evaluated": tot,
                "Personal-Attack Alerts": alt_cnt,
                "Alert Rate": f"{rate:.1f}%",
            })

        df_ckpt = pd.DataFrame(rows)
        st.dataframe(df_ckpt, use_container_width=True, hide_index=True)

        if sum(c["alerts"] for c in ckpt_counts.values()) > 0:
            chart = (
                alt.Chart(df_ckpt)
                .mark_bar(color="#0D9488", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Checkpoint:N", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Personal-Attack Alerts:Q", title="Personal-Attack Alerts"),
                    tooltip=["Checkpoint", "Evaluated", "Personal-Attack Alerts", "Alert Rate"],
                )
                .properties(height=220)
            )
            st.altair_chart(chart, use_container_width=True)

    with col2:
        st.markdown("#### Review Workflow Status Breakdown")
        status_data = []
        for r in reviews_by_status:
            status_data.append({
                "Review Status": r.get("review_status", "-"),
                "Count": r.get("review_count", 0),
            })

        if not status_data:
            st.info("No moderator reviews have been recorded in the database yet.")
        else:
            df_status = pd.DataFrame(status_data)
            st.dataframe(df_status, use_container_width=True, hide_index=True)

            chart_status = (
                alt.Chart(df_status)
                .mark_arc(innerRadius=45)
                .encode(
                    theta=alt.Theta("Count:Q"),
                    color=alt.Color(
                        "Review Status:N",
                        scale=alt.Scale(
                            domain=["PENDING", "IN_REVIEW", "RESOLVED", "DISMISSED"],
                            range=["#D97706", "#4F46E5", "#059669", "#64748B"]
                        ),
                    ),
                    tooltip=["Review Status", "Count"],
                )
                .properties(height=220)
            )
            st.altair_chart(chart_status, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<div style='color: #64748B; font-size: 0.8rem;'>"
        "<strong>Note:</strong> Distributions reflect database records from interactive assessments and test runs. "
        "They do not imply empirical online platform incidence rates or operational benchmark guarantees."
        "</div>",
        unsafe_allow_html=True,
    )
