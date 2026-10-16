"""
Overview view for ConversationGuardian Moderator Dashboard.

Displays:
- Core operational KPIs: Total Conversations, Total Assessments, Active Reviews, Alerts Raised.
- Two concise Altair charts: Alerts by Checkpoint, Reviews by Status.
"""

import streamlit as st
import pandas as pd
import altair as alt
from typing import Dict, Any

from conversation_guardian.storage import ConversationGuardianRepository
from conversation_guardian.ui.components import render_kpi


def render_overview_view(repo: ConversationGuardianRepository):
    st.subheader("System Overview & Operational Status")

    # 1. Fetch metrics
    try:
        metrics = repo.get_overview_metrics()
        alerts_by_ckpt = repo.aggregate_alerts_by_checkpoint()
        reviews_by_status = repo.aggregate_reviews_by_status()
    except Exception as exc:
        st.error(f"Error fetching operational metrics from database: {str(exc)}")
        return

    # 2. Render 4 KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    render_kpi(col1, "Total Conversations", metrics.get("total_conversations", 0), "Tracked discussion threads")
    render_kpi(col2, "Risk Assessments", metrics.get("total_assessments", 0), "Evaluated observation checkpoints")
    render_kpi(col3, "Active Reviews", metrics.get("active_reviews", 0), "Pending or In-Review tickets")
    render_kpi(col4, "Derailment Alerts", metrics.get("total_alerts", 0), "High-risk threshold exceedances")

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # 3. Two Compact Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Alerts by Observation Checkpoint (H=3)**")
        # Ensure all checkpoints (1, 3, 5) appear even if count is 0
        ckpt_map = {1: 0, 3: 0, 5: 0}
        for item in alerts_by_ckpt:
            ckpt = item.get("checkpoint_n_obs")
            if ckpt in ckpt_map:
                ckpt_map[ckpt] = item.get("alert_count", 0)

        df_ckpt = pd.DataFrame([
            {"Checkpoint": f"n = {k} ({k} turns)", "Alerts": v}
            for k, v in ckpt_map.items()
        ])

        if df_ckpt["Alerts"].sum() == 0:
            st.info("No derailment alerts currently recorded in the database.")
        else:
            chart1 = (
                alt.Chart(df_ckpt)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#0D9488")
                .encode(
                    x=alt.X("Checkpoint:N", axis=alt.Axis(labelAngle=0, title=None)),
                    y=alt.Y("Alerts:Q", axis=alt.Axis(tickMinStep=1, title="Total Alerts")),
                    tooltip=["Checkpoint", "Alerts"],
                )
                .properties(height=240)
            )
            st.altair_chart(chart1, use_container_width=True)

    with chart_col2:
        st.markdown("**Moderator Reviews by Current Status**")
        status_map = {"PENDING": 0, "IN_REVIEW": 0, "RESOLVED": 0, "DISMISSED": 0}
        for item in reviews_by_status:
            stat = item.get("review_status")
            if stat in status_map:
                status_map[stat] = item.get("review_count", 0)

        df_status = pd.DataFrame([
            {"Status": k, "Reviews": v}
            for k, v in status_map.items()
        ])

        if df_status["Reviews"].sum() == 0:
            st.info("No moderator reviews currently filed in the database.")
        else:
            # Color scale matching our accessible palette
            color_scale = alt.Scale(
                domain=["PENDING", "IN_REVIEW", "RESOLVED", "DISMISSED"],
                range=["#D97706", "#4F46E5", "#059669", "#64748B"]
            )
            chart2 = (
                alt.Chart(df_status)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Status:N", axis=alt.Axis(labelAngle=0, title=None)),
                    y=alt.Y("Reviews:Q", axis=alt.Axis(tickMinStep=1, title="Total Reviews")),
                    color=alt.Color("Status:N", scale=color_scale, legend=None),
                    tooltip=["Status", "Reviews"],
                )
                .properties(height=240)
            )
            st.altair_chart(chart2, use_container_width=True)
