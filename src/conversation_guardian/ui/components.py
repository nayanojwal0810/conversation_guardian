"""
Reusable UI layout and display components for ConversationGuardian Streamlit Dashboard.
"""

from typing import Dict, List, Any
import streamlit as st
import pandas as pd


def render_header(endpoint: str = "mongodb://127.0.0.1:27018", is_connected: bool = True):
    """Render top application header with system status."""
    status_color = "#10B981" if is_connected else "#EF4444"
    status_text = "DATABASE CONNECTED" if is_connected else "DISCONNECTED"
    
    html = f"""
    <div class="cg-header">
        <div>
            <h1>ConversationGuardian</h1>
            <p>Early Personal-Attack Forecasting & Moderator Audit Platform</p>
        </div>
        <div class="cg-header-status">
            <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:{status_color};"></span>
            <span>{status_text}</span> &nbsp;|&nbsp; <span>{endpoint}</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_kpi(col, title: str, value: Any, subtitle: str = ""):
    """Render a single KPI metric card into the specified Streamlit column."""
    with col:
        html = f"""
        <div class="cg-kpi-card">
            <div class="cg-kpi-label">{title}</div>
            <div class="cg-kpi-val">{value}</div>
            {f'<div class="cg-kpi-sub">{subtitle}</div>' if subtitle else ''}
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)


def render_alert_banner(alert: bool, risk_score: float, threshold: float, checkpoint: int):
    """Render high-contrast alert status banner."""
    if alert:
        html = f"""
        <div class="cg-alert-banner cg-alert-danger">
            <span style="font-size:1.2rem;">⚠️</span>
            <div>
                <strong>PERSONAL-ATTACK FORECAST ALERT RAISED</strong> &mdash; 
                Uncalibrated model score of <strong>{risk_score:.4f}</strong> meets or exceeds the validation-selected decision threshold of <strong>{threshold:.4f}</strong> at checkpoint n={checkpoint}.
            </div>
        </div>
        """
    else:
        html = f"""
        <div class="cg-alert-banner cg-alert-safe">
            <span style="font-size:1.2rem;">✓</span>
            <div>
                <strong>NOMINAL (NO ALERT)</strong> &mdash; 
                Uncalibrated model score of <strong>{risk_score:.4f}</strong> is below the validation-selected threshold of <strong>{threshold:.4f}</strong> at checkpoint n={checkpoint}.
            </div>
        </div>
        """
    st.markdown(html, unsafe_allow_html=True)


def render_uncalibrated_disclaimer():
    """Render mandatory interpretive disclaimer regarding model output scores."""
    st.markdown(
        """
        <div class="cg-disclaimer">
            <strong>Model Score Advisory:</strong> 
            The reported score is an uncalibrated model score from a validation-selected classifier, 
            <strong>not</strong> a calibrated probability. The classification threshold was selected using validation 
            data to optimize early forecasting utility. Predictions serve as an automated triage advisory 
            for human moderator review. Recorded decisions update internal audit logs and do not execute actions 
            on external chat platforms without a dedicated platform integration.
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_status_badge(status: Any) -> str:
    """Return HTML string for a colored status badge."""
    val = status.value if hasattr(status, "value") else str(status)
    clean_s = str(val).upper()
    css_class = f"cg-badge cg-badge-{clean_s.lower().replace('_', '-')}"
    return f'<span class="{css_class}">{clean_s}</span>'


def format_alert_badge(alert: bool) -> str:
    """Return HTML string for an alert badge."""
    css_class = "cg-badge cg-badge-alert-true" if alert else "cg-badge cg-badge-alert-false"
    text = "ALERT" if alert else "SAFE"
    return f'<span class="{css_class}">{text}</span>'


def render_signals_table(signals_summary: List[Dict[str, Any]]):
    """Render structured table of turn-by-turn conversational heuristic indicators."""
    if not signals_summary:
        st.info("No per-turn heuristic signals available.")
        return

    rows = []
    for sig in signals_summary:
        rows.append({
            "Turn": sig.get("turn_index", "-"),
            "Observed Message": (sig.get("text", "")[:75] + "...") if len(sig.get("text", "")) > 75 else sig.get("text", ""),
            "Tone State": sig.get("state", "-"),
            "VADER Compound": f"{sig.get('vader_compound', 0.0):+.4f}",
            "Hostility Match": "Yes" if sig.get("is_hostility_match") else "No",
            "Direct Address": sig.get("direct_address_count", 0),
            "Caps Ratio": f"{sig.get('caps_ratio', 0.0):.2%}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
