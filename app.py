"""
ConversationGuardian — Moderator Dashboard Application Entry Point.

Run with:
  streamlit run app.py
"""

import sys
from pathlib import Path

# Add project root and src to sys.path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import streamlit as st

from conversation_guardian.storage import (
    DatabaseConnectionError,
    get_mongodb_uri,
    get_database_name,
    get_database,
    setup_database_schema,
    ConversationGuardianRepository,
    StorageWorkflowService,
)
from conversation_guardian.ui.styles import CUSTOM_CSS
from conversation_guardian.ui.components import render_header
from conversation_guardian.ui.views.overview import render_overview_view
from conversation_guardian.ui.views.assess import render_assess_view
from conversation_guardian.ui.views.history import render_history_view
from conversation_guardian.ui.views.review import render_review_view
from conversation_guardian.ui.views.analytics import render_analytics_view


# Page configuration
st.set_page_config(
    page_title="ConversationGuardian — Moderator Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom analytics theme CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Connecting to ConversationGuardian database...")
def init_app_services():
    """
    Connect to MongoDB, verify schema, and initialize repository and prediction workflow.
    """
    uri = get_mongodb_uri()
    db_name = get_database_name()
    db = get_database(uri=uri, db_name=db_name, timeout_ms=3000)
    setup_database_schema(db)
    repo = ConversationGuardianRepository(db)
    workflow = StorageWorkflowService(repository=repo)
    return db, repo, workflow, uri, db_name


def main():
    # 1. Initialize backend connection
    db = None
    repo = None
    workflow = None
    uri = get_mongodb_uri()
    db_name = get_database_name()
    is_connected = False

    try:
        db, repo, workflow, uri, db_name = init_app_services()
        is_connected = True
    except DatabaseConnectionError as conn_err:
        render_header(endpoint=uri, is_connected=False)
        st.error(
            f"**Cannot connect to MongoDB at `{uri}`.**\n\n"
            f"**Error Details:** {str(conn_err)}\n\n"
            "**Troubleshooting Instructions:**\n"
            "1. Verify Docker Desktop is running.\n"
            "2. Start the project's MongoDB container: `docker compose up -d`\n"
            "3. Ensure the host port is mapped correctly (default: `127.0.0.1:27018`).\n"
            "4. Verify the `MONGODB_URI` environment variable if using custom endpoints."
        )
        return
    except Exception as exc:
        render_header(endpoint=uri, is_connected=False)
        st.error(
            f"**Application initialization error:** {str(exc)}\n\n"
            "Please ensure predictor bundles are prepared by running:\n"
            "`python scripts/run_sequence_benchmark.py`"
        )
        return

    # Render header
    render_header(endpoint=uri, is_connected=True)

    # 2. Sidebar Navigation and System Info
    nav_items = [
        "Overview",
        "Assess Conversation",
        "Assessment History",
        "Moderator Review",
        "Analytics",
    ]

    # Handle navigation state
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "Overview"

    with st.sidebar:
        st.markdown("### Navigation")
        selected_nav = st.radio(
            "Go to",
            options=nav_items,
            key="active_tab",
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### Active Models")
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #475569; line-height: 1.4;">
                <strong>n=1:</strong> Hybrid LogReg (&tau; = 0.15)<br>
                <strong>n=3:</strong> Hybrid LogReg (&tau; = 0.27)<br>
                <strong>n=5:</strong> AggregateState LogReg (&tau; = 0.42)<br>
                <strong>Horizon:</strong> H = 3 dialogue turns
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### Database Info")
        st.markdown(
            f"""
            <div style="font-size: 0.8rem; color: #475569; line-height: 1.4;">
                <strong>Host:</strong> <code>{uri}</code><br>
                <strong>Database:</strong> <code>{db_name}</code><br>
                <strong>Engine:</strong> MongoDB 7.0 Container
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Reconnect / Refresh DB"):
            st.cache_resource.clear()
            st.rerun()

    # 3. Route to active view
    if selected_nav == "Overview":
        render_overview_view(repo)
    elif selected_nav == "Assess Conversation":
        render_assess_view(workflow)
    elif selected_nav == "Assessment History":
        render_history_view(repo)
    elif selected_nav == "Moderator Review":
        render_review_view(repo)
    elif selected_nav == "Analytics":
        render_analytics_view(repo)


if __name__ == "__main__":
    main()
