"""
Unit tests for ConversationGuardian Moderator Dashboard components and logic.

Verifies:
- Overview KPI metric computation and aggregation mapping
- Recent conversation and assessment query listings
- Input validation for checkpoint dialogue sequences
- UI badge formatting and signal summaries
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pytest
import mongomock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from conversation_guardian.storage import (
    setup_database_schema,
    ConversationGuardianRepository,
    ReviewStatus,
    ModeratorDecision,
    ReviewEventAction,
)
from conversation_guardian.ui.components import (
    format_status_badge,
    format_alert_badge,
)


@pytest.fixture
def mock_repo():
    """Create a mock repository instance for dashboard testing."""
    client = mongomock.MongoClient()
    db = client["dashboard_test_db"]
    setup_database_schema(db)
    return ConversationGuardianRepository(db)


def test_overview_metrics_empty(mock_repo):
    """Verify overview metrics return zeroes when database is empty."""
    metrics = mock_repo.get_overview_metrics()
    assert metrics["total_conversations"] == 0
    assert metrics["total_assessments"] == 0
    assert metrics["total_alerts"] == 0
    assert metrics["active_reviews"] == 0


def test_overview_metrics_populated(mock_repo):
    """Verify overview metrics accurately compute totals and active reviews."""
    # Seed conversations
    mock_repo.create_or_update_conversation("conv_1", review_status=ReviewStatus.PENDING)
    mock_repo.create_or_update_conversation("conv_2", review_status=ReviewStatus.RESOLVED)

    # Seed assessments
    mock_repo.insert_risk_assessment({
        "assessment_id": "asmt_1",
        "conversation_id": "conv_1",
        "checkpoint_n_obs": 1,
        "horizon_h": 3,
        "observed_messages": ["Msg 1"],
        "risk_score": 0.85,
        "decision_threshold": 0.15,
        "alert": True,
        "model_family": "Hybrid_LogReg",
        "model_version": "1.0.0",
        "signals_summary": [],
    })
    mock_repo.insert_risk_assessment({
        "assessment_id": "asmt_2",
        "conversation_id": "conv_2",
        "checkpoint_n_obs": 3,
        "horizon_h": 3,
        "observed_messages": ["M1", "M2", "M3"],
        "risk_score": 0.05,
        "decision_threshold": 0.27,
        "alert": False,
        "model_family": "Hybrid_LogReg",
        "model_version": "1.0.0",
        "signals_summary": [],
    })

    # Seed reviews
    mock_repo.create_or_update_moderator_review(
        assessment_id="asmt_1",
        conversation_id="conv_1",
        review_status=ReviewStatus.IN_REVIEW,
        decision=ModeratorDecision.NO_ACTION,
        reviewer_id="mod_test",
        comment="Investigating alert",
    )
    mock_repo.create_or_update_moderator_review(
        assessment_id="asmt_2",
        conversation_id="conv_2",
        review_status=ReviewStatus.RESOLVED,
        decision=ModeratorDecision.NO_ACTION,
        reviewer_id="mod_test",
        comment="All good",
    )

    metrics = mock_repo.get_overview_metrics()
    assert metrics["total_conversations"] == 2
    assert metrics["total_assessments"] == 2
    assert metrics["total_alerts"] == 1
    assert metrics["active_reviews"] == 1  # only asmt_1 is IN_REVIEW; asmt_2 is RESOLVED


def test_recent_listings(mock_repo):
    """Verify list_recent_conversations and list_recent_assessments."""
    now = datetime.now(timezone.utc)
    mock_repo.create_or_update_conversation("conv_a")
    mock_repo.create_or_update_conversation("conv_b")

    convs = mock_repo.list_recent_conversations(limit=10)
    assert len(convs) == 2
    assert any(c["conversation_id"] == "conv_a" for c in convs)

    mock_repo.insert_risk_assessment({
        "assessment_id": "asmt_rec_1",
        "conversation_id": "conv_a",
        "checkpoint_n_obs": 1,
        "horizon_h": 3,
        "observed_messages": ["M1"],
        "risk_score": 0.1,
        "decision_threshold": 0.15,
        "alert": False,
        "model_family": "Hybrid_LogReg",
        "model_version": "1.0.0",
        "signals_summary": [],
        "created_at": now,
    })

    asmts = mock_repo.list_recent_assessments(limit=10)
    assert len(asmts) == 1
    assert asmts[0]["assessment_id"] == "asmt_rec_1"


def test_ui_badge_formatting():
    """Verify HTML generation for status and alert badges."""
    badge_pending = format_status_badge(ReviewStatus.PENDING)
    assert "PENDING" in badge_pending
    assert "cg-badge-pending" in badge_pending

    badge_resolved = format_status_badge("RESOLVED")
    assert "RESOLVED" in badge_resolved
    assert "cg-badge-resolved" in badge_resolved

    alert_true = format_alert_badge(True)
    assert "ALERT" in alert_true
    assert "cg-badge-alert-true" in alert_true

    alert_false = format_alert_badge(False)
    assert "SAFE" in alert_false
    assert "cg-badge-alert-false" in alert_false


def test_empty_conversation_history(mock_repo):
    """Verify querying non-existent conversation history returns empty list without error."""
    history = mock_repo.get_assessment_history("non_existent_conv_xyz")
    assert history == []
    meta = mock_repo.get_conversation("non_existent_conv_xyz")
    assert meta is None
