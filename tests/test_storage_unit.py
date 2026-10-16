"""
Unit tests for ConversationGuardian MongoDB persistence and audit layer.

Uses isolated test doubles (mongomock) to verify schema rules, validation,
CRUD operations, event trails, and aggregation pipelines without requiring
a live MongoDB daemon.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pytest
import mongomock
from unittest.mock import MagicMock

# Ensure src is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from conversation_guardian.storage import (
    DatabaseConnectionError,
    ping_connection,
    setup_database_schema,
    ConversationGuardianRepository,
    StorageWorkflowService,
    ReviewStatus,
    ModeratorDecision,
    ReviewEventAction,
    ValidationError,
    RepositoryError,
)


@pytest.fixture
def mock_db():
    """Create a fresh in-memory mongomock Database instance with schemas configured."""
    client = mongomock.MongoClient()
    db = client["test_conversation_guardian"]
    setup_database_schema(db)
    return db


@pytest.fixture
def repo(mock_db):
    """Instantiate a repository backed by the mongomock database."""
    return ConversationGuardianRepository(mock_db)


# =============================================================================
# 1. Connection Failure Handling
# =============================================================================

def test_connection_failure_handling():
    """Verify ping_connection raises DatabaseConnectionError when connection fails."""
    mock_client = MagicMock()
    mock_client.admin.command.side_effect = Exception("Connection refused on port 27017")
    mock_client.address = ("127.0.0.1", 27017)

    with pytest.raises(DatabaseConnectionError) as exc_info:
        ping_connection(mock_client)

    assert "Failed to connect to MongoDB" in str(exc_info.value)
    assert "docker compose up -d" in str(exc_info.value)


# =============================================================================
# 2. Validation: Required Fields and Enums
# =============================================================================

def test_enum_validation(repo):
    """Verify that invalid enum strings are strictly rejected with ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        repo.create_or_update_conversation(
            conversation_id="conv_val_01",
            review_status="INVALID_STATUS",
        )
    assert "Invalid review_status" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        repo.create_or_update_moderator_review(
            assessment_id="asmt_01",
            conversation_id="conv_01",
            review_status=ReviewStatus.IN_REVIEW,
            decision="INVALID_DECISION",
            reviewer_id="mod_1",
            comment="",
        )
    assert "Invalid decision" in str(exc_info.value)


def test_assessment_field_validation(repo):
    """Verify that assessment boundary rules (checkpoint, horizon, score range) are enforced."""
    base_data = {
        "assessment_id": "asmt_val_01",
        "conversation_id": "conv_01",
        "checkpoint_n_obs": 1,
        "horizon_h": 3,
        "observed_messages": ["A valid test message."],
        "risk_score": 0.42,
        "decision_threshold": 0.15,
        "alert": True,
        "model_family": "Hybrid_LogReg",
        "model_version": "1.0.0",
        "signals_summary": [],
    }

    # Missing field
    incomplete = dict(base_data)
    del incomplete["risk_score"]
    with pytest.raises(ValidationError) as exc:
        repo.insert_risk_assessment(incomplete)
    assert "Missing required assessment field" in str(exc.value)

    # Invalid checkpoint
    bad_ckpt = dict(base_data, checkpoint_n_obs=4)
    with pytest.raises(ValidationError) as exc:
        repo.insert_risk_assessment(bad_ckpt)
    assert "checkpoint_n_obs must be strictly in (1, 3, 5)" in str(exc.value)

    # Invalid horizon
    bad_h = dict(base_data, horizon_h=2)
    with pytest.raises(ValidationError) as exc:
        repo.insert_risk_assessment(bad_h)
    assert "horizon_h must be strictly 3" in str(exc.value)

    # Out-of-bounds risk score
    bad_score = dict(base_data, risk_score=1.5)
    with pytest.raises(ValidationError) as exc:
        repo.insert_risk_assessment(bad_score)
    assert "risk_score must be in [0.0, 1.0]" in str(exc.value)

    # Empty message
    bad_msg = dict(base_data, observed_messages=["  "])
    with pytest.raises(ValidationError) as exc:
        repo.insert_risk_assessment(bad_msg)
    assert "Message at index 0 must be a non-empty string" in str(exc.value)


# =============================================================================
# 3. Conversation Creation and Retrieval
# =============================================================================

def test_conversation_creation_and_retrieval(repo):
    """Verify conversation upsert sets defaults and updates fields cleanly."""
    conv_id = "conv_test_100"
    created = repo.create_or_update_conversation(
        conversation_id=conv_id,
        source="unit_test",
        review_status=ReviewStatus.PENDING,
    )
    assert created["conversation_id"] == conv_id
    assert created["review_status"] == "PENDING"
    assert created["latest_assessment_id"] is None
    assert "created_at" in created
    assert "updated_at" in created

    # Update latest assessment reference
    updated = repo.create_or_update_conversation(
        conversation_id=conv_id,
        latest_assessment_id="asmt_abc",
        review_status=ReviewStatus.IN_REVIEW,
    )
    assert updated["latest_assessment_id"] == "asmt_abc"
    assert updated["review_status"] == "IN_REVIEW"

    # Fetch back
    fetched = repo.get_conversation(conv_id)
    assert fetched == updated


# =============================================================================
# 4. Assessment Insertion, Retrieval & Chronological Ordering
# =============================================================================

def test_assessment_insertion_and_chronological_ordering(repo):
    """Verify risk assessments are saved immutably and returned in chronological order."""
    conv_id = "conv_seq_01"
    repo.create_or_update_conversation(conv_id)

    t0 = datetime.now(timezone.utc) - timedelta(minutes=10)
    t1 = datetime.now(timezone.utc) - timedelta(minutes=5)
    t2 = datetime.now(timezone.utc)

    asmt1 = {
        "assessment_id": "asmt_seq_1",
        "conversation_id": conv_id,
        "checkpoint_n_obs": 1,
        "horizon_h": 3,
        "observed_messages": ["Turn 1 message"],
        "risk_score": 0.20,
        "decision_threshold": 0.15,
        "alert": True,
        "model_family": "Hybrid_LogReg",
        "model_version": "1.0.0",
        "signals_summary": [],
        "created_at": t0,
    }
    asmt2 = {
        "assessment_id": "asmt_seq_3",
        "conversation_id": conv_id,
        "checkpoint_n_obs": 3,
        "horizon_h": 3,
        "observed_messages": ["Turn 1", "Turn 2", "Turn 3"],
        "risk_score": 0.45,
        "decision_threshold": 0.27,
        "alert": True,
        "model_family": "Hybrid_LogReg",
        "model_version": "1.0.0",
        "signals_summary": [],
        "created_at": t1,
    }
    asmt3 = {
        "assessment_id": "asmt_seq_5",
        "conversation_id": conv_id,
        "checkpoint_n_obs": 5,
        "horizon_h": 3,
        "observed_messages": ["T1", "T2", "T3", "T4", "T5"],
        "risk_score": 0.70,
        "decision_threshold": 0.42,
        "alert": True,
        "model_family": "AggregateState_LogReg",
        "model_version": "1.0.0",
        "signals_summary": [],
        "created_at": t2,
    }

    # Insert out of order
    repo.insert_risk_assessment(asmt2)
    repo.insert_risk_assessment(asmt1)
    repo.insert_risk_assessment(asmt3)

    # Verify single fetch
    fetched = repo.get_assessment("asmt_seq_1")
    assert fetched is not None
    assert fetched["assessment_id"] == "asmt_seq_1"

    # Verify chronological ordering
    history = repo.get_assessment_history(conv_id)
    assert len(history) == 3
    assert history[0]["assessment_id"] == "asmt_seq_1"
    assert history[1]["assessment_id"] == "asmt_seq_3"
    assert history[2]["assessment_id"] == "asmt_seq_5"

    # Latest assessment check
    latest = repo.get_latest_assessment(conv_id)
    assert latest is not None
    assert latest["assessment_id"] == "asmt_seq_5"


# =============================================================================
# 5. Unique Index Enforcement
# =============================================================================

def test_unique_index_enforcement(repo):
    """Verify unique constraint violations on assessment_id and event_id raise RepositoryError."""
    asmt = {
        "assessment_id": "asmt_duplicate_01",
        "conversation_id": "conv_dup",
        "checkpoint_n_obs": 1,
        "horizon_h": 3,
        "observed_messages": ["Message 1"],
        "risk_score": 0.12,
        "decision_threshold": 0.15,
        "alert": False,
        "model_family": "Hybrid_LogReg",
        "model_version": "1.0.0",
        "signals_summary": [],
    }

    repo.insert_risk_assessment(asmt)

    with pytest.raises(RepositoryError) as exc:
        repo.insert_risk_assessment(asmt)
    assert "already exists" in str(exc.value)

    # Unique event_id test
    ev = {
        "event_id": "evt_duplicate_01",
        "assessment_id": "asmt_01",
        "conversation_id": "conv_01",
        "action": ReviewEventAction.CREATE_REVIEW,
        "new_status": ReviewStatus.PENDING,
        "decision": ModeratorDecision.NO_ACTION,
        "actor_id": "mod_1",
        "comment": "init",
    }
    repo.append_review_event(ev)

    with pytest.raises(RepositoryError) as exc:
        repo.append_review_event(ev)
    assert "already exists" in str(exc.value)


# =============================================================================
# 6. Moderator Review & Append-Only Audit Trail
# =============================================================================

def test_review_and_audit_event_trail(repo):
    """Verify review creation, state updates, and immutable audit event persistence."""
    asmt_id = "asmt_review_01"
    conv_id = "conv_review_01"
    repo.create_or_update_conversation(conv_id)

    # Step 1: Initial review creation
    rev1 = repo.create_or_update_moderator_review(
        assessment_id=asmt_id,
        conversation_id=conv_id,
        review_status=ReviewStatus.IN_REVIEW,
        decision=ModeratorDecision.NO_ACTION,
        reviewer_id="mod_alpha",
        comment="Investigating flagged dialogue.",
    )
    assert rev1["review_status"] == "IN_REVIEW"
    assert rev1["decision"] == "NO_ACTION"

    # Step 2: Decision update (status remains IN_REVIEW)
    rev2 = repo.create_or_update_moderator_review(
        assessment_id=asmt_id,
        conversation_id=conv_id,
        review_status=ReviewStatus.IN_REVIEW,
        decision=ModeratorDecision.WARN_USER,
        reviewer_id="mod_alpha",
        comment="Issuing warning for aggressive language.",
    )
    assert rev2["decision"] == "WARN_USER"

    # Step 3: Status change to RESOLVED
    rev3 = repo.create_or_update_moderator_review(
        assessment_id=asmt_id,
        conversation_id=conv_id,
        review_status=ReviewStatus.RESOLVED,
        decision=ModeratorDecision.WARN_USER,
        reviewer_id="mod_alpha",
        comment="Warning delivered. Ticket resolved.",
    )
    assert rev3["review_status"] == "RESOLVED"

    # Audit events check: exactly 3 events must exist in order
    events = repo.get_review_history(assessment_id=asmt_id)
    assert len(events) == 3

    assert events[0]["action"] == "CREATE_REVIEW"
    assert events[0]["previous_status"] is None
    assert events[0]["new_status"] == "IN_REVIEW"

    assert events[1]["action"] == "UPDATE_DECISION"
    assert events[1]["new_status"] == "IN_REVIEW"
    assert events[1]["decision"] == "WARN_USER"

    assert events[2]["action"] == "CHANGE_STATUS"
    assert events[2]["previous_status"] == "IN_REVIEW"
    assert events[2]["new_status"] == "RESOLVED"

    # Verify conversation's review_status was also kept in sync
    conv = repo.get_conversation(conv_id)
    assert conv is not None
    assert conv["review_status"] == "RESOLVED"


# =============================================================================
# 7. Query Filters and Aggregations
# =============================================================================

def test_query_filters_and_aggregations(repo):
    """Verify alert listing filters and aggregation pipeline results."""
    # Seed assessments across checkpoints
    records = [
        {"checkpoint": 1, "alert": True, "model": "Hybrid_LogReg"},
        {"checkpoint": 1, "alert": True, "model": "Hybrid_LogReg"},
        {"checkpoint": 1, "alert": False, "model": "Hybrid_LogReg"},
        {"checkpoint": 3, "alert": True, "model": "Hybrid_LogReg"},
        {"checkpoint": 5, "alert": False, "model": "AggregateState_LogReg"},
    ]

    for i, r in enumerate(records):
        n = r["checkpoint"]
        repo.insert_risk_assessment({
            "assessment_id": f"asmt_agg_{i}",
            "conversation_id": "conv_agg",
            "checkpoint_n_obs": n,
            "horizon_h": 3,
            "observed_messages": [f"Message {k}" for k in range(n)],
            "risk_score": 0.5 if r["alert"] else 0.1,
            "decision_threshold": 0.25,
            "alert": r["alert"],
            "model_family": r["model"],
            "model_version": "1.0.0",
            "signals_summary": [],
        })

    # Alert listing with filter
    alerts_n1 = repo.list_alerts(checkpoint_n_obs=1)
    assert len(alerts_n1) == 2
    for a in alerts_n1:
        assert a["alert"] is True
        assert a["checkpoint_n_obs"] == 1

    # Aggregation: alerts by checkpoint
    checkpoint_counts = repo.aggregate_alerts_by_checkpoint()
    count_map = {item["checkpoint_n_obs"]: item["alert_count"] for item in checkpoint_counts}
    assert count_map[1] == 2
    assert count_map[3] == 1
    assert 5 not in count_map  # No alerts at n=5


# =============================================================================
# 8. Predictor-to-Storage Workflow Service Integration
# =============================================================================

def test_workflow_service_integration(repo):
    """Verify StorageWorkflowService executes prediction, persists assessment and updates conversation."""
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = {
        "checkpoint_n_obs": 3,
        "horizon_h": 3,
        "model_family": "Hybrid_LogReg",
        "model_version": "1.0.0",
        "risk_score": 0.6543,
        "threshold": 0.2700,
        "alert": True,
        "signals_summary": [{"turn_index": 1, "state": "ESCALATING"}],
    }

    workflow = StorageWorkflowService(repository=repo, predictor=mock_predictor)
    conv_id = "conv_wf_01"
    messages = ["Turn 1", "Turn 2", "Turn 3"]

    asmt = workflow.assess_and_persist(conv_id, messages, source="workflow_test")

    assert asmt["conversation_id"] == conv_id
    assert asmt["checkpoint_n_obs"] == 3
    assert asmt["alert"] is True
    assert asmt["risk_score"] == 0.6543

    # Verify conversation pointer
    conv = repo.get_conversation(conv_id)
    assert conv is not None
    assert conv["latest_assessment_id"] == asmt["assessment_id"]


# =============================================================================
# 9. Missing IDs and Unknown Assessments
# =============================================================================

def test_missing_ids_handling(repo):
    """Verify clean None returns for unknown records and ValidationError on empty IDs."""
    assert repo.get_conversation("nonexistent_conv") is None
    assert repo.get_assessment("nonexistent_asmt") is None
    assert repo.get_review("nonexistent_asmt") is None
    assert repo.get_latest_assessment("nonexistent_conv") is None

    with pytest.raises(ValidationError):
        repo.get_conversation("")

    with pytest.raises(ValidationError):
        repo.get_assessment("")

    with pytest.raises(ValidationError):
        repo.get_review("")
