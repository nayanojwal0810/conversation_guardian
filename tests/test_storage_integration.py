"""
Integration tests for ConversationGuardian against a live MongoDB instance.

These tests connect to the real MongoDB database configured via MONGODB_URI
(default: mongodb://127.0.0.1:27017).

If MongoDB is not reachable, tests are gracefully skipped with a clear,
actionable message instructing the developer to start MongoDB via Docker Compose.
"""

import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone
import pytest
from pymongo.errors import OperationFailure, DuplicateKeyError

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from conversation_guardian.storage import (
    DatabaseConnectionError,
    get_mongodb_uri,
    get_database,
    setup_database_schema,
    ConversationGuardianRepository,
    StorageWorkflowService,
    ReviewStatus,
    ModeratorDecision,
    COLL_CONVERSATIONS,
    COLL_RISK_ASSESSMENTS,
    COLL_MODERATOR_REVIEWS,
    COLL_REVIEW_EVENTS,
)


def is_live_mongodb_running() -> bool:
    """Check whether a real MongoDB instance is reachable within 1.5 seconds."""
    try:
        db = get_database(timeout_ms=1500)
        return True
    except (DatabaseConnectionError, Exception):
        return False


pytestmark = pytest.mark.skipif(
    not is_live_mongodb_running(),
    reason=(
        "Live MongoDB instance not reachable at configured MONGODB_URI. "
        "To run live integration tests, start MongoDB via: 'docker compose up -d'"
    ),
)


@pytest.fixture(scope="module")
def live_db():
    """Connect to live MongoDB and initialize schemas and indexes."""
    db = get_database(timeout_ms=3000)
    setup_database_schema(db)
    yield db


@pytest.fixture
def test_conv_id():
    """Generate a clean isolated test conversation ID."""
    return f"conv_live_test_{uuid.uuid4().hex[:8]}"


def test_live_schema_validation_rejection(live_db, test_conv_id):
    """Verify that MongoDB server-side $jsonSchema validator rejects invalid documents."""
    # Attempt to insert an assessment missing required fields directly into the collection
    invalid_doc = {
        "assessment_id": f"asmt_invalid_{uuid.uuid4().hex[:6]}",
        "conversation_id": test_conv_id,
        # Missing checkpoint_n_obs, horizon_h, risk_score, alert, etc.
    }
    with pytest.raises(OperationFailure) as exc_info:
        live_db[COLL_RISK_ASSESSMENTS].insert_one(invalid_doc)
    assert "Document failed validation" in str(exc_info.value) or "validation" in str(exc_info.value).lower()


def test_live_unique_index_rejection(live_db, test_conv_id):
    """Verify that live MongoDB enforces unique index on assessment_id."""
    asmt_id = f"asmt_uniq_{uuid.uuid4().hex[:8]}"
    valid_doc = {
        "assessment_id": asmt_id,
        "conversation_id": test_conv_id,
        "checkpoint_n_obs": 1,
        "horizon_h": 3,
        "observed_messages": ["A live test message turn."],
        "risk_score": 0.35,
        "decision_threshold": 0.15,
        "alert": True,
        "model_family": "Hybrid_LogReg",
        "model_version": "1.0.0",
        "signals_summary": [],
        "created_at": datetime.now(timezone.utc),
    }

    live_db[COLL_RISK_ASSESSMENTS].insert_one(valid_doc)

    # Attempt duplicate insert
    with pytest.raises(DuplicateKeyError):
        live_db[COLL_RISK_ASSESSMENTS].insert_one(valid_doc)

    # Cleanup
    live_db[COLL_RISK_ASSESSMENTS].delete_many({"conversation_id": test_conv_id})


def test_live_compound_index_query_plan(live_db, test_conv_id):
    """Verify that queries on (conversation_id, created_at) utilize index scan (IXSCAN)."""
    repo = ConversationGuardianRepository(live_db)
    # Insert sample record
    repo.insert_risk_assessment({
        "assessment_id": f"asmt_plan_{uuid.uuid4().hex[:6]}",
        "conversation_id": test_conv_id,
        "checkpoint_n_obs": 1,
        "horizon_h": 3,
        "observed_messages": ["Valid turn"],
        "risk_score": 0.2,
        "decision_threshold": 0.15,
        "alert": True,
        "model_family": "Hybrid_LogReg",
        "model_version": "1.0.0",
        "signals_summary": [],
    })

    # Query explain plan
    explain_plan = (
        live_db[COLL_RISK_ASSESSMENTS]
        .find({"conversation_id": test_conv_id})
        .sort("created_at", 1)
        .explain()
    )

    query_planner = explain_plan.get("queryPlanner", {})
    winning_plan = query_planner.get("winningPlan", {})

    # Plan should contain IXSCAN or FETCH over IXSCAN
    plan_str = str(winning_plan)
    assert "IXSCAN" in plan_str or "idx_assessment_conv_created" in plan_str

    # Cleanup
    live_db[COLL_RISK_ASSESSMENTS].delete_many({"conversation_id": test_conv_id})


def test_live_end_to_end_workflow(live_db, test_conv_id):
    """Verify live predictor execution, persistence, review updates, and audit event logging."""
    repo = ConversationGuardianRepository(live_db)
    workflow = StorageWorkflowService(repository=repo)

    messages = [
        "First turn message text.",
        "Second turn message text.",
        "Third turn message text.",
    ]

    # Run workflow
    asmt = workflow.assess_and_persist(
        conversation_id=test_conv_id,
        messages=messages,
        source="live_integration_test",
    )

    assert asmt["conversation_id"] == test_conv_id
    assert asmt["checkpoint_n_obs"] == 3
    assert "assessment_id" in asmt

    # Create review
    review = repo.create_or_update_moderator_review(
        assessment_id=asmt["assessment_id"],
        conversation_id=test_conv_id,
        review_status=ReviewStatus.RESOLVED,
        decision=ModeratorDecision.NO_ACTION,
        reviewer_id="mod_integration",
        comment="Live integration test review.",
    )
    assert review["review_status"] == "RESOLVED"

    # Verify audit events
    events = repo.get_review_history(assessment_id=asmt["assessment_id"])
    assert len(events) == 1
    assert events[0]["action"] == "CREATE_REVIEW"

    # Cleanup test documents
    live_db[COLL_CONVERSATIONS].delete_many({"conversation_id": test_conv_id})
    live_db[COLL_RISK_ASSESSMENTS].delete_many({"conversation_id": test_conv_id})
    live_db[COLL_MODERATOR_REVIEWS].delete_many({"conversation_id": test_conv_id})
    live_db[COLL_REVIEW_EVENTS].delete_many({"conversation_id": test_conv_id})
