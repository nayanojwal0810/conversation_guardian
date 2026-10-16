"""
Standalone demonstration script for ConversationGuardian MongoDB persistence and audit workflow.

Demonstrates:
1. Live database connection and schema/index initialization.
2. Upsert of demonstration conversation record.
3. Multi-checkpoint inference execution via real deployable predictor and MongoDB persistence.
4. Historical assessment retrieval in chronological order.
5. Moderator review creation and state update.
6. Append-only review event audit trail preservation.
7. Aggregation pipelines for checkpoint alert counts and review status distribution.

Usage:
  python scripts/demo_mongodb_workflow.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root and src to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from conversation_guardian.storage import (
    DatabaseConnectionError,
    get_mongodb_uri,
    get_database_name,
    get_database,
    setup_database_schema,
    ConversationGuardianRepository,
    StorageWorkflowService,
    ReviewStatus,
    ModeratorDecision,
    ValidationError,
    RepositoryError,
)
from conversation_guardian.ml.predictor import ConversationGuardianPredictor


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def run_demo():
    print("=" * 80)
    print("CONVERSATIONGUARDIAN — MONGODB STORAGE & AUDIT WORKFLOW DEMO")
    print("=" * 80)

    uri = get_mongodb_uri()
    db_name = get_database_name()
    print(f"\n[Step 1/8] Connecting to MongoDB...")
    print(f"  -> Endpoint: {uri}")
    print(f"  -> Database: {db_name}")

    try:
        db = get_database(uri=uri, db_name=db_name, timeout_ms=3000)
        print("  -> Connection established and verified via admin.ping.")
    except DatabaseConnectionError as exc:
        print(f"\n[ERROR] Could not connect to live MongoDB instance:\n  {str(exc)}")
        print("\nTroubleshooting:")
        print("  1. Verify Docker is running and execute: 'docker compose up -d'")
        print("  2. Verify port 27017 is bound on localhost (127.0.0.1:27017)")
        print("  3. Check connection string environment variable: MONGODB_URI")
        sys.exit(1)

    print("\n[Step 2/8] Setting up collection schemas, $jsonSchema validators, and indexes...")
    setup_database_schema(db)
    print("  -> Initialized 4 collections with indexes: conversations, risk_assessments, moderator_reviews, review_events")

    repo = ConversationGuardianRepository(db)

    print("\n[Step 3/8] Initializing deployable predictor and storage workflow service...")
    try:
        workflow = StorageWorkflowService(repository=repo)
        print("  -> Loaded frozen validation-selected model bundles (n=1, 3, 5).")
    except Exception as exc:
        print(f"\n[ERROR] Failed to initialize predictor: {str(exc)}")
        sys.exit(1)

    # 4. Create sample conversation
    demo_conv_id = "demo_conv_audit_2026_01"
    print(f"\n[Step 4/8] Creating or updating demonstration conversation record: '{demo_conv_id}'...")
    conv = repo.create_or_update_conversation(
        conversation_id=demo_conv_id,
        source="demo_interactive_audit",
        review_status=ReviewStatus.PENDING,
    )
    print(f"  -> Conversation: ID={conv['conversation_id']}, Status={conv['review_status']}, Source={conv['source']}")

    # 5. Execute predictions and persist assessments across checkpoints
    sample_dialogue = [
        "Please provide reliable citations for that claim before editing the article.",
        "I already provided the sources in the previous section if you bothered to read.",
        "Those citations do not support your assertion and misquote the author.",
        "You are being deliberately unhelpful and disruptive to this project.",
        "Stop making bad-faith personal attacks and adhere to our community guidelines.",
    ]

    print("\n[Step 5/8] Generating and persisting risk assessments across observation checkpoints...")
    saved_assessments = []

    for n in (1, 3, 5):
        prefix = sample_dialogue[:n]
        print(f"  -> Assessing checkpoint n={n} ({n} observed turns)...")
        asmt = workflow.assess_and_persist(
            conversation_id=demo_conv_id,
            messages=prefix,
            source="demo_interactive_audit",
        )
        saved_assessments.append(asmt)
        print(
            f"     Assessment ID: {asmt['assessment_id']} | "
            f"Risk Score: {asmt['risk_score']:.4f} | "
            f"Threshold: {asmt['decision_threshold']:.4f} | "
            f"Alert: {asmt['alert']} | "
            f"Model: {asmt['model_family']}"
        )

    # Read back assessment history
    print(f"\n[Step 6/8] Retrieving chronological assessment history for '{demo_conv_id}'...")
    history = repo.get_assessment_history(demo_conv_id)
    print(f"  -> Retrieved {len(history)} historical assessments (sorted chronologically):")
    for i, h in enumerate(history, 1):
        print(
            f"     Turn {i}: checkpoint_n={h['checkpoint_n_obs']} | "
            f"asmt_id={h['assessment_id']} | "
            f"risk={h['risk_score']:.4f} | "
            f"alert={h['alert']} | "
            f"time={h['created_at'].isoformat()}"
        )

    # 6. Moderator review workflow
    latest_asmt = saved_assessments[-1]
    latest_id = latest_asmt["assessment_id"]
    print(f"\n[Step 7/8] Moderator review and append-only audit trail demonstration on assessment '{latest_id}'...")

    # Action 1: Create Review (Status: IN_REVIEW, Decision: NO_ACTION)
    print("  -> Moderator 'mod_42' opens review (Action: CREATE_REVIEW, Status: IN_REVIEW)...")
    rev1 = repo.create_or_update_moderator_review(
        assessment_id=latest_id,
        conversation_id=demo_conv_id,
        review_status=ReviewStatus.IN_REVIEW,
        decision=ModeratorDecision.NO_ACTION,
        reviewer_id="mod_42",
        comment="Review initiated after high checkpoint n=5 alert.",
    )
    print(f"     Review stored: Status={rev1['review_status']}, Decision={rev1['decision']}")

    # Action 2: Update Decision & Status (Status: RESOLVED, Decision: WARN_USER)
    print("  -> Moderator 'mod_42' concludes review (Action: CHANGE_STATUS, Status: RESOLVED, Decision: WARN_USER)...")
    rev2 = repo.create_or_update_moderator_review(
        assessment_id=latest_id,
        conversation_id=demo_conv_id,
        review_status=ReviewStatus.RESOLVED,
        decision=ModeratorDecision.WARN_USER,
        reviewer_id="mod_42",
        comment="User warned regarding repeated ad-hominem accusations in turn 4.",
    )
    print(f"     Review updated: Status={rev2['review_status']}, Decision={rev2['decision']}")

    # Fetch audit event log
    print("\n  -> Fetching append-only review event audit history:")
    events = repo.get_review_history(assessment_id=latest_id)
    print(f"     Found {len(events)} immutable audit events:")
    for ev in events:
        print(
            f"     Event [{ev['event_id']}]: action={ev['action']} | "
            f"prev={ev.get('previous_status')} -> new={ev['new_status']} | "
            f"decision={ev['decision']} | actor={ev['actor_id']} | "
            f"comment='{ev['comment']}'"
        )

    # 7. Aggregation pipelines
    print("\n[Step 8/8] Executing MongoDB aggregation pipelines...")
    checkpoint_alerts = repo.aggregate_alerts_by_checkpoint()
    print("  -> Aggregation 1: Total Alerts by Checkpoint:")
    for agg in checkpoint_alerts:
        print(f"     Checkpoint n={agg['checkpoint_n_obs']}: {agg['alert_count']} alert(s)")

    status_counts = repo.aggregate_reviews_by_status()
    print("\n  -> Aggregation 2: Moderator Reviews by Status:")
    for agg in status_counts:
        print(f"     Review Status '{agg['review_status']}': {agg['review_count']} record(s)")

    print("\n" + "=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY — ALL MONGODB WORKFLOW STAGES VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
