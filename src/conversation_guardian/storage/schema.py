"""
MongoDB collection schemas, JSON Schema validators, enums, and index configurations.

Implements strict validation and indexed access patterns across:
1. conversations
2. risk_assessments
3. moderator_reviews
4. review_events (append-only)
"""

from enum import Enum
from typing import Dict, Any
from pymongo.database import Database
from pymongo import ASCENDING, IndexModel


# Collection names
COLL_CONVERSATIONS = "conversations"
COLL_RISK_ASSESSMENTS = "risk_assessments"
COLL_MODERATOR_REVIEWS = "moderator_reviews"
COLL_REVIEW_EVENTS = "review_events"


class ReviewStatus(str, Enum):
    """Allowed review statuses for conversations and assessments."""
    PENDING = "PENDING"
    IN_REVIEW = "IN_REVIEW"
    RESOLVED = "RESOLVED"
    DISMISSED = "DISMISSED"


class ModeratorDecision(str, Enum):
    """Allowed moderator decision actions."""
    NO_ACTION = "NO_ACTION"
    WARN_USER = "WARN_USER"
    ESCALATE = "ESCALATE"
    BLOCK = "BLOCK"


class ReviewEventAction(str, Enum):
    """Allowed event action types for the append-only review event audit log."""
    CREATE_REVIEW = "CREATE_REVIEW"
    UPDATE_DECISION = "UPDATE_DECISION"
    CHANGE_STATUS = "CHANGE_STATUS"


# JSON Schema Definitions for MongoDB Collection Validation
CONVERSATIONS_JSON_SCHEMA: Dict[str, Any] = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["conversation_id", "source", "created_at", "updated_at", "review_status"],
        "properties": {
            "_id": {"bsonType": "objectId"},
            "conversation_id": {
                "bsonType": "string",
                "minLength": 1,
                "description": "Unique, stable conversation identifier"
            },
            "source": {
                "bsonType": "string",
                "minLength": 1,
                "description": "Originating dataset or stream channel"
            },
            "created_at": {
                "bsonType": "date",
                "description": "UTC timestamp when the conversation record was created"
            },
            "updated_at": {
                "bsonType": "date",
                "description": "UTC timestamp of the latest conversation update"
            },
            "review_status": {
                "enum": [s.value for s in ReviewStatus],
                "description": "Current overall review status"
            },
            "latest_assessment_id": {
                "bsonType": ["string", "null"],
                "description": "Reference ID to the most recently generated risk assessment"
            }
        },
        "additionalProperties": False
    }
}

RISK_ASSESSMENTS_JSON_SCHEMA: Dict[str, Any] = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": [
            "assessment_id",
            "conversation_id",
            "checkpoint_n_obs",
            "horizon_h",
            "observed_messages",
            "risk_score",
            "decision_threshold",
            "alert",
            "model_family",
            "model_version",
            "signals_summary",
            "created_at"
        ],
        "properties": {
            "_id": {"bsonType": "objectId"},
            "assessment_id": {
                "bsonType": "string",
                "minLength": 1,
                "description": "Unique assessment identifier"
            },
            "conversation_id": {
                "bsonType": "string",
                "minLength": 1,
                "description": "Identifier of the conversation being assessed"
            },
            "checkpoint_n_obs": {
                "enum": [1, 3, 5],
                "description": "Number of observed turns strictly in {1, 3, 5}"
            },
            "horizon_h": {
                "enum": [3],
                "description": "Forecast horizon (strictly 3 dialogue turns)"
            },
            "observed_messages": {
                "bsonType": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {
                    "bsonType": "string",
                    "minLength": 1
                },
                "description": "Sequence of observed messages provided at this checkpoint"
            },
            "risk_score": {
                "bsonType": ["double", "int", "long"],
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Uncalibrated continuous risk score in [0.0, 1.0]"
            },
            "decision_threshold": {
                "bsonType": ["double", "int", "long"],
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Frozen validation-selected decision threshold"
            },
            "alert": {
                "bsonType": "bool",
                "description": "True if risk_score >= decision_threshold"
            },
            "model_family": {
                "bsonType": "string",
                "minLength": 1,
                "description": "Identifier of the selected model architecture"
            },
            "model_version": {
                "bsonType": "string",
                "minLength": 1,
                "description": "Release or bundle version of the predictor"
            },
            "signals_summary": {
                "bsonType": "array",
                "description": "Heuristic turn-level signals extracted strictly from observed text"
            },
            "created_at": {
                "bsonType": "date",
                "description": "UTC timestamp of assessment creation"
            }
        },
        "additionalProperties": False
    }
}

MODERATOR_REVIEWS_JSON_SCHEMA: Dict[str, Any] = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": [
            "assessment_id",
            "conversation_id",
            "review_status",
            "decision",
            "reviewer_id",
            "comment",
            "created_at",
            "updated_at"
        ],
        "properties": {
            "_id": {"bsonType": "objectId"},
            "assessment_id": {
                "bsonType": "string",
                "minLength": 1,
                "description": "Assessment ID being reviewed (unique per assessment)"
            },
            "conversation_id": {
                "bsonType": "string",
                "minLength": 1,
                "description": "Associated conversation identifier"
            },
            "review_status": {
                "enum": [s.value for s in ReviewStatus],
                "description": "Current status of the review"
            },
            "decision": {
                "enum": [d.value for d in ModeratorDecision],
                "description": "Moderator action or verdict"
            },
            "reviewer_id": {
                "bsonType": "string",
                "minLength": 1,
                "description": "Identifier of the reviewing human agent"
            },
            "comment": {
                "bsonType": "string",
                "description": "Moderator explanation or audit notes"
            },
            "created_at": {
                "bsonType": "date",
                "description": "UTC timestamp when the review record was created"
            },
            "updated_at": {
                "bsonType": "date",
                "description": "UTC timestamp of the most recent review update"
            }
        },
        "additionalProperties": False
    }
}

REVIEW_EVENTS_JSON_SCHEMA: Dict[str, Any] = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": [
            "event_id",
            "assessment_id",
            "conversation_id",
            "action",
            "new_status",
            "decision",
            "actor_id",
            "comment",
            "event_time"
        ],
        "properties": {
            "_id": {"bsonType": "objectId"},
            "event_id": {
                "bsonType": "string",
                "minLength": 1,
                "description": "Unique audit event identifier"
            },
            "assessment_id": {
                "bsonType": "string",
                "minLength": 1,
                "description": "Target assessment ID"
            },
            "conversation_id": {
                "bsonType": "string",
                "minLength": 1,
                "description": "Associated conversation identifier"
            },
            "action": {
                "enum": [a.value for a in ReviewEventAction],
                "description": "Action performed in this audit event"
            },
            "previous_status": {
                "bsonType": ["string", "null"],
                "description": "Status before this action occurred (null if initial creation)"
            },
            "new_status": {
                "enum": [s.value for s in ReviewStatus],
                "description": "Status after this action took effect"
            },
            "decision": {
                "enum": [d.value for d in ModeratorDecision],
                "description": "Decision associated with this audit event"
            },
            "actor_id": {
                "bsonType": "string",
                "minLength": 1,
                "description": "Identifier of the actor initiating the change"
            },
            "comment": {
                "bsonType": "string",
                "description": "Notes recorded alongside this action"
            },
            "event_time": {
                "bsonType": "date",
                "description": "UTC timestamp of the action event"
            }
        },
        "additionalProperties": False
    }
}


def setup_database_schema(db: Database) -> None:
    """
    Apply JSON Schema validators and create required indexes across all four collections.
    
    Safe to run idempotently on existing or fresh MongoDB databases.
    """
    collection_schemas = [
        (COLL_CONVERSATIONS, CONVERSATIONS_JSON_SCHEMA),
        (COLL_RISK_ASSESSMENTS, RISK_ASSESSMENTS_JSON_SCHEMA),
        (COLL_MODERATOR_REVIEWS, MODERATOR_REVIEWS_JSON_SCHEMA),
        (COLL_REVIEW_EVENTS, REVIEW_EVENTS_JSON_SCHEMA),
    ]

    existing_collections = db.list_collection_names()

    for col_name, schema in collection_schemas:
        if col_name not in existing_collections:
            try:
                db.create_collection(
                    col_name,
                    validator=schema,
                    validationAction="error",
                    validationLevel="strict",
                )
            except NotImplementedError:
                # Test doubles (e.g. mongomock) do not support collection validator options
                db.create_collection(col_name)
        else:
            try:
                db.command({
                    "collMod": col_name,
                    "validator": schema,
                    "validationAction": "error",
                    "validationLevel": "strict",
                })
            except Exception:
                # collMod may fail in restricted environments or mock drivers
                pass

    # 1. conversations indexes
    db[COLL_CONVERSATIONS].create_indexes([
        IndexModel([("conversation_id", ASCENDING)], unique=True, name="uniq_conversation_id"),
        IndexModel([("review_status", ASCENDING)], name="idx_conv_review_status"),
    ])

    # 2. risk_assessments indexes
    db[COLL_RISK_ASSESSMENTS].create_indexes([
        IndexModel([("assessment_id", ASCENDING)], unique=True, name="uniq_assessment_id"),
        IndexModel([("conversation_id", ASCENDING), ("created_at", ASCENDING)], name="idx_assessment_conv_created"),
        IndexModel([("checkpoint_n_obs", ASCENDING), ("alert", ASCENDING)], name="idx_checkpoint_alert"),
        IndexModel([("alert", ASCENDING), ("created_at", ASCENDING)], name="idx_alert_created"),
    ])

    # 3. moderator_reviews indexes
    db[COLL_MODERATOR_REVIEWS].create_indexes([
        IndexModel([("assessment_id", ASCENDING)], unique=True, name="uniq_review_assessment_id"),
        IndexModel([("conversation_id", ASCENDING)], name="idx_review_conversation_id"),
        IndexModel([("review_status", ASCENDING)], name="idx_review_status"),
    ])

    # 4. review_events indexes (append-only)
    db[COLL_REVIEW_EVENTS].create_indexes([
        IndexModel([("event_id", ASCENDING)], unique=True, name="uniq_event_id"),
        IndexModel([("conversation_id", ASCENDING), ("event_time", ASCENDING)], name="idx_event_conv_time"),
        IndexModel([("assessment_id", ASCENDING), ("event_time", ASCENDING)], name="idx_event_assessment_time"),
    ])
