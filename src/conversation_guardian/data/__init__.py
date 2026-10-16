"""Data preparation and validation module for ConversationGuardian."""

from .schemas import (
    FEATURE_COLUMN_WHITELIST,
    FORBIDDEN_AUDIT_COLUMNS,
    STATUS_ELIGIBLE,
    STATUS_EXCLUDED,
    REASON_POSITIVE_EVENT,
    REASON_NEGATIVE_EVENT,
    REASON_ATTACK_ALREADY_OBSERVED,
    REASON_CENSORED_INCOMPLETE_WINDOW,
    REASON_INSUFFICIENT_OBSERVED_PREFIX,
    UtteranceRecord,
    ConversationRecord,
    EventExample,
)
from .source_loader import load_cga_source_records
from .event_builder import build_event_dataset, build_conversation_event_examples

__all__ = [
    "FEATURE_COLUMN_WHITELIST",
    "FORBIDDEN_AUDIT_COLUMNS",
    "STATUS_ELIGIBLE",
    "STATUS_EXCLUDED",
    "REASON_POSITIVE_EVENT",
    "REASON_NEGATIVE_EVENT",
    "REASON_ATTACK_ALREADY_OBSERVED",
    "REASON_CENSORED_INCOMPLETE_WINDOW",
    "REASON_INSUFFICIENT_OBSERVED_PREFIX",
    "UtteranceRecord",
    "ConversationRecord",
    "EventExample",
    "load_cga_source_records",
    "build_event_dataset",
    "build_conversation_event_examples",
]
