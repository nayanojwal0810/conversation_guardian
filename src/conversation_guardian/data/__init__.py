"""Data preparation and validation module for ConversationGuardian."""

from .schemas import (
    MODEL_INPUT_ALLOWLIST,
    FEATURE_COLUMN_WHITELIST,
    METADATA_COLUMNS,
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
from .source_loader import (
    load_cga_source_records,
    validate_pair_integrity,
    validate_attack_annotation_consistency,
    validate_section_headers,
)
from .event_builder import build_event_dataset, build_conversation_event_examples

__all__ = [
    "MODEL_INPUT_ALLOWLIST",
    "FEATURE_COLUMN_WHITELIST",
    "METADATA_COLUMNS",
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
    "validate_pair_integrity",
    "validate_attack_annotation_consistency",
    "validate_section_headers",
    "build_event_dataset",
    "build_conversation_event_examples",
]
