"""Data structures, constants, and schema definitions for ConversationGuardian."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# Eligibility statuses
STATUS_ELIGIBLE = "ELIGIBLE"
STATUS_EXCLUDED = "EXCLUDED"

# Exclusion and labeling reasons
REASON_POSITIVE_EVENT = "POSITIVE_EVENT"
REASON_NEGATIVE_EVENT = "NEGATIVE_EVENT"
REASON_ATTACK_ALREADY_OBSERVED = "ATTACK_ALREADY_OBSERVED"
REASON_CENSORED_INCOMPLETE_WINDOW = "CENSORED_INCOMPLETE_WINDOW"
REASON_INSUFFICIENT_OBSERVED_PREFIX = "INSUFFICIENT_OBSERVED_PREFIX"

# Approved direct model-input predictor columns permitted for downstream model consumption
MODEL_INPUT_ALLOWLIST = frozenset([
    "observed_prefix_text",
])

# For backward compatibility, alias FEATURE_COLUMN_WHITELIST to the exact model input allowlist
FEATURE_COLUMN_WHITELIST = MODEL_INPUT_ALLOWLIST

# Metadata columns retained for tracking, joins, partitioning, or offline debugging,
# but NEVER passed as direct model predictors.
METADATA_COLUMNS = frozenset([
    "example_id",
    "conversation_id",
    "pair_id",
    "split",
    "checkpoint_n_obs",
    "horizon_h",
    "observed_prefix_length",
    "observed_utterance_ids",
    "observed_speakers",
    "observed_timestamps",
    "total_conversation_length_raw",
    "total_dialogue_length_filtered",
    "future_messages_available",
    "dataset_version",
])

# Forbidden audit / ground truth target columns strictly prohibited from model features
FORBIDDEN_AUDIT_COLUMNS = frozenset([
    "target_label",
    "eligibility_status",
    "reason",
    "has_attack_in_conversation",
    "first_attack_pos",
    "first_attack_utterance_id",
    "censoring_reason",
])


@dataclass
class UtteranceRecord:
    """Representation of a single utterance loaded from the official corpus."""
    id: str
    conversation_id: str
    speaker: str
    reply_to: Optional[str]
    timestamp: float
    text: str
    is_section_header: bool
    comment_has_personal_attack: bool
    toxicity: float = 0.0
    original_index: int = 0


@dataclass
class ConversationRecord:
    """Representation of an entire conversation thread."""
    conversation_id: str
    pair_id: str
    split: str
    conversation_has_personal_attack: bool
    page_title: str = ""
    page_id: int = 0
    utterances: List[UtteranceRecord] = field(default_factory=list)


@dataclass
class EventExample:
    """An event forecasting observation example at a fixed checkpoint."""
    # Tracking identifiers and metadata
    example_id: str
    conversation_id: str
    pair_id: str
    split: str
    checkpoint_n_obs: int
    horizon_h: int
    observed_prefix_length: int
    observed_prefix_text: str
    observed_utterance_ids: List[str]
    observed_speakers: List[str]
    observed_timestamps: List[float]

    # Offline target and audit metadata (strictly separated from features)
    target_label: Optional[int]
    eligibility_status: str
    reason: str
    has_attack_in_conversation: bool
    first_attack_pos: Optional[int]
    first_attack_utterance_id: Optional[str]
    total_conversation_length_raw: int
    total_dialogue_length_filtered: int
    future_messages_available: int
    dataset_version: str = "1.0.0-gate1"

    def to_feature_dict(self) -> Dict[str, Any]:
        """
        Return exclusively the exact approved direct model-input predictor fields.
        Identifiers, split names, timestamps, speaker identities, future metrics,
        and target labels are strictly excluded.
        """
        return {
            "observed_prefix_text": self.observed_prefix_text,
        }

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Return tracking identifiers and metadata without target labels."""
        return {
            "example_id": self.example_id,
            "conversation_id": self.conversation_id,
            "pair_id": self.pair_id,
            "split": self.split,
            "checkpoint_n_obs": self.checkpoint_n_obs,
            "horizon_h": self.horizon_h,
            "observed_prefix_length": self.observed_prefix_length,
            "observed_utterance_ids": self.observed_utterance_ids,
            "observed_speakers": self.observed_speakers,
            "observed_timestamps": self.observed_timestamps,
            "total_conversation_length_raw": self.total_conversation_length_raw,
            "total_dialogue_length_filtered": self.total_dialogue_length_filtered,
            "future_messages_available": self.future_messages_available,
            "dataset_version": self.dataset_version,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to complete record representation for audit and parquet persistence."""
        return {
            "example_id": self.example_id,
            "conversation_id": self.conversation_id,
            "pair_id": self.pair_id,
            "split": self.split,
            "checkpoint_n_obs": self.checkpoint_n_obs,
            "horizon_h": self.horizon_h,
            "observed_prefix_length": self.observed_prefix_length,
            "observed_prefix_text": self.observed_prefix_text,
            "observed_utterance_ids": self.observed_utterance_ids,
            "observed_speakers": self.observed_speakers,
            "observed_timestamps": self.observed_timestamps,
            "target_label": self.target_label,
            "eligibility_status": self.eligibility_status,
            "reason": self.reason,
            "has_attack_in_conversation": self.has_attack_in_conversation,
            "first_attack_pos": self.first_attack_pos,
            "first_attack_utterance_id": self.first_attack_utterance_id,
            "total_conversation_length_raw": self.total_conversation_length_raw,
            "total_dialogue_length_filtered": self.total_dialogue_length_filtered,
            "future_messages_available": self.future_messages_available,
            "dataset_version": self.dataset_version,
        }
