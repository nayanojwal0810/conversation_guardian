"""Event-forecasting dataset builder with strict boundary and censoring logic."""

from typing import List, Dict, Optional, Tuple
import pandas as pd

from .schemas import (
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


def filter_and_order_utterances(
    utterances: List[UtteranceRecord],
    exclude_section_headers: bool = True
) -> List[UtteranceRecord]:
    """
    Sort utterances chronologically by timestamp (stable sort preserving file order for ties).
    If exclude_section_headers is True, filter out utterances marked with is_section_header.
    Message positions 0 ... L-1 are assigned after this filtering.
    """
    # Python's Timsort is stable: ties in timestamp preserve original file order
    sorted_utts = sorted(utterances, key=lambda u: u.timestamp)
    if exclude_section_headers:
        return [u for u in sorted_utts if not u.is_section_header]
    return sorted_utts


def evaluate_forecast_checkpoint(
    filtered_utterances: List[UtteranceRecord],
    total_raw_length: int,
    conv_has_attack: bool,
    n_observed: int,
    horizon_h: int = 3,
) -> Tuple[str, str, Optional[int], Optional[int], Optional[str]]:
    """
    Evaluate eligibility and target label for a conversation at a fixed checkpoint.

    Returns:
        (eligibility_status, reason, target_label, first_attack_pos, first_attack_utterance_id)
    """
    L = len(filtered_utterances)

    # Locate all attack positions in filtered dialogue
    attack_positions = [
        (idx, u.id)
        for idx, u in enumerate(filtered_utterances)
        if u.comment_has_personal_attack
    ]
    first_attack_pos = attack_positions[0][0] if attack_positions else None
    first_attack_id = attack_positions[0][1] if attack_positions else None

    # Rule 1: Insufficient observed prefix
    if L < n_observed:
        return (
            STATUS_EXCLUDED,
            REASON_INSUFFICIENT_OBSERVED_PREFIX,
            None,
            first_attack_pos,
            first_attack_id,
        )

    # Rule 2: Attack already observed inside the observation window [0, n_observed - 1]
    if first_attack_pos is not None and first_attack_pos < n_observed:
        return (
            STATUS_EXCLUDED,
            REASON_ATTACK_ALREADY_OBSERVED,
            None,
            first_attack_pos,
            first_attack_id,
        )

    # Rule 3: Positive event inside forecast window [n_observed, n_observed + horizon_h - 1]
    window_end = n_observed + horizon_h - 1
    if first_attack_pos is not None and n_observed <= first_attack_pos <= window_end:
        # Attack occurs within the forecast window (even if conversation subsequently ends prematurely)
        return (
            STATUS_ELIGIBLE,
            REASON_POSITIVE_EVENT,
            1,
            first_attack_pos,
            first_attack_id,
        )

    # Rule 4 & 5: No attack occurred up to min(L - 1, window_end)
    # Check if the full future window was observed
    required_length = n_observed + horizon_h
    if L >= required_length:
        # All horizon_h future messages are observed and none has an attack
        return (
            STATUS_ELIGIBLE,
            REASON_NEGATIVE_EVENT,
            0,
            first_attack_pos,
            first_attack_id,
        )
    else:
        # Conversation ended before the full horizon_h future messages could be observed
        return (
            STATUS_EXCLUDED,
            REASON_CENSORED_INCOMPLETE_WINDOW,
            None,
            first_attack_pos,
            first_attack_id,
        )


def build_conversation_event_examples(
    conv: ConversationRecord,
    checkpoints: List[int] = [1, 3, 5],
    horizon_h: int = 3,
    exclude_section_headers: bool = True,
    dataset_version: str = "1.0.0-gate1",
) -> List[EventExample]:
    """Build EventExample objects for all specified checkpoints for a single conversation."""
    filtered_utts = filter_and_order_utterances(
        conv.utterances,
        exclude_section_headers=exclude_section_headers
    )
    total_raw_len = len(conv.utterances)
    total_filt_len = len(filtered_utts)

    examples: List[EventExample] = []

    for n_obs in checkpoints:
        status, reason, label, fa_pos, fa_id = evaluate_forecast_checkpoint(
            filtered_utterances=filtered_utts,
            total_raw_length=total_raw_len,
            conv_has_attack=conv.conversation_has_personal_attack,
            n_observed=n_obs,
            horizon_h=horizon_h,
        )

        # Slice observed prefix: strictly positions 0 ... min(L, n_obs) - 1
        observed_slice = filtered_utts[:n_obs] if total_filt_len >= n_obs else filtered_utts
        observed_texts = [u.text for u in observed_slice]
        observed_prefix_text = "\n\n".join(observed_texts)
        observed_uids = [u.id for u in observed_slice]
        observed_speakers = [u.speaker for u in observed_slice]
        observed_timestamps = [u.timestamp for u in observed_slice]
        future_avail = max(0, total_filt_len - n_obs)

        example_id = f"{conv.conversation_id}__obs{n_obs}_h{horizon_h}"

        example = EventExample(
            example_id=example_id,
            conversation_id=conv.conversation_id,
            pair_id=conv.pair_id,
            split=conv.split,
            checkpoint_n_obs=n_obs,
            horizon_h=horizon_h,
            observed_prefix_length=len(observed_slice),
            observed_prefix_text=observed_prefix_text,
            observed_utterance_ids=observed_uids,
            observed_speakers=observed_speakers,
            observed_timestamps=observed_timestamps,
            target_label=label,
            eligibility_status=status,
            reason=reason,
            has_attack_in_conversation=conv.conversation_has_personal_attack,
            first_attack_pos=fa_pos,
            first_attack_utterance_id=fa_id,
            total_conversation_length_raw=total_raw_len,
            total_dialogue_length_filtered=total_filt_len,
            future_messages_available=future_avail,
            dataset_version=dataset_version,
        )
        examples.append(example)

    return examples


def build_event_dataset(
    conversations: Dict[str, ConversationRecord],
    checkpoints: List[int] = [1, 3, 5],
    horizon_h: int = 3,
    exclude_section_headers: bool = True,
    dataset_version: str = "1.0.0-gate1",
) -> List[EventExample]:
    """Build EventExample objects for all conversations in the corpus."""
    all_examples: List[EventExample] = []
    # Sort conversation IDs for complete determinism
    for cid in sorted(conversations.keys()):
        conv = conversations[cid]
        exs = build_conversation_event_examples(
            conv=conv,
            checkpoints=checkpoints,
            horizon_h=horizon_h,
            exclude_section_headers=exclude_section_headers,
            dataset_version=dataset_version,
        )
        all_examples.extend(exs)
    return all_examples


def examples_to_dataframe(examples: List[EventExample]) -> pd.DataFrame:
    """Convert a list of EventExample objects to a Pandas DataFrame."""
    return pd.DataFrame([ex.to_dict() for ex in examples])
