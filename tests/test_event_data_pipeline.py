import sys
import unittest
from pathlib import Path
from typing import List, Dict, Any

# Ensure src directory is in sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from conversation_guardian.data.schemas import (
    STATUS_ELIGIBLE,
    STATUS_EXCLUDED,
    REASON_POSITIVE_EVENT,
    REASON_NEGATIVE_EVENT,
    REASON_ATTACK_ALREADY_OBSERVED,
    REASON_CENSORED_INCOMPLETE_WINDOW,
    REASON_INSUFFICIENT_OBSERVED_PREFIX,
    FEATURE_COLUMN_WHITELIST,
    FORBIDDEN_AUDIT_COLUMNS,
    UtteranceRecord,
    ConversationRecord,
    EventExample,
)
from conversation_guardian.data.event_builder import (
    filter_and_order_utterances,
    evaluate_forecast_checkpoint,
    build_conversation_event_examples,
    build_event_dataset,
)
from conversation_guardian.data.source_loader import (
    load_cga_source_records,
)


def create_synthetic_conv(
    conv_id: str,
    num_messages: int,
    attack_positions: List[int] = None,
    header_positions: List[int] = None,
    split: str = "train",
    pair_id: str = "partner_1",
) -> ConversationRecord:
    """Helper to generate deterministic synthetic conversations with controlled attributes."""
    if attack_positions is None:
        attack_positions = []
    if header_positions is None:
        header_positions = []

    utts = []
    for i in range(num_messages):
        is_hdr = i in header_positions
        is_att = i in attack_positions
        utts.append(
            UtteranceRecord(
                id=f"{conv_id}_u{i}",
                conversation_id=conv_id,
                speaker=f"speaker_{i % 2}",
                reply_to=f"{conv_id}_u{i-1}" if i > 0 else None,
                timestamp=1000.0 + (i * 10.0),
                text=f"== Section ==" if is_hdr else f"Message content {i}",
                is_section_header=is_hdr,
                comment_has_personal_attack=is_att,
                toxicity=0.9 if is_att else 0.05,
                original_index=i,
            )
        )

    has_attack = len(attack_positions) > 0
    return ConversationRecord(
        conversation_id=conv_id,
        pair_id=pair_id,
        split=split,
        conversation_has_personal_attack=has_attack,
        page_title="Test Page",
        page_id=12345,
        utterances=utts,
    )


class TestEventDataPipeline(unittest.TestCase):
    """Test suite for Gate-1 forecast target validation and data pipeline."""

    def test_01_ordering_and_section_header_treatment(self):
        """Criterion 1: Correct chronological ordering and section header removal."""
        # Create conv with section header at raw position 0 and 2
        u0 = UtteranceRecord("u0", "c1", "s1", None, 1000.0, "== Header ==", True, False)
        u1 = UtteranceRecord("u1", "c1", "s1", "u0", 1010.0, "Comment 1", False, False)
        u2 = UtteranceRecord("u2", "c1", "s2", "u1", 1020.0, "== Header 2 ==", True, False)
        u3 = UtteranceRecord("u3", "c1", "s2", "u1", 1030.0, "Comment 2", False, False)
        u4 = UtteranceRecord("u4", "c1", "s1", "u3", 1040.0, "Comment 3", False, False)

        # Unordered input
        raw = [u3, u0, u4, u2, u1]
        filtered = filter_and_order_utterances(raw, exclude_section_headers=True)

        # Must have exactly 3 comments, ordered u1 -> u3 -> u4
        self.assertEqual(len(filtered), 3)
        self.assertEqual([u.id for u in filtered], ["u1", "u3", "u4"])
        self.assertEqual([u.timestamp for u in filtered], [1010.0, 1030.0, 1040.0])

    def test_02_source_attack_annotation_mapping(self):
        """Criterion 2: Correct mapping of comment-level attack annotation."""
        # 6 dialogue messages, attack at filtered index 3
        conv = create_synthetic_conv(
            conv_id="c_att_map",
            num_messages=7,
            attack_positions=[4],
            header_positions=[0],  # Header at 0 means dialogue indices shift: 4 -> 3
        )
        filtered = filter_and_order_utterances(conv.utterances, exclude_section_headers=True)
        self.assertEqual(len(filtered), 6)
        self.assertTrue(filtered[3].comment_has_personal_attack)

        # At n_obs = 3, forecast window is [3, 4, 5], so index 3 must trigger POSITIVE_EVENT
        status, reason, label, fa_pos, fa_id = evaluate_forecast_checkpoint(
            filtered, 7, True, n_observed=3, horizon_h=3
        )
        self.assertEqual(status, STATUS_ELIGIBLE)
        self.assertEqual(reason, REASON_POSITIVE_EVENT)
        self.assertEqual(label, 1)
        self.assertEqual(fa_pos, 3)
        self.assertEqual(fa_id, "c_att_map_u4")

    def test_03_attack_occurring_before_observation_checkpoint(self):
        """Criterion 3: Attack occurring before observation checkpoint."""
        # 6 dialogue messages, attack at index 1 (inside observed prefix [0, 1, 2] for n_obs=3)
        conv = create_synthetic_conv(
            conv_id="c_prior_attack",
            num_messages=6,
            attack_positions=[1],
            header_positions=[],
        )
        filtered = filter_and_order_utterances(conv.utterances)
        status, reason, label, fa_pos, fa_id = evaluate_forecast_checkpoint(
            filtered, 6, True, n_observed=3, horizon_h=3
        )
        self.assertEqual(status, STATUS_EXCLUDED)
        self.assertEqual(reason, REASON_ATTACK_ALREADY_OBSERVED)
        self.assertIsNone(label)
        self.assertEqual(fa_pos, 1)

    def test_04_attack_occurring_at_first_future_position(self):
        """Criterion 4: Attack occurring exactly at the first future position (index = n_obs)."""
        # n_obs=3, forecast window=[3, 4, 5]. Attack at index 3
        conv = create_synthetic_conv(
            conv_id="c_att_first_future",
            num_messages=6,
            attack_positions=[3],
            header_positions=[],
        )
        filtered = filter_and_order_utterances(conv.utterances)
        status, reason, label, fa_pos, fa_id = evaluate_forecast_checkpoint(
            filtered, 6, True, n_observed=3, horizon_h=3
        )
        self.assertEqual(status, STATUS_ELIGIBLE)
        self.assertEqual(reason, REASON_POSITIVE_EVENT)
        self.assertEqual(label, 1)
        self.assertEqual(fa_pos, 3)

    def test_05_attack_occurring_at_final_position_of_forecast_window(self):
        """Criterion 5: Attack occurring at final position of forecast window (index = n_obs + 2)."""
        # n_obs=3, forecast window=[3, 4, 5]. Attack at index 5
        conv = create_synthetic_conv(
            conv_id="c_att_final_future",
            num_messages=7,
            attack_positions=[5],
            header_positions=[],
        )
        filtered = filter_and_order_utterances(conv.utterances)
        status, reason, label, fa_pos, fa_id = evaluate_forecast_checkpoint(
            filtered, 7, True, n_observed=3, horizon_h=3
        )
        self.assertEqual(status, STATUS_ELIGIBLE)
        self.assertEqual(reason, REASON_POSITIVE_EVENT)
        self.assertEqual(label, 1)
        self.assertEqual(fa_pos, 5)

    def test_06_attack_occurring_outside_forecast_window(self):
        """Criterion 6: Attack occurring outside forecast window (index > n_obs + 2)."""
        # n_obs=3, forecast window=[3, 4, 5]. Total length=8, attack at index 6
        conv = create_synthetic_conv(
            conv_id="c_att_outside_window",
            num_messages=8,
            attack_positions=[6],
            header_positions=[],
        )
        filtered = filter_and_order_utterances(conv.utterances)
        status, reason, label, fa_pos, fa_id = evaluate_forecast_checkpoint(
            filtered, 8, True, n_observed=3, horizon_h=3
        )
        # All 3 future messages (3, 4, 5) were observed and clean -> NEGATIVE_EVENT label 0
        self.assertEqual(status, STATUS_ELIGIBLE)
        self.assertEqual(reason, REASON_NEGATIVE_EVENT)
        self.assertEqual(label, 0)
        self.assertEqual(fa_pos, 6)

    def test_07_no_attack_with_complete_future_window(self):
        """Criterion 7: No attack with complete future window (L >= n_obs + 3)."""
        # n_obs=3, length=6 (indices 0, 1, 2, 3, 4, 5), no attack
        conv = create_synthetic_conv(
            conv_id="c_clean_complete",
            num_messages=6,
            attack_positions=[],
            header_positions=[],
        )
        filtered = filter_and_order_utterances(conv.utterances)
        status, reason, label, fa_pos, fa_id = evaluate_forecast_checkpoint(
            filtered, 6, False, n_observed=3, horizon_h=3
        )
        self.assertEqual(status, STATUS_ELIGIBLE)
        self.assertEqual(reason, REASON_NEGATIVE_EVENT)
        self.assertEqual(label, 0)
        self.assertIsNone(fa_pos)

    def test_08_no_attack_with_incomplete_future_window(self):
        """Criterion 8: No attack with incomplete future window (n_obs <= L < n_obs + 3)."""
        # n_obs=3, length=5 (indices 0, 1, 2, 3, 4) - only 2 future messages, not 3
        conv = create_synthetic_conv(
            conv_id="c_clean_incomplete",
            num_messages=5,
            attack_positions=[],
            header_positions=[],
        )
        filtered = filter_and_order_utterances(conv.utterances)
        status, reason, label, fa_pos, fa_id = evaluate_forecast_checkpoint(
            filtered, 5, False, n_observed=3, horizon_h=3
        )
        self.assertEqual(status, STATUS_EXCLUDED)
        self.assertEqual(reason, REASON_CENSORED_INCOMPLETE_WINDOW)
        self.assertIsNone(label)

    def test_09_conversation_shorter_than_observation_checkpoint(self):
        """Criterion 9: Conversation shorter than observation checkpoint (L < n_obs)."""
        # n_obs=5, length=4
        conv = create_synthetic_conv(
            conv_id="c_short",
            num_messages=4,
            attack_positions=[],
            header_positions=[],
        )
        filtered = filter_and_order_utterances(conv.utterances)
        status, reason, label, fa_pos, fa_id = evaluate_forecast_checkpoint(
            filtered, 4, False, n_observed=5, horizon_h=3
        )
        self.assertEqual(status, STATUS_EXCLUDED)
        self.assertEqual(reason, REASON_INSUFFICIENT_OBSERVED_PREFIX)
        self.assertIsNone(label)

    def test_10_no_conversation_or_pair_leakage_across_partitions(self):
        """Criterion 10: Partition and pair integrity on the actual corpus."""
        convs = load_cga_source_records()
        self.assertEqual(len(convs), 4188)

        splits = {"train": set(), "val": set(), "test": set()}
        pair_to_split = {}

        for cid, c in convs.items():
            self.assertIn(c.split, splits, f"Unknown split: {c.split}")
            splits[c.split].add(cid)

            # Check mutual pair
            self.assertIn(c.pair_id, convs, f"Pair {c.pair_id} not found in corpus")
            partner = convs[c.pair_id]
            self.assertEqual(
                partner.pair_id, cid, f"Pair reciprocity failed between {cid} and {c.pair_id}"
            )
            self.assertEqual(
                partner.split, c.split, f"Pair crossed splits: {cid} ({c.split}) vs {c.pair_id} ({partner.split})"
            )
            # Opposite attack labels
            self.assertNotEqual(
                c.conversation_has_personal_attack,
                partner.conversation_has_personal_attack,
                f"Paired convs have identical attack labels: {cid} and {c.pair_id}",
            )

            # Pair key
            pkey = tuple(sorted([cid, c.pair_id]))
            if pkey not in pair_to_split:
                pair_to_split[pkey] = c.split
            else:
                self.assertEqual(pair_to_split[pkey], c.split)

        # Check disjoint sets across partitions
        self.assertEqual(len(splits["train"] & splits["val"]), 0, "Train-Val conv overlap!")
        self.assertEqual(len(splits["train"] & splits["test"]), 0, "Train-Test conv overlap!")
        self.assertEqual(len(splits["val"] & splits["test"]), 0, "Val-Test conv overlap!")

        # Check total counts
        self.assertEqual(len(splits["train"]), 2508)
        self.assertEqual(len(splits["val"]), 840)
        self.assertEqual(len(splits["test"]), 840)
        self.assertEqual(len(pair_to_split), 2094)

    def test_11_reproducibility_of_dataset_construction(self):
        """Criterion 11: Dataset construction reproducibility and output determinism."""
        c1 = create_synthetic_conv("c1", 7, attack_positions=[4], header_positions=[0])
        c2 = create_synthetic_conv("c2", 6, attack_positions=[], header_positions=[])
        c3 = create_synthetic_conv("c3", 4, attack_positions=[1], header_positions=[])
        test_dict = {"c1": c1, "c2": c2, "c3": c3}

        ds1 = build_event_dataset(test_dict, checkpoints=[1, 3, 5], horizon_h=3)
        ds2 = build_event_dataset(test_dict, checkpoints=[1, 3, 5], horizon_h=3)

        self.assertEqual(len(ds1), len(ds2))
        for ex1, ex2 in zip(ds1, ds2):
            self.assertEqual(ex1.example_id, ex2.example_id)
            self.assertEqual(ex1.target_label, ex2.target_label)
            self.assertEqual(ex1.eligibility_status, ex2.eligibility_status)
            self.assertEqual(ex1.reason, ex2.reason)
            self.assertEqual(ex1.observed_prefix_text, ex2.observed_prefix_text)

    def test_12_prohibition_of_target_audit_fields_from_feature_whitelist(self):
        """Criterion 12: Strict isolation of ground truth / audit fields from feature whitelist."""
        # 1. Zero intersection between whitelist and forbidden sets
        overlap = FEATURE_COLUMN_WHITELIST & FORBIDDEN_AUDIT_COLUMNS
        self.assertEqual(len(overlap), 0, f"Whitelist contains forbidden audit columns: {overlap}")

        # 2. Test example to_feature_dict() does not contain any forbidden audit field
        conv = create_synthetic_conv("c_sec", 6, attack_positions=[4])
        examples = build_conversation_event_examples(conv, checkpoints=[3], horizon_h=3)
        self.assertEqual(len(examples), 1)
        ex = examples[0]

        feat_dict = ex.to_feature_dict()
        for forbidden in FORBIDDEN_AUDIT_COLUMNS:
            self.assertNotIn(
                forbidden,
                feat_dict,
                f"Feature dictionary leaked forbidden column: {forbidden}",
            )


if __name__ == "__main__":
    unittest.main()
