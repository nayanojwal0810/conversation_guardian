import sys
import unittest
import tempfile
import json
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
    MODEL_INPUT_ALLOWLIST,
    METADATA_COLUMNS,
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
    validate_pair_integrity,
    validate_attack_annotation_consistency,
    validate_section_headers,
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
    """Test suite for Gate-1 forecast target validation, boundary conditions, and integrity checks."""

    def test_01_attack_at_first_position_of_forecast_window(self):
        """Criterion 1: Attack occurring exactly at the first future position (index = n_obs)."""
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

    def test_02_attack_at_final_position_of_forecast_window(self):
        """Criterion 2: Attack occurring at final position of forecast window (index = n_obs + 2)."""
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

    def test_03_attack_outside_forecast_window(self):
        """Criterion 3: Attack occurring outside forecast window (index > n_obs + 2)."""
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

    def test_04_attack_in_window_with_premature_termination(self):
        """Criterion 4: Attack within forecast window, but dialogue terminates prematurely (positive holds)."""
        # n_obs=3, forecast window=[3, 4, 5]. Thread terminates at turn 4 (only 1 future message)
        conv = create_synthetic_conv(
            conv_id="c_att_early_term",
            num_messages=4,
            attack_positions=[3],
            header_positions=[],
        )
        filtered = filter_and_order_utterances(conv.utterances)
        status, reason, label, fa_pos, fa_id = evaluate_forecast_checkpoint(
            filtered, 4, True, n_observed=3, horizon_h=3
        )
        self.assertEqual(status, STATUS_ELIGIBLE)
        self.assertEqual(reason, REASON_POSITIVE_EVENT)
        self.assertEqual(label, 1)
        self.assertEqual(fa_pos, 3)

    def test_05_no_attack_with_complete_future_window(self):
        """Criterion 5: No attack with complete future window (L >= n_obs + 3)."""
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

    def test_06_no_attack_with_incomplete_future_window_censored(self):
        """Criterion 6: No attack with incomplete future window (n_obs <= L < n_obs + 3)."""
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

    def test_07_attack_already_observed_in_prefix(self):
        """Criterion 7: Attack occurring before observation checkpoint."""
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

    def test_08_feature_dict_exact_allowlist(self):
        """Criterion 8: Feature dictionary contains only the exact approved model-input fields."""
        conv = create_synthetic_conv("c_feat", 6, attack_positions=[4])
        examples = build_conversation_event_examples(conv, checkpoints=[3], horizon_h=3)
        self.assertEqual(len(examples), 1)
        ex = examples[0]

        feat_dict = ex.to_feature_dict()

        # 1. Exact match against approved model input allowlist
        self.assertEqual(set(feat_dict.keys()), set(MODEL_INPUT_ALLOWLIST))
        self.assertEqual(set(feat_dict.keys()), {"observed_prefix_text"})

        # 2. Assert all identifiers and partition fields are excluded
        excluded_fields = [
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
            "target_label",
            "first_attack_pos",
            "first_attack_utterance_id",
            "total_conversation_length_raw",
            "total_dialogue_length_filtered",
            "future_messages_available",
            "censoring_reason",
        ]
        for field in excluded_fields:
            self.assertNotIn(field, feat_dict, f"Direct model inputs leaked metadata/audit field: {field}")

    def test_09_invalid_source_dir_fails_fast(self):
        """Criterion 9: Explicitly invalid source directory raises FileNotFoundError without silent fallback."""
        with self.assertRaises(FileNotFoundError) as ctx:
            load_cga_source_records(corpus_dir="/nonexistent/directory/path/invalid_12345")
        self.assertIn("does not exist or is not a directory", str(ctx.exception))

    def test_10_source_validation_checks(self):
        """Criterion 10: Source annotation inconsistencies, missing conversation references, and pair flaws fail validation."""
        # 10a. Attack in healthy conversation
        c_bad_label = create_synthetic_conv("c_bad", 5, attack_positions=[2])
        c_bad_label.conversation_has_personal_attack = False  # Mismatch: has attack comment but marked healthy
        with self.assertRaises(ValueError) as ctx1:
            validate_attack_annotation_consistency({"c_bad": c_bad_label})
        self.assertIn("Attack annotation concordance violation", str(ctx1.exception))

        # 10b. Attacked conversation with zero attack comments
        c_no_att = create_synthetic_conv("c_no_att", 5, attack_positions=[])
        c_no_att.conversation_has_personal_attack = True  # Mismatch: marked attack but no attack comments
        with self.assertRaises(ValueError) as ctx2:
            validate_attack_annotation_consistency({"c_no_att": c_no_att})
        self.assertIn("Attack annotation concordance violation", str(ctx2.exception))

        # 10c. Pair reciprocity failure
        c_pair_a = create_synthetic_conv("c_pair_a", 5, pair_id="c_pair_b")
        c_pair_b = create_synthetic_conv("c_pair_b", 5, pair_id="c_pair_wrong")
        with self.assertRaises(ValueError) as ctx3:
            validate_pair_integrity({"c_pair_a": c_pair_a, "c_pair_b": c_pair_b})
        self.assertIn("Pair reciprocity failure", str(ctx3.exception))

        # 10d. Missing conversation ID in utterances fails fast
        with tempfile.TemporaryDirectory() as tmpdir:
            conv_path = Path(tmpdir) / "conversations.json"
            utt_path = Path(tmpdir) / "utterances.jsonl"
            with open(conv_path, "w", encoding="utf-8") as f:
                json.dump({
                    "c1": {
                        "split": "train",
                        "pair_id": "c2",
                        "conversation_has_personal_attack": False,
                    },
                    "c2": {
                        "split": "train",
                        "pair_id": "c1",
                        "conversation_has_personal_attack": True,
                    }
                }, f)
            with open(utt_path, "w", encoding="utf-8") as f:
                # Utterance referencing non-existent conversation c_ghost
                f.write(json.dumps({
                    "id": "u_bad",
                    "conversation_id": "c_ghost",
                    "speaker": "s1",
                    "timestamp": 1000.0,
                    "text": "orphaned utterance"
                }) + "\n")
            with self.assertRaises(ValueError) as ctx4:
                load_cga_source_records(corpus_dir=tmpdir)
            self.assertIn("referencing missing or unknown conversation IDs", str(ctx4.exception))

    def test_11_corpus_pair_integrity_and_split_preservation(self):
        """Criterion 11: Real corpus verification for pair reciprocity, split preservation, and label opposition."""
        convs = load_cga_source_records()
        self.assertEqual(len(convs), 4188)

        splits = {"train": set(), "val": set(), "test": set()}
        pair_to_split = {}

        for cid, c in convs.items():
            self.assertIn(c.split, splits)
            splits[c.split].add(cid)

            # Check mutual pair
            self.assertIn(c.pair_id, convs)
            partner = convs[c.pair_id]
            self.assertEqual(partner.pair_id, cid)
            self.assertEqual(partner.split, c.split)
            self.assertNotEqual(c.conversation_has_personal_attack, partner.conversation_has_personal_attack)

            pkey = tuple(sorted([cid, c.pair_id]))
            pair_to_split[pkey] = c.split

        self.assertEqual(len(splits["train"]), 2508)
        self.assertEqual(len(splits["val"]), 840)
        self.assertEqual(len(splits["test"]), 840)
        self.assertEqual(len(pair_to_split), 2094)

    def test_12_dataset_construction_determinism(self):
        """Criterion 12: Dataset construction reproducibility and determinism."""
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
            self.assertEqual(ex1.to_feature_dict(), ex2.to_feature_dict())

    def test_13_ordering_and_section_header_treatment(self):
        """Criterion 13: Correct chronological ordering and section header removal."""
        u0 = UtteranceRecord("u0", "c1", "s1", None, 1000.0, "== Header ==", True, False)
        u1 = UtteranceRecord("u1", "c1", "s1", "u0", 1010.0, "Comment 1", False, False)
        u2 = UtteranceRecord("u2", "c1", "s2", "u1", 1020.0, "== Header 2 ==", True, False)
        u3 = UtteranceRecord("u3", "c1", "s2", "u1", 1030.0, "Comment 2", False, False)
        u4 = UtteranceRecord("u4", "c1", "s1", "u3", 1040.0, "Comment 3", False, False)

        raw = [u3, u0, u4, u2, u1]
        filtered = filter_and_order_utterances(raw, exclude_section_headers=True)

        self.assertEqual(len(filtered), 3)
        self.assertEqual([u.id for u in filtered], ["u1", "u3", "u4"])
        self.assertEqual([u.timestamp for u in filtered], [1010.0, 1030.0, 1040.0])

    def test_14_conversation_shorter_than_observation_checkpoint(self):
        """Criterion 14: Conversation shorter than observation checkpoint (L < n_obs)."""
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


if __name__ == "__main__":
    unittest.main()
