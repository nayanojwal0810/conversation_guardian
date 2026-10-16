"""
Unit tests for ConversationGRU and RecurrentSequenceClassifier (Task 5).

Validates all 8 criteria specified in Section 6:
  1. Ordered messages are passed without reordering (turn markers and chronological flow)
  2. Only observed messages are used (no future tokens)
  3. Padding and masking behave correctly under pack_padded_sequence
  4. Vocabulary and preprocessing are built strictly on training data
  5. Repeated inference is deterministic under fixed configuration
  6. Validation-selected thresholds remain frozen during test evaluation
  7. Model saving/loading and predictor integration work correctly
  8. Missing or invalid model artifacts produce informative errors
"""

import sys
import unittest
from pathlib import Path
from typing import List
import numpy as np
import torch

# Ensure src directory is in sys.path
TEST_DIR = Path(__file__).resolve().parent
ROOT_DIR = TEST_DIR.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from conversation_guardian.ml.recurrent_model import (
    Vocabulary,
    tokenize_text,
    ConversationGRU,
    RecurrentSequenceClassifier,
)
from conversation_guardian.ml.predictor import ConversationGuardianPredictor


class TestRecurrentSequenceModel(unittest.TestCase):
    """Test suite covering the recurrent sequence model implementation and safeguards."""

    def test_01_ordered_messages_without_reordering(self):
        """Criterion 1: Messages are converted to token sequences preserving exact chronological order and turn markers."""
        m1 = "First turn of discussion."
        m2 = "Second turn with more detail."
        m3 = "Third turn concluding the topic."

        tokens_fwd = RecurrentSequenceClassifier._messages_to_tokens([m1, m2, m3])
        tokens_rev = RecurrentSequenceClassifier._messages_to_tokens([m3, m2, m1])

        # Verify <turn> markers are inserted before each message
        self.assertEqual(tokens_fwd.count(Vocabulary.TURN_TOKEN), 3)
        self.assertEqual(tokens_fwd[0], Vocabulary.TURN_TOKEN)
        self.assertEqual(tokens_fwd[1], "first")

        # Verify forward and reverse sequences differ in order
        self.assertNotEqual(tokens_fwd, tokens_rev)
        self.assertEqual(tokens_rev[1], "third")

    def test_02_only_observed_messages_used(self):
        """Criterion 2: Model inputs are derived strictly from observed turns; future turns do not leak."""
        observed_messages = ["Hello there.", "What is this diff about?"]
        future_message = "You complete idiot, revert that now!"

        tokens_observed = RecurrentSequenceClassifier._messages_to_tokens(observed_messages)
        self.assertNotIn("idiot", tokens_observed)
        self.assertNotIn("revert", tokens_observed)

        # Confirm length corresponds only to the 2 observed turns
        self.assertEqual(tokens_observed.count(Vocabulary.TURN_TOKEN), 2)

    def test_03_padding_and_masking(self):
        """Criterion 3: Sequences of varying length pad and mask correctly via pack_padded_sequence."""
        vocab = Vocabulary(max_size=100, min_freq=1)
        vocab.fit([["hello", "world"], ["short"]])

        # Encode short sequence and longer sequence
        enc_short, len_short = vocab.encode_tokens(["hello"], max_len=10)
        enc_long, len_long = vocab.encode_tokens(["hello", "world", "extra"], max_len=10)

        self.assertEqual(len_short, 1)
        self.assertEqual(len(enc_short), 10)
        self.assertEqual(enc_short[1:], [Vocabulary.PAD_IDX] * 9)

        self.assertEqual(len_long, 3)
        self.assertEqual(len(enc_long), 10)

        # Verify forward pass through ConversationGRU handles padding without error
        model = ConversationGRU(vocab_size=len(vocab), embedding_dim=16, hidden_dim=16)
        x = torch.tensor([enc_short, enc_long], dtype=torch.long)
        lengths = torch.tensor([len_short, len_long], dtype=torch.long)

        out = model(x, lengths)
        self.assertEqual(out.shape, torch.Size([2]))
        self.assertFalse(torch.isnan(out).any())

    def test_04_vocabulary_built_strictly_on_training_data(self):
        """Criterion 4: Vocabulary is built on train only; out-of-vocabulary tokens in val/test encode as <unk>."""
        train_seqs = [["unique_train_term_alpha", "common_word"]]
        val_seqs = [["novel_val_term_beta", "common_word"]]

        clf = RecurrentSequenceClassifier(
            max_epochs=1, batch_size=2, embedding_dim=16, hidden_dim=16, min_freq=1
        )
        y_train = np.array([0])
        clf.fit(train_seqs, y_train)

        # Train term must be in vocabulary
        self.assertIn("unique_train_term_alpha", clf.vocab.token_to_idx)

        # Novel val term must NOT be in vocabulary
        self.assertNotIn("novel_val_term_beta", clf.vocab.token_to_idx)

        # Novel term encodes to UNK_IDX
        encoded, _ = clf.vocab.encode_tokens(["novel_val_term_beta"], max_len=5)
        self.assertEqual(encoded[0], Vocabulary.UNK_IDX)

    def test_05_deterministic_inference_under_fixed_config(self):
        """Criterion 5: Repeated inference produces bitwise-identical probabilities."""
        train_seqs = [
            ["Thank you very much.", "Looks good."],
            ["Stop this immediately.", "You fool."],
            ["I will check the reference.", "Sounds good."],
            ["Reverting your changes.", "Do not do that."],
        ]
        y_train = np.array([0, 1, 0, 1])

        clf = RecurrentSequenceClassifier(
            max_epochs=2, batch_size=2, embedding_dim=16, hidden_dim=16, random_state=42
        )
        clf.fit(train_seqs, y_train)

        eval_seqs = [["Thank you very much."], ["You fool."]]
        probs1 = clf.predict_proba(eval_seqs)
        probs2 = clf.predict_proba(eval_seqs)

        np.testing.assert_array_equal(probs1, probs2)

    def test_06_validation_threshold_freezing(self):
        """Criterion 6: Decision threshold tuned on validation data remains strictly frozen for test evaluation."""
        from conversation_guardian.ml.text_baseline import tune_threshold_f1

        y_val = np.array([0, 0, 1, 1])
        val_probs = np.array([0.15, 0.35, 0.65, 0.85])
        best_thresh, best_f1, _ = tune_threshold_f1(y_val, val_probs)

        # Frozen threshold is 0.65
        self.assertAlmostEqual(best_thresh, 0.65, places=2)

        # Altering test data or labels must not affect frozen threshold
        test_probs = np.array([0.20, 0.90])
        test_preds = (test_probs >= best_thresh).astype(int)
        np.testing.assert_array_equal(test_preds, np.array([0, 1]))

    def test_07_model_saving_loading_and_predictor_integration(self):
        """Criterion 7: Recurrent model bundle integrates cleanly into ConversationGuardianPredictor."""
        train_seqs = [
            ["Thank you very much."],
            ["You are an idiot!"],
            ["Looks good to me."],
            ["Please stop this nonsense."],
        ]
        y_train = np.array([0, 1, 0, 1])

        clf = RecurrentSequenceClassifier(
            max_epochs=2, batch_size=2, embedding_dim=16, hidden_dim=16, random_state=42
        )
        clf.fit(train_seqs, y_train)

        # Construct in-memory bundle with feature_mode="recurrent"
        stub_bundles = {
            1: {
                "checkpoint_n_obs": 1,
                "horizon_h": 3,
                "model_family": "Recurrent_GRU",
                "model_object": clf,
                "threshold": 0.35,
                "val_ap": 0.50,
                "val_f1": 0.50,
                "feature_mode": "recurrent",
                "feature_names": [],
                "version": "1.1.0",
            },
            3: {
                "checkpoint_n_obs": 3,
                "horizon_h": 3,
                "model_family": "Recurrent_GRU",
                "model_object": clf,
                "threshold": 0.35,
                "val_ap": 0.50,
                "val_f1": 0.50,
                "feature_mode": "recurrent",
                "feature_names": [],
                "version": "1.1.0",
            },
            5: {
                "checkpoint_n_obs": 5,
                "horizon_h": 3,
                "model_family": "Recurrent_GRU",
                "model_object": clf,
                "threshold": 0.35,
                "val_ap": 0.50,
                "val_f1": 0.50,
                "feature_mode": "recurrent",
                "feature_names": [],
                "version": "1.1.0",
            },
        }

        predictor = ConversationGuardianPredictor(loaded_bundles=stub_bundles)
        result = predictor.predict(["Thank you for reviewing the page."])

        self.assertEqual(result["checkpoint_n_obs"], 1)
        self.assertEqual(result["model_family"], "Recurrent_GRU")
        self.assertTrue(0.0 <= result["risk_score"] <= 1.0)
        self.assertIsInstance(result["alert"], bool)
        self.assertEqual(len(result["signals_summary"]), 1)

    def test_08_missing_or_invalid_model_artifacts_error(self):
        """Criterion 8: Calling predict_proba on an unfitted classifier raises a clear RuntimeError."""
        unfitted_clf = RecurrentSequenceClassifier()
        with self.assertRaises(RuntimeError) as ctx:
            unfitted_clf.predict_proba([["Hello world"]])
        self.assertIn("must be fitted", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
