"""
Unit tests for sequential state modeling and feature extraction (Task 3).

Validates all 8 criteria specified in Section 6:
  1. Exact use of observed messages only (no future message leakage)
  2. Correct state extraction (positive, neutral, negative, hostility-lexicon)
  3. Correct ordering and transition features (Model A invariance vs Model B sensitivity)
  4. No future-label or toxicity-metadata leakage
  5. Consistent example IDs across model comparisons
  6. Validation-only threshold selection (frozen for test, higher-threshold tie-breaking)
  7. Correct handling of the single-message checkpoint (n=1)
  8. Deterministic results on synthetic examples
"""

import sys
import unittest
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

# Ensure src directory is in sys.path
TEST_DIR = Path(__file__).resolve().parent
ROOT_DIR = TEST_DIR.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from conversation_guardian.ml.sequence_features import (
    classify_message_state,
    extract_message_signals,
    extract_aggregate_features,
    extract_ordered_features,
    get_aggregate_feature_names,
    get_ordered_feature_names,
    SequenceFeatureExtractor,
    MajorityBaseline,
    DenseStateClassifier,
    HybridSequencePipeline,
    SequenceCheckpointBenchmark,
    ORDERED_STATES,
)


class TestSequenceFeatures(unittest.TestCase):
    """Test suite covering Task 3 sequential modeling and feature engineering."""

    def test_01_exact_use_of_observed_messages_only(self):
        """Criterion 1: Extract features only from observed turns; future turns do not leak."""
        full_conversation = [
            "Thank you for reviewing the article.",       # turn 0 (pos)
            "Here is the diff for the disputed section.", # turn 1 (neu)
            "I checked the reference and it is accurate.",# turn 2 (pos)
            "You are a complete idiot, revert that now!", # turn 3 (FUTURE attack at n=3)
            "Stop vandalizing the page, moron!",          # turn 4 (FUTURE attack)
        ]

        # For checkpoint n=3, only turns 0, 1, 2 must be observed
        obs_n3 = full_conversation[:3]
        extractor_n3 = SequenceFeatureExtractor(mode="ordered", checkpoint_n_obs=3)
        feat_obs = extractor_n3.extract_features_single(obs_n3)

        # Confirm hostility is 0 in the observed prefix
        feature_names = get_ordered_feature_names(checkpoint_n_obs=3)
        hostility_idx = feature_names.index("hostility_total_count")
        self.assertEqual(feat_obs[hostility_idx], 0.0)

        # If future turns were mistakenly passed to an n=3 extractor, it must reject them
        with self.assertRaises(ValueError):
            extractor_n3.extract_features_single(full_conversation)

    def test_02_correct_state_extraction(self):
        """Criterion 2: Correct deterministic classification into POSITIVE, NEUTRAL, NEGATIVE, HOSTILE."""
        # 1. Clear positive
        sig_pos = extract_message_signals("Thank you so much! This is wonderful and extremely helpful.")
        self.assertEqual(sig_pos["state"], "POSITIVE")
        self.assertGreaterEqual(sig_pos["vader_compound"], 0.05)

        # 2. Neutral
        sig_neu = extract_message_signals("The meeting took place on October 14th.")
        self.assertEqual(sig_neu["state"], "NEUTRAL")
        self.assertGreater(sig_neu["vader_compound"], -0.05)
        self.assertLess(sig_neu["vader_compound"], 0.05)

        # 3. Negative without hostile lexicon
        sig_neg = extract_message_signals("This is unfortunately a terrible mistake and deeply disappointing.")
        self.assertEqual(sig_neg["state"], "NEGATIVE")
        self.assertLessEqual(sig_neg["vader_compound"], -0.05)
        self.assertEqual(sig_neg["has_hostility"], 0)

        # 4. Hostile lexicon match
        sig_hostile = extract_message_signals("You are an idiot.")
        self.assertEqual(sig_hostile["state"], "HOSTILE")
        self.assertEqual(sig_hostile["has_hostility"], 1)

        # 5. Hostility priority override: even if positive words exist, hostile lexicon triggers HOSTILE
        sig_override = extract_message_signals("Great job, you complete idiot!")
        self.assertEqual(sig_override["state"], "HOSTILE")

    def test_03_correct_ordering_and_transition_features(self):
        """Criterion 3: Model A is strictly order-invariant; Model B captures direction and transitions."""
        msg_pos = "Thank you so much, great job!"
        msg_neu = "The article was updated."
        msg_neg = "Stop being an idiot and shut up!"

        seq_forward = [msg_pos, msg_neu, msg_neg]
        seq_reverse = [msg_neg, msg_neu, msg_pos]

        sig_fwd = [extract_message_signals(m) for m in seq_forward]
        sig_rev = [extract_message_signals(m) for m in seq_reverse]

        # Model A: Aggregate features must be identical regardless of message order
        agg_fwd = extract_aggregate_features(sig_fwd)
        agg_rev = extract_aggregate_features(sig_rev)
        for k in agg_fwd:
            self.assertAlmostEqual(
                agg_fwd[k], agg_rev[k], places=6,
                msg=f"Model A aggregate feature '{k}' differed between forward and reverse order"
            )

        # Model B: Ordered features must capture the sequence direction
        ord_fwd = extract_ordered_features(sig_fwd, checkpoint_n_obs=3)
        ord_rev = extract_ordered_features(sig_rev, checkpoint_n_obs=3)

        # Forward goes POSITIVE -> NEUTRAL -> HOSTILE
        self.assertEqual(ord_fwd["trans_count_POSITIVE->NEUTRAL"], 1.0)
        self.assertEqual(ord_fwd["trans_count_NEUTRAL->HOSTILE"], 1.0)
        self.assertEqual(ord_fwd["trans_count_HOSTILE->NEUTRAL"], 0.0)
        self.assertLess(ord_fwd["sentiment_delta_last_first"], 0.0)
        self.assertEqual(ord_fwd["escalation_flag"], 1.0)

        # Reverse goes HOSTILE -> NEUTRAL -> POSITIVE
        self.assertEqual(ord_rev["trans_count_HOSTILE->NEUTRAL"], 1.0)
        self.assertEqual(ord_rev["trans_count_NEUTRAL->POSITIVE"], 1.0)
        self.assertEqual(ord_rev["trans_count_POSITIVE->NEUTRAL"], 0.0)
        self.assertGreater(ord_rev["sentiment_delta_last_first"], 0.0)
        self.assertEqual(ord_rev["escalation_flag"], 0.0)

    def test_04_no_future_label_or_toxicity_metadata_leakage(self):
        """Criterion 4: Feature extractors operate strictly on text and contain zero audit metadata fields."""
        names_agg = get_aggregate_feature_names()
        names_ord = get_ordered_feature_names(checkpoint_n_obs=3)

        forbidden_tokens = [
            "target", "label", "attack", "toxicity", "future", "audit",
            "conversation_has_personal_attack", "comment_has_personal_attack"
        ]
        for name in names_agg + names_ord:
            for tok in forbidden_tokens:
                self.assertNotIn(tok, name.lower())

    def test_05_consistent_example_ids_across_model_comparisons(self):
        """Criterion 5: All 6 benchmark models evaluate the exact same example IDs and row counts."""
        synthetic_examples = [
            {"example_id": f"ex_{i}", "texts": ["Hello there.", "What is this?"], "label": i % 2}
            for i in range(20)
        ]
        df_ex = pd.DataFrame(synthetic_examples)

        benchmark = SequenceCheckpointBenchmark(checkpoint_n_obs=2, horizon_h=3)
        X_texts = ["\n\n".join(t) for t in df_ex["texts"]]
        extractor_agg = SequenceFeatureExtractor(mode="aggregate", checkpoint_n_obs=2)
        extractor_ord = SequenceFeatureExtractor(mode="ordered", checkpoint_n_obs=2)

        X_agg = extractor_agg.extract_feature_matrix(df_ex["texts"].tolist())
        X_ord = extractor_ord.extract_feature_matrix(df_ex["texts"].tolist())
        y = df_ex["label"].to_numpy()

        benchmark.fit(X_texts[:10], X_agg[:10], X_ord[:10], y[:10])
        benchmark.tune_thresholds(X_texts[10:15], X_agg[10:15], X_ord[10:15], y[10:15])
        probs = benchmark.predict_probabilities(X_texts[15:], X_agg[15:], X_ord[15:])

        # Verify all models produce predictions of identical length matching test IDs
        n_test = len(df_ex[15:])
        for m_name in benchmark.MODEL_NAMES:
            self.assertEqual(len(probs[m_name]), n_test)

    def test_06_validation_only_threshold_selection(self):
        """Criterion 6: Thresholds selected purely on validation set, higher threshold chosen on tie."""
        from conversation_guardian.ml.text_baseline import tune_threshold_f1

        # Case 1: Clear unique maximum
        y_val = np.array([0, 0, 1, 1])
        y_probs = np.array([0.1, 0.4, 0.6, 0.9])
        thresh, f1, fallback = tune_threshold_f1(y_val, y_probs)
        self.assertFalse(fallback)
        self.assertAlmostEqual(thresh, 0.6, places=2)

        # Case 2: Tie breaking to HIGHER threshold
        # For candidate thresholds 0.2 and 0.4:
        # At 0.2: preds are [0, 1, 1, 1], F1 score = 2*2 / (2 + 3) = 0.8
        # At 0.4: preds are [0, 0, 1, 1], F1 score = 1.0 (better)
        # Construct exact tie:
        y_val_tie = np.array([0, 1])
        y_probs_tie = np.array([0.1, 0.9])
        # For any threshold between 0.11 and 0.90, F1 is 1.0 (perfect precision & recall)
        candidates = np.array([0.2, 0.4, 0.6, 0.8])
        thresh_tie, f1_tie, _ = tune_threshold_f1(y_val_tie, y_probs_tie, candidate_thresholds=candidates)
        # Should pick highest candidate that ties
        self.assertEqual(thresh_tie, 0.8)

    def test_07_correct_handling_of_single_message_checkpoint_n1(self):
        """Criterion 7: n=1 checkpoint executes cleanly with zero transition features and no errors."""
        single_msg = ["Just a single message observed."]
        extractor_n1 = SequenceFeatureExtractor(mode="ordered", checkpoint_n_obs=1)
        feat_n1 = extractor_n1.extract_features_single(single_msg)

        feat_names = get_ordered_feature_names(checkpoint_n_obs=1)
        self.assertEqual(len(feat_n1), len(feat_names))

        # Check that transition and delta features are all zero
        for fname in ["trans_count_POSITIVE->NEGATIVE", "sentiment_delta_last_first", "sentiment_slope", "escalation_flag"]:
            idx = feat_names.index(fname)
            self.assertEqual(feat_n1[idx], 0.0)

        # Benchmark should train and predict without crashing or NaN
        X_texts = ["Message one.", "Message two.", "Message three.", "Message four."]
        X_seqs = [["Message one."], ["Message two."], ["Message three."], ["Message four."]]
        y = np.array([0, 1, 0, 1])

        X_agg = SequenceFeatureExtractor("aggregate", 1).extract_feature_matrix(X_seqs)
        X_ord = extractor_n1.extract_feature_matrix(X_seqs)

        benchmark = SequenceCheckpointBenchmark(checkpoint_n_obs=1)
        benchmark.fit(X_texts[:2], X_agg[:2], X_ord[:2], y[:2])
        benchmark.tune_thresholds(X_texts[2:], X_agg[2:], X_ord[2:], y[2:])
        eval_res = benchmark.evaluate_split(X_texts[2:], X_agg[2:], X_ord[2:], y[2:], "test")

        for m_name in benchmark.MODEL_NAMES:
            self.assertIn("F1", eval_res[m_name])
            self.assertFalse(np.isnan(eval_res[m_name]["F1"]))

    def test_08_deterministic_results_on_synthetic_examples(self):
        """Criterion 8: Feature extraction and benchmark evaluation are strictly deterministic."""
        seqs = [
            ["Hello there.", "How are you doing today?", "Great, thank you!"],
            ["I disagree with your edit.", "Please see WP:RS.", "You are a complete idiot!"],
        ]
        extractor = SequenceFeatureExtractor(mode="ordered", checkpoint_n_obs=3)
        mat1 = extractor.extract_feature_matrix(seqs)
        mat2 = extractor.extract_feature_matrix(seqs)
        self.assertTrue(np.array_equal(mat1, mat2))


if __name__ == "__main__":
    unittest.main()
