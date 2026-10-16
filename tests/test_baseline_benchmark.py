"""Unit tests for ML text baseline benchmark, threshold tuning, and leak-free evaluation."""

import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

# Ensure src directory is in sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from conversation_guardian.ml.text_baseline import (
    build_tfidf_pipeline,
    tune_threshold_f1,
    compute_benchmark_metrics,
    PriorBaseline,
    CheckpointBenchmark,
)
from scripts.run_baseline_benchmark import load_and_filter_events


class TestBaselineBenchmark(unittest.TestCase):
    """Test suite for baseline modeling, threshold freezing, and leak-free execution."""

    def setUp(self):
        # Synthetic data for testing
        self.synthetic_texts = [
            "You are doing great work on Wikipedia",
            "Please cite reliable secondary sources",
            "Stop reverting my edits you vandal",
            "This article needs significant improvement",
            "Your comment is completely idiotic and abusive",
            "Thank you for the helpful feedback",
            "Get lost and stop bothering the editors",
            "Let's discuss the proposed changes on talk page",
            "You are a terrible contributor and should be banned",
            "I agree with the suggested consensus",
        ]
        self.synthetic_labels = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=int)

    def test_01_exclusion_of_censored_and_ineligible_examples(self):
        """Test that data loader strictly filters out ineligible and censored examples."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            df = pd.DataFrame({
                "example_id": ["e1", "e2", "e3", "e4"],
                "observed_prefix_text": ["text 1", "text 2", "text 3", "text 4"],
                "eligibility_status": ["ELIGIBLE", "EXCLUDED", "ELIGIBLE", "EXCLUDED"],
                "target_label": [1.0, None, 0.0, None],
                "reason": ["POSITIVE_EVENT", "CENSORED_INCOMPLETE_WINDOW", "NEGATIVE_EVENT", "ATTACK_ALREADY_OBSERVED"],
            })
            df.to_parquet(tmp_path / "test_events.parquet", index=False)

            filtered = load_and_filter_events(tmp_path, "test")
            self.assertEqual(len(filtered), 2)
            self.assertEqual(filtered["example_id"].tolist(), ["e1", "e3"])
            self.assertEqual(filtered["target_label"].tolist(), [1, 0])
            self.assertTrue(all(status == "ELIGIBLE" for status in filtered["eligibility_status"]))

    def test_02_use_of_observed_text_only(self):
        """Test that the model pipeline strictly accepts text input and operates without metadata."""
        pipeline = build_tfidf_pipeline()
        X_train = self.synthetic_texts[:8]
        y_train = self.synthetic_labels[:8]

        # Pipeline fits exclusively on string list/series
        pipeline.fit(X_train, y_train)

        # Predicts on new text list
        X_test = self.synthetic_texts[8:]
        probs = pipeline.predict_proba(X_test)[:, 1]
        self.assertEqual(len(probs), 2)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_03_independent_model_training_per_checkpoint(self):
        """Test that CheckpointBenchmark models are completely isolated and independent per checkpoint."""
        bm_n1 = CheckpointBenchmark(checkpoint_n_obs=1, horizon_h=3, random_state=42)
        bm_n3 = CheckpointBenchmark(checkpoint_n_obs=3, horizon_h=3, random_state=42)

        X_n1 = ["Hello editor", "Thanks", "Get lost idiot"]
        y_n1 = np.array([0, 0, 1])
        X_n3 = ["Longer discussion turn 1\n\nTurn 2\n\nTurn 3", "Civil chat\n\nCivil\n\nCivil"]
        y_n3 = np.array([1, 0])

        bm_n1.fit(X_n1, y_n1)
        bm_n3.fit(X_n3, y_n3)

        self.assertNotEqual(bm_n1.checkpoint_n_obs, bm_n3.checkpoint_n_obs)
        self.assertIsNot(bm_n1.tfidf_model, bm_n3.tfidf_model)
        self.assertIsNot(bm_n1.prior_model, bm_n3.prior_model)
        self.assertAlmostEqual(bm_n1.prior_model.prior_prob, 1 / 3, places=4)
        self.assertAlmostEqual(bm_n3.prior_model.prior_prob, 1 / 2, places=4)

    def test_04_threshold_selection_using_validation_only(self):
        """Test that threshold is selected on validation data without touching test data."""
        bm = CheckpointBenchmark(checkpoint_n_obs=3, horizon_h=3)
        bm.fit(self.synthetic_texts[:6], self.synthetic_labels[:6])

        # Tune thresholds on validation
        X_val = self.synthetic_texts[6:8]
        y_val = self.synthetic_labels[6:8]
        res = bm.tune_thresholds(X_val, y_val)

        # Thresholds must be chosen and frozen
        self.assertIn("TFIDF_LogReg", res)
        self.assertIn("threshold", res["TFIDF_LogReg"])
        self.assertTrue(0.0 <= bm.tfidf_threshold <= 1.0)
        self.assertTrue(0.0 <= bm.prior_threshold <= 1.0)

    def test_05_evaluation_with_frozen_threshold(self):
        """Test that test evaluation uses the exact frozen threshold selected during validation."""
        bm = CheckpointBenchmark(checkpoint_n_obs=3, horizon_h=3)
        bm.fit(self.synthetic_texts[:6], self.synthetic_labels[:6])

        X_val = self.synthetic_texts[6:8]
        y_val = self.synthetic_labels[6:8]
        bm.tune_thresholds(X_val, y_val)

        frozen_thresh = bm.tfidf_threshold

        X_test = self.synthetic_texts[8:]
        y_test = self.synthetic_labels[8:]
        eval_res = bm.evaluate_split(X_test, y_test, "test")

        self.assertEqual(eval_res["TFIDF_LogReg"]["decision_threshold"], frozen_thresh)

    def test_06_successful_execution_on_small_synthetic_dataset(self):
        """Test end-to-end benchmark execution on a self-contained synthetic dataset."""
        bm = CheckpointBenchmark(checkpoint_n_obs=3, horizon_h=3)
        bm.fit(self.synthetic_texts[:6], self.synthetic_labels[:6])
        bm.tune_thresholds(self.synthetic_texts[6:8], self.synthetic_labels[6:8])
        results = bm.evaluate_split(self.synthetic_texts[8:], self.synthetic_labels[8:], "test")

        for m in ["PriorBaseline", "TFIDF_LogReg"]:
            self.assertIn("AP", results[m])
            self.assertIn("ROC_AUC", results[m])
            self.assertIn("Accuracy", results[m])
            self.assertIn("F1", results[m])
            self.assertIn("Precision", results[m])
            self.assertIn("Recall", results[m])
            self.assertIn("Brier_Score", results[m])
            self.assertIn("Confusion_Matrix", results[m])

    def test_07_handling_of_degenerate_validation_predictions(self):
        """Test that degenerate predictions (e.g., all 0 or 1, or empty) fall back gracefully without crashing."""
        # 1. All zero probabilities
        y_val = np.array([0, 1, 0, 1])
        zero_probs = np.zeros(4)
        thresh, f1, fallback_used = tune_threshold_f1(y_val, zero_probs, default_fallback=0.5)
        self.assertTrue(fallback_used)
        self.assertEqual(thresh, 0.5)
        self.assertEqual(f1, 0.0)

        # 2. Tied thresholds break ties by selecting HIGHER threshold
        y_val = np.array([0, 1])
        probs = np.array([0.2, 0.8])
        # For thresholds between 0.25 and 0.75, prediction is [0, 1], yielding F1=1.0
        candidate_grid = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        thresh_tie, f1_tie, fallback_tie = tune_threshold_f1(y_val, probs, candidate_thresholds=candidate_grid)
        self.assertFalse(fallback_tie)
        self.assertEqual(f1_tie, 1.0)
        # Must pick the highest tied candidate (0.7)
        self.assertEqual(thresh_tie, 0.7)


if __name__ == "__main__":
    unittest.main()
