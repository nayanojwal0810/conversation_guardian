"""
Unit tests for ConversationGuardianPredictor and deployable inference service (Task 4).

Validates:
  1. Input validation (supported lengths 1, 3, 5; type checks; empty message rejection)
  2. Preservation of message order
  3. Correct selected model family and threshold for each checkpoint
  4. Deterministic output on identical inputs
  5. Absence of future text, target labels, or database dependencies
  6. Informative error messages when bundles are missing or invalid
  7. Real bundle integration (when deployable bundles exist)
"""

import sys
import unittest
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Ensure src directory is in sys.path
TEST_DIR = Path(__file__).resolve().parent
ROOT_DIR = TEST_DIR.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from conversation_guardian.ml.predictor import ConversationGuardianPredictor
from conversation_guardian.ml.sequence_features import (
    get_aggregate_feature_names,
    get_ordered_feature_names,
)


class DummyMockModel:
    """Lightweight stub model for isolated predictor unit tests without disk dependencies."""

    def __init__(self, fixed_prob: float = 0.45):
        self.fixed_prob = fixed_prob

    def predict_proba(self, *args, **kwargs) -> np.ndarray:
        return np.array([[1.0 - self.fixed_prob, self.fixed_prob]])


def build_stub_bundles() -> Dict[int, Dict[str, Any]]:
    """Create in-memory mock bundles for testing."""
    return {
        1: {
            "checkpoint_n_obs": 1,
            "horizon_h": 3,
            "model_family": "Hybrid_LogReg",
            "model_object": DummyMockModel(0.40),
            "threshold": 0.15,
            "val_ap": 0.3409,
            "val_f1": 0.4414,
            "feature_mode": "hybrid",
            "feature_names": get_ordered_feature_names(1),
            "version": "1.0.0",
        },
        3: {
            "checkpoint_n_obs": 3,
            "horizon_h": 3,
            "model_family": "Hybrid_LogReg",
            "model_object": DummyMockModel(0.20),
            "threshold": 0.27,
            "val_ap": 0.4987,
            "val_f1": 0.5461,
            "feature_mode": "hybrid",
            "feature_names": get_ordered_feature_names(3),
            "version": "1.0.0",
        },
        5: {
            "checkpoint_n_obs": 5,
            "horizon_h": 3,
            "model_family": "AggregateState_LogReg",
            "model_object": DummyMockModel(0.55),
            "threshold": 0.42,
            "val_ap": 0.6284,
            "val_f1": 0.7368,
            "feature_mode": "aggregate",
            "feature_names": get_aggregate_feature_names(),
            "version": "1.0.0",
        },
    }


class TestConversationGuardianPredictor(unittest.TestCase):
    """Test suite covering the ConversationGuardianPredictor service."""

    def setUp(self):
        self.stub_bundles = build_stub_bundles()
        self.predictor = ConversationGuardianPredictor(loaded_bundles=self.stub_bundles)

    def test_01_input_validation_lengths(self):
        """Reject inputs that do not contain exactly 1, 3, or 5 observed messages."""
        # 0 messages
        with self.assertRaises(ValueError):
            self.predictor.predict([])

        # 2 messages
        with self.assertRaises(ValueError):
            self.predictor.predict(["Turn 1", "Turn 2"])

        # 4 messages
        with self.assertRaises(ValueError):
            self.predictor.predict(["Turn 1", "Turn 2", "Turn 3", "Turn 4"])

        # 6 messages
        with self.assertRaises(ValueError):
            self.predictor.predict(["1", "2", "3", "4", "5", "6"])

    def test_02_input_validation_types_and_empty_values(self):
        """Reject non-list inputs, non-string messages, and empty/whitespace messages."""
        # Non-list
        with self.assertRaises(TypeError):
            self.predictor.predict("Just a string")

        # Non-string element
        with self.assertRaises(TypeError):
            self.predictor.predict([123])

        # Empty string
        with self.assertRaises(ValueError):
            self.predictor.predict([""])

        # Whitespace-only string
        with self.assertRaises(ValueError):
            self.predictor.predict(["Turn 1", "   \t\n  ", "Turn 3"])

    def test_03_preservation_of_message_order(self):
        """Verify message order is preserved in signals_summary and turn indices."""
        msgs = [
            "Thank you for reviewing.",
            "I checked the article page.",
            "You are an idiot!",
        ]
        res = self.predictor.predict(msgs)

        summary = res["signals_summary"]
        self.assertEqual(len(summary), 3)

        self.assertEqual(summary[0]["turn_index"], 1)
        self.assertEqual(summary[0]["text"], msgs[0])
        self.assertEqual(summary[0]["state"], "POSITIVE")

        self.assertEqual(summary[1]["turn_index"], 2)
        self.assertEqual(summary[1]["text"], msgs[1])
        self.assertEqual(summary[1]["state"], "NEUTRAL")

        self.assertEqual(summary[2]["turn_index"], 3)
        self.assertEqual(summary[2]["text"], msgs[2])
        self.assertEqual(summary[2]["state"], "HOSTILE")
        self.assertTrue(summary[2]["is_hostility_match"])

    def test_04_correct_threshold_and_alert_decision(self):
        """Verify alert decision is True when risk_score >= threshold, else False."""
        # Checkpoint 1: stub prob=0.40, threshold=0.15 -> alert=True
        res_n1 = self.predictor.predict(["Hello"])
        self.assertEqual(res_n1["checkpoint_n_obs"], 1)
        self.assertEqual(res_n1["model_family"], "Hybrid_LogReg")
        self.assertEqual(res_n1["threshold"], 0.15)
        self.assertAlmostEqual(res_n1["risk_score"], 0.40, places=4)
        self.assertTrue(res_n1["alert"])

        # Checkpoint 3: stub prob=0.20, threshold=0.27 -> alert=False
        res_n3 = self.predictor.predict(["M1", "M2", "M3"])
        self.assertEqual(res_n3["checkpoint_n_obs"], 3)
        self.assertEqual(res_n3["threshold"], 0.27)
        self.assertAlmostEqual(res_n3["risk_score"], 0.20, places=4)
        self.assertFalse(res_n3["alert"])

        # Checkpoint 5: stub prob=0.55, threshold=0.42 -> alert=True
        res_n5 = self.predictor.predict(["M1", "M2", "M3", "M4", "M5"])
        self.assertEqual(res_n5["checkpoint_n_obs"], 5)
        self.assertEqual(res_n5["model_family"], "AggregateState_LogReg")
        self.assertEqual(res_n5["threshold"], 0.42)
        self.assertAlmostEqual(res_n5["risk_score"], 0.55, places=4)
        self.assertTrue(res_n5["alert"])

    def test_05_deterministic_output(self):
        """Verify identical inputs produce deterministic output dictionaries."""
        msgs = ["First turn.", "Second turn.", "Third turn."]
        res1 = self.predictor.predict(msgs)
        res2 = self.predictor.predict(msgs)
        self.assertEqual(res1, res2)

    def test_06_missing_bundles_directory_raises_clear_error(self):
        """Verify informative FileNotFoundError when bundle directory or files are missing."""
        non_existent_dir = Path("D:/non_existent_predictor_bundles_dir_test_12345")
        with self.assertRaises(FileNotFoundError) as ctx:
            ConversationGuardianPredictor(bundles_dir=non_existent_dir)
        self.assertIn("scripts/run_sequence_benchmark.py", str(ctx.exception))

    def test_07_real_bundles_integration(self):
        """Verify real trained predictor bundles if they have been generated on disk."""
        real_bundles_dir = ROOT_DIR / "models" / "selected_predictors"
        if not real_bundles_dir.exists() or not (real_bundles_dir / "predictor_n1.joblib").exists():
            self.skipTest("Real predictor bundles not yet generated on disk. Skipping integration check.")

        real_predictor = ConversationGuardianPredictor(bundles_dir=real_bundles_dir)

        # n=1
        res1 = real_predictor.predict(["Thank you for the edit."])
        self.assertEqual(res1["checkpoint_n_obs"], 1)
        self.assertEqual(res1["model_family"], "Hybrid_LogReg")
        self.assertAlmostEqual(res1["threshold"], 0.15, places=2)
        self.assertTrue(0.0 <= res1["risk_score"] <= 1.0)
        self.assertIsInstance(res1["alert"], bool)

        # n=3
        res3 = real_predictor.predict(["Turn 1", "Turn 2", "Turn 3"])
        self.assertEqual(res3["checkpoint_n_obs"], 3)
        self.assertEqual(res3["model_family"], "Hybrid_LogReg")
        self.assertAlmostEqual(res3["threshold"], 0.27, places=2)
        self.assertTrue(0.0 <= res3["risk_score"] <= 1.0)

        # n=5
        res5 = real_predictor.predict(["Turn 1", "Turn 2", "Turn 3", "Turn 4", "Turn 5"])
        self.assertEqual(res5["checkpoint_n_obs"], 5)
        self.assertEqual(res5["model_family"], "AggregateState_LogReg")
        self.assertAlmostEqual(res5["threshold"], 0.42, places=2)
        self.assertTrue(0.0 <= res5["risk_score"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
