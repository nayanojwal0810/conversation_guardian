"""
Deployable prediction service for ConversationGuardian.

Provides the ConversationGuardianPredictor class for online inference at observation
checkpoints n in {1, 3, 5} to forecast personal-attack emergence within the next 3 dialogue turns.

Input: An ordered list of observed message texts (strictly 1, 3, or 5 strings).
Output: Machine-readable risk score, validation-selected alert threshold, alert flag,
        model metadata, and compact heuristic tone signal summaries.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from conversation_guardian.ml.sequence_features import (
    extract_message_signals,
    extract_aggregate_features,
    extract_ordered_features,
)


class ConversationGuardianPredictor:
    """
    Reusable prediction service component for ConversationGuardian.
    
    Loads frozen, validation-selected model bundles for checkpoints n in {1, 3, 5}
    and executes CPU-friendly inference strictly on observed message text.
    """

    SUPPORTED_CHECKPOINTS = (1, 3, 5)
    HORIZON_H = 3

    def __init__(
        self,
        bundles_dir: Optional[Union[str, Path]] = None,
        loaded_bundles: Optional[Dict[int, Dict[str, Any]]] = None,
    ):
        """
        Initialize predictor by loading bundles from bundles_dir or using pre-loaded bundles.
        
        Args:
            bundles_dir: Path to directory containing predictor_n{1,3,5}.joblib files.
                         Defaults to <repo_root>/models/selected_predictors.
            loaded_bundles: Optional dictionary mapping checkpoint (1, 3, 5) to bundle dicts,
                            primarily used for testing with stub models.
        """
        self._analyzer = SentimentIntensityAnalyzer()
        self._bundles: Dict[int, Dict[str, Any]] = {}

        if loaded_bundles is not None:
            self._bundles = dict(loaded_bundles)
            self._validate_loaded_bundles()
        else:
            if bundles_dir is None:
                # Default path relative to project structure
                bundles_dir = (
                    Path(__file__).resolve().parent.parent.parent.parent
                    / "models"
                    / "selected_predictors"
                )
            self.bundles_dir = Path(bundles_dir)
            self._load_bundles_from_disk()

    def _load_bundles_from_disk(self) -> None:
        """Load frozen model bundles from the specified bundles directory."""
        if not self.bundles_dir.exists():
            raise FileNotFoundError(
                f"Predictor bundles directory does not exist: '{self.bundles_dir}'. "
                f"Please run 'python scripts/run_sequence_benchmark.py' to fit and save the deployable bundles."
            )

        for n in self.SUPPORTED_CHECKPOINTS:
            bundle_path = self.bundles_dir / f"predictor_n{n}.joblib"
            if not bundle_path.exists():
                raise FileNotFoundError(
                    f"Selected model bundle for checkpoint n={n} not found at '{bundle_path}'. "
                    f"Please generate deployable bundles by running: 'python scripts/run_sequence_benchmark.py'"
                )
            try:
                bundle = joblib.load(bundle_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load bundle at '{bundle_path}': {str(e)}. "
                    f"Please regenerate bundles with 'python scripts/run_sequence_benchmark.py'."
                ) from e

            self._bundles[n] = bundle

        self._validate_loaded_bundles()

    def _validate_loaded_bundles(self) -> None:
        """Verify that required checkpoints and bundle keys are present."""
        missing = [n for n in self.SUPPORTED_CHECKPOINTS if n not in self._bundles]
        if missing:
            raise ValueError(
                f"Predictor is missing required bundles for checkpoints: {missing}."
            )

        required_keys = ("model_object", "model_family", "threshold", "feature_mode")
        for n, bundle in self._bundles.items():
            for key in required_keys:
                if key not in bundle:
                    raise KeyError(
                        f"Bundle for checkpoint n={n} is missing required field '{key}'."
                    )

    def validate_messages(self, messages: Any) -> List[str]:
        """
        Validate input message list against length and content constraints.
        
        Requirements:
          - Input must be a list or tuple.
          - Length must be exactly 1, 3, or 5.
          - Every message must be a non-empty, non-whitespace string.
        """
        if not isinstance(messages, (list, tuple)):
            raise TypeError(
                f"Expected messages to be a list or tuple of strings, got {type(messages).__name__}."
            )

        n = len(messages)
        if n not in self.SUPPORTED_CHECKPOINTS:
            raise ValueError(
                f"Invalid message count: received {n} messages. "
                f"ConversationGuardianPredictor strictly requires exactly 1, 3, or 5 observed messages."
            )

        validated: List[str] = []
        for i, m in enumerate(messages):
            if not isinstance(m, str):
                raise TypeError(
                    f"Message at index {i} must be a string, got {type(m).__name__}."
                )
            if not m.strip():
                raise ValueError(
                    f"Message at index {i} is empty or whitespace-only. "
                    f"All observed messages must contain authentic dialogue text."
                )
            validated.append(m)

        return validated

    def predict(self, messages: List[str]) -> Dict[str, Any]:
        """
        Execute attack risk forecast for an observed message sequence.
        
        Args:
            messages: Ordered list of observed dialogue turns (len must be 1, 3, or 5).
            
        Returns:
            Dict containing:
              - checkpoint_n_obs: observation checkpoint (1, 3, or 5)
              - horizon_h: forecasting horizon (3)
              - model_family: selected model identifier (e.g., 'Hybrid_LogReg')
              - model_version: bundle release version
              - risk_score: estimated continuous risk score in [0, 1]
              - threshold: frozen decision threshold selected on validation F1
              - alert: boolean flag indicating whether risk_score >= threshold
              - signals_summary: list of per-turn heuristic signals
              - signal_disclaimer: interpretive notice
        """
        clean_messages = self.validate_messages(messages)
        n = len(clean_messages)
        bundle = self._bundles[n]
        model = bundle["model_object"]
        threshold = float(bundle["threshold"])
        feature_mode = bundle["feature_mode"]

        # Extract per-turn heuristic signals
        msg_signals = [extract_message_signals(m, self._analyzer) for m in clean_messages]
        signals_summary = [
            {
                "turn_index": i + 1,
                "text": m,
                "vader_compound": round(float(sig["vader_compound"]), 4),
                "state": sig["state"],
                "hostility_match_count": int(sig["hostility_count"]),
                "is_hostility_match": bool(sig["has_hostility"]),
                "direct_address_count": int(sig["direct_address_count"]),
                "caps_ratio": round(float(sig["caps_ratio"]), 4),
            }
            for i, (m, sig) in enumerate(zip(clean_messages, msg_signals))
        ]

        # Extract model inputs matching the chosen architecture
        if feature_mode == "hybrid":
            prefix_text = "\n\n".join(clean_messages)
            ord_feat = extract_ordered_features(msg_signals, checkpoint_n_obs=n)
            ord_vec = np.array([[ord_feat[fname] for fname in bundle["feature_names"]]], dtype=float)
            prob = float(model.predict_proba([prefix_text], ord_vec)[0, 1])

        elif feature_mode == "aggregate":
            agg_feat = extract_aggregate_features(msg_signals)
            agg_vec = np.array([[agg_feat[fname] for fname in bundle["feature_names"]]], dtype=float)
            prob = float(model.predict_proba(agg_vec)[0, 1])

        elif feature_mode == "ordered":
            ord_feat = extract_ordered_features(msg_signals, checkpoint_n_obs=n)
            ord_vec = np.array([[ord_feat[fname] for fname in bundle["feature_names"]]], dtype=float)
            prob = float(model.predict_proba(ord_vec)[0, 1])

        elif feature_mode == "text_tfidf":
            prefix_text = "\n\n".join(clean_messages)
            prob = float(model.predict_proba([prefix_text])[0, 1])

        elif feature_mode == "recurrent":
            prob = float(model.predict_proba([clean_messages])[0, 1])

        else:
            raise ValueError(f"Unknown bundle feature mode: '{feature_mode}'")

        alert = bool(prob >= threshold)

        return {
            "checkpoint_n_obs": n,
            "horizon_h": self.HORIZON_H,
            "model_family": bundle["model_family"],
            "model_version": bundle.get("version", "1.0.0"),
            "risk_score": prob,
            "threshold": threshold,
            "alert": alert,
            "signals_summary": signals_summary,
            "signal_disclaimer": (
                "Heuristic conversational indicators derived strictly from observed message text, "
                "NOT ground-truth toxicity or verified attack annotations. Output risk score is an "
                "uncalibrated model score, not an empirical real-world probability."
            ),
        }
