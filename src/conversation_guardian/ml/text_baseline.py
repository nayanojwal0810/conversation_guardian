"""Word and character TF-IDF text baseline models and evaluation routines."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
)


def build_tfidf_pipeline(
    random_state: int = 42,
    c_value: float = 1.0,
    max_iter: int = 1000,
    min_df_word: int = 1,
    min_df_char: int = 1,
) -> Pipeline:
    """
    Construct a scikit-learn Pipeline with word and character n-gram TF-IDF representations
    followed by Logistic Regression.
    """
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=min_df_word,
        max_features=10000,
        sublinear_tf=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=min_df_char,
        max_features=15000,
        sublinear_tf=True,
    )
    union = FeatureUnion([
        ("word", word_vec),
        ("char", char_vec),
    ])
    clf = LogisticRegression(
        C=c_value,
        max_iter=max_iter,
        solver="liblinear",
        random_state=random_state,
    )
    return Pipeline([
        ("features", union),
        ("classifier", clf),
    ])


class PriorBaseline:
    """Simple baseline predicting constant probability equal to training positive class prevalence."""

    def __init__(self):
        self.prior_prob: float = 0.5

    def fit(self, X: Any, y: Any) -> "PriorBaseline":
        y_arr = np.asarray(y, dtype=int)
        self.prior_prob = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.5
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        n = len(X)
        probs = np.full((n, 2), 1.0 - self.prior_prob)
        probs[:, 1] = self.prior_prob
        return probs

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        n = len(X)
        pred_label = 1 if self.prior_prob >= threshold else 0
        return np.full(n, pred_label, dtype=int)


def tune_threshold_f1(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    candidate_thresholds: Optional[np.ndarray] = None,
    default_fallback: float = 0.5,
) -> Tuple[float, float, bool]:
    """
    Select decision threshold that maximizes F1 score on validation data.
    If several thresholds tie for maximum F1, choose the HIGHER threshold.
    Handles degenerate cases gracefully without crashing and records fallback status.

    Returns:
        (best_threshold, best_f1, fallback_used)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float(default_fallback), 0.0, True

    if candidate_thresholds is None:
        # Standard candidate grid from 0.01 to 0.99
        candidate_thresholds = np.linspace(0.01, 0.99, 99)

    best_thresh = float(default_fallback)
    best_f1 = -1.0
    fallback_used = False

    # Iterate in ascending order; >= ensures higher threshold is selected on tie
    for t in sorted(candidate_thresholds):
        y_pred = (y_probs >= t).astype(int)
        score = float(f1_score(y_true, y_pred, zero_division=0))
        if score >= best_f1:
            best_f1 = score
            best_thresh = float(t)

    # Degenerate case: no threshold produced non-zero F1
    if best_f1 <= 0.0:
        best_thresh = float(default_fallback)
        best_f1 = 0.0
        fallback_used = True

    return best_thresh, best_f1, fallback_used


def compute_benchmark_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float,
    model_name: str,
    split_name: str,
    checkpoint_n_obs: int,
    horizon_h: int = 3,
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics for a binary forecasting model."""
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    y_pred = (y_probs >= threshold).astype(int)

    n_total = len(y_true)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    prevalence = float(n_pos / n_total) if n_total > 0 else 0.0

    # AP and ROC-AUC
    if len(np.unique(y_true)) > 1:
        # Check if probabilities are non-constant
        if len(np.unique(y_probs)) > 1:
            ap_score = float(average_precision_score(y_true, y_probs))
            try:
                roc_auc = float(roc_auc_score(y_true, y_probs))
            except ValueError:
                roc_auc = 0.5
        else:
            # Constant probability prediction (e.g., PriorBaseline)
            ap_score = prevalence
            roc_auc = 0.5
    else:
        ap_score = prevalence
        roc_auc = 0.5

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_probs))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "model_name": model_name,
        "split": split_name,
        "checkpoint_n_obs": checkpoint_n_obs,
        "horizon_h": horizon_h,
        "sample_count": n_total,
        "positive_count": n_pos,
        "negative_count": n_neg,
        "prevalence": round(prevalence, 4),
        "decision_threshold": round(float(threshold), 4),
        "AP": round(ap_score, 4),
        "ROC_AUC": round(roc_auc, 4),
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "Brier_Score": round(brier, 4),
        "Confusion_Matrix": cm,
    }


class CheckpointBenchmark:
    """
    Manages training, validation threshold selection, and test evaluation
    for a single observation checkpoint without partition leakage.
    """

    def __init__(
        self,
        checkpoint_n_obs: int,
        horizon_h: int = 3,
        random_state: int = 42,
    ):
        self.checkpoint_n_obs = checkpoint_n_obs
        self.horizon_h = horizon_h
        self.random_state = random_state

        self.tfidf_model: Optional[Pipeline] = None
        self.prior_model: Optional[PriorBaseline] = None

        self.tfidf_threshold: float = 0.5
        self.prior_threshold: float = 0.5
        self.tfidf_fallback_used: bool = False
        self.prior_fallback_used: bool = False

    def fit(self, X_train: List[str], y_train: np.ndarray) -> "CheckpointBenchmark":
        """Fit models strictly on eligible training examples."""
        self.prior_model = PriorBaseline()
        self.prior_model.fit(X_train, y_train)

        self.tfidf_model = build_tfidf_pipeline(random_state=self.random_state)
        self.tfidf_model.fit(X_train, y_train)
        return self

    def tune_thresholds(
        self,
        X_val: List[str],
        y_val: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate validation predictions and freeze optimal F1 thresholds.
        Does NOT look at test data.
        """
        if self.tfidf_model is None or self.prior_model is None:
            raise RuntimeError("Models must be fitted before tuning thresholds.")

        # Prior baseline
        prior_val_probs = self.prior_model.predict_proba(X_val)[:, 1]
        self.prior_threshold, prior_f1, self.prior_fallback_used = tune_threshold_f1(
            y_val, prior_val_probs, default_fallback=0.5
        )

        # TF-IDF Logistic Regression
        tfidf_val_probs = self.tfidf_model.predict_proba(X_val)[:, 1]
        self.tfidf_threshold, tfidf_f1, self.tfidf_fallback_used = tune_threshold_f1(
            y_val, tfidf_val_probs, default_fallback=0.5
        )

        return {
            "PriorBaseline": {
                "threshold": self.prior_threshold,
                "val_f1": prior_f1,
                "fallback_used": self.prior_fallback_used,
            },
            "TFIDF_LogReg": {
                "threshold": self.tfidf_threshold,
                "val_f1": tfidf_f1,
                "fallback_used": self.tfidf_fallback_used,
            },
        }

    def evaluate_split(
        self,
        X: List[str],
        y: np.ndarray,
        split_name: str
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate models on a given partition using frozen thresholds."""
        if self.tfidf_model is None or self.prior_model is None:
            raise RuntimeError("Models must be fitted before evaluation.")

        # Prior Baseline
        prior_probs = self.prior_model.predict_proba(X)[:, 1]
        prior_metrics = compute_benchmark_metrics(
            y_true=y,
            y_probs=prior_probs,
            threshold=self.prior_threshold,
            model_name="PriorBaseline",
            split_name=split_name,
            checkpoint_n_obs=self.checkpoint_n_obs,
            horizon_h=self.horizon_h,
        )

        # TF-IDF Logistic Regression
        tfidf_probs = self.tfidf_model.predict_proba(X)[:, 1]
        tfidf_metrics = compute_benchmark_metrics(
            y_true=y,
            y_probs=tfidf_probs,
            threshold=self.tfidf_threshold,
            model_name="TFIDF_LogReg",
            split_name=split_name,
            checkpoint_n_obs=self.checkpoint_n_obs,
            horizon_h=self.horizon_h,
        )

        return {
            "PriorBaseline": prior_metrics,
            "TFIDF_LogReg": tfidf_metrics,
        }
