"""CLI script to train and evaluate checkpoint-specific text baseline benchmarks."""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import joblib

# Ensure src directory is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from conversation_guardian.ml.text_baseline import (
    CheckpointBenchmark,
)


def load_and_filter_events(
    data_dir: Path,
    split_name: str
) -> pd.DataFrame:
    """Load partitioned event Parquet file and filter strictly to eligible examples."""
    parquet_path = data_dir / f"{split_name}_events.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Required event dataset not found at {parquet_path}. "
            f"Please run 'python scripts/prepare_cga_event_data.py' first."
        )

    df = pd.read_parquet(parquet_path)
    # Strictly filter to eligible examples with valid binary target label
    eligible_mask = (df["eligibility_status"] == "ELIGIBLE") & (df["target_label"].notna())
    df_eligible = df[eligible_mask].copy().reset_index(drop=True)
    df_eligible["target_label"] = df_eligible["target_label"].astype(int)
    # Fill any null text with empty string
    df_eligible["observed_prefix_text"] = df_eligible["observed_prefix_text"].fillna("").astype(str)
    return df_eligible


def main():
    parser = argparse.ArgumentParser(description="Run Checkpoint-Specific Text Baseline Benchmark")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(ROOT_DIR / "data" / "events"),
        help="Path to directory containing partitioned event parquet files",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(ROOT_DIR / "models"),
        help="Directory to save trained model artifacts",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str(ROOT_DIR / "reports" / "benchmarks"),
        help="Directory to save benchmark metrics and prediction records",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=int,
        default=[1, 3, 5],
        help="Observation checkpoints n_obs to benchmark (default: 1 3 5)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="Forecasting horizon H (default: 3)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    reports_dir = Path(args.reports_dir)

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CONVERSATIONGUARDIAN — TASK 2: TEXT BASELINE BENCHMARK (H = 3)")
    print("=" * 80)

    # 1. Load data
    print("\n[Step 1/4] Loading and filtering eligible partition datasets...")
    train_df = load_and_filter_events(data_dir, "train")
    val_df = load_and_filter_events(data_dir, "val")
    test_df = load_and_filter_events(data_dir, "test")

    print(f"  -> Train eligible: {len(train_df)} examples")
    print(f"  -> Val eligible:   {len(val_df)} examples")
    print(f"  -> Test eligible:  {len(test_df)} examples")

    all_metrics: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    # 2. Benchmark each checkpoint independently
    print(f"\n[Step 2/4] Fitting and evaluating models across checkpoints: {args.checkpoints}...")
    for n_obs in sorted(args.checkpoints):
        print(f"\n" + "-" * 80)
        print(f"OBSERVATION CHECKPOINT: n = {n_obs} observed messages (Horizon H = {args.horizon})")
        print("-" * 80)

        tr_sub = train_df[train_df["checkpoint_n_obs"] == n_obs].reset_index(drop=True)
        val_sub = val_df[val_df["checkpoint_n_obs"] == n_obs].reset_index(drop=True)
        te_sub = test_df[test_df["checkpoint_n_obs"] == n_obs].reset_index(drop=True)

        # Direct model inputs: ONLY observed_prefix_text
        X_train = tr_sub["observed_prefix_text"].tolist()
        y_train = tr_sub["target_label"].to_numpy()

        X_val = val_sub["observed_prefix_text"].tolist()
        y_val = val_sub["target_label"].to_numpy()

        X_test = te_sub["observed_prefix_text"].tolist()
        y_test = te_sub["target_label"].to_numpy()

        print(f"  Train: Total={len(y_train)}, Pos={int(y_train.sum())}, Neg={int(len(y_train) - y_train.sum())}, Prev={y_train.mean():.4f}")
        print(f"  Val:   Total={len(y_val)}, Pos={int(y_val.sum())}, Neg={int(len(y_val) - y_val.sum())}, Prev={y_val.mean():.4f}")
        print(f"  Test:  Total={len(y_test)}, Pos={int(y_test.sum())}, Neg={int(len(y_test) - y_test.sum())}, Prev={y_test.mean():.4f}")

        # Initialize benchmark
        benchmark = CheckpointBenchmark(
            checkpoint_n_obs=n_obs,
            horizon_h=args.horizon,
            random_state=args.random_state,
        )

        # Fit on train only
        print("  -> Fitting Prior Baseline and Word+Char TF-IDF Logistic Regression on train...")
        benchmark.fit(X_train, y_train)

        # Tune thresholds on validation
        print("  -> Selecting optimal F1 decision threshold on validation set...")
        tune_res = benchmark.tune_thresholds(X_val, y_val)
        prior_thresh = tune_res["PriorBaseline"]["threshold"]
        tfidf_thresh = tune_res["TFIDF_LogReg"]["threshold"]
        print(f"     Prior Baseline optimal threshold:   {prior_thresh:.4f} (Val F1: {tune_res['PriorBaseline']['val_f1']:.4f})")
        print(f"     TF-IDF LogReg optimal threshold:    {tfidf_thresh:.4f} (Val F1: {tune_res['TFIDF_LogReg']['val_f1']:.4f})")

        # Evaluate on validation
        val_eval = benchmark.evaluate_split(X_val, y_val, "val")
        all_metrics.extend([val_eval["PriorBaseline"], val_eval["TFIDF_LogReg"]])

        # Evaluate on test with frozen thresholds
        print("  -> Evaluating test split with frozen thresholds...")
        test_eval = benchmark.evaluate_split(X_test, y_test, "test")
        all_metrics.extend([test_eval["PriorBaseline"], test_eval["TFIDF_LogReg"]])

        for m_name in ["PriorBaseline", "TFIDF_LogReg"]:
            m_res = test_eval[m_name]
            summary_rows.append({
                "checkpoint": n_obs,
                "model": m_name,
                "split": "test",
                "sample_count": m_res["sample_count"],
                "positive_count": m_res["positive_count"],
                "prevalence": m_res["prevalence"],
                "threshold": m_res["decision_threshold"],
                "AP": m_res["AP"],
                "ROC_AUC": m_res["ROC_AUC"],
                "Accuracy": m_res["Accuracy"],
                "Precision": m_res["Precision"],
                "Recall": m_res["Recall"],
                "F1": m_res["F1"],
                "Brier_Score": m_res["Brier_Score"],
                "Confusion_Matrix": json.dumps(m_res["Confusion_Matrix"]),
            })

        # Save trained pipeline under models/
        model_save_path = models_dir / f"baseline_tfidf_logreg_n{n_obs}.joblib"
        joblib.dump(benchmark.tfidf_model, model_save_path)
        print(f"  -> Saved trained TF-IDF model to {model_save_path}")

        # Save test prediction records
        prior_test_probs = benchmark.prior_model.predict_proba(X_test)[:, 1]
        prior_test_preds = (prior_test_probs >= prior_thresh).astype(int)

        tfidf_test_probs = benchmark.tfidf_model.predict_proba(X_test)[:, 1]
        tfidf_test_preds = (tfidf_test_probs >= tfidf_thresh).astype(int)

        pred_df = pd.DataFrame({
            "example_id": te_sub["example_id"],
            "conversation_id": te_sub["conversation_id"],
            "pair_id": te_sub["pair_id"],
            "split": te_sub["split"],
            "checkpoint_n_obs": n_obs,
            "horizon_h": args.horizon,
            "true_label": y_test,
            "prior_prob": np.round(prior_test_probs, 4),
            "prior_pred": prior_test_preds,
            "prior_threshold": round(prior_thresh, 4),
            "tfidf_prob": np.round(tfidf_test_probs, 4),
            "tfidf_pred": tfidf_test_preds,
            "tfidf_threshold": round(tfidf_thresh, 4),
        })
        pred_csv_path = reports_dir / f"test_predictions_n{n_obs}.csv"
        pred_df.to_csv(pred_csv_path, index=False)
        print(f"  -> Saved test predictions ({len(pred_df)} rows) to {pred_csv_path}")

    # 3. Save benchmark reports
    print("\n[Step 3/4] Writing machine-readable benchmark reports...")
    metrics_json_path = reports_dir / "baseline_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  -> Saved full metrics to {metrics_json_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = reports_dir / "baseline_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"  -> Saved summary table to {summary_csv_path}")

    # 4. Terminal summary
    print("\n" + "=" * 80)
    print("BENCHMARK TEST EVALUATION SUMMARY (FROZEN VALIDATION THRESHOLDS)")
    print("=" * 80)
    display_df = summary_df[[
        "checkpoint", "model", "prevalence", "threshold", "AP", "ROC_AUC",
        "Accuracy", "Precision", "Recall", "F1", "Brier_Score"
    ]]
    print(display_df.to_string(index=False))
    print("=" * 80)
    print("BENCHMARK EXECUTION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
