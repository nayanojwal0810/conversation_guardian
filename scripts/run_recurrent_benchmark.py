"""
CLI script to execute Task 5: Recurrent Sequence Model Benchmark (GRU).

Trains and evaluates ConversationGRU across checkpoints n in {1, 3, 5} for horizon H = 3.
Compares validation AP against existing deployable models, freezes validation-selected F1 thresholds,
and evaluates test performance alongside all existing baselines.
"""

import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import joblib

# Ensure src directory is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from conversation_guardian.data.source_loader import load_cga_source_records
from conversation_guardian.ml.text_baseline import (
    compute_benchmark_metrics,
    tune_threshold_f1,
)
from conversation_guardian.ml.recurrent_model import RecurrentSequenceClassifier


def load_and_filter_events(data_dir: Path, split_name: str) -> pd.DataFrame:
    """Load partitioned event Parquet file and filter strictly to eligible examples."""
    parquet_path = data_dir / f"{split_name}_events.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Required event dataset not found at {parquet_path}. "
            f"Please run 'python scripts/prepare_cga_event_data.py' first."
        )

    df = pd.read_parquet(parquet_path)
    eligible_mask = (df["eligibility_status"] == "ELIGIBLE") & (df["target_label"].notna())
    df_eligible = df[eligible_mask].copy().reset_index(drop=True)
    df_eligible["target_label"] = df_eligible["target_label"].astype(int)
    return df_eligible


def recover_message_sequences(
    df_sub: pd.DataFrame,
    utterance_lookup: Dict[str, str],
    checkpoint_n_obs: int,
) -> Tuple[List[List[str]], List[List[str]]]:
    """Recover individual observed messages per example using observed_utterance_ids."""
    sequences_texts: List[List[str]] = []
    sequences_uids: List[List[str]] = []

    for idx, row in df_sub.iterrows():
        uids = list(row["observed_utterance_ids"])
        if len(uids) != checkpoint_n_obs:
            raise ValueError(
                f"Example {row['example_id']} has {len(uids)} observed utterance IDs, "
                f"expected exactly {checkpoint_n_obs}"
            )
        msgs = [utterance_lookup[uid] for uid in uids]
        sequences_texts.append(msgs)
        sequences_uids.append(uids)

    return sequences_texts, sequences_uids


def main():
    parser = argparse.ArgumentParser(description="Run Recurrent Sequence Model Benchmark (GRU)")
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
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=15,
        help="Maximum training epochs for GRU (default: 15)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience in epochs (default: 3)",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    reports_dir = Path(args.reports_dir)

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CONVERSATIONGUARDIAN — TASK 5: RECURRENT SEQUENCE MODEL BENCHMARK (GRU)")
    print("=" * 80)

    # 1. Load source corpus and build utterance lookup
    print("\n[Step 1/5] Loading official CGA corpus and creating utterance lookup...")
    source_convs = load_cga_source_records()
    utterance_lookup: Dict[str, str] = {
        u.id: u.text for c in source_convs.values() for u in c.utterances
    }
    print(f"  -> Indexed {len(utterance_lookup)} total utterances from source records.")

    # 2. Load partition event datasets
    print("\n[Step 2/5] Loading and filtering eligible partition datasets...")
    train_df = load_and_filter_events(data_dir, "train")
    val_df = load_and_filter_events(data_dir, "val")
    test_df = load_and_filter_events(data_dir, "test")

    print(f"  -> Train eligible: {len(train_df)} examples")
    print(f"  -> Val eligible:   {len(val_df)} examples")
    print(f"  -> Test eligible:  {len(test_df)} examples")

    all_metrics: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    model_configs: Dict[str, Any] = {}

    # Load existing benchmark summary for direct baseline comparisons
    seq_summary_path = reports_dir / "sequence_summary.csv"
    existing_baselines_df = None
    if seq_summary_path.exists():
        existing_baselines_df = pd.read_csv(seq_summary_path)

    # 3. Benchmark each checkpoint
    print(f"\n[Step 3/5] Benchmarking GRU across checkpoints: {args.checkpoints}...")
    for n_obs in sorted(args.checkpoints):
        print(f"\n" + "-" * 80)
        print(f"CHECKPOINT n = {n_obs} OBSERVED MESSAGES (HORIZON H = {args.horizon})")
        print("-" * 80)

        tr_sub = train_df[train_df["checkpoint_n_obs"] == n_obs].reset_index(drop=True)
        val_sub = val_df[val_df["checkpoint_n_obs"] == n_obs].reset_index(drop=True)
        te_sub = test_df[test_df["checkpoint_n_obs"] == n_obs].reset_index(drop=True)

        tr_msgs, _ = recover_message_sequences(tr_sub, utterance_lookup, n_obs)
        val_msgs, _ = recover_message_sequences(val_sub, utterance_lookup, n_obs)
        te_msgs, _ = recover_message_sequences(te_sub, utterance_lookup, n_obs)

        y_train = tr_sub["target_label"].to_numpy()
        y_val = val_sub["target_label"].to_numpy()
        y_test = te_sub["target_label"].to_numpy()

        # Checkpoint-specific max token length: 128 for n=1, 256 for n=3, 384 for n=5
        max_seq_len = 128 if n_obs == 1 else (256 if n_obs == 3 else 384)

        recurrent_clf = RecurrentSequenceClassifier(
            max_vocab_size=5000,
            embedding_dim=64,
            hidden_dim=64,
            num_layers=1,
            dropout=0.2,
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=64,
            max_epochs=args.max_epochs,
            patience=args.patience,
            max_length=max_seq_len,
            random_state=args.random_state,
        )

        print(f"  -> Fitting ConversationGRU on training set (vocab built on train only)...")
        t0 = time.time()
        recurrent_clf.fit(tr_msgs, y_train, val_msgs, y_val)
        fit_duration = time.time() - t0
        print(f"     Trained {recurrent_clf.epochs_trained} epochs in {fit_duration:.2f}s "
              f"(Best epoch: {recurrent_clf.best_epoch}, Best Val AP: {recurrent_clf.best_val_ap:.4f}, "
              f"Params: {recurrent_clf.trainable_params:,})")

        model_configs[f"n{n_obs}"] = recurrent_clf.get_config()

        # Validation prediction & threshold tuning
        print("  -> Tuning decision threshold on validation set (maximizing F1, tie-breaking to higher threshold)...")
        val_probs = recurrent_clf.predict_proba(val_msgs)[:, 1]
        best_thresh, best_val_f1, fallback = tune_threshold_f1(y_val, val_probs, default_fallback=0.5)
        print(f"     Validation Optimal Threshold: {best_thresh:.4f} (Val F1: {best_val_f1:.4f})")

        val_metrics = compute_benchmark_metrics(
            y_true=y_val,
            y_probs=val_probs,
            threshold=best_thresh,
            model_name="Recurrent_GRU",
            split_name="val",
            checkpoint_n_obs=n_obs,
            horizon_h=args.horizon,
        )
        all_metrics.append(val_metrics)

        # Check against existing deployable predictor
        predictors_dir = models_dir / "selected_predictors"
        bundle_path = predictors_dir / f"predictor_n{n_obs}.joblib"
        incumbent_val_ap = 0.0
        incumbent_family = "None"
        if bundle_path.exists():
            incumbent_bundle = joblib.load(bundle_path)
            incumbent_val_ap = incumbent_bundle.get("val_ap", 0.0)
            incumbent_family = incumbent_bundle.get("model_family", "Unknown")

        gru_val_ap = val_metrics["AP"]
        print(f"  -> [Model Selection Gate] GRU Val AP: {gru_val_ap:.4f} vs Incumbent '{incumbent_family}' Val AP: {incumbent_val_ap:.4f}")
        if gru_val_ap > incumbent_val_ap:
            print(f"     *** GRU WINS on validation AP! Updating deployable bundle for n={n_obs}. ***")
            new_bundle = {
                "checkpoint_n_obs": n_obs,
                "horizon_h": args.horizon,
                "model_family": "Recurrent_GRU",
                "model_object": recurrent_clf,
                "threshold": float(best_thresh),
                "val_ap": float(gru_val_ap),
                "val_f1": float(best_val_f1),
                "val_roc_auc": float(val_metrics["ROC_AUC"]),
                "feature_mode": "recurrent",
                "feature_names": [],
                "version": "1.1.0",
                "metadata": {
                    "dataset_version": "cga_v1",
                    "selection_criterion": "validation_average_precision",
                    "decision_rule": "f1_optimal_on_validation",
                },
            }
            joblib.dump(new_bundle, bundle_path)
        else:
            print(f"     *** Incumbent '{incumbent_family}' retains selection for n={n_obs} (Val AP {incumbent_val_ap:.4f} >= {gru_val_ap:.4f}). ***")

        # Evaluate test partition using frozen threshold
        print("  -> Evaluating test partition with frozen validation threshold...")
        test_probs = recurrent_clf.predict_proba(te_msgs)[:, 1]
        test_metrics = compute_benchmark_metrics(
            y_true=y_test,
            y_probs=test_probs,
            threshold=best_thresh,
            model_name="Recurrent_GRU",
            split_name="test",
            checkpoint_n_obs=n_obs,
            horizon_h=args.horizon,
        )
        all_metrics.append(test_metrics)

        summary_rows.append({
            "checkpoint": n_obs,
            "model": "Recurrent_GRU",
            "split": "test",
            "sample_count": test_metrics["sample_count"],
            "positive_count": test_metrics["positive_count"],
            "prevalence": test_metrics["prevalence"],
            "threshold": test_metrics["decision_threshold"],
            "AP": test_metrics["AP"],
            "ROC_AUC": test_metrics["ROC_AUC"],
            "Accuracy": test_metrics["Accuracy"],
            "Precision": test_metrics["Precision"],
            "Recall": test_metrics["Recall"],
            "F1": test_metrics["F1"],
            "Brier_Score": test_metrics["Brier_Score"],
            "Confusion_Matrix": json.dumps(test_metrics["Confusion_Matrix"]),
            "trainable_params": recurrent_clf.trainable_params,
            "training_time_sec": round(fit_duration, 2),
        })

        # Save test predictions (full float64 precision)
        pred_df = pd.DataFrame({
            "example_id": te_sub["example_id"],
            "conversation_id": te_sub["conversation_id"],
            "pair_id": te_sub["pair_id"],
            "split": te_sub["split"],
            "checkpoint_n_obs": n_obs,
            "horizon_h": args.horizon,
            "true_label": y_test,
            "gru_prob": test_probs,  # Full float64 precision
            "gru_pred": (test_probs >= best_thresh).astype(int),
            "gru_threshold": best_thresh,
        })
        pred_csv = reports_dir / f"recurrent_test_predictions_n{n_obs}.csv"
        pred_df.to_csv(pred_csv, index=False)
        print(f"  -> Saved full-precision GRU predictions ({len(pred_df)} rows) to {pred_csv}")

    # 4. Save metrics and configuration metadata
    print("\n[Step 4/5] Saving benchmark metrics and configuration metadata...")
    metrics_json_path = reports_dir / "recurrent_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  -> Saved full metrics to {metrics_json_path}")

    config_json_path = reports_dir / "recurrent_model_config.json"
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(model_configs, f, indent=2)
    print(f"  -> Saved model configurations to {config_json_path}")

    recurrent_summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = reports_dir / "recurrent_summary.csv"
    recurrent_summary_df.to_csv(summary_csv_path, index=False)
    print(f"  -> Saved summary table to {summary_csv_path}")

    # 5. Display comprehensive comparison table
    print("\n" + "=" * 90)
    print("TASK 5: RECURRENT SEQUENCE MODEL (GRU) VS EXISTING BASELINES (TEST SET)")
    print("=" * 90)

    # Combine with existing baselines for side-by-side terminal report
    if existing_baselines_df is not None:
        combined_rows = []
        for n_obs in sorted(args.checkpoints):
            base_sub = existing_baselines_df[existing_baselines_df["checkpoint"] == n_obs]
            gru_sub = recurrent_summary_df[recurrent_summary_df["checkpoint"] == n_obs]
            combined = pd.concat([base_sub, gru_sub], ignore_index=True)
            combined_rows.append(combined)
        full_display_df = pd.concat(combined_rows, ignore_index=True)
        cols_to_show = ["checkpoint", "model", "prevalence", "threshold", "AP", "ROC_AUC", "Accuracy", "Precision", "Recall", "F1", "Brier_Score"]
        print(full_display_df[[c for c in cols_to_show if c in full_display_df.columns]].to_string(index=False))
    else:
        print(recurrent_summary_df.to_string(index=False))

    print("=" * 90)
    print("RECURRENT SEQUENCE BENCHMARK COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
