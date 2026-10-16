"""
CLI script to execute Task 3: Sequential State Modeling Benchmark.

Compares six models across observation checkpoints n in {1, 3, 5} for horizon H = 3:
  1. MajorityBaseline (Training majority class, fixed 0.5 decision rule)
  2. PriorBaseline (Constant soft probability equal to training prevalence)
  3. TFIDF_TextBaseline (Word and character n-gram TF-IDF Logistic Regression)
  4. AggregateState_LogReg (Order-agnostic state/sentiment summaries)
  5. OrderedState_LogReg (Aggregate + transitions + trigrams + positions + trajectories)
  6. Hybrid_LogReg (Sparse concatenation of TF-IDF and standardized ordered features)
"""

import os
import sys
import json
import argparse
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
from conversation_guardian.ml.sequence_features import (
    extract_message_signals,
    SequenceFeatureExtractor,
    SequenceCheckpointBenchmark,
)


def load_and_filter_events(
    data_dir: Path,
    split_name: str,
) -> pd.DataFrame:
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
) -> Tuple[List[List[str]], List[List[str]], List[str]]:
    """
    Recover individual observed messages per example using observed_utterance_ids.
    
    Returns:
        (sequences_of_texts, sequences_of_uids, reconstructed_prefix_texts)
    """
    sequences_texts: List[List[str]] = []
    sequences_uids: List[List[str]] = []
    prefix_texts: List[str] = []

    for idx, row in df_sub.iterrows():
        uids = list(row["observed_utterance_ids"])
        if len(uids) != checkpoint_n_obs:
            raise ValueError(
                f"Example {row['example_id']} has {len(uids)} observed utterance IDs, "
                f"expected exactly {checkpoint_n_obs}"
            )
        msgs: List[str] = []
        for uid in uids:
            if uid not in utterance_lookup:
                raise KeyError(
                    f"Utterance ID '{uid}' from example '{row['example_id']}' not found in source corpus lookup."
                )
            msgs.append(utterance_lookup[uid])

        sequences_texts.append(msgs)
        sequences_uids.append(uids)
        prefix_texts.append("\n\n".join(msgs))

    return sequences_texts, sequences_uids, prefix_texts


def main():
    parser = argparse.ArgumentParser(description="Run Sequential State Modeling Benchmark (H = 3)")
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
    print("CONVERSATIONGUARDIAN — TASK 3: SEQUENTIAL STATE MODELING BENCHMARK (H = 3)")
    print("=" * 80)

    # 1. Load source corpus and build utterance lookup
    print("\n[Step 1/5] Loading official CGA corpus and creating utterance lookup...")
    source_convs = load_cga_source_records()
    utterance_lookup: Dict[str, str] = {
        u.id: u.text for c in source_convs.values() for u in c.utterances
    }
    print(f"  -> Successfully indexed {len(utterance_lookup)} total utterances from source records.")

    # Pre-extract utterance signal cache
    print("  -> Pre-computing heuristic message signals for unique utterances...")
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    utterance_cache: Dict[str, Dict[str, Any]] = {}
    for uid, text in utterance_lookup.items():
        utterance_cache[uid] = extract_message_signals(text, analyzer)
    print(f"  -> Cached message signals for {len(utterance_cache)} utterances.")

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

    # 3. Benchmark each checkpoint independently
    print(f"\n[Step 3/5] Benchmarking observation checkpoints: {args.checkpoints}...")
    for n_obs in sorted(args.checkpoints):
        print(f"\n" + "-" * 80)
        print(f"OBSERVATION CHECKPOINT: n = {n_obs} observed messages (Horizon H = {args.horizon})")
        print("-" * 80)

        tr_sub = train_df[train_df["checkpoint_n_obs"] == n_obs].reset_index(drop=True)
        val_sub = val_df[val_df["checkpoint_n_obs"] == n_obs].reset_index(drop=True)
        te_sub = test_df[test_df["checkpoint_n_obs"] == n_obs].reset_index(drop=True)

        # Recover observed message sequences strictly by utterance ID lookup
        tr_msgs, tr_uids, tr_texts = recover_message_sequences(tr_sub, utterance_lookup, n_obs)
        val_msgs, val_uids, val_texts = recover_message_sequences(val_sub, utterance_lookup, n_obs)
        te_msgs, te_uids, te_texts = recover_message_sequences(te_sub, utterance_lookup, n_obs)

        y_train = tr_sub["target_label"].to_numpy()
        y_val = val_sub["target_label"].to_numpy()
        y_test = te_sub["target_label"].to_numpy()

        print(f"  Train: Total={len(y_train)}, Pos={int(y_train.sum())}, Neg={int(len(y_train) - y_train.sum())}, Prev={y_train.mean():.4f}")
        print(f"  Val:   Total={len(y_val)}, Pos={int(y_val.sum())}, Neg={int(len(y_val) - y_val.sum())}, Prev={y_val.mean():.4f}")
        print(f"  Test:  Total={len(y_test)}, Pos={int(y_test.sum())}, Neg={int(len(y_test) - y_test.sum())}, Prev={y_test.mean():.4f}")

        # Extract Aggregate and Ordered features
        print(f"  -> Extracting Model A (Aggregate) and Model B (Ordered) feature matrices...")
        agg_extractor = SequenceFeatureExtractor(mode="aggregate", checkpoint_n_obs=n_obs)
        ord_extractor = SequenceFeatureExtractor(mode="ordered", checkpoint_n_obs=n_obs)

        tr_agg = agg_extractor.extract_feature_matrix(tr_msgs, utterance_cache=utterance_cache, sequence_utterance_ids=tr_uids)
        val_agg = agg_extractor.extract_feature_matrix(val_msgs, utterance_cache=utterance_cache, sequence_utterance_ids=val_uids)
        te_agg = agg_extractor.extract_feature_matrix(te_msgs, utterance_cache=utterance_cache, sequence_utterance_ids=te_uids)

        tr_ord = ord_extractor.extract_feature_matrix(tr_msgs, utterance_cache=utterance_cache, sequence_utterance_ids=tr_uids)
        val_ord = ord_extractor.extract_feature_matrix(val_msgs, utterance_cache=utterance_cache, sequence_utterance_ids=val_uids)
        te_ord = ord_extractor.extract_feature_matrix(te_msgs, utterance_cache=utterance_cache, sequence_utterance_ids=te_uids)

        print(f"     Aggregate feature matrix shape: {tr_agg.shape}")
        print(f"     Ordered feature matrix shape:   {tr_ord.shape}")

        # Instantiate benchmark orchestrator
        benchmark = SequenceCheckpointBenchmark(
            checkpoint_n_obs=n_obs,
            horizon_h=args.horizon,
            random_state=args.random_state,
        )

        # Fit all models strictly on train partition
        print("  -> Fitting all 6 models strictly on training set...")
        benchmark.fit(
            texts_train=tr_texts,
            agg_features_train=tr_agg,
            ord_features_train=tr_ord,
            y_train=y_train,
        )

        # Select decision thresholds on validation partition only
        print("  -> Tuning decision thresholds on validation partition (maximizing F1, tie-breaking to higher threshold)...")
        tune_res = benchmark.tune_thresholds(
            texts_val=val_texts,
            agg_features_val=val_agg,
            ord_features_val=val_ord,
            y_val=y_val,
        )
        for m_name, t_info in tune_res.items():
            fallback_note = " [FALLBACK]" if t_info["fallback_used"] else ""
            print(f"     {m_name:<24}: Thresh={t_info['threshold']:.4f}, Val F1={t_info['val_f1']:.4f}{fallback_note}")

        # Evaluate validation set
        val_eval = benchmark.evaluate_split(val_texts, val_agg, val_ord, y_val, "val")
        for m_name in SequenceCheckpointBenchmark.MODEL_NAMES:
            all_metrics.append(val_eval[m_name])

        # Evaluate test set with frozen thresholds
        print("  -> Evaluating test partition with frozen validation thresholds...")
        test_eval = benchmark.evaluate_split(te_texts, te_agg, te_ord, y_test, "test")
        for m_name in SequenceCheckpointBenchmark.MODEL_NAMES:
            m_res = test_eval[m_name]
            all_metrics.append(m_res)
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

        # Save model artifacts under models/
        joblib.dump(benchmark.aggregate_model, models_dir / f"aggregate_state_logreg_n{n_obs}.joblib")
        joblib.dump(benchmark.ordered_model, models_dir / f"ordered_state_logreg_n{n_obs}.joblib")
        joblib.dump(benchmark.hybrid_model, models_dir / f"hybrid_logreg_n{n_obs}.joblib")
        joblib.dump(benchmark.tfidf_model, models_dir / f"tfidf_text_logreg_n{n_obs}.joblib")

        # Compute full-precision test probabilities
        test_probs = benchmark.predict_probabilities(te_texts, te_agg, te_ord)
        pred_dict: Dict[str, Any] = {
            "example_id": te_sub["example_id"],
            "conversation_id": te_sub["conversation_id"],
            "pair_id": te_sub["pair_id"],
            "split": te_sub["split"],
            "checkpoint_n_obs": n_obs,
            "horizon_h": args.horizon,
            "true_label": y_test,
        }

        # Store unrounded full-precision scores and decisions
        prefix_mapping = {
            "MajorityBaseline": "majority",
            "PriorBaseline": "prior",
            "TFIDF_TextBaseline": "tfidf",
            "AggregateState_LogReg": "aggregate",
            "OrderedState_LogReg": "ordered",
            "Hybrid_LogReg": "hybrid",
        }
        for m_name in SequenceCheckpointBenchmark.MODEL_NAMES:
            short_prefix = prefix_mapping[m_name]
            probs_arr = test_probs[m_name]
            thresh = benchmark.thresholds[m_name]
            preds_arr = (probs_arr >= thresh).astype(int)

            pred_dict[f"{short_prefix}_prob"] = probs_arr  # Full float64 precision
            pred_dict[f"{short_prefix}_pred"] = preds_arr
            pred_dict[f"{short_prefix}_threshold"] = thresh

        pred_df = pd.DataFrame(pred_dict)
        pred_csv_path = reports_dir / f"sequence_test_predictions_n{n_obs}.csv"
        pred_df.to_csv(pred_csv_path, index=False)
        print(f"  -> Saved full-precision test predictions ({len(pred_df)} rows) to {pred_csv_path}")

    # 4. Save machine-readable outputs
    print("\n[Step 4/5] Saving benchmark metrics and summary tables...")
    metrics_json_path = reports_dir / "sequence_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  -> Saved full metrics to {metrics_json_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = reports_dir / "sequence_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"  -> Saved summary table to {summary_csv_path}")

    # 5. Display terminal summary
    print("\n" + "=" * 80)
    print("TASK 3 BENCHMARK TEST SUMMARY (FROZEN VALIDATION THRESHOLDS)")
    print("=" * 80)
    display_df = summary_df[[
        "checkpoint", "model", "prevalence", "threshold", "AP", "ROC_AUC",
        "Accuracy", "Precision", "Recall", "F1", "Brier_Score"
    ]]
    print(display_df.to_string(index=False))
    print("=" * 80)
    print("TASK 3 BENCHMARK EXECUTION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
