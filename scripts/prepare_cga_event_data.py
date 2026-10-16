"""CLI script to build the CGA-WIKI event forecasting dataset and generate Gate-1 audit artifacts."""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

# Ensure src directory is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
SRC_DIR = ROOT_DIR / "src"
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
    ConversationRecord,
    EventExample,
)
from conversation_guardian.data.source_loader import (
    load_cga_source_records,
    generate_source_manifest,
)
from conversation_guardian.data.event_builder import (
    filter_and_order_utterances,
    build_event_dataset,
    examples_to_dataframe,
)


def generate_dataset_summary(
    conversations: Dict[str, ConversationRecord],
    output_path: Path
) -> pd.DataFrame:
    """Generate dataset_summary.csv showing source, utterance, attack, and length counts by split."""
    summary_rows = []
    splits = ["train", "val", "test", "overall"]

    for sp in splits:
        if sp == "overall":
            subset = list(conversations.values())
        else:
            subset = [c for c in conversations.values() if c.split == sp]

        n_convs = len(subset)
        if n_convs == 0:
            continue

        raw_utts = sum(len(c.utterances) for c in subset)
        header_utts = sum(
            sum(1 for u in c.utterances if u.is_section_header) for c in subset
        )
        filtered_dialogue_utts = raw_utts - header_utts
        attack_convs = sum(1 for c in subset if c.conversation_has_personal_attack)
        healthy_convs = n_convs - attack_convs

        # Dialogue lengths
        lengths = [
            len([u for u in c.utterances if not u.is_section_header])
            for c in subset
        ]
        mean_len = float(np.mean(lengths)) if lengths else 0.0
        median_len = float(np.median(lengths)) if lengths else 0.0
        min_len = int(np.min(lengths)) if lengths else 0
        max_len = int(np.max(lengths)) if lengths else 0

        summary_rows.append({
            "split": sp,
            "total_conversations": n_convs,
            "total_raw_utterances": raw_utts,
            "section_header_utterances": header_utts,
            "filtered_dialogue_utterances": filtered_dialogue_utts,
            "attack_annotated_conversations": attack_convs,
            "healthy_conversations": healthy_convs,
            "attack_prevalence_pct": round(attack_convs / n_convs * 100, 2),
            "dialogue_length_mean": round(mean_len, 2),
            "dialogue_length_median": round(median_len, 2),
            "dialogue_length_min": min_len,
            "dialogue_length_max": max_len,
        })

    df = pd.DataFrame(summary_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def generate_pair_split_audit(
    conversations: Dict[str, ConversationRecord],
    output_path: Path
) -> pd.DataFrame:
    """Generate pair_split_audit.csv verifying mutual reciprocity and zero split leakage."""
    audit_rows = []
    splits = ["train", "val", "test", "overall"]

    for sp in splits:
        if sp == "overall":
            subset = conversations
        else:
            subset = {cid: c for cid, c in conversations.items() if c.split == sp}

        n_convs = len(subset)
        unique_pairs = set()
        mutual_reciprocal_count = 0
        cross_split_violations = 0
        opposite_label_pairs = 0
        label_concordance_violations = 0

        for cid, c in subset.items():
            pid = c.pair_id
            if not pid or pid not in conversations:
                continue

            partner = conversations[pid]
            # Mutual reciprocity: partner.pair_id == cid
            if partner.pair_id == cid:
                mutual_reciprocal_count += 1

            # Cross-split check: partner.split == c.split
            if partner.split != c.split:
                cross_split_violations += 1

            # Opposite label check
            if c.conversation_has_personal_attack != partner.conversation_has_personal_attack:
                opposite_label_pairs += 1
            else:
                label_concordance_violations += 1

            pair_key = tuple(sorted([cid, pid]))
            unique_pairs.add(pair_key)

        audit_rows.append({
            "split": sp,
            "total_conversations": n_convs,
            "unique_mutual_pairs": len(unique_pairs),
            "conversations_with_mutual_reciprocity": mutual_reciprocal_count,
            "reciprocity_compliance_pct": round(mutual_reciprocal_count / n_convs * 100, 2) if n_convs else 0,
            "cross_split_pair_violations": cross_split_violations,
            "opposite_label_concordance_count": opposite_label_pairs,
            "same_label_pair_violations": label_concordance_violations,
        })

    df = pd.DataFrame(audit_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def generate_forecast_eligibility_table(
    examples: List[EventExample],
    output_path: Path
) -> pd.DataFrame:
    """Generate forecast_eligibility_by_checkpoint.csv detailing eligibility counts and prevalence."""
    rows = []
    grouped = defaultdict(list)
    for ex in examples:
        grouped[(ex.checkpoint_n_obs, ex.split)].append(ex)

    for n_obs in sorted(list(set(ex.checkpoint_n_obs for ex in examples))):
        for sp in ["train", "val", "test"]:
            sub = grouped.get((n_obs, sp), [])
            n_tot = len(sub)
            if n_tot == 0:
                continue

            n_insuff = sum(1 for ex in sub if ex.reason == REASON_INSUFFICIENT_OBSERVED_PREFIX)
            n_att_obs = sum(1 for ex in sub if ex.reason == REASON_ATTACK_ALREADY_OBSERVED)
            n_pos = sum(1 for ex in sub if ex.reason == REASON_POSITIVE_EVENT)
            n_neg = sum(1 for ex in sub if ex.reason == REASON_NEGATIVE_EVENT)
            n_cens = sum(1 for ex in sub if ex.reason == REASON_CENSORED_INCOMPLETE_WINDOW)

            n_elig_pos = sum(1 for ex in sub if ex.eligibility_status == STATUS_ELIGIBLE and ex.target_label == 1)
            n_elig_neg = sum(1 for ex in sub if ex.eligibility_status == STATUS_ELIGIBLE and ex.target_label == 0)
            n_elig = n_elig_pos + n_elig_neg
            pos_prev = round(n_elig_pos / n_elig * 100, 2) if n_elig > 0 else 0.0

            rows.append({
                "checkpoint_n_obs": n_obs,
                "split": sp,
                "total_conversations_considered": n_tot,
                "insufficient_observed_prefix": n_insuff,
                "attack_already_observed": n_att_obs,
                "positive_event_within_window": n_pos,
                "negative_event_full_window": n_neg,
                "censored_incomplete_window": n_cens,
                "eligible_positives": n_elig_pos,
                "eligible_negatives": n_elig_neg,
                "total_eligible_examples": n_elig,
                "positive_prevalence_pct": pos_prev,
            })

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def generate_data_dictionary(output_path: Path) -> None:
    """Generate reports/gate1/data_dictionary.md."""
    content = """# ConversationGuardian Data Dictionary — Gate 1

## 1. Overview & Provenance

This dataset is constructed from the official Cornell ConvoKit `conversations-gone-awry-corpus` (CGA-WIKI; Zhang et al., ACL 2018).
All records are processed using strict clean-room boundary definitions for fixed-checkpoint early-warning prediction ($H = 3$, $n_{\\text{observed}} \\in \\{1, 3, 5\\}$).

---

## 2. Source Corpus Entity Schemas

### 2.1 Conversations (`conversations.json`)
- `conversation_id` (string): Unique identifier for the Wikipedia talk page discussion thread. Matches the root utterance revision ID.
- `pair_id` (string): Identifier of the paired discussion thread originating from the same talk page. In this corpus, `pair_id` is the exact `conversation_id` of the reciprocal partner thread.
- `split` (string): Official author partition (`train`, `val`, `test`). Preserved without modification.
- `conversation_has_personal_attack` (boolean): Ground-truth indicator of whether any human annotator marked an utterance in this thread as containing a personal attack.
- `page_title` (string): Title of the Wikipedia talk page.
- `page_id` (integer): Wikipedia article / user talk page identifier.
- `verified` (boolean): Quality flag from Cornell corpus curation.
- `pair_verified` (boolean): Flag indicating talk-page pair consistency.
- `annotation_year` (string): Year of Crowdflower / MTurk annotation (2018).

### 2.2 Utterances (`utterances.jsonl`)
- `id` (string): Unique utterance revision ID.
- `conversation_id` (string): Foreign key referencing the parent conversation.
- `speaker` (string): Wikipedia username or IP address of the commenter.
- `reply-to` (string / null): Utterance ID to which this comment responds.
- `timestamp` (float): Unix epoch timestamp of the Wikipedia revision.
- `text` (string): Raw comment text.
- `meta.is_section_header` (boolean): True if the utterance is a Wikipedia talk section heading (e.g., `== Discussion Topic ==`).
- `meta.comment_has_personal_attack` (boolean): True if this specific comment was annotated as containing a personal attack.
- `meta.toxicity` (float): Perspective API toxicity score recorded at corpus curation.
- `meta.parsed` (array of objects): Dependency parse tree tokens (retained in raw source, excluded from event feature matrices).

---

## 3. Preprocessing & Section Header Treatment Rule

1. **Ordering Rule:** Utterances within a conversation are sorted in ascending order of `timestamp`. Python's stable Timsort is used, preserving original source file order when timestamps are identical.
2. **Section Header Removal:** Utterances with `meta.is_section_header == True` are removed prior to computing message sequence positions.
3. **Dialogue Position Indexing:** Zero-based position indices $0, 1, 2, \\dots, L-1$ are assigned strictly to the remaining authentic dialogue comments.
4. **Filtered Length $L$:** Represents the count of authentic dialogue messages. All 4,188 conversations in the corpus have $L \\ge 3$ dialogue messages after header removal.

---

## 4. Forecasting Target Specification

- **Primary Target Definition:** Whether an annotated personal attack occurs within the next three messages ($H = 3$) following an observed prefix of length $n_{\\text{observed}}$.
- **Observation Checkpoints:** $n_{\\text{observed}} \\in \\{1, 3, 5\\}$.
- **Observed Input Window:** Zero-based positions $0 \\dots n_{\\text{observed}} - 1$.
- **Forecast Window:** Zero-based positions $n_{\\text{observed}} \\dots n_{\\text{observed}} + 2$.

### 4.1 Boundary Decision Rules
| Condition | Eligibility Status | Reason Code | Target Label | Description |
|:---|:---|:---|:---:|:---|
| $L < n_{\\text{observed}}$ | `EXCLUDED` | `INSUFFICIENT_OBSERVED_PREFIX` | `None` | Dialogue ended before the requested observation checkpoint. |
| $\\exists \\text{ attack at index } < n_{\\text{observed}}$ | `EXCLUDED` | `ATTACK_ALREADY_OBSERVED` | `None` | Attack already occurred inside the observed input; predicting derailment is no longer an early warning. |
| $\\exists \\text{ attack at index } \\in [n_{\\text{observed}}, n_{\\text{observed}} + 2]$ | `ELIGIBLE` | `POSITIVE_EVENT` | `1` | Attack occurs within the next 3 messages. Eligible even if thread terminates before 3 full future turns. |
| No attack in $[0, n_{\\text{observed}} + 2]$ AND $L \\ge n_{\\text{observed}} + 3$ | `ELIGIBLE` | `NEGATIVE_EVENT` | `0` | All 3 future messages are observed and verified free of personal attacks. |
| No attack observed AND $L < n_{\\text{observed}} + 3$ | `EXCLUDED` | `CENSORED_INCOMPLETE_WINDOW` | `None` | Dialogue ended peacefully before 3 future messages could be observed. Right-censored; excluded from binary classification. |

---

## 5. Strict Separation: Model Inputs vs. Metadata & Audit Fields

### 5.1 Model Input Predictor Allowlist (`MODEL_INPUT_ALLOWLIST`)
Direct predictors that machine learning models are permitted to consume:
- `observed_prefix_text`: Raw concatenated text of the observed comments ($0 \\dots n_{\\text{observed}} - 1$).

### 5.2 Metadata Columns (`METADATA_COLUMNS`)
Retained for tracking, partitioning, slicing, and off-line joins, but **never** passed directly into the model:
- `example_id`: Unique example key (`{conversation_id}__obs{n_obs}_h{h}`).
- `conversation_id`: Discussion thread identifier.
- `pair_id`: Partner discussion thread identifier.
- `split`: Dataset partition (`train`, `val`, `test`).
- `checkpoint_n_obs`: Observation checkpoint ($1, 3, 5$).
- `horizon_h`: Forecasting horizon ($3$).
- `observed_prefix_length`: Number of observed utterances ($n_{\\text{observed}}$).
- `observed_utterance_ids`: List of IDs for observed comments.
- `observed_speakers`: Raw speaker identities (excluded to prevent user-specific bias).
- `observed_timestamps`: Raw timestamp values.
- `total_conversation_length_raw`: Raw utterance count including headers.
- `total_dialogue_length_filtered`: Filtered utterance count ($L$).
- `future_messages_available`: Number of remaining future messages.
- `dataset_version`: Dataset build version string.

### 5.3 Forbidden Audit Fields (`FORBIDDEN_AUDIT_COLUMNS`)
Strictly prohibited from feature matrices and model inputs:
- `target_label`: Ground truth binary label ($1$ or $0$).
- `eligibility_status`: Eligibility flag (`ELIGIBLE`, `EXCLUDED`).
- `reason`: Classification/censoring reason code.
- `has_attack_in_conversation`: Conversation-level overall attack label.
- `first_attack_pos`: Zero-based index of the first annotated attack.
- `first_attack_utterance_id`: Revision ID of the first attack utterance.
- `censoring_reason`: Detailed censoring annotation.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")


def generate_target_boundary_examples(output_path: Path) -> None:
    """Generate reports/gate1/target_boundary_examples.md with synthetic walkthroughs."""
    content = """# ConversationGuardian Target Boundary Examples — Gate 1

This document specifies the exact boundary evaluation behavior across all eight canonical cases using both formal notation and synthetic conversation scenarios.

---

## Boundary Decision Matrix

| Scenario # | Filtered Dialogue Length ($L$) | Attack Position ($p_{\\text{att}}$) | Checkpoint ($n_{\\text{obs}}$) | Window Range | Eligibility Status | Reason Code | Target Label | Rule Rationale |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|:---|
| **Case 1** | $6$ | None | $3$ | $[3, 4, 5]$ | `ELIGIBLE` | `NEGATIVE_EVENT` | `0` | All 3 future messages observed, 0 attacks. |
| **Case 2** | $5$ | None | $3$ | $[3, 4, 5]$ | `EXCLUDED` | `CENSORED_INCOMPLETE_WINDOW` | `None` | Only 2 future messages observed ($L=5 < 6$); peaceful termination is censored. |
| **Case 3** | $4$ | None | $5$ | $[5, 6, 7]$ | `EXCLUDED` | `INSUFFICIENT_OBSERVED_PREFIX` | `None` | Thread length ($4$) is shorter than the observation checkpoint ($5$). |
| **Case 4** | $6$ | $1$ | $3$ | $[3, 4, 5]$ | `EXCLUDED` | `ATTACK_ALREADY_OBSERVED` | `None` | Attack occurred at position 1, inside observed input $[0, 1, 2]$. |
| **Case 5** | $6$ | $3$ | $3$ | $[3, 4, 5]$ | `ELIGIBLE` | `POSITIVE_EVENT` | `1` | Attack occurs exactly at the first future message ($p_{\\text{att}} = n_{\\text{obs}}$). |
| **Case 6** | $7$ | $5$ | $3$ | $[3, 4, 5]$ | `ELIGIBLE` | `POSITIVE_EVENT` | `1` | Attack occurs at the boundary position of the 3-message window ($p_{\\text{att}} = n_{\\text{obs}} + 2$). |
| **Case 7** | $8$ | $6$ | $3$ | $[3, 4, 5]$ | `ELIGIBLE` | `NEGATIVE_EVENT` | `0` | Attack occurs outside window ($p_{\\text{att}} = 6 > 5$); window $[3, 4, 5]$ is complete and clean. |
| **Case 8** | $4$ | $3$ | $3$ | $[3, 4, 5]$ | `ELIGIBLE` | `POSITIVE_EVENT` | `1` | Attack occurs at position 3, followed by immediate thread termination. Positive event holds despite incomplete window. |

---

## Detailed Case Walkthroughs

### Case 1: Fully Observed Negative Event
- **Setup:** A conversation between two editors discussing citations spans 6 turns ($0, 1, 2, 3, 4, 5$). No personal attacks occur anywhere.
- **Evaluation at $n_{\\text{obs}} = 3$:**
  - Observed input: comments $0, 1, 2$.
  - Forecast window: comments $3, 4, 5$.
  - Future messages available: $6 - 3 = 3$ (positions $3, 4, 5$).
  - Attacks in window: $0$.
  - Result: **ELIGIBLE**, `NEGATIVE_EVENT`, label = **0**.

### Case 2: Right-Censored Peaceful Dialogue
- **Setup:** A short conversation concludes politely in 5 turns ($0, 1, 2, 3, 4$). No personal attacks occur.
- **Evaluation at $n_{\\text{obs}} = 3$:**
  - Observed input: comments $0, 1, 2$.
  - Forecast window requires 3 future turns ($3, 4, 5$).
  - Future messages available: only $2$ turns ($3, 4$).
  - Attacks in window: $0$.
  - Result: **EXCLUDED**, `CENSORED_INCOMPLETE_WINDOW`, label = **None**.
  - *Academic Rationale:* Calling this an ordinary negative would artificially inflate model negative specificity by assuming a hypothetical 6th turn would never derail.

### Case 3: Insufficient Observation Prefix
- **Setup:** A conversation consists of 4 turns ($0, 1, 2, 3$).
- **Evaluation at $n_{\\text{obs}} = 5$:**
  - Observation requirement: 5 turns ($0, 1, 2, 3, 4$).
  - Available turns: 4.
  - Result: **EXCLUDED**, `INSUFFICIENT_OBSERVED_PREFIX`, label = **None**.

### Case 4: Attack Already Observed (Post-Derailment)
- **Setup:** An attack is launched early at turn 1. The thread continues for 5 more turns ($L = 6$).
- **Evaluation at $n_{\\text{obs}} = 3$:**
  - Observed input: turns $0, 1, 2$. Turn 1 is an attack.
  - Result: **EXCLUDED**, `ATTACK_ALREADY_OBSERVED`, label = **None**.
  - *Academic Rationale:* Forecasting whether an attack will occur when an attack has already occurred in plain view is a trivial detection task, not an early warning.

### Case 5: Immediate Next-Turn Derailment
- **Setup:** Turns $0, 1, 2$ are civil. At turn 3, an editor insults another contributor. The thread terminates at turn 6.
- **Evaluation at $n_{\\text{obs}} = 3$:**
  - Observed input: turns $0, 1, 2$.
  - Forecast window: turns $3, 4, 5$. Turn 3 contains an annotated attack.
  - Result: **ELIGIBLE**, `POSITIVE_EVENT`, label = **1**.

### Case 6: Window-Boundary Derailment
- **Setup:** Turns $0, 1, 2$ are civil. Turns $3$ and $4$ show escalating tension without personal attacks. Turn $5$ erupts into a personal attack.
- **Evaluation at $n_{\\text{obs}} = 3$:**
  - Forecast window: turns $3, 4, 5$.
  - Turn 5 is inside the 3-turn horizon ($p_{\\text{att}} = 3 + 3 - 1 = 5$).
  - Result: **ELIGIBLE**, `POSITIVE_EVENT`, label = **1**.

### Case 7: Delayed Derailment Outside Window
- **Setup:** An 8-turn conversation ($0 \\dots 7$). Turns $0 \\dots 5$ are free of attacks. At turn 6, an attack occurs.
- **Evaluation at $n_{\\text{obs}} = 3$:**
  - Forecast window: turns $3, 4, 5$.
  - Turns $3, 4, 5$ are fully observed and contain 0 attacks.
  - Attack at turn 6 is outside the $H=3$ window.
  - Result: **ELIGIBLE**, `NEGATIVE_EVENT`, label = **0**.
  - *Academic Rationale:* The model is tasked with predicting whether an attack happens *within the next 3 turns*. Since it does not occur within those 3 turns, the negative forecast is correct for that horizon.

### Case 8: Positive Event with Premature Termination
- **Setup:** Turns $0, 1, 2$ are observed. Turn 3 is a vitriolic attack, and the thread is immediately locked or abandoned ($L = 4$).
- **Evaluation at $n_{\\text{obs}} = 3$:**
  - Forecast window: turn 3 is observed and positive.
  - Even though turns 4 and 5 do not exist, a positive event has already manifested within the window.
  - Result: **ELIGIBLE**, `POSITIVE_EVENT`, label = **1**.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")


def generate_test_summary(output_path: Path) -> bool:
    """Run unittest suite programmatically and write reports/gate1/test_summary.txt. Return True if passed."""
    import unittest
    import io

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(ROOT_DIR / "tests"), pattern="test_*.py")

    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)

    status_str = "PASSED" if result.wasSuccessful() else "FAILED"
    output_lines = [
        "======================================================================",
        "CONVERSATIONGUARDIAN GATE-1 REGRESSION TEST SUITE EXECUTION SUMMARY",
        "======================================================================",
        f"Executed Command: python -m unittest discover -s tests -v",
        f"Total Tests Executed: {result.testsRun}",
        f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}",
        f"Failures: {len(result.failures)}",
        f"Errors: {len(result.errors)}",
        f"Status: {status_str}",
        "======================================================================",
        "TEST RESULTS DETAIL:",
        stream.getvalue(),
        "======================================================================",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines) + "\n")

    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description="Prepare CGA Event Forecasting Dataset — Gate 1")
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Path to raw ConvoKit conversations-gone-awry-corpus directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "data" / "events"),
        help="Output directory for generated event parquet files",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str(ROOT_DIR / "reports" / "gate1"),
        help="Output directory for Gate 1 audit reports",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0-gate1",
        help="Dataset build version",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="Forecasting horizon H (default: 3)",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=int,
        default=[1, 3, 5],
        help="Observation checkpoints n_obs (default: 1 3 5)",
    )

    args = parser.parse_args()
    reports_dir = Path(args.reports_dir)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("CONVERSATIONGUARDIAN — GATE 1: FRESH DATA PIPELINE BUILDER")
    print("=" * 70)

    # 1. Source manifest (portable provenance)
    print("\n[Step 1/6] Generating portable source manifest and computing checksums...")
    manifest = generate_source_manifest(args.source_dir)
    manifest_path = reports_dir / "source_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  -> Saved portable source manifest to {manifest_path}")

    # 2. Load raw source records with fail-fast validation
    print("\n[Step 2/6] Loading official ConvoKit corpus records with structural validation...")
    convs = load_cga_source_records(args.source_dir)
    print(f"  -> Successfully loaded and verified {len(convs)} conversations from source.")

    # 3. Source summaries and audits
    print("\n[Step 3/6] Generating dataset summary and pair-split audit tables...")
    summary_path = reports_dir / "dataset_summary.csv"
    generate_dataset_summary(convs, summary_path)
    print(f"  -> Saved dataset summary to {summary_path}")

    pair_audit_path = reports_dir / "pair_split_audit.csv"
    generate_pair_split_audit(convs, pair_audit_path)
    print(f"  -> Saved pair split audit to {pair_audit_path}")

    # 4. Build event forecasting dataset
    print(f"\n[Step 4/6] Building event forecasting dataset (checkpoints={args.checkpoints}, H={args.horizon})...")
    examples = build_event_dataset(
        conversations=convs,
        checkpoints=args.checkpoints,
        horizon_h=args.horizon,
        exclude_section_headers=True,
        dataset_version=args.version,
    )
    print(f"  -> Constructed {len(examples)} total observation examples across all checkpoints.")

    # Save eligibility table
    eligibility_path = reports_dir / "forecast_eligibility_by_checkpoint.csv"
    generate_forecast_eligibility_table(examples, eligibility_path)
    print(f"  -> Saved eligibility table to {eligibility_path}")

    # Save event parquet datasets partitioned by split
    output_dir.mkdir(parents=True, exist_ok=True)
    df_all = examples_to_dataframe(examples)
    for sp in ["train", "val", "test"]:
        sp_df = df_all[df_all["split"] == sp].reset_index(drop=True)
        sp_parquet_path = output_dir / f"{sp}_events.parquet"
        sp_df.to_parquet(sp_parquet_path, index=False)
        print(f"  -> Saved {len(sp_df)} {sp} event records to {sp_parquet_path}")

    # 5. Documentation artifacts
    print("\n[Step 5/6] Writing data dictionary and target boundary examples...")
    data_dict_path = reports_dir / "data_dictionary.md"
    generate_data_dictionary(data_dict_path)
    print(f"  -> Saved data dictionary to {data_dict_path}")

    boundary_path = reports_dir / "target_boundary_examples.md"
    generate_target_boundary_examples(boundary_path)
    print(f"  -> Saved target boundary examples to {boundary_path}")

    # 6. Run regression tests and emit test summary
    print("\n[Step 6/6] Executing unit test suite and generating test summary...")
    test_summary_path = reports_dir / "test_summary.txt"
    tests_passed = generate_test_summary(test_summary_path)
    print(f"  -> Saved test summary to {test_summary_path}")

    if not tests_passed:
        print("\n[ERROR] Regression test suite FAILED. Aborting pipeline with exit code 1.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("GATE 1 DATA PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
