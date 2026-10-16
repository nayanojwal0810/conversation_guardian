# ConversationGuardian — Gate 1

ConversationGuardian is an academic research and engineering platform for early warning and moderation of conversational derailment in online discussion platforms.

This repository implements **Gate 1: Fresh Data Pipeline & Forecast-Target Validation**, establishing a clean-room, leakage-free data construction process using the Cornell ConvoKit CGA-WIKI corpus (*conversations-gone-awry-corpus*; Zhang et al., ACL 2018).

---

## 1. Project Purpose & Precise Target Definition

The objective of Gate 1 is to define and validate an online conversational event-forecasting target without data leakage or post-attack lookahead.

### Forecasting Task
- **Primary Target:** Predict whether an annotated personal attack occurs within the next three messages ($H = 3$) following an observed prefix of length $n_{\text{observed}}$.
- **Observation Checkpoints:** Fixed conversational prefixes $n_{\text{observed}} \in \{1, 3, 5\}$.
- **Observed Window:** Dialogue turns $0 \dots n_{\text{observed}} - 1$.
- **Forecast Window:** Dialogue turns $n_{\text{observed}} \dots n_{\text{observed}} + 2$.

### Boundary Decision Rules
- **Insufficient Observed Prefix ($L < n_{\text{observed}}$):** Excluded (`INSUFFICIENT_OBSERVED_PREFIX`), target label is `None`.
- **Attack Already Observed ($p_{\text{att}} < n_{\text{observed}}$):** Excluded (`ATTACK_ALREADY_OBSERVED`), target label is `None`. Once derailment has already occurred, predicting future derailment is no longer an early warning.
- **Positive Event ($p_{\text{att}} \in [n_{\text{observed}}, n_{\text{observed}} + 2]$):** Eligible (`POSITIVE_EVENT`), target label is `1`. This holds even if the dialogue subsequently terminates before the full 3 turns elapse.
- **Negative Event (No attack in window and $L \ge n_{\text{observed}} + 3$):** Eligible (`NEGATIVE_EVENT`), target label is `0`.
- **Right-Censored Window (No attack in window and $L < n_{\text{observed}} + 3$):** Excluded (`CENSORED_INCOMPLETE_WINDOW`), target label is `None`. Peaceful dialogue that ends before 3 future turns can be observed is right-censored and excluded from binary classification.

---

## 2. Environment & Prerequisites

- **Supported Python Runtime:** Python 3.10 to 3.13 (Tested on Python 3.13.3 64-bit).
- **Core Dependencies:**
  - `convokit==4.1.2`
  - `pandas==3.0.3`
  - `numpy==2.4.6`
  - `pyarrow==24.0.0`
  - `scikit-learn==1.9.0`

### Setup Instructions

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt
```

---

## 3. Data Acquisition & Corpus Access

The pipeline uses the official Cornell ConvoKit `conversations-gone-awry-corpus` (4,188 Wikipedia talk page conversations; 30,021 utterances; 2,094 mutual pairs).

The source loader automatically downloads the official corpus using `convokit.download('conversations-gone-awry-corpus')` if not already present in the standard ConvoKit cache directory (`~/.convokit/saved-corpora/conversations-gone-awry-corpus`).

Alternatively, an existing local corpus directory can be explicitly specified:
```bash
python scripts/prepare_cga_event_data.py --source-dir /path/to/conversations-gone-awry-corpus
```
If `--source-dir` is provided but invalid or missing required files (`conversations.json`, `utterances.jsonl`), the loader fails fast with a clear error rather than falling back silently.

---

## 4. Running the Pipeline

Execute the standalone CLI data preparation script:

```bash
python scripts/prepare_cga_event_data.py
```

Optional CLI flags:
- `--source-dir <path>`: Path to local raw ConvoKit corpus (defaults to ConvoKit cache/download).
- `--output-dir <path>`: Destination directory for partitioned event Parquet files (default: `data/events`).
- `--reports-dir <path>`: Destination directory for audit reports (default: `reports/gate1`).
- `--horizon <int>`: Forecasting horizon $H$ (default: `3`).
- `--checkpoints <int ...>`: Observation checkpoints (default: `1 3 5`).

The script enforces strict validation checks:
- Pair reciprocity and zero cross-partition leakage across official `train`, `val`, and `test` splits.
- 100% agreement between conversation-level attack flags and turn-level attack annotations.
- Full execution of the regression test suite, exiting with a non-zero exit code if any test fails.

---

## 5. Running Tests

Execute the unit and regression test suite:

```bash
python -m unittest discover -s tests -v
```

All 12+ critical boundary conditions and integrity checks are covered.

---

## 6. Project Artifacts & Commit Policy

### Tracked in Git
- `src/conversation_guardian/`: Core data structures, source loader, event builder, and ML baseline routines.
- `scripts/prepare_cga_event_data.py`: Deterministic CLI dataset builder.
- `scripts/run_baseline_benchmark.py`: Word+Char TF-IDF Logistic Regression & Prior baseline benchmark CLI.
- `tests/`: Comprehensive unit and regression test suites.
- `reports/gate1/`: Audit artifacts and data dictionary.
- `reports/benchmarks/`: Benchmark evaluation metrics and test prediction records.

### Intentionally Not Committed (`.gitignore`)
- `data/` and `data/events/*.parquet`: Generated event dataset files are kept out of Git.
- `models/`: Trained model binaries (`*.joblib`, `*.pt`, `*.bin`).
- Raw corpus caches and virtual environments (`.venv/`, `__pycache__/`).
- Model checkpoints and credentials.

---

## 7. Running the Machine Learning Benchmark

Train and evaluate the checkpoint-specific word-and-character TF-IDF Logistic Regression baseline against the empirical prior baseline:

```bash
python scripts/run_baseline_benchmark.py
```

This fits models strictly on eligible training examples, selects optimal F1 classification thresholds on validation predictions, freezes the thresholds, and reports final test metrics (AP, ROC-AUC, Precision, Recall, F1, Accuracy, Brier score, Confusion matrix).