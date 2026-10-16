# ConversationGuardian Data Dictionary — Gate 1

## 1. Overview & Provenance

This dataset is constructed from the official Cornell ConvoKit `conversations-gone-awry-corpus` (CGA-WIKI; Zhang et al., ACL 2018).
All records are processed using strict clean-room boundary definitions for fixed-checkpoint early-warning prediction ($H = 3$, $n_{\text{observed}} \in \{1, 3, 5\}$).

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
3. **Dialogue Position Indexing:** Zero-based position indices $0, 1, 2, \dots, L-1$ are assigned strictly to the remaining authentic dialogue comments.
4. **Filtered Length $L$:** Represents the count of authentic dialogue messages. All 4,188 conversations in the corpus have $L \ge 3$ dialogue messages after header removal.

---

## 4. Forecasting Target Specification

- **Primary Target Definition:** Whether an annotated personal attack occurs within the next three messages ($H = 3$) following an observed prefix of length $n_{\text{observed}}$.
- **Observation Checkpoints:** $n_{\text{observed}} \in \{1, 3, 5\}$.
- **Observed Input Window:** Zero-based positions $0 \dots n_{\text{observed}} - 1$.
- **Forecast Window:** Zero-based positions $n_{\text{observed}} \dots n_{\text{observed}} + 2$.

### 4.1 Boundary Decision Rules
| Condition | Eligibility Status | Reason Code | Target Label | Description |
|:---|:---|:---|:---:|:---|
| $L < n_{\text{observed}}$ | `EXCLUDED` | `INSUFFICIENT_OBSERVED_PREFIX` | `None` | Dialogue ended before the requested observation checkpoint. |
| $\exists \text{ attack at index } < n_{\text{observed}}$ | `EXCLUDED` | `ATTACK_ALREADY_OBSERVED` | `None` | Attack already occurred inside the observed input; predicting derailment is no longer an early warning. |
| $\exists \text{ attack at index } \in [n_{\text{observed}}, n_{\text{observed}} + 2]$ | `ELIGIBLE` | `POSITIVE_EVENT` | `1` | Attack occurs within the next 3 messages. Eligible even if thread terminates before 3 full future turns. |
| No attack in $[0, n_{\text{observed}} + 2]$ AND $L \ge n_{\text{observed}} + 3$ | `ELIGIBLE` | `NEGATIVE_EVENT` | `0` | All 3 future messages are observed and verified free of personal attacks. |
| No attack observed AND $L < n_{\text{observed}} + 3$ | `EXCLUDED` | `CENSORED_INCOMPLETE_WINDOW` | `None` | Dialogue ended peacefully before 3 future messages could be observed. Right-censored; excluded from binary classification. |

---

## 5. Strict Separation: Model Inputs vs. Metadata & Audit Fields

### 5.1 Model Input Predictor Allowlist (`MODEL_INPUT_ALLOWLIST`)
Direct predictors that machine learning models are permitted to consume:
- `observed_prefix_text`: Raw concatenated text of the observed comments ($0 \dots n_{\text{observed}} - 1$).

### 5.2 Metadata Columns (`METADATA_COLUMNS`)
Retained for tracking, partitioning, slicing, and off-line joins, but **never** passed directly into the model:
- `example_id`: Unique example key (`{conversation_id}__obs{n_obs}_h{h}`).
- `conversation_id`: Discussion thread identifier.
- `pair_id`: Partner discussion thread identifier.
- `split`: Dataset partition (`train`, `val`, `test`).
- `checkpoint_n_obs`: Observation checkpoint ($1, 3, 5$).
- `horizon_h`: Forecasting horizon ($3$).
- `observed_prefix_length`: Number of observed utterances ($n_{\text{observed}}$).
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
