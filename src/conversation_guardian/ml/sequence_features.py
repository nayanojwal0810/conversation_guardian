"""
Sequential and conversational state feature extraction for personal-attack forecasting.

This module extracts lightweight, interpretable, CPU-friendly signals from dialogue
sequences using sentiment analysis (VADER), a deterministic hostility lexicon,
direct-address markers, and structural text properties.

NOTE: These signals are heuristic conversational indicators derived strictly from observed
message text, NOT ground-truth attack labels or validated toxicity scores. They must never
use future messages, target labels, or metadata from outside the observation window.
"""

import re
import itertools
from typing import Dict, List, Tuple, Optional, Any, FrozenSet
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Predefined Deterministic Lexicons and Thresholds
# ---------------------------------------------------------------------------

# Explicit, fixed hostility lexicon of overt insult words, profanities, and hostile phrases.
# Predeclared and frozen prior to model fitting and test evaluation.
HOSTILITY_LEXICON: FrozenSet[str] = frozenset([
    "asshole",
    "bastard",
    "bitch",
    "bullshit",
    "clown",
    "coward",
    "crap",
    "damn",
    "damned",
    "disgusting",
    "fool",
    "foolish",
    "fuck",
    "fucked",
    "fucking",
    "garbage",
    "get lost",
    "hate",
    "hateful",
    "hypocrite",
    "hypocrisy",
    "idiot",
    "idiotic",
    "ignorant",
    "incompetent",
    "jerk",
    "liar",
    "loser",
    "lying",
    "moron",
    "moronic",
    "nonsense",
    "pathetic",
    "piss off",
    "ridiculous",
    "rubbish",
    "scumbag",
    "shit",
    "shitty",
    "shut up",
    "stop lying",
    "stupid",
    "troll",
    "trolling",
    "vandal",
    "vandalism",
])

# Compiled regex for exact word/phrase boundary matching (case-insensitive)
_HOSTILITY_REGEX = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in sorted(HOSTILITY_LEXICON, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Second-person direct-address pronouns and @mentions
_DIRECT_ADDRESS_REGEX = re.compile(
    r"\b(?:you|your|yours|yourself|yourselves|u|ur)\b|@[a-zA-Z0-9_]+",
    re.IGNORECASE,
)

# Canonical discrete state categories
# Priority: HOSTILE match > POSITIVE (compound >= 0.05) > NEGATIVE (compound <= -0.05) > NEUTRAL
ORDERED_STATES: Tuple[str, ...] = ("POSITIVE", "NEUTRAL", "NEGATIVE", "HOSTILE")

VADER_POSITIVE_THRESHOLD: float = 0.05
VADER_NEGATIVE_THRESHOLD: float = -0.05


def classify_message_state(compound_score: float, hostility_count: int) -> str:
    """
    Deterministic categorical state assignment for a single dialogue message.
    
    Fixed rule hierarchy:
      1. HOSTILE: if message contains >= 1 match from HOSTILITY_LEXICON.
      2. POSITIVE: if VADER compound score >= +0.05.
      3. NEGATIVE: if VADER compound score <= -0.05.
      4. NEUTRAL: otherwise (-0.05 < compound < +0.05).
    """
    if hostility_count > 0:
        return "HOSTILE"
    if compound_score >= VADER_POSITIVE_THRESHOLD:
        return "POSITIVE"
    if compound_score <= VADER_NEGATIVE_THRESHOLD:
        return "NEGATIVE"
    return "NEUTRAL"


def extract_message_signals(
    text: str,
    analyzer: Optional[SentimentIntensityAnalyzer] = None,
) -> Dict[str, Any]:
    """
    Extract CPU-friendly interpretable signals for a single observed message.
    
    Returns:
        Dict with sentiment scores, hostility counts, direct-address counts,
        punctuation/capitalization metrics, and categorical state.
    """
    if analyzer is None:
        analyzer = SentimentIntensityAnalyzer()

    clean_text = str(text or "")
    sentiment_scores = analyzer.polarity_scores(clean_text)
    compound = float(sentiment_scores["compound"])
    pos = float(sentiment_scores["pos"])
    neg = float(sentiment_scores["neg"])
    neu = float(sentiment_scores["neu"])

    hostility_matches = _HOSTILITY_REGEX.findall(clean_text)
    hostility_count = len(hostility_matches)
    has_hostility = 1 if hostility_count > 0 else 0

    direct_address_matches = _DIRECT_ADDRESS_REGEX.findall(clean_text)
    direct_address_count = len(direct_address_matches)

    exclamation_count = clean_text.count("!")
    question_count = clean_text.count("?")

    char_len = len(clean_text)
    word_count = len(clean_text.split())

    # Uppercase ratio: uppercase alphabetic characters relative to total length
    caps_count = sum(1 for c in clean_text if c.isupper())
    caps_ratio = float(caps_count / max(char_len, 1))

    state = classify_message_state(compound, hostility_count)

    return {
        "text": clean_text,
        "vader_compound": compound,
        "vader_pos": pos,
        "vader_neg": neg,
        "vader_neu": neu,
        "hostility_count": hostility_count,
        "has_hostility": has_hostility,
        "direct_address_count": direct_address_count,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "caps_ratio": caps_ratio,
        "char_length": char_len,
        "word_count": word_count,
        "state": state,
    }


# ---------------------------------------------------------------------------
# Feature Extractors: Model A (Aggregate) and Model B (Ordered)
# ---------------------------------------------------------------------------

def get_aggregate_feature_names() -> List[str]:
    """Return ordered list of feature column names for Model A (Aggregate State)."""
    names: List[str] = []
    # State counts and proportions (order-agnostic)
    for s in ORDERED_STATES:
        names.append(f"state_count_{s}")
        names.append(f"state_prop_{s}")
    # Sentiment summaries
    names.extend([
        "compound_mean",
        "compound_min",
        "compound_max",
        "pos_mean",
        "neg_mean",
        "neu_mean",
        "neg_max",
    ])
    # Signal aggregates
    names.extend([
        "hostility_total_count",
        "hostility_msg_count",
        "hostility_prop",
        "direct_address_total",
        "direct_address_mean",
        "exclamation_total",
        "question_total",
        "caps_ratio_mean",
        "caps_ratio_max",
        "total_char_length",
        "mean_char_length",
        "total_word_count",
        "mean_word_count",
    ])
    return names


def extract_aggregate_features(message_signals: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Model A: Order-agnostic aggregate state and signal summaries.
    
    Contains strictly ZERO position-specific, transition, or trajectory features.
    Permuting the message order produces identical feature output.
    """
    n = len(message_signals)
    if n == 0:
        raise ValueError("Cannot extract features from an empty sequence of messages.")

    feat: Dict[str, float] = {}

    # State counts and proportions
    states = [m["state"] for m in message_signals]
    for s in ORDERED_STATES:
        cnt = states.count(s)
        feat[f"state_count_{s}"] = float(cnt)
        feat[f"state_prop_{s}"] = float(cnt / n)

    # Sentiment summaries
    compounds = [m["vader_compound"] for m in message_signals]
    pos_scores = [m["vader_pos"] for m in message_signals]
    neg_scores = [m["vader_neg"] for m in message_signals]
    neu_scores = [m["vader_neu"] for m in message_signals]

    feat["compound_mean"] = float(np.mean(compounds))
    feat["compound_min"] = float(np.min(compounds))
    feat["compound_max"] = float(np.max(compounds))
    feat["pos_mean"] = float(np.mean(pos_scores))
    feat["neg_mean"] = float(np.mean(neg_scores))
    feat["neu_mean"] = float(np.mean(neu_scores))
    feat["neg_max"] = float(np.max(neg_scores))

    # Signal aggregates
    hostility_counts = [m["hostility_count"] for m in message_signals]
    hostility_msgs = [m["has_hostility"] for m in message_signals]
    direct_addrs = [m["direct_address_count"] for m in message_signals]
    exclamations = [m["exclamation_count"] for m in message_signals]
    questions = [m["question_count"] for m in message_signals]
    caps_ratios = [m["caps_ratio"] for m in message_signals]
    char_lens = [m["char_length"] for m in message_signals]
    word_lens = [m["word_count"] for m in message_signals]

    feat["hostility_total_count"] = float(np.sum(hostility_counts))
    feat["hostility_msg_count"] = float(np.sum(hostility_msgs))
    feat["hostility_prop"] = float(np.mean(hostility_msgs))

    feat["direct_address_total"] = float(np.sum(direct_addrs))
    feat["direct_address_mean"] = float(np.mean(direct_addrs))

    feat["exclamation_total"] = float(np.sum(exclamations))
    feat["question_total"] = float(np.sum(questions))

    feat["caps_ratio_mean"] = float(np.mean(caps_ratios))
    feat["caps_ratio_max"] = float(np.max(caps_ratios))

    feat["total_char_length"] = float(np.sum(char_lens))
    feat["mean_char_length"] = float(np.mean(char_lens))

    feat["total_word_count"] = float(np.sum(word_lens))
    feat["mean_word_count"] = float(np.mean(word_lens))

    return feat


def get_ordered_feature_names(checkpoint_n_obs: int) -> List[str]:
    """Return ordered list of feature column names for Model B (Ordered State) at checkpoint n."""
    names = list(get_aggregate_feature_names())

    # 1. Adjacent state transitions (16 pairs)
    for s1, s2 in itertools.product(ORDERED_STATES, ORDERED_STATES):
        names.append(f"trans_count_{s1}->{s2}")
        names.append(f"trans_prop_{s1}->{s2}")

    # 2. State trigrams (64 triples)
    for s1, s2, s3 in itertools.product(ORDERED_STATES, ORDERED_STATES, ORDERED_STATES):
        names.append(f"trigram_count_{s1}->{s2}->{s3}")

    # 3. Position-specific states and values for positions 1..checkpoint_n_obs
    for pos_idx in range(1, checkpoint_n_obs + 1):
        for s in ORDERED_STATES:
            names.append(f"pos_{pos_idx}_state_{s}")
        names.append(f"pos_{pos_idx}_compound")
        names.append(f"pos_{pos_idx}_has_hostility")
        names.append(f"pos_{pos_idx}_direct_address")

    # 4. Trajectory and change features
    names.extend([
        "sentiment_delta_last_first",
        "hostility_delta_last_first",
        "sentiment_delta_last_prev",
        "hostility_delta_last_prev",
        "sentiment_slope",
        "escalation_flag",
    ])

    return names


def extract_ordered_features(
    message_signals: List[Dict[str, Any]],
    checkpoint_n_obs: int,
) -> Dict[str, float]:
    """
    Model B: Ordered state sequence features.
    
    Includes all aggregate features from Model A, plus:
      - Adjacent state transitions (count and proportion)
      - Ordered state trigrams (count)
      - Position-specific states and metrics for each observed turn
      - Sentiment and hostility trajectory deltas, linear slope, and escalation flag.
    
    Gracefully handles n=1: transition/trajectory features naturally have 0 observations.
    """
    n = len(message_signals)
    if n != checkpoint_n_obs:
        raise ValueError(
            f"Message count mismatch: received {n} messages, expected checkpoint n={checkpoint_n_obs}"
        )

    # 1. Base aggregate features
    feat = extract_aggregate_features(message_signals)

    states = [m["state"] for m in message_signals]
    compounds = [m["vader_compound"] for m in message_signals]
    hostilities = [m["has_hostility"] for m in message_signals]

    # 2. Adjacent transitions (S_i -> S_{i+1})
    trans_counts: Dict[Tuple[str, str], int] = {pair: 0 for pair in itertools.product(ORDERED_STATES, ORDERED_STATES)}
    if n >= 2:
        for i in range(n - 1):
            pair = (states[i], states[i + 1])
            trans_counts[pair] += 1

    denom_trans = max(n - 1, 1)
    for (s1, s2), cnt in trans_counts.items():
        feat[f"trans_count_{s1}->{s2}"] = float(cnt)
        feat[f"trans_prop_{s1}->{s2}"] = float(cnt / denom_trans) if n >= 2 else 0.0

    # 3. State trigrams (S_i -> S_{i+1} -> S_{i+2})
    trigram_counts: Dict[Tuple[str, str, str], int] = {
        trip: 0 for trip in itertools.product(ORDERED_STATES, ORDERED_STATES, ORDERED_STATES)
    }
    if n >= 3:
        for i in range(n - 2):
            trip = (states[i], states[i + 1], states[i + 2])
            trigram_counts[trip] += 1

    for trip, cnt in trigram_counts.items():
        feat[f"trigram_count_{trip[0]}->{trip[1]}->{trip[2]}"] = float(cnt)

    # 4. Position-specific states and metrics for positions 1..checkpoint_n_obs
    for pos_idx in range(1, checkpoint_n_obs + 1):
        msg = message_signals[pos_idx - 1]
        msg_state = msg["state"]
        for s in ORDERED_STATES:
            feat[f"pos_{pos_idx}_state_{s}"] = 1.0 if msg_state == s else 0.0
        feat[f"pos_{pos_idx}_compound"] = float(msg["vader_compound"])
        feat[f"pos_{pos_idx}_has_hostility"] = float(msg["has_hostility"])
        feat[f"pos_{pos_idx}_direct_address"] = float(msg["direct_address_count"])

    # 5. Trajectory and change features
    if n >= 2:
        feat["sentiment_delta_last_first"] = float(compounds[-1] - compounds[0])
        feat["hostility_delta_last_first"] = float(hostilities[-1] - hostilities[0])
        feat["sentiment_delta_last_prev"] = float(compounds[-1] - compounds[-2])
        feat["hostility_delta_last_prev"] = float(hostilities[-1] - hostilities[-2])

        # Linear slope of compound sentiment across positions 0..(n-1)
        x_coords = np.arange(n, dtype=float)
        slope, _ = np.polyfit(x_coords, compounds, 1)
        feat["sentiment_slope"] = float(slope)

        # Escalation flag: sharp sentiment drop (>= 0.20) or hostility emergence
        is_escalated = (compounds[-1] <= compounds[0] - 0.20) or (hostilities[-1] > 0 and hostilities[0] == 0)
        feat["escalation_flag"] = 1.0 if is_escalated else 0.0
    else:
        # n = 1: zero observations for transition/trajectory
        feat["sentiment_delta_last_first"] = 0.0
        feat["hostility_delta_last_first"] = 0.0
        feat["sentiment_delta_last_prev"] = 0.0
        feat["hostility_delta_last_prev"] = 0.0
        feat["sentiment_slope"] = 0.0
        feat["escalation_flag"] = 0.0

    return feat


class SequenceFeatureExtractor:
    """
    Extracts structured numeric feature matrices for a batch of observed message sequences.
    Ensures exact column alignment and deterministic ordering across splits.
    """

    def __init__(self, mode: str, checkpoint_n_obs: int):
        if mode not in ("aggregate", "ordered"):
            raise ValueError(f"Mode must be 'aggregate' or 'ordered', got '{mode}'")
        self.mode = mode
        self.checkpoint_n_obs = checkpoint_n_obs
        self.analyzer = SentimentIntensityAnalyzer()

        if mode == "aggregate":
            self.feature_names = get_aggregate_feature_names()
        else:
            self.feature_names = get_ordered_feature_names(checkpoint_n_obs)

    def extract_features_single(self, messages: List[str]) -> np.ndarray:
        """Extract 1D feature array for a single list of messages."""
        if len(messages) != self.checkpoint_n_obs:
            raise ValueError(
                f"Expected exactly {self.checkpoint_n_obs} messages, got {len(messages)}"
            )
        signals = [extract_message_signals(m, self.analyzer) for m in messages]
        if self.mode == "aggregate":
            feat_dict = extract_aggregate_features(signals)
        else:
            feat_dict = extract_ordered_features(signals, self.checkpoint_n_obs)

        return np.array([feat_dict[fname] for fname in self.feature_names], dtype=float)

    def extract_feature_matrix(
        self,
        sequences: List[List[str]],
        utterance_cache: Optional[Dict[str, Dict[str, Any]]] = None,
        sequence_utterance_ids: Optional[List[List[str]]] = None,
    ) -> np.ndarray:
        """
        Extract 2D feature matrix (N, num_features) for a list of message sequences.
        Optionally uses pre-extracted utterance_cache if sequence_utterance_ids is provided.
        """
        n_samples = len(sequences)
        n_features = len(self.feature_names)
        matrix = np.zeros((n_samples, n_features), dtype=float)

        use_cache = (utterance_cache is not None) and (sequence_utterance_ids is not None)

        for i in range(n_samples):
            msgs = sequences[i]
            if len(msgs) != self.checkpoint_n_obs:
                raise ValueError(
                    f"Sample {i} has {len(msgs)} messages, expected {self.checkpoint_n_obs}"
                )

            if use_cache:
                uids = sequence_utterance_ids[i]
                signals = [utterance_cache[uid] for uid in uids]
            else:
                signals = [extract_message_signals(m, self.analyzer) for m in msgs]

            if self.mode == "aggregate":
                feat_dict = extract_aggregate_features(signals)
            else:
                feat_dict = extract_ordered_features(signals, self.checkpoint_n_obs)

            matrix[i] = [feat_dict[fname] for fname in self.feature_names]

        return matrix


# ---------------------------------------------------------------------------
# Majority Baseline Comparator
# ---------------------------------------------------------------------------

class MajorityBaseline:
    """
    Majority-class comparator determined by training-set prevalence with a fixed 0.5 decision rule.
    Distinct from the PriorBaseline soft-probability model.
    """

    def __init__(self):
        self.majority_class: int = 0
        self.prior_prob: float = 0.5

    def fit(self, X: Any, y: Any) -> "MajorityBaseline":
        y_arr = np.asarray(y, dtype=int)
        self.prior_prob = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.5
        self.majority_class = 1 if self.prior_prob >= 0.5 else 0
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        n = len(X)
        probs = np.zeros((n, 2), dtype=float)
        probs[:, self.majority_class] = 1.0
        return probs

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        n = len(X)
        return np.full(n, self.majority_class, dtype=int)


# ---------------------------------------------------------------------------
# Modeling Pipelines and Benchmark Orchestration
# ---------------------------------------------------------------------------

import scipy.sparse
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from conversation_guardian.ml.text_baseline import (
    build_tfidf_pipeline,
    tune_threshold_f1,
    compute_benchmark_metrics,
    PriorBaseline,
)


class DenseStateClassifier:
    """
    Standardized Logistic Regression classifier for dense sequence feature matrices.
    Fitted strictly on training features with StandardScaler to prevent scale distortion.
    """

    def __init__(self, random_state: int = 42, c_value: float = 1.0, max_iter: int = 1000):
        self.random_state = random_state
        self.c_value = c_value
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            C=self.c_value,
            max_iter=self.max_iter,
            solver="liblinear",
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DenseStateClassifier":
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


class HybridSequencePipeline:
    """
    Hybrid model combining word-and-character TF-IDF sparse text representation
    with standardized ordered-state sequence features via scipy.sparse.hstack.
    """

    def __init__(
        self,
        random_state: int = 42,
        c_value: float = 1.0,
        max_iter: int = 1000,
        min_df_word: int = 1,
        min_df_char: int = 1,
    ):
        self.random_state = random_state
        self.c_value = c_value
        self.max_iter = max_iter

        self.word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=min_df_word,
            max_features=10000,
            sublinear_tf=True,
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=min_df_char,
            max_features=15000,
            sublinear_tf=True,
        )
        self.tfidf_union = FeatureUnion([
            ("word", self.word_vec),
            ("char", self.char_vec),
        ])
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            C=self.c_value,
            max_iter=self.max_iter,
            solver="liblinear",
            random_state=self.random_state,
        )

    def fit(
        self,
        texts: List[str],
        ordered_features: np.ndarray,
        y: np.ndarray,
    ) -> "HybridSequencePipeline":
        """Fit TF-IDF vectorizers, scaler, and logistic regression strictly on training data."""
        X_tfidf = self.tfidf_union.fit_transform(texts)
        X_ord_scaled = self.scaler.fit_transform(ordered_features)
        X_combined = scipy.sparse.hstack([
            X_tfidf,
            scipy.sparse.csr_matrix(X_ord_scaled),
        ])
        self.clf.fit(X_combined, y)
        return self

    def predict_proba(
        self,
        texts: List[str],
        ordered_features: np.ndarray,
    ) -> np.ndarray:
        """Predict class probabilities for test/eval data using fitted components."""
        X_tfidf = self.tfidf_union.transform(texts)
        X_ord_scaled = self.scaler.transform(ordered_features)
        X_combined = scipy.sparse.hstack([
            X_tfidf,
            scipy.sparse.csr_matrix(X_ord_scaled),
        ])
        return self.clf.predict_proba(X_combined)

    def predict(
        self,
        texts: List[str],
        ordered_features: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        probs = self.predict_proba(texts, ordered_features)[:, 1]
        return (probs >= threshold).astype(int)


class SequenceCheckpointBenchmark:
    """
    Orchestrates the 6-way model comparison for a single observation checkpoint:
      1. Majority Baseline (Training majority class, 0.5 rule)
      2. Prior Baseline (Training prevalence probability)
      3. TF-IDF Text Baseline (Word + Char TF-IDF LogReg)
      4. Aggregate-State Model (Order-agnostic state/sentiment summaries)
      5. Ordered-State Model (Aggregate + transitions + trigrams + positions + trajectories)
      6. Hybrid Model (TF-IDF + Ordered State via sparse hstack)
    """

    MODEL_NAMES = [
        "MajorityBaseline",
        "PriorBaseline",
        "TFIDF_TextBaseline",
        "AggregateState_LogReg",
        "OrderedState_LogReg",
        "Hybrid_LogReg",
    ]

    def __init__(
        self,
        checkpoint_n_obs: int,
        horizon_h: int = 3,
        random_state: int = 42,
    ):
        self.checkpoint_n_obs = checkpoint_n_obs
        self.horizon_h = horizon_h
        self.random_state = random_state

        # Instantiate models
        self.majority_model = MajorityBaseline()
        self.prior_model = PriorBaseline()
        self.tfidf_model = build_tfidf_pipeline(random_state=self.random_state)
        self.aggregate_model = DenseStateClassifier(random_state=self.random_state)
        self.ordered_model = DenseStateClassifier(random_state=self.random_state)
        self.hybrid_model = HybridSequencePipeline(random_state=self.random_state)

        # Frozen decision thresholds (default to 0.5 fallback)
        self.thresholds: Dict[str, float] = {m: 0.5 for m in self.MODEL_NAMES}
        self.val_f1s: Dict[str, float] = {m: 0.0 for m in self.MODEL_NAMES}
        self.fallbacks: Dict[str, bool] = {m: False for m in self.MODEL_NAMES}

    def fit(
        self,
        texts_train: List[str],
        agg_features_train: np.ndarray,
        ord_features_train: np.ndarray,
        y_train: np.ndarray,
    ) -> "SequenceCheckpointBenchmark":
        """Fit all models strictly on eligible training examples."""
        self.majority_model.fit(texts_train, y_train)
        self.prior_model.fit(texts_train, y_train)
        self.tfidf_model.fit(texts_train, y_train)
        self.aggregate_model.fit(agg_features_train, y_train)
        self.ordered_model.fit(ord_features_train, y_train)
        self.hybrid_model.fit(texts_train, ord_features_train, y_train)
        return self

    def predict_probabilities(
        self,
        texts: List[str],
        agg_features: np.ndarray,
        ord_features: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute positive class predicted probabilities for all models (full precision)."""
        return {
            "MajorityBaseline": self.majority_model.predict_proba(texts)[:, 1],
            "PriorBaseline": self.prior_model.predict_proba(texts)[:, 1],
            "TFIDF_TextBaseline": self.tfidf_model.predict_proba(texts)[:, 1],
            "AggregateState_LogReg": self.aggregate_model.predict_proba(agg_features)[:, 1],
            "OrderedState_LogReg": self.ordered_model.predict_proba(ord_features)[:, 1],
            "Hybrid_LogReg": self.hybrid_model.predict_proba(texts, ord_features)[:, 1],
        }

    def tune_thresholds(
        self,
        texts_val: List[str],
        agg_features_val: np.ndarray,
        ord_features_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Select decision threshold maximizing F1 on validation predictions only.
        Uses higher-threshold tie-breaking and freezes thresholds for test evaluation.
        """
        val_probs = self.predict_probabilities(texts_val, agg_features_val, ord_features_val)
        summary: Dict[str, Dict[str, Any]] = {}

        for m_name in self.MODEL_NAMES:
            thresh, best_f1, fallback = tune_threshold_f1(y_val, val_probs[m_name], default_fallback=0.5)
            self.thresholds[m_name] = thresh
            self.val_f1s[m_name] = best_f1
            self.fallbacks[m_name] = fallback
            summary[m_name] = {
                "threshold": thresh,
                "val_f1": best_f1,
                "fallback_used": fallback,
            }

        return summary

    def evaluate_split(
        self,
        texts: List[str],
        agg_features: np.ndarray,
        ord_features: np.ndarray,
        y: np.ndarray,
        split_name: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate all models on a partition using frozen validation thresholds."""
        probs_dict = self.predict_probabilities(texts, agg_features, ord_features)
        results: Dict[str, Dict[str, Any]] = {}

        for m_name in self.MODEL_NAMES:
            metrics = compute_benchmark_metrics(
                y_true=y,
                y_probs=probs_dict[m_name],
                threshold=self.thresholds[m_name],
                model_name=m_name,
                split_name=split_name,
                checkpoint_n_obs=self.checkpoint_n_obs,
                horizon_h=self.horizon_h,
            )
            results[m_name] = metrics

        return results

