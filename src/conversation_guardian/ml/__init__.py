"""Machine learning modeling and benchmarking module for ConversationGuardian."""

from .text_baseline import (
    build_tfidf_pipeline,
    tune_threshold_f1,
    compute_benchmark_metrics,
    PriorBaseline,
    CheckpointBenchmark,
)
from .sequence_features import (
    HOSTILITY_LEXICON,
    ORDERED_STATES,
    classify_message_state,
    extract_message_signals,
    extract_aggregate_features,
    extract_ordered_features,
    get_aggregate_feature_names,
    get_ordered_feature_names,
    SequenceFeatureExtractor,
    MajorityBaseline,
    DenseStateClassifier,
    HybridSequencePipeline,
    SequenceCheckpointBenchmark,
    select_deployable_model,
)
from .predictor import ConversationGuardianPredictor

__all__ = [
    "build_tfidf_pipeline",
    "tune_threshold_f1",
    "compute_benchmark_metrics",
    "PriorBaseline",
    "CheckpointBenchmark",
    "HOSTILITY_LEXICON",
    "ORDERED_STATES",
    "classify_message_state",
    "extract_message_signals",
    "extract_aggregate_features",
    "extract_ordered_features",
    "get_aggregate_feature_names",
    "get_ordered_feature_names",
    "SequenceFeatureExtractor",
    "MajorityBaseline",
    "DenseStateClassifier",
    "HybridSequencePipeline",
    "SequenceCheckpointBenchmark",
    "select_deployable_model",
    "ConversationGuardianPredictor",
]
