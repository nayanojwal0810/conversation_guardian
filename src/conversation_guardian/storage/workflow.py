"""
Workflow service bridging the deployable predictor and MongoDB persistence layer.

Executes inference on observed dialogue turns, captures heuristic signal summaries,
and records immutable assessment history while updating conversation metadata.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path

from conversation_guardian.ml.predictor import ConversationGuardianPredictor
from conversation_guardian.storage.repository import (
    ConversationGuardianRepository,
    ValidationError,
    RepositoryError,
)


class StorageWorkflowService:
    """
    Coordinates prediction execution and persistence into MongoDB.
    """

    def __init__(
        self,
        repository: ConversationGuardianRepository,
        predictor: Optional[ConversationGuardianPredictor] = None,
        bundles_dir: Optional[Path] = None,
    ):
        """
        Initialize the workflow service with a repository and predictor.
        
        Args:
            repository: Active ConversationGuardianRepository instance.
            predictor: Optional pre-configured predictor instance (for injection/testing).
            bundles_dir: Path to directory containing predictor_n{1,3,5}.joblib.
        """
        self.repository = repository
        if predictor is not None:
            self.predictor = predictor
        else:
            try:
                self.predictor = ConversationGuardianPredictor(bundles_dir=bundles_dir)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "Selected predictor model bundles are missing. "
                    "Please generate the deployable bundles first by executing:\n"
                    "  python scripts/run_sequence_benchmark.py\n"
                    f"Details: {str(exc)}"
                ) from exc

    def assess_and_persist(
        self,
        conversation_id: str,
        messages: List[str],
        source: str = "cga_corpus",
    ) -> Dict[str, Any]:
        """
        Execute prediction on observed messages and record assessment into MongoDB.
        
        Args:
            conversation_id: Unique identifier for the conversation.
            messages: List of observed messages (strictly 1, 3, or 5).
            source: Source channel or dataset identifier.
            
        Returns:
            The stored risk assessment document.
        """
        if not conversation_id or not isinstance(conversation_id, str):
            raise ValidationError("conversation_id must be a non-empty string.")

        # 1. Execute prediction using validation-selected models
        prediction = self.predictor.predict(messages)

        # 2. Build stable assessment record
        assessment_id = f"asmt_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        assessment_doc = {
            "assessment_id": assessment_id,
            "conversation_id": conversation_id,
            "checkpoint_n_obs": prediction["checkpoint_n_obs"],
            "horizon_h": prediction["horizon_h"],
            "observed_messages": list(messages),
            "risk_score": float(prediction["risk_score"]),
            "decision_threshold": float(prediction["threshold"]),
            "alert": bool(prediction["alert"]),
            "model_family": str(prediction["model_family"]),
            "model_version": str(prediction["model_version"]),
            "signals_summary": prediction["signals_summary"],
            "created_at": now,
        }

        # 3. Insert immutable risk assessment
        saved_assessment = self.repository.insert_risk_assessment(assessment_doc)

        # 4. Upsert conversation with latest_assessment_id reference
        self.repository.create_or_update_conversation(
            conversation_id=conversation_id,
            source=source,
            latest_assessment_id=assessment_id,
        )

        return saved_assessment
