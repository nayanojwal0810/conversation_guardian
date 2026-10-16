"""
Data-access and repository operations for ConversationGuardian MongoDB collections.

Implements clean, parameterized operations across:
- conversations
- risk_assessments (immutable history)
- moderator_reviews (current state)
- review_events (append-only audit log)
- aggregation pipelines (alerts by checkpoint, reviews by status)
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import PyMongoError, DuplicateKeyError

from conversation_guardian.storage.schema import (
    COLL_CONVERSATIONS,
    COLL_RISK_ASSESSMENTS,
    COLL_MODERATOR_REVIEWS,
    COLL_REVIEW_EVENTS,
    ReviewStatus,
    ModeratorDecision,
    ReviewEventAction,
)


class RepositoryError(RuntimeError):
    """Raised when a storage repository operation fails."""
    pass


class DocumentNotFoundError(RepositoryError):
    """Raised when a requested document does not exist."""
    pass


class ValidationError(RepositoryError):
    """Raised when input parameters fail domain or schema validation."""
    pass


class ConversationGuardianRepository:
    """
    Data-access repository providing CRUD, historical queries, and aggregations.
    """

    def __init__(self, db: Database):
        """
        Initialize repository with an active MongoDB database instance.
        """
        self.db: Database = db
        self.conversations: Collection = db[COLL_CONVERSATIONS]
        self.risk_assessments: Collection = db[COLL_RISK_ASSESSMENTS]
        self.moderator_reviews: Collection = db[COLL_MODERATOR_REVIEWS]
        self.review_events: Collection = db[COLL_REVIEW_EVENTS]

    @staticmethod
    def _utc_now() -> datetime:
        """Return current datetime with UTC timezone."""
        return datetime.now(timezone.utc)

    # -------------------------------------------------------------------------
    # 1. Conversations
    # -------------------------------------------------------------------------

    def create_or_update_conversation(
        self,
        conversation_id: str,
        source: str = "cga_corpus",
        review_status: Optional[Union[str, ReviewStatus]] = None,
        latest_assessment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upsert a conversation document.
        
        If the conversation is new, sets created_at and default review_status (PENDING).
        Always updates updated_at and any explicitly provided fields.
        """
        if not conversation_id or not isinstance(conversation_id, str):
            raise ValidationError("conversation_id must be a non-empty string.")

        now = self._utc_now()
        set_on_insert: Dict[str, Any] = {
            "conversation_id": conversation_id,
            "source": source,
            "created_at": now,
        }

        set_fields: Dict[str, Any] = {"updated_at": now}

        if review_status is not None:
            status_val = review_status.value if isinstance(review_status, ReviewStatus) else str(review_status)
            if status_val not in [s.value for s in ReviewStatus]:
                raise ValidationError(f"Invalid review_status: '{status_val}'. Must be one of {[s.value for s in ReviewStatus]}")
            set_fields["review_status"] = status_val
        else:
            set_on_insert["review_status"] = ReviewStatus.PENDING.value

        if latest_assessment_id is not None:
            if not isinstance(latest_assessment_id, str) or not latest_assessment_id.strip():
                raise ValidationError("latest_assessment_id must be a non-empty string.")
            set_fields["latest_assessment_id"] = latest_assessment_id
        else:
            set_on_insert["latest_assessment_id"] = None

        update_doc: Dict[str, Any] = {"$set": set_fields, "$setOnInsert": set_on_insert}

        try:
            self.conversations.update_one(
                {"conversation_id": conversation_id},
                update_doc,
                upsert=True,
            )
            doc = self.conversations.find_one({"conversation_id": conversation_id}, {"_id": 0})
            if doc is None:
                raise DocumentNotFoundError(f"Failed to retrieve conversation after upsert: '{conversation_id}'")
            return doc
        except PyMongoError as exc:
            raise RepositoryError(f"Database error while upserting conversation '{conversation_id}': {str(exc)}") from exc

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve conversation record by conversation_id."""
        if not conversation_id or not isinstance(conversation_id, str):
            raise ValidationError("conversation_id must be a non-empty string.")
        try:
            return self.conversations.find_one({"conversation_id": conversation_id}, {"_id": 0})
        except PyMongoError as exc:
            raise RepositoryError(f"Error fetching conversation '{conversation_id}': {str(exc)}") from exc

    # -------------------------------------------------------------------------
    # 2. Risk Assessments (Historical, Immutable Records)
    # -------------------------------------------------------------------------

    def insert_risk_assessment(self, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Insert an immutable risk assessment record.
        
        Validates required fields, observation checkpoints, score boundaries,
        and ensures timestamps are in UTC.
        """
        required_fields = [
            "assessment_id",
            "conversation_id",
            "checkpoint_n_obs",
            "horizon_h",
            "observed_messages",
            "risk_score",
            "decision_threshold",
            "alert",
            "model_family",
            "model_version",
            "signals_summary",
        ]
        for field in required_fields:
            if field not in assessment_data:
                raise ValidationError(f"Missing required assessment field: '{field}'")

        # Validate checkpoint & horizon
        checkpoint = assessment_data["checkpoint_n_obs"]
        if checkpoint not in (1, 3, 5):
            raise ValidationError(f"checkpoint_n_obs must be strictly in (1, 3, 5), got {checkpoint}")

        if assessment_data["horizon_h"] != 3:
            raise ValidationError(f"horizon_h must be strictly 3, got {assessment_data['horizon_h']}")

        # Validate messages
        messages = assessment_data["observed_messages"]
        if not isinstance(messages, (list, tuple)) or len(messages) != checkpoint:
            raise ValidationError(
                f"observed_messages count ({len(messages) if isinstance(messages, (list, tuple)) else 'non-list'}) "
                f"must match checkpoint_n_obs ({checkpoint})"
            )
        for idx, m in enumerate(messages):
            if not isinstance(m, str) or not m.strip():
                raise ValidationError(f"Message at index {idx} must be a non-empty string.")

        # Validate risk_score & threshold
        risk_score = float(assessment_data["risk_score"])
        threshold = float(assessment_data["decision_threshold"])
        if not (0.0 <= risk_score <= 1.0):
            raise ValidationError(f"risk_score must be in [0.0, 1.0], got {risk_score}")
        if not (0.0 <= threshold <= 1.0):
            raise ValidationError(f"decision_threshold must be in [0.0, 1.0], got {threshold}")

        if not isinstance(assessment_data["alert"], bool):
            raise ValidationError(f"alert must be a boolean, got {type(assessment_data['alert']).__name__}")

        # Construct doc to insert
        doc = dict(assessment_data)
        doc["risk_score"] = risk_score
        doc["decision_threshold"] = threshold
        doc["alert"] = bool(assessment_data["alert"])
        if "created_at" not in doc or not isinstance(doc["created_at"], datetime):
            doc["created_at"] = self._utc_now()

        # Remove _id if passed to avoid collision
        doc.pop("_id", None)

        try:
            self.risk_assessments.insert_one(doc)
            return self.get_assessment(doc["assessment_id"])  # type: ignore[return-value]
        except DuplicateKeyError as exc:
            raise RepositoryError(f"Assessment with assessment_id '{doc['assessment_id']}' already exists.") from exc
        except PyMongoError as exc:
            raise RepositoryError(f"Database error inserting risk assessment '{doc['assessment_id']}': {str(exc)}") from exc

    def get_assessment(self, assessment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific risk assessment by its unique assessment_id."""
        if not assessment_id or not isinstance(assessment_id, str):
            raise ValidationError("assessment_id must be a non-empty string.")
        try:
            return self.risk_assessments.find_one({"assessment_id": assessment_id}, {"_id": 0})
        except PyMongoError as exc:
            raise RepositoryError(f"Error fetching assessment '{assessment_id}': {str(exc)}") from exc

    def get_latest_assessment(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the latest risk assessment for a conversation.
        
        First queries the conversation's latest_assessment_id pointer; falls back to
        the most recent assessment by created_at.
        """
        if not conversation_id or not isinstance(conversation_id, str):
            raise ValidationError("conversation_id must be a non-empty string.")

        conv = self.get_conversation(conversation_id)
        if conv and conv.get("latest_assessment_id"):
            asmt = self.get_assessment(conv["latest_assessment_id"])
            if asmt:
                return asmt

        # Fallback to sorting by created_at descending
        try:
            return self.risk_assessments.find_one(
                {"conversation_id": conversation_id},
                {"_id": 0},
                sort=[("created_at", -1)],
            )
        except PyMongoError as exc:
            raise RepositoryError(f"Error fetching latest assessment for '{conversation_id}': {str(exc)}") from exc

    def get_assessment_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve complete historical sequence of assessments for a conversation,
        strictly ordered in chronological ascending order by created_at.
        """
        if not conversation_id or not isinstance(conversation_id, str):
            raise ValidationError("conversation_id must be a non-empty string.")
        try:
            cursor = self.risk_assessments.find(
                {"conversation_id": conversation_id},
                {"_id": 0},
            ).sort("created_at", 1)
            return list(cursor)
        except PyMongoError as exc:
            raise RepositoryError(f"Error fetching assessment history for '{conversation_id}': {str(exc)}") from exc

    def list_alerts(
        self,
        checkpoint_n_obs: Optional[int] = None,
        model_family: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        List assessments where an alert was raised, filtered by checkpoint or model family.
        """
        query: Dict[str, Any] = {"alert": True}
        if checkpoint_n_obs is not None:
            if checkpoint_n_obs not in (1, 3, 5):
                raise ValidationError(f"Invalid checkpoint_n_obs: {checkpoint_n_obs}")
            query["checkpoint_n_obs"] = checkpoint_n_obs
        if model_family is not None:
            query["model_family"] = str(model_family)

        try:
            cursor = self.risk_assessments.find(query, {"_id": 0}).sort("created_at", -1).limit(limit)
            return list(cursor)
        except PyMongoError as exc:
            raise RepositoryError(f"Error listing alerts: {str(exc)}") from exc

    # -------------------------------------------------------------------------
    # 3. Moderator Reviews & Append-Only Review Events
    # -------------------------------------------------------------------------

    def create_or_update_moderator_review(
        self,
        assessment_id: str,
        conversation_id: str,
        review_status: Union[str, ReviewStatus],
        decision: Union[str, ModeratorDecision],
        reviewer_id: str,
        comment: str,
    ) -> Dict[str, Any]:
        """
        Create or update the current moderator review for an assessment,
        and atomically or sequentially append an audit trail event.
        
        Note on Atomicity:
            In standalone MongoDB environments (without a replica set), multi-document
            transactions are not supported. This method executes the review update
            followed immediately by the review event append. In replica set environments,
            a transactional session may be supplied.
        """
        if not assessment_id or not isinstance(assessment_id, str):
            raise ValidationError("assessment_id must be a non-empty string.")
        if not conversation_id or not isinstance(conversation_id, str):
            raise ValidationError("conversation_id must be a non-empty string.")
        if not reviewer_id or not isinstance(reviewer_id, str):
            raise ValidationError("reviewer_id must be a non-empty string.")

        status_val = review_status.value if isinstance(review_status, ReviewStatus) else str(review_status)
        if status_val not in [s.value for s in ReviewStatus]:
            raise ValidationError(f"Invalid review_status: '{status_val}'. Allowed: {[s.value for s in ReviewStatus]}")

        decision_val = decision.value if isinstance(decision, ModeratorDecision) else str(decision)
        if decision_val not in [d.value for d in ModeratorDecision]:
            raise ValidationError(f"Invalid decision: '{decision_val}'. Allowed: {[d.value for d in ModeratorDecision]}")

        # Check existing review to determine audit action
        existing_review = self.get_review(assessment_id)
        now = self._utc_now()

        if existing_review is None:
            action = ReviewEventAction.CREATE_REVIEW.value
            previous_status = None
        else:
            previous_status = existing_review.get("review_status")
            if previous_status != status_val:
                action = ReviewEventAction.CHANGE_STATUS.value
            else:
                action = ReviewEventAction.UPDATE_DECISION.value

        review_doc = {
            "assessment_id": assessment_id,
            "conversation_id": conversation_id,
            "review_status": status_val,
            "decision": decision_val,
            "reviewer_id": reviewer_id,
            "comment": comment or "",
            "updated_at": now,
        }

        try:
            # 1. Upsert current review
            self.moderator_reviews.update_one(
                {"assessment_id": assessment_id},
                {
                    "$set": review_doc,
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            )

            # 2. Update conversation's overall review status
            self.create_or_update_conversation(
                conversation_id=conversation_id,
                review_status=status_val,
            )

            # 3. Append to review_events audit log (immutable event)
            event_id = f"evt_{uuid.uuid4().hex[:12]}"
            self.append_review_event({
                "event_id": event_id,
                "assessment_id": assessment_id,
                "conversation_id": conversation_id,
                "action": action,
                "previous_status": previous_status,
                "new_status": status_val,
                "decision": decision_val,
                "actor_id": reviewer_id,
                "comment": comment or "",
                "event_time": now,
            })

            saved_review = self.get_review(assessment_id)
            if saved_review is None:
                raise DocumentNotFoundError(f"Failed to retrieve review after saving for assessment '{assessment_id}'")
            return saved_review
        except PyMongoError as exc:
            raise RepositoryError(f"Database error while saving review for assessment '{assessment_id}': {str(exc)}") from exc

    def get_review(self, assessment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve current moderator review for an assessment by assessment_id."""
        if not assessment_id or not isinstance(assessment_id, str):
            raise ValidationError("assessment_id must be a non-empty string.")
        try:
            return self.moderator_reviews.find_one({"assessment_id": assessment_id}, {"_id": 0})
        except PyMongoError as exc:
            raise RepositoryError(f"Error fetching review for assessment '{assessment_id}': {str(exc)}") from exc

    def append_review_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Append an audit event to the append-only review_events collection.
        
        Strictly inserts new records; update and delete operations are not supported.
        """
        required_fields = [
            "event_id",
            "assessment_id",
            "conversation_id",
            "action",
            "new_status",
            "decision",
            "actor_id",
            "comment",
        ]
        for field in required_fields:
            if field not in event_data:
                raise ValidationError(f"Missing required review event field: '{field}'")

        action_val = event_data["action"].value if isinstance(event_data["action"], ReviewEventAction) else str(event_data["action"])
        if action_val not in [a.value for a in ReviewEventAction]:
            raise ValidationError(f"Invalid review event action: '{action_val}'")

        status_val = event_data["new_status"].value if isinstance(event_data["new_status"], ReviewStatus) else str(event_data["new_status"])
        if status_val not in [s.value for s in ReviewStatus]:
            raise ValidationError(f"Invalid review event new_status: '{status_val}'")

        decision_val = event_data["decision"].value if isinstance(event_data["decision"], ModeratorDecision) else str(event_data["decision"])
        if decision_val not in [d.value for d in ModeratorDecision]:
            raise ValidationError(f"Invalid review event decision: '{decision_val}'")

        doc = dict(event_data)
        doc["action"] = action_val
        doc["new_status"] = status_val
        doc["decision"] = decision_val
        if "event_time" not in doc or not isinstance(doc["event_time"], datetime):
            doc["event_time"] = self._utc_now()

        doc.pop("_id", None)

        try:
            self.review_events.insert_one(doc)
            # Fetch back
            ret = self.review_events.find_one({"event_id": doc["event_id"]}, {"_id": 0})
            if ret is None:
                raise DocumentNotFoundError(f"Failed to read appended review event: '{doc['event_id']}'")
            return ret
        except DuplicateKeyError as exc:
            raise RepositoryError(f"Review event with event_id '{doc['event_id']}' already exists.") from exc
        except PyMongoError as exc:
            raise RepositoryError(f"Database error inserting review event '{doc['event_id']}': {str(exc)}") from exc

    def get_review_history(
        self,
        assessment_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chronological history of review audit events.
        
        Can filter by assessment_id or conversation_id. Always ordered ascending by event_time.
        """
        if not assessment_id and not conversation_id:
            raise ValidationError("Must provide at least assessment_id or conversation_id.")

        query: Dict[str, Any] = {}
        if assessment_id:
            query["assessment_id"] = assessment_id
        if conversation_id:
            query["conversation_id"] = conversation_id

        try:
            cursor = self.review_events.find(query, {"_id": 0}).sort("event_time", 1)
            return list(cursor)
        except PyMongoError as exc:
            raise RepositoryError(f"Error fetching review event history: {str(exc)}") from exc

    # -------------------------------------------------------------------------
    # 4. Aggregations
    # -------------------------------------------------------------------------

    def aggregate_alerts_by_checkpoint(self) -> List[Dict[str, Any]]:
        """
        Run aggregation pipeline counting raised alerts grouped by checkpoint_n_obs.
        
        Returns:
            List of dicts: [{'checkpoint_n_obs': 1, 'alert_count': 12}, ...]
        """
        pipeline = [
            {"$match": {"alert": True}},
            {"$group": {"_id": "$checkpoint_n_obs", "alert_count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
            {"$project": {"checkpoint_n_obs": "$_id", "alert_count": 1, "_id": 0}},
        ]
        try:
            return list(self.risk_assessments.aggregate(pipeline))
        except PyMongoError as exc:
            raise RepositoryError(f"Error aggregating alerts by checkpoint: {str(exc)}") from exc

    def aggregate_reviews_by_status(self) -> List[Dict[str, Any]]:
        """
        Run aggregation pipeline counting moderator reviews grouped by review_status.
        
        Returns:
            List of dicts: [{'review_status': 'PENDING', 'review_count': 5}, ...]
        """
        pipeline = [
            {"$group": {"_id": "$review_status", "review_count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
            {"$project": {"review_status": "$_id", "review_count": 1, "_id": 0}},
        ]
        try:
            return list(self.moderator_reviews.aggregate(pipeline))
        except PyMongoError as exc:
            raise RepositoryError(f"Error aggregating reviews by status: {str(exc)}") from exc

    def get_overview_metrics(self) -> Dict[str, int]:
        """
        Retrieve high-level summary KPIs for the moderator dashboard.
        """
        try:
            total_convs = self.conversations.count_documents({})
            total_asmts = self.risk_assessments.count_documents({})
            total_alerts = self.risk_assessments.count_documents({"alert": True})
            active_reviews = self.moderator_reviews.count_documents(
                {"review_status": {"$in": [ReviewStatus.PENDING.value, ReviewStatus.IN_REVIEW.value]}}
            )
            return {
                "total_conversations": total_convs,
                "total_assessments": total_asmts,
                "total_alerts": total_alerts,
                "active_reviews": active_reviews,
            }
        except PyMongoError as exc:
            raise RepositoryError(f"Error computing overview metrics: {str(exc)}") from exc

    def list_recent_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve recently updated conversations for dashboard search and selection.
        """
        try:
            cursor = self.conversations.find({}, {"_id": 0}).sort("updated_at", -1).limit(limit)
            return list(cursor)
        except PyMongoError as exc:
            raise RepositoryError(f"Error listing recent conversations: {str(exc)}") from exc

    def list_recent_assessments(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve recently created assessments for dashboard review selection.
        """
        try:
            cursor = self.risk_assessments.find({}, {"_id": 0}).sort("created_at", -1).limit(limit)
            return list(cursor)
        except PyMongoError as exc:
            raise RepositoryError(f"Error listing recent assessments: {str(exc)}") from exc
