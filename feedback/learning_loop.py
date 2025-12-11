"""
Rased (راصد) - Learning Loop
Real-time feedback integration for continuous model improvement.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import os
import sys

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import config


class FeedbackType(Enum):
    """Types of feedback signals."""
    FALSE_POSITIVE = "false_positive"    # User passed challenge
    TRUE_POSITIVE = "true_positive"      # Confirmed fraud
    FALSE_NEGATIVE = "false_negative"    # Missed fraud (reported later)
    TRUE_NEGATIVE = "true_negative"      # Correctly allowed
    USER_REPORT = "user_report"          # User-reported issue


@dataclass
class FeedbackEvent:
    """Feedback event for learning."""
    feedback_id: str
    user_id: str
    event_id: str
    response_id: str
    
    feedback_type: FeedbackType
    original_risk_score: int
    original_decision: str
    
    # Verification details
    challenge_method: Optional[str] = None
    challenge_passed: Optional[bool] = None
    
    # Context
    timestamp: datetime = None
    processing_latency_ms: float = 0.0
    
    # Adjustments made
    score_adjustment: float = 0.0
    embedding_updated: bool = False
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            "feedback_id": self.feedback_id,
            "user_id": self.user_id,
            "event_id": self.event_id,
            "response_id": self.response_id,
            "feedback_type": self.feedback_type.value,
            "original_risk_score": self.original_risk_score,
            "original_decision": self.original_decision,
            "challenge_method": self.challenge_method,
            "challenge_passed": self.challenge_passed,
            "timestamp": self.timestamp.isoformat(),
            "score_adjustment": self.score_adjustment,
            "embedding_updated": self.embedding_updated,
        }


class LearningLoop:
    """
    Real-time learning loop for Rased.
    
    Processes feedback signals to:
    1. Update user embeddings (Digital Twins)
    2. Adjust risk scoring weights
    3. Improve model predictions
    4. Track system performance
    
    Key principle: When a user passes a challenge (false positive),
    the system learns that this behavior was actually valid.
    """
    
    def __init__(
        self,
        twin_manager=None,
        model=None,
        storage_path: str = "feedback",
    ):
        """
        Initialize learning loop.
        
        Args:
            twin_manager: DigitalTwinManager instance
            model: RasedLSTMModel instance
            storage_path: Path to store feedback data
        """
        self.twin_manager = twin_manager
        self.model = model
        self.storage_path = storage_path
        
        # Learning rates
        self.false_positive_lr = config.feedback.false_positive_learning_rate
        self.true_positive_lr = config.feedback.true_positive_learning_rate
        
        # Feedback buffer
        self.feedback_buffer: List[FeedbackEvent] = []
        self.max_buffer_size = 10000
        
        # Performance tracking
        self.performance_metrics = {
            "total_feedback": 0,
            "false_positives": 0,
            "true_positives": 0,
            "false_negatives": 0,
            "true_negatives": 0,
            "user_reports": 0,
        }
        
        # Risk score adjustments per user
        self.user_adjustments: Dict[str, float] = {}
        
        # Load existing feedback
        self._load_feedback()
    
    def process_feedback(
        self,
        user_id: str,
        event_id: str,
        response_id: str,
        feedback_type: FeedbackType,
        original_risk_score: int,
        original_decision: str,
        challenge_method: Optional[str] = None,
        challenge_passed: Optional[bool] = None,
        event_data: Optional[Dict] = None,
    ) -> FeedbackEvent:
        """
        Process a feedback signal.
        
        Args:
            user_id: User who triggered the event
            event_id: Original event ID
            response_id: Security response ID
            feedback_type: Type of feedback
            original_risk_score: Original risk score
            original_decision: Original decision made
            challenge_method: Verification method used (if any)
            challenge_passed: Whether challenge was passed (if any)
            event_data: Original event data for learning
        
        Returns:
            FeedbackEvent with adjustments made
        """
        import uuid
        feedback_id = f"fb_{uuid.uuid4().hex[:12]}"
        
        feedback = FeedbackEvent(
            feedback_id=feedback_id,
            user_id=user_id,
            event_id=event_id,
            response_id=response_id,
            feedback_type=feedback_type,
            original_risk_score=original_risk_score,
            original_decision=original_decision,
            challenge_method=challenge_method,
            challenge_passed=challenge_passed,
        )
        
        # Update performance metrics
        self._update_metrics(feedback_type)
        
        # Apply learning based on feedback type
        if feedback_type == FeedbackType.FALSE_POSITIVE:
            feedback = self._handle_false_positive(feedback, event_data)
        elif feedback_type == FeedbackType.TRUE_POSITIVE:
            feedback = self._handle_true_positive(feedback, event_data)
        elif feedback_type == FeedbackType.FALSE_NEGATIVE:
            feedback = self._handle_false_negative(feedback, event_data)
        elif feedback_type == FeedbackType.USER_REPORT:
            feedback = self._handle_user_report(feedback, event_data)
        
        # Add to buffer
        self.feedback_buffer.append(feedback)
        if len(self.feedback_buffer) > self.max_buffer_size:
            self.feedback_buffer = self.feedback_buffer[-self.max_buffer_size:]
        
        # Periodic save
        if len(self.feedback_buffer) % 100 == 0:
            self._save_feedback()
        
        return feedback
    
    def _handle_false_positive(
        self,
        feedback: FeedbackEvent,
        event_data: Optional[Dict],
    ) -> FeedbackEvent:
        """
        Handle false positive: user passed the challenge.
        
        The system challenged the user, but they successfully verified.
        This means the behavior was actually legitimate.
        
        Actions:
        1. Update Digital Twin to accept this pattern
        2. Reduce future risk scores for similar behavior
        3. Adjust user-specific scoring threshold
        """
        user_id = feedback.user_id
        
        # Record in Digital Twin (if available)
        if self.twin_manager:
            self.twin_manager.record_false_positive(user_id)
            
            # Update embedding to be more tolerant of this behavior
            if event_data:
                # Create adjustment vector based on event features
                adjustment = self._create_embedding_adjustment(
                    event_data, 
                    direction="tolerant",
                    magnitude=self.false_positive_lr,
                )
                self.twin_manager.update_embedding(
                    user_id, 
                    adjustment,
                    learning_rate=self.false_positive_lr,
                )
                feedback.embedding_updated = True
        
        # Adjust user's risk threshold
        current_adjustment = self.user_adjustments.get(user_id, 0.0)
        self.user_adjustments[user_id] = current_adjustment - 0.05  # Lower threshold
        
        feedback.score_adjustment = -5  # Points to subtract from future similar events
        
        return feedback
    
    def _handle_true_positive(
        self,
        feedback: FeedbackEvent,
        event_data: Optional[Dict],
    ) -> FeedbackEvent:
        """
        Handle true positive: confirmed fraud was correctly detected.
        
        The system blocked or challenged, and it was confirmed as fraud.
        
        Actions:
        1. Reinforce the detection pattern
        2. Update Digital Twin with fraud marker
        3. Increase sensitivity for similar patterns
        """
        user_id = feedback.user_id
        
        # Record in Digital Twin
        if self.twin_manager:
            self.twin_manager.record_confirmed_fraud(user_id)
            
            # Update embedding to be more sensitive to this behavior
            if event_data:
                adjustment = self._create_embedding_adjustment(
                    event_data,
                    direction="sensitive",
                    magnitude=self.true_positive_lr,
                )
                self.twin_manager.update_embedding(
                    user_id,
                    adjustment,
                    learning_rate=self.true_positive_lr,
                )
                feedback.embedding_updated = True
        
        # Increase user's risk threshold
        current_adjustment = self.user_adjustments.get(user_id, 0.0)
        self.user_adjustments[user_id] = current_adjustment + 0.1
        
        feedback.score_adjustment = 10  # Points to add for similar future events
        
        return feedback
    
    def _handle_false_negative(
        self,
        feedback: FeedbackEvent,
        event_data: Optional[Dict],
    ) -> FeedbackEvent:
        """
        Handle false negative: fraud was missed and reported later.
        
        The system allowed a transaction that turned out to be fraud.
        This is the most critical feedback for improvement.
        
        Actions:
        1. Significantly increase sensitivity for this pattern
        2. Flag account for enhanced monitoring
        3. Trigger model retraining consideration
        """
        user_id = feedback.user_id
        
        # Record in Digital Twin with high weight
        if self.twin_manager:
            self.twin_manager.record_confirmed_fraud(user_id)
            
            if event_data:
                # Strong adjustment toward sensitivity
                adjustment = self._create_embedding_adjustment(
                    event_data,
                    direction="sensitive",
                    magnitude=self.true_positive_lr * 2,  # Double the learning
                )
                self.twin_manager.update_embedding(
                    user_id,
                    adjustment,
                    learning_rate=self.true_positive_lr * 2,
                )
                feedback.embedding_updated = True
        
        # Significant increase in risk threshold
        current_adjustment = self.user_adjustments.get(user_id, 0.0)
        self.user_adjustments[user_id] = current_adjustment + 0.2
        
        feedback.score_adjustment = 20  # Significant boost for future
        
        return feedback
    
    def _handle_user_report(
        self,
        feedback: FeedbackEvent,
        event_data: Optional[Dict],
    ) -> FeedbackEvent:
        """Handle user-reported issue (not directly fraud-related)."""
        # Log for manual review
        feedback.score_adjustment = 0
        return feedback
    
    def _create_embedding_adjustment(
        self,
        event_data: Dict,
        direction: str,
        magnitude: float,
    ) -> np.ndarray:
        """
        Create an embedding adjustment vector based on event.
        
        Args:
            event_data: Event that triggered the feedback
            direction: "tolerant" or "sensitive"
            magnitude: Adjustment magnitude
        
        Returns:
            Adjustment vector
        """
        embedding_dim = 64  # Default dimension
        
        # Create base adjustment
        adjustment = np.zeros(embedding_dim)
        
        # Use event features to determine adjustment direction
        # This is a simplified version - production would use gradients
        
        if direction == "tolerant":
            # Move toward accepting this pattern
            adjustment[:8] = -magnitude  # Reduce anomaly sensitivity
        else:
            # Move toward detecting this pattern
            adjustment[:8] = magnitude  # Increase anomaly sensitivity
        
        # Feature-specific adjustments
        if event_data.get("is_vpn"):
            adjustment[8:12] = magnitude if direction == "sensitive" else -magnitude
        
        if event_data.get("country_code") != "SA":
            adjustment[12:16] = magnitude if direction == "sensitive" else -magnitude
        
        # Normalize
        norm = np.linalg.norm(adjustment)
        if norm > 0:
            adjustment = adjustment / norm * magnitude
        
        return adjustment
    
    def _update_metrics(self, feedback_type: FeedbackType):
        """Update performance metrics."""
        self.performance_metrics["total_feedback"] += 1
        
        if feedback_type == FeedbackType.FALSE_POSITIVE:
            self.performance_metrics["false_positives"] += 1
        elif feedback_type == FeedbackType.TRUE_POSITIVE:
            self.performance_metrics["true_positives"] += 1
        elif feedback_type == FeedbackType.FALSE_NEGATIVE:
            self.performance_metrics["false_negatives"] += 1
        elif feedback_type == FeedbackType.TRUE_NEGATIVE:
            self.performance_metrics["true_negatives"] += 1
        elif feedback_type == FeedbackType.USER_REPORT:
            self.performance_metrics["user_reports"] += 1
    
    def get_user_adjustment(self, user_id: str) -> float:
        """Get risk score adjustment for a user."""
        return self.user_adjustments.get(user_id, 0.0)
    
    def get_performance_metrics(self) -> Dict:
        """Get system performance metrics."""
        metrics = dict(self.performance_metrics)
        
        total = metrics["total_feedback"]
        if total > 0:
            metrics["false_positive_rate"] = metrics["false_positives"] / total
            metrics["true_positive_rate"] = metrics["true_positives"] / total
            metrics["false_negative_rate"] = metrics["false_negatives"] / total
            
            # Precision and recall (if we have enough data)
            tp = metrics["true_positives"]
            fp = metrics["false_positives"]
            fn = metrics["false_negatives"]
            
            if tp + fp > 0:
                metrics["precision"] = tp / (tp + fp)
            else:
                metrics["precision"] = 0.0
            
            if tp + fn > 0:
                metrics["recall"] = tp / (tp + fn)
            else:
                metrics["recall"] = 0.0
            
            if metrics["precision"] + metrics["recall"] > 0:
                metrics["f1_score"] = (
                    2 * metrics["precision"] * metrics["recall"] /
                    (metrics["precision"] + metrics["recall"])
                )
            else:
                metrics["f1_score"] = 0.0
        
        return metrics
    
    def get_recent_feedback(
        self,
        user_id: Optional[str] = None,
        feedback_type: Optional[FeedbackType] = None,
        limit: int = 100,
    ) -> List[FeedbackEvent]:
        """Get recent feedback events with optional filtering."""
        events = self.feedback_buffer
        
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        
        if feedback_type:
            events = [e for e in events if e.feedback_type == feedback_type]
        
        return events[-limit:]
    
    def should_retrain(self) -> Tuple[bool, str]:
        """
        Determine if model retraining is needed.
        
        Returns:
            Tuple of (should_retrain, reason)
        """
        metrics = self.performance_metrics
        total = metrics["total_feedback"]
        
        if total < config.feedback.min_samples_for_retrain:
            return False, f"Insufficient samples ({total})"
        
        # Check false negative rate (missed fraud)
        fn_rate = metrics["false_negatives"] / total if total > 0 else 0
        if fn_rate > 0.05:  # More than 5% missed fraud
            return True, f"High false negative rate: {fn_rate:.1%}"
        
        # Check false positive rate (user friction)
        fp_rate = metrics["false_positives"] / total if total > 0 else 0
        if fp_rate > 0.30:  # More than 30% false positives
            return True, f"High false positive rate: {fp_rate:.1%}"
        
        return False, "Performance within acceptable range"
    
    def _save_feedback(self):
        """Save feedback to storage."""
        os.makedirs(self.storage_path, exist_ok=True)
        
        data = {
            "metrics": self.performance_metrics,
            "adjustments": self.user_adjustments,
            "recent_feedback": [f.to_dict() for f in self.feedback_buffer[-1000:]],
        }
        
        filepath = os.path.join(self.storage_path, "feedback_state.json")
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load_feedback(self):
        """Load feedback from storage."""
        filepath = os.path.join(self.storage_path, "feedback_state.json")
        
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
            
            self.performance_metrics = data.get("metrics", self.performance_metrics)
            self.user_adjustments = data.get("adjustments", {})


if __name__ == "__main__":
    # Demo learning loop
    loop = LearningLoop()
    
    # Simulate feedback events
    print("Processing feedback events:\n")
    
    # False positive - user passed challenge
    fb1 = loop.process_feedback(
        user_id="user_001",
        event_id="evt_001",
        response_id="resp_001",
        feedback_type=FeedbackType.FALSE_POSITIVE,
        original_risk_score=55,
        original_decision="challenge",
        challenge_method="otp_sms",
        challenge_passed=True,
    )
    print(f"1. False Positive (user passed challenge)")
    print(f"   User: {fb1.user_id}, Score adjustment: {fb1.score_adjustment}")
    
    # True positive - confirmed fraud
    fb2 = loop.process_feedback(
        user_id="user_002",
        event_id="evt_002",
        response_id="resp_002",
        feedback_type=FeedbackType.TRUE_POSITIVE,
        original_risk_score=85,
        original_decision="block",
    )
    print(f"2. True Positive (confirmed fraud)")
    print(f"   User: {fb2.user_id}, Score adjustment: {fb2.score_adjustment}")
    
    # False negative - missed fraud
    fb3 = loop.process_feedback(
        user_id="user_003",
        event_id="evt_003",
        response_id="resp_003",
        feedback_type=FeedbackType.FALSE_NEGATIVE,
        original_risk_score=25,
        original_decision="allow",
    )
    print(f"3. False Negative (missed fraud)")
    print(f"   User: {fb3.user_id}, Score adjustment: {fb3.score_adjustment}")
    
    # Performance metrics
    print("\nPerformance Metrics:")
    metrics = loop.get_performance_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")
    
    # Check retraining
    should_retrain, reason = loop.should_retrain()
    print(f"\nShould retrain model: {should_retrain}")
    print(f"Reason: {reason}")
