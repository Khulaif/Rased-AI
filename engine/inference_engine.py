"""
Rased (راصد) - Inference Engine
Real-time prediction service combining LSTM model and Digital Twin analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import time
import os
import sys

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import config
from models.lstm_model import RasedLSTMModel, ModelPrediction
from models.digital_twin import DigitalTwinManager
from data.processors.vectorizer import FeatureVectorizer


@dataclass
class InferenceResult:
    """Complete inference result for a user event."""
    user_id: str
    event_id: str
    timestamp: datetime
    
    # Model predictions
    predicted_action: str
    action_confidence: float
    model_anomaly_score: float
    
    # Digital Twin analysis
    twin_deviation_score: float
    deviation_breakdown: Dict[str, float]
    
    # Composite scores
    raw_risk_score: float
    normalized_risk_score: int  # 0-100
    
    # Context
    is_new_user: bool
    user_history_length: int
    processing_time_ms: float
    
    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "predicted_action": self.predicted_action,
            "action_confidence": self.action_confidence,
            "model_anomaly_score": self.model_anomaly_score,
            "twin_deviation_score": self.twin_deviation_score,
            "deviation_breakdown": self.deviation_breakdown,
            "raw_risk_score": self.raw_risk_score,
            "normalized_risk_score": self.normalized_risk_score,
            "is_new_user": self.is_new_user,
            "user_history_length": self.user_history_length,
            "processing_time_ms": self.processing_time_ms,
        }


class InferenceEngine:
    """
    Real-time inference engine for Rased.
    
    Combines:
    1. LSTM model predictions (sequence-based)
    2. Digital Twin deviation analysis (behavioral patterns)
    3. Contextual risk factors
    
    To produce a unified risk assessment.
    """
    
    def __init__(
        self,
        model: Optional[RasedLSTMModel] = None,
        twin_manager: Optional[DigitalTwinManager] = None,
        vectorizer: Optional[FeatureVectorizer] = None,
        model_path: Optional[str] = None,
    ):
        """
        Initialize inference engine.
        
        Args:
            model: Pre-loaded LSTM model (creates new if None)
            twin_manager: Digital Twin manager (creates new if None)
            vectorizer: Feature vectorizer (creates new if None)
            model_path: Path to load saved model from
        """
        # Initialize vectorizer
        self.vectorizer = vectorizer or FeatureVectorizer()
        
        # Initialize model
        if model:
            self.model = model
        elif model_path and os.path.exists(model_path):
            self.model = RasedLSTMModel(input_dim=self.vectorizer.total_dim)
            self.model.load(model_path)
        else:
            self.model = RasedLSTMModel(input_dim=self.vectorizer.total_dim)
        
        # Initialize Digital Twin manager
        self.twin_manager = twin_manager or DigitalTwinManager()
        
        # User event history for sequence building
        self.user_history: Dict[str, List[Dict]] = {}
        self.max_history_length = 100  # Per user
        
        # Risk weight configuration
        self.risk_weights = {
            "model_prediction": config.risk.weights["model_prediction_error"],
            "twin_deviation": config.risk.weights["contextual_anomaly"],
            "historical": config.risk.weights["historical_risk"],
        }
        
        # Statistics
        self.stats = {
            "total_inferences": 0,
            "total_latency_ms": 0.0,
            "high_risk_events": 0,
        }
    
    def infer(self, event: Dict) -> InferenceResult:
        """
        Perform inference on a user event.
        
        Args:
            event: Event data dictionary
        
        Returns:
            InferenceResult with risk assessment
        """
        start_time = time.perf_counter()
        
        user_id = event.get("user_id", "unknown")
        event_id = event.get("event_id", f"evt_{datetime.now().timestamp()}")
        action = event.get("action", "unknown")
        
        # === 1. Build Event Sequence ===
        sequence = self._build_sequence(user_id, event)
        
        # === 2. LSTM Model Prediction ===
        prediction = self.model.predict(
            sequence,
            user_id=user_id,
            actual_action=action,
        )
        
        # === 3. Digital Twin Analysis ===
        deviations = self.twin_manager.calculate_deviation(user_id, event)
        twin_deviation = deviations.get("composite", 0.0)
        
        # === 4. Historical Risk Factor ===
        profile = self.twin_manager.get_user_risk_profile(user_id)
        historical_risk = profile.get("average_risk_score", 0.0) / 100.0
        
        # Adjust for confirmed fraud history
        if profile.get("confirmed_fraud_count", 0) > 0:
            historical_risk = min(1.0, historical_risk + 0.3)
        
        # === 5. Composite Risk Score ===
        raw_risk = (
            self.risk_weights["model_prediction"] * prediction.anomaly_score +
            self.risk_weights["twin_deviation"] * twin_deviation +
            self.risk_weights["historical"] * historical_risk
        )
        
        # Apply contextual adjustments
        raw_risk = self._apply_context_adjustments(raw_risk, event, profile)
        
        # Normalize to 0-100
        normalized_risk = int(min(100, max(0, raw_risk * 100)))
        
        # === 6. Update State ===
        # Update user history
        self._update_history(user_id, event)
        
        # Update Digital Twin
        is_session_start = action == "login"
        self.twin_manager.update_twin(user_id, event, is_session_start)
        
        # Record risk score
        self.twin_manager.record_risk_score(user_id, normalized_risk)
        
        # Calculate processing time
        processing_time = (time.perf_counter() - start_time) * 1000
        
        # Update stats
        self.stats["total_inferences"] += 1
        self.stats["total_latency_ms"] += processing_time
        if normalized_risk >= 70:
            self.stats["high_risk_events"] += 1
        
        # Build result
        result = InferenceResult(
            user_id=user_id,
            event_id=event_id,
            timestamp=datetime.now(),
            predicted_action=prediction.predicted_action,
            action_confidence=prediction.prediction_confidence,
            model_anomaly_score=prediction.anomaly_score,
            twin_deviation_score=twin_deviation,
            deviation_breakdown=deviations,
            raw_risk_score=raw_risk,
            normalized_risk_score=normalized_risk,
            is_new_user=(profile["total_sessions"] < 5),
            user_history_length=len(self.user_history.get(user_id, [])),
            processing_time_ms=processing_time,
        )
        
        return result
    
    def _build_sequence(self, user_id: str, current_event: Dict) -> np.ndarray:
        """Build event sequence for LSTM input."""
        # Get user's past events
        history = self.user_history.get(user_id, [])
        
        # Combine with current event
        events = history + [current_event]
        
        # Take last N events
        sequence_length = config.model.sequence_length
        recent_events = events[-sequence_length:]
        
        # Vectorize
        sequence = self.vectorizer.vectorize_sequence(
            recent_events,
            sequence_length=sequence_length,
        )
        
        return sequence
    
    def _update_history(self, user_id: str, event: Dict):
        """Update user's event history."""
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        
        self.user_history[user_id].append(event)
        
        # Trim to max length
        if len(self.user_history[user_id]) > self.max_history_length:
            self.user_history[user_id] = self.user_history[user_id][-self.max_history_length:]
    
    def _apply_context_adjustments(
        self,
        base_risk: float,
        event: Dict,
        profile: Dict,
    ) -> float:
        """Apply contextual adjustments to risk score."""
        adjusted_risk = base_risk
        
        # === High-Value Transaction Boost ===
        amount = event.get("transaction_amount", 0.0)
        if amount > 50000:  # High value
            adjusted_risk *= 1.3
        elif amount > 10000:  # Medium value
            adjusted_risk *= 1.1
        
        # === New Beneficiary Boost ===
        beneficiary = event.get("beneficiary_id", "")
        if beneficiary and "new" in str(beneficiary).lower():
            adjusted_risk *= 1.2
        
        # === Rapid Transaction Boost ===
        action = event.get("action", "")
        time_since_last = event.get("time_since_last_action", 30.0)
        
        if action == "confirm_ownership_transfer" and time_since_last < 2.0:
            # Very fast confirmation - suspicious
            adjusted_risk *= 1.4
        
        # === VPN/Tor Boost ===
        if event.get("is_vpn") or event.get("is_tor"):
            adjusted_risk *= 1.25
        
        # === Low IP Reputation Boost ===
        ip_reputation = event.get("ip_reputation_score", 100.0)
        if ip_reputation < 50:
            adjusted_risk *= 1.3
        elif ip_reputation < 75:
            adjusted_risk *= 1.1
        
        # === New User Discount ===
        # Be more lenient with new users (less data to judge)
        if profile["total_sessions"] < 5:
            adjusted_risk *= 0.8
        
        # === Known Good User Discount ===
        # Users with history of false positives get slight discount
        false_positive_rate = profile.get("false_positive_rate", 0.0)
        if false_positive_rate > 0.3 and profile["confirmed_fraud_count"] == 0:
            adjusted_risk *= 0.9
        
        return min(1.0, adjusted_risk)
    
    def batch_infer(self, events: List[Dict]) -> List[InferenceResult]:
        """
        Perform inference on multiple events.
        
        Args:
            events: List of event dictionaries
        
        Returns:
            List of InferenceResults
        """
        return [self.infer(event) for event in events]
    
    def get_user_risk_summary(self, user_id: str) -> Dict:
        """Get risk summary for a user."""
        profile = self.twin_manager.get_user_risk_profile(user_id)
        twin = self.twin_manager.get_or_create_twin(user_id)
        
        return {
            **profile,
            "typical_hours": twin.typical_hours,
            "typical_countries": twin.typical_countries,
            "known_devices": len(twin.known_devices),
            "avg_navigation_speed": twin.avg_navigation_speed,
        }
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        avg_latency = (
            self.stats["total_latency_ms"] / max(1, self.stats["total_inferences"])
        )
        
        return {
            **self.stats,
            "avg_latency_ms": avg_latency,
            "high_risk_rate": (
                self.stats["high_risk_events"] / max(1, self.stats["total_inferences"])
            ),
            "active_users": len(self.user_history),
            "total_twins": len(self.twin_manager.twins),
        }
    
    def save_state(self, path: str):
        """Save engine state to directory."""
        os.makedirs(path, exist_ok=True)
        
        # Save model
        self.model.save(os.path.join(path, "model"))
        
        # Save vectorizer
        self.vectorizer.save(os.path.join(path, "vectorizer.json"))
        
        # Save Digital Twins
        self.twin_manager.storage_path = os.path.join(path, "twins")
        self.twin_manager.save_twins()
        
        print(f"Engine state saved to {path}")
    
    def load_state(self, path: str):
        """Load engine state from directory."""
        # Load model
        model_path = os.path.join(path, "model")
        if os.path.exists(model_path):
            self.model.load(model_path)
        
        # Load vectorizer
        vectorizer_path = os.path.join(path, "vectorizer.json")
        if os.path.exists(vectorizer_path):
            self.vectorizer.load(vectorizer_path)
        
        # Load Digital Twins
        twins_path = os.path.join(path, "twins")
        if os.path.exists(twins_path):
            self.twin_manager.storage_path = twins_path
            self.twin_manager._load_twins()
        
        print(f"Engine state loaded from {path}")


if __name__ == "__main__":
    # Demo inference
    engine = InferenceEngine()
    
    # Simulate some events
    events = [
        {
            "event_id": "evt_001",
            "user_id": "user_001",
            "action": "login",
            "hour_of_day": 10,
            "day_of_week": 1,
            "latitude": 24.7136,
            "longitude": 46.6753,
            "country_code": "SA",
            "device_hash": "abc123",
            "device_type": "mobile",
            "os_family": "iOS",
            "ip_reputation_score": 95.0,
            "time_since_last_action": 0.0,
            "is_vpn": False,
        },
        {
            "event_id": "evt_002",
            "user_id": "user_001",
            "action": "view_dashboard",
            "hour_of_day": 10,
            "day_of_week": 1,
            "latitude": 24.7136,
            "longitude": 46.6753,
            "country_code": "SA",
            "device_hash": "abc123",
            "device_type": "mobile",
            "os_family": "iOS",
            "ip_reputation_score": 95.0,
            "time_since_last_action": 5.0,
            "is_vpn": False,
        },
        {
            "event_id": "evt_003",
            "user_id": "user_001",
            "action": "confirm_transfer",
            "hour_of_day": 10,
            "day_of_week": 1,
            "latitude": 55.7558,  # Moscow!
            "longitude": 37.6173,
            "country_code": "RU",  # Foreign country
            "device_hash": "xyz999",  # Different device
            "device_type": "desktop",
            "os_family": "Windows",
            "ip_reputation_score": 40.0,
            "time_since_last_action": 0.3,  # Very fast
            "is_vpn": True,
            "transaction_amount": 100000.0,
            "beneficiary_id": "new_beneficiary_999",
        },
    ]
    
    print("Running inference on sample events:\n")
    
    for event in events:
        result = engine.infer(event)
        
        print(f"Event: {result.event_id}")
        print(f"  Action: {event['action']}")
        print(f"  Country: {event['country_code']}")
        print(f"  Risk Score: {result.normalized_risk_score}/100")
        print(f"  Model Anomaly: {result.model_anomaly_score:.2%}")
        print(f"  Twin Deviation: {result.twin_deviation_score:.2%}")
        print(f"  Processing Time: {result.processing_time_ms:.1f}ms")
        print()
    
    # Print stats
    print("Engine Stats:")
    for key, value in engine.get_stats().items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
