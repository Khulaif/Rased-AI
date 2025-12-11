"""
Rased (راصد) - System Tests
Comprehensive tests for the fraud detection system.
"""

import pytest
import numpy as np
from datetime import datetime
import sys
import os

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestConfiguration:
    """Test configuration module."""
    
    def test_config_initialization(self):
        """Test that config initializes properly."""
        from config.settings import config, RiskLevel
        
        assert config is not None
        assert config.risk.green_max == 30
        assert config.risk.yellow_max == 70
        assert config.model.sequence_length == 10
    
    def test_risk_level_classification(self):
        """Test risk level classification."""
        from config.settings import config, RiskLevel
        
        assert config.risk.get_risk_level(15) == RiskLevel.GREEN
        assert config.risk.get_risk_level(50) == RiskLevel.YELLOW
        assert config.risk.get_risk_level(85) == RiskLevel.RED


class TestVectorizer:
    """Test feature vectorization."""
    
    def test_vectorizer_initialization(self):
        """Test vectorizer initialization."""
        from data.processors.vectorizer import FeatureVectorizer
        
        vectorizer = FeatureVectorizer()
        assert vectorizer.total_dim > 0
        assert len(vectorizer.action_vocab) == 11
    
    def test_event_vectorization(self):
        """Test single event vectorization."""
        from data.processors.vectorizer import FeatureVectorizer
        
        vectorizer = FeatureVectorizer()
        
        event = {
            "action": "login",
            "hour_of_day": 10,
            "day_of_week": 1,
            "latitude": 24.7136,
            "longitude": 46.6753,
            "country_code": "SA",
            "device_type": "mobile",
            "os_family": "iOS",
            "time_since_last_action": 5.0,
        }
        
        vector = vectorizer.vectorize_event(event)
        
        assert vector.shape == (vectorizer.total_dim,)
        assert not np.isnan(vector).any()
    
    def test_sequence_vectorization(self):
        """Test sequence vectorization with padding."""
        from data.processors.vectorizer import FeatureVectorizer
        
        vectorizer = FeatureVectorizer()
        
        events = [
            {"action": "login", "hour_of_day": 10},
            {"action": "view_dashboard", "hour_of_day": 10},
        ]
        
        sequence = vectorizer.vectorize_sequence(events, sequence_length=10)
        
        assert sequence.shape == (10, vectorizer.total_dim)


class TestLSTMModel:
    """Test LSTM model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        from models.lstm_model import RasedLSTMModel
        
        model = RasedLSTMModel(input_dim=36)
        
        assert model.input_dim == 36
        assert model.num_actions == 11
        assert len(model.weights) > 0
    
    def test_model_forward_pass(self):
        """Test forward pass through model."""
        from models.lstm_model import RasedLSTMModel
        
        model = RasedLSTMModel(input_dim=36)
        sequence = np.random.randn(10, 36)
        
        action_probs, anomaly_score, hidden = model.forward(sequence, "test_user")
        
        assert action_probs.shape == (11,)
        assert np.isclose(action_probs.sum(), 1.0, atol=0.01)
        assert 0 <= anomaly_score <= 1
    
    def test_model_prediction(self):
        """Test prediction output."""
        from models.lstm_model import RasedLSTMModel
        
        model = RasedLSTMModel(input_dim=36)
        sequence = np.random.randn(10, 36)
        
        prediction = model.predict(sequence, "test_user", actual_action="login")
        
        assert prediction.predicted_action in model.idx_to_action.values()
        assert 0 <= prediction.anomaly_score <= 1
        assert prediction.prediction_confidence > 0


class TestDigitalTwin:
    """Test Digital Twin manager."""
    
    def test_twin_creation(self):
        """Test Digital Twin creation."""
        from models.digital_twin import DigitalTwinManager
        
        manager = DigitalTwinManager()
        twin = manager.get_or_create_twin("user_001")
        
        assert twin.user_id == "user_001"
        assert twin.embedding.shape == (64,)
    
    def test_twin_update(self):
        """Test Digital Twin update."""
        from models.digital_twin import DigitalTwinManager
        
        manager = DigitalTwinManager()
        
        event = {
            "hour_of_day": 10,
            "day_of_week": 1,
            "latitude": 24.7136,
            "longitude": 46.6753,
            "country_code": "SA",
            "device_hash": "abc123",
            "device_type": "mobile",
            "os_family": "iOS",
            "time_since_last_action": 5.0,
        }
        
        manager.update_twin("user_001", event, is_session_start=True)
        twin = manager.twins["user_001"]
        
        assert twin.total_sessions == 1
        assert 10 in twin.typical_hours
        assert "SA" in twin.typical_countries
    
    def test_deviation_calculation(self):
        """Test deviation calculation."""
        from models.digital_twin import DigitalTwinManager
        
        manager = DigitalTwinManager()
        
        # First, establish normal behavior
        normal_event = {
            "hour_of_day": 10,
            "latitude": 24.7136,
            "longitude": 46.6753,
            "country_code": "SA",
            "device_hash": "abc123",
            "device_type": "mobile",
            "os_family": "iOS",
            "time_since_last_action": 5.0,
        }
        manager.update_twin("user_001", normal_event, is_session_start=True)
        
        # Test deviation for anomalous event
        anomalous_event = {
            "hour_of_day": 3,
            "latitude": 55.7558,  # Moscow
            "longitude": 37.6173,
            "country_code": "RU",
            "device_hash": "xyz999",
            "device_type": "desktop",
            "os_family": "Windows",
            "time_since_last_action": 0.1,  # Very fast
        }
        
        deviations = manager.calculate_deviation("user_001", anomalous_event)
        
        assert deviations["location"] > 0  # Should detect location change
        assert deviations["device"] > 0    # Should detect device change
        assert deviations["navigation_speed"] > 0  # Should detect fast navigation


class TestInferenceEngine:
    """Test inference engine."""
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        from engine.inference_engine import InferenceEngine
        
        engine = InferenceEngine()
        
        assert engine.model is not None
        assert engine.vectorizer is not None
        assert engine.twin_manager is not None
    
    def test_inference(self):
        """Test complete inference."""
        from engine.inference_engine import InferenceEngine
        
        engine = InferenceEngine()
        
        event = {
            "event_id": "evt_001",
            "user_id": "user_001",
            "action": "login",
            "hour_of_day": 10,
            "day_of_week": 1,
            "latitude": 24.7136,
            "longitude": 46.6753,
            "country_code": "SA",
            "device_type": "mobile",
            "os_family": "iOS",
            "time_since_last_action": 0.0,
        }
        
        result = engine.infer(event)
        
        assert result.user_id == "user_001"
        assert 0 <= result.normalized_risk_score <= 100
        assert result.processing_time_ms > 0


class TestRiskScorer:
    """Test risk scorer."""
    
    def test_normal_scoring(self):
        """Test scoring for normal behavior."""
        from engine.risk_scorer import RiskScorer
        from config.settings import RiskLevel
        
        scorer = RiskScorer()
        
        assessment = scorer.calculate_risk(
            model_anomaly_score=0.1,
            twin_deviations={"location": 0.1, "device": 0.0, "timing": 0.1, "navigation_speed": 0.1},
            historical_profile={"average_risk_score": 10, "confirmed_fraud_count": 0, "account_age_days": 365},
            event={"action": "view_dashboard", "country_code": "SA", "is_vpn": False, "ip_reputation_score": 95},
        )
        
        assert assessment.normalized_score < 50
        assert assessment.risk_level in [RiskLevel.GREEN, RiskLevel.YELLOW]
    
    def test_suspicious_scoring(self):
        """Test scoring for suspicious behavior."""
        from engine.risk_scorer import RiskScorer
        from config.settings import RiskLevel
        
        scorer = RiskScorer()
        
        assessment = scorer.calculate_risk(
            model_anomaly_score=0.8,
            twin_deviations={"location": 0.9, "device": 0.8, "timing": 0.5, "navigation_speed": 0.9},
            historical_profile={"average_risk_score": 30, "confirmed_fraud_count": 0, "account_age_days": 30},
            event={
                "action": "confirm_transfer",
                "country_code": "RU",
                "is_vpn": True,
                "ip_reputation_score": 30,
                "time_since_last_action": 0.5,
                "transaction_amount": 100000,
            },
        )
        
        assert assessment.normalized_score > 50
        assert len(assessment.primary_risk_factors) > 0


class TestDecisionMatrix:
    """Test decision matrix."""
    
    def test_green_decision(self):
        """Test decision for low risk."""
        from engine.decision_matrix import DecisionMatrix, ResponseAction
        from config.settings import RiskLevel
        
        matrix = DecisionMatrix()
        
        response = matrix.decide(
            risk_score=20,
            risk_level=RiskLevel.GREEN,
            event={"user_id": "user_001", "action": "view_dashboard"},
        )
        
        assert response.action == ResponseAction.ALLOW
        assert not response.requires_user_action
    
    def test_yellow_decision(self):
        """Test decision for medium risk."""
        from engine.decision_matrix import DecisionMatrix, ResponseAction
        from config.settings import RiskLevel
        
        matrix = DecisionMatrix()
        
        response = matrix.decide(
            risk_score=55,
            risk_level=RiskLevel.YELLOW,
            event={"user_id": "user_001", "action": "confirm_transfer", "device_type": "mobile", "os_family": "iOS"},
        )
        
        assert response.action in [
            ResponseAction.CHALLENGE_OTP,
            ResponseAction.CHALLENGE_FACE,
            ResponseAction.CHALLENGE_NAFATH,
        ]
        assert response.requires_user_action
        assert response.challenge_method is not None
    
    def test_red_decision(self):
        """Test decision for high risk."""
        from engine.decision_matrix import DecisionMatrix, ResponseAction
        from config.settings import RiskLevel
        
        matrix = DecisionMatrix()
        
        response = matrix.decide(
            risk_score=95,
            risk_level=RiskLevel.RED,
            event={"user_id": "user_001", "action": "confirm_transfer"},
        )
        
        assert response.action == ResponseAction.BLOCK
        assert response.transaction_blocked


class TestLearningLoop:
    """Test learning loop."""
    
    def test_false_positive_feedback(self):
        """Test false positive feedback processing."""
        from feedback.learning_loop import LearningLoop, FeedbackType
        
        loop = LearningLoop()
        
        feedback = loop.process_feedback(
            user_id="user_001",
            event_id="evt_001",
            response_id="resp_001",
            feedback_type=FeedbackType.FALSE_POSITIVE,
            original_risk_score=55,
            original_decision="challenge",
            challenge_passed=True,
        )
        
        assert feedback.feedback_type == FeedbackType.FALSE_POSITIVE
        assert feedback.score_adjustment < 0  # Should reduce future scores
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        from feedback.learning_loop import LearningLoop, FeedbackType
        
        loop = LearningLoop()
        
        # Add some feedback
        loop.process_feedback("u1", "e1", "r1", FeedbackType.FALSE_POSITIVE, 50, "challenge")
        loop.process_feedback("u2", "e2", "r2", FeedbackType.TRUE_POSITIVE, 80, "block")
        
        metrics = loop.get_performance_metrics()
        
        assert metrics["total_feedback"] == 2
        assert metrics["false_positives"] == 1
        assert metrics["true_positives"] == 1


class TestMessageQueue:
    """Test message queue."""
    
    def test_in_memory_queue(self):
        """Test in-memory queue operations."""
        from queue.message_queue import InMemoryQueue, QueueMessage
        from datetime import datetime
        
        queue = InMemoryQueue()
        
        # Publish message
        msg = QueueMessage(
            message_id="msg_001",
            payload={"action": "test"},
            timestamp=datetime.now(),
        )
        success = queue.publish(msg)
        assert success
        assert queue.size() == 1
        
        # Consume message
        consumed = queue.consume(timeout=1.0)
        assert consumed is not None
        assert consumed.message_id == "msg_001"
        
        # Acknowledge
        queue.acknowledge("msg_001")
        stats = queue.get_stats()
        assert stats["acknowledged"] == 1


class TestSyntheticDataGenerator:
    """Test synthetic data generator."""
    
    def test_user_profile_generation(self):
        """Test user profile generation."""
        from data.generators.synthetic_data import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator(seed=42)
        profiles = generator.generate_user_profiles(num_users=10)
        
        assert len(profiles) == 10
        assert all(p.user_id for p in profiles)
        assert all(p.device_hash for p in profiles)
    
    def test_normal_session_generation(self):
        """Test normal session generation."""
        from data.generators.synthetic_data import SyntheticDataGenerator
        from datetime import datetime
        
        generator = SyntheticDataGenerator(seed=42)
        profiles = generator.generate_user_profiles(num_users=1)
        
        events = generator.generate_normal_session(profiles[0], datetime.now())
        
        assert len(events) >= 3
        assert events[0].action == "login"
        assert events[-1].action == "logout"
        assert not any(e.is_anomaly for e in events)
    
    def test_anomalous_session_generation(self):
        """Test anomalous session generation."""
        from data.generators.synthetic_data import SyntheticDataGenerator
        from datetime import datetime
        
        generator = SyntheticDataGenerator(seed=42)
        profiles = generator.generate_user_profiles(num_users=1)
        
        events = generator.generate_anomalous_session(
            profiles[0], 
            datetime.now(),
            anomaly_type="location_change"
        )
        
        assert len(events) >= 3
        assert all(e.is_anomaly for e in events)
        assert events[0].anomaly_type == "location_change"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
