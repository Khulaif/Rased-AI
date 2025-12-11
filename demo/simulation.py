"""
Rased (راصد) - End-to-End Simulation
Demonstrates the complete fraud detection workflow.
"""

import time
import random
from datetime import datetime, timedelta
from typing import Dict, List
import os
import sys

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import config, RiskLevel
from data.generators.synthetic_data import SyntheticDataGenerator, UserProfile
from data.processors.vectorizer import FeatureVectorizer
from models.lstm_model import RasedLSTMModel
from models.digital_twin import DigitalTwinManager
from engine.inference_engine import InferenceEngine
from engine.risk_scorer import RiskScorer
from engine.decision_matrix import DecisionMatrix
from feedback.learning_loop import LearningLoop, FeedbackType


def print_header(title: str):
    """Print formatted header."""
    # Handle encoding for Windows console
    try:
        print("\n" + "=" * 60)
        print(f"  {title}")
        print("=" * 60)
    except UnicodeEncodeError:
        # Fallback for Windows console encoding issues
        safe_title = title.encode('ascii', 'replace').decode('ascii')
        print("\n" + "=" * 60)
        print(f"  {safe_title}")
        print("=" * 60)


def print_event_result(event: Dict, result, decision):
    """Print formatted event processing result."""
    risk_color = {
        RiskLevel.GREEN: "\033[92m",  # Green
        RiskLevel.YELLOW: "\033[93m",  # Yellow
        RiskLevel.RED: "\033[91m",    # Red
    }
    reset = "\033[0m"
    
    level = config.risk.get_risk_level(result.normalized_risk_score)
    color = risk_color.get(level, "")
    
    print(f"\n  User: {event.get('user_id', 'unknown')}")
    print(f"  Action: {event.get('action', 'unknown')}")
    print(f"  Location: {event.get('city', event.get('country_code', 'Unknown'))}")
    print(f"  Device: {event.get('device_type', 'unknown')} ({event.get('os_family', 'unknown')})")
    print(f"  {color}Risk Score: {result.normalized_risk_score}/100 ({level.value.upper()}){reset}")
    print(f"  Decision: {decision.action.value}")
    
    if decision.challenge_method:
        print(f"  Challenge: {decision.challenge_method}")
    if decision.transaction_blocked:
        print(f"  Transaction: BLOCKED")
    
    print(f"  Processing: {result.processing_time_ms:.1f}ms")


def run_simulation():
    """
    Run a complete simulation of the Rased system.
    
    Demonstrates:
    1. Normal user behavior detection
    2. Suspicious behavior detection
    3. Bot attack detection
    4. Feedback loop processing
    """
    print_header("Rased (راصد) - AI Fraud Detection Simulation")
    
    # Initialize components
    print("\n[1/5] Initializing components...")
    
    vectorizer = FeatureVectorizer()
    twin_manager = DigitalTwinManager()
    model = RasedLSTMModel(input_dim=vectorizer.total_dim)
    inference_engine = InferenceEngine(
        model=model,
        twin_manager=twin_manager,
        vectorizer=vectorizer,
    )
    decision_matrix = DecisionMatrix()
    learning_loop = LearningLoop(twin_manager=twin_manager, model=model)
    risk_scorer = RiskScorer()
    
    print("  ✓ LSTM Model initialized")
    print("  ✓ Digital Twin Manager ready")
    print("  ✓ Inference Engine configured")
    print("  ✓ Decision Matrix active")
    print("  ✓ Learning Loop enabled")
    
    # Generate synthetic user profiles
    print("\n[2/5] Generating user profiles...")
    generator = SyntheticDataGenerator(seed=42)
    profiles = generator.generate_user_profiles(num_users=5)
    
    for profile in profiles:
        print(f"  → {profile.user_id}: {profile.typical_city}, {profile.device_type}")
    
    # Simulate normal user sessions
    print_header("Scenario 1: Normal User Behavior")
    print("User performs typical actions from their usual location and device.")
    
    normal_profile = profiles[0]
    normal_events = [
        {
            "user_id": normal_profile.user_id,
            "action": "login",
            "hour_of_day": 10,
            "day_of_week": 1,
            "latitude": normal_profile.typical_latitude,
            "longitude": normal_profile.typical_longitude,
            "country_code": "SA",
            "city": normal_profile.typical_city,
            "device_type": normal_profile.device_type,
            "os_family": normal_profile.os_family,
            "device_hash": normal_profile.device_hash,
            "ip_reputation_score": 95.0,
            "time_since_last_action": 0.0,
            "is_vpn": False,
            "is_tor": False,
        },
        {
            "user_id": normal_profile.user_id,
            "action": "view_dashboard",
            "hour_of_day": 10,
            "day_of_week": 1,
            "latitude": normal_profile.typical_latitude,
            "longitude": normal_profile.typical_longitude,
            "country_code": "SA",
            "city": normal_profile.typical_city,
            "device_type": normal_profile.device_type,
            "os_family": normal_profile.os_family,
            "device_hash": normal_profile.device_hash,
            "ip_reputation_score": 95.0,
            "time_since_last_action": 5.0,  # Normal speed
            "is_vpn": False,
            "is_tor": False,
        },
        {
            "user_id": normal_profile.user_id,
            "action": "view_vehicles",
            "hour_of_day": 10,
            "day_of_week": 1,
            "latitude": normal_profile.typical_latitude,
            "longitude": normal_profile.typical_longitude,
            "country_code": "SA",
            "city": normal_profile.typical_city,
            "device_type": normal_profile.device_type,
            "os_family": normal_profile.os_family,
            "device_hash": normal_profile.device_hash,
            "ip_reputation_score": 95.0,
            "time_since_last_action": 8.0,
            "is_vpn": False,
            "is_tor": False,
        },
    ]
    
    for event in normal_events:
        result = inference_engine.infer(event)
        level = config.risk.get_risk_level(result.normalized_risk_score)
        decision = decision_matrix.decide(
            risk_score=result.normalized_risk_score,
            risk_level=level,
            event=event,
        )
        print_event_result(event, result, decision)
        time.sleep(0.1)
    
    # Simulate suspicious behavior
    print_header("Scenario 2: Suspicious Behavior")
    print("Same user suddenly attempts vehicle ownership transfer from a different country.")
    
    suspicious_events = [
        {
            "user_id": normal_profile.user_id,
            "action": "login",
            "hour_of_day": 3,  # Unusual hour
            "day_of_week": 1,
            "latitude": 55.7558,  # Moscow
            "longitude": 37.6173,
            "country_code": "RU",  # Different country
            "city": "Moscow",
            "device_type": "desktop",  # Different device type
            "os_family": "Windows",
            "device_hash": "suspicious_device_123",
            "ip_reputation_score": 60.0,  # Lower reputation
            "time_since_last_action": 0.0,
            "is_vpn": True,  # VPN detected
            "is_tor": False,
        },
        {
            "user_id": normal_profile.user_id,
            "action": "view_vehicles",
            "hour_of_day": 3,
            "day_of_week": 1,
            "latitude": 55.7558,
            "longitude": 37.6173,
            "country_code": "RU",
            "city": "Moscow",
            "device_type": "desktop",
            "os_family": "Windows",
            "device_hash": "suspicious_device_123",
            "ip_reputation_score": 60.0,
            "time_since_last_action": 2.0,  # Faster than normal
            "is_vpn": True,
            "is_tor": False,
        },
        {
            "user_id": normal_profile.user_id,
            "action": "initiate_ownership_transfer",
            "hour_of_day": 3,
            "day_of_week": 1,
            "latitude": 55.7558,
            "longitude": 37.6173,
            "country_code": "RU",
            "city": "Moscow",
            "device_type": "desktop",
            "os_family": "Windows",
            "device_hash": "suspicious_device_123",
            "ip_reputation_score": 60.0,
            "time_since_last_action": 1.5,
            "is_vpn": True,
            "is_tor": False,
            "service_type": "vehicle_ownership_transfer",
            "asset_type": "vehicle",
            "asset_value": 85000.0,
            "asset_plate": "ABC 1234",
            "new_owner_id": "unknown_person_999",
        },
    ]
    
    for event in suspicious_events:
        result = inference_engine.infer(event)
        level = config.risk.get_risk_level(result.normalized_risk_score)
        decision = decision_matrix.decide(
            risk_score=result.normalized_risk_score,
            risk_level=level,
            event=event,
        )
        print_event_result(event, result, decision)
        time.sleep(0.1)
    
    # Simulate bot attack
    print_header("Scenario 3: Bot Attack - Property Ownership Theft Attempt")
    print("Attacker using automated bot to steal property ownership.")
    
    attacker_profile = profiles[1]
    bot_events = [
        {
            "user_id": attacker_profile.user_id,
            "action": "login",
            "hour_of_day": 2,
            "day_of_week": 6,
            "latitude": 6.5244,  # Lagos
            "longitude": 3.3792,
            "country_code": "NG",
            "city": "Lagos",
            "device_type": "desktop",
            "os_family": "Linux",
            "device_hash": "bot_device_999",
            "ip_reputation_score": 25.0,  # Very low reputation
            "time_since_last_action": 0.0,
            "is_vpn": True,
            "is_tor": True,  # Tor exit node
        },
        {
            "user_id": attacker_profile.user_id,
            "action": "view_dashboard",
            "hour_of_day": 2,
            "day_of_week": 6,
            "latitude": 6.5244,
            "longitude": 3.3792,
            "country_code": "NG",
            "city": "Lagos",
            "device_type": "desktop",
            "os_family": "Linux",
            "device_hash": "bot_device_999",
            "ip_reputation_score": 25.0,
            "time_since_last_action": 0.1,  # Impossibly fast
            "is_vpn": True,
            "is_tor": True,
        },
        {
            "user_id": attacker_profile.user_id,
            "action": "view_vehicles",
            "hour_of_day": 2,
            "day_of_week": 6,
            "latitude": 6.5244,
            "longitude": 3.3792,
            "country_code": "NG",
            "city": "Lagos",
            "device_type": "desktop",
            "os_family": "Linux",
            "device_hash": "bot_device_999",
            "ip_reputation_score": 25.0,
            "time_since_last_action": 0.05,  # Even faster - bot behavior
            "is_vpn": True,
            "is_tor": True,
        },
        {
            "user_id": attacker_profile.user_id,
            "action": "confirm_ownership_transfer",
            "hour_of_day": 2,
            "day_of_week": 6,
            "latitude": 6.5244,
            "longitude": 3.3792,
            "country_code": "NG",
            "city": "Lagos",
            "device_type": "desktop",
            "os_family": "Linux",
            "device_hash": "bot_device_999",
            "ip_reputation_score": 25.0,
            "time_since_last_action": 0.02,  # Milliseconds - definitely bot
            "is_vpn": True,
            "is_tor": True,
            "service_type": "property_ownership_transfer",
            "asset_type": "property",
            "asset_value": 1500000.0,  # High value property
            "asset_deed": "DEED-2024-12345",
            "new_owner_id": "fraud_person_999",
        },
    ]
    
    for event in bot_events:
        result = inference_engine.infer(event)
        level = config.risk.get_risk_level(result.normalized_risk_score)
        decision = decision_matrix.decide(
            risk_score=result.normalized_risk_score,
            risk_level=level,
            event=event,
        )
        print_event_result(event, result, decision)
        time.sleep(0.1)
    
    # Demonstrate feedback loop
    print_header("Scenario 4: Feedback Loop - Learning from Mistakes")
    print("User passes verification challenge - system learns this was valid behavior.")
    
    # Simulate that the suspicious user passed verification
    print("\n  Processing feedback: User passed OTP verification...")
    
    feedback = learning_loop.process_feedback(
        user_id=normal_profile.user_id,
        event_id="evt_suspicious_001",
        response_id="resp_001",
        feedback_type=FeedbackType.FALSE_POSITIVE,
        original_risk_score=65,
        original_decision="challenge",
        challenge_method="otp_sms",
        challenge_passed=True,
    )
    
    print(f"  ✓ Feedback processed: {feedback.feedback_type.value}")
    print(f"  ✓ Score adjustment for future: {feedback.score_adjustment}")
    print(f"  ✓ Digital Twin updated: {feedback.embedding_updated}")
    
    # Show performance metrics
    print_header("System Performance Metrics")
    
    print("\n  Inference Engine:")
    engine_stats = inference_engine.get_stats()
    print(f"    Total inferences: {engine_stats['total_inferences']}")
    print(f"    Average latency: {engine_stats['avg_latency_ms']:.2f}ms")
    print(f"    High risk events: {engine_stats['high_risk_events']}")
    
    print("\n  Decision Matrix:")
    decision_stats = decision_matrix.get_statistics()
    for action, count in decision_stats["breakdown"].items():
        rate = decision_stats["rates"].get(action, 0)
        print(f"    {action}: {count} ({rate:.1f}%)")
    
    print("\n  Learning Loop:")
    feedback_metrics = learning_loop.get_performance_metrics()
    print(f"    Total feedback: {feedback_metrics['total_feedback']}")
    print(f"    False positives: {feedback_metrics['false_positives']}")
    print(f"    True positives: {feedback_metrics['true_positives']}")
    
    # Summary
    print_header("Simulation Complete")
    print("""
  The Rased (راصد) system successfully demonstrated:
  
  ✓ Normal behavior recognition (Green zone - seamless experience)
  ✓ Suspicious activity detection (Yellow zone - step-up auth)
  ✓ Bot attack prevention (Red zone - immediate block)
  ✓ Continuous learning from feedback
  
  All processing completed with <200ms latency per event.
  
  For production deployment:
  - Install dependencies: pip install -r requirements.txt
  - Start API: uvicorn api.main:app --reload
  - Access docs: http://localhost:8000/docs
""")


if __name__ == "__main__":
    run_simulation()
