"""
Rased - Demo
مثال تطبيقي لكشف الاحتيال باستخدام LSTM
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from engine.inference import InferenceEngine


def print_result(title, result, status_char):
    """طباعة نتيجة التحليل"""
    print(f"\n{status_char} {title}")
    print("-" * 40)
    print(f"   Predicted: {result.prediction.predicted_action}")
    print(f"   Confidence: {result.prediction.confidence:.1%}")
    print(f"   Anomaly Score: {result.prediction.anomaly_score:.2f}")
    
    if result.recommendation == "allow":
        status = "[OK] Allow"
    elif result.recommendation == "verify":
        status = "[!] Verify"
    else:
        status = "[X] Block"
    
    print(f"   Recommendation: {status}")
    
    if result.prediction.reason:
        print(f"   Reason: {result.prediction.reason}")


def main():
    print("=" * 50)
    print("   Rased - AI Fraud Detection System")
    print("=" * 50)
    
    engine = InferenceEngine()
    
    # Scenario 1: Normal user
    print("\n[1] Scenario 1: Normal User")
    normal_events = [
        {"action": "open_app", "timestamp": 0, "duration": 1},
        {"action": "login", "timestamp": 3, "duration": 8},
        {"action": "browse_home", "timestamp": 15, "duration": 20},
        {"action": "search", "timestamp": 40, "duration": 5},
        {"action": "view_product", "timestamp": 50, "duration": 45},
        {"action": "add_to_cart", "timestamp": 100, "duration": 3},
    ]
    
    result = engine.analyze_sequence("user_normal", normal_events)
    print_result("Normal User Behavior", result, "[USER]")
    
    # Scenario 2: Bot
    print("\n[2] Scenario 2: Bot (Non-human Speed)")
    bot_events = [
        {"action": "open_app", "timestamp": 0, "duration": 0.02},
        {"action": "login", "timestamp": 0.05, "duration": 0.02},
        {"action": "checkout", "timestamp": 0.1, "duration": 0.02},
    ]
    
    result = engine.analyze_sequence("user_bot", bot_events)
    print_result("Bot Behavior", result, "[BOT]")
    
    # Scenario 3: Suspicious
    print("\n[3] Scenario 3: Suspicious (Skipping Steps)")
    suspicious_events = [
        {"action": "open_app", "timestamp": 0, "duration": 0.5},
        {"action": "login", "timestamp": 0.8, "duration": 1},
        {"action": "checkout", "timestamp": 2, "duration": 1},
    ]
    
    result = engine.analyze_sequence("user_suspicious", suspicious_events)
    print_result("Suspicious Behavior", result, "[?]")
    
    # Scenario 4: Real-time analysis
    print("\n" + "=" * 50)
    print("   [4] Real-time Analysis (Event by Event)")
    print("=" * 50)
    
    realtime_events = [
        ("open_app", 0, 1),
        ("login", 3, 5),
        ("browse_home", 10, 10),
        ("view_product", 25, 30),
        ("add_to_cart", 60, 2),
        ("checkout", 65, 10),
    ]
    
    for action, timestamp, duration in realtime_events:
        result = engine.process_event(
            user_id="user_realtime",
            action=action,
            timestamp=timestamp,
            duration=duration
        )
        
        if result.recommendation == "allow":
            status = "[OK]"
        elif result.recommendation == "verify":
            status = "[!]"
        else:
            status = "[X]"
        
        print(f"   {status} {action}: "
              f"anomaly={result.prediction.anomaly_score:.2f} "
              f"-> {result.recommendation}")
    
    # Stats
    print("\n" + "=" * 50)
    stats = engine.get_stats()
    print(f"Total Analyses: {stats['total_analyses']}")
    print(f"Anomalies Detected: {stats['model_stats']['anomalies_detected']}")
    print(f"Anomaly Rate: {stats['model_stats']['anomaly_rate']:.1%}")
    print("=" * 50)


if __name__ == "__main__":
    main()
