"""
Rased - Smart Twin Demo
مثال تطبيقي للتوأم الذكي

يوضح كيفية:
1. بناء ملف سلوكي للمستخدم
2. كشف سلوك مختلف (شخص آخر يستخدم الحساب)
3. التكامل مع محرك الاستدلال
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.twins.smart_twin import SmartTwin
from engine.inference import InferenceEngine


def print_divider(title=""):
    print("\n" + "=" * 55)
    if title:
        print(f"   {title}")
        print("=" * 55)


def demo_smart_twin_standalone():
    """عرض Smart Twin بشكل مستقل"""
    print_divider("🧬 Smart Twin - Standalone Demo")
    
    twin = SmartTwin()
    
    # ===== بناء الملف السلوكي =====
    print("\n📝 Step 1: Building User Profile from History")
    print("-" * 45)
    
    # تاريخ سلوك المستخدم الطبيعي
    user_history = [
        {"action": "open_app", "time_delta": 0, "timestamp": 1000},
        {"action": "login", "time_delta": 3.5, "timestamp": 1003.5},
        {"action": "browse_home", "time_delta": 5.0, "timestamp": 1008.5},
        {"action": "search", "time_delta": 12.0, "timestamp": 1020.5},
        {"action": "view_product", "time_delta": 8.0, "timestamp": 1028.5},
        {"action": "add_to_cart", "time_delta": 35.0, "timestamp": 1063.5},
        {"action": "browse_home", "time_delta": 4.0, "timestamp": 1067.5},
        {"action": "search", "time_delta": 10.0, "timestamp": 1077.5},
        {"action": "view_product", "time_delta": 7.0, "timestamp": 1084.5},
        {"action": "add_to_cart", "time_delta": 25.0, "timestamp": 1109.5},
        {"action": "checkout", "time_delta": 60.0, "timestamp": 1169.5},
    ]
    
    profile = twin.build_profile("user_khulaif", user_history)
    
    print(f"   ✅ Profile created for: {profile.user_id}")
    print(f"   📊 Total events analyzed: {profile.total_events}")
    print(f"   ⭐ Favorite actions: {', '.join(profile.favorite_actions)}")
    print(f"   ⏱️  Average speed: {profile.speed_mean:.1f}s between actions")
    
    # ===== اختبار 1: السلوك الطبيعي =====
    print("\n✅ Step 2: Testing NORMAL Behavior")
    print("-" * 45)
    
    normal_behavior = [
        {"action": "open_app", "time_delta": 0},
        {"action": "login", "time_delta": 4.0},
        {"action": "browse_home", "time_delta": 6.0},
        {"action": "search", "time_delta": 11.0},
        {"action": "view_product", "time_delta": 9.0},
        {"action": "add_to_cart", "time_delta": 28.0},
    ]
    
    result = twin.compare_behavior("user_khulaif", normal_behavior)
    
    print(f"   Deviation Score: {result['deviation_score']:.2f}")
    print(f"   Risk Level: {result['risk_level'].upper()}")
    print(f"   Is Suspicious: {'❌ YES' if result['is_suspicious'] else '✅ NO'}")
    print(f"   Analysis: {result['reason']}")
    
    # ===== اختبار 2: شخص آخر يستخدم الحساب =====
    print("\n🚨 Step 3: Testing SUSPICIOUS Behavior (Different Person?)")
    print("-" * 45)
    
    suspicious_behavior = [
        {"action": "open_app", "time_delta": 0},
        {"action": "login", "time_delta": 0.3},       # سريع جداً!
        {"action": "change_settings", "time_delta": 0.2},  # إجراء غير معتاد
        {"action": "view_profile", "time_delta": 0.15},    # إجراء غير معتاد
        {"action": "logout", "time_delta": 0.1},      # سريع جداً!
    ]
    
    result = twin.compare_behavior("user_khulaif", suspicious_behavior)
    
    print(f"   Deviation Score: {result['deviation_score']:.2f}")
    print(f"   Risk Level: {result['risk_level'].upper()}")
    print(f"   Is Suspicious: {'❌ YES' if result['is_suspicious'] else '✅ NO'}")
    print(f"   Analysis: {result['reason']}")
    
    if result.get('details'):
        print("\n   📋 Detailed Analysis:")
        for key, value in result['details'].items():
            print(f"      • {key}: {value:.2f}")
    
    # ===== اختبار 3: بوت =====
    print("\n🤖 Step 4: Testing BOT Behavior")
    print("-" * 45)
    
    bot_behavior = [
        {"action": "open_app", "time_delta": 0},
        {"action": "login", "time_delta": 0.05},
        {"action": "checkout", "time_delta": 0.03},
    ]
    
    result = twin.compare_behavior("user_khulaif", bot_behavior)
    
    print(f"   Deviation Score: {result['deviation_score']:.2f}")
    print(f"   Risk Level: {result['risk_level'].upper()}")
    print(f"   Is Suspicious: {'❌ YES' if result['is_suspicious'] else '✅ NO'}")
    print(f"   Analysis: {result['reason']}")
    
    return twin


def demo_integrated_analysis():
    """عرض التكامل مع محرك الاستدلال"""
    print_divider("🔗 Integrated Analysis Demo")
    
    twin = SmartTwin()
    engine = InferenceEngine()
    
    # بناء ملف سلوكي
    print("\n📝 Building Smart Twin profile...")
    
    history = [
        {"action": "open_app", "time_delta": 0, "timestamp": 0},
        {"action": "login", "time_delta": 3, "timestamp": 3},
        {"action": "browse_home", "time_delta": 5, "timestamp": 8},
        {"action": "search", "time_delta": 10, "timestamp": 18},
        {"action": "view_product", "time_delta": 8, "timestamp": 26},
        {"action": "add_to_cart", "time_delta": 30, "timestamp": 56},
    ]
    
    twin.build_profile("integrated_user", history)
    print("   ✅ Profile built!")
    
    # تحليل سلوك جديد
    print("\n🔍 Analyzing new behavior with both systems...")
    print("-" * 45)
    
    new_events = [
        {"action": "open_app", "timestamp": 100, "duration": 1},
        {"action": "login", "timestamp": 103, "duration": 3},
        {"action": "checkout", "timestamp": 106, "duration": 1},  # تخطي خطوات!
    ]
    
    # تحليل بمحرك الاستدلال (LSTM)
    lstm_result = engine.analyze_sequence("integrated_user", new_events)
    
    # تحليل بالتوأم الذكي
    twin_events = [
        {"action": "open_app", "time_delta": 0},
        {"action": "login", "time_delta": 3},
        {"action": "checkout", "time_delta": 3},
    ]
    twin_result = twin.compare_behavior("integrated_user", twin_events)
    
    print("\n   📊 LSTM Analysis (Pattern Detection):")
    print(f"      • Anomaly Score: {lstm_result.prediction.anomaly_score:.2f}")
    print(f"      • Recommendation: {lstm_result.recommendation.upper()}")
    
    print("\n   🧬 Smart Twin Analysis (Profile Matching):")
    print(f"      • Deviation Score: {twin_result['deviation_score']:.2f}")
    print(f"      • Risk Level: {twin_result['risk_level'].upper()}")
    
    # الدرجة المدمجة
    combined_score = (lstm_result.prediction.anomaly_score + twin_result['deviation_score']) / 2
    
    print("\n   🔗 Combined Analysis:")
    print(f"      • Combined Score: {combined_score:.2f}")
    
    if combined_score > 0.7:
        print("      • Final Decision: 🚫 BLOCK")
    elif combined_score > 0.4:
        print("      • Final Decision: ⚠️ VERIFY")
    else:
        print("      • Final Decision: ✅ ALLOW")


def main():
    print("\n" + "=" * 55)
    print("   [RASED] Smart Twin Demo")
    print("   نظام التوأم الذكي لكشف الاحتيال")
    print("=" * 55)
    
    # Demo 1: Smart Twin المستقل
    twin = demo_smart_twin_standalone()
    
    # Demo 2: التكامل مع محرك الاستدلال
    demo_integrated_analysis()
    
    # ملخص
    print_divider("📈 Summary")
    stats = twin.get_stats()
    print(f"\n   Total Profiles Created: {stats['total_profiles']}")
    print("\n   Smart Twin helps detect:")
    print("   • 👤 Different person using the account")
    print("   • 🤖 Bot behavior")
    print("   • ⚡ Unusual action speed")
    print("   • 🔀 Unusual action sequences")
    
    print("\n" + "=" * 55)
    print("   ✅ Demo completed successfully!")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
