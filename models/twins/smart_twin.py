"""
راصد (Rased) - Smart Twin
التوأم الذكي: نموذج سلوكي فريد لكل مستخدم

المبدأ:
- يتعلم نمط السلوك المعتاد للمستخدم
- يكشف أي انحراف عن هذا النمط
- يساعد في اكتشاف استخدام غير مصرح به للحساب
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class UserProfile:
    """الملف السلوكي للمستخدم"""
    user_id: str
    
    # أنماط التوقيت
    typical_hours: List[int] = field(default_factory=list)  # ساعات الاستخدام المعتادة
    avg_session_duration: float = 0.0  # متوسط مدة الجلسة
    avg_time_between_actions: float = 0.0  # متوسط الوقت بين الإجراءات
    
    # أنماط السلوك
    action_frequencies: Dict[str, float] = field(default_factory=dict)  # تكرار كل إجراء
    common_sequences: List[Tuple[str, str]] = field(default_factory=list)  # التسلسلات الشائعة
    favorite_actions: List[str] = field(default_factory=list)  # الإجراءات المفضلة
    
    # إحصائيات
    total_events: int = 0
    total_sessions: int = 0
    last_updated: Optional[float] = None
    
    # معايير الكشف
    speed_mean: float = 1.0  # متوسط السرعة
    speed_std: float = 0.5   # الانحراف المعياري للسرعة
    
    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "typical_hours": self.typical_hours,
            "avg_session_duration": self.avg_session_duration,
            "avg_time_between_actions": self.avg_time_between_actions,
            "action_frequencies": self.action_frequencies,
            "favorite_actions": self.favorite_actions,
            "total_events": self.total_events,
            "total_sessions": self.total_sessions,
        }


class SmartTwin:
    """
    التوأم الذكي - Smart Twin
    
    يُنشئ نموذج سلوكي فريد لكل مستخدم ويستخدمه لكشف:
    1. شخص آخر يستخدم الحساب
    2. سلوك غير معتاد من المستخدم نفسه
    3. محاولات اختراق الحساب
    """
    
    # عتبات الانحراف
    LOW_DEVIATION = 0.3      # انحراف منخفض (طبيعي)
    MEDIUM_DEVIATION = 0.6   # انحراف متوسط (مشبوه)
    HIGH_DEVIATION = 0.8     # انحراف عالي (خطر)
    
    # الإجراءات المدعومة
    ACTIONS = [
        "open_app", "login", "browse_home", "search",
        "view_product", "add_to_cart", "checkout",
        "view_profile", "change_settings", "logout"
    ]
    
    def __init__(self, learning_rate: float = 0.1):
        """
        تهيئة Smart Twin
        
        Args:
            learning_rate: معدل التعلم للتحديث التدريجي
        """
        self.learning_rate = learning_rate
        self.profiles: Dict[str, UserProfile] = {}
    
    def build_profile(self, user_id: str, events: List[Dict]) -> UserProfile:
        """
        بناء الملف السلوكي من تاريخ الأحداث
        
        Args:
            user_id: معرف المستخدم
            events: قائمة الأحداث التاريخية
            
        Returns:
            UserProfile: الملف السلوكي
        """
        if len(events) < 2:
            # إنشاء ملف افتراضي
            profile = UserProfile(user_id=user_id)
            self.profiles[user_id] = profile
            return profile
        
        # تحليل الأحداث
        actions = [e.get('action', 'browse_home') for e in events]
        time_deltas = [e.get('time_delta', 1.0) for e in events if 'time_delta' in e]
        timestamps = [e.get('timestamp', 0) for e in events if 'timestamp' in e]
        
        # حساب تكرار الإجراءات
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        total = len(actions)
        action_frequencies = {a: c / total for a, c in action_counts.items()}
        
        # الإجراءات المفضلة (الأكثر تكراراً)
        sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
        favorite_actions = [a for a, _ in sorted_actions[:3]]
        
        # التسلسلات الشائعة
        sequences = []
        for i in range(len(actions) - 1):
            sequences.append((actions[i], actions[i + 1]))
        
        sequence_counts = {}
        for seq in sequences:
            sequence_counts[seq] = sequence_counts.get(seq, 0) + 1
        
        common_sequences = sorted(sequence_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        common_sequences = [seq for seq, _ in common_sequences]
        
        # إحصائيات السرعة
        if time_deltas:
            speed_mean = np.mean(time_deltas)
            speed_std = np.std(time_deltas) if len(time_deltas) > 1 else 0.5
        else:
            speed_mean = 1.0
            speed_std = 0.5
        
        # ساعات الاستخدام المعتادة
        typical_hours = []
        for ts in timestamps:
            if ts > 0:
                try:
                    hour = datetime.fromtimestamp(ts).hour
                    typical_hours.append(hour)
                except:
                    pass
        
        if typical_hours:
            # أكثر 3 ساعات شيوعاً
            hour_counts = {}
            for h in typical_hours:
                hour_counts[h] = hour_counts.get(h, 0) + 1
            typical_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            typical_hours = [h for h, _ in typical_hours]
        
        # إنشاء الملف
        profile = UserProfile(
            user_id=user_id,
            typical_hours=typical_hours,
            avg_time_between_actions=speed_mean,
            action_frequencies=action_frequencies,
            common_sequences=common_sequences,
            favorite_actions=favorite_actions,
            total_events=len(events),
            total_sessions=1,
            last_updated=datetime.now().timestamp(),
            speed_mean=speed_mean,
            speed_std=max(speed_std, 0.1)  # حد أدنى للتباين
        )
        
        self.profiles[user_id] = profile
        return profile
    
    def update_profile(self, user_id: str, events: List[Dict]) -> UserProfile:
        """
        تحديث الملف السلوكي بأحداث جديدة
        
        Args:
            user_id: معرف المستخدم
            events: الأحداث الجديدة
            
        Returns:
            UserProfile: الملف المُحدّث
        """
        if user_id not in self.profiles:
            return self.build_profile(user_id, events)
        
        current = self.profiles[user_id]
        
        # بناء ملف مؤقت من الأحداث الجديدة
        new_profile = self.build_profile(f"temp_{user_id}", events)
        
        # دمج الملفين (تحديث تدريجي)
        lr = self.learning_rate
        
        # تحديث متوسط السرعة
        current.speed_mean = (1 - lr) * current.speed_mean + lr * new_profile.speed_mean
        current.speed_std = (1 - lr) * current.speed_std + lr * new_profile.speed_std
        current.avg_time_between_actions = current.speed_mean
        
        # تحديث تكرار الإجراءات
        for action, freq in new_profile.action_frequencies.items():
            if action in current.action_frequencies:
                current.action_frequencies[action] = (
                    (1 - lr) * current.action_frequencies[action] + lr * freq
                )
            else:
                current.action_frequencies[action] = freq * lr
        
        # إحصائيات
        current.total_events += len(events)
        current.total_sessions += 1
        current.last_updated = datetime.now().timestamp()
        
        # تنظيف الملف المؤقت
        del self.profiles[f"temp_{user_id}"]
        
        return current
    
    def compare_behavior(
        self,
        user_id: str,
        current_events: List[Dict]
    ) -> Dict:
        """
        مقارنة السلوك الحالي بالملف السلوكي
        
        Args:
            user_id: معرف المستخدم
            current_events: الأحداث الحالية
            
        Returns:
            dict: نتيجة المقارنة
        """
        if user_id not in self.profiles:
            return {
                "deviation_score": 0.0,
                "is_suspicious": False,
                "reason": "No profile exists for comparison",
                "details": {}
            }
        
        profile = self.profiles[user_id]
        deviation_scores = []
        details = {}
        reasons = []
        
        # 1. تحليل السرعة
        time_deltas = [e.get('time_delta', 1.0) for e in current_events if 'time_delta' in e]
        if time_deltas:
            current_speed = np.mean(time_deltas)
            
            # Z-score للسرعة
            if profile.speed_std > 0:
                speed_z = abs(current_speed - profile.speed_mean) / profile.speed_std
                speed_deviation = min(1.0, speed_z / 3.0)  # تطبيع لـ 0-1
            else:
                speed_deviation = 0.0
            
            deviation_scores.append(speed_deviation * 0.3)  # وزن 30%
            details["speed_deviation"] = speed_deviation
            
            if speed_deviation > 0.5:
                reasons.append(f"Unusual speed: {current_speed:.2f}s (expected: {profile.speed_mean:.2f}s)")
        
        # 2. تحليل الإجراءات
        actions = [e.get('action', 'browse_home') for e in current_events]
        if actions:
            # نسبة الإجراءات غير المعتادة
            unusual_count = 0
            for action in actions:
                if action not in profile.action_frequencies:
                    unusual_count += 1
                elif profile.action_frequencies.get(action, 0) < 0.05:
                    unusual_count += 0.5
            
            action_deviation = unusual_count / len(actions)
            deviation_scores.append(action_deviation * 0.4)  # وزن 40%
            details["action_deviation"] = action_deviation
            
            if action_deviation > 0.3:
                reasons.append(f"Unusual actions detected ({unusual_count:.0f} of {len(actions)})")
        
        # 3. تحليل التسلسلات
        if len(actions) >= 2:
            current_sequences = [(actions[i], actions[i+1]) for i in range(len(actions)-1)]
            unusual_sequences = 0
            
            for seq in current_sequences:
                if seq not in profile.common_sequences:
                    unusual_sequences += 1
            
            if current_sequences:
                sequence_deviation = unusual_sequences / len(current_sequences)
                deviation_scores.append(sequence_deviation * 0.3)  # وزن 30%
                details["sequence_deviation"] = sequence_deviation
                
                if sequence_deviation > 0.5:
                    reasons.append("Unusual action sequences")
        
        # حساب الدرجة النهائية
        if deviation_scores:
            final_score = sum(deviation_scores)
        else:
            final_score = 0.0
        
        # تحديد مستوى الخطر
        if final_score > self.HIGH_DEVIATION:
            risk_level = "high"
        elif final_score > self.MEDIUM_DEVIATION:
            risk_level = "medium"
        elif final_score > self.LOW_DEVIATION:
            risk_level = "low"
        else:
            risk_level = "normal"
        
        return {
            "deviation_score": min(1.0, final_score),
            "risk_level": risk_level,
            "is_suspicious": final_score > self.MEDIUM_DEVIATION,
            "reason": "; ".join(reasons) if reasons else "Behavior matches profile",
            "details": details
        }
    
    def get_deviation_score(self, user_id: str, current_events: List[Dict]) -> float:
        """
        حساب درجة الانحراف فقط
        
        Args:
            user_id: معرف المستخدم
            current_events: الأحداث الحالية
            
        Returns:
            float: درجة الانحراف (0-1)
        """
        result = self.compare_behavior(user_id, current_events)
        return result["deviation_score"]
    
    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """الحصول على الملف السلوكي"""
        return self.profiles.get(user_id)
    
    def has_profile(self, user_id: str) -> bool:
        """هل يوجد ملف سلوكي للمستخدم؟"""
        return user_id in self.profiles
    
    def get_stats(self) -> Dict:
        """إحصائيات Smart Twin"""
        return {
            "total_profiles": len(self.profiles),
            "profiles": [
                {"user_id": uid, "events": p.total_events}
                for uid, p in self.profiles.items()
            ]
        }


# Demo
if __name__ == "__main__":
    print("=" * 50)
    print("   Smart Twin - التوأم الذكي")
    print("=" * 50)
    
    twin = SmartTwin()
    
    # بناء ملف سلوكي من تاريخ المستخدم
    print("\n📝 بناء الملف السلوكي للمستخدم...")
    
    history = [
        {"action": "open_app", "time_delta": 0, "timestamp": 1000},
        {"action": "login", "time_delta": 3, "timestamp": 1003},
        {"action": "browse_home", "time_delta": 5, "timestamp": 1008},
        {"action": "search", "time_delta": 10, "timestamp": 1018},
        {"action": "view_product", "time_delta": 8, "timestamp": 1026},
        {"action": "add_to_cart", "time_delta": 30, "timestamp": 1056},
        {"action": "browse_home", "time_delta": 5, "timestamp": 1061},
        {"action": "search", "time_delta": 12, "timestamp": 1073},
        {"action": "view_product", "time_delta": 6, "timestamp": 1079},
        {"action": "checkout", "time_delta": 45, "timestamp": 1124},
    ]
    
    profile = twin.build_profile("user_khulaif", history)
    print(f"   ✅ تم بناء الملف: {profile.total_events} حدث")
    print(f"   الإجراءات المفضلة: {profile.favorite_actions}")
    print(f"   متوسط السرعة: {profile.speed_mean:.1f}s")
    
    # اختبار 1: سلوك مطابق
    print("\n🔍 اختبار 1: سلوك مطابق للملف...")
    same_behavior = [
        {"action": "open_app", "time_delta": 0},
        {"action": "login", "time_delta": 4},
        {"action": "browse_home", "time_delta": 6},
        {"action": "search", "time_delta": 8},
        {"action": "view_product", "time_delta": 7},
    ]
    
    result = twin.compare_behavior("user_khulaif", same_behavior)
    print(f"   درجة الانحراف: {result['deviation_score']:.2f}")
    print(f"   مستوى الخطر: {result['risk_level']}")
    print(f"   مشبوه؟ {result['is_suspicious']}")
    
    # اختبار 2: سلوك مختلف (شخص آخر؟)
    print("\n⚠️ اختبار 2: سلوك مختلف تماماً...")
    different_behavior = [
        {"action": "open_app", "time_delta": 0},
        {"action": "login", "time_delta": 0.5},  # سريع جداً
        {"action": "change_settings", "time_delta": 0.3},  # غير معتاد
        {"action": "view_profile", "time_delta": 0.2},  # غير معتاد
        {"action": "logout", "time_delta": 0.1},  # سريع جداً
    ]
    
    result = twin.compare_behavior("user_khulaif", different_behavior)
    print(f"   درجة الانحراف: {result['deviation_score']:.2f}")
    print(f"   مستوى الخطر: {result['risk_level']}")
    print(f"   مشبوه؟ {result['is_suspicious']}")
    print(f"   السبب: {result['reason']}")
    
    print("\n" + "=" * 50)
    print("   ✅ Smart Twin يعمل بنجاح!")
    print("=" * 50)
