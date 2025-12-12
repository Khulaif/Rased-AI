"""
راصد (Rased) - Inference Engine
محرك الاستدلال: تشغيل نموذج LSTM على أحداث المستخدم
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import Dict, List, Optional
from dataclasses import dataclass

from models.lstm_model import RasedLSTM, PredictionResult
from data.event_processor import EventProcessor


@dataclass
class AnalysisResult:
    """نتيجة تحليل سلوك المستخدم"""
    user_id: str
    prediction: PredictionResult
    sequence_length: int
    recommendation: str  # "allow" | "verify" | "block"
    
    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "predicted_action": self.prediction.predicted_action,
            "confidence": self.prediction.confidence,
            "anomaly_score": self.prediction.anomaly_score,
            "is_anomaly": self.prediction.is_anomaly,
            "reason": self.prediction.reason,
            "sequence_length": self.sequence_length,
            "recommendation": self.recommendation
        }


class InferenceEngine:
    """
    محرك الاستدلال
    
    يجمع بين:
    1. معالج الأحداث (EventProcessor)
    2. نموذج LSTM (RasedLSTM)
    
    ويُنتج توصية بناءً على درجة الشذوذ
    """
    
    # عتبات القرار
    ALLOW_THRESHOLD = 0.3     # أقل من هذا = سماح
    VERIFY_THRESHOLD = 0.7    # بين هذا والسابق = تحقق
    # أعلى من VERIFY = حظر
    
    def __init__(self):
        """تهيئة محرك الاستدلال"""
        self.model = RasedLSTM()
        self.processor = EventProcessor()
        self.total_analyses = 0
    
    def process_event(
        self,
        user_id: str,
        action: str,
        timestamp: Optional[float] = None,
        duration: float = 0.0
    ) -> AnalysisResult:
        """
        معالجة حدث جديد وتحليله
        
        Args:
            user_id: معرف المستخدم
            action: الإجراء الحالي
            timestamp: وقت الحدث
            duration: مدة الإجراء
            
        Returns:
            نتيجة التحليل مع التوصية
        """
        # معالجة الحدث
        self.processor.process_event(
            user_id=user_id,
            action=action,
            timestamp=timestamp,
            duration=duration
        )
        
        # الحصول على تسلسل المستخدم
        sequence = self.processor.get_user_sequence(user_id)
        
        # التنبؤ
        prediction = self.model.predict(sequence, actual_action=action)
        
        # تحديد التوصية
        if prediction.anomaly_score < self.ALLOW_THRESHOLD:
            recommendation = "allow"
        elif prediction.anomaly_score < self.VERIFY_THRESHOLD:
            recommendation = "verify"
        else:
            recommendation = "block"
        
        self.total_analyses += 1
        
        return AnalysisResult(
            user_id=user_id,
            prediction=prediction,
            sequence_length=len(sequence),
            recommendation=recommendation
        )
    
    def analyze_sequence(
        self,
        user_id: str,
        events: List[Dict]
    ) -> AnalysisResult:
        """
        تحليل تسلسل كامل من الأحداث
        
        Args:
            user_id: معرف المستخدم
            events: قائمة الأحداث
            
        Returns:
            نتيجة التحليل
        """
        # مسح التسلسل القديم
        self.processor.clear_user_sequence(user_id)
        
        # معالجة كل حدث
        for event in events:
            self.processor.process_event(
                user_id=user_id,
                action=event.get('action', 'browse_home'),
                timestamp=event.get('timestamp'),
                duration=event.get('duration', 0)
            )
        
        # الحصول على التسلسل المعالج
        sequence = self.processor.get_user_sequence(user_id)
        
        # آخر إجراء
        last_action = events[-1].get('action') if events else None
        
        # التنبؤ
        prediction = self.model.predict(sequence, actual_action=last_action)
        
        # تحديد التوصية
        if prediction.anomaly_score < self.ALLOW_THRESHOLD:
            recommendation = "allow"
        elif prediction.anomaly_score < self.VERIFY_THRESHOLD:
            recommendation = "verify"
        else:
            recommendation = "block"
        
        self.total_analyses += 1
        
        return AnalysisResult(
            user_id=user_id,
            prediction=prediction,
            sequence_length=len(sequence),
            recommendation=recommendation
        )
    
    def get_stats(self) -> Dict:
        """إحصائيات المحرك"""
        return {
            "total_analyses": self.total_analyses,
            "model_stats": self.model.get_stats(),
            "processor_stats": self.processor.get_stats()
        }


# Demo
if __name__ == "__main__":
    engine = InferenceEngine()
    
    print("=== محرك الاستدلال - راصد ===\n")
    
    # سيناريو 1: سلوك طبيعي
    print("📱 سيناريو 1: مستخدم طبيعي")
    normal_events = [
        {"action": "open_app", "timestamp": 0, "duration": 1},
        {"action": "login", "timestamp": 3, "duration": 5},
        {"action": "browse_home", "timestamp": 10, "duration": 15},
        {"action": "search", "timestamp": 28, "duration": 5},
        {"action": "view_product", "timestamp": 35, "duration": 30},
        {"action": "add_to_cart", "timestamp": 70, "duration": 2},
    ]
    
    result = engine.analyze_sequence("user_normal", normal_events)
    print(f"   التوصية: {result.recommendation}")
    print(f"   درجة الشذوذ: {result.prediction.anomaly_score:.2f}")
    print(f"   التنبؤ التالي: {result.prediction.predicted_action}")
    
    # سيناريو 2: بوت
    print("\n🤖 سيناريو 2: بوت")
    bot_events = [
        {"action": "open_app", "timestamp": 0, "duration": 0.05},
        {"action": "login", "timestamp": 0.1, "duration": 0.05},
        {"action": "checkout", "timestamp": 0.2, "duration": 0.05},
    ]
    
    result = engine.analyze_sequence("user_bot", bot_events)
    print(f"   التوصية: {result.recommendation}")
    print(f"   درجة الشذوذ: {result.prediction.anomaly_score:.2f}")
    print(f"   السبب: {result.prediction.reason}")
    
    # سيناريو 3: سلوك مشبوه
    print("\n⚠️ سيناريو 3: سلوك مشبوه")
    suspicious_events = [
        {"action": "open_app", "timestamp": 0, "duration": 1},
        {"action": "login", "timestamp": 1, "duration": 2},
        {"action": "change_settings", "timestamp": 3.5, "duration": 1},
        {"action": "checkout", "timestamp": 5, "duration": 0.5},
    ]
    
    result = engine.analyze_sequence("user_suspicious", suspicious_events)
    print(f"   التوصية: {result.recommendation}")
    print(f"   درجة الشذوذ: {result.prediction.anomaly_score:.2f}")
    
    print("\n" + "="*40)
    print(f"📊 الإحصائيات: {engine.get_stats()}")
