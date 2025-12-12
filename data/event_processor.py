"""
راصد (Rased) - Event Processor
معالج الأحداث: تحويل الأحداث الخام لتنسيق يفهمه LSTM
"""

from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class Event:
    """حدث مستخدم واحد"""
    action: str                 # نوع الإجراء
    timestamp: float            # وقت الحدث (Unix timestamp)
    duration: float = 0.0       # مدة الحدث (ثانية)
    metadata: Optional[Dict] = None  # بيانات إضافية


class EventProcessor:
    """
    معالج أحداث المستخدم
    
    يحول الأحداث الخام إلى تنسيق مناسب للـ LSTM:
    - حساب الفارق الزمني بين الأحداث
    - تطبيع القيم
    - تصفية الأحداث غير الصالحة
    """
    
    # الإجراءات الصالحة
    VALID_ACTIONS = {
        "open_app", "login", "browse_home", "search",
        "view_product", "add_to_cart", "checkout",
        "view_profile", "change_settings", "logout"
    }
    
    def __init__(self, max_sequence_length: int = 50):
        """
        تهيئة معالج الأحداث
        
        Args:
            max_sequence_length: أقصى طول للتسلسل المحفوظ
        """
        self.max_sequence_length = max_sequence_length
        self.user_sequences: Dict[str, List[Dict]] = {}
    
    def process_event(
        self,
        user_id: str,
        action: str,
        timestamp: Optional[float] = None,
        duration: float = 0.0,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        معالجة حدث جديد وإضافته لتسلسل المستخدم
        
        Args:
            user_id: معرف المستخدم
            action: نوع الإجراء
            timestamp: وقت الحدث (اختياري، يستخدم الوقت الحالي)
            duration: مدة الإجراء
            metadata: بيانات إضافية
            
        Returns:
            الحدث المعالج
        """
        # التحقق من صحة الإجراء
        if action not in self.VALID_ACTIONS:
            action = "browse_home"  # إجراء افتراضي
        
        # الوقت
        if timestamp is None:
            timestamp = datetime.now().timestamp()
        
        # الحصول على تسلسل المستخدم
        if user_id not in self.user_sequences:
            self.user_sequences[user_id] = []
        
        sequence = self.user_sequences[user_id]
        
        # حساب الفارق الزمني
        if sequence:
            last_timestamp = sequence[-1].get('timestamp', timestamp)
            time_delta = max(0, timestamp - last_timestamp)
        else:
            time_delta = 0.0
        
        # إنشاء الحدث المعالج
        processed_event = {
            'action': action,
            'timestamp': timestamp,
            'time_delta': time_delta,
            'duration': max(0, duration),
            'metadata': metadata or {}
        }
        
        # إضافة للتسلسل
        sequence.append(processed_event)
        
        # الاحتفاظ بآخر N حدث فقط
        if len(sequence) > self.max_sequence_length:
            self.user_sequences[user_id] = sequence[-self.max_sequence_length:]
        
        return processed_event
    
    def get_user_sequence(self, user_id: str) -> List[Dict]:
        """
        الحصول على تسلسل أحداث المستخدم
        
        Args:
            user_id: معرف المستخدم
            
        Returns:
            قائمة الأحداث
        """
        return self.user_sequences.get(user_id, [])
    
    def clear_user_sequence(self, user_id: str):
        """مسح تسلسل المستخدم"""
        if user_id in self.user_sequences:
            del self.user_sequences[user_id]
    
    def get_stats(self) -> Dict:
        """إحصائيات المعالج"""
        return {
            "total_users": len(self.user_sequences),
            "total_events": sum(len(seq) for seq in self.user_sequences.values())
        }


# Demo
if __name__ == "__main__":
    processor = EventProcessor()
    
    # محاكاة أحداث مستخدم
    events = [
        ("user_123", "open_app", 0, 1),
        ("user_123", "login", 2, 5),
        ("user_123", "browse_home", 3, 10),
        ("user_123", "search", 5, 3),
    ]
    
    import time
    base_time = time.time()
    
    for user_id, action, delay, duration in events:
        processor.process_event(
            user_id=user_id,
            action=action,
            timestamp=base_time + delay,
            duration=duration
        )
    
    sequence = processor.get_user_sequence("user_123")
    print(f"عدد الأحداث: {len(sequence)}")
    for event in sequence:
        print(f"  - {event['action']}: time_delta={event['time_delta']:.1f}s")
