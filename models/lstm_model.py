"""
راصد (Rased) - LSTM Sequence Model
نموذج LSTM لتحليل تسلسل سلوك المستخدم وكشف الشذوذ

المبدأ:
- المدخل: تسلسل أحداث المستخدم (sequence of events)
- المخرج: التنبؤ بالإجراء التالي + درجة الشذوذ
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PredictionResult:
    """نتيجة التنبؤ من نموذج LSTM"""
    predicted_action: str           # الإجراء المتوقع التالي
    action_probabilities: Dict[str, float]  # احتمالات كل إجراء
    confidence: float               # نسبة الثقة بالتنبؤ
    anomaly_score: float            # درجة الشذوذ (0=طبيعي, 1=شاذ)
    is_anomaly: bool                # هل السلوك شاذ؟
    reason: Optional[str] = None    # سبب الشذوذ إن وجد


class RasedLSTM:
    """
    نموذج LSTM لتحليل سلوك المستخدم
    
    المبدأ:
    1. يتعلم النموذج أنماط السلوك الطبيعي من تسلسل الأحداث
    2. عند كل حدث جديد، يتنبأ بالإجراء التالي المتوقع
    3. إذا كان الإجراء الفعلي مختلفاً جداً عن المتوقع = شذوذ
    4. إذا كان التوقيت سريعاً جداً (غير بشري) = بوت
    """
    
    # قائمة الإجراءات المدعومة
    ACTIONS = [
        "open_app",         # فتح التطبيق
        "login",            # تسجيل الدخول
        "browse_home",      # تصفح الرئيسية
        "search",           # بحث
        "view_product",     # عرض منتج/خدمة
        "add_to_cart",      # إضافة للسلة
        "checkout",         # الدفع
        "view_profile",     # عرض الملف الشخصي
        "change_settings",  # تغيير الإعدادات
        "logout",           # تسجيل الخروج
    ]
    
    # حدود الشذوذ
    ANOMALY_THRESHOLD = 0.7      # عتبة الشذوذ
    MIN_HUMAN_TIME = 0.5         # الحد الأدنى للوقت البشري (ثانية)
    BOT_SPEED_THRESHOLD = 0.3    # عتبة سرعة البوت (ثانية)
    
    def __init__(
        self,
        sequence_length: int = 10,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
    ):
        """
        تهيئة نموذج LSTM
        
        Args:
            sequence_length: طول التسلسل المستخدم للتنبؤ
            hidden_dim: أبعاد الطبقة المخفية
            embedding_dim: أبعاد تمثيل الحدث
        """
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_actions = len(self.ACTIONS)
        
        # خريطة الإجراءات
        self.action_to_idx = {a: i for i, a in enumerate(self.ACTIONS)}
        self.idx_to_action = {i: a for i, a in enumerate(self.ACTIONS)}
        
        # تهيئة الأوزان (Xavier initialization)
        self._initialize_weights()
        
        # إحصائيات
        self.total_predictions = 0
        self.anomalies_detected = 0
    
    def _initialize_weights(self):
        """تهيئة أوزان الشبكة"""
        input_dim = self.embedding_dim + 2  # embedding + time_delta + duration
        
        # Embedding layer
        self.embedding = np.random.randn(self.num_actions, self.embedding_dim) * 0.1
        
        # LSTM weights (simplified single layer)
        scale = np.sqrt(2.0 / (input_dim + self.hidden_dim))
        
        # Forget gate
        self.Wf = np.random.randn(self.hidden_dim, input_dim + self.hidden_dim) * scale
        self.bf = np.zeros(self.hidden_dim)
        
        # Input gate
        self.Wi = np.random.randn(self.hidden_dim, input_dim + self.hidden_dim) * scale
        self.bi = np.zeros(self.hidden_dim)
        
        # Cell gate
        self.Wc = np.random.randn(self.hidden_dim, input_dim + self.hidden_dim) * scale
        self.bc = np.zeros(self.hidden_dim)
        
        # Output gate
        self.Wo = np.random.randn(self.hidden_dim, input_dim + self.hidden_dim) * scale
        self.bo = np.zeros(self.hidden_dim)
        
        # Output layer (action prediction)
        self.Wa = np.random.randn(self.num_actions, self.hidden_dim) * 0.1
        self.ba = np.zeros(self.num_actions)
        
        # Anomaly detection layer
        self.Wanomaly = np.random.randn(1, self.hidden_dim) * 0.1
        self.banomaly = np.zeros(1)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation"""
        return np.tanh(x)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def _encode_event(self, event: Dict) -> np.ndarray:
        """
        تحويل حدث واحد لمتجه رقمي
        
        Args:
            event: {
                'action': اسم الإجراء,
                'time_delta': الوقت منذ الحدث السابق (ثانية),
                'duration': مدة الحدث (ثانية)
            }
        """
        action = event.get('action', 'browse_home')
        time_delta = event.get('time_delta', 1.0)
        duration = event.get('duration', 1.0)
        
        # الحصول على embedding الإجراء
        action_idx = self.action_to_idx.get(action, 0)
        action_emb = self.embedding[action_idx]
        
        # تطبيع الوقت (log scale)
        time_feature = np.log1p(time_delta) / 5.0  # normalize
        duration_feature = np.log1p(duration) / 5.0
        
        # دمج المتجهات
        return np.concatenate([action_emb, [time_feature, duration_feature]])
    
    def _lstm_step(
        self,
        x: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        خطوة LSTM واحدة
        
        Args:
            x: المدخل الحالي
            h_prev: الحالة المخفية السابقة
            c_prev: حالة الخلية السابقة
            
        Returns:
            (hidden_state, cell_state)
        """
        # Concatenate input and previous hidden state
        combined = np.concatenate([x, h_prev])
        
        # Gates
        f = self._sigmoid(self.Wf @ combined + self.bf)  # Forget gate
        i = self._sigmoid(self.Wi @ combined + self.bi)  # Input gate
        c_tilde = self._tanh(self.Wc @ combined + self.bc)  # Candidate
        o = self._sigmoid(self.Wo @ combined + self.bo)  # Output gate
        
        # New cell and hidden states
        c = f * c_prev + i * c_tilde
        h = o * self._tanh(c)
        
        return h, c
    
    def forward(self, events: List[Dict]) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Forward pass عبر تسلسل الأحداث
        
        Args:
            events: قائمة الأحداث
            
        Returns:
            (action_probs, hidden_state, anomaly_score)
        """
        # تهيئة الحالة
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        
        # معالجة كل حدث في التسلسل
        for event in events[-self.sequence_length:]:
            x = self._encode_event(event)
            h, c = self._lstm_step(x, h, c)
        
        # التنبؤ بالإجراء التالي
        action_logits = self.Wa @ h + self.ba
        action_probs = self._softmax(action_logits)
        
        # درجة الشذوذ
        anomaly_raw = self.Wanomaly @ h + self.banomaly
        anomaly_score = self._sigmoid(anomaly_raw)[0]
        
        return action_probs, h, anomaly_score
    
    def predict(
        self,
        events: List[Dict],
        actual_action: Optional[str] = None
    ) -> PredictionResult:
        """
        التنبؤ بالإجراء التالي وكشف الشذوذ
        
        Args:
            events: تسلسل أحداث المستخدم
            actual_action: الإجراء الفعلي (للمقارنة)
            
        Returns:
            PredictionResult
        """
        if len(events) == 0:
            return PredictionResult(
                predicted_action="open_app",
                action_probabilities={a: 1/self.num_actions for a in self.ACTIONS},
                confidence=0.1,
                anomaly_score=0.0,
                is_anomaly=False
            )
        
        # Forward pass
        action_probs, hidden, base_anomaly = self.forward(events)
        
        # الإجراء المتوقع
        pred_idx = np.argmax(action_probs)
        predicted_action = self.idx_to_action[pred_idx]
        confidence = float(action_probs[pred_idx])
        
        # احتمالات كل إجراء
        probs_dict = {self.idx_to_action[i]: float(p) for i, p in enumerate(action_probs)}
        
        # حساب الشذوذ
        anomaly_score = base_anomaly
        reason = None
        
        # فحص السرعة (كشف البوت)
        if events:
            last_event = events[-1]
            time_delta = last_event.get('time_delta', 1.0)
            
            if time_delta < self.BOT_SPEED_THRESHOLD:
                # سلوك سريع جداً = بوت محتمل
                anomaly_score = max(anomaly_score, 0.9)
                reason = f"Non-human speed: {time_delta:.2f}s between events"
            elif time_delta < self.MIN_HUMAN_TIME:
                # سلوك سريع = مشبوه
                anomaly_score = max(anomaly_score, 0.6)
                reason = f"Suspicious speed: {time_delta:.2f}s"
        
        # إذا كان الإجراء الفعلي معطى، قارنه بالمتوقع
        if actual_action and actual_action in self.action_to_idx:
            actual_idx = self.action_to_idx[actual_action]
            actual_prob = action_probs[actual_idx]
            
            # إذا كان الإجراء الفعلي غير متوقع بشدة
            if actual_prob < 0.05:
                anomaly_score = max(anomaly_score, 0.8)
                reason = f"Unexpected action: {actual_action} (probability {actual_prob:.1%})"
            elif actual_prob < 0.15:
                anomaly_score = max(anomaly_score, 0.5)
                reason = f"Unusual action: {actual_action} (probability {actual_prob:.1%})"
        
        is_anomaly = anomaly_score > self.ANOMALY_THRESHOLD
        
        # تحديث الإحصائيات
        self.total_predictions += 1
        if is_anomaly:
            self.anomalies_detected += 1
        
        return PredictionResult(
            predicted_action=predicted_action,
            action_probabilities=probs_dict,
            confidence=confidence,
            anomaly_score=float(anomaly_score),
            is_anomaly=is_anomaly,
            reason=reason
        )
    
    def get_stats(self) -> Dict:
        """إحصائيات النموذج"""
        return {
            "total_predictions": self.total_predictions,
            "anomalies_detected": self.anomalies_detected,
            "anomaly_rate": self.anomalies_detected / max(1, self.total_predictions)
        }


# Demo
if __name__ == "__main__":
    model = RasedLSTM()
    
    # مثال: سلوك طبيعي
    normal_events = [
        {"action": "open_app", "time_delta": 0, "duration": 1},
        {"action": "login", "time_delta": 2, "duration": 5},
        {"action": "browse_home", "time_delta": 3, "duration": 10},
        {"action": "search", "time_delta": 5, "duration": 3},
        {"action": "view_product", "time_delta": 2, "duration": 30},
    ]
    
    result = model.predict(normal_events, actual_action="add_to_cart")
    print("=== سلوك طبيعي ===")
    print(f"التنبؤ: {result.predicted_action}")
    print(f"الثقة: {result.confidence:.1%}")
    print(f"درجة الشذوذ: {result.anomaly_score:.2f}")
    print(f"شاذ؟ {result.is_anomaly}")
    
    # مثال: سلوك بوت (سريع جداً)
    bot_events = [
        {"action": "open_app", "time_delta": 0, "duration": 0.1},
        {"action": "login", "time_delta": 0.1, "duration": 0.1},
        {"action": "checkout", "time_delta": 0.1, "duration": 0.1},
    ]
    
    result = model.predict(bot_events)
    print("\n=== سلوك بوت ===")
    print(f"درجة الشذوذ: {result.anomaly_score:.2f}")
    print(f"شاذ؟ {result.is_anomaly}")
    print(f"السبب: {result.reason}")
