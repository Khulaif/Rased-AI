# راصد - Rased

## 🎯 نظام كشف الاحتيال باستخدام LSTM

**راصد** هو نموذج LSTM يحلل تسلسل سلوك المستخدم لكشف الشذوذ والاحتيال قبل وقوعه.

---

## 💡 الفكرة

```
تسلسل الأحداث → LSTM → التنبؤ بالإجراء التالي → كشف الشذوذ → قرار
```

### المبدأ
سلوك المستخدم = **تسلسل زمني للأحداث**. الـ LSTM يتعلم هذا التسلسل:

1. **التنبؤ**: يتوقع الإجراء التالي للمستخدم
2. **كشف الشذوذ**: إذا كان السلوك غريباً (سريع جداً، تخطي خطوات) = شاذ
3. **القرار**: سماح / تحقق إضافي / حظر

### مثال
```
المستخدم العادي:
  1. فتح التطبيق (t=0)
  2. تسجيل الدخول (t=3s)
  3. تصفح (t=15s)
  4. بحث (t=40s)
  5. عرض منتج (t=50s)  → توقع: إضافة للسلة ✅

البوت:
  1. فتح التطبيق (t=0)
  2. تسجيل الدخول (t=0.05s)  ← سريع جداً!
  3. الدفع (t=0.1s)           ← 🚫 حظر
```

---

## 📁 الهيكل

```
Rased/
├── models/
│   ├── lstm_model.py          # نموذج LSTM
│   └── twins/
│       └── smart_twin.py      # التوأم الذكي 🆕
├── data/
│   └── event_processor.py     # معالجة الأحداث
├── engine/
│   └── inference.py           # محرك الاستدلال
├── examples/
│   ├── demo.py                # مثال تطبيقي
│   └── smart_twin_demo.py     # مثال التوأم الذكي 🆕
└── README.md
```

---

## 🧬 Smart Twin - التوأم الذكي

ميزة جديدة تُنشئ **ملف سلوكي فريد** لكل مستخدم وتستخدمه لكشف:
- 👤 **شخص آخر يستخدم الحساب** (سلوك مختلف عن المعتاد)
- 🤖 **البوتات** (سرعة غير بشرية)
- ⚡ **تغيير مفاجئ في السلوك** (إجراءات غير معتادة)

### كيف يعمل؟
```
تاريخ السلوك → بناء الملف السلوكي → مقارنة السلوك الجديد → كشف الانحراف
```

### مثال الاستخدام
```python
from models.twins.smart_twin import SmartTwin

twin = SmartTwin()

# بناء الملف السلوكي من تاريخ المستخدم
history = [
    {"action": "open_app", "time_delta": 0},
    {"action": "login", "time_delta": 3},
    {"action": "browse_home", "time_delta": 5},
    {"action": "search", "time_delta": 10},
]
twin.build_profile("user_123", history)

# مقارنة سلوك جديد
new_behavior = [
    {"action": "open_app", "time_delta": 0},
    {"action": "login", "time_delta": 0.1},  # سريع جداً!
    {"action": "change_settings", "time_delta": 0.2},  # غير معتاد!
]
result = twin.compare_behavior("user_123", new_behavior)

print(f"درجة الانحراف: {result['deviation_score']}")  # 0.85
print(f"مشبوه؟ {result['is_suspicious']}")  # True
print(f"السبب: {result['reason']}")  # Unusual speed...
```

### تشغيل مثال التوأم الذكي
```bash
python examples/smart_twin_demo.py
```

---

## 🚀 التشغيل

### تثبيت المتطلبات
```bash
pip install numpy
```

### تشغيل المثال
```bash
python examples/demo.py
```

### المخرج المتوقع
```
==================================================
   🔍 راصد - نظام كشف الاحتيال بالذكاء الاصطناعي
==================================================

👤 سيناريو 1: مستخدم طبيعي
   درجة الشذوذ: 0.15
   التوصية: ✅ سماح

🤖 سيناريو 2: بوت (سرعة غير بشرية)
   درجة الشذوذ: 0.92
   التوصية: 🚫 حظر
   السبب: سرعة غير بشرية: 0.05s بين الأحداث
```

---

## 🔧 الاستخدام البرمجي

```python
from engine.inference import InferenceEngine

engine = InferenceEngine()

# تحليل تسلسل أحداث
events = [
    {"action": "open_app", "timestamp": 0, "duration": 1},
    {"action": "login", "timestamp": 3, "duration": 5},
    {"action": "browse_home", "timestamp": 10, "duration": 15},
]

result = engine.analyze_sequence("user_123", events)

print(f"درجة الشذوذ: {result.prediction.anomaly_score}")
print(f"التوصية: {result.recommendation}")  # allow / verify / block
```

---

## 📊 مستويات القرار

| درجة الشذوذ | القرار | الوصف |
|-------------|--------|-------|
| 0.0 - 0.3 | ✅ سماح | سلوك طبيعي |
| 0.3 - 0.7 | ⚠️ تحقق | سلوك مشبوه، طلب OTP |
| 0.7 - 1.0 | 🚫 حظر | سلوك شاذ (بوت/احتيال) |

---

## 🔍 ما يكتشفه النظام

- **🤖 البوتات**: سرعة غير بشرية (< 0.3 ثانية بين الأحداث)
- **⏭️ تخطي الخطوات**: الذهاب للدفع مباشرة بدون تصفح
- **❓ إجراءات غير متوقعة**: إجراء لا يتناسب مع السياق

---

## 🔗 GitHub

https://github.com/Khulaif/Rased-AI
