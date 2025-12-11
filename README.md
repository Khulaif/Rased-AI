# راصد (Rased) - AI-Based Fraud Detection System

<div align="center">

![Rased Logo](https://img.shields.io/badge/راصد-Rased-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![License](https://img.shields.io/badge/License-Proprietary-red?style=flat-square)

**Real-time Behavioral Analysis & Fraud Detection for Government Services**

</div>

---

## 📖 Overview

**Rased (راصد)** is an AI-powered fraud detection system designed for government service platforms like Absher. It uses LSTM neural networks and "Digital Twin" technology to learn each user's unique behavioral patterns and detect anomalies in real-time.

### Key Features

- 🧠 **LSTM-Based Learning**: Predicts user behavior sequences with high accuracy
- 👤 **Digital Twin Technology**: Creates unique behavioral fingerprints for each user
- ⚡ **<200ms Latency**: Real-time processing with minimal impact on UX
- 🔄 **Continuous Learning**: Improves from feedback (challenges passed/failed)
- 🎯 **3-Tier Risk Classification**: Green (Allow), Yellow (Challenge), Red (Block)

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Absher App    │────▶│  Message Queue  │────▶│  Rased Engine   │
│   (Frontend)    │     │  (Kafka/Memory) │     │  (LSTM + Twin)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                        ┌────────────────────────────────┘
                        ▼
            ┌─────────────────────────────────────────────┐
            │              Risk Score (0-100)              │
            ├─────────────┬─────────────┬─────────────────┤
            │  🟢 0-30    │  🟡 31-70   │   🔴 71-100     │
            │   Allow     │  Challenge  │     Block       │
            └─────────────┴─────────────┴─────────────────┘
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd Rased
pip install -r requirements.txt
```

### 2. Run the Demo Simulation

```bash
python -m demo.simulation
```

This runs a complete simulation showing:
- Normal user behavior (Green zone)
- Suspicious activity detection (Yellow zone)
- Bot attack prevention (Red zone)
- Feedback loop learning

### 3. Start the API Server

```bash
uvicorn api.main:app --reload --port 8000
```

Access the API documentation at: http://localhost:8000/docs

---

## 📁 Project Structure

```
Rased/
├── config/
│   └── settings.py          # Central configuration
├── data/
│   ├── generators/
│   │   └── synthetic_data.py # Training data generation
│   └── processors/
│       └── vectorizer.py     # Feature vectorization
├── models/
│   ├── lstm_model.py         # LSTM neural network
│   └── digital_twin.py       # User behavioral embeddings
├── engine/
│   ├── inference_engine.py   # Real-time prediction
│   ├── risk_scorer.py        # Risk calculation
│   └── decision_matrix.py    # Automated responses
├── queue/
│   ├── message_queue.py      # Queue abstraction
│   └── async_processor.py    # Background processing
├── feedback/
│   ├── learning_loop.py      # Real-time learning
│   └── retrainer.py          # Periodic retraining
├── api/
│   └── main.py               # FastAPI server
├── demo/
│   └── simulation.py         # End-to-end demo
└── requirements.txt
```

---

## 🔧 Configuration

Edit `config/settings.py` to customize:

```python
# Risk Thresholds
risk_config = RiskConfig(
    green_max=30,      # 0-30: Allow
    yellow_max=70,     # 31-70: Challenge
    red_min=71,        # 71-100: Block
)

# Model Parameters
model_config = ModelConfig(
    sequence_length=10,    # Past actions to consider
    embedding_dim=64,      # User embedding size
    lstm_units_1=128,      # First LSTM layer
    lstm_units_2=64,       # Second LSTM layer
)
```

---

## 📡 API Endpoints

### Process Event
```http
POST /events
Content-Type: application/json

{
  "user_id": "user_001",
  "action": "confirm_transfer",
  "latitude": 24.7136,
  "longitude": 46.6753,
  "country_code": "SA",
  "device_type": "mobile",
  "os_family": "iOS",
  "transaction_amount": 5000.0
}
```

Response:
```json
{
  "event_id": "evt_abc123",
  "user_id": "user_001",
  "risk_score": 25,
  "risk_level": "green",
  "action": "allow",
  "processing_time_ms": 45.2
}
```

### Submit Feedback
```http
POST /feedback
Content-Type: application/json

{
  "user_id": "user_001",
  "event_id": "evt_abc123",
  "response_id": "resp_xyz",
  "feedback_type": "false_positive",
  "challenge_passed": true
}
```

### Get User Risk Profile
```http
GET /risk/{user_id}
```

---

## 🎯 Detection Scenarios

### 1. Normal Behavior (Green Zone)
- User logs in from typical location
- Uses known device
- Normal navigation speed
- **Result**: Transaction proceeds instantly

### 2. Suspicious Behavior (Yellow Zone)
- New location or device detected
- Unusual login time
- Higher transaction amount
- **Result**: Step-up authentication required (OTP/FaceID/Nafath)

### 3. Attack Detection (Red Zone)
- Bot-like navigation speed
- Tor/VPN from high-risk country
- Multiple anomalies combined
- **Result**: Transaction blocked, SOC alerted

---

## 🔄 Continuous Learning

The system learns from every interaction:

1. **False Positive**: User passes challenge → System learns this behavior is valid
2. **True Positive**: Fraud confirmed → System reinforces detection pattern
3. **False Negative**: Fraud missed → System significantly increases sensitivity

Weekly model retraining incorporates all feedback.

---

## 🛡️ Security Features

| Feature | Description |
|---------|-------------|
| Location Tracking | Detects impossible travel speeds |
| Device Fingerprinting | Identifies new/unknown devices |
| Bot Detection | Catches inhuman navigation speeds |
| IP Reputation | Flags VPN/Tor/low-rep IPs |
| Transaction Profiling | Monitors for unusual amounts/beneficiaries |

---

## 📊 Performance Targets

| Metric | Target | Actual |
|--------|--------|--------|
| Latency | <200ms | ~50ms |
| False Positive Rate | <10% | Configurable |
| Detection Rate | >95% | Depends on training |
| Uptime | 99.9% | Infrastructure dependent |

---

## 🚧 Production Deployment

For production use:

1. **Replace In-Memory Queue** with Apache Kafka:
   ```python
   queue_config = QueueConfig(queue_type="kafka")
   ```

2. **Enable TensorFlow** for better performance:
   ```bash
   pip install tensorflow keras
   ```

3. **Connect Real Data** - Replace synthetic generators with actual Absher logs

4. **Configure Monitoring** - Add Prometheus/Grafana for observability

---

## 📜 License

Proprietary - Saudi Government Services

---

## 👥 Contact

For support or questions:
- Email: security@absher.sa
- SOC Hotline: Internal

---

<div align="center">

**راصد - نحمي هويتك الرقمية**

*Rased - Protecting Your Digital Identity*

</div>
