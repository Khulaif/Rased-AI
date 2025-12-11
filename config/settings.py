"""
Rased (راصد) - Configuration Settings
Central configuration for all system parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List
from enum import Enum


class RiskLevel(Enum):
    """Risk classification levels."""
    GREEN = "green"      # 0-30: Normal behavior
    YELLOW = "yellow"    # 31-70: Suspicious, requires step-up auth
    RED = "red"          # 71-100: Critical, block immediately


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    
    # Temporal features
    temporal_features: List[str] = field(default_factory=lambda: [
        "hour_of_day",           # 0-23
        "day_of_week",           # 0-6
        "login_frequency",       # Logins per day (rolling average)
        "session_duration",      # Minutes
        "time_since_last_action",# Seconds
        "navigation_speed",      # Actions per minute
    ])
    
    # Spatial features
    spatial_features: List[str] = field(default_factory=lambda: [
        "latitude",
        "longitude", 
        "ip_reputation_score",   # 0-100 (100 = trusted)
        "country_code",
        "city_hash",
        "travel_speed_kmh",      # Speed between last two logins
        "is_vpn",                # Boolean
        "is_tor",                # Boolean
    ])
    
    # Device fingerprinting features
    device_features: List[str] = field(default_factory=lambda: [
        "device_type",           # mobile, desktop, tablet
        "os_family",             # iOS, Android, Windows, macOS
        "os_version",
        "browser_family",
        "screen_width",
        "screen_height",
        "device_hash",           # Unique device fingerprint
        "battery_level",         # 0-100 (if available)
        "is_charging",           # Boolean
    ])
    
    # Ownership transfer features (for Absher services)
    ownership_features: List[str] = field(default_factory=lambda: [
        "service_type",             # Vehicle Transfer, Property Transfer, etc.
        "asset_type",               # vehicle, property, commercial_register
        "asset_value",              # Estimated value in SAR
        "new_owner_id",             # ID of the person receiving ownership
        "new_owner_known",          # Is this a known contact?
        "transfer_frequency",       # Similar transfers in last 30 days
    ])
    
    # Action sequence features (for LSTM)
    action_features: List[str] = field(default_factory=lambda: [
        "login",
        "view_dashboard",
        "view_vehicles",
        "view_traffic_violations",
        "view_visas",
        "view_passports",
        "view_properties",
        "initiate_ownership_transfer",
        "confirm_ownership_transfer",
        "cancel_ownership_transfer",
        "change_settings",
        "logout",
    ])


@dataclass
class ModelConfig:
    """LSTM model configuration."""
    
    # Sequence parameters
    sequence_length: int = 10          # Number of past actions to consider
    embedding_dim: int = 64            # User embedding dimension
    
    # LSTM architecture
    lstm_units_1: int = 128            # First LSTM layer
    lstm_units_2: int = 64             # Second LSTM layer
    dense_units: int = 32              # Dense layer before output
    dropout_rate: float = 0.3
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    validation_split: float = 0.2
    
    # Anomaly detection
    anomaly_threshold: float = 0.5     # Prediction error threshold
    

@dataclass
class RiskConfig:
    """Risk scoring configuration."""
    
    # Thresholds
    green_max: int = 30
    yellow_max: int = 70
    red_min: int = 71
    
    # Weight distribution for composite score
    weights: Dict[str, float] = field(default_factory=lambda: {
        "model_prediction_error": 0.40,  # LSTM prediction deviation
        "contextual_anomaly": 0.30,      # Device, location changes
        "historical_risk": 0.30,         # Past suspicious activities
    })
    
    # Step-up authentication options
    step_up_methods: List[str] = field(default_factory=lambda: [
        "otp_sms",
        "otp_email", 
        "face_id",
        "nafath_verification",
        "security_questions",
    ])
    
    def get_risk_level(self, score: int) -> RiskLevel:
        """Classify risk score into level."""
        if score <= self.green_max:
            return RiskLevel.GREEN
        elif score <= self.yellow_max:
            return RiskLevel.YELLOW
        else:
            return RiskLevel.RED


@dataclass
class QueueConfig:
    """Message queue configuration."""
    
    queue_type: str = "memory"         # memory, kafka, rabbitmq
    max_queue_size: int = 10000
    processing_timeout_ms: int = 200   # Max latency target
    
    # Kafka settings (for production)
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "rased_events"
    kafka_consumer_group: str = "rased_inference"
    
    # RabbitMQ settings (alternative)
    rabbitmq_host: str = "localhost"
    rabbitmq_port: int = 5672
    rabbitmq_queue: str = "rased_events"


@dataclass
class FeedbackConfig:
    """Continuous learning configuration."""
    
    # Learning rates for feedback
    false_positive_learning_rate: float = 0.01
    true_positive_learning_rate: float = 0.05
    
    # Retraining schedule
    retrain_interval_days: int = 7     # Weekly retraining
    min_samples_for_retrain: int = 1000
    
    # Model versioning
    max_model_versions: int = 5        # Keep last N versions


@dataclass 
class RasedConfig:
    """Master configuration for Rased system."""
    
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    
    # System settings
    debug_mode: bool = True
    log_level: str = "INFO"
    
    # Paths
    model_save_path: str = "models/trained"
    data_path: str = "data"


# Global configuration instance
config = RasedConfig()
