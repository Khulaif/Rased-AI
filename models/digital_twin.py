"""
Rased (راصد) - Digital Twin Manager
Manages user-specific behavioral embeddings (Digital Twins).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os

# Add parent path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import config


@dataclass
class DigitalTwin:
    """
    Digital Twin representation of a user's behavioral profile.
    
    This captures the learned patterns of normal behavior for each user,
    enabling the detection of anomalies when actual behavior deviates.
    """
    user_id: str
    embedding: np.ndarray
    
    # Behavioral statistics
    avg_session_duration: float = 0.0
    avg_actions_per_session: float = 0.0
    avg_navigation_speed: float = 2.0  # actions per minute
    
    # Temporal patterns
    typical_hours: List[int] = field(default_factory=list)
    typical_days: List[int] = field(default_factory=list)
    
    # Spatial patterns
    typical_locations: List[Tuple[float, float]] = field(default_factory=list)
    typical_countries: List[str] = field(default_factory=list)
    
    # Device patterns
    known_devices: List[str] = field(default_factory=list)
    primary_device_type: str = "mobile"
    primary_os: str = "iOS"
    
    # Transaction patterns
    avg_transaction_amount: float = 0.0
    max_transaction_amount: float = 0.0
    typical_beneficiaries: List[str] = field(default_factory=list)
    
    # History tracking
    total_sessions: int = 0
    total_actions: int = 0
    last_activity: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Risk history
    historical_risk_scores: List[float] = field(default_factory=list)
    false_positive_count: int = 0
    confirmed_fraud_count: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "embedding": self.embedding.tolist(),
            "avg_session_duration": self.avg_session_duration,
            "avg_actions_per_session": self.avg_actions_per_session,
            "avg_navigation_speed": self.avg_navigation_speed,
            "typical_hours": self.typical_hours,
            "typical_days": self.typical_days,
            "typical_locations": self.typical_locations,
            "typical_countries": self.typical_countries,
            "known_devices": self.known_devices,
            "primary_device_type": self.primary_device_type,
            "primary_os": self.primary_os,
            "avg_transaction_amount": self.avg_transaction_amount,
            "max_transaction_amount": self.max_transaction_amount,
            "typical_beneficiaries": self.typical_beneficiaries,
            "total_sessions": self.total_sessions,
            "total_actions": self.total_actions,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "historical_risk_scores": self.historical_risk_scores[-100:],  # Keep last 100
            "false_positive_count": self.false_positive_count,
            "confirmed_fraud_count": self.confirmed_fraud_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DigitalTwin':
        """Create from dictionary."""
        twin = cls(
            user_id=data["user_id"],
            embedding=np.array(data["embedding"]),
        )
        twin.avg_session_duration = data.get("avg_session_duration", 0.0)
        twin.avg_actions_per_session = data.get("avg_actions_per_session", 0.0)
        twin.avg_navigation_speed = data.get("avg_navigation_speed", 2.0)
        twin.typical_hours = data.get("typical_hours", [])
        twin.typical_days = data.get("typical_days", [])
        twin.typical_locations = [tuple(loc) for loc in data.get("typical_locations", [])]
        twin.typical_countries = data.get("typical_countries", [])
        twin.known_devices = data.get("known_devices", [])
        twin.primary_device_type = data.get("primary_device_type", "mobile")
        twin.primary_os = data.get("primary_os", "iOS")
        twin.avg_transaction_amount = data.get("avg_transaction_amount", 0.0)
        twin.max_transaction_amount = data.get("max_transaction_amount", 0.0)
        twin.typical_beneficiaries = data.get("typical_beneficiaries", [])
        twin.total_sessions = data.get("total_sessions", 0)
        twin.total_actions = data.get("total_actions", 0)
        twin.last_activity = datetime.fromisoformat(data["last_activity"]) if data.get("last_activity") else None
        twin.created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now()
        twin.last_updated = datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else datetime.now()
        twin.historical_risk_scores = data.get("historical_risk_scores", [])
        twin.false_positive_count = data.get("false_positive_count", 0)
        twin.confirmed_fraud_count = data.get("confirmed_fraud_count", 0)
        
        return twin


class DigitalTwinManager:
    """
    Manages Digital Twin profiles for all users.
    
    Key responsibilities:
    1. Create and update user Digital Twins
    2. Compare current behavior against Digital Twin
    3. Calculate behavioral deviation scores
    4. Update embeddings based on feedback
    """
    
    def __init__(self, embedding_dim: int = 64, storage_path: str = "models/twins"):
        """
        Initialize Digital Twin manager.
        
        Args:
            embedding_dim: Dimension of user embedding vectors
            storage_path: Path to store Digital Twin data
        """
        self.embedding_dim = embedding_dim
        self.storage_path = storage_path
        self.twins: Dict[str, DigitalTwin] = {}
        
        # Contextual weights for deviation calculation
        self.deviation_weights = {
            "location": 0.25,
            "device": 0.20,
            "timing": 0.15,
            "navigation_speed": 0.20,
            "transaction": 0.20,
        }
        
        # Load existing twins
        self._load_twins()
    
    def _load_twins(self):
        """Load saved Digital Twins from storage."""
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path, exist_ok=True)
            return
        
        twins_file = os.path.join(self.storage_path, "twins.json")
        if os.path.exists(twins_file):
            with open(twins_file, "r") as f:
                data = json.load(f)
            
            for user_id, twin_data in data.items():
                self.twins[user_id] = DigitalTwin.from_dict(twin_data)
            
            print(f"Loaded {len(self.twins)} Digital Twins")
    
    def save_twins(self):
        """Save all Digital Twins to storage."""
        os.makedirs(self.storage_path, exist_ok=True)
        
        data = {
            user_id: twin.to_dict()
            for user_id, twin in self.twins.items()
        }
        
        twins_file = os.path.join(self.storage_path, "twins.json")
        with open(twins_file, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(self.twins)} Digital Twins")
    
    def get_or_create_twin(self, user_id: str) -> DigitalTwin:
        """
        Get existing Digital Twin or create new one.
        
        Args:
            user_id: User identifier
        
        Returns:
            DigitalTwin for the user
        """
        if user_id not in self.twins:
            # Create new Digital Twin with random initial embedding
            embedding = np.random.randn(self.embedding_dim) * 0.1
            self.twins[user_id] = DigitalTwin(
                user_id=user_id,
                embedding=embedding,
            )
        
        return self.twins[user_id]
    
    def update_twin(
        self,
        user_id: str,
        event: Dict,
        is_session_start: bool = False,
    ) -> DigitalTwin:
        """
        Update Digital Twin with new event data.
        
        Args:
            user_id: User identifier
            event: Event data dictionary
            is_session_start: Whether this is the start of a new session
        
        Returns:
            Updated DigitalTwin
        """
        twin = self.get_or_create_twin(user_id)
        
        # Update activity tracking
        twin.total_actions += 1
        twin.last_activity = datetime.now()
        twin.last_updated = datetime.now()
        
        if is_session_start:
            twin.total_sessions += 1
        
        # Update temporal patterns
        hour = event.get("hour_of_day", 12)
        day = event.get("day_of_week", 0)
        
        if hour not in twin.typical_hours:
            if len(twin.typical_hours) < 12:
                twin.typical_hours.append(hour)
        
        if day not in twin.typical_days:
            if len(twin.typical_days) < 7:
                twin.typical_days.append(day)
        
        # Update spatial patterns
        lat = event.get("latitude", 0.0)
        lng = event.get("longitude", 0.0)
        country = event.get("country_code", "SA")
        
        location = (lat, lng)
        if not self._location_exists(twin.typical_locations, location):
            if len(twin.typical_locations) < 10:
                twin.typical_locations.append(location)
        
        if country not in twin.typical_countries:
            if len(twin.typical_countries) < 5:
                twin.typical_countries.append(country)
        
        # Update device patterns
        device_hash = event.get("device_hash", "")
        device_type = event.get("device_type", "mobile")
        os_family = event.get("os_family", "iOS")
        
        if device_hash and device_hash not in twin.known_devices:
            if len(twin.known_devices) < 5:
                twin.known_devices.append(device_hash)
        
        # Update primary device (most common)
        twin.primary_device_type = device_type
        twin.primary_os = os_family
        
        # Update navigation speed (rolling average)
        time_since_last = event.get("time_since_last_action", 30.0)
        if time_since_last > 0:
            current_speed = 60.0 / time_since_last
            twin.avg_navigation_speed = (
                0.9 * twin.avg_navigation_speed + 0.1 * current_speed
            )
        
        # Update transaction patterns
        amount = event.get("transaction_amount", 0.0)
        beneficiary = event.get("beneficiary_id")
        
        if amount > 0:
            twin.avg_transaction_amount = (
                0.9 * twin.avg_transaction_amount + 0.1 * amount
            )
            twin.max_transaction_amount = max(twin.max_transaction_amount, amount)
        
        if beneficiary and beneficiary not in twin.typical_beneficiaries:
            if len(twin.typical_beneficiaries) < 20:
                twin.typical_beneficiaries.append(beneficiary)
        
        return twin
    
    def _location_exists(
        self, 
        locations: List[Tuple[float, float]], 
        new_location: Tuple[float, float],
        threshold_km: float = 50.0
    ) -> bool:
        """Check if a similar location already exists."""
        for lat, lng in locations:
            distance = self._haversine_distance(lat, lng, new_location[0], new_location[1])
            if distance < threshold_km:
                return True
        return False
    
    def _haversine_distance(
        self, 
        lat1: float, 
        lng1: float, 
        lat2: float, 
        lng2: float
    ) -> float:
        """Calculate distance between two points in km."""
        R = 6371  # Earth's radius in km
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lng = np.radians(lng2 - lng1)
        
        a = (np.sin(delta_lat / 2) ** 2 +
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lng / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c
    
    def calculate_deviation(
        self,
        user_id: str,
        event: Dict,
    ) -> Dict[str, float]:
        """
        Calculate behavioral deviation from Digital Twin.
        
        Args:
            user_id: User identifier
            event: Current event data
        
        Returns:
            Dictionary with deviation scores for each category
        """
        twin = self.get_or_create_twin(user_id)
        deviations = {}
        
        # === Location Deviation ===
        lat = event.get("latitude", 0.0)
        lng = event.get("longitude", 0.0)
        country = event.get("country_code", "SA")
        
        location_score = 0.0
        
        # Check if country is known
        if country not in twin.typical_countries:
            location_score += 0.5
        
        # Check distance from known locations
        min_distance = float('inf')
        for known_lat, known_lng in twin.typical_locations:
            distance = self._haversine_distance(lat, lng, known_lat, known_lng)
            min_distance = min(min_distance, distance)
        
        if min_distance != float('inf'):
            # Normalize: > 1000km is fully anomalous
            location_score += min(0.5, min_distance / 2000.0)
        
        deviations["location"] = min(1.0, location_score)
        
        # === Device Deviation ===
        device_hash = event.get("device_hash", "")
        device_type = event.get("device_type", "mobile")
        os_family = event.get("os_family", "iOS")
        
        device_score = 0.0
        
        if device_hash not in twin.known_devices:
            device_score += 0.4
        
        if device_type != twin.primary_device_type:
            device_score += 0.3
        
        if os_family != twin.primary_os:
            device_score += 0.3
        
        deviations["device"] = min(1.0, device_score)
        
        # === Timing Deviation ===
        hour = event.get("hour_of_day", 12)
        day = event.get("day_of_week", 0)
        
        timing_score = 0.0
        
        if hour not in twin.typical_hours:
            # Calculate distance to nearest typical hour
            if twin.typical_hours:
                min_hour_diff = min(
                    min(abs(hour - h), 24 - abs(hour - h))
                    for h in twin.typical_hours
                )
                timing_score += min(0.5, min_hour_diff / 12.0)
            else:
                timing_score += 0.1  # New user, slight penalty
        
        if day not in twin.typical_days:
            timing_score += 0.3
        
        deviations["timing"] = min(1.0, timing_score)
        
        # === Navigation Speed Deviation ===
        time_since_last = event.get("time_since_last_action", 30.0)
        
        speed_score = 0.0
        
        if time_since_last > 0:
            current_speed = 60.0 / time_since_last
            
            # Compare to typical speed
            speed_ratio = current_speed / max(0.1, twin.avg_navigation_speed)
            
            if speed_ratio > 10:  # 10x faster than normal = bot behavior
                speed_score = 1.0
            elif speed_ratio > 3:  # 3x faster = suspicious
                speed_score = 0.5
            elif speed_ratio < 0.1:  # 10x slower = unusual
                speed_score = 0.3
        
        deviations["navigation_speed"] = speed_score
        
        # === Transaction Deviation ===
        amount = event.get("transaction_amount", 0.0)
        beneficiary = event.get("beneficiary_id")
        
        transaction_score = 0.0
        
        if amount > 0:
            # Check if amount is significantly higher than usual
            if twin.max_transaction_amount > 0:
                amount_ratio = amount / twin.max_transaction_amount
                if amount_ratio > 5:  # 5x higher than max
                    transaction_score += 0.5
                elif amount_ratio > 2:  # 2x higher
                    transaction_score += 0.3
            
            # Check if new beneficiary
            if beneficiary and beneficiary not in twin.typical_beneficiaries:
                transaction_score += 0.3
        
        deviations["transaction"] = min(1.0, transaction_score)
        
        # === Composite Score ===
        weighted_sum = sum(
            deviations[category] * weight
            for category, weight in self.deviation_weights.items()
        )
        deviations["composite"] = weighted_sum
        
        return deviations
    
    def update_embedding(
        self,
        user_id: str,
        adjustment: np.ndarray,
        learning_rate: float = 0.01,
    ):
        """
        Update user embedding based on feedback.
        
        Args:
            user_id: User identifier
            adjustment: Embedding adjustment vector
            learning_rate: Learning rate for update
        """
        twin = self.get_or_create_twin(user_id)
        twin.embedding += learning_rate * adjustment
        
        # Normalize embedding
        norm = np.linalg.norm(twin.embedding)
        if norm > 0:
            twin.embedding /= norm
    
    def record_risk_score(self, user_id: str, score: float):
        """Record a risk score for the user."""
        twin = self.get_or_create_twin(user_id)
        twin.historical_risk_scores.append(score)
        
        # Keep only last 100 scores
        if len(twin.historical_risk_scores) > 100:
            twin.historical_risk_scores = twin.historical_risk_scores[-100:]
    
    def record_false_positive(self, user_id: str):
        """Record a false positive (challenge passed) for the user."""
        twin = self.get_or_create_twin(user_id)
        twin.false_positive_count += 1
    
    def record_confirmed_fraud(self, user_id: str):
        """Record confirmed fraud for the user."""
        twin = self.get_or_create_twin(user_id)
        twin.confirmed_fraud_count += 1
    
    def get_user_risk_profile(self, user_id: str) -> Dict:
        """Get risk profile summary for a user."""
        twin = self.get_or_create_twin(user_id)
        
        avg_risk = (
            np.mean(twin.historical_risk_scores)
            if twin.historical_risk_scores
            else 0.0
        )
        
        return {
            "user_id": user_id,
            "total_sessions": twin.total_sessions,
            "total_actions": twin.total_actions,
            "average_risk_score": avg_risk,
            "false_positive_rate": (
                twin.false_positive_count / max(1, len(twin.historical_risk_scores))
            ),
            "confirmed_fraud_count": twin.confirmed_fraud_count,
            "known_devices_count": len(twin.known_devices),
            "known_locations_count": len(twin.typical_locations),
            "account_age_days": (
                (datetime.now() - twin.created_at).days
            ),
        }


if __name__ == "__main__":
    # Demo Digital Twin
    manager = DigitalTwinManager()
    
    # Create a twin
    twin = manager.get_or_create_twin("user_001")
    print(f"Created twin for {twin.user_id}")
    print(f"Embedding shape: {twin.embedding.shape}")
    
    # Simulate events
    event = {
        "hour_of_day": 10,
        "day_of_week": 1,
        "latitude": 24.7136,
        "longitude": 46.6753,
        "country_code": "SA",
        "device_hash": "abc123",
        "device_type": "mobile",
        "os_family": "iOS",
        "time_since_last_action": 30.0,
        "transaction_amount": 0.0,
    }
    
    # Update twin
    manager.update_twin("user_001", event, is_session_start=True)
    
    # Calculate deviation for a normal event
    deviations = manager.calculate_deviation("user_001", event)
    print(f"\nNormal event deviations: {deviations}")
    
    # Calculate deviation for anomalous event
    anomalous_event = {
        **event,
        "latitude": 55.7558,  # Moscow
        "longitude": 37.6173,
        "country_code": "RU",
        "device_type": "desktop",
        "os_family": "Windows",
        "device_hash": "xyz999",
        "time_since_last_action": 0.5,  # Very fast
    }
    
    deviations = manager.calculate_deviation("user_001", anomalous_event)
    print(f"\nAnomalous event deviations: {deviations}")
