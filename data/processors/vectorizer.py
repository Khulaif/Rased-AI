"""
Rased (راصد) - Feature Vectorizer
Converts raw events into LSTM-ready vector format.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import hashlib
import json
import os

# Add parent path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.settings import config


class FeatureVectorizer:
    """
    Converts raw user events into numerical vectors for LSTM processing.
    
    Features are grouped into:
    - Temporal: time-based features (hour, day, frequency)
    - Spatial: location-based features (lat/lng, IP reputation)
    - Device: device fingerprint features
    - Action: one-hot encoded user actions
    """
    
    def __init__(self):
        self.config = config.features
        
        # Action vocabulary for one-hot encoding
        self.action_vocab = {
            action: idx for idx, action in enumerate(self.config.action_features)
        }
        self.num_actions = len(self.action_vocab)
        
        # Device type vocabulary
        self.device_vocab = {"mobile": 0, "desktop": 1, "tablet": 2}
        
        # OS vocabulary
        self.os_vocab = {"iOS": 0, "Android": 1, "Windows": 2, "macOS": 3, "Linux": 4}
        
        # Country vocabulary (top countries)
        self.country_vocab = {"SA": 0, "AE": 1, "EG": 2, "TR": 3, "US": 4, "GB": 5}
        
        # Service type vocabulary
        self.service_vocab = {
            "vehicle_transfer": 0,
            "visa_renewal": 1,
            "passport_renewal": 2,
            "traffic_violation_payment": 3,
            "iqama_renewal": 4,
            "exit_reentry_visa": 5,
        }
        
        # Feature dimension calculation
        self.temporal_dim = 6
        self.spatial_dim = 8
        self.device_dim = 7
        self.action_dim = self.num_actions
        self.transaction_dim = 4
        
        self.total_dim = (
            self.temporal_dim + 
            self.spatial_dim + 
            self.device_dim + 
            self.action_dim + 
            self.transaction_dim
        )
        
        # Normalization stats (will be computed during fit)
        self.stats: Dict[str, Dict[str, float]] = {}
        self.is_fitted = False
        
    def fit(self, events: List[Dict]) -> 'FeatureVectorizer':
        """
        Compute normalization statistics from training data.
        
        Args:
            events: List of event dictionaries
        
        Returns:
            Self for method chaining
        """
        # Collect numeric features for normalization
        numeric_features = {
            "latitude": [],
            "longitude": [],
            "ip_reputation_score": [],
            "time_since_last_action": [],
            "transaction_amount": [],
            "screen_width": [],
            "screen_height": [],
            "battery_level": [],
        }
        
        for event in events:
            for feature in numeric_features:
                if feature in event and event[feature] is not None:
                    numeric_features[feature].append(float(event[feature]))
        
        # Compute mean and std for each feature
        for feature, values in numeric_features.items():
            if values:
                self.stats[feature] = {
                    "mean": np.mean(values),
                    "std": np.std(values) if np.std(values) > 0 else 1.0,
                    "min": np.min(values),
                    "max": np.max(values),
                }
            else:
                self.stats[feature] = {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0}
        
        self.is_fitted = True
        return self
    
    def normalize(self, value: float, feature: str) -> float:
        """Normalize a value using stored statistics."""
        if not self.is_fitted or feature not in self.stats:
            return value
        
        stats = self.stats[feature]
        # Z-score normalization
        return (value - stats["mean"]) / stats["std"]
    
    def vectorize_event(self, event: Dict) -> np.ndarray:
        """
        Convert a single event to a feature vector.
        
        Args:
            event: Event dictionary with all features
        
        Returns:
            NumPy array of shape (total_dim,)
        """
        vector = np.zeros(self.total_dim)
        idx = 0
        
        # === Temporal Features (6 dims) ===
        # Hour encoding (cyclical)
        hour = event.get("hour_of_day", 0)
        vector[idx] = np.sin(2 * np.pi * hour / 24)
        vector[idx + 1] = np.cos(2 * np.pi * hour / 24)
        idx += 2
        
        # Day of week encoding (cyclical)
        day = event.get("day_of_week", 0)
        vector[idx] = np.sin(2 * np.pi * day / 7)
        vector[idx + 1] = np.cos(2 * np.pi * day / 7)
        idx += 2
        
        # Time since last action (normalized)
        time_since = event.get("time_since_last_action", 0.0)
        vector[idx] = self.normalize(time_since, "time_since_last_action")
        idx += 1
        
        # Is weekend
        vector[idx] = 1.0 if day >= 5 else 0.0
        idx += 1
        
        # === Spatial Features (8 dims) ===
        # Latitude/Longitude (normalized)
        vector[idx] = self.normalize(event.get("latitude", 0.0), "latitude")
        vector[idx + 1] = self.normalize(event.get("longitude", 0.0), "longitude")
        idx += 2
        
        # IP reputation (normalized to 0-1)
        ip_rep = event.get("ip_reputation_score", 100.0)
        vector[idx] = ip_rep / 100.0
        idx += 1
        
        # Country (one-hot, 5 dims max)
        country = event.get("country_code", "SA")
        country_idx = self.country_vocab.get(country, len(self.country_vocab) - 1)
        country_vec = np.zeros(len(self.country_vocab))
        country_vec[country_idx] = 1.0
        vector[idx:idx + 3] = country_vec[:3]  # Only first 3 countries
        idx += 3
        
        # VPN and Tor flags
        vector[idx] = 1.0 if event.get("is_vpn", False) else 0.0
        vector[idx + 1] = 1.0 if event.get("is_tor", False) else 0.0
        idx += 2
        
        # === Device Features (7 dims) ===
        # Device type (one-hot, 3 dims)
        device = event.get("device_type", "mobile")
        device_idx = self.device_vocab.get(device, 0)
        device_vec = np.zeros(3)
        device_vec[device_idx] = 1.0
        vector[idx:idx + 3] = device_vec
        idx += 3
        
        # OS family (embedded as single value)
        os_family = event.get("os_family", "iOS")
        os_idx = self.os_vocab.get(os_family, 0)
        vector[idx] = os_idx / len(self.os_vocab)
        idx += 1
        
        # Screen size (normalized)
        vector[idx] = self.normalize(event.get("screen_width", 390), "screen_width")
        vector[idx + 1] = self.normalize(event.get("screen_height", 844), "screen_height")
        idx += 2
        
        # Battery level (normalized to 0-1)
        vector[idx] = event.get("battery_level", 100) / 100.0
        idx += 1
        
        # === Action Features (one-hot, num_actions dims) ===
        action = event.get("action", "login")
        action_idx = self.action_vocab.get(action, 0)
        action_vec = np.zeros(self.num_actions)
        action_vec[action_idx] = 1.0
        vector[idx:idx + self.num_actions] = action_vec
        idx += self.num_actions
        
        # === Transaction Features (4 dims) ===
        # Has transaction
        has_tx = 1.0 if event.get("service_type") else 0.0
        vector[idx] = has_tx
        idx += 1
        
        # Transaction amount (normalized, log-scaled)
        amount = event.get("transaction_amount", 0.0)
        if amount > 0:
            vector[idx] = np.log1p(amount) / 15.0  # Max ~3M SAR -> ~15
        idx += 1
        
        # Service type (embedded as single value)
        service = event.get("service_type")
        if service:
            service_idx = self.service_vocab.get(service, 0)
            vector[idx] = service_idx / len(self.service_vocab)
        idx += 1
        
        # New beneficiary flag
        beneficiary = event.get("beneficiary_id")
        if beneficiary and "new" in str(beneficiary).lower():
            vector[idx] = 1.0
        idx += 1
        
        return vector
    
    def vectorize_sequence(
        self, 
        events: List[Dict], 
        sequence_length: int = 10
    ) -> np.ndarray:
        """
        Convert a sequence of events to a 2D array for LSTM.
        
        Args:
            events: List of event dictionaries
            sequence_length: Fixed sequence length (will pad or truncate)
        
        Returns:
            NumPy array of shape (sequence_length, total_dim)
        """
        vectors = [self.vectorize_event(e) for e in events]
        
        # Pad or truncate to fixed length
        if len(vectors) < sequence_length:
            # Pad with zeros at the beginning
            padding = [np.zeros(self.total_dim)] * (sequence_length - len(vectors))
            vectors = padding + vectors
        elif len(vectors) > sequence_length:
            # Take most recent events
            vectors = vectors[-sequence_length:]
        
        return np.array(vectors)
    
    def prepare_training_data(
        self,
        events: List[Dict],
        sequence_length: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training.
        
        Creates sequences where:
        - X: sequence of past N events
        - y_action: next action (one-hot)
        - y_anomaly: is the sequence anomalous (binary)
        
        Args:
            events: Sorted list of all events
            sequence_length: Number of past events to use
        
        Returns:
            Tuple of (X, y_action, y_anomaly)
        """
        # Group events by user
        user_events: Dict[str, List[Dict]] = {}
        for event in events:
            user_id = event.get("user_id", "unknown")
            if user_id not in user_events:
                user_events[user_id] = []
            user_events[user_id].append(event)
        
        X_sequences = []
        y_actions = []
        y_anomalies = []
        
        for user_id, user_evts in user_events.items():
            # Sort by timestamp
            user_evts.sort(key=lambda e: e.get("timestamp", ""))
            
            # Create sliding window sequences
            for i in range(sequence_length, len(user_evts)):
                # Past events
                past_events = user_evts[i - sequence_length:i]
                sequence = self.vectorize_sequence(past_events, sequence_length)
                
                # Target: next action
                next_event = user_evts[i]
                next_action = next_event.get("action", "login")
                action_idx = self.action_vocab.get(next_action, 0)
                action_one_hot = np.zeros(self.num_actions)
                action_one_hot[action_idx] = 1.0
                
                # Target: is anomaly
                is_anomaly = 1.0 if next_event.get("is_anomaly", False) else 0.0
                
                X_sequences.append(sequence)
                y_actions.append(action_one_hot)
                y_anomalies.append(is_anomaly)
        
        return (
            np.array(X_sequences),
            np.array(y_actions),
            np.array(y_anomalies),
        )
    
    def save(self, path: str):
        """Save vectorizer state to file."""
        state = {
            "stats": self.stats,
            "action_vocab": self.action_vocab,
            "is_fitted": self.is_fitted,
            "total_dim": self.total_dim,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
    
    def load(self, path: str):
        """Load vectorizer state from file."""
        with open(path, "r") as f:
            state = json.load(f)
        self.stats = state["stats"]
        self.action_vocab = state["action_vocab"]
        self.is_fitted = state["is_fitted"]
        return self


if __name__ == "__main__":
    # Demo vectorization
    sample_event = {
        "event_id": "evt_001",
        "user_id": "user_001",
        "timestamp": "2024-01-15T10:30:00",
        "action": "view_dashboard",
        "hour_of_day": 10,
        "day_of_week": 0,
        "time_since_last_action": 5.0,
        "latitude": 24.7136,
        "longitude": 46.6753,
        "ip_reputation_score": 95.0,
        "country_code": "SA",
        "is_vpn": False,
        "is_tor": False,
        "device_type": "mobile",
        "os_family": "iOS",
        "screen_width": 390,
        "screen_height": 844,
        "battery_level": 85,
        "service_type": None,
        "transaction_amount": 0.0,
        "is_anomaly": False,
    }
    
    vectorizer = FeatureVectorizer()
    # Fit with dummy data
    vectorizer.fit([sample_event])
    
    vector = vectorizer.vectorize_event(sample_event)
    print(f"Vector shape: {vector.shape}")
    print(f"Vector: {vector}")
    print(f"Total dimensions: {vectorizer.total_dim}")
