"""
Rased (راصد) - Synthetic Data Generator
Generates realistic user behavior sequences for training and testing.
"""

import random
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import json


@dataclass
class UserProfile:
    """Simulated user profile with consistent behavior patterns."""
    user_id: str
    name: str
    
    # Typical behavior patterns
    typical_login_hours: List[int] = field(default_factory=lambda: [8, 9, 10, 18, 19, 20])
    typical_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Weekdays
    typical_city: str = "Riyadh"
    typical_latitude: float = 24.7136
    typical_longitude: float = 46.6753
    
    # Device info
    device_type: str = "mobile"
    os_family: str = "iOS"
    os_version: str = "17.0"
    device_hash: str = ""
    screen_width: int = 390
    screen_height: int = 844
    
    # Behavior characteristics
    avg_session_duration: int = 15  # minutes
    avg_actions_per_session: int = 8
    navigation_speed: float = 2.0  # actions per minute (normal human speed)
    
    # Common action sequences
    common_sequences: List[List[str]] = field(default_factory=lambda: [
        ["login", "view_dashboard", "view_vehicles", "logout"],
        ["login", "view_dashboard", "view_traffic_violations", "logout"],
        ["login", "view_dashboard", "view_visas", "logout"],
    ])
    
    def __post_init__(self):
        if not self.device_hash:
            self.device_hash = hashlib.md5(
                f"{self.user_id}{self.device_type}{self.os_family}".encode()
            ).hexdigest()[:16]


@dataclass
class UserEvent:
    """Single user action event."""
    event_id: str
    user_id: str
    timestamp: datetime
    action: str
    
    # Temporal features
    hour_of_day: int = 0
    day_of_week: int = 0
    time_since_last_action: float = 0.0  # seconds
    
    # Spatial features
    latitude: float = 0.0
    longitude: float = 0.0
    ip_address: str = ""
    ip_reputation_score: float = 100.0
    country_code: str = "SA"
    city: str = ""
    is_vpn: bool = False
    is_tor: bool = False
    
    # Device features  
    device_type: str = ""
    os_family: str = ""
    os_version: str = ""
    device_hash: str = ""
    screen_width: int = 0
    screen_height: int = 0
    battery_level: int = 100
    is_charging: bool = False
    
    # Transaction features (if applicable)
    service_type: Optional[str] = None
    transaction_amount: float = 0.0
    beneficiary_id: Optional[str] = None
    
    # Labels
    is_anomaly: bool = False
    anomaly_type: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "hour_of_day": self.hour_of_day,
            "day_of_week": self.day_of_week,
            "time_since_last_action": self.time_since_last_action,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "ip_address": self.ip_address,
            "ip_reputation_score": self.ip_reputation_score,
            "country_code": self.country_code,
            "city": self.city,
            "is_vpn": self.is_vpn,
            "is_tor": self.is_tor,
            "device_type": self.device_type,
            "os_family": self.os_family,
            "os_version": self.os_version,
            "device_hash": self.device_hash,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "battery_level": self.battery_level,
            "is_charging": self.is_charging,
            "service_type": self.service_type,
            "transaction_amount": self.transaction_amount,
            "beneficiary_id": self.beneficiary_id,
            "is_anomaly": self.is_anomaly,
            "anomaly_type": self.anomaly_type,
        }


class SyntheticDataGenerator:
    """
    Generates synthetic user behavior data for training the LSTM model.
    
    Creates realistic patterns with:
    - 90% normal behavior
    - 10% anomalous behavior (for training detection)
    """
    
    # Saudi Arabian cities with coordinates
    SA_CITIES = {
        "Riyadh": (24.7136, 46.6753),
        "Jeddah": (21.4858, 39.1925),
        "Mecca": (21.4225, 39.8262),
        "Medina": (24.5247, 39.5692),
        "Dammam": (26.4207, 50.0888),
        "Khobar": (26.2172, 50.1971),
    }
    
    # Foreign cities (for anomaly simulation)
    FOREIGN_CITIES = {
        "Dubai": (25.2048, 55.2708, "AE"),
        "Cairo": (30.0444, 31.2357, "EG"),
        "Istanbul": (41.0082, 28.9784, "TR"),
        "Moscow": (55.7558, 37.6173, "RU"),
        "Lagos": (6.5244, 3.3792, "NG"),
    }
    
    ACTIONS = [
        "login",
        "view_dashboard",
        "view_vehicles",
        "view_traffic_violations",
        "view_visas",
        "view_passports",
        "initiate_transfer",
        "confirm_transfer",
        "cancel_transfer",
        "change_settings",
        "logout",
    ]
    
    SERVICE_TYPES = [
        "vehicle_transfer",
        "visa_renewal",
        "passport_renewal",
        "traffic_violation_payment",
        "iqama_renewal",
        "exit_reentry_visa",
    ]
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        self.event_counter = 0
        
    def generate_user_profiles(self, num_users: int = 100) -> List[UserProfile]:
        """Generate diverse user profiles."""
        profiles = []
        
        device_types = ["mobile", "desktop", "tablet"]
        os_options = {
            "mobile": [("iOS", "17.0"), ("iOS", "16.5"), ("Android", "14"), ("Android", "13")],
            "desktop": [("Windows", "11"), ("Windows", "10"), ("macOS", "14.0")],
            "tablet": [("iOS", "17.0"), ("Android", "14")],
        }
        
        screen_sizes = {
            "mobile": [(390, 844), (393, 873), (360, 800)],
            "desktop": [(1920, 1080), (2560, 1440), (1366, 768)],
            "tablet": [(1024, 768), (1194, 834)],
        }
        
        for i in range(num_users):
            city = random.choice(list(self.SA_CITIES.keys()))
            lat, lng = self.SA_CITIES[city]
            
            device = random.choice(device_types)
            os_family, os_version = random.choice(os_options[device])
            screen = random.choice(screen_sizes[device])
            
            # Vary login patterns
            if random.random() < 0.3:
                # Night owl user
                login_hours = [20, 21, 22, 23, 0, 1]
            elif random.random() < 0.5:
                # Early bird
                login_hours = [5, 6, 7, 8, 9]
            else:
                # Normal office hours
                login_hours = [8, 9, 10, 14, 15, 16, 17, 18]
            
            profile = UserProfile(
                user_id=f"user_{i:05d}",
                name=f"User {i}",
                typical_login_hours=login_hours,
                typical_days=random.sample(range(7), k=random.randint(4, 7)),
                typical_city=city,
                typical_latitude=lat + random.uniform(-0.1, 0.1),
                typical_longitude=lng + random.uniform(-0.1, 0.1),
                device_type=device,
                os_family=os_family,
                os_version=os_version,
                screen_width=screen[0],
                screen_height=screen[1],
                avg_session_duration=random.randint(5, 30),
                avg_actions_per_session=random.randint(4, 15),
                navigation_speed=random.uniform(1.0, 3.0),
            )
            profiles.append(profile)
            
        return profiles
    
    def generate_normal_session(
        self, 
        profile: UserProfile, 
        base_time: datetime
    ) -> List[UserEvent]:
        """Generate a normal behavior session for a user."""
        events = []
        
        # Determine session length
        num_actions = random.randint(
            max(3, profile.avg_actions_per_session - 3),
            profile.avg_actions_per_session + 3
        )
        
        # Start with login
        current_time = base_time
        current_action = "login"
        
        for i in range(num_actions):
            # Calculate time since last action
            if i == 0:
                time_since_last = 0.0
            else:
                # Normal human navigation speed
                avg_time = 60.0 / profile.navigation_speed
                time_since_last = max(2.0, np.random.normal(avg_time, avg_time * 0.3))
                current_time += timedelta(seconds=time_since_last)
            
            # Generate event
            event = self._create_event(
                profile=profile,
                timestamp=current_time,
                action=current_action,
                time_since_last=time_since_last,
                is_anomaly=False,
            )
            events.append(event)
            
            # Determine next action
            if i == num_actions - 1:
                current_action = "logout"
            elif current_action == "login":
                current_action = "view_dashboard"
            elif current_action == "view_dashboard":
                current_action = random.choice([
                    "view_vehicles", "view_traffic_violations", 
                    "view_visas", "view_passports", "change_settings"
                ])
            elif current_action in ["view_vehicles"]:
                if random.random() < 0.3:
                    current_action = "initiate_transfer"
                else:
                    current_action = random.choice(["view_dashboard", "logout"])
            elif current_action == "initiate_transfer":
                current_action = random.choice(["confirm_transfer", "cancel_transfer"])
            else:
                current_action = random.choice(["view_dashboard", "logout"])
        
        return events
    
    def generate_anomalous_session(
        self, 
        profile: UserProfile, 
        base_time: datetime,
        anomaly_type: str = "random"
    ) -> List[UserEvent]:
        """Generate anomalous behavior session."""
        
        if anomaly_type == "random":
            anomaly_type = random.choice([
                "location_change",
                "device_change", 
                "bot_behavior",
                "unusual_time",
                "rapid_high_value_transfer",
            ])
        
        events = []
        num_actions = random.randint(3, 8)
        current_time = base_time
        
        for i in range(num_actions):
            if i == 0:
                time_since_last = 0.0
                current_action = "login"
            else:
                if anomaly_type == "bot_behavior":
                    # Unnaturally fast navigation
                    time_since_last = random.uniform(0.1, 0.5)
                else:
                    time_since_last = random.uniform(5.0, 30.0)
                current_time += timedelta(seconds=time_since_last)
                
                if i == num_actions - 1:
                    current_action = "logout"
                elif anomaly_type == "rapid_high_value_transfer":
                    if i == 1:
                        current_action = "view_vehicles"
                    elif i == 2:
                        current_action = "initiate_transfer"
                    else:
                        current_action = "confirm_transfer"
                else:
                    current_action = random.choice(self.ACTIONS[1:-1])
            
            # Create event with anomaly modifications
            event = self._create_anomalous_event(
                profile=profile,
                timestamp=current_time,
                action=current_action,
                time_since_last=time_since_last,
                anomaly_type=anomaly_type,
            )
            events.append(event)
        
        return events
    
    def _create_event(
        self,
        profile: UserProfile,
        timestamp: datetime,
        action: str,
        time_since_last: float,
        is_anomaly: bool = False,
    ) -> UserEvent:
        """Create a normal user event."""
        self.event_counter += 1
        
        # Small variations in location (GPS drift)
        lat = profile.typical_latitude + random.uniform(-0.01, 0.01)
        lng = profile.typical_longitude + random.uniform(-0.01, 0.01)
        
        # Generate consistent IP for the city
        ip_base = hash(profile.typical_city) % 200 + 1
        ip_address = f"{ip_base}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
        
        # Transaction details if applicable
        service_type = None
        amount = 0.0
        beneficiary = None
        
        if action in ["initiate_transfer", "confirm_transfer"]:
            service_type = random.choice(self.SERVICE_TYPES)
            amount = random.uniform(100, 5000)
            beneficiary = f"beneficiary_{random.randint(1, 20):03d}"
        
        return UserEvent(
            event_id=f"evt_{self.event_counter:010d}",
            user_id=profile.user_id,
            timestamp=timestamp,
            action=action,
            hour_of_day=timestamp.hour,
            day_of_week=timestamp.weekday(),
            time_since_last_action=time_since_last,
            latitude=lat,
            longitude=lng,
            ip_address=ip_address,
            ip_reputation_score=random.uniform(85, 100),
            country_code="SA",
            city=profile.typical_city,
            is_vpn=False,
            is_tor=False,
            device_type=profile.device_type,
            os_family=profile.os_family,
            os_version=profile.os_version,
            device_hash=profile.device_hash,
            screen_width=profile.screen_width,
            screen_height=profile.screen_height,
            battery_level=random.randint(20, 100),
            is_charging=random.random() < 0.3,
            service_type=service_type,
            transaction_amount=amount,
            beneficiary_id=beneficiary,
            is_anomaly=is_anomaly,
            anomaly_type=None,
        )
    
    def _create_anomalous_event(
        self,
        profile: UserProfile,
        timestamp: datetime,
        action: str,
        time_since_last: float,
        anomaly_type: str,
    ) -> UserEvent:
        """Create an anomalous event based on anomaly type."""
        
        # Start with normal event
        event = self._create_event(
            profile=profile,
            timestamp=timestamp,
            action=action,
            time_since_last=time_since_last,
            is_anomaly=True,
        )
        
        event.anomaly_type = anomaly_type
        
        if anomaly_type == "location_change":
            # Sudden location change to foreign country
            city_name = random.choice(list(self.FOREIGN_CITIES.keys()))
            lat, lng, country = self.FOREIGN_CITIES[city_name]
            event.latitude = lat + random.uniform(-0.1, 0.1)
            event.longitude = lng + random.uniform(-0.1, 0.1)
            event.country_code = country
            event.city = city_name
            event.ip_reputation_score = random.uniform(30, 60)
            
        elif anomaly_type == "device_change":
            # Completely different device
            event.device_type = "desktop" if profile.device_type == "mobile" else "mobile"
            event.os_family = "Windows" if profile.os_family in ["iOS", "Android"] else "Android"
            event.os_version = "10" if event.os_family == "Windows" else "13"
            event.device_hash = hashlib.md5(str(random.random()).encode()).hexdigest()[:16]
            event.screen_width = 1920
            event.screen_height = 1080
            
        elif anomaly_type == "bot_behavior":
            # Impossibly fast navigation
            event.time_since_last_action = time_since_last  # Already set to very low value
            
        elif anomaly_type == "unusual_time":
            # Unusual hour (opposite of normal pattern)
            if 8 in profile.typical_login_hours:
                # Normal day user, unusual is 3 AM
                new_hour = 3
            else:
                new_hour = 10
            event.hour_of_day = new_hour
            
        elif anomaly_type == "rapid_high_value_transfer":
            if action == "confirm_transfer":
                event.transaction_amount = random.uniform(50000, 200000)
                event.beneficiary_id = f"new_beneficiary_{random.randint(1000, 9999)}"
            event.is_vpn = True
            event.ip_reputation_score = random.uniform(20, 50)
        
        return event
    
    def generate_training_data(
        self,
        num_users: int = 100,
        sessions_per_user: int = 50,
        anomaly_ratio: float = 0.1,
        time_range_days: int = 180,
    ) -> Tuple[List[UserProfile], List[UserEvent]]:
        """
        Generate complete training dataset.
        
        Args:
            num_users: Number of unique users to simulate
            sessions_per_user: Average sessions per user
            anomaly_ratio: Ratio of anomalous sessions (default 10%)
            time_range_days: Time range to spread sessions over (default 6 months)
        
        Returns:
            Tuple of (user_profiles, all_events)
        """
        print(f"Generating synthetic data for {num_users} users...")
        
        profiles = self.generate_user_profiles(num_users)
        all_events = []
        
        start_date = datetime.now() - timedelta(days=time_range_days)
        
        for profile in profiles:
            # Vary sessions per user
            user_sessions = random.randint(
                sessions_per_user - 20, 
                sessions_per_user + 20
            )
            
            for _ in range(user_sessions):
                # Random date within time range
                random_days = random.randint(0, time_range_days)
                
                # Pick a typical hour for this user (with some variation)
                if random.random() < 0.9:  # 90% typical hours
                    hour = random.choice(profile.typical_login_hours)
                else:
                    hour = random.randint(0, 23)
                
                minute = random.randint(0, 59)
                
                session_date = start_date + timedelta(
                    days=random_days,
                    hours=hour,
                    minutes=minute,
                )
                
                # Generate normal or anomalous session
                if random.random() < anomaly_ratio:
                    events = self.generate_anomalous_session(profile, session_date)
                else:
                    events = self.generate_normal_session(profile, session_date)
                
                all_events.extend(events)
        
        # Sort events by timestamp
        all_events.sort(key=lambda e: e.timestamp)
        
        print(f"Generated {len(all_events)} events")
        print(f"  - Normal events: {sum(1 for e in all_events if not e.is_anomaly)}")
        print(f"  - Anomalous events: {sum(1 for e in all_events if e.is_anomaly)}")
        
        return profiles, all_events
    
    def save_data(
        self,
        profiles: List[UserProfile],
        events: List[UserEvent],
        output_dir: str = "data",
    ):
        """Save generated data to JSON files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save profiles
        profiles_data = []
        for p in profiles:
            profiles_data.append({
                "user_id": p.user_id,
                "name": p.name,
                "typical_city": p.typical_city,
                "device_type": p.device_type,
                "os_family": p.os_family,
            })
        
        with open(os.path.join(output_dir, "profiles.json"), "w") as f:
            json.dump(profiles_data, f, indent=2)
        
        # Save events
        events_data = [e.to_dict() for e in events]
        with open(os.path.join(output_dir, "events.json"), "w") as f:
            json.dump(events_data, f, indent=2, default=str)
        
        print(f"Data saved to {output_dir}/")


if __name__ == "__main__":
    # Demo generation
    generator = SyntheticDataGenerator(seed=42)
    profiles, events = generator.generate_training_data(
        num_users=50,
        sessions_per_user=30,
        anomaly_ratio=0.1,
    )
    generator.save_data(profiles, events)
