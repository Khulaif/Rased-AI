"""
Rased (راصد) - Risk Scorer
Calculates composite risk scores from multiple signals.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import os
import sys

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import config, RiskLevel


@dataclass
class RiskComponent:
    """Individual risk component with value and weight."""
    name: str
    value: float  # 0.0 to 1.0
    weight: float
    reason: str


@dataclass
class RiskAssessment:
    """Complete risk assessment with breakdown."""
    raw_score: float
    normalized_score: int  # 0-100
    risk_level: RiskLevel
    components: List[RiskComponent]
    primary_risk_factors: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            "raw_score": self.raw_score,
            "normalized_score": self.normalized_score,
            "risk_level": self.risk_level.value,
            "components": [
                {"name": c.name, "value": c.value, "weight": c.weight, "reason": c.reason}
                for c in self.components
            ],
            "primary_risk_factors": self.primary_risk_factors,
            "timestamp": self.timestamp.isoformat(),
        }


class RiskScorer:
    """
    Calculates composite risk scores from multiple signals.
    
    Combines:
    1. Model prediction error (LSTM anomaly score)
    2. Contextual anomalies (location, device, timing)
    3. Historical risk factors (past behavior, confirmed fraud)
    4. Transaction risk (amount, new beneficiary)
    5. Real-time signals (IP reputation, VPN usage)
    """
    
    def __init__(self):
        """Initialize risk scorer with default weights."""
        self.config = config.risk
        
        # Base weights (must sum to 1.0)
        self.base_weights = {
            "model_prediction": 0.30,
            "contextual": 0.25,
            "historical": 0.15,
            "transaction": 0.15,
            "realtime": 0.15,
        }
        
        # Context-specific weight adjustments
        self.transaction_weight_boost = 1.5  # Boost for transaction events
        
        # Thresholds for risk factors
        self.thresholds = {
            "high_value_amount": 50000,
            "medium_value_amount": 10000,
            "low_ip_reputation": 50,
            "suspicious_navigation_speed": 10,  # Actions per minute
            "bot_navigation_speed": 30,
        }
    
    def calculate_risk(
        self,
        model_anomaly_score: float,
        twin_deviations: Dict[str, float],
        historical_profile: Dict,
        event: Dict,
    ) -> RiskAssessment:
        """
        Calculate comprehensive risk score.
        
        Args:
            model_anomaly_score: LSTM model anomaly output (0-1)
            twin_deviations: Digital Twin deviation breakdown
            historical_profile: User's historical risk profile
            event: Current event data
        
        Returns:
            RiskAssessment with detailed breakdown
        """
        components = []
        
        # === 1. Model Prediction Risk ===
        model_component = RiskComponent(
            name="model_prediction",
            value=min(1.0, model_anomaly_score),
            weight=self.base_weights["model_prediction"],
            reason=self._get_model_reason(model_anomaly_score),
        )
        components.append(model_component)
        
        # === 2. Contextual Risk ===
        contextual_score = self._calculate_contextual_risk(twin_deviations)
        contextual_component = RiskComponent(
            name="contextual",
            value=contextual_score,
            weight=self.base_weights["contextual"],
            reason=self._get_contextual_reason(twin_deviations),
        )
        components.append(contextual_component)
        
        # === 3. Historical Risk ===
        historical_score = self._calculate_historical_risk(historical_profile)
        historical_component = RiskComponent(
            name="historical",
            value=historical_score,
            weight=self.base_weights["historical"],
            reason=self._get_historical_reason(historical_profile),
        )
        components.append(historical_component)
        
        # === 4. Transaction Risk ===
        transaction_score, transaction_reason = self._calculate_transaction_risk(event)
        transaction_weight = self.base_weights["transaction"]
        
        # Boost weight for high-value transactions
        if event.get("transaction_amount", 0) > self.thresholds["high_value_amount"]:
            transaction_weight *= self.transaction_weight_boost
        
        transaction_component = RiskComponent(
            name="transaction",
            value=transaction_score,
            weight=transaction_weight,
            reason=transaction_reason,
        )
        components.append(transaction_component)
        
        # === 5. Real-time Signals ===
        realtime_score, realtime_reason = self._calculate_realtime_risk(event)
        realtime_component = RiskComponent(
            name="realtime",
            value=realtime_score,
            weight=self.base_weights["realtime"],
            reason=realtime_reason,
        )
        components.append(realtime_component)
        
        # === Calculate Composite Score ===
        total_weight = sum(c.weight for c in components)
        raw_score = sum(c.value * c.weight for c in components) / total_weight
        
        # Normalize to 0-100
        normalized_score = int(min(100, max(0, raw_score * 100)))
        
        # Determine risk level
        risk_level = self.config.get_risk_level(normalized_score)
        
        # Identify primary risk factors
        primary_factors = self._identify_primary_factors(components, twin_deviations, event)
        
        return RiskAssessment(
            raw_score=raw_score,
            normalized_score=normalized_score,
            risk_level=risk_level,
            components=components,
            primary_risk_factors=primary_factors,
            timestamp=datetime.now(),
        )
    
    def _calculate_contextual_risk(self, deviations: Dict[str, float]) -> float:
        """Calculate contextual risk from Digital Twin deviations."""
        # Weighted combination of deviation factors
        weights = {
            "location": 0.30,
            "device": 0.25,
            "timing": 0.20,
            "navigation_speed": 0.25,
        }
        
        score = 0.0
        for factor, weight in weights.items():
            if factor in deviations:
                score += deviations[factor] * weight
        
        return min(1.0, score)
    
    def _calculate_historical_risk(self, profile: Dict) -> float:
        """Calculate historical risk from user profile."""
        score = 0.0
        
        # Average historical risk
        avg_risk = profile.get("average_risk_score", 0.0) / 100.0
        score += avg_risk * 0.4
        
        # Confirmed fraud history (major factor)
        fraud_count = profile.get("confirmed_fraud_count", 0)
        if fraud_count > 0:
            score += min(0.5, fraud_count * 0.25)
        
        # Account age factor (newer accounts slightly riskier)
        account_age_days = profile.get("account_age_days", 365)
        if account_age_days < 30:
            score += 0.2
        elif account_age_days < 90:
            score += 0.1
        
        # False positive adjustment (reduce score for users with many FPs)
        false_positive_rate = profile.get("false_positive_rate", 0.0)
        if false_positive_rate > 0.5 and fraud_count == 0:
            score *= 0.7  # 30% discount
        
        return min(1.0, score)
    
    def _calculate_transaction_risk(self, event: Dict) -> Tuple[float, str]:
        """Calculate transaction-specific risk."""
        score = 0.0
        reasons = []
        
        action = event.get("action", "")
        amount = event.get("transaction_amount", 0.0)
        beneficiary = event.get("beneficiary_id", "")
        
        # Only score if it's a transaction action
        if action not in ["initiate_ownership_transfer", "confirm_ownership_transfer"]:
            return 0.0, "Non-transaction action"
        
        # Amount-based risk
        if amount > self.thresholds["high_value_amount"]:
            score += 0.4
            reasons.append(f"High-value transaction: {amount:,.0f} SAR")
        elif amount > self.thresholds["medium_value_amount"]:
            score += 0.2
            reasons.append(f"Medium-value transaction: {amount:,.0f} SAR")
        
        # New beneficiary risk
        if beneficiary and "new" in str(beneficiary).lower():
            score += 0.3
            reasons.append("New beneficiary")
        
        # Fast confirmation risk
        time_since_last = event.get("time_since_last_action", 30.0)
        if action == "confirm_ownership_transfer" and time_since_last < 2.0:
            score += 0.3
            reasons.append("Very rapid confirmation")
        
        reason = "; ".join(reasons) if reasons else "Normal transaction"
        return min(1.0, score), reason
    
    def _calculate_realtime_risk(self, event: Dict) -> Tuple[float, str]:
        """Calculate real-time signal risk."""
        score = 0.0
        reasons = []
        
        # VPN/Tor usage
        if event.get("is_vpn"):
            score += 0.25
            reasons.append("VPN detected")
        
        if event.get("is_tor"):
            score += 0.4
            reasons.append("Tor exit node detected")
        
        # IP reputation
        ip_reputation = event.get("ip_reputation_score", 100.0)
        if ip_reputation < self.thresholds["low_ip_reputation"]:
            score += 0.3
            reasons.append(f"Low IP reputation: {ip_reputation:.0f}")
        elif ip_reputation < 75:
            score += 0.1
            reasons.append(f"Medium IP reputation: {ip_reputation:.0f}")
        
        # Navigation speed (bot detection)
        time_since_last = event.get("time_since_last_action", 30.0)
        if time_since_last > 0:
            speed = 60.0 / time_since_last  # Actions per minute
            
            if speed > self.thresholds["bot_navigation_speed"]:
                score += 0.5
                reasons.append(f"Bot-like speed: {speed:.1f} actions/min")
            elif speed > self.thresholds["suspicious_navigation_speed"]:
                score += 0.2
                reasons.append(f"Fast navigation: {speed:.1f} actions/min")
        
        # Foreign country
        country = event.get("country_code", "SA")
        if country != "SA":
            score += 0.2
            reasons.append(f"Foreign location: {country}")
        
        reason = "; ".join(reasons) if reasons else "Normal signals"
        return min(1.0, score), reason
    
    def _get_model_reason(self, score: float) -> str:
        """Get explanation for model prediction score."""
        if score > 0.7:
            return "High deviation from predicted behavior"
        elif score > 0.4:
            return "Moderate deviation from predicted behavior"
        elif score > 0.2:
            return "Slight deviation from predicted behavior"
        else:
            return "Behavior matches predictions"
    
    def _get_contextual_reason(self, deviations: Dict[str, float]) -> str:
        """Get explanation for contextual risk."""
        high_factors = [
            k for k, v in deviations.items()
            if v > 0.5 and k != "composite"
        ]
        
        if not high_factors:
            return "Normal contextual patterns"
        
        factor_names = {
            "location": "unusual location",
            "device": "unfamiliar device",
            "timing": "unusual time",
            "navigation_speed": "unusual navigation speed",
            "transaction": "unusual transaction pattern",
        }
        
        return "Detected: " + ", ".join(
            factor_names.get(f, f) for f in high_factors[:3]
        )
    
    def _get_historical_reason(self, profile: Dict) -> str:
        """Get explanation for historical risk."""
        fraud_count = profile.get("confirmed_fraud_count", 0)
        if fraud_count > 0:
            return f"History of {fraud_count} confirmed fraud event(s)"
        
        avg_risk = profile.get("average_risk_score", 0.0)
        if avg_risk > 50:
            return f"Elevated historical risk average: {avg_risk:.0f}"
        
        account_age = profile.get("account_age_days", 365)
        if account_age < 30:
            return "New account (less than 30 days)"
        
        return "Normal historical profile"
    
    def _identify_primary_factors(
        self,
        components: List[RiskComponent],
        deviations: Dict[str, float],
        event: Dict,
    ) -> List[str]:
        """Identify the primary risk factors."""
        factors = []
        
        # High-impact components
        for component in components:
            if component.value > 0.5:
                factors.append(f"{component.name}: {component.reason}")
        
        # Specific high-risk indicators
        if deviations.get("location", 0) > 0.7:
            factors.append("Significant location deviation")
        
        if deviations.get("navigation_speed", 0) > 0.8:
            factors.append("Bot-like behavior detected")
        
        if event.get("is_vpn") and event.get("transaction_amount", 0) > 0:
            factors.append("VPN used during transaction")
        
        return factors[:5]  # Top 5 factors
    
    def adjust_for_false_positive(
        self,
        assessment: RiskAssessment,
        user_id: str,
    ) -> RiskAssessment:
        """
        Adjust scoring after a challenge is successfully passed.
        
        This reduces the impact of similar events in the future.
        """
        # Reduce the score by 20% for this type of event
        adjusted_raw = assessment.raw_score * 0.8
        adjusted_normalized = int(adjusted_raw * 100)
        
        new_level = self.config.get_risk_level(adjusted_normalized)
        
        return RiskAssessment(
            raw_score=adjusted_raw,
            normalized_score=adjusted_normalized,
            risk_level=new_level,
            components=assessment.components,
            primary_risk_factors=assessment.primary_risk_factors + ["Adjusted for false positive"],
            timestamp=datetime.now(),
        )


if __name__ == "__main__":
    # Demo risk scoring
    scorer = RiskScorer()
    
    # Normal user scenario
    normal_assessment = scorer.calculate_risk(
        model_anomaly_score=0.1,
        twin_deviations={
            "location": 0.1,
            "device": 0.0,
            "timing": 0.1,
            "navigation_speed": 0.2,
            "composite": 0.1,
        },
        historical_profile={
            "average_risk_score": 15.0,
            "confirmed_fraud_count": 0,
            "account_age_days": 365,
            "false_positive_rate": 0.1,
        },
        event={
            "action": "view_dashboard",
            "country_code": "SA",
            "is_vpn": False,
            "ip_reputation_score": 95.0,
            "time_since_last_action": 30.0,
        },
    )
    
    print("Normal User Assessment:")
    print(f"  Score: {normal_assessment.normalized_score}/100")
    print(f"  Level: {normal_assessment.risk_level.value}")
    print(f"  Factors: {normal_assessment.primary_risk_factors}")
    print()
    
    # Suspicious scenario
    suspicious_assessment = scorer.calculate_risk(
        model_anomaly_score=0.7,
        twin_deviations={
            "location": 0.9,
            "device": 0.8,
            "timing": 0.3,
            "navigation_speed": 0.9,
            "composite": 0.7,
        },
        historical_profile={
            "average_risk_score": 25.0,
            "confirmed_fraud_count": 0,
            "account_age_days": 30,
            "false_positive_rate": 0.0,
        },
        event={
            "action": "confirm_transfer",
            "country_code": "RU",
            "is_vpn": True,
            "ip_reputation_score": 40.0,
            "time_since_last_action": 0.5,
            "transaction_amount": 100000.0,
            "beneficiary_id": "new_beneficiary_999",
        },
    )
    
    print("Suspicious User Assessment:")
    print(f"  Score: {suspicious_assessment.normalized_score}/100")
    print(f"  Level: {suspicious_assessment.risk_level.value}")
    print(f"  Factors: {suspicious_assessment.primary_risk_factors}")
    print()
    
    # Component breakdown
    print("Component Breakdown:")
    for component in suspicious_assessment.components:
        print(f"  {component.name}: {component.value:.2f} (weight: {component.weight:.2f})")
        print(f"    Reason: {component.reason}")
