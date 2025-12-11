"""
Rased (راصد) - Decision Matrix
Automated response system based on risk classification.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import random
import os
import sys

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import config, RiskLevel


class ResponseAction(Enum):
    """Possible response actions."""
    ALLOW = "allow"              # Allow transaction silently
    CHALLENGE_OTP = "challenge_otp"        # Request OTP verification
    CHALLENGE_FACE = "challenge_face"      # Request FaceID verification
    CHALLENGE_NAFATH = "challenge_nafath"  # Request Nafath verification
    CHALLENGE_QUESTIONS = "challenge_questions"  # Security questions
    DELAY = "delay"              # Introduce artificial delay
    BLOCK = "block"              # Block transaction
    ALERT_SOC = "alert_soc"      # Alert Security Operations Center
    REQUIRE_REVIEW = "require_review"  # Require manual review


@dataclass
class SecurityResponse:
    """Security response to be executed."""
    action: ResponseAction
    risk_score: int
    risk_level: RiskLevel
    
    # Response details
    message_to_user: str
    internal_code: str
    requires_user_action: bool
    
    # Challenge details (if applicable)
    challenge_method: Optional[str] = None
    challenge_timeout_seconds: int = 300  # 5 minutes default
    
    # Alert details (if applicable)
    alert_priority: str = "normal"  # low, normal, high, critical
    alert_recipients: List[str] = field(default_factory=list)
    
    # Transaction handling
    transaction_blocked: bool = False
    pending_review: bool = False
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    response_id: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "action": self.action.value,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level.value,
            "message_to_user": self.message_to_user,
            "internal_code": self.internal_code,
            "requires_user_action": self.requires_user_action,
            "challenge_method": self.challenge_method,
            "challenge_timeout_seconds": self.challenge_timeout_seconds,
            "alert_priority": self.alert_priority,
            "transaction_blocked": self.transaction_blocked,
            "pending_review": self.pending_review,
            "timestamp": self.timestamp.isoformat(),
            "response_id": self.response_id,
        }


class DecisionMatrix:
    """
    Decision matrix for automated security responses.
    
    Maps risk scores to appropriate actions:
    - Green (0-30): Allow silently
    - Yellow (31-70): Step-up authentication
    - Red (71-100): Block and alert
    
    Considers:
    - Risk score and level
    - Transaction type and value
    - User history and preferences
    - Available verification methods
    """
    
    # User-friendly messages in Arabic and English
    MESSAGES = {
        "allow": {
            "ar": "تمت العملية بنجاح",
            "en": "Transaction completed successfully",
        },
        "challenge": {
            "ar": "يرجى التحقق من هويتك للمتابعة",
            "en": "Please verify your identity to continue",
        },
        "block": {
            "ar": "تم إيقاف العملية لأسباب أمنية. يرجى التواصل مع الدعم",
            "en": "Transaction blocked for security reasons. Please contact support",
        },
        "review": {
            "ar": "العملية قيد المراجعة. سيتم إشعارك بالنتيجة",
            "en": "Transaction under review. You will be notified of the result",
        },
    }
    
    def __init__(self):
        """Initialize decision matrix."""
        self.config = config.risk
        
        # Response counters for analytics
        self.response_counts = {
            "allow": 0,
            "challenge": 0,
            "block": 0,
            "review": 0,
        }
        
        # User challenge history (for method selection)
        self.user_challenge_history: Dict[str, List[str]] = {}
        
        # Callbacks for external integrations
        self._alert_callback: Optional[Callable] = None
        self._challenge_callback: Optional[Callable] = None
        self._block_callback: Optional[Callable] = None
    
    def decide(
        self,
        risk_score: int,
        risk_level: RiskLevel,
        event: Dict,
        user_profile: Optional[Dict] = None,
        primary_factors: Optional[List[str]] = None,
    ) -> SecurityResponse:
        """
        Make a security decision based on risk assessment.
        
        Args:
            risk_score: Normalized risk score (0-100)
            risk_level: Risk classification
            event: Current event data
            user_profile: User's profile data
            primary_factors: Primary risk factors identified
        
        Returns:
            SecurityResponse with action to take
        """
        import uuid
        response_id = f"resp_{uuid.uuid4().hex[:12]}"
        
        if risk_level == RiskLevel.GREEN:
            response = self._handle_green(risk_score, event, response_id)
        elif risk_level == RiskLevel.YELLOW:
            response = self._handle_yellow(risk_score, event, user_profile, primary_factors, response_id)
        else:  # RED
            response = self._handle_red(risk_score, event, user_profile, primary_factors, response_id)
        
        # Execute callbacks if registered
        self._execute_callbacks(response, event)
        
        return response
    
    def _handle_green(
        self,
        risk_score: int,
        event: Dict,
        response_id: str,
    ) -> SecurityResponse:
        """Handle low-risk (green) events."""
        self.response_counts["allow"] += 1
        
        return SecurityResponse(
            action=ResponseAction.ALLOW,
            risk_score=risk_score,
            risk_level=RiskLevel.GREEN,
            message_to_user="",  # No message needed for allow
            internal_code="ALLOW_SILENT",
            requires_user_action=False,
            response_id=response_id,
        )
    
    def _handle_yellow(
        self,
        risk_score: int,
        event: Dict,
        user_profile: Optional[Dict],
        primary_factors: Optional[List[str]],
        response_id: str,
    ) -> SecurityResponse:
        """Handle medium-risk (yellow) events with step-up authentication."""
        self.response_counts["challenge"] += 1
        
        user_id = event.get("user_id", "unknown")
        
        # Select appropriate challenge method
        challenge_method = self._select_challenge_method(
            risk_score=risk_score,
            event=event,
            user_id=user_id,
            user_profile=user_profile,
        )
        
        # Determine action based on challenge method
        action_map = {
            "otp_sms": ResponseAction.CHALLENGE_OTP,
            "otp_email": ResponseAction.CHALLENGE_OTP,
            "face_id": ResponseAction.CHALLENGE_FACE,
            "nafath_verification": ResponseAction.CHALLENGE_NAFATH,
            "security_questions": ResponseAction.CHALLENGE_QUESTIONS,
        }
        action = action_map.get(challenge_method, ResponseAction.CHALLENGE_OTP)
        
        # Record challenge in history
        if user_id not in self.user_challenge_history:
            self.user_challenge_history[user_id] = []
        self.user_challenge_history[user_id].append(challenge_method)
        
        # Timeout based on risk score (higher risk = shorter timeout)
        if risk_score > 60:
            timeout = 180  # 3 minutes
        elif risk_score > 45:
            timeout = 300  # 5 minutes
        else:
            timeout = 600  # 10 minutes
        
        return SecurityResponse(
            action=action,
            risk_score=risk_score,
            risk_level=RiskLevel.YELLOW,
            message_to_user=self.MESSAGES["challenge"]["en"],
            internal_code=f"CHALLENGE_{challenge_method.upper()}",
            requires_user_action=True,
            challenge_method=challenge_method,
            challenge_timeout_seconds=timeout,
            alert_priority="normal",
            response_id=response_id,
        )
    
    def _handle_red(
        self,
        risk_score: int,
        event: Dict,
        user_profile: Optional[Dict],
        primary_factors: Optional[List[str]],
        response_id: str,
    ) -> SecurityResponse:
        """Handle high-risk (red) events with blocking."""
        user_id = event.get("user_id", "unknown")
        
        # Determine if immediate block or pending review
        if risk_score >= 90:
            # Immediate block for very high risk
            self.response_counts["block"] += 1
            
            return SecurityResponse(
                action=ResponseAction.BLOCK,
                risk_score=risk_score,
                risk_level=RiskLevel.RED,
                message_to_user=self.MESSAGES["block"]["en"],
                internal_code="BLOCK_IMMEDIATE",
                requires_user_action=False,
                transaction_blocked=True,
                pending_review=False,
                alert_priority="critical",
                alert_recipients=["soc@absher.sa", "fraud-team@absher.sa"],
                response_id=response_id,
            )
        else:
            # Pending review for moderately high risk
            self.response_counts["review"] += 1
            
            return SecurityResponse(
                action=ResponseAction.REQUIRE_REVIEW,
                risk_score=risk_score,
                risk_level=RiskLevel.RED,
                message_to_user=self.MESSAGES["review"]["en"],
                internal_code="PENDING_REVIEW",
                requires_user_action=False,
                transaction_blocked=True,
                pending_review=True,
                alert_priority="high",
                alert_recipients=["review-team@absher.sa"],
                response_id=response_id,
            )
    
    def _select_challenge_method(
        self,
        risk_score: int,
        event: Dict,
        user_id: str,
        user_profile: Optional[Dict],
    ) -> str:
        """
        Select the most appropriate challenge method.
        
        Considers:
        - Risk level (higher = stronger verification)
        - Device capabilities (FaceID only on supported devices)
        - User preferences
        - Recent challenge history (avoid repeating)
        """
        available_methods = list(self.config.step_up_methods)
        
        # Remove methods based on device
        device_type = event.get("device_type", "mobile")
        os_family = event.get("os_family", "")
        
        if device_type == "desktop" or os_family not in ["iOS", "Android"]:
            # Remove face_id for non-mobile or non-supported OS
            if "face_id" in available_methods:
                available_methods.remove("face_id")
        
        # Prioritize based on risk score
        if risk_score >= 60:
            # High-medium risk: prefer stronger methods
            priority_order = ["nafath_verification", "face_id", "otp_sms", "otp_email", "security_questions"]
        elif risk_score >= 45:
            # Medium risk: prefer balanced methods
            priority_order = ["face_id", "otp_sms", "nafath_verification", "otp_email", "security_questions"]
        else:
            # Low-medium risk: prefer convenient methods
            priority_order = ["otp_sms", "otp_email", "face_id", "security_questions", "nafath_verification"]
        
        # Avoid recently used methods
        recent_challenges = self.user_challenge_history.get(user_id, [])[-3:]
        
        for method in priority_order:
            if method in available_methods and method not in recent_challenges:
                return method
        
        # Fallback to first available
        return available_methods[0] if available_methods else "otp_sms"
    
    def register_alert_callback(self, callback: Callable[[SecurityResponse, Dict], None]):
        """Register callback for SOC alerts."""
        self._alert_callback = callback
    
    def register_challenge_callback(self, callback: Callable[[SecurityResponse, Dict], None]):
        """Register callback for challenge initiation."""
        self._challenge_callback = callback
    
    def register_block_callback(self, callback: Callable[[SecurityResponse, Dict], None]):
        """Register callback for transaction blocks."""
        self._block_callback = callback
    
    def _execute_callbacks(self, response: SecurityResponse, event: Dict):
        """Execute registered callbacks based on response."""
        if response.action == ResponseAction.BLOCK and self._block_callback:
            self._block_callback(response, event)
        
        if response.action in [ResponseAction.ALERT_SOC, ResponseAction.BLOCK, ResponseAction.REQUIRE_REVIEW]:
            if self._alert_callback:
                self._alert_callback(response, event)
        
        if response.requires_user_action and self._challenge_callback:
            self._challenge_callback(response, event)
    
    def handle_challenge_result(
        self,
        response_id: str,
        user_id: str,
        passed: bool,
        challenge_method: str,
    ) -> Dict[str, Any]:
        """
        Handle the result of a challenge verification.
        
        Args:
            response_id: Original response ID
            user_id: User who completed the challenge
            passed: Whether the challenge was passed
            challenge_method: Method used for verification
        
        Returns:
            Result with next steps
        """
        if passed:
            return {
                "status": "verified",
                "action": "allow",
                "feedback": "false_positive",
                "message": "Verification successful. Transaction can proceed.",
            }
        else:
            return {
                "status": "failed",
                "action": "block",
                "feedback": "suspicious",
                "message": "Verification failed. Transaction blocked.",
                "escalate": True,
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get decision statistics."""
        total = sum(self.response_counts.values())
        
        if total == 0:
            return {"total": 0, "breakdown": self.response_counts}
        
        return {
            "total": total,
            "breakdown": self.response_counts,
            "rates": {
                k: (v / total * 100) for k, v in self.response_counts.items()
            },
        }


if __name__ == "__main__":
    # Demo decision matrix
    matrix = DecisionMatrix()
    
    # Test scenarios
    scenarios = [
        {
            "name": "Normal User - View Dashboard",
            "risk_score": 15,
            "risk_level": RiskLevel.GREEN,
            "event": {
                "user_id": "user_001",
                "action": "view_dashboard",
                "device_type": "mobile",
                "os_family": "iOS",
            },
        },
        {
            "name": "Suspicious - New Location Transfer",
            "risk_score": 55,
            "risk_level": RiskLevel.YELLOW,
            "event": {
                "user_id": "user_002",
                "action": "confirm_transfer",
                "device_type": "mobile",
                "os_family": "iOS",
            },
            "factors": ["Location deviation", "New beneficiary"],
        },
        {
            "name": "High Risk - Bot Behavior",
            "risk_score": 85,
            "risk_level": RiskLevel.RED,
            "event": {
                "user_id": "user_003",
                "action": "confirm_transfer",
                "device_type": "desktop",
                "os_family": "Windows",
            },
            "factors": ["Bot-like speed", "VPN detected", "High-value transaction"],
        },
        {
            "name": "Critical - Confirmed Attack",
            "risk_score": 95,
            "risk_level": RiskLevel.RED,
            "event": {
                "user_id": "user_004",
                "action": "confirm_transfer",
                "device_type": "desktop",
                "os_family": "Linux",
            },
            "factors": ["Tor exit node", "Foreign location", "Bot behavior", "New device"],
        },
    ]
    
    print("Decision Matrix Demo\n" + "=" * 50)
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"  Risk Score: {scenario['risk_score']}")
        print(f"  Risk Level: {scenario['risk_level'].value}")
        
        response = matrix.decide(
            risk_score=scenario["risk_score"],
            risk_level=scenario["risk_level"],
            event=scenario["event"],
            primary_factors=scenario.get("factors"),
        )
        
        print(f"  Response: {response.action.value}")
        print(f"  User Message: {response.message_to_user or '(none)'}")
        if response.challenge_method:
            print(f"  Challenge: {response.challenge_method} (timeout: {response.challenge_timeout_seconds}s)")
        if response.transaction_blocked:
            print(f"  Transaction: BLOCKED")
        if response.alert_priority != "normal":
            print(f"  Alert Priority: {response.alert_priority}")
    
    print("\n" + "=" * 50)
    print("Statistics:")
    stats = matrix.get_statistics()
    for k, v in stats["breakdown"].items():
        rate = stats["rates"].get(k, 0)
        print(f"  {k}: {v} ({rate:.1f}%)")
