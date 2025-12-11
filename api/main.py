"""
Rased (راصد) - FastAPI Application
Main API server for the fraud detection system.
"""

from typing import Dict, List, Optional
from datetime import datetime
import os
import sys

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Try to import FastAPI, provide fallback if not installed
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn")

from config.settings import config, RiskLevel
from engine.inference_engine import InferenceEngine
from engine.decision_matrix import DecisionMatrix, SecurityResponse
from feedback.learning_loop import LearningLoop, FeedbackType
from queue.async_processor import AsyncProcessor, ProcessingResult


# Pydantic models for API
if FASTAPI_AVAILABLE:
    class UserEvent(BaseModel):
        """User action event."""
        user_id: str
        action: str
        timestamp: Optional[str] = None
        
        # Temporal
        hour_of_day: Optional[int] = None
        day_of_week: Optional[int] = None
        time_since_last_action: Optional[float] = 30.0
        
        # Spatial
        latitude: Optional[float] = 24.7136
        longitude: Optional[float] = 46.6753
        country_code: Optional[str] = "SA"
        city: Optional[str] = "Riyadh"
        ip_address: Optional[str] = None
        ip_reputation_score: Optional[float] = 100.0
        is_vpn: Optional[bool] = False
        is_tor: Optional[bool] = False
        
        # Device
        device_type: Optional[str] = "mobile"
        os_family: Optional[str] = "iOS"
        os_version: Optional[str] = "17.0"
        device_hash: Optional[str] = None
        screen_width: Optional[int] = 390
        screen_height: Optional[int] = 844
        battery_level: Optional[int] = 100
        
        # Transaction
        service_type: Optional[str] = None
        transaction_amount: Optional[float] = 0.0
        beneficiary_id: Optional[str] = None


    class FeedbackRequest(BaseModel):
        """Feedback submission request."""
        user_id: str
        event_id: str
        response_id: str
        feedback_type: str  # false_positive, true_positive, etc.
        challenge_passed: Optional[bool] = None
        challenge_method: Optional[str] = None


    class RiskResponse(BaseModel):
        """Risk assessment response."""
        event_id: str
        user_id: str
        risk_score: int
        risk_level: str
        action: str
        message: Optional[str] = None
        challenge_method: Optional[str] = None
        processing_time_ms: float


def create_app() -> 'FastAPI':
    """Create and configure FastAPI application."""
    
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required. Install with: pip install fastapi uvicorn")
    
    app = FastAPI(
        title="Rased (راصد) - AI Fraud Detection",
        description="Real-time behavioral analysis and fraud detection system for Absher",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize components
    inference_engine = InferenceEngine()
    decision_matrix = DecisionMatrix()
    learning_loop = LearningLoop()
    
    # Async processor for background processing
    async_processor = AsyncProcessor()
    
    def process_event_async(event: Dict) -> ProcessingResult:
        """Background processing callback."""
        result = inference_engine.infer(event)
        
        response = decision_matrix.decide(
            risk_score=result.normalized_risk_score,
            risk_level=config.risk.get_risk_level(result.normalized_risk_score),
            event=event,
            primary_factors=None,
        )
        
        return ProcessingResult(
            message_id="",
            success=True,
            risk_score=result.normalized_risk_score,
            decision=response.action.value,
            processing_time_ms=result.processing_time_ms,
        )
    
    async_processor.set_inference_callback(process_event_async)
    
    # ========== Health Endpoints ==========
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
        }
    
    @app.get("/stats")
    async def get_stats():
        """Get system statistics."""
        return {
            "inference_engine": inference_engine.get_stats(),
            "decision_matrix": decision_matrix.get_statistics(),
            "learning_loop": learning_loop.get_performance_metrics(),
            "async_processor": async_processor.get_stats(),
        }
    
    # ========== Event Endpoints ==========
    
    @app.post("/events", response_model=RiskResponse)
    async def process_event(event: UserEvent, background_tasks: BackgroundTasks):
        """
        Process a user event and return risk assessment.
        
        This is the main endpoint called by the Absher frontend.
        """
        import uuid
        event_id = f"evt_{uuid.uuid4().hex[:12]}"
        
        # Convert to dict
        event_dict = event.dict()
        event_dict["event_id"] = event_id
        
        # Auto-fill timestamp if not provided
        if not event_dict.get("timestamp"):
            event_dict["timestamp"] = datetime.now().isoformat()
        
        # Auto-fill hour/day if not provided
        now = datetime.now()
        if event_dict.get("hour_of_day") is None:
            event_dict["hour_of_day"] = now.hour
        if event_dict.get("day_of_week") is None:
            event_dict["day_of_week"] = now.weekday()
        
        # Perform inference
        result = inference_engine.infer(event_dict)
        
        # Get decision
        risk_level = config.risk.get_risk_level(result.normalized_risk_score)
        response = decision_matrix.decide(
            risk_score=result.normalized_risk_score,
            risk_level=risk_level,
            event=event_dict,
            primary_factors=None,
        )
        
        return RiskResponse(
            event_id=event_id,
            user_id=event.user_id,
            risk_score=result.normalized_risk_score,
            risk_level=risk_level.value,
            action=response.action.value,
            message=response.message_to_user if response.requires_user_action else None,
            challenge_method=response.challenge_method,
            processing_time_ms=result.processing_time_ms,
        )
    
    @app.post("/events/async")
    async def process_event_async_endpoint(event: UserEvent):
        """
        Submit event for async processing (non-blocking).
        
        Returns immediately with event ID for tracking.
        """
        import uuid
        event_id = f"evt_{uuid.uuid4().hex[:12]}"
        
        event_dict = event.dict()
        event_dict["event_id"] = event_id
        event_dict["timestamp"] = datetime.now().isoformat()
        
        # Submit to async processor
        try:
            async_processor.submit_event(event_dict)
            return {
                "event_id": event_id,
                "status": "submitted",
                "message": "Event queued for processing",
            }
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
    
    # ========== Risk Endpoints ==========
    
    @app.get("/risk/{user_id}")
    async def get_user_risk(user_id: str):
        """Get current risk profile for a user."""
        summary = inference_engine.get_user_risk_summary(user_id)
        return summary
    
    # ========== Feedback Endpoints ==========
    
    @app.post("/feedback")
    async def submit_feedback(feedback: FeedbackRequest):
        """
        Submit feedback on a security decision.
        
        Called when:
        - User passes a challenge (false_positive)
        - Fraud is confirmed (true_positive)
        - Fraud is reported later (false_negative)
        """
        try:
            feedback_type = FeedbackType(feedback.feedback_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid feedback type: {feedback.feedback_type}"
            )
        
        result = learning_loop.process_feedback(
            user_id=feedback.user_id,
            event_id=feedback.event_id,
            response_id=feedback.response_id,
            feedback_type=feedback_type,
            original_risk_score=0,  # Would be looked up in production
            original_decision="",
            challenge_passed=feedback.challenge_passed,
            challenge_method=feedback.challenge_method,
        )
        
        return {
            "feedback_id": result.feedback_id,
            "status": "processed",
            "score_adjustment": result.score_adjustment,
            "embedding_updated": result.embedding_updated,
        }
    
    @app.get("/feedback/metrics")
    async def get_feedback_metrics():
        """Get feedback performance metrics."""
        return learning_loop.get_performance_metrics()
    
    # ========== Admin Endpoints ==========
    
    @app.post("/admin/retrain")
    async def trigger_retrain(background_tasks: BackgroundTasks):
        """Trigger model retraining (admin only)."""
        # In production, this would be protected by authentication
        return {
            "status": "acknowledged",
            "message": "Retraining job will be scheduled",
        }
    
    @app.get("/admin/config")
    async def get_config():
        """Get current configuration (admin only)."""
        return {
            "risk_thresholds": {
                "green_max": config.risk.green_max,
                "yellow_max": config.risk.yellow_max,
                "red_min": config.risk.red_min,
            },
            "model_config": {
                "sequence_length": config.model.sequence_length,
                "embedding_dim": config.model.embedding_dim,
            },
            "queue_config": {
                "queue_type": config.queue.queue_type,
                "processing_timeout_ms": config.queue.processing_timeout_ms,
            },
        }
    
    # Start async processor
    @app.on_event("startup")
    async def startup_event():
        """Start background services."""
        async_processor.start()
        print("Rased API started")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Stop background services."""
        async_processor.stop()
        print("Rased API stopped")
    
    return app


# Create app instance
if FASTAPI_AVAILABLE:
    app = create_app()


if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        import uvicorn
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
        )
    else:
        print("Please install FastAPI: pip install fastapi uvicorn")
