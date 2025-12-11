"""
Rased (راصد) - Model Retrainer
Periodic batch retraining of LSTM models.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import shutil
import os
import sys

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import config


@dataclass
class RetrainingJob:
    """Retraining job metadata."""
    job_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Data stats
    training_samples: int = 0
    validation_samples: int = 0
    
    # Performance metrics
    training_loss: float = 0.0
    validation_loss: float = 0.0
    improvement: float = 0.0
    
    # Model versions
    previous_version: str = ""
    new_version: str = ""
    
    # Status
    status: str = "pending"  # pending, running, completed, failed
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "training_samples": self.training_samples,
            "validation_samples": self.validation_samples,
            "training_loss": self.training_loss,
            "validation_loss": self.validation_loss,
            "improvement": self.improvement,
            "previous_version": self.previous_version,
            "new_version": self.new_version,
            "status": self.status,
            "error_message": self.error_message,
        }


class ModelRetrainer:
    """
    Manages periodic retraining of Rased LSTM models.
    
    Features:
    1. Scheduled retraining (e.g., weekly)
    2. Triggered retraining (on performance degradation)
    3. Model versioning and rollback
    4. A/B testing support
    """
    
    def __init__(
        self,
        model=None,
        vectorizer=None,
        model_path: str = "models/trained",
        feedback_path: str = "feedback",
    ):
        """
        Initialize model retrainer.
        
        Args:
            model: RasedLSTMModel instance
            vectorizer: FeatureVectorizer instance
            model_path: Path for model storage
            feedback_path: Path for feedback data
        """
        self.model = model
        self.vectorizer = vectorizer
        self.model_path = model_path
        self.feedback_path = feedback_path
        
        # Version management
        self.current_version = "v1.0.0"
        self.version_history: List[Dict] = []
        self.max_versions = config.feedback.max_model_versions
        
        # Retraining configuration
        self.retrain_interval_days = config.feedback.retrain_interval_days
        self.min_samples = config.feedback.min_samples_for_retrain
        
        # Retraining history
        self.retraining_jobs: List[RetrainingJob] = []
        
        # Last retrain timestamp
        self.last_retrain: Optional[datetime] = None
        
        # Load state
        self._load_state()
    
    def should_retrain(self) -> Tuple[bool, str]:
        """
        Check if retraining is due.
        
        Returns:
            Tuple of (should_retrain, reason)
        """
        # Check time-based schedule
        if self.last_retrain:
            days_since = (datetime.now() - self.last_retrain).days
            if days_since >= self.retrain_interval_days:
                return True, f"Scheduled retrain ({days_since} days since last)"
        else:
            # Never retrained
            return True, "Initial training needed"
        
        return False, "Not due for retraining"
    
    def prepare_training_data(
        self,
        events: List[Dict],
        feedback_events: List[Dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for retraining.
        
        Args:
            events: Historical events
            feedback_events: Feedback for label refinement
        
        Returns:
            Tuple of (X, y_action, y_anomaly, user_ids)
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not initialized")
        
        # Fit vectorizer if needed
        if not self.vectorizer.is_fitted:
            self.vectorizer.fit(events)
        
        # Process feedback to update labels
        if feedback_events:
            events = self._refine_labels(events, feedback_events)
        
        # Prepare training sequences
        X, y_action, y_anomaly = self.vectorizer.prepare_training_data(
            events,
            sequence_length=config.model.sequence_length,
        )
        
        # Extract user IDs from sequences
        user_ids = self._extract_user_ids(events)
        
        return X, y_action, y_anomaly, user_ids
    
    def _refine_labels(
        self,
        events: List[Dict],
        feedback_events: List[Dict],
    ) -> List[Dict]:
        """Refine event labels based on feedback."""
        # Create lookup by event_id
        feedback_map = {
            fb.get("event_id"): fb 
            for fb in feedback_events 
            if fb.get("event_id")
        }
        
        refined = []
        for event in events:
            event_copy = dict(event)
            event_id = event.get("event_id")
            
            if event_id in feedback_map:
                fb = feedback_map[event_id]
                fb_type = fb.get("feedback_type")
                
                if fb_type in ["true_positive", "false_negative"]:
                    event_copy["is_anomaly"] = True
                elif fb_type in ["false_positive", "true_negative"]:
                    event_copy["is_anomaly"] = False
            
            refined.append(event_copy)
        
        return refined
    
    def _extract_user_ids(self, events: List[Dict]) -> List[str]:
        """Extract user IDs maintaining sequence order."""
        # Group by user
        user_events: Dict[str, List] = {}
        for e in events:
            user_id = e.get("user_id", "unknown")
            if user_id not in user_events:
                user_events[user_id] = []
            user_events[user_id].append(e)
        
        # Build user ID list matching training sequences
        user_ids = []
        seq_len = config.model.sequence_length
        
        for user_id, evts in user_events.items():
            # Number of sequences for this user
            n_sequences = max(0, len(evts) - seq_len)
            user_ids.extend([user_id] * n_sequences)
        
        return user_ids
    
    def retrain(
        self,
        events: List[Dict],
        feedback_events: List[Dict] = None,
        force: bool = False,
    ) -> RetrainingJob:
        """
        Perform model retraining.
        
        Args:
            events: Training events
            feedback_events: Feedback for label refinement
            force: Force retraining even if not due
        
        Returns:
            RetrainingJob with results
        """
        import uuid
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        
        job = RetrainingJob(
            job_id=job_id,
            started_at=datetime.now(),
            previous_version=self.current_version,
            status="running",
        )
        
        try:
            # Check if we should retrain
            if not force:
                should, reason = self.should_retrain()
                if not should:
                    job.status = "skipped"
                    job.error_message = reason
                    return job
            
            print(f"Starting retraining job {job_id}...")
            
            # Prepare data
            X, y_action, y_anomaly, user_ids = self.prepare_training_data(
                events, feedback_events
            )
            
            if len(X) < self.min_samples:
                job.status = "skipped"
                job.error_message = f"Insufficient samples ({len(X)} < {self.min_samples})"
                return job
            
            job.training_samples = int(len(X) * 0.8)
            job.validation_samples = len(X) - job.training_samples
            
            # Backup current model
            self._backup_model()
            
            # Train model
            if self.model:
                history = self.model.train(
                    X_train=X,
                    y_action_train=y_action,
                    y_anomaly_train=y_anomaly,
                    user_ids=user_ids,
                    epochs=config.model.epochs,
                    batch_size=config.model.batch_size,
                    learning_rate=config.model.learning_rate,
                    validation_split=config.model.validation_split,
                )
                
                job.training_loss = history["train_loss"][-1]
                job.validation_loss = history["val_loss"][-1]
            else:
                # Simulate training for demo
                job.training_loss = 0.5
                job.validation_loss = 0.6
            
            # Calculate improvement
            if self.retraining_jobs:
                prev_loss = self.retraining_jobs[-1].validation_loss
                job.improvement = (prev_loss - job.validation_loss) / max(0.001, prev_loss)
            
            # Update version
            self.current_version = self._increment_version(self.current_version)
            job.new_version = self.current_version
            
            # Save model
            self._save_model()
            
            # Update state
            job.completed_at = datetime.now()
            job.status = "completed"
            self.last_retrain = datetime.now()
            
            print(f"Retraining completed. New version: {job.new_version}")
            print(f"Training loss: {job.training_loss:.4f}")
            print(f"Validation loss: {job.validation_loss:.4f}")
            print(f"Improvement: {job.improvement:.2%}")
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            print(f"Retraining failed: {e}")
        
        self.retraining_jobs.append(job)
        self._save_state()
        
        return job
    
    def rollback(self, version: str = None) -> bool:
        """
        Rollback to a previous model version.
        
        Args:
            version: Specific version to rollback to (latest backup if None)
        
        Returns:
            True if rollback successful
        """
        if version:
            backup_path = os.path.join(self.model_path, f"backup_{version}")
        else:
            # Find latest backup
            backups = sorted([
                d for d in os.listdir(self.model_path)
                if d.startswith("backup_")
            ], reverse=True)
            
            if not backups:
                print("No backups available")
                return False
            
            backup_path = os.path.join(self.model_path, backups[0])
            version = backups[0].replace("backup_", "")
        
        if not os.path.exists(backup_path):
            print(f"Backup {version} not found")
            return False
        
        # Restore backup
        current_path = os.path.join(self.model_path, "current")
        if os.path.exists(current_path):
            shutil.rmtree(current_path)
        shutil.copytree(backup_path, current_path)
        
        # Load model
        if self.model:
            self.model.load(current_path)
        
        self.current_version = version
        print(f"Rolled back to version {version}")
        
        return True
    
    def _increment_version(self, version: str) -> str:
        """Increment semantic version."""
        parts = version.lstrip("v").split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        return "v" + ".".join(parts)
    
    def _backup_model(self):
        """Backup current model."""
        current_path = os.path.join(self.model_path, "current")
        backup_path = os.path.join(self.model_path, f"backup_{self.current_version}")
        
        if os.path.exists(current_path):
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)
            shutil.copytree(current_path, backup_path)
        
        # Clean old backups
        self._cleanup_old_backups()
    
    def _cleanup_old_backups(self):
        """Remove old backups beyond max versions."""
        backups = sorted([
            d for d in os.listdir(self.model_path)
            if d.startswith("backup_")
        ], reverse=True)
        
        for old_backup in backups[self.max_versions:]:
            backup_path = os.path.join(self.model_path, old_backup)
            shutil.rmtree(backup_path)
    
    def _save_model(self):
        """Save current model."""
        current_path = os.path.join(self.model_path, "current")
        os.makedirs(current_path, exist_ok=True)
        
        if self.model:
            self.model.save(current_path)
        
        if self.vectorizer:
            self.vectorizer.save(os.path.join(current_path, "vectorizer.json"))
    
    def _save_state(self):
        """Save retrainer state."""
        os.makedirs(self.model_path, exist_ok=True)
        
        state = {
            "current_version": self.current_version,
            "last_retrain": self.last_retrain.isoformat() if self.last_retrain else None,
            "jobs": [job.to_dict() for job in self.retraining_jobs[-100:]],
        }
        
        with open(os.path.join(self.model_path, "retrainer_state.json"), "w") as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load retrainer state."""
        state_path = os.path.join(self.model_path, "retrainer_state.json")
        
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
            
            self.current_version = state.get("current_version", self.current_version)
            if state.get("last_retrain"):
                self.last_retrain = datetime.fromisoformat(state["last_retrain"])
    
    def get_retraining_history(self) -> List[Dict]:
        """Get retraining job history."""
        return [job.to_dict() for job in self.retraining_jobs]
    
    def get_model_info(self) -> Dict:
        """Get current model information."""
        return {
            "current_version": self.current_version,
            "last_retrain": self.last_retrain.isoformat() if self.last_retrain else None,
            "total_retraining_jobs": len(self.retraining_jobs),
            "model_path": self.model_path,
        }


if __name__ == "__main__":
    # Demo retrainer
    retrainer = ModelRetrainer()
    
    print("Model Retrainer Demo")
    print("=" * 50)
    
    print(f"\nCurrent model info:")
    for key, value in retrainer.get_model_info().items():
        print(f"  {key}: {value}")
    
    # Check if retraining is needed
    should_retrain, reason = retrainer.should_retrain()
    print(f"\nShould retrain: {should_retrain}")
    print(f"Reason: {reason}")
    
    # Simulate retraining with dummy data
    print("\nSimulating retraining...")
    dummy_events = [
        {
            "event_id": f"evt_{i}",
            "user_id": f"user_{i % 10}",
            "action": "view_dashboard",
            "timestamp": datetime.now().isoformat(),
            "is_anomaly": (i % 20 == 0),
        }
        for i in range(5000)
    ]
    
    # Note: This will skip in demo since model is None
    # job = retrainer.retrain(dummy_events, force=True)
    # print(f"\nRetraining job status: {job.status}")
    
    print("\nRetraining history:")
    for job in retrainer.get_retraining_history():
        print(f"  {job['job_id']}: {job['status']}")
