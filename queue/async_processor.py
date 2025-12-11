"""
Rased (راصد) - Async Event Processor
Background processing of events using message queue.
"""

from __future__ import absolute_import
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import os
import sys

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import config
from queue.message_queue import MessageQueue, QueueMessage, create_queue


@dataclass
class ProcessingResult:
    """Result of event processing."""
    message_id: str
    success: bool
    risk_score: int = 0
    decision: str = ""
    processing_time_ms: float = 0.0
    error: Optional[str] = None


class AsyncProcessor:
    """
    Asynchronous event processor using message queue.
    
    Features:
    - Background worker pool
    - Automatic retry on failure
    - Backpressure handling
    - Statistics tracking
    """
    
    def __init__(
        self,
        queue: Optional[MessageQueue] = None,
        num_workers: int = 4,
    ):
        """
        Initialize async processor.
        
        Args:
            queue: MessageQueue instance (creates default if None)
            num_workers: Number of worker threads
        """
        self.queue = queue or create_queue(config.queue.queue_type)
        self.num_workers = num_workers
        self._workers: List[threading.Thread] = []
        self._running = False
        self._inference_callback: Optional[Callable] = None
        self._results: Dict[str, ProcessingResult] = {}
        self._lock = threading.Lock()
        self._stats = {
            "events_submitted": 0,
            "events_processed": 0,
            "events_failed": 0,
        }
    
    def set_inference_callback(self, callback: Callable[[Dict], ProcessingResult]):
        """
        Set the callback function for processing events.
        
        Args:
            callback: Function that takes an event dict and returns ProcessingResult
        """
        self._inference_callback = callback
    
    def start(self):
        """Start the processor workers."""
        if self._running:
            return
        
        self._running = True
        
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"AsyncProcessor-Worker-{i}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)
        
        print(f"Started {self.num_workers} async processor workers")
    
    def stop(self):
        """Stop the processor workers."""
        self._running = False
        
        for worker in self._workers:
            worker.join(timeout=2.0)
        
        self._workers.clear()
        print("Async processor stopped")
    
    def submit_event(self, event: Dict) -> str:
        """
        Submit an event for async processing.
        
        Args:
            event: Event data to process
        
        Returns:
            Message ID for tracking
        
        Raises:
            RuntimeError: If processor is not running or queue is full
        """
        if not self._running:
            raise RuntimeError("Async processor is not running")
        
        import uuid
        message_id = event.get("event_id") or f"msg_{uuid.uuid4().hex[:12]}"
        
        message = QueueMessage(
            message_id=message_id,
            payload=event,
            timestamp=datetime.now(),
            priority=self._calculate_priority(event),
        )
        
        success = self.queue.publish(message)
        if not success:
            raise RuntimeError("Queue is full, cannot accept more events")
        
        with self._lock:
            self._stats["events_submitted"] += 1
        
        return message_id
    
    def get_result(self, message_id: str) -> Optional[ProcessingResult]:
        """
        Get the result for a processed message.
        
        Args:
            message_id: ID of the message
        
        Returns:
            ProcessingResult if available, None otherwise
        """
        with self._lock:
            return self._results.get(message_id)
    
    def get_stats(self) -> Dict:
        """Get processor statistics."""
        with self._lock:
            return {
                **self._stats,
                "queue_size": self.queue.size(),
                "workers_active": len([w for w in self._workers if w.is_alive()]),
            }
    
    def _worker_loop(self):
        """Main worker loop for processing events."""
        while self._running:
            try:
                # Consume message from queue
                message = self.queue.consume(timeout=0.5)
                
                if message is None:
                    continue
                
                # Process the event
                result = self._process_message(message)
                
                # Store result
                with self._lock:
                    self._results[message.message_id] = result
                    
                    if result.success:
                        self._stats["events_processed"] += 1
                    else:
                        self._stats["events_failed"] += 1
                
                # Acknowledge or reject
                if result.success:
                    self.queue.acknowledge(message.message_id)
                else:
                    self.queue.reject(message.message_id, requeue=True)
                    
            except Exception as e:
                print(f"Worker error: {e}")
                time.sleep(0.1)
    
    def _process_message(self, message: QueueMessage) -> ProcessingResult:
        """
        Process a single message.
        
        Args:
            message: QueueMessage to process
        
        Returns:
            ProcessingResult
        """
        start_time = time.time()
        
        try:
            if self._inference_callback:
                result = self._inference_callback(message.payload)
                result.message_id = message.message_id
                return result
            else:
                # Default processing - just mark as successful
                processing_time = (time.time() - start_time) * 1000
                return ProcessingResult(
                    message_id=message.message_id,
                    success=True,
                    processing_time_ms=processing_time,
                )
                
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return ProcessingResult(
                message_id=message.message_id,
                success=False,
                error=str(e),
                processing_time_ms=processing_time,
            )
    
    def _calculate_priority(self, event: Dict) -> int:
        """
        Calculate message priority based on event characteristics.
        
        Higher priority for:
        - High-value transactions
        - Sensitive actions
        - Known risky patterns
        """
        priority = 0
        
        # Transaction amount
        amount = event.get("transaction_amount", 0)
        if amount > 100000:
            priority += 3
        elif amount > 10000:
            priority += 2
        elif amount > 1000:
            priority += 1
        
        # Sensitive actions
        action = event.get("action", "")
        if action in ["confirm_ownership_transfer", "change_password", "add_new_owner"]:
            priority += 2
        
        # Risk indicators
        if event.get("is_tor", False):
            priority += 2
        if event.get("is_vpn", False):
            priority += 1
        
        return priority


if __name__ == "__main__":
    # Demo
    processor = AsyncProcessor(num_workers=2)
    
    # Set a simple callback
    def demo_callback(event: Dict) -> ProcessingResult:
        time.sleep(0.1)  # Simulate processing
        return ProcessingResult(
            message_id="",
            success=True,
            risk_score=42,
            decision="allow",
            processing_time_ms=100.0,
        )
    
    processor.set_inference_callback(demo_callback)
    processor.start()
    
    # Submit some events
    for i in range(5):
        msg_id = processor.submit_event({
            "user_id": f"user_{i}",
            "action": "test",
            "transaction_amount": i * 100,
        })
        print(f"Submitted: {msg_id}")
    
    # Wait and check results
    time.sleep(1)
    print(f"\nStats: {processor.get_stats()}")
    
    processor.stop()
