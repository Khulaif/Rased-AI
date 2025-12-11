"""
Rased (راصد) - Message Queue Interface
Abstract message queue for asynchronous event processing.
"""

from __future__ import absolute_import
import queue as stdlib_queue  # Rename to avoid conflict
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import json
import os

# Add parent path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import config


@dataclass
class QueueMessage:
    """Message wrapper for queue processing."""
    message_id: str
    payload: Dict
    timestamp: datetime
    priority: int = 0  # Higher = more urgent
    retries: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict:
        return {
            "message_id": self.message_id,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
            "retries": self.retries,
        }


class MessageQueue(ABC):
    """
    Abstract base class for message queue implementations.
    
    Supports different backends:
    - In-memory (for development/demo)
    - Apache Kafka (for production)
    - RabbitMQ (alternative production option)
    """
    
    @abstractmethod
    def publish(self, message: QueueMessage) -> bool:
        """Publish a message to the queue."""
        pass
    
    @abstractmethod
    def consume(self, timeout: float = 1.0) -> Optional[QueueMessage]:
        """Consume a message from the queue."""
        pass
    
    @abstractmethod
    def acknowledge(self, message_id: str) -> bool:
        """Acknowledge successful processing of a message."""
        pass
    
    @abstractmethod
    def reject(self, message_id: str, requeue: bool = True) -> bool:
        """Reject a message (optionally requeue for retry)."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get current queue size."""
        pass
    
    @abstractmethod
    def close(self):
        """Close the queue connection."""
        pass


class InMemoryQueue(MessageQueue):
    """
    In-memory message queue for development and testing.
    
    Features:
    - Thread-safe operations
    - Priority queue support
    - Message acknowledgment tracking
    - Automatic retry logic
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize in-memory queue.
        
        Args:
            max_size: Maximum queue size
        """
        self.max_size = max_size
        self._queue = stdlib_queue.PriorityQueue(maxsize=max_size)
        self._pending: Dict[str, QueueMessage] = {}  # Messages being processed
        self._lock = threading.Lock()
        self._message_counter = 0
        self._stats = {
            "published": 0,
            "consumed": 0,
            "acknowledged": 0,
            "rejected": 0,
            "requeued": 0,
        }
    
    def publish(self, message: QueueMessage) -> bool:
        """
        Publish a message to the queue.
        
        Args:
            message: QueueMessage to publish
        
        Returns:
            True if successful, False if queue is full
        """
        try:
            # Priority queue uses (priority, counter, message) tuple
            # Negative priority for higher-priority-first ordering
            with self._lock:
                self._message_counter += 1
                counter = self._message_counter
            
            self._queue.put(
                (-message.priority, counter, message),
                block=False
            )
            
            with self._lock:
                self._stats["published"] += 1
            
            return True
            
        except stdlib_queue.Full:
            return False
    
    def consume(self, timeout: float = 1.0) -> Optional[QueueMessage]:
        """
        Consume a message from the queue.
        
        Args:
            timeout: Seconds to wait for a message
        
        Returns:
            QueueMessage or None if timeout
        """
        try:
            _, _, message = self._queue.get(timeout=timeout)
            
            with self._lock:
                self._pending[message.message_id] = message
                self._stats["consumed"] += 1
            
            return message
            
        except stdlib_queue.Empty:
            return None
    
    def acknowledge(self, message_id: str) -> bool:
        """
        Acknowledge successful processing.
        
        Args:
            message_id: ID of processed message
        
        Returns:
            True if message was pending
        """
        with self._lock:
            if message_id in self._pending:
                del self._pending[message_id]
                self._stats["acknowledged"] += 1
                return True
            return False
    
    def reject(self, message_id: str, requeue: bool = True) -> bool:
        """
        Reject a message.
        
        Args:
            message_id: ID of message to reject
            requeue: Whether to requeue for retry
        
        Returns:
            True if message was pending
        """
        with self._lock:
            if message_id not in self._pending:
                return False
            
            message = self._pending[message_id]
            del self._pending[message_id]
            self._stats["rejected"] += 1
            
            if requeue and message.retries < message.max_retries:
                # Requeue with incremented retry count
                message.retries += 1
                self._stats["requeued"] += 1
                
        # Requeue outside the lock
        if requeue and message.retries <= message.max_retries:
            return self.publish(message)
        
        return True
    
    def size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()
    
    def pending_count(self) -> int:
        """Get number of messages being processed."""
        with self._lock:
            return len(self._pending)
    
    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        with self._lock:
            return {
                **self._stats,
                "queue_size": self._queue.qsize(),
                "pending_count": len(self._pending),
            }
    
    def close(self):
        """Close the queue (no-op for in-memory)."""
        pass


class KafkaQueue(MessageQueue):
    """
    Apache Kafka message queue implementation.
    
    Note: This is a placeholder implementation. For production use,
    install confluent-kafka or kafka-python and implement properly.
    """
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "rased_events",
        consumer_group: str = "rased_inference",
    ):
        """
        Initialize Kafka queue.
        
        Args:
            bootstrap_servers: Kafka broker addresses
            topic: Topic name
            consumer_group: Consumer group ID
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.consumer_group = consumer_group
        
        # Placeholder - in production, initialize actual Kafka clients
        self._producer = None
        self._consumer = None
        self._pending: Dict[str, QueueMessage] = {}
        
        print(f"[PLACEHOLDER] Kafka queue configured for {bootstrap_servers}/{topic}")
        print("For production, install 'confluent-kafka' and implement actual Kafka integration")
        
        # Fall back to in-memory for demo
        self._fallback = InMemoryQueue()
    
    def publish(self, message: QueueMessage) -> bool:
        """Publish to Kafka (falls back to in-memory)."""
        # In production: serialize and send to Kafka
        return self._fallback.publish(message)
    
    def consume(self, timeout: float = 1.0) -> Optional[QueueMessage]:
        """Consume from Kafka (falls back to in-memory)."""
        # In production: poll Kafka consumer
        return self._fallback.consume(timeout)
    
    def acknowledge(self, message_id: str) -> bool:
        """Acknowledge (commit offset in Kafka)."""
        # In production: commit consumer offset
        return self._fallback.acknowledge(message_id)
    
    def reject(self, message_id: str, requeue: bool = True) -> bool:
        """Reject message."""
        # In production: handle based on dead-letter queue strategy
        return self._fallback.reject(message_id, requeue)
    
    def size(self) -> int:
        """Get queue size (approximate for Kafka)."""
        return self._fallback.size()
    
    def close(self):
        """Close Kafka connections."""
        if self._producer:
            self._producer.flush()
        if self._consumer:
            self._consumer.close()
        self._fallback.close()


def create_queue(queue_type: str = "memory") -> MessageQueue:
    """
    Factory function to create appropriate queue type.
    
    Args:
        queue_type: "memory", "kafka", or "rabbitmq"
    
    Returns:
        MessageQueue instance
    """
    if queue_type == "memory":
        return InMemoryQueue(max_size=config.queue.max_queue_size)
    elif queue_type == "kafka":
        return KafkaQueue(
            bootstrap_servers=config.queue.kafka_bootstrap_servers,
            topic=config.queue.kafka_topic,
            consumer_group=config.queue.kafka_consumer_group,
        )
    else:
        raise ValueError(f"Unknown queue type: {queue_type}")


if __name__ == "__main__":
    # Demo queue usage
    q = create_queue("memory")
    
    # Publish messages
    for i in range(5):
        msg = QueueMessage(
            message_id=f"msg_{i}",
            payload={"action": "test", "index": i},
            timestamp=datetime.now(),
            priority=i,  # Higher index = higher priority
        )
        q.publish(msg)
        print(f"Published: {msg.message_id} (priority {msg.priority})")
    
    print(f"\nQueue size: {q.size()}")
    
    # Consume messages (should come in priority order)
    print("\nConsuming messages:")
    while q.size() > 0:
        msg = q.consume(timeout=0.1)
        if msg:
            print(f"  Consumed: {msg.message_id} (priority {msg.priority})")
            q.acknowledge(msg.message_id)
    
    print(f"\nFinal stats: {q.get_stats() if hasattr(q, 'get_stats') else 'N/A'}")
