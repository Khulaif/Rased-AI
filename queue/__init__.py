# Queue package - uses lazy imports to avoid conflicts with standard library
from __future__ import absolute_import

def get_message_queue():
    """Lazy import to avoid circular imports."""
    from .message_queue import MessageQueue, InMemoryQueue
    return MessageQueue, InMemoryQueue

def get_async_processor():
    """Lazy import async processor."""
    from .async_processor import AsyncProcessor, ProcessingResult
    return AsyncProcessor, ProcessingResult
