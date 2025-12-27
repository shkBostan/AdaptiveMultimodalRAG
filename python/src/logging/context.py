"""
Context management for correlation IDs and other logging context.

Author: s Bostan
Created on: Nov, 2025
"""

import contextvars
import uuid
from typing import Optional

# Context variables for logging
correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'correlation_id', default=None
)
user_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'user_id', default=None
)
request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'request_id', default=None
)


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def set_correlation_id(correlation_id_value: Optional[str]) -> None:
    """Set correlation ID in context."""
    correlation_id.set(correlation_id_value)


def get_correlation_id() -> Optional[str]:
    """Get correlation ID from context."""
    return correlation_id.get(None)


def set_user_id(user_id_value: Optional[str]) -> None:
    """Set user ID in context."""
    user_id.set(user_id_value)


def get_user_id() -> Optional[str]:
    """Get user ID from context."""
    return user_id.get(None)


def set_request_id(request_id_value: Optional[str]) -> None:
    """Set request ID in context."""
    request_id.set(request_id_value)


def get_request_id() -> Optional[str]:
    """Get request ID from context."""
    return request_id.get(None)


def initialize_context(
    correlation_id_value: Optional[str] = None,
    user_id_value: Optional[str] = None,
    request_id_value: Optional[str] = None
) -> str:
    """Initialize logging context for a request."""
    if correlation_id_value is None:
        correlation_id_value = generate_correlation_id()
    
    set_correlation_id(correlation_id_value)
    
    if user_id_value:
        set_user_id(user_id_value)
    
    if request_id_value:
        set_request_id(request_id_value)
    
    return correlation_id_value


def clear_context() -> None:
    """Clear all context variables."""
    correlation_id.set(None)
    user_id.set(None)
    request_id.set(None)


def get_context() -> dict:
    """Get all context as a dictionary."""
    return {
        'correlation_id': get_correlation_id(),
        'user_id': get_user_id(),
        'request_id': get_request_id()
    }

