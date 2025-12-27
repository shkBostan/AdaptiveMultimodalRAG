"""
Logging module for AdaptiveMultimodalRAG.

This module provides centralized logging functionality including:
- Configuration and setup
- Structured JSON formatting
- Correlation ID context management
- Data sanitization for security

Author: s Bostan
Created on: Nov, 2025
"""

from .config import setup_logging, get_logger
from .formatter import JSONFormatter, CorrelationIDFilter, SafeFormatter
from .context import (
    initialize_context,
    get_correlation_id,
    set_correlation_id,
    clear_context,
    get_context,
    get_user_id,
    get_request_id
)
from .sanitizer import (
    sanitize_api_key,
    sanitize_token,
    sanitize_email,
    sanitize_card_number,
    sanitize_ssn,
    sanitize_password,
    sanitize_phone_number,
    might_be_sensitive
)
from .middleware import CorrelationIDMiddleware

__all__ = [
    # Configuration
    'setup_logging',
    'get_logger',
    # Formatters
    'SafeFormatter',
    'JSONFormatter',
    'CorrelationIDFilter',
    # Context
    'initialize_context',
    'get_correlation_id',
    'set_correlation_id',
    'clear_context',
    'get_context',
    'get_user_id',
    'get_request_id',
    # Sanitization
    'sanitize_api_key',
    'sanitize_token',
    'sanitize_email',
    'sanitize_card_number',
    'sanitize_ssn',
    'sanitize_password',
    'sanitize_phone_number',
    'might_be_sensitive',
    # Middleware
    'CorrelationIDMiddleware',
]

