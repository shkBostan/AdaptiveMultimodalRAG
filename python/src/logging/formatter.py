"""
JSON formatter for structured logging.

Author: s Bostan
Created on: Nov, 2025
"""

import json
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, Optional


class SafeFormatter(logging.Formatter):
    """Formatter that safely handles missing attributes like correlationId."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record, handling missing attributes gracefully."""
        # Ensure correlation_id is set if not present
        if not hasattr(record, 'correlation_id'):
            # Try to get from context
            try:
                from .context import get_correlation_id
                correlation_id = get_correlation_id()
                if correlation_id:
                    record.correlation_id = correlation_id
            except Exception:
                pass
        
        # Set correlationId (camelCase) for format string compatibility
        if hasattr(record, 'correlation_id') and record.correlation_id:
            record.correlationId = record.correlation_id
        elif not hasattr(record, 'correlationId'):
            record.correlationId = '-'
        
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add correlation ID if present
        if hasattr(record, 'correlation_id') and record.correlation_id:
            log_data["correlationId"] = record.correlation_id
        
        # Add user ID if present
        if hasattr(record, 'user_id') and record.user_id:
            log_data["userId"] = record.user_id
        
        # Add request ID if present
        if hasattr(record, 'request_id') and record.request_id:
            log_data["requestId"] = record.request_id
        
        # Add thread name
        if record.threadName:
            log_data["threadName"] = record.threadName
        
        # Add process ID
        log_data["processId"] = record.process
        
        # Add exception information if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "stackTrace": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in [
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'message', 'pathname', 'process', 'processName', 'relativeCreated',
                'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
                'correlation_id', 'correlationId', 'user_id', 'request_id'
            ]:
                log_data[key] = value
        
        return json.dumps(log_data, default=str)


class CorrelationIDFilter(logging.Filter):
    """Filter to add correlation ID to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID from context if available."""
        from .context import get_correlation_id
        
        correlation_id = get_correlation_id()
        if correlation_id:
            record.correlation_id = correlation_id
        
        return True
