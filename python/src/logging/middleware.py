"""
FastAPI middleware for correlation ID handling.

Author: s Bostan
Created on: Nov, 2025
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .context import initialize_context, get_correlation_id, clear_context

logger = logging.getLogger(__name__)


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Middleware to handle correlation IDs for request tracking."""
    
    CORRELATION_ID_HEADER = "X-Correlation-ID"
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with correlation ID."""
        # Get correlation ID from header or generate new one
        correlation_id = request.headers.get(self.CORRELATION_ID_HEADER)
        if not correlation_id:
            correlation_id = initialize_context()
        else:
            initialize_context(correlation_id)
        
        # Add correlation ID to request state
        request.state.correlation_id = correlation_id
        
        # Log request start
        start_time = time.time()
        logger.info(
            "Request started",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else None
            }
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Add correlation ID to response header
            response.headers[self.CORRELATION_ID_HEADER] = correlation_id
            
            # Log request completion
            duration = time.time() - start_time
            logger.info(
                "Request completed",
                extra={
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration * 1000
                }
            )
            
            return response
            
        except Exception as e:
            # Log request error
            duration = time.time() - start_time
            logger.error(
                "Request failed",
                extra={
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration_ms": duration * 1000
                },
                exc_info=True
            )
            raise
        
        finally:
            # Clean up context
            clear_context()

