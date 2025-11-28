"""
Examples demonstrating proper logging practices in Python.

Author: s Bostan
Created on: Nov, 2025
"""

import logging
import time
from typing import Dict, Any

from python.utils.logger import get_logger
from python.utils.logging_context import initialize_context, get_correlation_id
from python.utils.log_sanitizer import sanitize_api_key, sanitize_email

logger = get_logger(__name__)


def process_document(document_id: str, user_id: str):
    """Example: Service method with proper logging."""
    # Entry log with context
    logger.info(
        "Processing document",
        extra={
            "document_id": document_id,
            "user_id": user_id,
            "action": "process_document"
        }
    )
    
    start_time = time.time()
    
    try:
        # Debug log for detailed information
        logger.debug(
            "Retrieving document from repository",
            extra={"document_id": document_id}
        )
        
        # Simulate business logic
        content = retrieve_document(document_id)
        
        # Info log for important business events
        logger.info(
            "Document retrieved successfully",
            extra={
                "document_id": document_id,
                "content_length": len(content)
            }
        )
        
        # Process document
        processed = process_content(content)
        
        duration = time.time() - start_time
        
        # Exit log with metrics
        logger.info(
            "Document processing completed",
            extra={
                "document_id": document_id,
                "duration_ms": duration * 1000,
                "status": "success"
            }
        )
        
        return processed
        
    except DocumentNotFoundException as e:
        logger.warning(
            "Document not found",
            extra={
                "document_id": document_id,
                "error": str(e)
            },
            exc_info=True
        )
        raise
        
    except Exception as e:
        duration = time.time() - start_time
        
        # Error log with full context
        logger.error(
            "Document processing failed",
            extra={
                "document_id": document_id,
                "user_id": user_id,
                "duration_ms": duration * 1000,
                "error": str(e),
                "error_type": type(e).__name__
            },
            exc_info=True  # Include stack trace
        )
        raise ProcessingException("Failed to process document") from e


async def handle_api_request(request_id: str, user_id: str, query: str) -> Dict[str, Any]:
    """Example: API endpoint logging."""
    # Initialize logging context
    correlation_id = initialize_context(None, user_id, request_id)
    
    logger.info(
        "API request received",
        extra={
            "endpoint": "/api/query",
            "method": "POST",
            "user_id": user_id,
            "query_length": len(query),
            "correlation_id": correlation_id
        }
    )
    
    start_time = time.time()
    
    try:
        # Process request
        result = await process_query(query)
        
        duration = time.time() - start_time
        
        logger.info(
            "API request completed successfully",
            extra={
                "request_id": request_id,
                "correlation_id": correlation_id,
                "result_size": len(result),
                "duration_ms": duration * 1000
            }
        )
        
        return result
        
    except Exception as e:
        duration = time.time() - start_time
        
        logger.error(
            "API request failed",
            extra={
                "request_id": request_id,
                "user_id": user_id,
                "correlation_id": correlation_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "duration_ms": duration * 1000
            },
            exc_info=True
        )
        raise
    
    finally:
        from python.utils.logging_context import clear_context
        clear_context()


def authenticate_user(username: str, password: str, api_key: str):
    """Example: Logging with sensitive data sanitization."""
    # NEVER log passwords
    # BAD: logger.info("Login attempt", extra={"password": password})
    
    # GOOD: Log only safe information
    logger.info(
        "User login attempt",
        extra={
            "username": username,
            "api_key": sanitize_api_key(api_key)
        }
    )
    
    # Simulate authentication
    authenticated = perform_authentication(username, password)
    
    if authenticated:
        logger.info(
            "User authenticated successfully",
            extra={"username": username}
        )
    else:
        logger.warning(
            "Authentication failed",
            extra={"username": username}
        )


async def process_async_operation(task_id: str):
    """Example: Async operation with correlation ID propagation."""
    # Get current correlation ID
    correlation_id = get_correlation_id()
    
    logger.info(
        "Starting async operation",
        extra={
            "task_id": task_id,
            "correlation_id": correlation_id
        }
    )
    
    try:
        logger.debug(
            "Async operation executing",
            extra={"task_id": task_id}
        )
        
        # Perform async work
        await perform_async_work(task_id)
        
        logger.info(
            "Async operation completed",
            extra={"task_id": task_id}
        )
        
    except Exception as e:
        logger.error(
            "Async operation failed",
            extra={
                "task_id": task_id,
                "error": str(e)
            },
            exc_info=True
        )
        raise


def process_batch(items: list):
    """Example: Performance logging."""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Processing batch",
            extra={"batch_size": len(items)}
        )
    
    start_time = time.time()
    processed = 0
    
    for item in items:
        try:
            process_item(item)
            processed += 1
        except Exception as e:
            logger.warning(
                "Failed to process item",
                extra={
                    "item": item,
                    "error": str(e)
                },
                exc_info=True
            )
    
    duration = time.time() - start_time
    
    logger.info(
        "Batch processing completed",
        extra={
            "total_items": len(items),
            "processed_items": processed,
            "failed_items": len(items) - processed,
            "duration_ms": duration * 1000,
            "items_per_second": len(items) / duration if duration > 0 else 0
        }
    )


# Placeholder functions
def retrieve_document(doc_id: str) -> str:
    return "content"

def process_content(content: str) -> str:
    return content

async def process_query(query: str) -> Dict[str, Any]:
    return {"result": "processed"}

def perform_authentication(username: str, password: str) -> bool:
    return True

async def perform_async_work(task_id: str):
    pass

def process_item(item: str):
    pass

# Placeholder exceptions
class DocumentNotFoundException(Exception):
    pass

class ProcessingException(Exception):
    pass

