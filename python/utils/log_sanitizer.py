"""
Utilities for sanitizing sensitive data before logging.

Author: s Bostan
Created on: Nov, 2025
"""

from typing import Optional

MASK = "***"
VISIBLE_CHARS = 4


def sanitize_api_key(api_key: Optional[str]) -> str:
    """Sanitize API key - shows first 4 and last 4 characters."""
    if not api_key or len(api_key) <= VISIBLE_CHARS * 2:
        return MASK
    return f"{api_key[:VISIBLE_CHARS]}{MASK}{api_key[-VISIBLE_CHARS:]}"


def sanitize_token(token: Optional[str]) -> str:
    """Sanitize token - shows first 4 characters only."""
    if not token or len(token) <= VISIBLE_CHARS:
        return MASK
    return f"{token[:VISIBLE_CHARS]}{MASK}"


def sanitize_email(email: Optional[str]) -> str:
    """Sanitize email address - shows first 2 characters of local part."""
    if not email or '@' not in email:
        return MASK
    
    try:
        local, domain = email.split('@', 1)
        visible_chars = min(2, len(local))
        return f"{local[:visible_chars]}{MASK}@{domain}"
    except Exception:
        return MASK


def sanitize_card_number(card_number: Optional[str]) -> str:
    """Sanitize credit card number - shows last 4 digits only."""
    if not card_number:
        return MASK
    
    # Remove spaces and dashes
    cleaned = ''.join(c for c in card_number if c.isdigit())
    if len(cleaned) < 4:
        return MASK
    
    return f"{MASK}{cleaned[-4:]}"


def sanitize_ssn(ssn: Optional[str]) -> str:
    """Sanitize SSN - shows last 4 digits only."""
    if not ssn:
        return MASK
    
    cleaned = ''.join(c for c in ssn if c.isdigit())
    if len(cleaned) < 4:
        return MASK
    
    return f"{MASK}{cleaned[-4:]}"


def sanitize_password(password: Optional[str]) -> str:
    """Sanitize password - always returns mask."""
    return MASK


def sanitize_phone_number(phone_number: Optional[str]) -> str:
    """Sanitize phone number - shows last 4 digits."""
    if not phone_number:
        return MASK
    
    cleaned = ''.join(c for c in phone_number if c.isdigit())
    if len(cleaned) < 4:
        return MASK
    
    return f"{MASK}{cleaned[-4:]}"


def might_be_sensitive(value: Optional[str]) -> bool:
    """Check if a string might contain sensitive data."""
    if not value:
        return False
    
    lower = value.lower()
    sensitive_keywords = ['password', 'secret', 'key', 'token', 'api', 'auth', 'credential']
    return any(keyword in lower for keyword in sensitive_keywords)

