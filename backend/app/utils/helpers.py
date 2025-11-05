"""
General Helper Functions.

Common utility functions for formatting, validation, and data processing.
"""

import re
import secrets
import string
import time
import logging
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Callable, Any
from urllib.parse import urlparse

from fastapi import Request


logger = logging.getLogger(__name__)


# ============================================================
# FORMATTING UTILITIES
# ============================================================

def format_bytes(bytes_size: int, precision: int = 2) -> str:
    """
    Format bytes to human-readable string.
    
    Args:
        bytes_size: Size in bytes
        precision: Decimal precision
        
    Returns:
        Formatted string (e.g., "1.23 MB")
        
    Example:
        >>> format_bytes(1048576)
        '1.00 MB'
    """
    if bytes_size == 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    unit_index = 0
    size = float(bytes_size)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.{precision}f} {units[unit_index]}"


def format_number(number: float, precision: int = 2) -> str:
    """
    Format number with thousand separators.
    
    Args:
        number: Number to format
        precision: Decimal precision
        
    Returns:
        Formatted string (e.g., "1,234.56")
        
    Example:
        >>> format_number(1234567.89)
        '1,234,567.89'
    """
    return f"{number:,.{precision}f}"


def format_percentage(value: float, precision: int = 1) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Value between 0 and 1
        precision: Decimal precision
        
    Returns:
        Formatted percentage string (e.g., "45.5%")
        
    Example:
        >>> format_percentage(0.455)
        '45.5%'
    """
    return f"{value * 100:.{precision}f}%"


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate string to maximum length.
    
    Args:
        text: String to truncate
        max_length: Maximum length
        suffix: Suffix to append if truncated
        
    Returns:
        Truncated string
        
    Example:
        >>> truncate_string("Long text here", 10)
        'Long te...'
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


# ============================================================
# STRING UTILITIES
# ============================================================

def generate_random_string(
    length: int = 32,
    use_uppercase: bool = True,
    use_lowercase: bool = True,
    use_digits: bool = True,
    use_special: bool = False
) -> str:
    """
    Generate cryptographically secure random string.
    
    Args:
        length: String length
        use_uppercase: Include uppercase letters
        use_lowercase: Include lowercase letters
        use_digits: Include digits
        use_special: Include special characters
        
    Returns:
        Random string
        
    Example:
        >>> generate_random_string(16)
        'aB3dE5fG7hI9jK1l'
    """
    chars = ""
    
    if use_uppercase:
        chars += string.ascii_uppercase
    if use_lowercase:
        chars += string.ascii_lowercase
    if use_digits:
        chars += string.digits
    if use_special:
        chars += "!@#$%^&*"
    
    if not chars:
        raise ValueError("At least one character set must be enabled")
    
    return ''.join(secrets.choice(chars) for _ in range(length))


def sanitize_input(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize user input by removing dangerous characters.
    
    Args:
        text: Input text
        max_length: Optional maximum length
        
    Returns:
        Sanitized string
        
    Example:
        >>> sanitize_input("<script>alert('xss')</script>")
        'scriptalert(xss)/script'
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove SQL injection patterns
    text = re.sub(r'(--|;|\'|\")', '', text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    
    # Trim whitespace
    text = text.strip()
    
    # Limit length
    if max_length:
        text = text[:max_length]
    
    return text


# ============================================================
# VALIDATION UTILITIES
# ============================================================

def is_valid_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
        
    Example:
        >>> is_valid_email("user@example.com")
        True
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def is_valid_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid, False otherwise
        
    Example:
        >>> is_valid_url("https://example.com")
        True
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def validate_file_extension(filename: str, allowed_extensions: list[str]) -> bool:
    """
    Validate file extension.
    
    Args:
        filename: Filename to validate
        allowed_extensions: List of allowed extensions (e.g., ['.csv', '.xlsx'])
        
    Returns:
        True if valid, False otherwise
        
    Example:
        >>> validate_file_extension("data.csv", [".csv", ".xlsx"])
        True
    """
    if not filename:
        return False
    
    ext = filename.lower().split('.')[-1]
    return f".{ext}" in [e.lower() for e in allowed_extensions]


def get_mime_type(filename: str) -> str:
    """
    Get MIME type from filename.
    
    Args:
        filename: Filename
        
    Returns:
        MIME type string
        
    Example:
        >>> get_mime_type("data.csv")
        'text/csv'
    """
    mime_types = {
        '.csv': 'text/csv',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.xls': 'application/vnd.ms-excel',
        '.json': 'application/json',
        '.pdf': 'application/pdf',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.txt': 'text/plain',
        '.html': 'text/html',
        '.parquet': 'application/octet-stream',
    }
    
    ext = filename.lower().split('.')[-1]
    return mime_types.get(f".{ext}", 'application/octet-stream')


# ============================================================
# DATE UTILITIES
# ============================================================

def parse_date(date_string: str, formats: Optional[list[str]] = None) -> Optional[datetime]:
    """
    Parse date string with multiple format attempts.
    
    Args:
        date_string: Date string to parse
        formats: List of format strings to try
        
    Returns:
        Datetime object or None if parsing fails
        
    Example:
        >>> parse_date("2025-01-15")
        datetime.datetime(2025, 1, 15, 0, 0)
    """
    if formats is None:
        formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%d-%m-%Y',
            '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
        ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    
    return None


def calculate_age(birth_date: datetime) -> int:
    """
    Calculate age from birth date.
    
    Args:
        birth_date: Birth date
        
    Returns:
        Age in years
        
    Example:
        >>> calculate_age(datetime(2000, 1, 1))
        25
    """
    today = datetime.now()
    age = today.year - birth_date.year
    
    # Adjust if birthday hasn't occurred this year
    if today.month < birth_date.month or \
       (today.month == birth_date.month and today.day < birth_date.day):
        age -= 1
    
    return age


# ============================================================
# REQUEST UTILITIES
# ============================================================

def get_client_ip(request: Request) -> str:
    """
    Get client IP address from request.
    
    Args:
        request: FastAPI request object
        
    Returns:
        IP address string
        
    Example:
        >>> get_client_ip(request)
        '192.168.1.1'
    """
    # Check X-Forwarded-For header (proxy/load balancer)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    
    # Check X-Real-IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to direct client
    return request.client.host if request.client else "unknown"


# ============================================================
# DECORATORS
# ============================================================

def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Wrapped function
        
    Example:
        @timing_decorator
        def slow_function():
            time.sleep(1)
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        elapsed = end_time - start_time
        logger.info(f"{func.__name__} took {elapsed:.4f} seconds")
        
        return result
    
    return wrapper


def retry_decorator(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator to retry function on failure with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Decorator function
        
    Example:
        @retry_decorator(max_retries=3, delay=1.0)
        def unstable_api_call():
            # API call that might fail
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts")
                        raise
                    
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {str(e)}. "
                        f"Retrying in {current_delay}s..."
                    )
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        
        return wrapper
    return decorator
