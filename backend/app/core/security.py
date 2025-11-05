"""
Security utilities for authentication and authorization.

Includes JWT token management, password hashing, and security helpers.
"""

import secrets
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Union

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from fastapi.security import OAuth2PasswordBearer, HTTPBearer

from app.core.config import settings

logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# ============================================================
# PASSWORD HASHING
# ============================================================

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,
)


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    ✅ CRITICAL: This function is called by auth.py endpoints.
    
    Args:
        password: Plain text password to hash
        
    Returns:
        Hashed password string
    """
    return pwd_context.hash(password)


def get_password_hash(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Alias for hash_password for backward compatibility.
    
    Args:
        password: Plain text password to hash
        
    Returns:
        Hashed password string
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password.
    
    Args:
        plain_password: The plain text password to verify
        hashed_password: The hashed password to compare against
        
    Returns:
        True if password matches, False otherwise
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        return False


def validate_password_strength(password: str) -> tuple[bool, Optional[str]]:
    """
    Validate password meets security requirements.
    
    Checks:
    - Minimum length (configurable)
    - Uppercase letters (configurable)
    - Lowercase letters (configurable)
    - Digits (configurable)
    - Special characters (configurable)
    
    Args:
        password: Password to validate
        
    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str])
    """
    min_length = getattr(settings, 'PASSWORD_MIN_LENGTH', 8)
    
    if len(password) < min_length:
        return False, f"Password must be at least {min_length} characters"
    
    if getattr(settings, 'PASSWORD_REQUIRE_UPPERCASE', True) and not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    
    if getattr(settings, 'PASSWORD_REQUIRE_LOWERCASE', True) and not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    
    if getattr(settings, 'PASSWORD_REQUIRE_DIGITS', True) and not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit"
    
    if getattr(settings, 'PASSWORD_REQUIRE_SPECIAL', False):
        special_chars = set("!@#$%^&*()_+-=[]{}|;:,.<>?")
        if not any(c in special_chars for c in password):
            return False, "Password must contain at least one special character"
    
    return True, None


# ============================================================
# JWT TOKEN MANAGEMENT
# ============================================================

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login",
    auto_error=True
)

http_bearer = HTTPBearer(auto_error=True)


def create_access_token(
    subject: Union[str, int] = None,
    expires_delta: Optional[timedelta] = None,
    additional_claims: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a JWT access token.
    
    Supports two calling patterns:
    1. create_access_token(subject="user_id") - Simple pattern
    2. create_access_token(data={"sub": "user_id"}) - Advanced pattern
    
    Args:
        subject: The subject (typically user ID)
        expires_delta: Optional custom expiration time
        additional_claims: Optional additional data to include in token
        data: Optional complete data dict to encode
        
    Returns:
        Encoded JWT token string
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    # Support both calling patterns
    if data:
        to_encode = data.copy()
        # Ensure type is set if not present
        if "type" not in to_encode:
            to_encode["type"] = "access"
    else:
        to_encode = {
            "sub": str(subject) if subject else "",
            "type": "access",
        }
    
    # Update with standard claims
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    })
    
    # Add any additional claims
    if additional_claims:
        to_encode.update(additional_claims)
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=ALGORITHM
    )
    return encoded_jwt


def create_refresh_token(
    subject: Union[str, int] = None,
    expires_delta: Optional[timedelta] = None,
    data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a JWT refresh token with token rotation support.
    
    Supports two calling patterns:
    1. create_refresh_token(subject="user_id") - Simple pattern
    2. create_refresh_token(data={"sub": "user_id"}) - Advanced pattern
    
    Args:
        subject: The subject (typically user ID)
        expires_delta: Optional custom expiration time
        data: Optional complete data dict to encode
        
    Returns:
        Encoded JWT refresh token string
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            days=getattr(settings, 'REFRESH_TOKEN_EXPIRE_DAYS', 7)
        )
    
    # Support both calling patterns
    if data:
        to_encode = data.copy()
        # Ensure type is set if not present
        if "type" not in to_encode:
            to_encode["type"] = "refresh"
    else:
        to_encode = {
            "sub": str(subject) if subject else "",
            "type": "refresh",
        }
    
    # Update with standard claims
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": secrets.token_urlsafe(32),  # Unique token ID for revocation
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=ALGORITHM
    )
    return encoded_jwt


def decode_token(token: str, token_type: str = "access") -> Dict[str, Any]:
    """
    Decode and validate a JWT token.
    
    Args:
        token: JWT token string to decode
        token_type: Expected token type ('access' or 'refresh')
        
    Returns:
        Decoded token payload dictionary
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        
        # Verify token type
        token_type_claim = payload.get("type")
        if token_type_claim != token_type:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token type. Expected {token_type}, got {token_type_claim}",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return payload
        
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def create_email_verification_token(email: str) -> str:
    """
    Create a token for email verification.
    
    Args:
        email: User's email address
        
    Returns:
        Encoded verification token
    """
    expire_hours = getattr(settings, 'EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS', 24)
    delta = timedelta(hours=expire_hours)
    expire = datetime.now(timezone.utc) + delta
    
    to_encode = {
        "exp": expire,
        "sub": email,
        "type": "email_verification",
        "iat": datetime.now(timezone.utc),
    }
    
    return jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=ALGORITHM
    )


def create_password_reset_token(email: str) -> str:
    """
    Create a token for password reset.
    
    Args:
        email: User's email address
        
    Returns:
        Encoded reset token
    """
    expire_hours = getattr(settings, 'EMAIL_RESET_TOKEN_EXPIRE_HOURS', 24)
    delta = timedelta(hours=expire_hours)
    expire = datetime.now(timezone.utc) + delta
    
    to_encode = {
        "exp": expire,
        "sub": email,
        "type": "password_reset",
        "iat": datetime.now(timezone.utc),
    }
    
    return jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=ALGORITHM
    )


def verify_email_token(token: str) -> Optional[str]:
    """
    Verify email verification token and extract email.
    
    Args:
        token: Verification token
        
    Returns:
        Email address if valid, None otherwise
    """
    try:
        payload = decode_token(token, token_type="email_verification")
        return payload.get("sub")
    except HTTPException:
        return None


def verify_password_reset_token(token: str) -> Optional[str]:
    """
    Verify password reset token and extract email.
    
    Args:
        token: Reset token
        
    Returns:
        Email address if valid, None otherwise
    """
    try:
        payload = decode_token(token, token_type="password_reset")
        return payload.get("sub")
    except HTTPException:
        return None


# ============================================================
# API KEY GENERATION
# ============================================================


def generate_api_key(prefix: str = "sk") -> str:
    """
    Generate a secure API key.
    
    Args:
        prefix: Prefix for the API key (default: 'sk' for secret key)
        
    Returns:
        Generated API key string (e.g., "sk_xxxxxxxxxxxxx")
    """
    random_part = secrets.token_urlsafe(32)
    return f"{prefix}_{random_part}"


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key for secure storage.
    
    Args:
        api_key: Plain API key
        
    Returns:
        Hashed API key
    """
    return get_password_hash(api_key)


def verify_api_key(plain_key: str, hashed_key: str) -> bool:
    """
    Verify an API key against stored hash.
    
    Args:
        plain_key: Plain API key to verify
        hashed_key: Stored hashed API key
        
    Returns:
        True if keys match, False otherwise
    """
    return verify_password(plain_key, hashed_key)


# ============================================================
# SECURITY HELPERS
# ============================================================


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a secure random token.
    
    Args:
        length: Length of the token in bytes
        
    Returns:
        URL-safe token string
    """
    return secrets.token_urlsafe(length)


def constant_time_compare(val1: str, val2: str) -> bool:
    """
    Compare two strings in constant time to prevent timing attacks.
    
    Args:
        val1: First string
        val2: Second string
        
    Returns:
        True if strings match, False otherwise
    """
    return secrets.compare_digest(val1, val2)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal attacks.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove path separators and other dangerous characters
    dangerous_chars = ["../", "..\\", "/", "\\", "\x00"]
    for char in dangerous_chars:
        filename = filename.replace(char, "")
    
    # Limit filename length
    max_length = 255
    if len(filename) > max_length:
        if "." in filename:
            name, ext = filename.rsplit(".", 1)
            name = name[:max_length - len(ext) - 1]
            filename = f"{name}.{ext}"
        else:
            filename = filename[:max_length]
    
    return filename


def create_rate_limit_key(identifier: str, endpoint: str) -> str:
    """
    Create a cache key for rate limiting.
    
    Args:
        identifier: User ID or IP address
        endpoint: API endpoint path
        
    Returns:
        Rate limit cache key
    """
    return f"rate_limit:{identifier}:{endpoint}"


def get_security_headers() -> Dict[str, str]:
    """
    Get recommended security headers for HTTP responses.
    
    Returns:
        Dictionary of security headers
    """
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    }


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Constants
    "ALGORITHM",
    "ACCESS_TOKEN_EXPIRE_MINUTES",
    "REFRESH_TOKEN_EXPIRE_DAYS",
    # Password functions
    "pwd_context",
    "hash_password",
    "get_password_hash",
    "verify_password",
    "validate_password_strength",
    # OAuth2
    "oauth2_scheme",
    "http_bearer",
    # JWT tokens
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "create_email_verification_token",
    "create_password_reset_token",
    "verify_email_token",
    "verify_password_reset_token",
    # API keys
    "generate_api_key",
    "hash_api_key",
    "verify_api_key",
    # Helpers
    "generate_secure_token",
    "constant_time_compare",
    "sanitize_filename",
    "create_rate_limit_key",
    "get_security_headers",
]
