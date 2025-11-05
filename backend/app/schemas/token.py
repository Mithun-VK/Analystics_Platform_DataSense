"""
Token-related Pydantic schemas.
Handles JWT tokens, refresh tokens, and verification tokens.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator


class Token(BaseModel):
    """
    Access token response schema.
    
    Returned after successful authentication.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 1800,
            }
        }
    )
    
    access_token: str = Field(
        ...,
        description="JWT access token for authentication",
        min_length=1,
    )
    
    refresh_token: Optional[str] = Field(
        None,
        description="JWT refresh token for obtaining new access tokens",
    )
    
    token_type: str = Field(
        default="bearer",
        description="Token type (always 'bearer')",
        pattern="^bearer$",
    )
    
    expires_in: int = Field(
        ...,
        description="Token expiration time in seconds",
        gt=0,
        example=1800,
    )


class TokenPayload(BaseModel):
    """
    JWT token payload schema.
    
    Decoded token data for internal use.
    """
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )
    
    sub: str = Field(
        ...,
        description="Subject (user ID)",
        min_length=1,
    )
    
    exp: datetime = Field(
        ...,
        description="Expiration timestamp",
    )
    
    iat: Optional[datetime] = Field(
        None,
        description="Issued at timestamp",
    )
    
    type: str = Field(
        ...,
        description="Token type (access, refresh, etc.)",
        pattern="^(access|refresh|email_verification|password_reset)$",
    )
    
    jti: Optional[str] = Field(
        None,
        description="JWT ID for token revocation",
    )


class RefreshToken(BaseModel):
    """
    Refresh token request schema.
    
    Used to obtain a new access token.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        }
    )
    
    refresh_token: str = Field(
        ...,
        description="Valid refresh token",
        min_length=1,
    )


class EmailVerificationToken(BaseModel):
    """
    Email verification token schema.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        }
    )
    
    token: str = Field(
        ...,
        description="Email verification token",
        min_length=1,
    )


class PasswordResetToken(BaseModel):
    """
    Password reset token request schema.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "new_password": "NewSecurePass123!",
            }
        }
    )
    
    token: str = Field(
        ...,
        description="Password reset token",
        min_length=1,
    )
    
    new_password: str = Field(
        ...,
        description="New password",
        min_length=8,
        max_length=128,
    )
    
    @field_validator("new_password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Validate password meets security requirements."""
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in v):
            raise ValueError("Password must contain at least one special character")
        return v
