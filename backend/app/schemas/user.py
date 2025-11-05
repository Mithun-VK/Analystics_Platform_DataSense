"""
User-related Pydantic schemas.
Handles user creation, updates, authentication, and profile management.
"""

from datetime import datetime
from typing import Optional, Dict, Any

from pydantic import (
    BaseModel,
    EmailStr,
    Field,
    ConfigDict,
    field_validator,
    model_validator,
)

from app.models.user import UserRole, UserSubscription


class UserBase(BaseModel):
    """Base user schema with common fields."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        str_min_length=1,
        use_enum_values=True,
    )
    
    email: Optional[EmailStr] = Field(
        None,
        description="User email address",
        max_length=255,
    )
    
    username: Optional[str] = Field(
        None,
        description="Unique username",
        min_length=3,
        max_length=50,
        pattern="^[a-zA-Z0-9_-]+$",
    )
    
    full_name: Optional[str] = Field(
        None,
        description="User's full name",
        max_length=255,
    )


class UserCreate(UserBase):
    """User creation schema - Required fields for user registration."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "user@example.com",
                "username": "johndoe",
                "full_name": "John Doe",
                "password": "SecurePass123!",
            }
        }
    )
    
    email: EmailStr = Field(..., description="User email address (required)")
    username: str = Field(..., description="Unique username (required)", min_length=3, max_length=50)
    full_name: str = Field(..., description="User's full name (required)", max_length=255)
    password: str = Field(..., description="User password", min_length=8, max_length=128)
    
    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate username format."""
        if not v.isalnum() and not all(c in "_-" for c in v if not c.isalnum()):
            raise ValueError("Username can only contain letters, numbers, underscores, and hyphens")
        if v.lower() in ["admin", "root", "system", "api"]:
            raise ValueError("This username is reserved")
        return v
    
    @field_validator("password")
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


class UserLogin(BaseModel):
    """User login schema - Credentials for authentication."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {"username": "johndoe", "password": "SecurePass123!"}
        }
    )
    
    username: str = Field(..., description="Username or email", min_length=1)
    password: str = Field(..., description="User password", min_length=1)


class UserUpdate(UserBase):
    """User update schema - All fields optional for partial updates."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "full_name": "John Updated Doe",
                "bio": "Software engineer and data enthusiast",
                "timezone": "America/New_York",
            }
        }
    )
    
    bio: Optional[str] = Field(None, description="User biography", max_length=500)
    phone_number: Optional[str] = Field(None, description="Phone number", pattern="^\\+?[1-9]\\d{1,14}$")
    timezone: Optional[str] = Field(None, description="User timezone", max_length=50)
    language: Optional[str] = Field(None, description="Preferred language", max_length=10, pattern="^[a-z]{2}(-[A-Z]{2})?$")


class UserPasswordChange(BaseModel):
    """Password change schema - Requires current password for security."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "current_password": "OldPass123!",
                "new_password": "NewSecurePass123!",
                "confirm_password": "NewSecurePass123!",
            }
        }
    )
    
    current_password: str = Field(..., description="Current password", min_length=1)
    new_password: str = Field(..., description="New password", min_length=8, max_length=128)
    confirm_password: str = Field(..., description="Confirm new password", min_length=8, max_length=128)
    
    @model_validator(mode="after")
    def validate_passwords_match(self) -> "UserPasswordChange":
        """Validate new passwords match."""
        if self.new_password != self.confirm_password:
            raise ValueError("New passwords do not match")
        if self.current_password == self.new_password:
            raise ValueError("New password must be different from current password")
        return self
    
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


class UserPasswordReset(BaseModel):
    """Password reset request schema."""
    
    model_config = ConfigDict(
        json_schema_extra={"example": {"email": "user@example.com"}}
    )
    
    email: EmailStr = Field(..., description="Email address to send reset link")


class UserEmailVerification(BaseModel):
    """Email verification request schema."""
    
    model_config = ConfigDict(
        json_schema_extra={"example": {"email": "user@example.com"}}
    )
    
    email: EmailStr = Field(..., description="Email address to verify")


class UserPreferences(BaseModel):
    """User preferences schema."""
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "timezone": "America/New_York",
                "language": "en",
                "theme": "dark",
                "email_notifications": True,
            }
        }
    )
    
    timezone: str = Field(default="UTC", description="User's timezone")
    language: str = Field(default="en", description="Preferred language (ISO 639-1)")
    theme: Optional[str] = Field(default="light", description="UI theme (light/dark/auto)")
    email_notifications: bool = Field(default=True, description="Enable email notifications")
    dataset_notifications: bool = Field(default=True, description="Dataset processing notifications")
    marketing_emails: bool = Field(default=False, description="Marketing emails")


class UserSubscriptionUpdate(BaseModel):
    """Subscription tier update schema."""
    
    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={"example": {"subscription": "premium"}}
    )
    
    subscription: UserSubscription = Field(..., description="New subscription tier")


class UserResponse(UserBase):
    """User response schema - Returned by API (excludes sensitive data)."""
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "email": "user@example.com",
                "username": "johndoe",
                "full_name": "John Doe",
                "role": "user",
                "subscription": "premium",
                "is_active": True,
                "is_verified": True,
                "created_at": "2025-01-01T00:00:00Z",
            }
        }
    )
    
    id: int = Field(..., description="User unique identifier", gt=0)
    email: EmailStr = Field(..., description="User email address")
    username: str = Field(..., description="Username")
    full_name: str = Field(..., description="User's full name")
    role: UserRole = Field(..., description="User role")
    subscription: UserSubscription = Field(..., description="Subscription tier")
    is_active: bool = Field(..., description="Whether user account is active")
    is_verified: bool = Field(..., description="Whether email has been verified")
    created_at: datetime = Field(..., description="Account creation timestamp")
    last_login_at: Optional[datetime] = Field(None, description="Last login timestamp")


class UserProfile(UserResponse):
    """Detailed user profile schema - Includes additional profile information."""
    
    model_config = ConfigDict(from_attributes=True)
    
    bio: Optional[str] = None
    phone_number: Optional[str] = None
    profile_picture_url: Optional[str] = None
    timezone: str
    language: str
    storage_used_bytes: int
    datasets_count: int
    subscription_expires_at: Optional[datetime] = None
    updated_at: datetime


class UserStats(BaseModel):
    """User statistics and usage metrics."""
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "user_id": 1,
                "email": "user@example.com",
                "username": "johndoe",
                "account_age_days": 30,
                "total_datasets": 5,
                "storage_used_formatted": "125.5 MB",
            }
        }
    )
    
    user_id: int
    email: str
    username: str
    full_name: str
    
    # Account info
    account_age_days: int = Field(description="Days since account creation")
    subscription: str
    subscription_expires_at: Optional[datetime] = None
    days_until_expiry: Optional[int] = None
    
    # Usage metrics
    total_datasets: int = Field(description="Total datasets uploaded")
    storage_used_bytes: int = Field(description="Storage used in bytes")
    storage_used_formatted: str = Field(description="Human-readable storage size")
    datasets_limit: int = Field(description="Maximum datasets allowed")
    storage_limit_bytes: int = Field(description="Maximum storage allowed")
    
    # Activity metrics
    total_logins: int = Field(description="Total number of logins")
    last_login_at: Optional[datetime] = Field(description="Last login timestamp")
    days_since_last_login: Optional[int] = None
    
    # Feature usage
    eda_reports_generated: int = Field(default=0, description="EDA reports generated")
    ai_insights_generated: int = Field(default=0, description="AI insights generated")
    visualizations_created: int = Field(default=0, description="Visualizations created")
    
    # Account status
    is_active: bool
    is_verified: bool
    is_premium: bool = Field(description="Whether user has premium or higher")


# Export all schemas
__all__ = [
    "UserBase",
    "UserCreate",
    "UserLogin",
    "UserUpdate",
    "UserPasswordChange",
    "UserPasswordReset",
    "UserEmailVerification",
    "UserPreferences",
    "UserSubscriptionUpdate",
    "UserResponse",
    "UserProfile",
    "UserStats",
]
