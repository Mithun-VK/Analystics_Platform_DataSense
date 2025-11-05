"""
User model and related entities.
Handles user authentication, profiles, and subscriptions.
"""

import enum
from datetime import datetime, timezone, timedelta
from typing import List, Optional, TYPE_CHECKING

from sqlalchemy import (
    String, 
    Integer, 
    Boolean, 
    Enum, 
    Text,
    DateTime,
    CheckConstraint,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, SoftDeleteMixin

# Avoid circular import
if TYPE_CHECKING:
    from app.models.dataset import Dataset


class UserRole(str, enum.Enum):
    """User role enumeration."""
    ADMIN = "admin"
    USER = "user"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class UserSubscription(str, enum.Enum):
    """User subscription tier enumeration."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class User(Base, TimestampMixin, SoftDeleteMixin):
    """
    User model for authentication and user management.
    
    Attributes:
        id: Primary key
        email: Unique email address (used for login)
        username: Unique username
        hashed_password: Bcrypt hashed password
        full_name: User's full name
        role: User role (admin, user, premium, enterprise)
        subscription: Current subscription tier
        is_active: Whether user account is active
        is_verified: Whether email has been verified
        email_verified_at: Timestamp of email verification
        last_login_at: Last successful login timestamp
        login_count: Total number of successful logins
        failed_login_attempts: Consecutive failed login attempts
        locked_until: Account lock timestamp (for security)
        profile_picture_url: URL to user's profile picture
        phone_number: Optional phone number
        timezone: User's timezone preference
        language: User's language preference
        api_key_hash: Hashed API key for programmatic access
        datasets: Relationship to user's datasets
    """
    
    __tablename__ = "users"
    
    # Primary Key
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="User unique identifier"
    )
    
    # Authentication Fields
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="User email address (unique)"
    )
    
    username: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
        index=True,
        comment="Username (unique)"
    )
    
    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Bcrypt hashed password"
    )
    
    # Profile Information
    full_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="User's full name"
    )
    
    profile_picture_url: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="URL to profile picture"
    )
    
    phone_number: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        comment="User's phone number"
    )
    
    bio: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="User biography/description"
    )
    
    # Authorization & Subscription
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, native_enum=False, length=20),
        nullable=False,
        default=UserRole.USER,
        server_default=UserRole.USER.value,
        index=True,
        comment="User role"
    )
    
    subscription: Mapped[UserSubscription] = mapped_column(
        Enum(UserSubscription, native_enum=False, length=20),
        nullable=False,
        default=UserSubscription.FREE,
        server_default=UserSubscription.FREE.value,
        index=True,
        comment="Subscription tier"
    )
    
    subscription_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Subscription expiration timestamp"
    )
    
    # Account Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
        index=True,
        comment="Whether user account is active"
    )
    
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        index=True,
        comment="Whether email has been verified"
    )
    
    email_verified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Email verification timestamp"
    )
    
    # Security Fields
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last successful login timestamp"
    )
    
    login_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Total number of logins"
    )
    
    failed_login_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Consecutive failed login attempts"
    )
    
    locked_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Account lock expiration (security measure)"
    )
    
    password_changed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last password change timestamp"
    )
    
    # API Access
    api_key_hash: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        unique=True,
        comment="Hashed API key for programmatic access"
    )
    
    api_key_created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="API key creation timestamp"
    )
    
    # User Preferences
    timezone: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="UTC",
        server_default="UTC",
        comment="User's timezone"
    )
    
    language: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="en",
        server_default="en",
        comment="User's preferred language"
    )
    
    # Usage Statistics
    storage_used_bytes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Total storage used in bytes"
    )
    
    datasets_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Total number of datasets"
    )
    
    # Relationships
    datasets: Mapped[List["Dataset"]] = relationship(
        "Dataset",
        back_populates="owner",
        lazy="selectin",
        cascade="all, delete-orphan",
        order_by="desc(Dataset.created_at)",
    )
    
    # Table Constraints - SQLite Compatible
    __table_args__ = (
        # Username must be at least 3 characters
        CheckConstraint(
            "length(username) >= 3",
            name="username_min_length"
        ),
        # Storage cannot be negative
        CheckConstraint(
            "storage_used_bytes >= 0",
            name="storage_non_negative"
        ),
        # Dataset count cannot be negative
        CheckConstraint(
            "datasets_count >= 0",
            name="datasets_count_non_negative"
        ),
        # Login count cannot be negative
        CheckConstraint(
            "login_count >= 0",
            name="login_count_non_negative"
        ),
        # Failed attempts cannot be negative
        CheckConstraint(
            "failed_login_attempts >= 0",
            name="failed_attempts_non_negative"
        ),
        # Composite indexes for common queries
        Index("ix_users_email_active", "email", "is_active"),
        Index("ix_users_subscription_role", "subscription", "role"),
        Index("ix_users_username_active", "username", "is_active"),
        {"comment": "Users table for authentication and profiles"}
    )
    
    # ============================================================
    # HELPER METHODS
    # ============================================================
    
    def is_subscription_active(self) -> bool:
        """Check if user's subscription is currently active."""
        if self.subscription == UserSubscription.FREE:
            return True
        if not self.subscription_expires_at:
            return False
        return datetime.now(timezone.utc) < self.subscription_expires_at
    
    def is_premium_or_higher(self) -> bool:
        """Check if user has premium or enterprise subscription."""
        return self.subscription in [
            UserSubscription.PREMIUM,
            UserSubscription.ENTERPRISE
        ]
    
    def is_account_locked(self) -> bool:
        """Check if account is currently locked."""
        if not self.locked_until:
            return False
        return datetime.now(timezone.utc) < self.locked_until
    
    def can_upload_dataset(self, file_size_bytes: int) -> tuple[bool, str]:
        """
        Check if user can upload a dataset of given size.
        
        Args:
            file_size_bytes: Size of file to upload
            
        Returns:
            Tuple of (can_upload, reason_if_cannot)
        """
        from app.core.config import settings
        
        # Check if subscription is active
        if not self.is_subscription_active():
            return False, "Subscription has expired"
        
        # Check subscription limits
        if self.subscription == UserSubscription.FREE:
            if self.datasets_count >= settings.FREE_TIER_DATASET_LIMIT:
                return False, f"Free tier limit of {settings.FREE_TIER_DATASET_LIMIT} datasets reached"
            if file_size_bytes > settings.FREE_TIER_FILE_SIZE_LIMIT:
                from app.utils.helpers import format_bytes
                return False, f"File size exceeds free tier limit of {format_bytes(settings.FREE_TIER_FILE_SIZE_LIMIT)}"
        elif self.subscription in [UserSubscription.PREMIUM, UserSubscription.ENTERPRISE]:
            if self.datasets_count >= settings.PREMIUM_TIER_DATASET_LIMIT:
                return False, f"Dataset limit of {settings.PREMIUM_TIER_DATASET_LIMIT} reached"
            if file_size_bytes > settings.PREMIUM_TIER_FILE_SIZE_LIMIT:
                from app.utils.helpers import format_bytes
                return False, f"File size exceeds limit of {format_bytes(settings.PREMIUM_TIER_FILE_SIZE_LIMIT)}"
        
        return True, ""
    
    def increment_login(self) -> None:
        """Increment login count and update last login timestamp."""
        self.login_count += 1
        self.last_login_at = datetime.now(timezone.utc)
        self.failed_login_attempts = 0
        self.locked_until = None  # Clear any lock
    
    def record_failed_login(self) -> None:
        """
        Record a failed login attempt.
        
        Locks account after 5 failed attempts for 30 minutes.
        """
        self.failed_login_attempts += 1
        
        # Lock account after 5 failed attempts
        if self.failed_login_attempts >= 5:
            self.locked_until = datetime.now(timezone.utc) + timedelta(minutes=30)
    
    def reset_failed_attempts(self) -> None:
        """Reset failed login attempts counter and unlock account."""
        self.failed_login_attempts = 0
        self.locked_until = None
    
    def verify_email(self) -> None:
        """Mark email as verified."""
        self.is_verified = True
        self.email_verified_at = datetime.now(timezone.utc)
    
    def update_storage_usage(self, bytes_delta: int) -> None:
        """
        Update storage usage.
        
        Args:
            bytes_delta: Change in storage (positive or negative)
        """
        self.storage_used_bytes = max(0, self.storage_used_bytes + bytes_delta)
    
    def update_dataset_count(self, count_delta: int) -> None:
        """
        Update dataset count.
        
        Args:
            count_delta: Change in count (positive or negative)
        """
        self.datasets_count = max(0, self.datasets_count + count_delta)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"User(id={self.id}, email='{self.email}', "
            f"username='{self.username}', role={self.role.value}, "
            f"subscription={self.subscription.value})"
        )
