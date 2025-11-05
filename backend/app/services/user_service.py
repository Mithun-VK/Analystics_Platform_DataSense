"""
User Service.

Handles user profile management, preferences, subscription management,
and user-related business logic separate from authentication.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, status
from sqlalchemy import select, func, or_
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.core.config import settings
from app.models.user import User, UserRole, UserSubscription
from app.schemas.user import (
    UserUpdate,
    UserProfile,
    UserPreferences,
    UserSubscriptionUpdate,
)
from app.database import get_db


logger = logging.getLogger(__name__)


class UserService:
    """
    User management service.
    
    Implements:
    - User profile management (CRUD)
    - Subscription tier management
    - User preferences and settings
    - Usage statistics tracking
    - User search and filtering
    - Account status management
    """
    
    def __init__(self, db: Session):
        """
        Initialize user service.
        
        Args:
            db: SQLAlchemy database session
        """
        self.db = db
    
    # ============================================================
    # USER RETRIEVAL
    # ============================================================
    
    def get_user_by_id(self, user_id: int) -> User:
        """
        Get user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User instance
            
        Raises:
            HTTPException: If user not found
        """
        user = self.db.get(User, user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        
        return user
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email address.
        
        Args:
            email: Email address
            
        Returns:
            User instance or None
        """
        stmt = select(User).where(User.email == email.lower())
        result = self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Get user by username.
        
        Args:
            username: Username
            
        Returns:
            User instance or None
        """
        stmt = select(User).where(User.username == username.lower())
        result = self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    def get_user_profile(self, user_id: int) -> UserProfile:
        """
        Get detailed user profile.
        
        Args:
            user_id: User ID
            
        Returns:
            UserProfile schema
            
        Raises:
            HTTPException: If user not found
        """
        user = self.get_user_by_id(user_id)
        return UserProfile.model_validate(user)
    
    # ============================================================
    # USER UPDATES
    # ============================================================
    
    def update_user_profile(
        self,
        user_id: int,
        update_data: UserUpdate
    ) -> User:
        """
        Update user profile information.
        
        Args:
            user_id: User ID
            update_data: Update data
            
        Returns:
            Updated user instance
            
        Raises:
            HTTPException: If update fails
        """
        user = self.get_user_by_id(user_id)
        
        # Check for email/username uniqueness if being updated
        update_dict = update_data.model_dump(exclude_unset=True)
        
        if 'email' in update_dict and update_dict['email'] != user.email:
            existing = self.get_user_by_email(update_dict['email'])
            if existing and existing.id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already in use",
                )
        
        if 'username' in update_dict and update_dict['username'] != user.username:
            existing = self.get_user_by_username(update_dict['username'])
            if existing and existing.id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already taken",
                )
        
        # Update fields
        for field, value in update_dict.items():
            if hasattr(user, field):
                setattr(user, field, value)
        
        try:
            self.db.commit()
            self.db.refresh(user)
            logger.info(f"Updated profile for user {user_id}")
            return user
        except IntegrityError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Profile update failed",
            ) from e
    
    def update_user_preferences(
        self,
        user_id: int,
        preferences: UserPreferences
    ) -> User:
        """
        Update user preferences.
        
        Args:
            user_id: User ID
            preferences: User preferences
            
        Returns:
            Updated user instance
        """
        user = self.get_user_by_id(user_id)
        
        # Store preferences (could be in separate table in production)
        pref_dict = preferences.model_dump(exclude_unset=True)
        
        # Update user fields based on preferences
        if 'timezone' in pref_dict:
            user.timezone = pref_dict['timezone']
        if 'language' in pref_dict:
            user.language = pref_dict['language']
        
        self.db.commit()
        self.db.refresh(user)
        
        logger.info(f"Updated preferences for user {user_id}")
        return user
    
    # ============================================================
    # SUBSCRIPTION MANAGEMENT
    # ============================================================
    
    def update_subscription(
        self,
        user_id: int,
        subscription_update: UserSubscriptionUpdate
    ) -> User:
        """
        Update user subscription tier.
        
        Args:
            user_id: User ID
            subscription_update: Subscription update data
            
        Returns:
            Updated user instance
        """
        user = self.get_user_by_id(user_id)
        
        # Update subscription
        user.subscription = subscription_update.subscription
        
        # Set expiration date based on tier
        if subscription_update.subscription != UserSubscription.FREE:
            from datetime import timedelta
            # Premium subscriptions typically last 30 days
            user.subscription_expires_at = datetime.now(timezone.utc) + timedelta(days=30)
        else:
            user.subscription_expires_at = None
        
        self.db.commit()
        self.db.refresh(user)
        
        logger.info(
            f"Updated subscription for user {user_id} to {subscription_update.subscription}"
        )
        return user
    
    def check_subscription_status(self, user_id: int) -> Dict[str, Any]:
        """
        Check user's subscription status.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with subscription status details
        """
        user = self.get_user_by_id(user_id)
        
        is_active = user.is_subscription_active()
        days_remaining = None
        
        if user.subscription_expires_at:
            days_remaining = (user.subscription_expires_at - datetime.now(timezone.utc)).days
            days_remaining = max(0, days_remaining)
        
        return {
            "subscription": user.subscription.value,
            "is_active": is_active,
            "expires_at": user.subscription_expires_at.isoformat() if user.subscription_expires_at else None,
            "days_remaining": days_remaining,
            "is_premium": user.is_premium_or_higher(),
        }
    
    # ============================================================
    # USAGE STATISTICS
    # ============================================================
    
    def get_user_statistics(self, user_id: int) -> Dict[str, Any]:
        """
        Get user usage statistics.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with usage statistics
        """
        user = self.get_user_by_id(user_id)
        
        from app.utils.helpers import format_bytes
        
        return {
            "datasets_count": user.datasets_count,
            "storage_used_bytes": user.storage_used_bytes,
            "storage_used_formatted": format_bytes(user.storage_used_bytes),
            "storage_limit_bytes": self._get_storage_limit(user),
            "storage_limit_formatted": format_bytes(self._get_storage_limit(user)),
            "storage_percentage": (user.storage_used_bytes / self._get_storage_limit(user) * 100) if self._get_storage_limit(user) > 0 else 0,
            "login_count": user.login_count,
            "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
            "account_age_days": (datetime.now(timezone.utc) - user.created_at).days,
        }
    
    def _get_storage_limit(self, user: User) -> int:
        """Get storage limit based on subscription tier."""
        if user.subscription == UserSubscription.FREE:
            return settings.FREE_TIER_FILE_SIZE_LIMIT * settings.FREE_TIER_DATASET_LIMIT
        elif user.subscription in [UserSubscription.PREMIUM, UserSubscription.ENTERPRISE]:
            return settings.PREMIUM_TIER_FILE_SIZE_LIMIT * settings.PREMIUM_TIER_DATASET_LIMIT
        else:
            return settings.FREE_TIER_FILE_SIZE_LIMIT * settings.FREE_TIER_DATASET_LIMIT
    
    def update_storage_usage(
        self,
        user_id: int,
        bytes_delta: int
    ) -> None:
        """
        Update user's storage usage.
        
        Args:
            user_id: User ID
            bytes_delta: Change in storage (positive or negative)
        """
        user = self.get_user_by_id(user_id)
        user.storage_used_bytes += bytes_delta
        user.storage_used_bytes = max(0, user.storage_used_bytes)  # Prevent negative
        
        self.db.commit()
        logger.debug(f"Updated storage for user {user_id}: {bytes_delta:+d} bytes")
    
    def update_dataset_count(
        self,
        user_id: int,
        count_delta: int
    ) -> None:
        """
        Update user's dataset count.
        
        Args:
            user_id: User ID
            count_delta: Change in count (positive or negative)
        """
        user = self.get_user_by_id(user_id)
        user.datasets_count += count_delta
        user.datasets_count = max(0, user.datasets_count)  # Prevent negative
        
        self.db.commit()
        logger.debug(f"Updated dataset count for user {user_id}: {count_delta:+d}")
    
    # ============================================================
    # USER SEARCH & LISTING
    # ============================================================
    
    def list_users(
        self,
        skip: int = 0,
        limit: int = 20,
        search: Optional[str] = None,
        role_filter: Optional[UserRole] = None,
        subscription_filter: Optional[UserSubscription] = None,
        is_active: Optional[bool] = None,
    ) -> Tuple[List[User], int]:
        """
        List users with pagination and filters.
        
        Args:
            skip: Offset for pagination
            limit: Number of items per page
            search: Search query for email/username
            role_filter: Filter by role
            subscription_filter: Filter by subscription
            is_active: Filter by active status
            
        Returns:
            Tuple of (users list, total count)
        """
        query = select(User)
        
        # Apply filters
        if search:
            search_term = f"%{search}%"
            query = query.where(
                or_(
                    User.email.ilike(search_term),
                    User.username.ilike(search_term),
                    User.full_name.ilike(search_term),
                )
            )
        
        if role_filter:
            query = query.where(User.role == role_filter)
        
        if subscription_filter:
            query = query.where(User.subscription == subscription_filter)
        
        if is_active is not None:
            query = query.where(User.is_active == is_active)
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total = self.db.execute(count_query).scalar_one()
        
        # Apply pagination and ordering
        query = query.order_by(User.created_at.desc()).offset(skip).limit(limit)
        
        result = self.db.execute(query)
        users = result.scalars().all()
        
        return list(users), total
    
    # ============================================================
    # ACCOUNT STATUS MANAGEMENT
    # ============================================================
    
    def activate_user(self, user_id: int) -> User:
        """
        Activate user account.
        
        Args:
            user_id: User ID
            
        Returns:
            Updated user instance
        """
        user = self.get_user_by_id(user_id)
        user.is_active = True
        
        self.db.commit()
        self.db.refresh(user)
        
        logger.info(f"Activated user account {user_id}")
        return user
    
    def deactivate_user(self, user_id: int) -> User:
        """
        Deactivate user account.
        
        Args:
            user_id: User ID
            
        Returns:
            Updated user instance
        """
        user = self.get_user_by_id(user_id)
        user.is_active = False
        
        self.db.commit()
        self.db.refresh(user)
        
        logger.info(f"Deactivated user account {user_id}")
        return user
    
    def delete_user(
        self,
        user_id: int,
        hard_delete: bool = False
    ) -> None:
        """
        Delete user account.
        
        Args:
            user_id: User ID
            hard_delete: Permanently delete (vs soft delete)
        """
        user = self.get_user_by_id(user_id)
        
        if hard_delete:
            # Delete all user's datasets first (cascading)
            self.db.delete(user)
            logger.info(f"Hard deleted user {user_id}")
        else:
            # Soft delete
            user.soft_delete()
            user.is_active = False
            logger.info(f"Soft deleted user {user_id}")
        
        self.db.commit()
    
    # ============================================================
    # ROLE MANAGEMENT
    # ============================================================
    
    def update_user_role(
        self,
        user_id: int,
        new_role: UserRole
    ) -> User:
        """
        Update user role (admin function).
        
        Args:
            user_id: User ID
            new_role: New role
            
        Returns:
            Updated user instance
        """
        user = self.get_user_by_id(user_id)
        user.role = new_role
        
        self.db.commit()
        self.db.refresh(user)
        
        logger.info(f"Updated role for user {user_id} to {new_role}")
        return user
    
    # ============================================================
    # HELPER METHODS
    # ============================================================
    
    def check_user_limits(
        self,
        user: User,
        check_type: str = "dataset"
    ) -> Tuple[bool, str]:
        """
        Check if user has reached their limits.
        
        Args:
            user: User instance
            check_type: Type of limit to check (dataset, storage)
            
        Returns:
            Tuple of (can_proceed, message)
        """
        if check_type == "dataset":
            if user.subscription == UserSubscription.FREE:
                if user.datasets_count >= settings.FREE_TIER_DATASET_LIMIT:
                    return False, f"Free tier limit of {settings.FREE_TIER_DATASET_LIMIT} datasets reached"
            elif user.subscription in [UserSubscription.PREMIUM, UserSubscription.ENTERPRISE]:
                if user.datasets_count >= settings.PREMIUM_TIER_DATASET_LIMIT:
                    return False, f"Dataset limit of {settings.PREMIUM_TIER_DATASET_LIMIT} reached"
        
        elif check_type == "storage":
            storage_limit = self._get_storage_limit(user)
            if user.storage_used_bytes >= storage_limit:
                from app.utils.helpers import format_bytes
                return False, f"Storage limit of {format_bytes(storage_limit)} reached"
        
        return True, ""


# ============================================================
# DEPENDENCY INJECTION HELPER
# ============================================================

def get_user_service(db: Session = Depends(get_db)) -> UserService:
    """
    Dependency for injecting UserService.
    
    Args:
        db: Database session
        
    Returns:
        UserService instance
    """
    return UserService(db)
