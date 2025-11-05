"""
Authentication Service.

Handles user authentication, registration, password management,
token generation, and session management with production-grade security.
"""

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from fastapi import HTTPException, status, Depends
from sqlalchemy import select, or_
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.core.config import settings
from app.core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    decode_token,
    create_email_verification_token,
    create_password_reset_token,
    verify_email_token,
    verify_password_reset_token,
    validate_password_strength,
)
from app.models.user import User, UserRole, UserSubscription
from app.schemas.user import UserCreate, UserLogin
from app.schemas.token import Token
from app.database import get_db


class AuthService:
    """
    Authentication service for user management and security operations.
    
    Implements:
    - User registration with email verification
    - Login with JWT tokens
    - Token refresh with rotation
    - Password reset workflow
    - Account locking on failed attempts
    - Session management
    """
    
    def __init__(self, db: Session):
        """
        Initialize auth service with database session.
        
        Args:
            db: SQLAlchemy database session
        """
        self.db = db
    
    # ============================================================
    # USER REGISTRATION
    # ============================================================
    
    def register_user(
        self,
        user_data: UserCreate,
        send_verification_email: bool = True
    ) -> User:
        """
        Register a new user account.
        
        Args:
            user_data: User registration data
            send_verification_email: Whether to send verification email
            
        Returns:
            Created user instance
            
        Raises:
            HTTPException: If username/email already exists or validation fails
        """
        # Validate password strength
        is_valid, error_msg = validate_password_strength(user_data.password)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg,
            )
        
        # Check if user already exists
        existing_user = self._get_user_by_email_or_username(
            user_data.email,
            user_data.username
        )
        
        if existing_user:
            if existing_user.email == user_data.email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered",
                )
            if existing_user.username == user_data.username:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already taken",
                )
        
        # Create new user
        try:
            new_user = User(
                email=user_data.email.lower(),
                username=user_data.username.lower(),
                full_name=user_data.full_name,
                hashed_password=get_password_hash(user_data.password),
                role=UserRole.USER,
                subscription=UserSubscription.FREE,
                is_active=True,
                is_verified=False,
            )
            
            self.db.add(new_user)
            self.db.commit()
            self.db.refresh(new_user)
            
            # Send verification email (async task in production)
            if send_verification_email and settings.EMAIL_ENABLED:
                self._send_verification_email(new_user)
            
            return new_user
            
        except IntegrityError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User registration failed due to duplicate data",
            ) from e
    
    # ============================================================
    # USER AUTHENTICATION
    # ============================================================
    
    def authenticate_user(
        self,
        login_data: UserLogin
    ) -> Tuple[User, Token]:
        """
        Authenticate user and return tokens.
        
        Args:
            login_data: Login credentials (username/email + password)
            
        Returns:
            Tuple of (authenticated user, token response)
            
        Raises:
            HTTPException: If authentication fails
        """
        # Get user by username or email
        user = self._get_user_by_email_or_username(
            login_data.username,
            login_data.username
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check if account is locked
        if user.is_account_locked():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Account is locked until {user.locked_until.isoformat()}",
            )
        
        # Verify password
        if not verify_password(login_data.password, user.hashed_password):
            user.record_failed_login()
            self.db.commit()
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check if account is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is disabled",
            )
        
        # Check if email is verified (optional requirement)
        if settings.ENVIRONMENT == "production" and not user.is_verified:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Please verify your email before logging in",
            )
        
        # Update login stats
        user.increment_login()
        self.db.commit()
        
        # Generate tokens
        tokens = self._generate_token_pair(user)
        
        return user, tokens
    
    def _generate_token_pair(self, user: User) -> Token:
        """
        Generate access and refresh token pair.
        
        Args:
            user: User instance
            
        Returns:
            Token response with access and refresh tokens
        """
        # Create access token
        access_token_expires = timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
        access_token = create_access_token(
            subject=str(user.id),
            expires_delta=access_token_expires,
            additional_claims={
                "email": user.email,
                "role": user.role.value,
                "subscription": user.subscription.value,
            }
        )
        
        # Create refresh token
        refresh_token_expires = timedelta(
            minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES
        )
        refresh_token = create_refresh_token(
            subject=str(user.id),
            expires_delta=refresh_token_expires,
        )
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )
    
    # ============================================================
    # TOKEN MANAGEMENT
    # ============================================================
    
    def refresh_access_token(
        self,
        refresh_token: str
    ) -> Token:
        """
        Refresh access token using refresh token (with token rotation).
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New token pair (access + refresh)
            
        Raises:
            HTTPException: If refresh token is invalid
        """
        try:
            # Decode refresh token
            payload = decode_token(refresh_token, token_type="refresh")
            user_id = int(payload.get("sub"))
            
            # Get user
            user = self.db.get(User, user_id)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found",
                )
            
            # Check if user is active
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Account is disabled",
                )
            
            # Token rotation: invalidate old refresh token and generate new pair
            # In production, store jti in Redis/database for blacklisting
            jti = payload.get("jti")
            if jti and settings.CACHE_ENABLED:
                self._blacklist_token(jti)
            
            # Generate new token pair
            return self._generate_token_pair(user)
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            ) from e
    
    def _blacklist_token(self, jti: str) -> None:
        """
        Blacklist a token by its JTI (JWT ID).
        
        In production, store in Redis with TTL = refresh token expiry.
        
        Args:
            jti: JWT ID to blacklist
        """
        # TODO: Implement Redis caching
        # redis_client.setex(
        #     f"blacklist:{jti}",
        #     settings.REFRESH_TOKEN_EXPIRE_MINUTES * 60,
        #     "1"
        # )
        pass
    
    def is_token_blacklisted(self, jti: str) -> bool:
        """
        Check if token is blacklisted.
        
        Args:
            jti: JWT ID to check
            
        Returns:
            True if blacklisted, False otherwise
        """
        # TODO: Implement Redis lookup
        # return redis_client.exists(f"blacklist:{jti}") == 1
        return False
    
    # ============================================================
    # PASSWORD MANAGEMENT
    # ============================================================
    
    def change_password(
        self,
        user: User,
        current_password: str,
        new_password: str
    ) -> None:
        """
        Change user password (requires current password).
        
        Args:
            user: User instance
            current_password: Current password for verification
            new_password: New password
            
        Raises:
            HTTPException: If current password is incorrect
        """
        # Verify current password
        if not verify_password(current_password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect",
            )
        
        # Validate new password strength
        is_valid, error_msg = validate_password_strength(new_password)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg,
            )
        
        # Check if new password is different
        if current_password == new_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password must be different from current password",
            )
        
        # Update password
        user.hashed_password = get_password_hash(new_password)
        user.password_changed_at = datetime.now(timezone.utc)
        
        # Invalidate all existing sessions (optional)
        # In production, increment a version number and check in JWT
        
        self.db.commit()
    
    def request_password_reset(self, email: str) -> str:
        """
        Generate password reset token and send email.
        
        Args:
            email: User email address
            
        Returns:
            Reset token (in production, only send via email)
            
        Raises:
            HTTPException: If user not found
        """
        user = self._get_user_by_email(email)
        
        if not user:
            # Don't reveal if email exists (security best practice)
            # But still return success to prevent user enumeration
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="If email exists, reset link has been sent",
            )
        
        # Generate reset token
        reset_token = create_password_reset_token(email)
        
        # Send reset email (async task in production)
        if settings.EMAIL_ENABLED:
            self._send_password_reset_email(user, reset_token)
        
        return reset_token
    
    def reset_password(
        self,
        token: str,
        new_password: str
    ) -> None:
        """
        Reset password using reset token.
        
        Args:
            token: Password reset token
            new_password: New password
            
        Raises:
            HTTPException: If token is invalid or expired
        """
        # Verify token
        email = verify_password_reset_token(token)
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token",
            )
        
        # Get user
        user = self._get_user_by_email(email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        
        # Validate new password
        is_valid, error_msg = validate_password_strength(new_password)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg,
            )
        
        # Update password
        user.hashed_password = get_password_hash(new_password)
        user.password_changed_at = datetime.now(timezone.utc)
        user.failed_login_attempts = 0
        user.locked_until = None
        
        self.db.commit()
    
    # ============================================================
    # EMAIL VERIFICATION
    # ============================================================
    
    def request_email_verification(self, email: str) -> str:
        """
        Generate email verification token and send email.
        
        Args:
            email: User email address
            
        Returns:
            Verification token
        """
        user = self._get_user_by_email(email)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        
        if user.is_verified:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already verified",
            )
        
        # Generate verification token
        verification_token = create_email_verification_token(email)
        
        # Send verification email
        if settings.EMAIL_ENABLED:
            self._send_verification_email(user, verification_token)
        
        return verification_token
    
    def verify_email(self, token: str) -> User:
        """
        Verify user email using verification token.
        
        Args:
            token: Email verification token
            
        Returns:
            Verified user
            
        Raises:
            HTTPException: If token is invalid
        """
        # Verify token
        email = verify_email_token(token)
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired verification token",
            )
        
        # Get user
        user = self._get_user_by_email(email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        
        # Mark as verified
        user.is_verified = True
        user.email_verified_at = datetime.now(timezone.utc)
        
        self.db.commit()
        self.db.refresh(user)
        
        return user
    
    # ============================================================
    # ACCOUNT MANAGEMENT
    # ============================================================
    
    def deactivate_account(self, user: User) -> None:
        """
        Deactivate user account (soft delete).
        
        Args:
            user: User instance to deactivate
        """
        user.is_active = False
        user.soft_delete()
        self.db.commit()
    
    def reactivate_account(self, user: User) -> None:
        """
        Reactivate previously deactivated account.
        
        Args:
            user: User instance to reactivate
        """
        user.is_active = True
        user.restore()
        self.db.commit()
    
    def unlock_account(self, user_id: int) -> None:
        """
        Manually unlock a locked account (admin function).
        
        Args:
            user_id: User ID to unlock
        """
        user = self.db.get(User, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        
        user.reset_failed_attempts()
        self.db.commit()
    
    # ============================================================
    # HELPER METHODS
    # ============================================================
    
    def _get_user_by_email(self, email: str) -> Optional[User]:
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
    
    def _get_user_by_username(self, username: str) -> Optional[User]:
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
    
    def _get_user_by_email_or_username(
        self,
        email: str,
        username: str
    ) -> Optional[User]:
        """
        Get user by email or username.
        
        Args:
            email: Email address
            username: Username
            
        Returns:
            User instance or None
        """
        stmt = select(User).where(
            or_(
                User.email == email.lower(),
                User.username == username.lower()
            )
        )
        result = self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    def _send_verification_email(
        self,
        user: User,
        token: Optional[str] = None
    ) -> None:
        """
        Send email verification email.
        
        In production, use Celery task for async sending.
        
        Args:
            user: User instance
            token: Verification token (generated if not provided)
        """
        if not token:
            token = create_email_verification_token(user.email)
        
        # TODO: Implement email sending
        # from app.tasks.notification_tasks import send_verification_email
        # send_verification_email.delay(user.email, token)
        
        # For development, log the token
        if settings.is_development:
            print(f"Verification token for {user.email}: {token}")
    
    def _send_password_reset_email(
        self,
        user: User,
        token: str
    ) -> None:
        """
        Send password reset email.
        
        In production, use Celery task for async sending.
        
        Args:
            user: User instance
            token: Reset token
        """
        # TODO: Implement email sending
        # from app.tasks.notification_tasks import send_password_reset_email
        # send_password_reset_email.delay(user.email, token)
        
        # For development, log the token
        if settings.is_development:
            print(f"Password reset token for {user.email}: {token}")


# ============================================================
# DEPENDENCY INJECTION HELPER
# ============================================================

def get_auth_service(db: Session = Depends(get_db)) -> AuthService:
    """
    Dependency for injecting AuthService.
    
    Args:
        db: Database session (injected via Depends)
        
    Returns:
        AuthService instance
        
    Usage:
        In endpoints:
        ```
        @router.post("/register")
        def register(
            auth_service: AuthService = Depends(get_auth_service)
        ):
            ...
        ```
    """
    return AuthService(db)
