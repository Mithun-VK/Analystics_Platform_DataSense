"""
API Dependencies.

Common dependencies used across multiple endpoints including authentication,
authorization, database sessions, pagination, and rate limiting.
"""

import logging
from typing import Optional, Generator, Callable
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, status, Query, Request
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import ALGORITHM
from app.database import get_db
from app.models.user import User, UserRole, UserSubscription


logger = logging.getLogger(__name__)


# ============================================================
# OAUTH2 SCHEME
# ============================================================

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login/form",
    auto_error=True
)

# Alternative: HTTP Bearer scheme (for flexibility)
http_bearer = HTTPBearer(auto_error=True)


# ============================================================
# DATABASE DEPENDENCY
# ============================================================

def get_db_session() -> Generator[Session, None, None]:
    """
    Get database session dependency.
    
    This is an alias for the main get_db function.
    Use this in endpoints that need database access.
    
    Yields:
        Database session
        
    Example:
        @router.get("/items/")
        def get_items(db: Session = Depends(get_db_session)):
            return db.query(Item).all()
    """
    return get_db()


# ============================================================
# AUTHENTICATION DEPENDENCIES
# ============================================================

def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> User:
    """
    Get current authenticated user from JWT token.
    
    Validates JWT token and returns the user object.
    Checks if user is active and account is not locked.
    
    Args:
        db: Database session
        token: JWT access token from Authorization header
        
    Returns:
        Current authenticated user object
        
    Raises:
        HTTPException 401: Invalid credentials or token expired
        HTTPException 403: User account disabled or locked
        
    Example:
        @router.get("/profile")
        def get_profile(current_user: User = Depends(get_current_user)):
            return current_user
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode JWT token
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        
        # Extract user_id and token type
        user_id: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if user_id is None or token_type != "access":
            logger.warning(f"Invalid token format: user_id={user_id}, type={token_type}")
            raise credentials_exception
        
        # Get user from database
        user = db.get(User, int(user_id))
        
        if user is None:
            logger.warning(f"User not found: {user_id}")
            raise credentials_exception
        
        # Check if user is active
        if not user.is_active:
            logger.warning(f"Inactive user attempted access: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is disabled. Please contact support."
            )
        
        # Check if account is locked
        if user.is_account_locked():
            logger.warning(f"Locked account attempted access: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is temporarily locked due to security reasons. Please try again later."
            )
        
        # Update last activity
        user.last_login_at = datetime.now(timezone.utc)
        db.commit()
        
        return user
    
    except JWTError as e:
        logger.error(f"JWT validation error: {str(e)}")
        raise credentials_exception
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise credentials_exception


def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user.
    
    Additional check to ensure user is active.
    This is mostly redundant with get_current_user but kept for clarity.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Active user object
        
    Raises:
        HTTPException 403: User account is not active
        
    Example:
        @router.get("/items/")
        def get_items(user: User = Depends(get_current_active_user)):
            return get_user_items(user)
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is not active"
        )
    return current_user


def get_current_verified_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current verified user (email verified).
    
    Requires email verification in production environment.
    In development, this check is relaxed.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Verified user object
        
    Raises:
        HTTPException 403: Email not verified (production only)
        
    Example:
        @router.post("/datasets/upload")
        def upload_dataset(user: User = Depends(get_current_verified_user)):
            # Only verified users can upload
            pass
    """
    if not current_user.is_verified and settings.is_production:
        logger.warning(f"Unverified user attempted protected action: {current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Please verify your email address to access this feature. Check your inbox for verification link."
        )
    return current_user


# ============================================================
# AUTHORIZATION DEPENDENCIES (ROLE-BASED)
# ============================================================

def require_premium_user(
    current_user: User = Depends(get_current_verified_user)
) -> User:
    """
    Require premium or higher subscription.
    
    Checks if user has active premium, professional, or enterprise subscription.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Premium user object
        
    Raises:
        HTTPException 403: No premium subscription or subscription expired
        
    Example:
        @router.post("/insights/generate")
        def generate_insights(user: User = Depends(require_premium_user)):
            # Premium feature
            pass
    """
    if not current_user.is_premium_or_higher():
        logger.warning(f"Non-premium user attempted premium feature: {current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This feature requires a premium subscription. Please upgrade your account."
        )
    
    # Check if subscription is active
    if not current_user.is_subscription_active():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your subscription has expired. Please renew to access premium features."
        )
    
    return current_user


def require_professional_user(
    current_user: User = Depends(get_current_verified_user)
) -> User:
    """
    Require professional or enterprise subscription.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Professional user object
        
    Raises:
        HTTPException 403: No professional subscription
        
    Example:
        @router.post("/automl/train")
        def train_model(user: User = Depends(require_professional_user)):
            # Professional feature
            pass
    """
    if current_user.subscription not in [UserSubscription.PROFESSIONAL, UserSubscription.ENTERPRISE]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This feature requires a professional or enterprise subscription."
        )
    
    if not current_user.is_subscription_active():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your subscription has expired. Please renew to access this feature."
        )
    
    return current_user


def require_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Require admin role.
    
    Only users with admin role can access admin endpoints.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Admin user object
        
    Raises:
        HTTPException 403: User is not admin
        
    Example:
        @router.get("/admin/users")
        def list_all_users(admin: User = Depends(require_admin_user)):
            # Admin only
            pass
    """
    if current_user.role != UserRole.ADMIN:
        logger.warning(f"Non-admin user attempted admin access: {current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required. You do not have sufficient permissions."
        )
    return current_user


def require_role(allowed_roles: list[UserRole]) -> Callable:
    """
    Create a dependency that requires specific roles.
    
    Factory function that creates a dependency for specific role requirements.
    
    Args:
        allowed_roles: List of allowed roles
        
    Returns:
        Dependency function
        
    Example:
        @router.get("/moderator/reports")
        def get_reports(user: User = Depends(require_role([UserRole.ADMIN, UserRole.MODERATOR]))):
            pass
    """
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {[role.value for role in allowed_roles]}"
            )
        return current_user
    
    return role_checker


# ============================================================
# OPTIONAL AUTHENTICATION
# ============================================================

def get_optional_user(
    db: Session = Depends(get_db),
    token: Optional[str] = Depends(oauth2_scheme)
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise.
    
    Useful for endpoints that work for both authenticated and anonymous users.
    
    Args:
        db: Database session
        token: Optional JWT token
        
    Returns:
        User object if authenticated, None otherwise
        
    Example:
        @router.get("/public/datasets")
        def list_datasets(user: Optional[User] = Depends(get_optional_user)):
            if user:
                # Show user's private datasets too
                pass
            else:
                # Show only public datasets
                pass
    """
    if not token:
        return None
    
    try:
        return get_current_user(db, token)
    except HTTPException:
        return None


# ============================================================
# PAGINATION DEPENDENCIES
# ============================================================

class PaginationParams:
    """
    Pagination parameters for list endpoints.
    
    Provides consistent pagination across all list endpoints.
    """
    
    def __init__(
        self,
        skip: int = Query(0, ge=0, description="Number of records to skip"),
        limit: int = Query(20, ge=1, le=100, description="Number of records to return")
    ):
        self.skip = skip
        self.limit = limit
        self.page = (skip // limit) + 1


def get_pagination_params(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(20, ge=1, le=100, description="Number of records per page")
) -> PaginationParams:
    """
    Get pagination parameters.
    
    Args:
        skip: Number of records to skip (offset)
        limit: Number of records per page (max 100)
        
    Returns:
        PaginationParams object with skip, limit, and page
        
    Example:
        @router.get("/items/")
        def list_items(pagination: PaginationParams = Depends(get_pagination_params)):
            return get_items(skip=pagination.skip, limit=pagination.limit)
    """
    return PaginationParams(skip=skip, limit=limit)


# ============================================================
# RATE LIMITING DEPENDENCIES
# ============================================================

def check_rate_limit(
    request: Request,
    current_user: Optional[User] = Depends(get_optional_user)
) -> None:
    """
    Check rate limits for API requests.
    
    Rate limits:
    - Anonymous: 10 requests per minute
    - Free tier: 60 requests per minute
    - Premium: 300 requests per minute
    - Admin: No limit
    
    Args:
        request: FastAPI request object
        current_user: Optional current user
        
    Raises:
        HTTPException 429: Rate limit exceeded
        
    Note:
        This is a placeholder. In production, use Redis for distributed rate limiting.
        
    Example:
        @router.get("/api/resource")
        def get_resource(_: None = Depends(check_rate_limit)):
            pass
    """
    # TODO: Implement Redis-based rate limiting in production
    # from app.utils.rate_limiter import check_rate_limit
    
    # For now, this is a no-op
    # In production, you would:
    # 1. Get user identifier (IP or user_id)
    # 2. Check Redis counter
    # 3. Increment counter
    # 4. Raise 429 if limit exceeded
    
    pass


# ============================================================
# REQUEST VALIDATION DEPENDENCIES
# ============================================================

def get_client_ip(request: Request) -> str:
    """
    Get client IP address from request.
    
    Handles proxy headers (X-Forwarded-For, X-Real-IP).
    
    Args:
        request: FastAPI request object
        
    Returns:
        Client IP address string
        
    Example:
        @router.post("/track")
        def track_event(ip: str = Depends(get_client_ip)):
            log_event(ip)
    """
    from app.utils.helpers import get_client_ip as extract_ip
    return extract_ip(request)


def get_user_agent(request: Request) -> Optional[str]:
    """
    Get user agent from request headers.
    
    Args:
        request: FastAPI request object
        
    Returns:
        User agent string or None
        
    Example:
        @router.get("/analytics")
        def track(user_agent: str = Depends(get_user_agent)):
            pass
    """
    return request.headers.get("user-agent")


# ============================================================
# DATASET OWNERSHIP DEPENDENCY
# ============================================================

def verify_dataset_owner(
    dataset_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> int:
    """
    Verify that current user owns the specified dataset.
    
    Args:
        dataset_id: Dataset ID from path parameter
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Dataset ID if user is owner
        
    Raises:
        HTTPException 404: Dataset not found
        HTTPException 403: User is not the owner
        
    Example:
        @router.delete("/datasets/{dataset_id}")
        def delete_dataset(
            dataset_id: int = Depends(verify_dataset_owner),
            db: Session = Depends(get_db)
        ):
            # User is verified as owner
            pass
    """
    from app.models.dataset import Dataset
    
    dataset = db.get(Dataset, dataset_id)
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    if dataset.owner_id != current_user.id and current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this dataset"
        )
    
    return dataset_id


# ============================================================
# FEATURE FLAG DEPENDENCIES
# ============================================================

def require_feature(feature_name: str) -> Callable:
    """
    Create a dependency that checks if a feature is enabled.
    
    Factory function for feature flag checking.
    
    Args:
        feature_name: Feature flag name from settings
        
    Returns:
        Dependency function
        
    Example:
        @router.post("/automl/train")
        def train_model(_: None = Depends(require_feature("FEATURE_AUTOML"))):
            pass
    """
    def feature_checker() -> None:
        feature_enabled = getattr(settings, feature_name, False)
        
        if not feature_enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"This feature is currently disabled: {feature_name}"
            )
    
    return feature_checker


def require_ai_insights() -> None:
    """
    Check if AI insights feature is enabled.
    
    Raises:
        HTTPException 503: Feature not enabled
        
    Example:
        @router.post("/insights/generate")
        def generate_insights(_: None = Depends(require_ai_insights)):
            pass
    """
    if not settings.FEATURE_AI_INSIGHTS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI insights feature is not enabled"
        )


def require_automl() -> None:
    """
    Check if AutoML feature is enabled.
    
    Raises:
        HTTPException 503: Feature not enabled
    """
    if not settings.FEATURE_AUTOML:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AutoML feature is not enabled"
        )


# ============================================================
# EXPORT ALL DEPENDENCIES
# ============================================================

__all__ = [
    # Database
    "get_db_session",
    
    # Authentication
    "get_current_user",
    "get_current_active_user",
    "get_current_verified_user",
    "get_optional_user",
    
    # Authorization
    "require_premium_user",
    "require_professional_user",
    "require_admin_user",
    "require_role",
    
    # Pagination
    "PaginationParams",
    "get_pagination_params",
    
    # Rate Limiting
    "check_rate_limit",
    
    # Request Info
    "get_client_ip",
    "get_user_agent",
    
    # Ownership
    "verify_dataset_owner",
    
    # Feature Flags
    "require_feature",
    "require_ai_insights",
    "require_automl",
]
