"""
User Management Endpoints.

Handles user profile management, account settings, subscription management,
and admin user operations.
"""

import logging
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User, UserRole, UserSubscription
from app.schemas.user import (
    UserResponse,
    UserUpdate,
    UserPasswordChange,
    UserPreferences,
    UserStats,
)
from app.schemas.response import SuccessResponse, MessageResponse
from app.core.deps import (
    get_current_user,
    get_current_verified_user,
    require_admin_user,
    get_pagination_params,
    PaginationParams,
)
from app.core.security import get_password_hash, verify_password
from app.services.user_service import UserService, get_user_service


logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
# CURRENT USER PROFILE
# ============================================================

@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get Current User Profile",
    description="Get the authenticated user's profile information."
)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get current user's profile.
    
    **Returns:**
    - Complete user profile with:
      - Basic info (id, email, username, full_name)
      - Account status (is_active, is_verified)
      - Role and subscription details
      - Usage statistics (datasets_count, storage_used)
      - Timestamps (created_at, last_login_at)
    
    **Authentication:**
    - Requires valid access token
    
    **Use Cases:**
    - Display user profile in UI
    - Check subscription status
    - Verify account details
    """
    return UserResponse.model_validate(current_user)


@router.get(
    "/me/stats",
    response_model=UserStats,
    summary="Get User Statistics",
    description="Get detailed statistics about user's account and usage."
)
async def get_user_statistics(
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
) -> Any:
    """
    Get comprehensive user statistics.
    
    **Returns:**
    - Usage metrics:
      - Total datasets uploaded
      - Storage used (bytes and formatted)
      - Total API calls (if tracked)
      - Active sessions
    - Account info:
      - Account age
      - Days until subscription expires
      - Feature limits
    - Activity summary:
      - Last login
      - Total logins
      - Recent activity
    
    **Authentication:**
    - Requires valid access token
    """
    try:
        stats = user_service.get_user_stats(current_user.id)
        return stats
    except Exception as e:
        logger.error(f"Failed to get user stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user statistics"
        )


# ============================================================
# UPDATE PROFILE
# ============================================================

@router.patch(
    "/me",
    response_model=SuccessResponse[UserResponse],
    summary="Update User Profile",
    description="Update the current user's profile information."
)
async def update_user_profile(
    update_data: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
) -> Any:
    """
    Update user profile.
    
    **Updatable Fields:**
    - full_name: User's full name
    - bio: User biography/description
    - phone_number: Contact number
    - timezone: User's timezone (e.g., "America/New_York")
    - language: Preferred language (e.g., "en", "es")
    
    **Note:**
    - Email and username cannot be changed via this endpoint
    - Use separate endpoints for email/username updates
    
    **Returns:**
    - Updated user profile
    
    **Errors:**
    - 400: Invalid data
    - 422: Validation error
    
    **Example Request:**
    ```
    {
      "full_name": "John Smith",
      "bio": "Data Scientist at TechCorp",
      "timezone": "America/Los_Angeles",
      "language": "en"
    }
    ```
    """
    try:
        logger.info(f"Updating profile for user {current_user.id}")
        
        updated_user = user_service.update_user(
            user_id=current_user.id,
            update_data=update_data
        )
        
        logger.info(f"Profile updated successfully for user {current_user.id}")
        
        return SuccessResponse(
            success=True,
            message="Profile updated successfully",
            data=UserResponse.model_validate(updated_user)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile update failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )


@router.post(
    "/me/avatar",
    response_model=SuccessResponse[UserResponse],
    summary="Upload Profile Picture",
    description="Upload or update user's profile picture."
)
async def upload_profile_picture(
    file: UploadFile = File(..., description="Profile picture (max 5MB, jpg/png)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
) -> Any:
    """
    Upload profile picture.
    
    **Requirements:**
    - File format: JPEG, PNG, or GIF
    - Max file size: 5 MB
    - Recommended dimensions: 400x400 pixels
    - Image will be automatically resized and cropped
    
    **Process:**
    1. Validates file type and size
    2. Resizes to 400x400 (square)
    3. Optimizes for web
    4. Saves to storage
    5. Updates user profile
    
    **Returns:**
    - Updated user profile with new avatar URL
    
    **Errors:**
    - 400: Invalid file type or size
    - 413: File too large
    """
    try:
        logger.info(f"Uploading profile picture for user {current_user.id}")
        
        # Validate file type
        allowed_types = ["image/jpeg", "image/png", "image/gif"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Only JPEG, PNG, and GIF are allowed."
            )
        
        # Validate file size (5MB)
        max_size = 5 * 1024 * 1024
        file_content = await file.read()
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File too large. Maximum size is 5MB."
            )
        
        # Upload and process
        updated_user = await user_service.upload_profile_picture(
            user_id=current_user.id,
            file_content=file_content,
            filename=file.filename
        )
        
        logger.info(f"Profile picture uploaded successfully for user {current_user.id}")
        
        return SuccessResponse(
            success=True,
            message="Profile picture uploaded successfully",
            data=UserResponse.model_validate(updated_user)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile picture upload failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload profile picture"
        )


# ============================================================
# PASSWORD MANAGEMENT
# ============================================================

@router.post(
    "/me/change-password",
    response_model=MessageResponse,
    summary="Change Password",
    description="Change the current user's password."
)
async def change_password(
    password_data: UserPasswordChange,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Change user password.
    
    **Required Fields:**
    - current_password: Current password for verification
    - new_password: New password (must meet requirements)
    
    **Password Requirements:**
    - Minimum 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character (!@#$%^&*()_+-=[]{}|;:,.<>?)
    
    **Security:**
    - Requires current password verification
    - Invalidates all existing refresh tokens
    - Sends notification email
    - Updates password_changed_at timestamp
    
    **Returns:**
    - Success message
    
    **Errors:**
    - 400: Current password incorrect
    - 400: New password doesn't meet requirements
    - 400: New password same as current
    """
    try:
        logger.info(f"Password change requested for user {current_user.id}")
        
        # Verify current password
        if not verify_password(password_data.current_password, current_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Check if new password is different
        if verify_password(password_data.new_password, current_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password must be different from current password"
            )
        
        # Validate new password strength
        from app.core.security import validate_password_strength
        is_valid, error_message = validate_password_strength(password_data.new_password)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_message
            )
        
        # Update password
        from datetime import datetime, timezone
        current_user.hashed_password = get_password_hash(password_data.new_password)
        current_user.password_changed_at = datetime.now(timezone.utc)
        db.commit()
        
        logger.info(f"Password changed successfully for user {current_user.id}")
        
        # TODO: Send notification email
        # TODO: Invalidate all refresh tokens
        
        return MessageResponse(
            message="Password changed successfully. Please login again with your new password."
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )


# ============================================================
# PREFERENCES
# ============================================================

@router.get(
    "/me/preferences",
    response_model=UserPreferences,
    summary="Get User Preferences",
    description="Get user's application preferences."
)
async def get_user_preferences(
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get user preferences.
    
    **Returns:**
    - timezone: User's timezone
    - language: Preferred language
    - theme: UI theme preference (if stored)
    - notification_settings: Email/push notification preferences
    - display_preferences: UI display settings
    """
    return UserPreferences(
        timezone=current_user.timezone,
        language=current_user.language,
        # Add more preferences as needed
    )


@router.patch(
    "/me/preferences",
    response_model=SuccessResponse[UserPreferences],
    summary="Update User Preferences",
    description="Update user's application preferences."
)
async def update_user_preferences(
    preferences: UserPreferences,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Update user preferences.
    
    **Updatable Preferences:**
    - timezone: User's timezone
    - language: Preferred language
    - theme: UI theme (light/dark)
    - notification_settings: Email/push preferences
    
    **Returns:**
    - Updated preferences
    """
    try:
        current_user.timezone = preferences.timezone
        current_user.language = preferences.language
        db.commit()
        
        return SuccessResponse(
            success=True,
            message="Preferences updated successfully",
            data=preferences
        )
    
    except Exception as e:
        logger.error(f"Preferences update failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update preferences"
        )


# ============================================================
# ACCOUNT MANAGEMENT
# ============================================================

@router.post(
    "/me/deactivate",
    response_model=MessageResponse,
    summary="Deactivate Account",
    description="Deactivate the current user's account (soft delete)."
)
async def deactivate_account(
    password: str = Query(..., description="Password for confirmation"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Deactivate user account.
    
    **Query Parameters:**
    - password: User's password for confirmation
    
    **Process:**
    - Verifies password
    - Marks account as inactive
    - Retains data for 30 days
    - Can be reactivated by logging in
    
    **Note:**
    - This is a soft delete
    - Data is retained and can be recovered
    - For permanent deletion, contact support
    
    **Returns:**
    - Success message
    
    **Errors:**
    - 400: Incorrect password
    """
    try:
        # Verify password
        if not verify_password(password, current_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Incorrect password"
            )
        
        # Deactivate account
        current_user.is_active = False
        from datetime import datetime, timezone
        current_user.deleted_at = datetime.now(timezone.utc)
        db.commit()
        
        logger.info(f"Account deactivated for user {current_user.id}")
        
        return MessageResponse(
            message="Account deactivated successfully. You can reactivate by logging in within 30 days."
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Account deactivation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to deactivate account"
        )


@router.delete(
    "/me",
    response_model=MessageResponse,
    summary="Delete Account Permanently",
    description="Permanently delete the user's account and all data."
)
async def delete_account_permanently(
    password: str = Query(..., description="Password for confirmation"),
    confirmation: str = Query(..., description="Type 'DELETE' to confirm"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
) -> Any:
    """
    Permanently delete account.
    
    **⚠️ WARNING: This action is irreversible!**
    
    **Query Parameters:**
    - password: User's password
    - confirmation: Must type "DELETE" exactly
    
    **What Gets Deleted:**
    - User account
    - All datasets
    - All uploaded files
    - All generated reports
    - All insights
    - All visualizations
    
    **Returns:**
    - Success message
    
    **Errors:**
    - 400: Incorrect password or confirmation
    """
    try:
        # Verify confirmation
        if confirmation != "DELETE":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Confirmation text must be 'DELETE'"
            )
        
        # Verify password
        if not verify_password(password, current_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Incorrect password"
            )
        
        # Delete account
        user_service.delete_user_permanently(current_user.id)
        
        logger.warning(f"Account permanently deleted: {current_user.id}")
        
        return MessageResponse(
            message="Account and all data permanently deleted"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Account deletion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete account"
        )


# ============================================================
# ADMIN: USER MANAGEMENT
# ============================================================

@router.get(
    "/",
    response_model=dict,
    summary="List All Users (Admin)",
    description="Get paginated list of all users (admin only)."
)
async def list_users(
    pagination: PaginationParams = Depends(get_pagination_params),
    role: Optional[UserRole] = Query(None, description="Filter by role"),
    subscription: Optional[UserSubscription] = Query(None, description="Filter by subscription"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    search: Optional[str] = Query(None, description="Search in email, username, or name"),
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin_user),
    user_service: UserService = Depends(get_user_service),
) -> Any:
    """
    List all users (Admin only).
    
    **Query Parameters:**
    - skip: Pagination offset
    - limit: Results per page (max 100)
    - role: Filter by role
    - subscription: Filter by subscription tier
    - is_active: Filter by active status
    - search: Search query
    
    **Returns:**
    - Paginated list of users
    - Total count
    - Filters applied
    
    **Authentication:**
    - Requires admin role
    """
    try:
        users, total = user_service.get_all_users(
            skip=pagination.skip,
            limit=pagination.limit,
            role=role,
            subscription=subscription,
            is_active=is_active,
            search=search,
        )
        
        return {
            "items": [UserResponse.model_validate(u) for u in users],
            "total": total,
            "page": pagination.page,
            "size": pagination.limit,
            "pages": (total + pagination.limit - 1) // pagination.limit,
        }
    
    except Exception as e:
        logger.error(f"Failed to list users: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users"
        )


@router.get(
    "/{user_id}",
    response_model=UserResponse,
    summary="Get User by ID (Admin)",
    description="Get detailed information about a specific user (admin only)."
)
async def get_user_by_id(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin_user),
) -> Any:
    """
    Get user by ID (Admin only).
    
    **Path Parameters:**
    - user_id: User ID
    
    **Returns:**
    - Complete user information
    
    **Authentication:**
    - Requires admin role
    
    **Errors:**
    - 404: User not found
    """
    user = db.get(User, user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse.model_validate(user)


@router.patch(
    "/{user_id}/role",
    response_model=SuccessResponse[UserResponse],
    summary="Update User Role (Admin)",
    description="Update a user's role (admin only)."
)
async def update_user_role(
    user_id: int,
    new_role: UserRole = Query(..., description="New role"),
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin_user),
) -> Any:
    """
    Update user role (Admin only).
    
    **Path Parameters:**
    - user_id: User ID
    
    **Query Parameters:**
    - new_role: New role to assign
    
    **Available Roles:**
    - user: Regular user
    - premium: Premium user
    - admin: Administrator
    - enterprise: Enterprise user
    
    **Returns:**
    - Updated user object
    
    **Errors:**
    - 404: User not found
    """
    user = db.get(User, user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user.role = new_role
    db.commit()
    
    logger.info(f"Admin {admin.id} changed role of user {user_id} to {new_role.value}")
    
    return SuccessResponse(
        success=True,
        message=f"User role updated to {new_role.value}",
        data=UserResponse.model_validate(user)
    )


@router.post(
    "/{user_id}/deactivate",
    response_model=MessageResponse,
    summary="Deactivate User (Admin)",
    description="Deactivate a user account (admin only)."
)
async def deactivate_user_admin(
    user_id: int,
    reason: str = Query(..., description="Reason for deactivation"),
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin_user),
) -> Any:
    """
    Deactivate user (Admin only).
    
    **Path Parameters:**
    - user_id: User ID
    
    **Query Parameters:**
    - reason: Reason for deactivation
    
    **Returns:**
    - Success message
    
    **Errors:**
    - 404: User not found
    """
    user = db.get(User, user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user.is_active = False
    db.commit()
    
    logger.warning(
        f"Admin {admin.id} deactivated user {user_id}. Reason: {reason}"
    )
    
    return MessageResponse(
        message=f"User {user_id} deactivated successfully"
    )
