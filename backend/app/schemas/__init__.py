"""
Pydantic schemas package.
Exports all request/response schemas for API validation.
"""

from app.schemas.token import (
    Token,
    TokenPayload,
    RefreshToken,
    EmailVerificationToken,
    PasswordResetToken,
)
from app.schemas.user import (
    UserBase,
    UserCreate,
    UserUpdate,
    UserResponse,
    UserLogin,
    UserProfile,
    UserPasswordChange,
    UserPasswordReset,
    UserEmailVerification,
    UserSubscriptionUpdate,
    UserPreferences,
)
from app.schemas.dataset import (
    DatasetBase,
    DatasetCreate,
    DatasetUpdate,
    DatasetResponse,
    DatasetList,
    DatasetDetail,
    DatasetUpload,
    DatasetStatisticsResponse,
    DatasetInsightResponse,
    DatasetVisualizationResponse,
    DatasetCleaningConfig,
    DatasetEDAConfig,
)
from app.schemas.response import (
    SuccessResponse,
    ErrorResponse,
    PaginatedResponse,
    MessageResponse,
)

__all__ = [
    # Token schemas
    "Token",
    "TokenPayload",
    "RefreshToken",
    "EmailVerificationToken",
    "PasswordResetToken",
    # User schemas
    "UserBase",
    "UserCreate",
    "UserUpdate",
    "UserResponse",
    "UserLogin",
    "UserProfile",
    "UserPasswordChange",
    "UserPasswordReset",
    "UserEmailVerification",
    "UserSubscriptionUpdate",
    "UserPreferences",
    # Dataset schemas
    "DatasetBase",
    "DatasetCreate",
    "DatasetUpdate",
    "DatasetResponse",
    "DatasetList",
    "DatasetDetail",
    "DatasetUpload",
    "DatasetStatisticsResponse",
    "DatasetInsightResponse",
    "DatasetVisualizationResponse",
    "DatasetCleaningConfig",
    "DatasetEDAConfig",
    # Response schemas
    "SuccessResponse",
    "ErrorResponse",
    "PaginatedResponse",
    "MessageResponse",
]
