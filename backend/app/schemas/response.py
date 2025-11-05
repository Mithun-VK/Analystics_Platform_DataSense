"""
Generic API response schemas.
Standard response wrappers for consistency.
"""

from typing import Any, Optional, Generic, TypeVar, List

from pydantic import BaseModel, Field, ConfigDict


T = TypeVar("T")


class SuccessResponse(BaseModel, Generic[T]):
    """
    Generic success response wrapper.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "data": {"id": 1, "name": "Example"},
            }
        }
    )
    
    success: bool = Field(
        default=True,
        description="Operation success status",
    )
    
    message: str = Field(
        ...,
        description="Success message",
    )
    
    data: Optional[T] = Field(
        None,
        description="Response data",
    )


class ErrorResponse(BaseModel):
    """
    Error response schema.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": False,
                "error": "Validation Error",
                "message": "Invalid input data",
                "details": ["Field 'email' is required"],
            }
        }
    )
    
    success: bool = Field(
        default=False,
        description="Operation success status",
    )
    
    error: str = Field(
        ...,
        description="Error type",
    )
    
    message: str = Field(
        ...,
        description="Error message",
    )
    
    details: Optional[List[str]] = Field(
        None,
        description="Detailed error information",
    )


class MessageResponse(BaseModel):
    """
    Simple message response schema.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Operation completed successfully"
            }
        }
    )
    
    message: str = Field(
        ...,
        description="Response message",
    )


class PaginatedResponse(BaseModel, Generic[T]):
    """
    Generic paginated response wrapper.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [],
                "total": 100,
                "page": 1,
                "size": 20,
                "pages": 5,
            }
        }
    )
    
    items: List[T] = Field(
        ...,
        description="List of items",
    )
    
    total: int = Field(
        ...,
        description="Total number of items",
        ge=0,
    )
    
    page: int = Field(
        ...,
        description="Current page number",
        ge=1,
    )
    
    size: int = Field(
        ...,
        description="Number of items per page",
        ge=1,
        le=100,
    )
    
    pages: int = Field(
        ...,
        description="Total number of pages",
        ge=0,
    )
