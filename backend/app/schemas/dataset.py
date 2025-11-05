"""
Dataset-related Pydantic schemas.
Handles dataset uploads, processing, and analysis results.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_validator,
    model_validator,
)

from app.models.dataset import DatasetStatus


class DatasetBase(BaseModel):
    """
    Base dataset schema with common fields.
    """
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )
    
    name: Optional[str] = Field(
        None,
        description="Dataset name",
        min_length=1,
        max_length=255,
    )
    
    description: Optional[str] = Field(
        None,
        description="Dataset description",
        max_length=2000,
    )
    
    tags: Optional[List[str]] = Field(
        None,
        description="Tags for categorization",
        max_length=20,
    )
    
    is_public: Optional[bool] = Field(
        None,
        description="Whether dataset is publicly accessible",
    )
    
    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate tags format."""
        if v is None:
            return v
        
        # Remove duplicates and empty strings
        tags = list(set(tag.strip().lower() for tag in v if tag.strip()))
        
        # Limit number of tags
        if len(tags) > 10:
            raise ValueError("Maximum 10 tags allowed")
        
        # Validate tag length
        for tag in tags:
            if len(tag) > 50:
                raise ValueError("Tag length cannot exceed 50 characters")
        
        return tags


class DatasetCreate(DatasetBase):
    """
    Dataset creation schema.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Sales Data 2024",
                "description": "Annual sales data with customer demographics",
                "tags": ["sales", "2024", "customer-data"],
            }
        }
    )
    
    name: str = Field(
        ...,
        description="Dataset name (required)",
        min_length=1,
        max_length=255,
    )


class DatasetUpdate(DatasetBase):
    """
    Dataset update schema.
    
    All fields are optional for partial updates.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Updated Dataset Name",
                "description": "Updated description",
                "tags": ["updated", "new-tag"],
                "is_public": False,
            }
        }
    )


class DatasetUpload(BaseModel):
    """
    Dataset upload metadata schema.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Customer Data",
                "description": "Customer purchase history",
                "file_type": "csv",
            }
        }
    )
    
    name: str = Field(
        ...,
        description="Dataset name",
        min_length=1,
        max_length=255,
    )
    
    description: Optional[str] = Field(
        None,
        description="Dataset description",
        max_length=2000,
    )
    
    file_type: Optional[str] = Field(
        None,
        description="File type override",
        pattern="^(csv|xlsx|xls|json|parquet)$",
    )


class DatasetCleaningConfig(BaseModel):
    """
    Data cleaning configuration schema.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "remove_duplicates": True,
                "handle_missing": "drop",
                "columns_to_drop": ["unnecessary_col"],
                "fill_values": {"age": 0, "name": "Unknown"},
            }
        }
    )
    
    remove_duplicates: bool = Field(
        default=True,
        description="Remove duplicate rows",
    )
    
    handle_missing: str = Field(
        default="drop",
        description="How to handle missing values",
        pattern="^(drop|fill|forward|backward)$",
    )
    
    columns_to_drop: Optional[List[str]] = Field(
        None,
        description="Columns to drop from dataset",
    )
    
    fill_values: Optional[Dict[str, Any]] = Field(
        None,
        description="Values to fill for specific columns",
    )
    
    outlier_detection: bool = Field(
        default=False,
        description="Detect and handle outliers",
    )


class DatasetEDAConfig(BaseModel):
    """
    EDA generation configuration schema.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "generate_correlations": True,
                "generate_distributions": True,
                "sample_size": 10000,
            }
        }
    )
    
    generate_correlations: bool = Field(
        default=True,
        description="Generate correlation matrices",
    )
    
    generate_distributions: bool = Field(
        default=True,
        description="Generate distribution plots",
    )
    
    sample_size: Optional[int] = Field(
        None,
        description="Sample size for large datasets",
        gt=0,
        le=100000,
    )
    
    minimal_report: bool = Field(
        default=False,
        description="Generate minimal EDA report for faster processing",
    )


class DatasetStatisticsResponse(BaseModel):
    """
    Dataset statistics response schema.
    """
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "dataset_id": 1,
                "numerical_stats": {
                    "age": {"mean": 35.5, "std": 12.3, "min": 18, "max": 65}
                },
                "categorical_stats": {
                    "gender": {"unique": 2, "top": "Male", "freq": 550}
                },
            }
        }
    )
    
    id: int
    dataset_id: int
    numerical_stats: Optional[Dict[str, Any]] = None
    categorical_stats: Optional[Dict[str, Any]] = None
    correlation_matrix: Optional[Dict[str, Any]] = None
    distributions: Optional[Dict[str, Any]] = None


class DatasetInsightResponse(BaseModel):
    """
    Dataset insight response schema.
    """
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "dataset_id": 1,
                "title": "Strong Correlation Detected",
                "content": "There is a strong positive correlation (0.87) between age and income.",
                "insight_type": "correlation",
                "confidence_score": 0.92,
                "created_at": "2025-01-01T00:00:00Z",
            }
        }
    )
    
    id: int
    dataset_id: int
    title: str
    content: str
    insight_type: str
    confidence_score: Optional[float] = None
    model_used: Optional[str] = None
    is_helpful: Optional[bool] = None
    created_at: datetime


class DatasetVisualizationResponse(BaseModel):
    """
    Dataset visualization response schema.
    """
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "dataset_id": 1,
                "title": "Age Distribution",
                "chart_type": "histogram",
                "chart_url": "https://storage.example.com/charts/chart_123.png",
                "data_columns": ["age"],
                "order": 1,
            }
        }
    )
    
    id: int
    dataset_id: int
    title: str
    chart_type: str
    chart_url: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    data_columns: Optional[List[str]] = None
    order: int


class DatasetResponse(DatasetBase):
    """
    Basic dataset response schema.
    """
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "name": "Sales Data 2024",
                "description": "Annual sales data",
                "file_name": "sales_2024.csv",
                "file_size_bytes": 1048576,
                "file_type": "csv",
                "status": "completed",
                "owner_id": 1,
                "row_count": 10000,
                "column_count": 15,
                "created_at": "2025-01-01T00:00:00Z",
            }
        }
    )
    
    id: int
    name: str
    description: Optional[str] = None
    file_name: str
    file_size_bytes: int
    file_type: str
    status: DatasetStatus
    owner_id: int
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    tags: Optional[List[str]] = None
    is_public: bool
    view_count: int
    download_count: int
    created_at: datetime
    updated_at: datetime
    
    @field_validator("file_size_bytes")
    @classmethod
    def format_file_size(cls, v: int) -> int:
        """Validate file size is positive."""
        if v <= 0:
            raise ValueError("File size must be positive")
        return v


class DatasetDetail(DatasetResponse):
    """
    Detailed dataset response schema.
    
    Includes processing status, statistics, insights, and visualizations.
    """
    
    model_config = ConfigDict(
        from_attributes=True,
    )
    
    file_path: Optional[str] = None
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    processing_error: Optional[str] = None
    processing_duration_seconds: Optional[float] = None
    columns_info: Optional[Dict[str, Any]] = None
    missing_values_count: Optional[int] = None
    duplicate_rows_count: Optional[int] = None
    data_quality_score: Optional[float] = None
    eda_report_url: Optional[str] = None
    eda_report_generated_at: Optional[datetime] = None
    metadata_json: Optional[Dict[str, Any]] = None
    share_token: Optional[str] = None
    last_accessed_at: Optional[datetime] = None
    statistics: Optional[DatasetStatisticsResponse] = None
    insights: Optional[List[DatasetInsightResponse]] = None
    visualizations: Optional[List[DatasetVisualizationResponse]] = None


class DatasetList(BaseModel):
    """
    Paginated dataset list response schema.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [],
                "total": 50,
                "page": 1,
                "size": 20,
                "pages": 3,
            }
        }
    )
    
    items: List[DatasetResponse] = Field(
        ...,
        description="List of datasets",
    )
    
    total: int = Field(
        ...,
        description="Total number of datasets",
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
    )
    
    pages: int = Field(
        ...,
        description="Total number of pages",
        ge=0,
    )
