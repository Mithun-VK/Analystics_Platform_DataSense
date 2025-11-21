"""
Dataset model and related entities.
Handles data file uploads, processing, and analysis results.
"""

import enum
from datetime import datetime
from typing import List, Optional, Dict, Any ,TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.user import User
from sqlalchemy import (
    String,
    Integer,
    BigInteger,
    Boolean,
    Enum,
    Text,
    JSON,
    ForeignKey,
    DateTime,
    Float,
    CheckConstraint,
    Index,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, SoftDeleteMixin


class DatasetStatus(str, enum.Enum):
    """Dataset processing status enumeration."""
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    CLEANING = "cleaning"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class Dataset(Base, TimestampMixin, SoftDeleteMixin):
    """
    Dataset model for uploaded data files.
    
    Stores metadata about uploaded files, processing status,
    and analysis results including insights and visualizations.
    
    Attributes:
        id: Primary key
        name: User-defined dataset name
        description: Optional dataset description
        file_name: Original uploaded file name
        file_path: Path to stored file (local or S3)
        file_size_bytes: File size in bytes
        file_type: File extension/type (csv, xlsx, etc.)
        status: Current processing status
        owner_id: Foreign key to user who owns this dataset
        owner: Relationship to User model
        row_count: Number of rows in dataset
        column_count: Number of columns in dataset
        columns_info: JSON metadata about columns
        statistics: Relationship to dataset statistics
        insights: List of AI-generated insights
        visualizations: List of generated visualizations
        processing_started_at: When processing began
        processing_completed_at: When processing completed
        processing_error: Error message if processing failed
        eda_report_url: URL to generated EDA report
        tags: List of user-defined tags
        is_public: Whether dataset is publicly accessible
        download_count: Number of times downloaded
        view_count: Number of times viewed
    """
    
    __tablename__ = "datasets"
    
    # Primary Key
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Dataset unique identifier"
    )
    
    # Basic Information
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Dataset name"
    )
    
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Dataset description"
    )
    
    # File Information
    file_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Original file name"
    )
    
    file_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        unique=True,
        comment="Path to stored file"
    )
    
    file_size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="File size in bytes"
    )
    
    file_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        comment="File type/extension (csv, xlsx, etc.)"
    )
    
    file_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment="SHA-256 hash of file content"
    )
    
    # Processing Status
    status: Mapped[DatasetStatus] = mapped_column(
        Enum(DatasetStatus, native_enum=False, length=20),
        nullable=False,
        default=DatasetStatus.UPLOADING,
        server_default=DatasetStatus.UPLOADING.value,
        index=True,
        comment="Processing status"
    )
    
    processing_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Processing start timestamp"
    )
    
    processing_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Processing completion timestamp"
    )
    
    processing_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Error message if processing failed"
    )
    
    processing_duration_seconds: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Total processing time in seconds"
    )
    
    # Ownership
    owner_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User ID who owns this dataset"
    )
    
    owner: Mapped["User"] = relationship(
        "User",
        back_populates="datasets",
        lazy="joined",
    )
    
    # Data Dimensions
    row_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of rows in dataset"
    )
    
    column_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of columns in dataset"
    )
    
    # Column Metadata (stored as JSON)
    columns_info: Mapped[Dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="JSON metadata about columns (types, nulls, etc.)"
    )
    
    # Data Quality Metrics
    missing_values_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Total missing values across all columns"
    )
    
    duplicate_rows_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of duplicate rows detected"
    )
    
    data_quality_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Overall data quality score (0-100)"
    )
    
    # Analysis Results
    eda_report_url: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="URL to generated EDA HTML report"
    )
    
    eda_report_generated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="EDA report generation timestamp"
    )
    
    # Relationships
    statistics: Mapped[Optional["DatasetStatistics"]] = relationship(
        "DatasetStatistics",
        back_populates="dataset",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    
    insights: Mapped[List["DatasetInsight"]] = relationship(
        "DatasetInsight",
        back_populates="dataset",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="desc(DatasetInsight.created_at)",
    )
    
    visualizations: Mapped[List["DatasetVisualization"]] = relationship(
        "DatasetVisualization",
        back_populates="dataset",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="DatasetVisualization.order",
    )
    
    # Metadata & Tags
    tags: Mapped[List[str] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="User-defined tags for categorization"
    )
    
    metadata_json: Mapped[Dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Additional metadata as JSON"
    )
    
    # Sharing & Visibility
    is_public: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        index=True,
        comment="Whether dataset is publicly accessible"
    )
    
    share_token: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        unique=True,
        index=True,
        comment="Token for sharing dataset via link"
    )
    
    # Usage Statistics
    view_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of times viewed"
    )
    
    download_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of times downloaded"
    )
    orders_path: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="Path to analytics orders csv"
    )
    customers_path: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="Path to analytics customers csv"
    )
    products_path: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="Path to analytics products csv"
    )
    marketing_path: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="Path to analytics marketing csv"
    )

    last_accessed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last access timestamp"
    )
    
    # Table Constraints
    __table_args__ = (
        CheckConstraint(
            "file_size_bytes > 0",
            name="file_size_positive"
        ),
        CheckConstraint(
            "row_count >= 0",
            name="row_count_non_negative"
        ),
        CheckConstraint(
            "column_count > 0",
            name="column_count_positive"
        ),
        CheckConstraint(
            "data_quality_score >= 0 AND data_quality_score <= 100",
            name="quality_score_range"
        ),
        Index("ix_datasets_owner_status", "owner_id", "status"),
        Index("ix_datasets_created_status", "created_at", "status"),
        Index("ix_datasets_public_status", "is_public", "status"),
        {"comment": "Datasets table for uploaded data files"}
    )
    
    # Helper Methods
    def is_processing(self) -> bool:
        """Check if dataset is currently being processed."""
        return self.status in [
            DatasetStatus.PROCESSING,
            DatasetStatus.CLEANING,
            DatasetStatus.ANALYZING,
        ]
        
    def has_failed(self) -> bool:
        """Check if dataset processing failed."""
        return self.status == DatasetStatus.FAILED
    
    def get_file_size_mb(self) -> float:
        """Get file size in megabytes."""
        return round(self.file_size_bytes / (1024 * 1024), 2)
    
    def increment_view_count(self) -> None:
        """Increment view counter."""
        self.view_count += 1
        self.last_accessed_at = datetime.now()
    
    def increment_download_count(self) -> None:
        """Increment download counter."""
        self.download_count += 1
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Dataset(id={self.id}, name='{self.name}', "
            f"status={self.status}, rows={self.row_count}, "
            f"cols={self.column_count})"
        )
    def is_ready(self) -> bool:
        """
        Check if all analytic file paths are present for analytics.
        (Optionally: you may combine with classic status check for backward compatibility)
        """
        return all([
            getattr(self, "orders_path", None),
            getattr(self, "customers_path", None),
            getattr(self, "products_path", None),
            getattr(self, "marketing_path", None)
        ]) or (self.file_path and self.status == DatasetStatus.COMPLETED)


class DatasetStatistics(Base, TimestampMixin):
    """
    Statistical summary of a dataset.
    
    Stores computed statistics for numerical and categorical columns.
    """
    
    __tablename__ = "dataset_statistics"
    
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    
    dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    
    dataset: Mapped["Dataset"] = relationship(
        "Dataset",
        back_populates="statistics",
    )
    
    # Numerical Statistics
    numerical_stats: Mapped[Dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Statistics for numerical columns (mean, std, etc.)"
    )
    
    # Categorical Statistics
    categorical_stats: Mapped[Dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Statistics for categorical columns (unique, top, freq)"
    )
    
    # Correlation Matrix
    correlation_matrix: Mapped[Dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Correlation matrix for numerical columns"
    )
    
    # Distribution Info
    distributions: Mapped[Dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Distribution information for columns"
    )
    
    __table_args__ = (
        {"comment": "Statistical summaries for datasets"}
    )


class DatasetInsight(Base, TimestampMixin):
    """
    AI-generated insight about a dataset.
    
    Stores insights generated by LLMs analyzing the dataset.
    """
    
    __tablename__ = "dataset_insights"
    
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    
    dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    dataset: Mapped["Dataset"] = relationship(
        "Dataset",
        back_populates="insights",
    )
    
    # Insight Content
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Insight title/summary"
    )
    
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Detailed insight content"
    )
    
    insight_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Type of insight (correlation, outlier, trend, etc.)"
    )
    
    confidence_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Confidence score for this insight (0-1)"
    )
    
    # AI Model Info
    model_used: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="AI model used to generate insight"
    )
    
    generation_time_seconds: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Time taken to generate insight"
    )
    
    # User Feedback
    is_helpful: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
        comment="User feedback: was this insight helpful?"
    )
    
    __table_args__ = (
        Index("ix_insights_dataset_type", "dataset_id", "insight_type"),
        {"comment": "AI-generated insights for datasets"}
    )


class DatasetVisualization(Base, TimestampMixin):
    """
    Visualization/chart generated for a dataset.
    
    Stores metadata about generated charts and their configurations.
    """
    
    __tablename__ = "dataset_visualizations"
    
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    
    dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    dataset: Mapped["Dataset"] = relationship(
        "Dataset",
        back_populates="visualizations",
    )
    
    # Visualization Info
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Visualization title"
    )
    
    chart_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Type of chart (bar, line, scatter, etc.)"
    )
    
    chart_url: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="URL to saved chart image"
    )
    
    # Configuration
    config: Mapped[Dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Chart configuration (axes, colors, etc.)"
    )
    
    data_columns: Mapped[List[str] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Columns used in this visualization"
    )
    
    # Display Order
    order: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Display order for visualizations"
    )
    
    __table_args__ = (
        Index("ix_visualizations_dataset_type", "dataset_id", "chart_type"),
        {"comment": "Generated visualizations for datasets"}
    )
class CleaningAuditLog(Base, TimestampMixin):
    """
    Audit log for data cleaning operations.
    
    Tracks detailed history of all cleaning operations performed on datasets
    including transformations, quality metrics, and configuration used.
    
    Attributes:
        id: Primary key
        dataset_id: Foreign key to dataset
        session_id: Unique session identifier for this cleaning operation
        started_at: When cleaning started
        completed_at: When cleaning completed
        duration_seconds: Total processing time
        status: Operation status (success, failed, partial)
        config_used: Cleaning configuration that was used
        original_shape: Original dataset dimensions
        final_shape: Final dataset dimensions after cleaning
        rows_removed: Number of rows removed
        columns_removed: Number of columns removed
        transformations_applied: List of transformations performed
        cleaning_steps: Detailed step-by-step log
        quality_metrics: Quality metrics calculated
        quality_alerts: Quality alerts generated
        original_file_path: Path to original file
        cleaned_file_path: Path to cleaned file
        error_message: Error message if operation failed
        user_id: User who initiated the cleaning
    """
    
    __tablename__ = "cleaning_audit_logs"
    
    # Primary Key
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Audit log unique identifier"
    )
    
    # Dataset Reference
    dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Dataset ID this cleaning was performed on"
    )
    
    dataset: Mapped["Dataset"] = relationship(
        "Dataset",
        backref="cleaning_audit_logs",
        lazy="joined",
    )
    
    # Session Identification
    session_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
        index=True,
        comment="Unique session ID for this cleaning operation"
    )
    
    # Timing Information
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Cleaning start timestamp"
    )
    
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Cleaning completion timestamp"
    )
    
    duration_seconds: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Total cleaning duration in seconds"
    )
    
    # Operation Status
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Operation status: success, failed, partial"
    )
    
    # Configuration Used
    config_used: Mapped[Dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Cleaning configuration that was used"
    )
    
    # Data Dimensions
    original_shape: Mapped[Dict[str, int] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Original dataset shape (rows, columns)"
    )
    
    final_shape: Mapped[Dict[str, int] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Final dataset shape after cleaning"
    )
    
    rows_removed: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of rows removed during cleaning"
    )
    
    columns_removed: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of columns removed during cleaning"
    )
    
    # Transformations Applied
    transformations_applied: Mapped[List[str] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="List of transformation names applied"
    )
    
    cleaning_steps: Mapped[List[Dict[str, Any]] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Detailed step-by-step cleaning log"
    )
    
    # Quality Metrics
    quality_metrics: Mapped[Dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Quality metrics calculated after cleaning"
    )
    
    quality_alerts: Mapped[List[Dict[str, Any]] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Quality alerts generated during cleaning"
    )
    
    # File Paths
    original_file_path: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="Path to original file before cleaning"
    )
    
    cleaned_file_path: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="Path to cleaned file after processing"
    )
    
    # Error Tracking
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Error message if cleaning operation failed"
    )
    
    # User Tracking
    user_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="User who initiated this cleaning operation"
    )
    
    user: Mapped[Optional["User"]] = relationship(
        "User",
        backref="cleaning_operations",
        lazy="joined",
    )
    
    # Table Constraints
    __table_args__ = (
        CheckConstraint(
            "duration_seconds >= 0",
            name="duration_non_negative"
        ),
        CheckConstraint(
            "rows_removed >= 0",
            name="rows_removed_non_negative"
        ),
        CheckConstraint(
            "columns_removed >= 0",
            name="columns_removed_non_negative"
        ),
        Index("ix_cleaning_audit_dataset_status", "dataset_id", "status"),
        Index("ix_cleaning_audit_started", "started_at"),
        Index("ix_cleaning_audit_user_dataset", "user_id", "dataset_id"),
        {"comment": "Audit trail for data cleaning operations"}
    )
    
    # Helper Methods
    def is_successful(self) -> bool:
        """Check if cleaning operation was successful."""
        return self.status == "success"
    
    def get_duration_minutes(self) -> float:
        """Get duration in minutes."""
        if self.duration_seconds:
            return round(self.duration_seconds / 60, 2)
        return 0.0
    
    def get_data_reduction_percent(self) -> float:
        """Calculate percentage of data removed."""
        if self.original_shape and self.rows_removed:
            original_rows = self.original_shape.get("rows", 0)
            if original_rows > 0:
                return round((self.rows_removed / original_rows) * 100, 2)
        return 0.0
    
    def get_quality_score(self) -> float:
        """Get overall quality score from metrics."""
        if self.quality_metrics:
            return self.quality_metrics.get("overall_score", 0.0)
        return 0.0
    
    def get_quality_grade(self) -> str:
        """Get quality grade from metrics."""
        if self.quality_metrics:
            return self.quality_metrics.get("quality_grade", "N/A")
        return "N/A"
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CleaningAuditLog(id={self.id}, session='{self.session_id}', "
            f"dataset_id={self.dataset_id}, status={self.status}, "
            f"duration={self.duration_seconds}s)"
        )
