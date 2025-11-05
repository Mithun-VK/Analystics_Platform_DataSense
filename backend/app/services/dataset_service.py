"""
Dataset Service.

Handles dataset management including file uploads, CRUD operations,
metadata extraction, and dataset processing orchestration.
"""

import hashlib
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, BinaryIO

import pandas as pd
from fastapi import Depends, HTTPException, status, UploadFile
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import IntegrityError

from app.core.config import settings
from app.models.dataset import (
    Dataset,
    DatasetStatus,
    DatasetStatistics,
    DatasetInsight,
    DatasetVisualization,
)
from app.models.user import User
from app.schemas.dataset import (
    DatasetCreate,
    DatasetUpdate,
    DatasetUpload,
)
from app.database import get_db

import logging

logger = logging.getLogger(__name__)


class DatasetService:
    """
    Dataset service for managing data files and analysis.
    
    Implements:
    - File upload with validation
    - Dataset CRUD operations
    - Metadata extraction
    - Storage management (local/S3)
    - Data quality checks
    - Pagination and filtering
    """
    
    # Supported file types
    ALLOWED_EXTENSIONS = {
        ".csv": "text/csv",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
        ".json": "application/json",
        ".parquet": "application/octet-stream",
    }
    
    def __init__(self, db: Session):
        """
        Initialize dataset service with database session.
        
        Args:
            db: SQLAlchemy database session
        """
        self.db = db
        self._ensure_upload_directory()
    
    def _ensure_upload_directory(self) -> None:
        """Create upload directory if it doesn't exist."""
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================================
    # FILE UPLOAD & VALIDATION
    # ============================================================
    
    async def upload_dataset(
        self,
        file: UploadFile,
        user: User,
        metadata: DatasetUpload
    ) -> Dataset:
        """
        Upload and process a dataset file.
        
        Args:
            file: Uploaded file
            user: Owner user
            metadata: Dataset metadata
            
        Returns:
            Created dataset instance
            
        Raises:
            HTTPException: If validation fails or upload errors occur
        """
        try:
            logger.info(f"Starting upload for user {user.id}")
            
            # Validate file extension only (don't read yet)
            self._validate_file(file)
            
            # Generate unique file path
            file_path = self._generate_file_path(user.id, file.filename)
            
            # Calculate file hash while saving
            file_hash = await self._save_file(file, file_path)
            
            # Check for duplicate files
            existing = await self._check_duplicate_file(user.id, file_hash)
            if existing:
                # Clean up uploaded file
                os.remove(file_path)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"This file already exists as dataset: {existing.name}",
                )
            
            # Extract file metadata
            file_stats = os.stat(file_path)
            file_type = self._get_file_extension(file.filename)
            
            # Verify file has content
            if file_stats.st_size == 0:
                os.remove(file_path)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="File is empty",
                )
            
            # Create dataset record
            try:
                dataset = Dataset(
                    name=metadata.name,
                    description=metadata.description,
                    file_name=file.filename,
                    file_path=str(file_path),
                    file_size_bytes=file_stats.st_size,
                    file_type=file_type,
                    file_hash=file_hash,
                    status=DatasetStatus.UPLOADED,
                    owner_id=user.id,
                )
                
                self.db.add(dataset)
                
                # Update user statistics
                user.datasets_count += 1
                user.storage_used_bytes += file_stats.st_size
                
                self.db.commit()
                self.db.refresh(dataset)
                
                logger.info(f"Dataset {dataset.id} uploaded successfully by user {user.id}")
                
                # Trigger async processing
                self._trigger_dataset_processing(dataset.id)
                
                return dataset
                
            except IntegrityError as e:
                self.db.rollback()
                # Clean up file on error
                if os.path.exists(file_path):
                    os.remove(file_path)
                logger.error(f"Database error during upload: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create dataset record",
                ) from e
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Upload failed: {str(e)}"
            ) from e
    
    def _validate_file(self, file: UploadFile) -> None:
        """
        Validate uploaded file (check extension and size only).
        
        Args:
            file: Uploaded file
            
        Raises:
            HTTPException: If validation fails
        """
        # Check filename
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required",
            )
        
        # Check file extension
        file_ext = self._get_file_extension(file.filename)
        if file_ext not in self.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type {file_ext} not supported. "
                       f"Allowed types: {', '.join(self.ALLOWED_EXTENSIONS.keys())}",
            )
        
        # Check file size if available
        if file.size and file.size > settings.MAX_UPLOAD_SIZE:
            max_mb = settings.MAX_UPLOAD_SIZE / (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds maximum allowed size of {max_mb:.0f}MB",
            )
    
    async def _save_file(
        self,
        file: UploadFile,
        file_path: Path
    ) -> str:
        """
        Save uploaded file and calculate hash.
        
        Args:
            file: Uploaded file
            file_path: Destination path
            
        Returns:
            SHA-256 hash of file content
        """
        hasher = hashlib.sha256()
        
        try:
            # ✅ CRITICAL: Seek to beginning to reset file pointer
            # (In case it was read before)
            await file.seek(0)
            
            # Read entire file into memory
            content = await file.read()
            
            if not content or len(content) == 0:
                raise ValueError("File is empty after read")
            
            logger.info(f"Read {len(content)} bytes from file {file.filename}")
            
            # Write to disk and compute hash
            with open(file_path, "wb") as buffer:
                buffer.write(content)
                hasher.update(content)
            
            file_hash = hasher.hexdigest()
            logger.info(f"File saved: {file_path} ({len(content)} bytes, hash: {file_hash[:8]}...)")
            
            return file_hash
            
        except Exception as e:
            # Clean up partial file on error
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
            logger.error(f"File save failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save file: {str(e)}",
            ) from e
    
    def _generate_file_path(
        self,
        user_id: int,
        filename: str
    ) -> Path:
        """
        Generate unique file path for uploaded file.
        
        Args:
            user_id: User ID
            filename: Original filename
            
        Returns:
            Unique file path
        """
        # Sanitize filename
        safe_filename = self._sanitize_filename(filename)
        
        # Create user-specific subdirectory
        user_dir = Path(settings.UPLOAD_DIR) / f"user_{user_id}"
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(safe_filename)
        unique_filename = f"{name}_{timestamp}{ext}"
        
        # Ensure uniqueness
        file_path = user_dir / unique_filename
        counter = 1
        while file_path.exists():
            unique_filename = f"{name}_{timestamp}_{counter}{ext}"
            file_path = user_dir / unique_filename
            counter += 1
        
        logger.info(f"Generated file path: {file_path}")
        return file_path
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent security issues.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove path components
        filename = os.path.basename(filename)
        
        # Remove dangerous characters
        dangerous_chars = ['..', '/', '\\', '\x00', ':', '*', '?', '"', '<', '>', '|']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        name, ext = os.path.splitext(filename)
        if len(name) > 200:
            name = name[:200]
        
        return f"{name}{ext}"
    
    def _get_file_extension(self, filename: str) -> str:
        """Get lowercase file extension."""
        return os.path.splitext(filename)[1].lower()
    
    async def _check_duplicate_file(
        self,
        user_id: int,
        file_hash: str
    ) -> Optional[Dataset]:
        """
        Check if file with same hash already exists for user.
        
        Args:
            user_id: User ID
            file_hash: File hash to check
            
        Returns:
            Existing dataset or None
        """
        stmt = select(Dataset).where(
            and_(
                Dataset.owner_id == user_id,
                Dataset.file_hash == file_hash,
                Dataset.is_deleted == False,
            )
        )
        result = self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    def _trigger_dataset_processing(self, dataset_id: int) -> None:
        """
        Trigger async dataset processing.
        
        In production, use Celery task.
        
        Args:
            dataset_id: Dataset ID to process
        """
        # For MVP, process synchronously
        if settings.DEBUG:
            self._extract_metadata_sync(dataset_id)
    
    # ============================================================
    # METADATA EXTRACTION
    # ============================================================
    
    def _extract_metadata_sync(self, dataset_id: int) -> None:
        """
        Extract metadata from dataset (synchronous for MVP).
        
        Args:
            dataset_id: Dataset ID
        """
        dataset = self.db.get(Dataset, dataset_id)
        if not dataset:
            logger.warning(f"Dataset {dataset_id} not found for metadata extraction")
            return
        
        try:
            dataset.status = DatasetStatus.PROCESSING
            dataset.processing_started_at = datetime.now()
            self.db.commit()
            
            logger.info(f"Extracting metadata for dataset {dataset_id}")
            
            # Read file based on type
            df = self._read_dataframe(dataset.file_path, dataset.file_type)
            
            # Extract basic metadata
            dataset.row_count = len(df)
            dataset.column_count = len(df.columns)
            
            logger.info(f"Dataset {dataset_id}: {dataset.row_count} rows, {dataset.column_count} columns")
            
            # Extract column information
            columns_info = self._extract_column_info(df)
            dataset.columns_info = columns_info
            
            # Calculate data quality metrics
            dataset.missing_values_count = int(df.isnull().sum().sum())
            dataset.duplicate_rows_count = int(df.duplicated().sum())
            
            # Calculate quality score (0-100)
            total_cells = dataset.row_count * dataset.column_count
            if total_cells > 0:
                missing_ratio = dataset.missing_values_count / total_cells
                duplicate_ratio = dataset.duplicate_rows_count / dataset.row_count if dataset.row_count > 0 else 0
                quality_score = 100 * (1 - (missing_ratio * 0.7 + duplicate_ratio * 0.3))
                dataset.data_quality_score = round(max(0, min(100, quality_score)), 2)
            else:
                dataset.data_quality_score = 0.0
            
            # Mark as completed
            dataset.status = DatasetStatus.COMPLETED
            dataset.processing_completed_at = datetime.now()
            
            if dataset.processing_started_at:
                duration = (dataset.processing_completed_at - dataset.processing_started_at).total_seconds()
                dataset.processing_duration_seconds = duration
            
            self.db.commit()
            logger.info(f"Metadata extraction completed for dataset {dataset_id}. Quality score: {dataset.data_quality_score}")
            
        except Exception as e:
            dataset.status = DatasetStatus.FAILED
            dataset.processing_error = str(e)
            self.db.commit()
            logger.error(f"Metadata extraction failed for dataset {dataset_id}: {str(e)}", exc_info=True)
    
    def _read_dataframe(self, file_path: str, file_type: str) -> pd.DataFrame:
        """
        Read file into pandas DataFrame.
        
        Args:
            file_path: Path to file
            file_type: File extension
            
        Returns:
            DataFrame
            
        Raises:
            Exception: If reading fails
        """
        try:
            if file_type == ".csv":
                return pd.read_csv(file_path)
            elif file_type in [".xlsx", ".xls"]:
                return pd.read_excel(file_path)
            elif file_type == ".json":
                return pd.read_json(file_path)
            elif file_type == ".parquet":
                return pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {str(e)}")
            raise
    
    def _extract_column_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract detailed column information.
        
        Args:
            df: DataFrame
            
        Returns:
            Dictionary with column metadata
        """
        columns_info = {}
        
        for col in df.columns:
            col_data = df[col]
            
            columns_info[col] = {
                "dtype": str(col_data.dtype),
                "null_count": int(col_data.isnull().sum()),
                "null_percentage": round(col_data.isnull().sum() / len(df) * 100, 2) if len(df) > 0 else 0,
                "unique_count": int(col_data.nunique()),
                "is_numeric": pd.api.types.is_numeric_dtype(col_data),
                "is_datetime": pd.api.types.is_datetime64_any_dtype(col_data),
            }
            
            # Add statistics for numeric columns
            if columns_info[col]["is_numeric"]:
                columns_info[col].update({
                    "min": float(col_data.min()) if not pd.isna(col_data.min()) else None,
                    "max": float(col_data.max()) if not pd.isna(col_data.max()) else None,
                    "mean": float(col_data.mean()) if not pd.isna(col_data.mean()) else None,
                    "median": float(col_data.median()) if not pd.isna(col_data.median()) else None,
                    "std": float(col_data.std()) if not pd.isna(col_data.std()) else None,
                })
            
            # Add top values for categorical columns
            if not columns_info[col]["is_numeric"] and columns_info[col]["unique_count"] < 50:
                top_values = col_data.value_counts().head(5).to_dict()
                columns_info[col]["top_values"] = {str(k): int(v) for k, v in top_values.items()}
        
        return columns_info
    
    # ============================================================
    # CRUD OPERATIONS
    # ============================================================
    
    def get_dataset(
        self,
        dataset_id: int,
        user: Optional[User] = None,
        include_details: bool = False
    ) -> Dataset:
        """
        Get dataset by ID with access control.
        
        Args:
            dataset_id: Dataset ID
            user: Current user (for access control)
            include_details: Include relationships
            
        Returns:
            Dataset instance
            
        Raises:
            HTTPException: If not found or access denied
        """
        query = select(Dataset).where(Dataset.id == dataset_id)
        
        result = self.db.execute(query)
        dataset = result.scalar_one_or_none()
        
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found",
            )
        
        # Check access permissions
        if not self._can_access_dataset(dataset, user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this dataset",
            )
        
        # Increment view count
        if hasattr(dataset, 'increment_view_count'):
            dataset.increment_view_count()
            self.db.commit()
        
        return dataset
    
    def get_user_datasets(
        self,
        user_id: int,
        skip: int = 0,
        limit: int = 20,
        status_filter: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Tuple[List[Dataset], int]:
        """
        Get paginated list of user's datasets.
        
        Args:
            user_id: User ID
            skip: Offset for pagination
            limit: Number of items per page
            status_filter: Filter by status
            search: Search query for name/description
            
        Returns:
            Tuple of (datasets list, total count)
        """
        # Build query
        query = select(Dataset).where(
            and_(
                Dataset.owner_id == user_id,
                Dataset.is_deleted == False,
            )
        )
        
        # Apply filters
        if status_filter:
            query = query.where(Dataset.status == status_filter)
        
        if search:
            search_term = f"%{search}%"
            query = query.where(
                or_(
                    Dataset.name.ilike(search_term),
                    Dataset.description.ilike(search_term),
                )
            )
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total = self.db.execute(count_query).scalar_one()
        
        # Apply pagination and ordering
        query = query.order_by(Dataset.created_at.desc()).offset(skip).limit(limit)
        
        result = self.db.execute(query)
        datasets = result.scalars().all()
        
        return list(datasets), total
    
    def update_dataset(
        self,
        dataset_id: int,
        user: User,
        update_data: DatasetUpdate
    ) -> Dataset:
        """
        Update dataset metadata.
        
        Args:
            dataset_id: Dataset ID
            user: Current user
            update_data: Update data
            
        Returns:
            Updated dataset
            
        Raises:
            HTTPException: If not found or access denied
        """
        dataset = self.get_dataset(dataset_id, user)
        
        # Check ownership
        if dataset.owner_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only dataset owner can update it",
            )
        
        # Update fields
        update_dict = update_data.model_dump(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(dataset, field, value)
        
        self.db.commit()
        self.db.refresh(dataset)
        
        return dataset
    
    def delete_dataset(
        self,
        dataset_id: int,
        user: User,
        hard_delete: bool = False
    ) -> None:
        """
        Delete dataset (soft or hard delete).
        
        Args:
            dataset_id: Dataset ID
            user: Current user
            hard_delete: Permanently delete file and record
            
        Raises:
            HTTPException: If not found or access denied
        """
        dataset = self.get_dataset(dataset_id, user)
        
        # Check ownership
        if dataset.owner_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only dataset owner can delete it",
            )
        
        if hard_delete:
            # Delete physical file
            try:
                if os.path.exists(dataset.file_path):
                    os.remove(dataset.file_path)
                    logger.info(f"Deleted file: {dataset.file_path}")
            except Exception as e:
                logger.error(f"Failed to delete file: {str(e)}")
            
            # Update user statistics
            user.datasets_count = max(0, user.datasets_count - 1)
            user.storage_used_bytes = max(0, user.storage_used_bytes - dataset.file_size_bytes)
            
            # Delete database record
            self.db.delete(dataset)
            logger.info(f"Hard deleted dataset {dataset_id}")
        else:
            # Soft delete
            dataset.is_deleted = True
            logger.info(f"Soft deleted dataset {dataset_id}")
        
        self.db.commit()
    
    # ============================================================
    # HELPER METHODS
    # ============================================================
    
    def _can_access_dataset(
        self,
        dataset: Dataset,
        user: Optional[User]
    ) -> bool:
        """
        Check if user can access dataset.
        
        Args:
            dataset: Dataset instance
            user: Current user (None for anonymous)
            
        Returns:
            True if user can access, False otherwise
        """
        # Public datasets are accessible to everyone
        if getattr(dataset, 'is_public', False):
            return True
        
        # No user means anonymous access
        if not user:
            return False
        
        # Owner always has access
        if dataset.owner_id == user.id:
            return True
        
        # Admins have access to all
        from app.models.user import UserRole
        if hasattr(user, 'role') and user.role == UserRole.ADMIN:
            return True
        
        return False


# ============================================================
# DEPENDENCY INJECTION HELPER
# ============================================================

def get_dataset_service(db: Session = Depends(get_db)) -> DatasetService:
    """
    Dependency for injecting DatasetService.
    
    Args:
        db: Database session
        
    Returns:
        DatasetService instance
    """
    return DatasetService(db)
