"""
Dataset Endpoints.

Handles dataset upload, management, listing, and basic operations.
Provides CRUD operations for user datasets with access control.
"""

import logging
from typing import Any, Optional
from pathlib import Path
from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    UploadFile,
    File,
    Form,
    Query,
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.models.dataset import Dataset, DatasetStatus
from app.schemas.dataset import (
    DatasetResponse,
    DatasetDetail,
    DatasetList,
    DatasetUpdate,
    DatasetUpload,
)
from app.schemas.response import MessageResponse, SuccessResponse
from app.services.dataset_service import DatasetService, get_dataset_service
from app.core.deps import get_current_verified_user, get_current_user
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
# DATASET UPLOAD
# ============================================================

@router.post(
    "/upload",
    response_model=SuccessResponse[DatasetResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Upload Dataset",
    description="Upload a new dataset file (CSV, Excel, JSON, or Parquet)."
)
async def upload_dataset(
    file: UploadFile = File(..., description="Dataset file to upload"),
    name: str = Form(..., description="Dataset name", min_length=1, max_length=255),
    description: Optional[str] = Form(None, description="Dataset description", max_length=2000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> Any:
    """
    Upload a new dataset file.
    
    **Supported Formats:**
    - CSV (.csv)
    - Excel (.xlsx, .xls)
    - JSON (.json)
    - Parquet (.parquet)
    
    **File Size Limits:**
    - Free tier: 10 MB
    - Premium tier: 500 MB
    
    **Process:**
    1. Validates file type and size
    2. Checks user's upload limits
    3. Saves file securely
    4. Extracts metadata
    5. Triggers background processing
    
    **Returns:**
    - Dataset object with initial metadata
    - Processing status (will be updated async)
    
    **Errors:**
    - 400: Invalid file type or size
    - 403: User quota exceeded
    - 413: File too large
    - 422: Validation error
    """
    try:
        logger.info(f"Dataset upload initiated by user {current_user.id}: {name}")
        
        # ✅ FIX: Check subscription expiry with timezone-aware datetime
        if current_user.subscription_expires_at:
            now = datetime.now(timezone.utc)
            expiry = current_user.subscription_expires_at
            
            # Convert naive datetime to aware if needed
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            
            if expiry < now:
                logger.warning(f"Upload rejected: subscription expired for user {current_user.id}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Subscription expired. Please renew your subscription."
                )
        
        # Validate file extension
        allowed_extensions = {'.csv', '.xlsx', '.xls', '.json', '.parquet'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type {file_ext} not supported. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Read file content
        content = await file.read()
        
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is empty"
            )
        
        # Create upload metadata
        metadata = DatasetUpload(
            name=name,
            description=description,
        )
        
        # Upload dataset via service
        dataset = await dataset_service.upload_dataset(
            file=file,
            user=current_user,
            metadata=metadata
        )
        
        logger.info(f"Dataset uploaded successfully: {dataset.id}")
        
        return SuccessResponse(
            success=True,
            message="Dataset uploaded successfully. Processing will begin shortly.",
            data=DatasetResponse.model_validate(dataset)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset upload failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


# ============================================================
# LIST DATASETS
# ============================================================

@router.get(
    "",
    response_model=DatasetList,
    summary="List User Datasets",
    description="Get paginated list of user's datasets with filtering."
)
async def list_datasets(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(20, ge=1, le=100, description="Number of records to return"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    search: Optional[str] = Query(None, description="Search in name/description"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> Any:
    """
    List user's datasets with pagination and filtering.
    
    **Query Parameters:**
    - skip: Pagination offset (default: 0)
    - limit: Results per page (default: 20, max: 100)
    - status_filter: Filter by processing status
    - search: Search in dataset name or description
    
    **Returns:**
    - Paginated list of datasets
    - Total count
    - Page information
    
    **Status Values:**
    - uploaded: File uploaded, not processed
    - processing: Currently processing
    - analyzing: Running EDA
    - cleaning: Data cleaning in progress
    - completed: Ready for use
    - failed: Processing failed
    """
    try:
        logger.info(f"Listing datasets for user {current_user.id}")
        
        # Get datasets
        datasets, total = dataset_service.get_user_datasets(
            user_id=current_user.id,
            skip=skip,
            limit=limit,
            status_filter=status_filter,
            search=search,
        )
        
        # Calculate pagination
        pages = (total + limit - 1) // limit  # Ceiling division
        
        return DatasetList(
            items=[DatasetResponse.model_validate(ds) for ds in datasets],
            total=total,
            page=skip // limit + 1,
            size=limit,
            pages=pages,
        )
    
    except Exception as e:
        logger.error(f"Failed to list datasets: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve datasets"
        )


# ============================================================
# GET DATASET DETAILS
# ============================================================

@router.get(
    "/{dataset_id}",
    response_model=DatasetDetail,
    summary="Get Dataset Details",
    description="Get detailed information about a specific dataset."
)
async def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> Any:
    """
    Get detailed information about a dataset.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Returns:**
    - Complete dataset information
    - Processing status and metrics
    - Column information
    - Data quality scores
    - Statistics (if available)
    
    **Errors:**
    - 404: Dataset not found
    - 403: Access denied
    """
    try:
        logger.info(f"Fetching dataset {dataset_id} for user {current_user.id}")
        
        # Get dataset with details
        dataset = dataset_service.get_dataset(
            dataset_id=dataset_id,
            user=current_user,
            include_details=True
        )
        
        return DatasetDetail.model_validate(dataset)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dataset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dataset"
        )


# ============================================================
# UPDATE DATASET
# ============================================================

@router.patch(
    "/{dataset_id}",
    response_model=SuccessResponse[DatasetResponse],
    summary="Update Dataset",
    description="Update dataset metadata (name, description, tags, visibility)."
)
async def update_dataset(
    dataset_id: int,
    update_data: DatasetUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> Any:
    """
    Update dataset metadata.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Updatable Fields:**
    - name: Dataset name
    - description: Dataset description
    - tags: List of tags for categorization
    - is_public: Public visibility (premium feature)
    
    **Returns:**
    - Updated dataset object
    
    **Errors:**
    - 404: Dataset not found
    - 403: Not owner or access denied
    - 422: Validation error
    """
    try:
        logger.info(f"Updating dataset {dataset_id} for user {current_user.id}")
        
        # Update dataset
        dataset = dataset_service.update_dataset(
            dataset_id=dataset_id,
            user=current_user,
            update_data=update_data
        )
        
        logger.info(f"Dataset {dataset_id} updated successfully")
        
        return SuccessResponse(
            success=True,
            message="Dataset updated successfully",
            data=DatasetResponse.model_validate(dataset)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update dataset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update dataset"
        )


# ============================================================
# DELETE DATASET
# ============================================================

@router.delete(
    "/{dataset_id}",
    response_model=MessageResponse,
    summary="Delete Dataset",
    description="Delete a dataset and its associated files."
)
async def delete_dataset(
    dataset_id: int,
    hard_delete: bool = Query(False, description="Permanently delete (cannot be recovered)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> Any:
    """
    Delete a dataset.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Query Parameters:**
    - hard_delete: If true, permanently deletes file and record
                   If false, soft delete (can be restored)
    
    **Note:**
    - Soft delete: Dataset marked as deleted but data retained
    - Hard delete: File and database record permanently removed
    
    **Returns:**
    - Success message
    
    **Errors:**
    - 404: Dataset not found
    - 403: Not owner or access denied
    """
    try:
        logger.info(
            f"Deleting dataset {dataset_id} for user {current_user.id} "
            f"(hard_delete={hard_delete})"
        )
        
        # Delete dataset
        dataset_service.delete_dataset(
            dataset_id=dataset_id,
            user=current_user,
            hard_delete=hard_delete
        )
        
        delete_type = "permanently deleted" if hard_delete else "deleted"
        logger.info(f"Dataset {dataset_id} {delete_type}")
        
        return MessageResponse(
            message=f"Dataset {delete_type} successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete dataset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete dataset"
        )


# ============================================================
# DOWNLOAD DATASET
# ============================================================

@router.get(
    "/{dataset_id}/download",
    summary="Download Dataset",
    description="Download the dataset file."
)
async def download_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> Any:
    """
    Download dataset file.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Returns:**
    - File download response
    
    **Note:**
    - Download count is tracked
    - Original file format is preserved
    
    **Errors:**
    - 404: Dataset not found
    - 403: Access denied
    """
    try:
        logger.info(f"Download requested for dataset {dataset_id}")
        
        # Get dataset
        dataset = dataset_service.get_dataset(
            dataset_id=dataset_id,
            user=current_user
        )
        
        # Check if file exists
        if not Path(dataset.file_path).exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset file not found"
            )
        
        # Increment download count
        dataset.increment_download_count()
        db.commit()
        
        logger.info(f"Dataset {dataset_id} downloaded by user {current_user.id}")
        
        # Return file
        return FileResponse(
            path=dataset.file_path,
            filename=dataset.file_name,
            media_type='application/octet-stream'
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download dataset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download dataset"
        )


# ============================================================
# DATASET STATISTICS
# ============================================================

@router.get(
    "/{dataset_id}/stats",
    response_model=dict,
    summary="Get Dataset Statistics",
    description="Get basic statistics about the dataset."
)
async def get_dataset_stats(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> Any:
    """
    Get dataset statistics.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Returns:**
    - Row and column counts
    - File size
    - Data quality metrics
    - Processing status
    
    **Errors:**
    - 404: Dataset not found
    - 403: Access denied
    """
    try:
        dataset = dataset_service.get_dataset(
            dataset_id=dataset_id,
            user=current_user
        )
        
        # Format file size
        def format_bytes(bytes_size):
            for unit in ['B', 'KB', 'MB', 'GB']:
                if bytes_size < 1024.0:
                    return f"{bytes_size:.2f} {unit}"
                bytes_size /= 1024.0
            return f"{bytes_size:.2f} TB"
        
        return {
            "dataset_id": dataset.id,
            "name": dataset.name,
            "status": dataset.status.value if hasattr(dataset.status, 'value') else str(dataset.status),
            "row_count": dataset.row_count or 0,
            "column_count": dataset.column_count or 0,
            "file_size": format_bytes(dataset.file_size_bytes or 0),
            "file_size_bytes": dataset.file_size_bytes or 0,
            "missing_values": getattr(dataset, 'missing_values_count', 0) or 0,
            "duplicate_rows": getattr(dataset, 'duplicate_rows_count', 0) or 0,
            "data_quality_score": getattr(dataset, 'data_quality_score', 0) or 0,
            "view_count": getattr(dataset, 'view_count', 0) or 0,
            "download_count": getattr(dataset, 'download_count', 0) or 0,
            "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
            "updated_at": dataset.updated_at.isoformat() if dataset.updated_at else None,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dataset stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )
