"""Exploratory Data Analysis (EDA) Endpoints.

Production-grade implementation with full feature support including:
- Comprehensive statistical analysis
- Correlation and distribution analysis
- Interactive HTML report generation
- Data quality assessment
- Performance optimization
- Async processing for large datasets
- Enterprise-level error handling and logging
"""

import logging
import asyncio
import os
from typing import Any, Optional, Dict
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.models.dataset import Dataset, DatasetStatus
from app.schemas.dataset import DatasetEDAConfig, DatasetStatisticsResponse
from app.schemas.response import SuccessResponse, MessageResponse
from app.services.eda_service import EDAService, get_eda_service
from app.services.dataset_service import DatasetService, get_dataset_service
from app.core.deps import get_current_verified_user
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _verify_dataset_access(
    dataset_id: int,
    current_user: User,
    dataset_service: DatasetService
) -> Dataset:
    """Verify user has access to dataset."""
    dataset = dataset_service.get_dataset(
        dataset_id=dataset_id,
        user=current_user
    )
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
    
    return dataset


def _check_dataset_ready(dataset: Dataset) -> None:
    """Check if dataset is ready for analysis."""
    if not dataset.is_ready():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset not ready for analysis. Status: {dataset.status}"
        )


# ============================================================
# 1. GENERATE EDA REPORT (ASYNC)
# ============================================================

@router.post(
    "/{dataset_id}/generate",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Generate EDA Report",
    description="Generate comprehensive exploratory data analysis report with all statistics and visualizations.",
    status_code=202
)
async def generate_eda_report(
    dataset_id: int,
    config: Optional[DatasetEDAConfig] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    eda_service: EDAService = Depends(get_eda_service),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> SuccessResponse[Dict[str, Any]]:
    """
    Generate comprehensive EDA report (asynchronous).
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Request Body (Optional):**
    ```json
    {
        "minimal_report": false,
        "sample_size": null,
        "generate_correlations": true,
        "generate_distributions": true
    }
    ```
    
    **Generated Report Includes:**
    - Overview statistics (rows, columns, memory usage, missing/duplicates)
    - Numerical column statistics (mean, std, quartiles, outliers, skewness, kurtosis)
    - Categorical column statistics (unique values, top values, entropy)
    - Correlation matrices (Pearson and Spearman)
    - Distribution analysis (normality tests, distribution types)
    - Data quality assessment (completeness, uniqueness, validity scores)
    - Interactive visualizations and recommendations
    - HTML report file
    
    **Configuration Options:**
    - minimal_report: Set to true for faster processing (~30 sec)
    - sample_size: Limit rows for analysis (for memory efficiency)
    - generate_correlations: Set to false to skip correlation analysis
    - generate_distributions: Set to false to skip distribution analysis
    
    **Response (202 Accepted):**
    ```json
    {
        "success": true,
        "message": "EDA report generated successfully",
        "data": {
            "dataset_id": 11,
            "status": "completed",
            "config_used": {...},
            "report_url": "path/to/report.html",
            "statistics": {...},
            "generated_at": "2025-11-05T20:00:00Z",
            "recommendations": [...]
        }
    }
    ```
    
    **Performance:**
    - Small datasets (<1MB): ~10-30 seconds
    - Medium datasets (1-50MB): ~30-120 seconds
    - Large datasets (>50MB): 2-5 minutes (use minimal_report=true)
    
    **Errors:**
    - 404: Dataset not found
    - 403: Access denied
    - 400: Dataset not ready
    - 409: Report already being generated
    - 500: Generation failed
    """
    try:
        logger.info(
            f"🔄 EDA report generation requested for dataset {dataset_id} "
            f"by user {current_user.id}"
        )
        
        # Set default config
        if config is None:
            config = DatasetEDAConfig()
        
        # Verify dataset access
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        _check_dataset_ready(dataset)
        
        # Check if analysis already in progress
        if dataset.status == DatasetStatus.ANALYZING:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="EDA analysis already in progress for this dataset"
            )
        
        # Generate report (async)
        results = await eda_service.generate_eda_report(
            dataset_id=dataset_id,
            config=config.model_dump()
        )
        
        logger.info(f"✅ EDA report generated successfully for dataset {dataset_id}")
        
        return SuccessResponse(
            success=True,
            message="EDA report generated successfully",
            data=results
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"❌ EDA generation failed for dataset {dataset_id}: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"EDA generation failed: {str(e)}"
        )


# ============================================================
# 2. GET STATISTICS
# ============================================================

@router.get(
    "/{dataset_id}/statistics",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Get Dataset Statistics",
    description="Get detailed statistical analysis (numerical and categorical)."
)
async def get_statistics(
    dataset_id: int,
    include_correlations: bool = Query(False, description="Include correlation matrices"),
    include_distributions: bool = Query(False, description="Include distribution analysis"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    eda_service: EDAService = Depends(get_eda_service),
) -> SuccessResponse[Dict[str, Any]]:
    """
    Get detailed statistics for a dataset.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Query Parameters:**
    - include_correlations: Include correlation matrices (default: false)
    - include_distributions: Include distribution analysis (default: false)
    
    **Returns:**
    ```json
    {
        "success": true,
        "message": "Statistics retrieved successfully",
        "data": {
            "dataset_id": 11,
            "dataset_name": "sales_data.csv",
            "status": "success",
            "overview": {
                "total_rows": 29469,
                "total_columns": 9,
                "memory_usage_mb": 4.5,
                "missing_percentage": 0.56,
                "duplicate_percentage": 0.08,
                "numerical_columns": 6,
                "categorical_columns": 3
            },
            "numerical": {
                "age": {
                    "count": 29469,
                    "missing": 150,
                    "mean": 45.67,
                    "std": 12.34,
                    "min": 0,
                    "max": 100,
                    "median": 45,
                    "q1": 35,
                    "q3": 56,
                    "iqr": 21,
                    "skewness": 0.45,
                    "kurtosis": -0.23,
                    "outliers": 234,
                    "outliers_percentage": 0.79
                }
            },
            "categorical": {
                "category": {
                    "count": 29469,
                    "missing": 10,
                    "unique": 15,
                    "top": "Category A",
                    "top_frequency": 8945,
                    "entropy": 3.45
                }
            }
        }
    }
    ```
    
    **Statistics Include:**
    - Numerical: mean, median, std, min, max, quartiles, outliers, skewness, kurtosis
    - Categorical: unique count, top value, entropy
    - Overview: rows, columns, memory, missing, duplicates
    
    **Errors:**
    - 404: Dataset not found
    - 403: Access denied
    - 400: Dataset not ready
    - 500: Analysis failed
    """
    try:
        logger.info(f"📊 Statistics requested for dataset {dataset_id}")
        
        # Verify access
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        _check_dataset_ready(dataset)
        
        # Load data and generate statistics
        df = await asyncio.to_thread(
            eda_service._read_dataframe,
            dataset.file_path,
            dataset.file_type
        )
        
        statistics = await asyncio.to_thread(
            eda_service._generate_statistics,
            df,
            include_correlations,
            include_distributions
        )
        
        logger.info(f"✅ Statistics generated for dataset {dataset_id}")
        
        return SuccessResponse(
            success=True,
            message="Statistics retrieved successfully",
            data={
                "dataset_id": dataset_id,
                "dataset_name": dataset.name,
                "status": "success",
                **statistics
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Statistics retrieval failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )


# ============================================================
# 3. GET CORRELATIONS
# ============================================================

@router.get(
    "/{dataset_id}/correlations",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Get Correlation Analysis",
    description="Get correlation matrices and strongly correlated features."
)
async def get_correlations(
    dataset_id: int,
    method: str = Query("pearson", description="Correlation method (pearson or spearman)"),
    threshold: float = Query(0.7, ge=0, le=1, description="Strong correlation threshold (0-1)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    eda_service: EDAService = Depends(get_eda_service),
) -> SuccessResponse[Dict[str, Any]]:
    """
    Get correlation analysis for numerical features.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Query Parameters:**
    - method: Correlation method (pearson or spearman, default: pearson)
    - threshold: Minimum |correlation| for strong flag (0-1, default: 0.7)
    
    **Returns:**
    ```json
    {
        "success": true,
        "message": "Correlations retrieved successfully",
        "data": {
            "dataset_id": 11,
            "dataset_name": "sales_data.csv",
            "status": "success",
            "method": "pearson",
            "threshold": 0.7,
            "numerical_columns_count": 8,
            "pearson": {"age": {"salary": 0.85}},
            "spearman": {"age": {"salary": 0.82}},
            "strong_correlations": [
                {
                    "column1": "age",
                    "column2": "salary",
                    "correlation": 0.85,
                    "strength": "strong positive"
                }
            ],
            "strong_correlations_count": 12
        }
    }
    ```
    
    **Use Cases:**
    - Feature engineering (identify redundant features)
    - Multicollinearity detection
    - Understanding feature relationships
    - Dimensionality reduction planning
    
    **Errors:**
    - 404: Dataset not found
    - 403: Access denied
    - 400: Dataset not ready or <2 numerical columns
    - 500: Analysis failed
    """
    try:
        logger.info(f"🔗 Correlations requested for dataset {dataset_id}")
        
        # Verify access
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        _check_dataset_ready(dataset)
        
        # Load data
        df = await asyncio.to_thread(
            eda_service._read_dataframe,
            dataset.file_path,
            dataset.file_type
        )
        
        # Check numerical columns
        numerical_cols = df.select_dtypes(include=['number']).columns
        if len(numerical_cols) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dataset must have at least 2 numerical columns for correlation analysis"
            )
        
        # Calculate correlations
        correlations = await asyncio.to_thread(
            eda_service._calculate_correlations,
            df
        )
        
        # Filter by threshold
        if "strong_correlations" in correlations:
            correlations["strong_correlations"] = [
                corr for corr in correlations["strong_correlations"]
                if abs(corr["correlation"]) >= threshold
            ]
            correlations["strong_correlations_count"] = len(
                correlations["strong_correlations"]
            )
        
        logger.info(f"✅ Correlations calculated for dataset {dataset_id}")
        
        return SuccessResponse(
            success=True,
            message="Correlations retrieved successfully",
            data={
                "dataset_id": dataset_id,
                "dataset_name": dataset.name,
                "status": "success",
                "method": method,
                "threshold": threshold,
                "numerical_columns_count": len(numerical_cols),
                **correlations
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Correlation analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve correlations: {str(e)}"
        )


# ============================================================
# 4. GET DISTRIBUTIONS
# ============================================================

@router.get(
    "/{dataset_id}/distributions",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Get Distribution Analysis",
    description="Get distribution analysis for numerical features."
)
async def get_distributions(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    eda_service: EDAService = Depends(get_eda_service),
) -> SuccessResponse[Dict[str, Any]]:
    """
    Get distribution analysis for numerical columns.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Returns:**
    ```json
    {
        "success": true,
        "message": "Distributions retrieved successfully",
        "data": {
            "dataset_id": 11,
            "dataset_name": "sales_data.csv",
            "status": "success",
            "total_numerical_columns": 6,
            "distributions": {
                "age": {
                    "distribution_type": "right_skewed",
                    "normality_test": {
                        "test": "shapiro-wilk",
                        "statistic": 0.945,
                        "p_value": 0.0001,
                        "is_normal": false,
                        "confidence": "95%"
                    }
                }
            },
            "distribution_summary": {
                "normal_columns": 3,
                "right_skewed": 2,
                "left_skewed": 0,
                "heavy_tailed": 1,
                "light_tailed": 0,
                "unknown": 0
            }
        }
    }
    ```
    
    **Distribution Types:**
    - normal: Approximately normally distributed
    - right_skewed: Tail extends to the right
    - left_skewed: Tail extends to the left
    - heavy_tailed: More outliers than normal
    - light_tailed: Fewer outliers than normal
    - unknown: Cannot determine
    
    **Normality Test (Shapiro-Wilk):**
    - p_value > 0.05: Normally distributed
    - p_value ≤ 0.05: NOT normally distributed
    - is_normal: Boolean at 95% confidence
    
    **Errors:**
    - 404: Dataset not found
    - 403: Access denied
    - 400: Dataset not ready or no numerical columns
    - 500: Analysis failed
    """
    try:
        logger.info(f"📈 Distributions requested for dataset {dataset_id}")
        
        # Verify access
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        _check_dataset_ready(dataset)
        
        # Load data
        df = await asyncio.to_thread(
            eda_service._read_dataframe,
            dataset.file_path,
            dataset.file_type
        )
        
        # Analyze distributions
        distributions = await asyncio.to_thread(
            eda_service._analyze_distributions,
            df
        )
        
        if not distributions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No numerical columns found for distribution analysis"
            )
        
        # Generate summary
        distribution_summary = {
            "normal_columns": 0,
            "right_skewed": 0,
            "left_skewed": 0,
            "heavy_tailed": 0,
            "light_tailed": 0,
            "unknown": 0,
        }
        
        for col, dist_info in distributions.items():
            dist_type = dist_info.get("distribution_type", "unknown")
            if dist_type in distribution_summary:
                distribution_summary[dist_type] += 1
        
        logger.info(f"✅ Distributions analyzed for dataset {dataset_id}")
        
        return SuccessResponse(
            success=True,
            message="Distributions retrieved successfully",
            data={
                "dataset_id": dataset_id,
                "dataset_name": dataset.name,
                "status": "success",
                "distributions": distributions,
                "distribution_summary": distribution_summary,
                "total_numerical_columns": len(distributions)
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Distribution analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve distributions: {str(e)}"
        )


# ============================================================
# 5. GET COMPREHENSIVE SUMMARY
# ============================================================

@router.get(
    "/{dataset_id}/summary",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Get EDA Summary",
    description="Get comprehensive EDA summary combining all analyses."
)
async def get_eda_summary(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    eda_service: EDAService = Depends(get_eda_service),
) -> SuccessResponse[Dict[str, Any]]:
    """
    Get complete EDA summary with all analyses combined.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Returns Comprehensive Report:**
    ```json
    {
        "success": true,
        "message": "EDA summary retrieved successfully",
        "data": {
            "dataset_id": 11,
            "dataset_name": "sales_data.csv",
            "status": "success",
            "overview": {...},
            "numerical": {...},
            "categorical": {...},
            "correlations": {...},
            "distributions": {...},
            "data_quality": {
                "overall_score": 92.5,
                "completeness": 98.7,
                "uniqueness": 99.2,
                "validity": 82.3
            },
            "recommendations": [
                "Excellent data quality (92.5%)",
                "Strong correlation: age & salary (0.85)"
            ],
            "report_url": "http://...",
            "generated_at": "2025-11-05T20:00:00Z",
            "analysis_duration_seconds": 45.2
        }
    }
    ```
    
    **Data Quality Score (0-100):**
    - 90-100: Excellent (ready for analysis)
    - 70-89: Good (minor issues)
    - 50-69: Fair (moderate issues)
    - <50: Poor (needs cleaning)
    
    **Errors:**
    - 404: Dataset not found
    - 403: Access denied
    - 400: Dataset not ready
    - 500: Analysis failed
    """
    try:
        logger.info(f"📋 EDA summary requested for dataset {dataset_id}")
        
        # Verify access
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        _check_dataset_ready(dataset)
        
        start_time = datetime.now()
        
        # Load data
        df = await asyncio.to_thread(
            eda_service._read_dataframe,
            dataset.file_path,
            dataset.file_type
        )
        
        # Generate full statistics
        statistics = await asyncio.to_thread(
            eda_service._generate_statistics,
            df,
            generate_correlations=True,
            generate_distributions=True
        )
        
        # Generate recommendations
        recommendations = await asyncio.to_thread(
            eda_service._generate_recommendations,
            statistics
        )
        
        # Calculate duration
        analysis_duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(
            f"✅ EDA summary generated for dataset {dataset_id} "
            f"in {analysis_duration:.2f} seconds"
        )
        
        return SuccessResponse(
            success=True,
            message="EDA summary retrieved successfully",
            data={
                "dataset_id": dataset_id,
                "dataset_name": dataset.name,
                "status": "success",
                "overview": statistics.get("overview"),
                "numerical": statistics.get("numerical"),
                "categorical": statistics.get("categorical"),
                "correlations": statistics.get("correlations"),
                "distributions": statistics.get("distributions"),
                "data_quality": statistics.get("data_quality"),
                "recommendations": recommendations,
                "report_url": getattr(dataset, "eda_report_url", None),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "analysis_duration_seconds": round(analysis_duration, 2),
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ EDA summary retrieval failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve summary: {str(e)}"
        )


# ============================================================
# 6. GET EDA REPORT FILE
# ============================================================

@router.get(
    "/{dataset_id}/report",
    summary="Download EDA Report",
    description="Download the generated HTML EDA report file."
)
async def get_eda_report(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> FileResponse:
    """
    Download the generated EDA report in HTML format.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Response:**
    - HTML file with interactive visualizations
    - Includes all statistical analyses
    - Can be opened in any browser
    
    **Prerequisites:**
    - Report must be generated first using POST /generate
    
    **File Format:**
    - Type: HTML5
    - Interactive: Yes (ydata-profiling)
    - Size: Typically 5-20 MB
    
    **Use Cases:**
    - Share analysis with stakeholders
    - Archive for documentation
    - Offline analysis
    
    **Errors:**
    - 404: Dataset or report not found
    - 403: Access denied
    - 400: Report not yet generated
    - 500: File retrieval failed
    """
    try:
        logger.info(f"📥 EDA report download requested for dataset {dataset_id}")
        
        # Verify access
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        
        # Check if report exists
        report_url = getattr(dataset, "eda_report_url", None)
        if not report_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="EDA report not generated yet. Call POST /generate first."
            )
        
        # Verify file exists
        if not os.path.exists(report_url):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Report file not found"
            )
        
        logger.info(f"✅ Report file retrieved for dataset {dataset_id}")
        
        return FileResponse(
            path=report_url,
            filename=f"eda_report_{dataset_id}.html",
            media_type="text/html"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Report download failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve report"
        )


# ============================================================
# 7. REGENERATE STATISTICS
# ============================================================

@router.post(
    "/{dataset_id}/regenerate",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Regenerate Statistics",
    description="Force regeneration of EDA statistics and report."
)
async def regenerate_eda(
    dataset_id: int,
    config: Optional[DatasetEDAConfig] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    eda_service: EDAService = Depends(get_eda_service),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> SuccessResponse[Dict[str, Any]]:
    """
    Regenerate EDA statistics and report.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Request Body (Optional):**
    ```json
    {
        "minimal_report": false,
        "sample_size": null,
        "generate_correlations": true,
        "generate_distributions": true
    }
    ```
    
    **Use Cases:**
    - Dataset was updated or cleaned
    - Previous generation failed
    - Want fresh analysis with different config
    
    **Returns:**
    ```json
    {
        "success": true,
        "message": "EDA analysis regenerated successfully",
        "data": {...}
    }
    ```
    
    **Note:**
    - This will overwrite existing report and statistics
    - Use minimal_report=true for faster regeneration
    - Process runs asynchronously
    
    **Errors:**
    - 404: Dataset not found
    - 403: Access denied
    - 400: Dataset not ready
    - 500: Regeneration failed
    """
    try:
        logger.info(f"🔄 EDA regeneration requested for dataset {dataset_id}")
        
        if config is None:
            config = DatasetEDAConfig()
        
        # Verify access
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        _check_dataset_ready(dataset)
        
        # Regenerate
        results = await eda_service.generate_eda_report(
            dataset_id=dataset_id,
            config=config.model_dump()
        )
        
        logger.info(f"✅ EDA regeneration completed for dataset {dataset_id}")
        
        return SuccessResponse(
            success=True,
            message="EDA analysis regenerated successfully",
            data=results
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ EDA regeneration failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to regenerate EDA"
        )


# ============================================================
# 8. GET EDA STATUS
# ============================================================

@router.get(
    "/{dataset_id}/status",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Get EDA Status",
    description="Check if EDA analysis is ready or currently processing."
)
async def get_eda_status(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> SuccessResponse[Dict[str, Any]]:
    """
    Check current status of EDA analysis for dataset.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Returns:**
    ```json
    {
        "success": true,
        "message": "EDA status retrieved",
        "data": {
            "dataset_id": 11,
            "status": "completed",
            "is_ready": true,
            "generated_at": "2025-11-05T20:00:00Z",
            "report_available": true,
            "file_size_mb": 4.5
        }
    }
    ```
    
    **Status Values:**
    - completed: Analysis done, results available
    - processing: Analysis in progress
    - failed: Previous analysis failed
    - pending: No analysis started yet
    
    **Use Cases:**
    - Check if analysis is complete before fetching results
    - Monitor long-running analysis
    - Determine if report is available
    
    **Errors:**
    - 404: Dataset not found
    - 403: Access denied
    - 500: Status check failed
    """
    try:
        logger.info(f"✓ EDA status requested for dataset {dataset_id}")
        
        # Verify access
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        
        return SuccessResponse(
            success=True,
            message="EDA status retrieved",
            data={
                "dataset_id": dataset_id,
                "status": dataset.status.value if hasattr(dataset, 'status') else "unknown",
                "is_ready": dataset.is_ready(),
                "generated_at": (
                    dataset.eda_report_generated_at.isoformat()
                    if hasattr(dataset, 'eda_report_generated_at') and dataset.eda_report_generated_at
                    else None
                ),
                "report_available": bool(getattr(dataset, "eda_report_url", None)),
                "file_size_mb": round(dataset.file_size_bytes / 1024 / 1024, 2) if dataset.file_size_bytes else 0,
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Status check failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check EDA status"
        )
