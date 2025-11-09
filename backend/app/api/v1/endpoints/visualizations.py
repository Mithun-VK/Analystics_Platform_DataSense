"""
Visualization Endpoints - Enhanced.

Handles chart generation, interactive visualizations, data plotting, and exports.
Provides various chart types with customization options and multi-format export support.
"""

import logging
from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.schemas.response import SuccessResponse, MessageResponse
from app.services.visualization_service import (
    VisualizationService,
    get_visualization_service,
    ChartType,
)
from app.services.dataset_service import DatasetService, get_dataset_service
from app.core.deps import get_current_verified_user

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================
# GENERATE VISUALIZATION
# ============================================================

@router.post(
    "/{dataset_id}/generate",
    response_model=SuccessResponse[dict],
    summary="Generate Visualization",
    description="Generate a chart/visualization for a dataset with customizable options."
)
async def generate_visualization(
    dataset_id: int,
    chart_type: ChartType = Query(..., description="Type of chart to generate"),
    x_column: Optional[str] = Query(None, description="Column for X-axis"),
    y_column: Optional[str] = Query(None, description="Column for Y-axis"),
    color_column: Optional[str] = Query(None, description="Column for color grouping"),
    title: Optional[str] = Query(None, description="Chart title"),
    width: int = Query(800, ge=400, le=2000, description="Chart width in pixels"),
    height: int = Query(600, ge=300, le=1500, description="Chart height in pixels"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    viz_service: VisualizationService = Depends(get_visualization_service),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> Any:
    """
    Generate a visualization for dataset with full customization.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Query Parameters:**
    - chart_type: Type of chart (bar, line, scatter, pie, box, histogram, heatmap, violin, area, time_series)
    - x_column: Column name for X-axis
    - y_column: Column name for Y-axis (required for some chart types)
    - color_column: Column for color grouping (optional)
    - title: Custom chart title (optional)
    - width: Chart width in pixels (400-2000, default: 800)
    - height: Chart height in pixels (300-1500, default: 600)
    
    **Supported Chart Types:**
    
    1. **Bar Chart** - Compare values across categories
    2. **Line Chart** - Show trends over time/sequence
    3. **Scatter Plot** - Show relationships between variables
    4. **Pie Chart** - Show proportions/parts of a whole
    5. **Box Plot** - Show distribution and outliers
    6. **Histogram** - Show frequency distribution
    7. **Heatmap/Correlation** - Show correlations between numerical columns
    8. **Violin Plot** - Show distribution with quartiles
    9. **Area Chart** - Show cumulative trends
    10. **Time Series** - Show temporal trends
    
    **Returns:**
    ```
    {
      "success": true,
      "message": "scatter chart generated successfully",
      "data": {
        "dataset_id": 11,
        "title": "Age vs Salary",
        "chart_type": "scatter",
        "chart_url": "/path/to/chart.html",
        "config": {
          "x_column": "Age",
          "y_column": "Salary",
          "correlation": 0.85
        },
        "data_columns": ["Age", "Salary"]
      }
    }
    ```
    """
    try:
        logger.info(
            f"Generating {chart_type.value} visualization for dataset {dataset_id} "
            f"by user {current_user.id}"
        )
        
        # Verify access to dataset
        dataset = dataset_service.get_dataset(
            dataset_id=dataset_id,
            user=current_user
        )
        
        # Check if dataset is ready for visualization
        if not dataset.is_ready():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dataset not ready for visualization. Current status: {dataset.status}. "
                       f"Please wait for processing to complete."
            )
        
        # Generate visualization
        chart_data = await viz_service.generate_chart(
            dataset_id=dataset_id,
            chart_type=chart_type,
            x_column=x_column,
            y_column=y_column,
            color_column=color_column,
            title=title,
            width=width,
            height=height,
        )
        
        logger.info(f"Visualization generated successfully for dataset {dataset_id}")
        
        return SuccessResponse(
            success=True,
            message=f"{chart_type.value} chart generated successfully",
            data=chart_data
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visualization generation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate visualization: {str(e)}"
        )


# ============================================================
# GENERATE AUTOMATED VISUALIZATION SUITE
# ============================================================

@router.post(
    "/{dataset_id}/generate-suite",
    response_model=SuccessResponse[dict],
    summary="Generate Visualization Suite",
    description="Generate comprehensive set of visualizations automatically."
)
async def generate_visualization_suite(
    dataset_id: int,
    max_charts: int = Query(10, ge=1, le=20, description="Maximum number of charts"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    viz_service: VisualizationService = Depends(get_visualization_service),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> Any:
    """
    Generate automated comprehensive visualization suite.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Query Parameters:**
    - max_charts: Maximum number of charts (1-20, default: 10)
    
    **Automatic Charts Generated:**
    
    **Distribution Analysis:**
    - Histograms for top numerical columns
    - Box plots for outlier detection
    - Violin plots for distribution shape
    
    **Correlation Analysis:**
    - Correlation heatmap for all numerical columns
    - Scatter plots for highly correlated pairs (correlation > 0.5)
    
    **Categorical Analysis:**
    - Bar charts for categorical columns
    - Pie charts for proportions
    
    **Returns:**
    ```
    {
      "success": true,
      "message": "Generated 8 visualizations",
      "data": {
        "charts": [...],
        "total_generated": 8,
        "dataset_id": 11
      }
    }
    ```
    """
    try:
        logger.info(f"Generating visualization suite for dataset {dataset_id}")
        
        # Verify access
        dataset = dataset_service.get_dataset(
            dataset_id=dataset_id,
            user=current_user
        )
        
        # Check if dataset is ready
        if not dataset.is_ready():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dataset not ready for visualization"
            )
        
        # Generate suite
        charts = await viz_service.generate_automated_visualizations(
            dataset_id=dataset_id,
            max_charts=max_charts,
        )
        
        logger.info(f"Generated {len(charts)} charts for dataset {dataset_id}")
        
        return SuccessResponse(
            success=True,
            message=f"Generated {len(charts)} visualizations",
            data={
                "charts": charts,
                "total_generated": len(charts),
                "dataset_id": dataset_id,
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visualization suite generation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate visualization suite"
        )


# ============================================================
# EXPORT CHART - MULTI-FORMAT
# ============================================================

@router.post(
    "/{dataset_id}/export",
    summary="Export Chart",
    description="Export chart in multiple formats (PNG, SVG, PDF, HTML, JPEG)."
)
async def export_chart(
    dataset_id: int,
    chart_id: int = Query(..., description="Chart/Visualization ID to export"),
    format: str = Query("png", regex="^(png|svg|pdf|html|jpeg)$", description="Export format"),
    width: int = Query(1200, ge=400, le=4000, description="Export width in pixels"),
    height: int = Query(900, ge=300, le=3000, description="Export height in pixels"),
    scale: float = Query(2.0, ge=1.0, le=4.0, description="DPI scale for raster formats"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    viz_service: VisualizationService = Depends(get_visualization_service),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> FileResponse:
    """
    Export chart/visualization in multiple formats.
    
    **Supported Formats:**
    - **PNG** - Raster format with transparency (recommended for web/documents)
    - **JPEG** - Compressed format, smallest file size (web/email optimized)
    - **SVG** - Vector format, infinitely scalable (perfect for presentations)
    - **PDF** - Print-ready professional format (archiving and formal reports)
    - **HTML** - Fully interactive Plotly chart (web-ready)
    
    **Returns:** Binary file download
    """
    try:
        logger.info(
            f"Exporting chart {chart_id} from dataset {dataset_id} as {format.upper()} "
            f"({width}x{height}, scale={scale}) by user {current_user.id}"
        )
        
        # Verify dataset access
        dataset = dataset_service.get_dataset(
            dataset_id=dataset_id,
            user=current_user
        )
        
        # Export chart
        export_result = await viz_service.export_chart(
            dataset_id=dataset_id,
            chart_id=chart_id,
            export_format=format,
            width=width,
            height=height,
            scale=scale,
        )
        
        # Check if export was successful
        if not export_result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Export failed"
            )
        
        file_path = export_result.get("file_path")
        file_name = export_result.get("file_name")
        mime_type = export_result.get("mime_type")
        
        logger.info(f"Chart exported successfully: {file_path} ({file_name})")
        
        # Return file for download
        return FileResponse(
            path=file_path,
            filename=file_name,
            media_type=mime_type,
            headers={
                "Content-Disposition": f'attachment; filename="{file_name}"',
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "public, max-age=3600",
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chart export failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export chart: {str(e)}"
        )


# ============================================================
# GET AVAILABLE COLUMNS FOR VISUALIZATION
# ============================================================

@router.get(
    "/{dataset_id}/columns",
    response_model=dict,
    summary="Get Available Columns",
    description="Get list of columns available for visualization with metadata."
)
async def get_available_columns(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> Any:
    """
    Get columns available for visualization with type information.
    
    **Returns:**
    ```
    {
      "numerical": ["Age", "Salary", "Years_Experience"],
      "categorical": ["Department", "City", "Role"],
      "temporal": ["Hire_Date"],
      "boolean": ["Is_Active"],
      "metadata": {
        "Age": {
          "dtype": "int64",
          "min": 22,
          "max": 65,
          "unique_count": 43,
          "null_count": 2
        }
      }
    }
    ```
    """
    try:
        # Verify access
        dataset = dataset_service.get_dataset(
            dataset_id=dataset_id,
            user=current_user,
            include_details=True
        )
        
        columns_info = {
            "numerical": [],
            "categorical": [],
            "temporal": [],
            "boolean": [],
            "metadata": {}
        }
        
        # Parse column information
        if dataset.columns_info:
            for col_name, col_info in dataset.columns_info.items():
                dtype = col_info.get("dtype", "")
                
                if "int" in dtype or "float" in dtype:
                    columns_info["numerical"].append(col_name)
                elif "datetime" in dtype:
                    columns_info["temporal"].append(col_name)
                elif "bool" in dtype:
                    columns_info["boolean"].append(col_name)
                else:
                    columns_info["categorical"].append(col_name)
                
                columns_info["metadata"][col_name] = col_info
        
        logger.info(f"Retrieved column info for dataset {dataset_id}")
        
        return columns_info
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get columns: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve column information"
        )


# ============================================================
# GENERATE INTERACTIVE DASHBOARD - ENHANCED
# ============================================================

@router.get(
    "/{dataset_id}/dashboard",
    response_class=HTMLResponse,
    summary="Generate Interactive Dashboard",
    description="Generate interactive HTML dashboard with multiple visualizations, data quality metrics, and statistics."
)
async def generate_dashboard_endpoint(
    dataset_id: int,
    regenerate: bool = Query(False, description="Force regenerate all charts"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    viz_service: VisualizationService = Depends(get_visualization_service),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> HTMLResponse:
    """
    Generate interactive HTML dashboard with comprehensive visualizations.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Query Parameters:**
    - regenerate: Force regenerate all charts (default: false)
    
    **Returns:** Interactive HTML dashboard page
    
    **Dashboard Features:**
    
    **📊 Statistics Cards:**
    - Total Rows - Number of data records
    - Total Columns - Number of features (numerical, categorical breakdown)
    - Data Quality Score - Comprehensive quality percentage (0-100%)
    - Memory Size - Dataset memory footprint
    - Last Updated - Processing timestamp
    
    **📈 Data Quality Analysis:**
    - Completeness - Percentage of non-null values
    - Uniqueness - Duplicate detection rate
    - Validity - Data type consistency checks
    - Consistency - Outlier detection using IQR method
    - Visual quality progress bar with percentage
    
    **🎨 Interactive Visualizations:**
    - Automatically generated charts based on data structure
    - Histograms for numerical distributions
    - Correlation heatmaps for relationships
    - Scatter plots with trendlines
    - Bar charts for categorical analysis
    - Box plots for outlier detection
    - All charts support zoom, pan, and hover tooltips
    
    **💎 Design Features:**
    - Responsive grid layout (adapts to screen size)
    - Modern gradient backgrounds and animations
    - Hover effects on stat cards and charts
    - Professional color scheme
    - Mobile-friendly design
    - Print-optimized layout
    
    **⚡ Performance:**
    - Async chart generation for speed
    - Cached visualizations (use regenerate=true to refresh)
    - Lazy loading for large datasets
    - Optimized HTML rendering
    
    **Status Codes:**
    - 200: Dashboard generated successfully
    - 400: Dataset not ready for visualization
    - 404: Dataset not found
    - 401: Unauthorized (invalid token)
    - 500: Server error
    
    **Example Usage:**
    ```
    # Generate dashboard with cached charts
    GET /api/v1/visualizations/11/dashboard
    
    # Force regenerate all visualizations
    GET /api/v1/visualizations/11/dashboard?regenerate=true
    ```
    
    **Dashboard Quality Metrics:**
    - Completeness: 95.8% (percentage of non-null values)
    - Uniqueness: 100% (no duplicate rows detected)
    - Validity: 98.2% (data type consistency)
    - Consistency: 94.5% (outlier detection)
    - Overall Score: 96.1% (weighted average)
    
    **Sample Response:**
    Returns complete HTML page with:
    - Beautiful gradient header with dataset name
    - 5 statistics cards with key metrics
    - Animated quality progress bar
    - Grid of interactive Plotly charts
    - Professional footer with timestamp
    """
    try:
        logger.info(
            f"Generating dashboard for dataset {dataset_id} "
            f"(regenerate={regenerate}) by user {current_user.id}"
        )
        
        # Verify dataset access
        dataset = dataset_service.get_dataset(
            dataset_id=dataset_id,
            user=current_user
        )
        
        # Check if dataset is ready
        if not dataset.is_ready():
            error_html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Dataset Not Ready</title>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        margin: 0;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    }}
                    .message-box {{
                        background: white;
                        padding: 40px;
                        border-radius: 12px;
                        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                        text-align: center;
                        max-width: 500px;
                    }}
                    h2 {{ color: #2c3e50; margin-bottom: 15px; }}
                    p {{ color: #7f8c8d; line-height: 1.6; }}
                    .status {{ 
                        display: inline-block;
                        padding: 8px 16px;
                        background: #f39c12;
                        color: white;
                        border-radius: 6px;
                        font-weight: 600;
                        margin: 15px 0;
                    }}
                    .refresh-btn {{
                        display: inline-block;
                        margin-top: 20px;
                        padding: 12px 30px;
                        background: #3498db;
                        color: white;
                        text-decoration: none;
                        border-radius: 6px;
                        font-weight: 600;
                        transition: background 0.3s;
                    }}
                    .refresh-btn:hover {{ background: #2980b9; }}
                </style>
            </head>
            <body>
                <div class="message-box">
                    <h2>⏳ Dataset is Processing</h2>
                    <p>The dataset is still being analyzed and prepared for visualization.</p>
                    <div class="status">{dataset.status}</div>
                    <p>This usually takes a few moments. Please refresh the page or check back shortly.</p>
                    <a href="javascript:location.reload()" class="refresh-btn">🔄 Refresh Page</a>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=error_html, status_code=200)
        
        # Call the service method to generate complete dashboard
        dashboard_html = await viz_service.generate_dashboard(
            dataset_id=dataset_id,
            regenerate=regenerate
        )
        
        logger.info(f"Dashboard generated successfully for dataset {dataset_id}")
        
        return HTMLResponse(content=dashboard_html, status_code=200)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard generation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate dashboard: {str(e)}"
        )


# ============================================================
# LIST ALL VISUALIZATIONS FOR DATASET
# ============================================================

@router.get(
    "/{dataset_id}/list",
    response_model=SuccessResponse[dict],
    summary="List Visualizations",
    description="Get list of all visualizations for a dataset."
)
async def list_visualizations(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> Any:
    """
    Get list of all visualizations created for a dataset.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Returns:**
    ```
    {
      "success": true,
      "message": "Found 5 visualizations",
      "data": {
        "visualizations": [
          {
            "id": 1,
            "title": "Age Distribution",
            "chart_type": "histogram",
            "chart_url": "/path/to/chart.html",
            "created_at": "2025-11-08T10:30:00",
            "order": 1
          }
        ],
        "total": 5,
        "dataset_id": 11
      }
    }
    ```
    """
    try:
        # Verify access
        dataset = dataset_service.get_dataset(
            dataset_id=dataset_id,
            user=current_user
        )
        
        # Get visualizations from database
        from app.models.dataset import DatasetVisualization
        visualizations = db.query(DatasetVisualization).filter(
            DatasetVisualization.dataset_id == dataset_id
        ).order_by(DatasetVisualization.order).all()
        
        viz_list = [
            {
                "id": viz.id,
                "title": viz.title,
                "chart_type": viz.chart_type,
                "chart_url": viz.chart_url,
                "created_at": viz.created_at.isoformat() if viz.created_at else None,
                "order": viz.order,
            }
            for viz in visualizations
        ]
        
        logger.info(f"Retrieved {len(viz_list)} visualizations for dataset {dataset_id}")
        
        return SuccessResponse(
            success=True,
            message=f"Found {len(viz_list)} visualizations",
            data={
                "visualizations": viz_list,
                "total": len(viz_list),
                "dataset_id": dataset_id,
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list visualizations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve visualizations"
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@router.get(
    "/health",
    summary="Health Check",
    description="Check visualization service health."
)
async def visualization_health(
    viz_service: VisualizationService = Depends(get_visualization_service),
) -> dict:
    """
    Check if visualization service is operational.
    
    **Returns:**
    ```
    {
      "status": "healthy",
      "service": "visualization",
      "version": "1.0.0",
      "features": {
        "chart_types": 15,
        "export_formats": 5,
        "dashboard": true,
        "automated_suite": true
      }
    }
    ```
    """
    return {
        "status": "healthy",
        "service": "visualization",
        "version": "1.0.0",
        "features": {
            "chart_types": 15,
            "export_formats": 5,
            "dashboard": True,
            "automated_suite": True,
            "data_quality_analysis": True,
        }
    }
