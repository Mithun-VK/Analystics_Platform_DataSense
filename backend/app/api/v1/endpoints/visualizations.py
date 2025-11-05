"""
Visualization Endpoints.

Handles chart generation, interactive visualizations, and data plotting.
Provides various chart types with customization options.
"""

import logging
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
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
    description="Generate a chart/visualization for a dataset column or columns."
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
    Generate a visualization for dataset.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Query Parameters:**
    - chart_type: Type of chart (bar, line, scatter, pie, box, histogram, heatmap)
    - x_column: Column name for X-axis
    - y_column: Column name for Y-axis (required for some chart types)
    - color_column: Column for color grouping (optional)
    - title: Custom chart title (optional)
    - width: Chart width in pixels (400-2000)
    - height: Chart height in pixels (300-1500)
    
    **Chart Types:**
    
    **Bar Chart:**
    - x_column: Categorical column
    - y_column: Numerical column
    - Shows comparison across categories
    
    **Line Chart:**
    - x_column: Sequential column (time series)
    - y_column: Numerical column
    - Shows trends over time
    
    **Scatter Plot:**
    - x_column: Numerical column
    - y_column: Numerical column
    - color_column: Optional grouping
    - Shows relationships between variables
    
    **Pie Chart:**
    - x_column: Categorical column
    - y_column: Numerical column (or count if not specified)
    - Shows proportions
    
    **Box Plot:**
    - x_column: Categorical column (optional)
    - y_column: Numerical column
    - Shows distribution and outliers
    
    **Histogram:**
    - x_column: Numerical column
    - Shows frequency distribution
    
    **Heatmap:**
    - Shows correlation between all numerical columns
    - No column selection needed
    
    **Returns:**
    - Chart configuration (Plotly JSON)
    - Chart HTML (for embedding)
    - Chart image URL (if saved)
    
    **Errors:**
    - 404: Dataset not found
    - 403: Access denied
    - 400: Invalid chart configuration
    """
    try:
        logger.info(
            f"Generating {chart_type.value} visualization for dataset {dataset_id} "
            f"by user {current_user.id}"
        )
        
        # Verify access
        dataset = dataset_service.get_dataset(
            dataset_id=dataset_id,
            user=current_user
        )
        
        # Check if dataset is ready
        if not dataset.is_ready():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dataset not ready for visualization. Current status: {dataset.status}"
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
        logger.error(f"Visualization generation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate visualization: {str(e)}"
        )


# ============================================================
# GET VISUALIZATION RECOMMENDATIONS
# ============================================================

@router.get(
    "/{dataset_id}/recommendations",
    response_model=List[dict],
    summary="Get Visualization Recommendations",
    description="Get recommended chart types based on dataset structure."
)
async def get_visualization_recommendations(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    viz_service: VisualizationService = Depends(get_visualization_service),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> Any:
    """
    Get visualization recommendations based on data types.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Returns:**
    - List of recommended visualizations with:
      - chart_type: Recommended chart type
      - reason: Why this chart is recommended
      - suggested_columns: Which columns to use
      - priority: High, Medium, or Low
      - example_config: Sample configuration
    
    **Recommendation Logic:**
    - Analyzes column data types
    - Identifies categorical vs numerical columns
    - Detects time series data
    - Suggests appropriate chart types
    - Ranks by relevance
    
    **Example Response:**
    ```
    [
      {
        "chart_type": "scatter",
        "reason": "Multiple numerical columns found - good for correlation analysis",
        "suggested_columns": {
          "x": "Age",
          "y": "Salary",
          "color": "Department"
        },
        "priority": "high",
        "example_config": {
          "chart_type": "scatter",
          "x_column": "Age",
          "y_column": "Salary",
          "color_column": "Department"
        }
      }
    ]
    ```
    
    **Errors:**
    - 404: Dataset not found
    - 403: Access denied
    """
    try:
        logger.info(f"Getting visualization recommendations for dataset {dataset_id}")
        
        # Verify access
        dataset = dataset_service.get_dataset(
            dataset_id=dataset_id,
            user=current_user
        )
        
        # Get recommendations
        recommendations = viz_service.get_recommendations(dataset_id)
        
        return recommendations
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recommendations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get visualization recommendations"
        )


# ============================================================
# GENERATE MULTIPLE CHARTS
# ============================================================

@router.post(
    "/{dataset_id}/generate-suite",
    response_model=SuccessResponse[dict],
    summary="Generate Visualization Suite",
    description="Generate a comprehensive set of visualizations for a dataset."
)
async def generate_visualization_suite(
    dataset_id: int,
    include_distributions: bool = Query(True, description="Include distribution plots"),
    include_correlations: bool = Query(True, description="Include correlation heatmap"),
    include_comparisons: bool = Query(True, description="Include comparison charts"),
    max_charts: int = Query(10, ge=1, le=20, description="Maximum number of charts"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    viz_service: VisualizationService = Depends(get_visualization_service),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> Any:
    """
    Generate a comprehensive set of visualizations.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Query Parameters:**
    - include_distributions: Generate histograms for numerical columns
    - include_correlations: Generate correlation heatmap
    - include_comparisons: Generate comparison charts for categorical data
    - max_charts: Maximum number of charts to generate (1-20)
    
    **Generated Charts Include:**
    
    **Distribution Analysis:**
    - Histograms for all numerical columns
    - Box plots for outlier detection
    - Density plots
    
    **Correlation Analysis:**
    - Correlation heatmap
    - Scatter plots for highly correlated pairs
    
    **Categorical Analysis:**
    - Bar charts for categorical columns
    - Pie charts for proportions
    - Count plots
    
    **Time Series (if detected):**
    - Line charts for trends
    - Seasonal decomposition
    
    **Returns:**
    - List of generated charts
    - Summary statistics
    - Recommended next steps
    
    **Errors:**
    - 404: Dataset not found
    - 403: Access denied
    - 400: Dataset not ready
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
        suite = await viz_service.generate_visualization_suite(
            dataset_id=dataset_id,
            include_distributions=include_distributions,
            include_correlations=include_correlations,
            include_comparisons=include_comparisons,
            max_charts=max_charts,
        )
        
        logger.info(
            f"Generated {len(suite['charts'])} charts for dataset {dataset_id}"
        )
        
        return SuccessResponse(
            success=True,
            message=f"Generated {len(suite['charts'])} visualizations",
            data=suite
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visualization suite generation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate visualization suite"
        )


# ============================================================
# EXPORT CHART
# ============================================================

@router.post(
    "/{dataset_id}/export",
    summary="Export Chart",
    description="Export a chart as image file (PNG, SVG, or PDF)."
)
async def export_chart(
    dataset_id: int,
    chart_type: ChartType = Query(..., description="Type of chart"),
    x_column: Optional[str] = Query(None),
    y_column: Optional[str] = Query(None),
    format: str = Query("png", regex="^(png|svg|pdf)$", description="Export format"),
    width: int = Query(1200, ge=400, le=4000),
    height: int = Query(900, ge=300, le=3000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    viz_service: VisualizationService = Depends(get_visualization_service),
) -> Any:
    """
    Export chart as image file.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Query Parameters:**
    - chart_type: Type of chart to export
    - x_column: X-axis column
    - y_column: Y-axis column
    - format: Export format (png, svg, pdf)
    - width: Image width (400-4000)
    - height: Image height (300-3000)
    
    **Returns:**
    - File download response
    
    **Supported Formats:**
    - PNG: Raster image, good for web
    - SVG: Vector image, scalable
    - PDF: Print-ready format
    
    **Errors:**
    - 404: Dataset not found
    - 403: Access denied
    """
    from fastapi.responses import FileResponse
    
    try:
        logger.info(f"Exporting chart for dataset {dataset_id} as {format}")
        
        # Generate and export chart
        file_path = await viz_service.export_chart(
            dataset_id=dataset_id,
            chart_type=chart_type,
            x_column=x_column,
            y_column=y_column,
            format=format,
            width=width,
            height=height,
        )
        
        return FileResponse(
            path=file_path,
            filename=f"chart_{dataset_id}.{format}",
            media_type=f"image/{format}" if format != "pdf" else "application/pdf"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chart export failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export chart"
        )


# ============================================================
# GET AVAILABLE COLUMNS
# ============================================================

@router.get(
    "/{dataset_id}/columns",
    response_model=dict,
    summary="Get Available Columns",
    description="Get list of columns available for visualization."
)
async def get_available_columns(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> Any:
    """
    Get columns available for visualization.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Returns:**
    - Numerical columns (suitable for Y-axis, aggregations)
    - Categorical columns (suitable for X-axis, grouping)
    - Temporal columns (suitable for time series)
    - Boolean columns
    - Column metadata (data type, unique values, etc.)
    
    **Example Response:**
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
          "unique_count": 43
        }
      }
    }
    ```
    
    **Errors:**
    - 404: Dataset not found
    - 403: Access denied
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
# INTERACTIVE DASHBOARD
# ============================================================

@router.get(
    "/{dataset_id}/dashboard",
    summary="Generate Interactive Dashboard",
    description="Generate an interactive HTML dashboard with multiple visualizations."
)
async def generate_dashboard(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    viz_service: VisualizationService = Depends(get_visualization_service),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> Any:
    """
    Generate interactive HTML dashboard.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Returns:**
    - HTML file with interactive dashboard
    
    **Dashboard Includes:**
    - Summary statistics
    - Key visualizations
    - Interactive filters
    - Download options
    - Responsive layout
    
    **Errors:**
    - 404: Dataset not found
    - 403: Access denied
    """
    from fastapi.responses import HTMLResponse
    
    try:
        logger.info(f"Generating dashboard for dataset {dataset_id}")
        
        # Verify access
        dataset = dataset_service.get_dataset(
            dataset_id=dataset_id,
            user=current_user
        )
        
        # Generate dashboard HTML
        dashboard_html = await viz_service.generate_dashboard(dataset_id)
        
        return HTMLResponse(content=dashboard_html)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard generation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate dashboard"
        )
