"""
Visualization Service for Advanced Chart Generation.

Handles automated and custom visualization generation using
Plotly, Matplotlib, and Seaborn. Creates interactive, publication-quality
charts with intelligent defaults and comprehensive export options.
"""

import logging
import io
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from enum import Enum

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib

from app.database import get_db
matplotlib.use('Agg')  # Non-interactive backend for server
import seaborn as sns
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.dataset import Dataset, DatasetVisualization


logger = logging.getLogger(__name__)


class ChartType(str, Enum):
    """Chart type enumeration."""
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    LINE = "line"
    BAR = "bar"
    BOX = "box"
    VIOLIN = "violin"
    HEATMAP = "heatmap"
    CORRELATION = "correlation"
    PAIR_PLOT = "pair_plot"
    DISTRIBUTION = "distribution"
    TIME_SERIES = "time_series"
    PIE = "pie"
    TREEMAP = "treemap"
    SUNBURST = "sunburst"
    AREA = "area"


class ChartLibrary(str, Enum):
    """Chart library enumeration."""
    PLOTLY = "plotly"
    MATPLOTLIB = "matplotlib"
    SEABORN = "seaborn"


class VisualizationService:
    """
    Visualization service for generating advanced charts.
    
    Implements:
    - Automated chart recommendations
    - Interactive Plotly visualizations
    - Static Matplotlib/Seaborn charts
    - Multi-variable analysis
    - Custom color schemes
    - Export to multiple formats (HTML, PNG, SVG)
    - Responsive design
    """
    
    # Plotly template for consistent styling
    PLOTLY_TEMPLATE = "plotly_white"
    
    # Color palettes
    COLOR_PALETTES = {
        "default": px.colors.qualitative.Plotly,
        "pastel": px.colors.qualitative.Pastel,
        "vivid": px.colors.qualitative.Vivid,
        "bold": px.colors.qualitative.Bold,
        "safe": px.colors.qualitative.Safe,
    }
    
    def __init__(self, db: Session):
        """
        Initialize visualization service.
        
        Args:
            db: SQLAlchemy database session
        """
        self.db = db
        
        # Set seaborn style
        sns.set_theme(style="whitegrid", palette="husl")
        
        # Configure matplotlib
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 100
    
    # ============================================================
    # AUTOMATED VISUALIZATION GENERATION
    # ============================================================
    
    async def generate_automated_visualizations(
        self,
        dataset_id: int,
        max_charts: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Automatically generate appropriate visualizations for dataset.
        
        Args:
            dataset_id: Dataset ID
            max_charts: Maximum number of charts to generate
            
        Returns:
            List of generated visualizations
            
        Raises:
            HTTPException: If dataset not found or generation fails
        """
        # Get dataset
        dataset = self.db.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found",
            )
        
        if not dataset.is_ready():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dataset not ready. Status: {dataset.status}",
            )
        
        try:
            logger.info(f"Generating automated visualizations for dataset {dataset_id}")
            
            # Load data
            df = self._read_dataframe(dataset.file_path, dataset.file_type)
            
            # Analyze data structure
            analysis = self._analyze_data_structure(df)
            
            # Generate recommended charts
            charts = []
            order = 1
            
            # 1. Numerical distributions
            if analysis["numerical_cols"]:
                for col in analysis["numerical_cols"][:3]:  # Top 3
                    chart = await self._create_distribution_chart(
                        df, col, dataset_id, order
                    )
                    if chart:
                        charts.append(chart)
                        order += 1
            
            # 2. Correlation heatmap
            if len(analysis["numerical_cols"]) >= 2:
                chart = await self._create_correlation_heatmap(
                    df, analysis["numerical_cols"], dataset_id, order
                )
                if chart:
                    charts.append(chart)
                    order += 1
            
            # 3. Categorical distributions
            if analysis["categorical_cols"]:
                for col in analysis["categorical_cols"][:2]:  # Top 2
                    if df[col].nunique() <= 20:  # Reasonable number of categories
                        chart = await self._create_categorical_chart(
                            df, col, dataset_id, order
                        )
                        if chart:
                            charts.append(chart)
                            order += 1
            
            # 4. Scatter plots for interesting relationships
            if len(analysis["numerical_cols"]) >= 2:
                # Use correlation to find interesting pairs
                corr_matrix = df[analysis["numerical_cols"]].corr()
                interesting_pairs = self._find_interesting_correlations(
                    corr_matrix, top_n=2
                )
                
                for col1, col2, corr in interesting_pairs:
                    chart = await self._create_scatter_plot(
                        df, col1, col2, dataset_id, order,
                        hue_col=analysis["categorical_cols"][0] if analysis["categorical_cols"] else None
                    )
                    if chart:
                        charts.append(chart)
                        order += 1
            
            # 5. Box plots for outlier detection
            if analysis["numerical_cols"]:
                chart = await self._create_box_plot(
                    df, analysis["numerical_cols"][:5], dataset_id, order
                )
                if chart:
                    charts.append(chart)
                    order += 1
            
            # Limit to max_charts
            charts = charts[:max_charts]
            
            # Save to database
            saved_charts = self._save_visualizations(charts)
            
            logger.info(f"Generated {len(saved_charts)} visualizations for dataset {dataset_id}")
            
            return [self._visualization_to_dict(v) for v in saved_charts]
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Visualization generation failed: {str(e)}",
            ) from e
        
    # ============================================================
    # PUBLIC API - MAIN CHART GENERATION METHOD
    # ============================================================
    async def generate_chart(
        self,
        dataset_id: int,
        chart_type: ChartType,
        x_column: Optional[str] = None,
        y_column: Optional[str] = None,
        color_column: Optional[str] = None,
        title: Optional[str] = None,
        width: int = 800,
        height: int = 600,
    ) -> Dict[str, Any]:
        """
        Generate chart visualization (main public method).
    
        Args:
            dataset_id: Dataset ID
            chart_type: Type of chart
            x_column: X-axis column
            y_column: Y-axis column
            color_column: Color grouping column
            title: Chart title
            width: Chart width
            height: Chart height
        
        Returns:
            Chart configuration and data
        """
    # Get dataset
        dataset = self.db.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found"
            )
    
        if not dataset.is_ready():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dataset not ready. Status: {dataset.status}"
            )
    
        try:
            logger.info(
                f"Generating {chart_type.value} chart for dataset {dataset_id} "
                f"(x={x_column}, y={y_column})"
            )
        
            # Load data
            df = self._read_dataframe(dataset.file_path, dataset.file_type)
        
            # Create configuration dict
            config = {
                "x_column": x_column,
                "y_column": y_column,
                "color_column": color_column,
                "title": title,
                "width": width,
                "height": height,
            }
        
            # Route to appropriate chart creator
            if chart_type == ChartType.HISTOGRAM:
                if not x_column:
                    # Use first numerical column
                    numerical_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numerical_cols) == 0:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="No numerical columns found for histogram"
                        )
                    x_column = numerical_cols[0]
                    config["x_column"] = x_column

                return await self._create_histogram(df, config, dataset_id)
            
            elif chart_type == ChartType.SCATTER:
                if not x_column or not y_column:
                    numerical_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numerical_cols) < 2:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Need at least 2 numerical columns for scatter plot"
                        )
                    x_column = x_column or numerical_cols[0]
                    y_column = y_column or numerical_cols[1]
                    config["x_column"] = x_column
                    config["y_column"] = y_column
            
                return await self._create_scatter(df, config, dataset_id)
            
            elif chart_type == ChartType.LINE:
                if not x_column or not y_column:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Both x_column and y_column required for line chart"
                    )
                return await self._create_line_chart(df, config, dataset_id)
            
            elif chart_type in [ChartType.HEATMAP, ChartType.CORRELATION]:
                return await self._create_heatmap(df, config, dataset_id)
            
            elif chart_type == ChartType.PIE:
                if not x_column:
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                    if len(categorical_cols) == 0:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="No categorical columns found for pie chart"
                        )
                    x_column = categorical_cols[0]
                    config["x_column"] = x_column
            
                return await self._create_pie_chart(df, config, dataset_id)
            
            elif chart_type == ChartType.BAR:
                if not x_column:
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                    if len(categorical_cols) == 0:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="No categorical columns found for bar chart"
                        )   
                    x_column = categorical_cols[0]
                    config["x_column"] = x_column
            
                return await self._create_bar_chart(df, config, dataset_id)
            
            elif chart_type == ChartType.BOX:
                if not y_column:
                    numerical_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numerical_cols) == 0:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="No numerical columns found for box plot"
                        )
                    config["columns"] = numerical_cols[:5].tolist()
            
                return await self._create_box_plot_custom(df, config, dataset_id)
            
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Chart type {chart_type.value} not supported"
                )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Chart generation failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Chart generation failed: {str(e)}"
            ) from e

    
    # ============================================================
    # CHART CREATION - DISTRIBUTIONS
    # ============================================================
    
    async def _create_distribution_chart(
        self,
        df: pd.DataFrame,
        column: str,
        dataset_id: int,
        order: int
    ) -> Optional[Dict[str, Any]]:
        """
        Create distribution chart for numerical column.
        
        Args:
            df: DataFrame
            column: Column name
            dataset_id: Dataset ID
            order: Display order
            
        Returns:
            Chart dictionary
        """
        try:
            # Create histogram with KDE overlay
            fig = go.Figure()
            
            # Histogram
            fig.add_trace(go.Histogram(
                x=df[column],
                name='Distribution',
                nbinsx=50,
                marker_color='rgb(55, 126, 184)',
                opacity=0.7,
            ))
            
            # Add mean and median lines
            mean_val = df[column].mean()
            median_val = df[column].median()
            
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_val:.2f}",
                annotation_position="top"
            )
            
            fig.add_vline(
                x=median_val,
                line_dash="dot",
                line_color="green",
                annotation_text=f"Median: {median_val:.2f}",
                annotation_position="bottom"
            )
            
            # Update layout
            fig.update_layout(
                title=f"Distribution of {column}",
                xaxis_title=column,
                yaxis_title="Frequency",
                template=self.PLOTLY_TEMPLATE,
                hovermode='x unified',
                showlegend=True,
            )
            
            # Save chart
            chart_url = await self._save_plotly_chart(fig, dataset_id, f"dist_{column}")
            
            return {
                "dataset_id": dataset_id,
                "title": f"Distribution of {column}",
                "chart_type": ChartType.DISTRIBUTION.value,
                "chart_url": chart_url,
                "config": {
                    "column": column,
                    "mean": float(mean_val),
                    "median": float(median_val),
                    "std": float(df[column].std()),
                },
                "data_columns": [column],
                "order": order,
            }
            
        except Exception as e:
            logger.error(f"Failed to create distribution chart for {column}: {str(e)}")
            return None
    
    # ============================================================
    # CHART CREATION - CORRELATION
    # ============================================================
    
    async def _create_correlation_heatmap(
        self,
        df: pd.DataFrame,
        columns: List[str],
        dataset_id: int,
        order: int
    ) -> Optional[Dict[str, Any]]:
        """
        Create correlation heatmap for numerical columns.
        
        Args:
            df: DataFrame
            columns: Column names
            dataset_id: Dataset ID
            order: Display order
            
        Returns:
            Chart dictionary
        """
        try:
            # Calculate correlation matrix
            corr_matrix = df[columns].corr()
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation"),
            ))
            
            # Update layout
            fig.update_layout(
                title="Correlation Matrix",
                xaxis_title="Features",
                yaxis_title="Features",
                template=self.PLOTLY_TEMPLATE,
                height=600,
                width=700,
            )
            
            # Save chart
            chart_url = await self._save_plotly_chart(fig, dataset_id, "correlation_heatmap")
            
            return {
                "dataset_id": dataset_id,
                "title": "Correlation Matrix",
                "chart_type": ChartType.CORRELATION.value,
                "chart_url": chart_url,
                "config": {"columns": columns},
                "data_columns": columns,
                "order": order,
            }
            
        except Exception as e:
            logger.error(f"Failed to create correlation heatmap: {str(e)}")
            return None
    
    # ============================================================
    # CHART CREATION - CATEGORICAL
    # ============================================================
    
    async def _create_categorical_chart(
        self,
        df: pd.DataFrame,
        column: str,
        dataset_id: int,
        order: int
    ) -> Optional[Dict[str, Any]]:
        """
        Create bar chart for categorical column.
        
        Args:
            df: DataFrame
            column: Column name
            dataset_id: Dataset ID
            order: Display order
            
        Returns:
            Chart dictionary
        """
        try:
            # Calculate value counts
            value_counts = df[column].value_counts().head(20)  # Top 20 categories
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    marker_color='rgb(77, 175, 74)',
                    text=value_counts.values,
                    textposition='auto',
                )
            ])
            
            # Update layout
            fig.update_layout(
                title=f"Distribution of {column}",
                xaxis_title=column,
                yaxis_title="Count",
                template=self.PLOTLY_TEMPLATE,
                xaxis_tickangle=-45,
            )
            
            # Save chart
            chart_url = await self._save_plotly_chart(fig, dataset_id, f"cat_{column}")
            
            return {
                "dataset_id": dataset_id,
                "title": f"Distribution of {column}",
                "chart_type": ChartType.BAR.value,
                "chart_url": chart_url,
                "config": {
                    "column": column,
                    "categories": int(len(value_counts)),
                },
                "data_columns": [column],
                "order": order,
            }
            
        except Exception as e:
            logger.error(f"Failed to create categorical chart for {column}: {str(e)}")
            return None
    
    # ============================================================
    # CHART CREATION - SCATTER PLOT
    # ============================================================
    
    async def _create_scatter_plot(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        dataset_id: int,
        order: int,
        hue_col: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create scatter plot for two numerical columns.
        
        Args:
            df: DataFrame
            x_col: X-axis column
            y_col: Y-axis column
            dataset_id: Dataset ID
            order: Display order
            hue_col: Optional color grouping column
            
        Returns:
            Chart dictionary
        """
        try:
            # Calculate correlation
            correlation = df[[x_col, y_col]].corr().iloc[0, 1]
            
            # Create scatter plot
            if hue_col and hue_col in df.columns:
                # Colored by category
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    color=hue_col,
                    trendline="ols",
                    template=self.PLOTLY_TEMPLATE,
                    title=f"{x_col} vs {y_col} (r={correlation:.3f})",
                )
            else:
                # Single color with trendline
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    trendline="ols",
                    template=self.PLOTLY_TEMPLATE,
                    title=f"{x_col} vs {y_col} (r={correlation:.3f})",
                )
                
                fig.update_traces(marker=dict(color='rgb(55, 126, 184)'))
            
            # Update layout
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col,
                hovermode='closest',
            )
            
            # Save chart
            chart_url = await self._save_plotly_chart(
                fig, dataset_id, f"scatter_{x_col}_{y_col}"
            )
            
            return {
                "dataset_id": dataset_id,
                "title": f"{x_col} vs {y_col}",
                "chart_type": ChartType.SCATTER.value,
                "chart_url": chart_url,
                "config": {
                    "x_column": x_col,
                    "y_column": y_col,
                    "correlation": float(correlation),
                    "hue_column": hue_col,
                },
                "data_columns": [x_col, y_col] + ([hue_col] if hue_col else []),
                "order": order,
            }
            
        except Exception as e:
            logger.error(f"Failed to create scatter plot: {str(e)}")
            return None
    
    # ============================================================
    # CHART CREATION - BOX PLOT
    # ============================================================
    
    async def _create_box_plot(
        self,
        df: pd.DataFrame,
        columns: List[str],
        dataset_id: int,
        order: int
    ) -> Optional[Dict[str, Any]]:
        """
        Create box plot for outlier detection.
        
        Args:
            df: DataFrame
            columns: Column names
            dataset_id: Dataset ID
            order: Display order
            
        Returns:
            Chart dictionary
        """
        try:
            # Create box plot
            fig = go.Figure()
            
            for col in columns:
                fig.add_trace(go.Box(
                    y=df[col],
                    name=col,
                    boxmean='sd',  # Show mean and standard deviation
                ))
            
            # Update layout
            fig.update_layout(
                title="Box Plot - Outlier Detection",
                yaxis_title="Value",
                template=self.PLOTLY_TEMPLATE,
                showlegend=True,
                height=500,
            )
            
            # Save chart
            chart_url = await self._save_plotly_chart(fig, dataset_id, "box_plot")
            
            return {
                "dataset_id": dataset_id,
                "title": "Box Plot - Outlier Detection",
                "chart_type": ChartType.BOX.value,
                "chart_url": chart_url,
                "config": {"columns": columns},
                "data_columns": columns,
                "order": order,
            }
            
        except Exception as e:
            logger.error(f"Failed to create box plot: {str(e)}")
            return None
    
    # ============================================================
    # CUSTOM CHART GENERATION
    # ============================================================
    
    async def create_custom_chart(
        self,
        dataset_id: int,
        chart_type: ChartType,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create custom chart with specified configuration.
        
        Args:
            dataset_id: Dataset ID
            chart_type: Type of chart to create
            config: Chart configuration
            
        Returns:
            Created visualization dictionary
            
        Raises:
            HTTPException: If creation fails
        """
        dataset = self.db.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found",
            )
        
        try:
            df = self._read_dataframe(dataset.file_path, dataset.file_type)
            
            # Route to appropriate chart creator
            if chart_type == ChartType.HISTOGRAM:
                chart = await self._create_histogram(df, config, dataset_id)
            elif chart_type == ChartType.SCATTER:
                chart = await self._create_scatter(df, config, dataset_id)
            elif chart_type == ChartType.LINE:
                chart = await self._create_line_chart(df, config, dataset_id)
            elif chart_type == ChartType.HEATMAP:
                chart = await self._create_heatmap(df, config, dataset_id)
            elif chart_type == ChartType.PIE:
                chart = await self._create_pie_chart(df, config, dataset_id)
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Chart type {chart_type} not supported",
                )
            
            # Save to database
            saved_chart = self._save_visualizations([chart])[0]
            
            return self._visualization_to_dict(saved_chart)
            
        except Exception as e:
            logger.error(f"Custom chart creation failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Chart creation failed: {str(e)}",
            ) from e
    
    # ============================================================
    # DATA ANALYSIS
    # ============================================================
    
    def _analyze_data_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data structure to recommend appropriate charts.
        
        Args:
            df: DataFrame
            
        Returns:
            Analysis dictionary
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        return {
            "numerical_cols": numerical_cols,
            "categorical_cols": categorical_cols,
            "datetime_cols": datetime_cols,
            "total_cols": len(df.columns),
            "total_rows": len(df),
        }
    
    def _find_interesting_correlations(
        self,
        corr_matrix: pd.DataFrame,
        top_n: int = 3,
        threshold: float = 0.5
    ) -> List[Tuple[str, str, float]]:
        """
        Find interesting correlations for visualization.
        
        Args:
            corr_matrix: Correlation matrix
            top_n: Number of pairs to return
            threshold: Minimum correlation threshold
            
        Returns:
            List of (col1, col2, correlation) tuples
        """
        interesting = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr = corr_matrix.iloc[i, j]
                
                if abs(corr) > threshold:
                    interesting.append((col1, col2, corr))
        
        # Sort by absolute correlation
        interesting.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return interesting[:top_n]
    
    # ============================================================
    # FILE OPERATIONS
    # ============================================================
    
    async def _save_plotly_chart(
        self,
        fig: go.Figure,
        dataset_id: int,
        chart_name: str
    ) -> str:
        """
        Save Plotly chart as HTML file.
        
        Args:
            fig: Plotly figure
            dataset_id: Dataset ID
            chart_name: Chart identifier
            
        Returns:
            URL/path to saved chart
        """
        # Get dataset to find owner
        dataset = self.db.get(Dataset, dataset_id)
        
        # Create charts directory
        charts_dir = Path(settings.UPLOAD_DIR) / f"user_{dataset.owner_id}" / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{chart_name}_{timestamp}.html"
        filepath = charts_dir / filename
        
        # Save as HTML
        fig.write_html(
            str(filepath),
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d'],
            }
        )
        
        logger.info(f"Saved chart to {filepath}")
        
        # In production, upload to S3
        if settings.USE_S3:
            return self._upload_chart_to_s3(filepath, dataset)
        
        return str(filepath)
    
    def _upload_chart_to_s3(self, filepath: Path, dataset: Dataset) -> str:
        """Upload chart to S3 (placeholder)."""
        # TODO: Implement S3 upload
        return str(filepath)
    
    def _read_dataframe(self, file_path: str, file_type: str) -> pd.DataFrame:
        """Read file into pandas DataFrame."""
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
    
    # ============================================================
    # DATABASE OPERATIONS
    # ============================================================
    
    def _save_visualizations(
        self,
        charts: List[Dict[str, Any]]
    ) -> List[DatasetVisualization]:
        """
        Save visualizations to database.
        
        Args:
            charts: List of chart dictionaries
            
        Returns:
            List of saved DatasetVisualization instances
        """
        saved_charts = []
        
        for chart_data in charts:
            viz = DatasetVisualization(
                dataset_id=chart_data["dataset_id"],
                title=chart_data["title"],
                chart_type=chart_data["chart_type"],
                chart_url=chart_data["chart_url"],
                config=chart_data.get("config"),
                data_columns=chart_data.get("data_columns"),
                order=chart_data["order"],
            )
            
            self.db.add(viz)
            saved_charts.append(viz)
        
        self.db.commit()
        
        return saved_charts
    
    def _visualization_to_dict(self, viz: DatasetVisualization) -> Dict[str, Any]:
        """Convert DatasetVisualization to dictionary."""
        return {
            "id": viz.id,
            "dataset_id": viz.dataset_id,
            "title": viz.title,
            "chart_type": viz.chart_type,
            "chart_url": viz.chart_url,
            "config": viz.config,
            "data_columns": viz.data_columns,
            "order": viz.order,
            "created_at": viz.created_at.isoformat(),
        }

  
# ============================================================
# DEPENDENCY INJECTION HELPER
# ============================================================

def get_visualization_service(db: Session = Depends(get_db)) -> VisualizationService:
    """
    Dependency for injecting VisualizationService.
    
    Args:
        db: Database session
        
    Returns:
        VisualizationService instance
    """
    return VisualizationService(db)
