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
        """Initialize visualization service."""
        self.db = db
        sns.set_theme(style="whitegrid", palette="husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 100
    
    # ============================================================
    # AUTOMATED VISUALIZATION GENERATION
    # ============================================================
    
    async def generate_automated_visualizations(
            self,
            dataset_id: int,
            max_charts: int = 20  # Increase max charts default to 20
        ) -> List[Dict[str, Any]]:
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

            df = self._read_dataframe(dataset.file_path, dataset.file_type)
            analysis = self._analyze_data_structure(df)

            charts = []
            order = 1

            # 1. Numerical distributions (increase from 3 to 7)
            if analysis["numerical_cols"]:
                for col in analysis["numerical_cols"][:7]:
                    chart = await self._create_distribution_chart(df, col, dataset_id, order)
                    if chart:
                        charts.append(chart)
                        order += 1

            # 2. Correlation heatmap (keep as is)
            if len(analysis["numerical_cols"]) >= 2:
                chart = await self._create_correlation_heatmap(df, analysis["numerical_cols"], dataset_id, order)
                if chart:
                    charts.append(chart)
                    order += 1

        # 3. Categorical distributions (increase from 2 to 5)
            if analysis["categorical_cols"]:
                for col in analysis["categorical_cols"][:5]:
                    if df[col].nunique() <= 20:
                        chart = await self._create_categorical_chart(df, col, dataset_id, order)
                        if chart:
                            charts.append(chart)
                            order += 1

        # 4. Scatter plots (increase top_n from 2 to 5)
            if len(analysis["numerical_cols"]) >= 2:
                corr_matrix = df[analysis["numerical_cols"]].corr()
                interesting_pairs = self._find_interesting_correlations(corr_matrix, top_n=5)
                for col1, col2, corr in interesting_pairs:
                    chart = await self._create_scatter_plot(
                        df, col1, col2, dataset_id, order,
                        hue_col=analysis["categorical_cols"][0] if analysis["categorical_cols"] else None
                    )
                    if chart:
                        charts.append(chart)
                        order += 1

            # 5. Box plots (increase from 5 to 7 numerical cols)
            if analysis["numerical_cols"]:
                chart = await self._create_box_plot(df, analysis["numerical_cols"][:7], dataset_id, order)
                if chart:
                    charts.append(chart)
                    order += 1

            # Limit charts to max_charts number at the end
            charts = charts[:max_charts]
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
        """Generate chart visualization (main public method)."""
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
        
            df = self._read_dataframe(dataset.file_path, dataset.file_type)
        
            config = {
                "x_column": x_column,
                "y_column": y_column,
                "color_column": color_column,
                "title": title,
                "width": width,
                "height": height,
            }
        
            if chart_type == ChartType.HISTOGRAM:
                if not x_column:
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
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                if len(numerical_cols) == 0:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="No numerical columns found for box plot"
                    )
                config["columns"] = numerical_cols[:5].tolist()
                return await self._create_box_plot_custom(df, config, dataset_id)
            
            elif chart_type == ChartType.AREA:
                if not x_column or not y_column:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Both x_column and y_column required for area chart"
                    )
                return await self._create_area_chart(df, config, dataset_id)
            
            elif chart_type == ChartType.VIOLIN:
                if not y_column:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="y_column required for violin plot"
                    )
                return await self._create_violin_plot(df, config, dataset_id)
            
            elif chart_type == ChartType.TIME_SERIES:
                if not x_column or not y_column:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Both x_column and y_column required for time series"
                    )
                return await self._create_time_series(df, config, dataset_id)
            
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
    # CHART CREATION - HISTOGRAM ✓
    # ============================================================
    
    async def _create_histogram(
        self,
        df: pd.DataFrame,
        config: dict,
        dataset_id: int
    ) -> Dict[str, Any]:
        """Create histogram visualization."""
        try:
            x_col = config.get('x_column')
            bins = config.get('bins', 30)
            title = config.get('title') or f'Histogram of {x_col}'
            width = config.get('width', 800)
            height = config.get('height', 600)
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=df[x_col].dropna(),
                nbinsx=bins,
                marker_color='rgba(55, 126, 184, 0.7)',
                name='Frequency',
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            ))
            
            mean_val = df[x_col].mean()
            median_val = df[x_col].median()
            std_val = df[x_col].std()
            
            fig.add_vline(x=mean_val, line_dash="dash", line_color="red", line_width=2,
                         annotation_text=f"Mean: {mean_val:.2f}", annotation_position="top right")
            fig.add_vline(x=median_val, line_dash="dot", line_color="green", line_width=2,
                         annotation_text=f"Median: {median_val:.2f}", annotation_position="bottom right")
            
            fig.update_layout(
                title=title, xaxis_title=x_col, yaxis_title='Frequency',
                template=self.PLOTLY_TEMPLATE, hovermode='x unified', showlegend=True,
                width=width, height=height, plot_bgcolor='rgba(240, 240, 240, 0.5)',
            )
            
            chart_url = await self._save_plotly_chart(fig, dataset_id, f"histogram_{x_col}")
            
            return {
                "dataset_id": dataset_id, "title": title,
                "chart_type": ChartType.HISTOGRAM.value, "chart_url": chart_url,
                "config": {"column": x_col, "mean": float(mean_val), "median": float(median_val),
                          "std": float(std_val), "bins": bins},
                "data_columns": [x_col], "order": 1,
            }
        except Exception as e:
            logger.error(f"Failed to create histogram: {str(e)}")
            raise

    # ============================================================
    # CHART CREATION - SCATTER ✓
    # ============================================================
    
    async def _create_scatter(
        self,
        df: pd.DataFrame,
        config: dict,
        dataset_id: int
    ) -> Dict[str, Any]:
        """Create scatter plot visualization."""
        try:
            x_col = config.get('x_column')
            y_col = config.get('y_column')
            color_col = config.get('color_column')
            title = config.get('title') or f'{x_col} vs {y_col}'
            width = config.get('width', 800)
            height = config.get('height', 600)
            
            valid_data = df[[x_col, y_col]].dropna()
            correlation = valid_data[x_col].corr(valid_data[y_col])
            
            if color_col and color_col in df.columns and df[color_col].dtype == 'object':
                fig = px.scatter(
                    df, x=x_col, y=y_col, color=color_col, trendline="ols",
                    trendline_color_override="red", template=self.PLOTLY_TEMPLATE,
                    title=title, width=width, height=height,
                    hover_data={x_col: ':.2f', y_col: ':.2f'},
                )
            else:
                fig = px.scatter(
                    df, x=x_col, y=y_col, trendline="ols",
                    trendline_color_override="red", template=self.PLOTLY_TEMPLATE,
                    title=title, width=width, height=height,
                )
                fig.update_traces(
                    marker=dict(color='rgba(55, 126, 184, 0.6)', size=8,
                               line=dict(width=0.5, color='white')),
                    selector=dict(mode='markers')
                )
            
            fig.update_layout(
                xaxis_title=x_col, yaxis_title=y_col, hovermode='closest',
                plot_bgcolor='rgba(240, 240, 240, 0.5)',
            )
            
            chart_url = await self._save_plotly_chart(fig, dataset_id, f"scatter_{x_col}_{y_col}")
            
            return {
                "dataset_id": dataset_id, "title": title,
                "chart_type": ChartType.SCATTER.value, "chart_url": chart_url,
                "config": {"x_column": x_col, "y_column": y_col, "color_column": color_col,
                          "correlation": float(correlation)},
                "data_columns": [x_col, y_col] + ([color_col] if color_col else []),
                "order": 1,
            }
        except Exception as e:
            logger.error(f"Failed to create scatter plot: {str(e)}")
            raise

    # ============================================================
    # CHART CREATION - LINE CHART ✓
    # ============================================================
    
    async def _create_line_chart(
        self,
        df: pd.DataFrame,
        config: dict,
        dataset_id: int
    ) -> Dict[str, Any]:
        """Create line chart visualization."""
        try:
            x_col = config.get('x_column')
            y_col = config.get('y_column')
            color_col = config.get('color_column')
            title = config.get('title') or f'{y_col} vs {x_col}'
            width = config.get('width', 800)
            height = config.get('height', 600)
            
            df_sorted = df.sort_values(x_col)
            
            if color_col and color_col in df.columns:
                fig = px.line(
                    df_sorted, x=x_col, y=y_col, color=color_col,
                    template=self.PLOTLY_TEMPLATE, title=title, width=width,
                    height=height, markers=True,
                )
            else:
                fig = px.line(
                    df_sorted, x=x_col, y=y_col,
                    template=self.PLOTLY_TEMPLATE, title=title, width=width,
                    height=height, markers=True,
                )
                fig.update_traces(
                    line=dict(color='rgba(55, 126, 184, 0.8)', width=2),
                    marker=dict(size=6),
                )
            
            fig.update_layout(
                xaxis_title=x_col, yaxis_title=y_col, hovermode='x unified',
                plot_bgcolor='rgba(240, 240, 240, 0.5)',
            )
            
            chart_url = await self._save_plotly_chart(fig, dataset_id, f"line_{x_col}_{y_col}")
            
            return {
                "dataset_id": dataset_id, "title": title,
                "chart_type": ChartType.LINE.value, "chart_url": chart_url,
                "config": {"x_column": x_col, "y_column": y_col, "color_column": color_col},
                "data_columns": [x_col, y_col] + ([color_col] if color_col else []),
                "order": 1,
            }
        except Exception as e:
            logger.error(f"Failed to create line chart: {str(e)}")
            raise

    # ============================================================
    # CHART CREATION - HEATMAP/CORRELATION ✓
    # ============================================================
    
    async def _create_heatmap(
        self,
        df: pd.DataFrame,
        config: dict,
        dataset_id: int
    ) -> Dict[str, Any]:
        """Create heatmap/correlation visualization."""
        try:
            title = config.get('title') or 'Correlation Matrix'
            width = config.get('width', 800)
            height = config.get('height', 700)
            
            numerical_cols = config.get('columns') or df.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            
            if len(numerical_cols) < 2:
                raise ValueError("Need at least 2 numerical columns for heatmap")
            
            corr_matrix = df[numerical_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
                text=np.round(corr_matrix.values, 2), texttemplate='%{text:.2f}',
                textfont={"size": 10},
                hovertemplate='%{y} - %{x}: %{z:.3f}<extra></extra>',
                colorbar=dict(title="Correlation", thickness=15, len=0.7),
            ))
            
            fig.update_layout(
                title=title, xaxis_title="Features", yaxis_title="Features",
                template=self.PLOTLY_TEMPLATE, height=height, width=width,
                xaxis_tickangle=-45, plot_bgcolor='white',
            )
            
            chart_url = await self._save_plotly_chart(fig, dataset_id, "heatmap_correlation")
            
            return {
                "dataset_id": dataset_id, "title": title,
                "chart_type": ChartType.CORRELATION.value, "chart_url": chart_url,
                "config": {"columns": numerical_cols}, "data_columns": numerical_cols, "order": 1,
            }
        except Exception as e:
            logger.error(f"Failed to create heatmap: {str(e)}")
            raise

    # ============================================================
    # CHART CREATION - PIE CHART ✓
    # ============================================================
    
    async def _create_pie_chart(
        self,
        df: pd.DataFrame,
        config: dict,
        dataset_id: int
    ) -> Dict[str, Any]:
        """Create pie chart visualization."""
        try:
            col = config.get('x_column')
            title = config.get('title') or f'Distribution of {col}'
            width = config.get('width', 800)
            height = config.get('height', 600)
            
            value_counts = df[col].value_counts().head(15)
            
            fig = go.Figure(data=[go.Pie(
                labels=value_counts.index, values=value_counts.values,
                textposition='inside', textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>',
                marker=dict(line=dict(color='white', width=2)),
            )])
            
            fig.update_layout(
                title=title, template=self.PLOTLY_TEMPLATE, width=width, height=height,
            )
            
            chart_url = await self._save_plotly_chart(fig, dataset_id, f"pie_{col}")
            
            return {
                "dataset_id": dataset_id, "title": title,
                "chart_type": ChartType.PIE.value, "chart_url": chart_url,
                "config": {"column": col, "categories": int(len(value_counts))},
                "data_columns": [col], "order": 1,
            }
        except Exception as e:
            logger.error(f"Failed to create pie chart: {str(e)}")
            raise

    # ============================================================
    # CHART CREATION - BAR CHART ✓
    # ============================================================
    
    async def _create_bar_chart(
        self,
        df: pd.DataFrame,
        config: dict,
        dataset_id: int
    ) -> Dict[str, Any]:
        """Create bar chart visualization."""
        try:
            col = config.get('x_column')
            y_col = config.get('y_column')
            title = config.get('title') or f'{col} Distribution'
            width = config.get('width', 800)
            height = config.get('height', 600)
            
            if y_col:
                bar_data = df.groupby(col)[y_col].sum().sort_values(ascending=False).head(20)
            else:
                bar_data = df[col].value_counts().head(20)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=bar_data.index, y=bar_data.values,
                    marker_color='rgba(77, 175, 74, 0.7)',
                    text=bar_data.values, textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>',
                )
            ])
            
            fig.update_layout(
                title=title, xaxis_title=col, yaxis_title='Count' if not y_col else y_col,
                template=self.PLOTLY_TEMPLATE, xaxis_tickangle=-45,
                width=width, height=height, plot_bgcolor='rgba(240, 240, 240, 0.5)',
            )
            
            chart_url = await self._save_plotly_chart(fig, dataset_id, f"bar_{col}")
            
            return {
                "dataset_id": dataset_id, "title": title,
                "chart_type": ChartType.BAR.value, "chart_url": chart_url,
                "config": {"column": col, "y_column": y_col, "categories": int(len(bar_data))},
                "data_columns": [col] + ([y_col] if y_col else []), "order": 1,
            }
        except Exception as e:
            logger.error(f"Failed to create bar chart: {str(e)}")
            raise

    # ============================================================
    # CHART CREATION - BOX PLOT CUSTOM ✓
    # ============================================================
    
    async def _create_box_plot_custom(
        self,
        df: pd.DataFrame,
        config: dict,
        dataset_id: int
    ) -> Dict[str, Any]:
        """Create custom box plot visualization."""
        try:
            columns = config.get('columns')
            title = config.get('title') or 'Box Plot Analysis'
            width = config.get('width', 800)
            height = config.get('height', 600)
            
            if not columns:
                raise ValueError("columns required for box plot")
            
            fig = go.Figure()
            
            for col in columns:
                fig.add_trace(go.Box(
                    y=df[col].dropna(), name=col, boxmean='sd',
                    hovertemplate='<b>%{fullData.name}</b><br>Value: %{y:.2f}<extra></extra>',
                ))
            
            fig.update_layout(
                title=title, yaxis_title="Value", template=self.PLOTLY_TEMPLATE,
                showlegend=True, height=height, width=width,
                plot_bgcolor='rgba(240, 240, 240, 0.5)',
            )
            
            chart_url = await self._save_plotly_chart(fig, dataset_id, "box_plot_custom")
            
            return {
                "dataset_id": dataset_id, "title": title,
                "chart_type": ChartType.BOX.value, "chart_url": chart_url,
                "config": {"columns": columns}, "data_columns": columns, "order": 1,
            }
        except Exception as e:
            logger.error(f"Failed to create box plot: {str(e)}")
            raise

    # ============================================================
    # CHART CREATION - AREA CHART (BONUS)
    # ============================================================
    
    async def _create_area_chart(
        self,
        df: pd.DataFrame,
        config: dict,
        dataset_id: int
    ) -> Dict[str, Any]:
        """Create area chart visualization."""
        try:
            x_col = config.get('x_column')
            y_col = config.get('y_column')
            title = config.get('title') or f'{y_col} over {x_col}'
            width = config.get('width', 800)
            height = config.get('height', 600)
            
            df_sorted = df.sort_values(x_col)
            
            fig = px.area(
                df_sorted, x=x_col, y=y_col, title=title,
                template=self.PLOTLY_TEMPLATE, width=width, height=height,
            )
            
            fig.update_traces(
                fillcolor='rgba(55, 126, 184, 0.5)',
                line=dict(color='rgba(55, 126, 184, 0.8)', width=2),
            )
            
            fig.update_layout(
                xaxis_title=x_col, yaxis_title=y_col, hovermode='x unified',
                plot_bgcolor='rgba(240, 240, 240, 0.5)',
            )
            
            chart_url = await self._save_plotly_chart(fig, dataset_id, f"area_{x_col}_{y_col}")
            
            return {
                "dataset_id": dataset_id, "title": title,
                "chart_type": ChartType.AREA.value, "chart_url": chart_url,
                "config": {"x_column": x_col, "y_column": y_col},
                "data_columns": [x_col, y_col], "order": 1,
            }
        except Exception as e:
            logger.error(f"Failed to create area chart: {str(e)}")
            raise

    # ============================================================
    # CHART CREATION - VIOLIN PLOT (BONUS)
    # ============================================================
    
    async def _create_violin_plot(
        self,
        df: pd.DataFrame,
        config: dict,
        dataset_id: int
    ) -> Dict[str, Any]:
        """Create violin plot visualization."""
        try:
            y_col = config.get('y_column')
            x_col = config.get('x_column')
            title = config.get('title') or f'Violin Plot of {y_col}'
            width = config.get('width', 800)
            height = config.get('height', 600)
            
            if x_col and x_col in df.columns:
                fig = px.violin(
                    df, x=x_col, y=y_col, title=title,
                    template=self.PLOTLY_TEMPLATE, width=width, height=height,
                    box=True, points="outliers",
                )
            else:
                fig = px.violin(
                    df, y=y_col, title=title,
                    template=self.PLOTLY_TEMPLATE, width=width, height=height,
                    box=True, points="outliers",
                )
            
            fig.update_layout(
                yaxis_title=y_col, plot_bgcolor='rgba(240, 240, 240, 0.5)',
            )
            
            chart_url = await self._save_plotly_chart(fig, dataset_id, f"violin_{y_col}")
            
            return {
                "dataset_id": dataset_id, "title": title,
                "chart_type": ChartType.VIOLIN.value, "chart_url": chart_url,
                "config": {"y_column": y_col, "x_column": x_col},
                "data_columns": [y_col] + ([x_col] if x_col else []), "order": 1,
            }
        except Exception as e:
            logger.error(f"Failed to create violin plot: {str(e)}")
            raise

    # ============================================================
    # CHART CREATION - TIME SERIES (BONUS)
    # ============================================================
    
    async def _create_time_series(
        self,
        df: pd.DataFrame,
        config: dict,
        dataset_id: int
    ) -> Dict[str, Any]:
        """Create time series visualization."""
        try:
            x_col = config.get('x_column')
            y_col = config.get('y_column')
            title = config.get('title') or f'{y_col} Time Series'
            width = config.get('width', 800)
            height = config.get('height', 600)
            
            df_copy = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_copy[x_col]):
                df_copy[x_col] = pd.to_datetime(df_copy[x_col])
            
            df_copy = df_copy.sort_values(x_col)
            
            fig = px.line(
                df_copy, x=x_col, y=y_col, title=title,
                template=self.PLOTLY_TEMPLATE, width=width, height=height,
                markers=True,
            )
            
            fig.update_traces(
                line=dict(color='rgba(55, 126, 184, 0.8)', width=2),
                marker=dict(size=4),
            )
            
            fig.update_xaxes(
                rangeslider_visible=False,
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="1w", step="day"),
                        dict(count=1, label="1m", step="month"),
                        dict(step="all", label="All"),
                    ])
                ),
                type="date",
            )
            
            fig.update_layout(
                xaxis_title=x_col, yaxis_title=y_col, hovermode='x unified',
                plot_bgcolor='rgba(240, 240, 240, 0.5)',
            )
            
            chart_url = await self._save_plotly_chart(fig, dataset_id, f"timeseries_{y_col}")
            
            return {
                "dataset_id": dataset_id, "title": title,
                "chart_type": ChartType.TIME_SERIES.value, "chart_url": chart_url,
                "config": {"x_column": x_col, "y_column": y_col},
                "data_columns": [x_col, y_col], "order": 1,
            }
        except Exception as e:
            logger.error(f"Failed to create time series: {str(e)}")
            raise
    
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
        """Create distribution chart for numerical column."""
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=df[column].dropna(), name='Distribution', nbinsx=50,
                marker_color='rgb(55, 126, 184)', opacity=0.7,
            ))
            
            mean_val = df[column].mean()
            median_val = df[column].median()
            
            fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {mean_val:.2f}", annotation_position="top")
            fig.add_vline(x=median_val, line_dash="dot", line_color="green",
                         annotation_text=f"Median: {median_val:.2f}", annotation_position="bottom")
            
            fig.update_layout(
                title=f"Distribution of {column}", xaxis_title=column,
                yaxis_title="Frequency", template=self.PLOTLY_TEMPLATE,
                hovermode='x unified', showlegend=True,
            )
            
            chart_url = await self._save_plotly_chart(fig, dataset_id, f"dist_{column}")
            
            return {
                "dataset_id": dataset_id, "title": f"Distribution of {column}",
                "chart_type": ChartType.DISTRIBUTION.value, "chart_url": chart_url,
                "config": {
                    "column": column, "mean": float(mean_val),
                    "median": float(median_val), "std": float(df[column].std()),
                },
                "data_columns": [column], "order": order,
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
        """Create correlation heatmap for numerical columns."""
        try:
            corr_matrix = df[columns].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                colorscale='RdBu', zmid=0, text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}', textfont={"size": 10},
                colorbar=dict(title="Correlation"),
            ))
            
            fig.update_layout(
                title="Correlation Matrix", xaxis_title="Features", yaxis_title="Features",
                template=self.PLOTLY_TEMPLATE, height=600, width=700,
            )
            
            chart_url = await self._save_plotly_chart(fig, dataset_id, "correlation_heatmap")
            
            return {
                "dataset_id": dataset_id, "title": "Correlation Matrix",
                "chart_type": ChartType.CORRELATION.value, "chart_url": chart_url,
                "config": {"columns": columns}, "data_columns": columns, "order": order,
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
        """Create bar chart for categorical column."""
        try:
            value_counts = df[column].value_counts().head(20)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=value_counts.index, y=value_counts.values,
                    marker_color='rgb(77, 175, 74)', text=value_counts.values,
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title=f"Distribution of {column}", xaxis_title=column,
                yaxis_title="Count", template=self.PLOTLY_TEMPLATE,
                xaxis_tickangle=-45,
            )
            
            chart_url = await self._save_plotly_chart(fig, dataset_id, f"cat_{column}")
            
            return {
                "dataset_id": dataset_id, "title": f"Distribution of {column}",
                "chart_type": ChartType.BAR.value, "chart_url": chart_url,
                "config": {"column": column, "categories": int(len(value_counts))},
                "data_columns": [column], "order": order,
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
        """Create scatter plot for two numerical columns."""
        try:
            correlation = df[[x_col, y_col]].corr().iloc[0, 1]
            
            if hue_col and hue_col in df.columns:
                fig = px.scatter(
                    df, x=x_col, y=y_col, color=hue_col, trendline="ols",
                    template=self.PLOTLY_TEMPLATE, title=f"{x_col} vs {y_col} (r={correlation:.3f})",
                )
            else:
                fig = px.scatter(
                    df, x=x_col, y=y_col, trendline="ols",
                    template=self.PLOTLY_TEMPLATE, title=f"{x_col} vs {y_col} (r={correlation:.3f})",
                )
                fig.update_traces(marker=dict(color='rgb(55, 126, 184)'))
            
            fig.update_layout(
                xaxis_title=x_col, yaxis_title=y_col, hovermode='closest',
            )
            
            chart_url = await self._save_plotly_chart(
                fig, dataset_id, f"scatter_{x_col}_{y_col}"
            )
            
            return {
                "dataset_id": dataset_id, "title": f"{x_col} vs {y_col}",
                "chart_type": ChartType.SCATTER.value, "chart_url": chart_url,
                "config": {
                    "x_column": x_col, "y_column": y_col,
                    "correlation": float(correlation), "hue_column": hue_col,
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
        """Create box plot for outlier detection."""
        try:
            fig = go.Figure()
            
            for col in columns:
                fig.add_trace(go.Box(
                    y=df[col], name=col, boxmean='sd',
                ))
            
            fig.update_layout(
                title="Box Plot - Outlier Detection", yaxis_title="Value",
                template=self.PLOTLY_TEMPLATE, showlegend=True, height=500,
            )
            
            chart_url = await self._save_plotly_chart(fig, dataset_id, "box_plot")
            
            return {
                "dataset_id": dataset_id, "title": "Box Plot - Outlier Detection",
                "chart_type": ChartType.BOX.value, "chart_url": chart_url,
                "config": {"columns": columns}, "data_columns": columns, "order": order,
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
        """Create custom chart with specified configuration."""
        dataset = self.db.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found",
            )
        
        try:
            df = self._read_dataframe(dataset.file_path, dataset.file_type)
            
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
            elif chart_type == ChartType.BAR:
                chart = await self._create_bar_chart(df, config, dataset_id)
            elif chart_type == ChartType.BOX:
                chart = await self._create_box_plot_custom(df, config, dataset_id)
            elif chart_type == ChartType.AREA:
                chart = await self._create_area_chart(df, config, dataset_id)
            elif chart_type == ChartType.VIOLIN:
                chart = await self._create_violin_plot(df, config, dataset_id)
            elif chart_type == ChartType.TIME_SERIES:
                chart = await self._create_time_series(df, config, dataset_id)
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Chart type {chart_type} not supported",
                )
            
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
        """Analyze data structure to recommend appropriate charts."""
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
        """Find interesting correlations for visualization."""
        interesting = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr = corr_matrix.iloc[i, j]
                
                if abs(corr) > threshold:
                    interesting.append((col1, col2, corr))
        
        interesting.sort(key=lambda x: abs(x[2]), reverse=True)
        return interesting[:top_n]
    
    def _get_data_quality_score(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive data quality metrics.
    
        Args:
            df: DataFrame to analyze
        
        Returns:
            Dictionary with quality metrics and overall score
        """
        try:
            total_cells = df.shape[0] * df.shape[1]
        
            # 1. Completeness: Percentage of non-null values
            null_count = df.isnull().sum().sum()
            completeness = ((total_cells - null_count) / total_cells) * 100 if total_cells > 0 else 0
        
            # 2. Uniqueness: Detect duplicate rows
            duplicate_count = df.duplicated().sum()
            uniqueness = ((len(df) - duplicate_count) / len(df)) * 100 if len(df) > 0 else 0
        
            # 3. Validity: Check data type consistency
            validity_issues = 0
            total_checks = 0
        
            for col in df.columns:
                total_checks += 1
                try:
                    # Check if numerical columns have valid numbers
                    if df[col].dtype in [np.float64, np.int64]:
                        invalid_nums = df[col].apply(
                            lambda x: pd.isna(x) or np.isinf(x) if pd.notna(x) else False
                        ).sum()
                        if invalid_nums > len(df) * 0.01:  # More than 1% invalid
                            validity_issues += 1
                        
                    # Check if categorical columns have reasonable cardinality
                    elif df[col].dtype == 'object':
                        unique_ratio = df[col].nunique() / len(df)
                        if unique_ratio > 0.95:  # Too many unique values (might be ID column)
                            validity_issues += 1
                        
                except Exception:
                    validity_issues += 1
        
            validity = ((total_checks - validity_issues) / total_checks) * 100 if total_checks > 0 else 100
        
            # 4. Consistency: Check for outliers in numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            outlier_ratio = 0
        
            if len(numerical_cols) > 0:
                outlier_counts = []
                for col in numerical_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                    outlier_counts.append(outliers)
            
                avg_outliers = np.mean(outlier_counts)
                outlier_ratio = (avg_outliers / len(df)) * 100 if len(df) > 0 else 0
        
            consistency = max(0, 100 - outlier_ratio * 2)  # Penalize outliers but not too harshly
        
            # Calculate weighted overall score
            overall_score = (
                completeness * 0.40 +  # 40% weight on completeness
                uniqueness * 0.25 +    # 25% weight on uniqueness
                validity * 0.25 +      # 25% weight on validity
                consistency * 0.10     # 10% weight on consistency
            )
        
            return {
                "overall_score": round(overall_score, 2),
                "completeness": round(completeness, 2),
                "uniqueness": round(uniqueness, 2),
                "validity": round(validity, 2),
                "consistency": round(consistency, 2),
                "null_count": int(null_count),
                "duplicate_count": int(duplicate_count),
                "outlier_ratio": round(outlier_ratio, 2),
                "total_cells": int(total_cells),
            }
        
        except Exception as e:
            logger.error(f"Data quality calculation failed: {str(e)}")
            # Return default values on error
            return {
                "overall_score": 0.0,
                "completeness": 0.0,
                "uniqueness": 0.0,
                "validity": 0.0,
                "consistency": 0.0,
                "null_count": 0,
                "duplicate_count": 0,
                "outlier_ratio": 0.0,
                "total_cells": 0,
            }


    def _generate_dashboard_stats(
        self,
        df: pd.DataFrame,
        dataset: Dataset,
        data_quality: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive statistics for dashboard display.
    
        Args:
            df: DataFrame being visualized
            dataset: Dataset model object
            data_quality: Pre-calculated quality metrics
        
        Returns:
            Dictionary with all dashboard statistics
        """
        try:
            # Calculate memory size
            memory_bytes = df.memory_usage(deep=True).sum()
            if memory_bytes < 1024:
                memory_size = f"{memory_bytes} B"
            elif memory_bytes < 1024 ** 2:
                memory_size = f"{memory_bytes / 1024:.1f} KB"
            elif memory_bytes < 1024 ** 3:
                memory_size = f"{memory_bytes / (1024 ** 2):.1f} MB"
            else:
                memory_size = f"{memory_bytes / (1024 ** 3):.2f} GB"
        
            # Count column types
            numerical_cols = len(df.select_dtypes(include=[np.number]).columns)
            categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
            datetime_cols = len(df.select_dtypes(include=['datetime64']).columns)
        
            # Format last updated timestamp
            if hasattr(dataset, 'updated_at') and dataset.updated_at:
                last_updated = dataset.updated_at.strftime('%Y-%m-%d %H:%M')
            elif hasattr(dataset, 'created_at') and dataset.created_at:
                last_updated = dataset.created_at.strftime('%Y-%m-%d %H:%M')
            else:
                last_updated = datetime.now().strftime('%Y-%m-%d %H:%M')

            return {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "numerical_cols": numerical_cols,
                "categorical_cols": categorical_cols,
                "datetime_cols": datetime_cols,
                "quality_score": data_quality["overall_score"],
                "memory_size": memory_size,
                "memory_bytes": memory_bytes,
                "last_updated": last_updated,
                "null_percentage": round((data_quality["null_count"] / data_quality["total_cells"]) * 100, 2) 
                    if data_quality["total_cells"] > 0 else 0,
                "duplicate_percentage": round((data_quality["duplicate_count"] / len(df)) * 100, 2) 
                    if len(df) > 0 else 0,
            }
        
        except Exception as e:
            logger.error(f"Failed to generate dashboard stats: {str(e)}")
            # Return minimal stats on error
            return {
                "total_rows": len(df) if df is not None else 0,
                "total_columns": len(df.columns) if df is not None else 0,
                "numerical_cols": 0,
                "categorical_cols": 0,
                "datetime_cols": 0,
                "quality_score": 0.0,
                "memory_size": "Unknown",
                "memory_bytes": 0,
                "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M'),
                "null_percentage": 0.0,
                "duplicate_percentage": 0.0,
            }

    # ============================================================
    # FILE OPERATIONS
    # ============================================================
    
    async def _save_plotly_chart(
        self,
        fig: go.Figure,
        dataset_id: int,
        chart_name: str
    ) -> str:
        """Save Plotly chart as HTML file."""
        dataset = self.db.get(Dataset, dataset_id)
        
        charts_dir = Path(settings.UPLOAD_DIR) / f"user_{dataset.owner_id}" / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{chart_name}_{timestamp}.html"
        filepath = charts_dir / filename
        
        fig.write_html(
            str(filepath),
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d'],
            }
        )
        
        logger.info(f"Saved chart to {filepath}")
        
        if settings.USE_S3:
            return self._upload_chart_to_s3(filepath, dataset)
        
        return str(filepath)
    
    def _upload_chart_to_s3(self, filepath: Path, dataset: Dataset) -> str:
        """Upload chart to S3 (placeholder)."""
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
    # CHART EXPORT FUNCTIONALITY
    # ============================================================

    async def export_chart(
        self,
        dataset_id: int,
        chart_id: int,
        export_format: str = "png",
        width: int = 1200,
        height: int = 900,
        scale: float = 2.0
    ) -> Dict[str, Any]:
        """
        Export visualization in multiple formats (PNG, SVG, PDF, HTML, JPEG).
    
        Args:
            dataset_id: Dataset ID
            chart_id: Chart/Visualization ID
            export_format: Export format ('png', 'svg', 'pdf', 'html', 'jpeg')
            width: Export width in pixels
            height: Export height in pixels
            scale: Scale factor for raster formats (default 2.0 for high DPI)
    
        Returns:
            Export result with file path/URL and metadata
    
        Raises:
            HTTPException: If export fails
        """
        try:
            # Validate dataset exists
            dataset = self.db.get(Dataset, dataset_id)
            if not dataset:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Dataset not found",
                )
        
            # Validate export format
            valid_formats = ['png', 'svg', 'pdf', 'html', 'jpeg']
            export_format = export_format.lower()
            if export_format not in valid_formats:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid export format. Supported: {', '.join(valid_formats)}"
                )
        
            logger.info(
                f"Exporting chart {chart_id} from dataset {dataset_id} "
                f"as {export_format.upper()} (size: {width}x{height})"
            )
        
            # Recreate chart based on stored configuration
            # First, get the dataset
            df = self._read_dataframe(dataset.file_path, dataset.file_type)
        
            # Route to appropriate export method
            if export_format == "png":
                result = await self._export_as_png(
                    df, chart_id, width, height, scale, dataset_id
                )
            elif export_format == "svg":
                result = await self._export_as_svg(
                    df, chart_id, width, height, dataset_id
                )
            elif export_format == "pdf":
                result = await self._export_as_pdf(
                    df, chart_id, width, height, dataset_id
                )
            elif export_format == "html":
                result = await self._export_as_html(
                    df, chart_id, width, height, dataset_id
                )
            elif export_format == "jpeg":
                result = await self._export_as_jpeg(
                    df, chart_id, width, height, scale, dataset_id
                )
        
            logger.info(f"Successfully exported chart {chart_id} as {export_format.upper()}")
        
            return result
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Chart export failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Chart export failed: {str(e)}"
            ) from e

# ============================================================
# EXPORT - PNG FORMAT
# ============================================================

    async def _export_as_png(
        self,
        df: pd.DataFrame,
        chart_id: int,
        width: int,
        height: int,
        scale: float,
        dataset_id: int
    ) -> Dict[str, Any]:
        """Export chart as PNG (raster format)."""
        try:
            # Create sample histogram for demonstration
            fig = self._create_export_figure(df, chart_id, width, height)
        
            # Create export directory
            export_dir = Path(settings.UPLOAD_DIR) / "exports" / f"dataset_{dataset_id}"
            export_dir.mkdir(parents=True, exist_ok=True)
        
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chart_{chart_id}_{timestamp}.png"
            filepath = export_dir / filename
        
            # Export as PNG with high quality
            fig.write_image(
                str(filepath),
                width=width,
                height=height,
                scale=scale,
            )
        
        # Get file size
            file_size = filepath.stat().st_size
        
            logger.info(f"Exported PNG to {filepath} ({file_size} bytes)")
        
            return {
                "success": True,
                "format": "png",
                "file_path": str(filepath),
                "file_name": filename,
                "file_size": file_size,
                "width": width,
                "height": height,
                "mime_type": "image/png",
                "timestamp": timestamp,
                "url": f"/api/v1/exports/download/{filename}",
            }
        
        except Exception as e:
            logger.error(f"PNG export failed: {str(e)}")
            raise

# ============================================================
# EXPORT - SVG FORMAT
# ============================================================

    async def _export_as_svg(
        self,
        df: pd.DataFrame,
        chart_id: int,
        width: int,
        height: int,
        dataset_id: int
    ) -> Dict[str, Any]:
        """Export chart as SVG (scalable vector format)."""
        try:
            fig = self._create_export_figure(df, chart_id, width, height)
        
            export_dir = Path(settings.UPLOAD_DIR) / "exports" / f"dataset_{dataset_id}"
            export_dir.mkdir(parents=True, exist_ok=True)
        
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chart_{chart_id}_{timestamp}.svg"
            filepath = export_dir / filename
        
            # Export as SVG (vector format - no quality loss on scaling)
            fig.write_image(
                str(filepath),
                format="svg",
                width=width,
                height=height,
            )
        
            file_size = filepath.stat().st_size
        
            logger.info(f"Exported SVG to {filepath} ({file_size} bytes)")
        
            return {
                "success": True,
                "format": "svg",
                "file_path": str(filepath),
                "file_name": filename,
                "file_size": file_size,
                "width": width,
                "height": height,
                "mime_type": "image/svg+xml",
                "timestamp": timestamp,
                "url": f"/api/v1/exports/download/{filename}",
                "scalable": True,
            }
        
        except Exception as e:
            logger.error(f"SVG export failed: {str(e)}")
            raise

# ============================================================
# EXPORT - PDF FORMAT
# ============================================================

    async def _export_as_pdf(
        self,
        df: pd.DataFrame,
        chart_id: int,
        width: int,
        height: int,
        dataset_id: int
    ) -> Dict[str, Any]:
        """Export chart as PDF (print-friendly format)."""
        try:
            fig = self._create_export_figure(df, chart_id, width, height)
        
            export_dir = Path(settings.UPLOAD_DIR) / "exports" / f"dataset_{dataset_id}"
            export_dir.mkdir(parents=True, exist_ok=True)
        
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chart_{chart_id}_{timestamp}.pdf"
            filepath = export_dir / filename
        
            # Export as PDF with optimized settings
            fig.write_image(
                str(filepath),
                format="pdf",
                width=width,
                height=height,
            )
        
            file_size = filepath.stat().st_size

            logger.info(f"Exported PDF to {filepath} ({file_size} bytes)")
        
            return {
                "success": True,
                "format": "pdf",
                "file_path": str(filepath),
                "file_name": filename,
                "file_size": file_size,
                "width": width,
                "height": height,
                "mime_type": "application/pdf",
                "timestamp": timestamp,
                "url": f"/api/v1/exports/download/{filename}",
                "printable": True,
            }
        
        except Exception as e:
            logger.error(f"PDF export failed: {str(e)}")
            raise

# ============================================================
# EXPORT - JPEG FORMAT
# ============================================================

    async def _export_as_jpeg(
        self,
        df: pd.DataFrame,
        chart_id: int,
        width: int,
        height: int,
        scale: float,
        dataset_id: int
    ) -> Dict[str, Any]:
        """Export chart as JPEG (compressed format)."""
        try:
            fig = self._create_export_figure(df, chart_id, width, height)
        
            export_dir = Path(settings.UPLOAD_DIR) / "exports" / f"dataset_{dataset_id}"
            export_dir.mkdir(parents=True, exist_ok=True)
        
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chart_{chart_id}_{timestamp}.jpg"
            filepath = export_dir / filename
        
            # Export as JPEG with quality optimization
            fig.write_image(
                str(filepath),
                format="jpeg",
                width=width,
                height=height,
                scale=scale,
                quality=95,  # High quality compression
            )

            file_size = filepath.stat().st_size
        
            logger.info(f"Exported JPEG to {filepath} ({file_size} bytes)")
        
            return {
                "success": True,
                "format": "jpeg",
                "file_path": str(filepath),
                "file_name": filename,
                "file_size": file_size,
                "width": width,
                "height": height,
                "mime_type": "image/jpeg",
                "timestamp": timestamp,
                "url": f"/api/v1/exports/download/{filename}",
                "quality": 95,
            }
        
        except Exception as e:
            logger.error(f"JPEG export failed: {str(e)}")
            raise

# ============================================================
# EXPORT - HTML FORMAT
# ============================================================

    async def _export_as_html(
        self,
        df: pd.DataFrame,
        chart_id: int,
        width: int,
        height: int,
        dataset_id: int
    ) -> Dict[str, Any]:
        """Export chart as standalone HTML (interactive)."""
        try:
            fig = self._create_export_figure(df, chart_id, width, height)
        
            export_dir = Path(settings.UPLOAD_DIR) / "exports" / f"dataset_{dataset_id}"
            export_dir.mkdir(parents=True, exist_ok=True)
        
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chart_{chart_id}_{timestamp}.html"
            filepath = export_dir / filename
        
            # Export as standalone HTML with enhanced styling
            html_string = fig.to_html(
                include_plotlyjs='cdn',
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'chart_{chart_id}',
                        'height': height,
                        'width': width,
                        'scale': 1
                    }
                }
            )
        
            # Enhance HTML with CSS styling and metadata
            enhanced_html = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta name="description" content="Exported Data Visualization">
        <title>Data Visualization Export - Chart {chart_id}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 30px;
            }}
            .header {{
                border-bottom: 3px solid #667eea;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .header h1 {{
                margin: 0;
                color: #333;
                font-size: 28px;
            }}
            .metadata {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
                font-size: 14px;
                color: #666;
            }}
            .metadata-item {{
                background: #f5f5f5;
                padding: 10px 15px;
                border-radius: 6px;
                border-left: 4px solid #667eea;
            }}
            .metadata-label {{
                font-weight: 600;
                color: #333;
            }}
            .plot-container {{
                margin-top: 30px;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }}
            .footer {{
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #e0e0e0;
                font-size: 12px;
                color: #999;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Data Visualization Export</h1>
            </div>
        
            <div class="metadata">
                <div class="metadata-item">
                    <div class="metadata-label">Chart ID</div>
                    <div>{chart_id}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Export Date</div>
                    <div>{timestamp}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Format</div>
                    <div>Interactive HTML</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Dimensions</div>
                    <div>{width}x{height}px</div>
                </div>
            </div>
        
            <div class="plot-container">
                {html_string}
            </div>
        
            <div class="footer">
                <p>Generated by DataSense Visualization Service | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </div>
    </body>
    </html>
    """
        
            # Write enhanced HTML to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(enhanced_html)
        
            file_size = filepath.stat().st_size
        
            logger.info(f"Exported HTML to {filepath} ({file_size} bytes)")
        
            return {
                "success": True,
                "format": "html",
                "file_path": str(filepath),
                "file_name": filename,
                "file_size": file_size,
                "width": width,
                "height": height,
                "mime_type": "text/html",
                "timestamp": timestamp,
                "url": f"/api/v1/exports/download/{filename}",
                "interactive": True,
                "responsive": True,
            }
        
        except Exception as e:
            logger.error(f"HTML export failed: {str(e)}")
            raise

# ============================================================
# EXPORT - BATCH EXPORT
# ============================================================

    async def export_multiple_charts(
        self,
        dataset_id: int,
        chart_ids: List[int],
        export_format: str = "png",
        width: int = 1200,
        height: int = 900
    ) -> Dict[str, Any]:
        """
        Export multiple charts in a single batch operation.
    
        Args:
            dataset_id: Dataset ID
            chart_ids: List of chart IDs to export
            export_format: Export format (png, svg, pdf, html, jpeg)
            width: Export width
            height: Export height
    
        Returns:
            Batch export result with all files
        """
        try:
            logger.info(f"Batch exporting {len(chart_ids)} charts as {export_format.upper()}")
        
            results = []
            failed = []
        
            for chart_id in chart_ids:
                try:
                    result = await self.export_chart(
                    dataset_id, chart_id, export_format, width, height
                    )
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to export chart {chart_id}: {str(e)}")
                    failed.append({"chart_id": chart_id, "error": str(e)})
        
            return {
                "success": len(results) > 0,
                "total_charts": len(chart_ids),
                "exported": len(results),
                "failed": len(failed),
                "format": export_format,
                "exports": results,
                "failures": failed,
                "timestamp": datetime.now().isoformat(),
            }
        
        except Exception as e:
            logger.error(f"Batch export failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch export failed: {str(e)}"
            ) from e
    async def generate_dashboard(
        self,
        dataset_id: int,
        regenerate: bool = False
    ) -> str:
        """
        Generate interactive HTML dashboard with multiple visualizations.
    
        Args:
            dataset_id: Dataset ID to create dashboard for
            regenerate: Force regenerate all charts if True
        
        Returns:
            Complete HTML dashboard string with embedded charts
        
        Raises:
            HTTPException: If dataset not found or generation fails
        """
        try:
            # Validate dataset exists and is ready
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
        
            logger.info(f"Generating dashboard for dataset {dataset_id}")
        
            # Load dataframe for analysis
            df = self._read_dataframe(dataset.file_path, dataset.file_type)
        
            # Check if visualizations exist
            existing_viz = self.db.query(DatasetVisualization).filter(
                DatasetVisualization.dataset_id == dataset_id
            ).all()
        
            # Generate charts if none exist or regenerate requested
            if not existing_viz or regenerate:
                logger.info(f"Generating automated visualizations for dataset {dataset_id}")
                await self.generate_automated_visualizations(dataset_id, max_charts=10)
            
                # Refresh visualizations after generation
                existing_viz = self.db.query(DatasetVisualization).filter(
                    DatasetVisualization.dataset_id == dataset_id
                ).order_by(DatasetVisualization.order).all()
        
            # Calculate data quality score
            data_quality = self._get_data_quality_score(df)
        
            # Generate dashboard statistics
            stats = self._generate_dashboard_stats(df, dataset, data_quality)
        
            # Embed all charts into HTML
            charts_html = self._embed_charts_in_dashboard(existing_viz, dataset_id)
        
            # Build complete dashboard HTML
            dashboard_html = self._build_dashboard_html(
                dataset=dataset,
                stats=stats,
                charts_html=charts_html,
                total_charts=len(existing_viz)
            )
        
            logger.info(f"Successfully generated dashboard with {len(existing_viz)} charts")
        
            return dashboard_html
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Dashboard generation failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Dashboard generation failed: {str(e)}"
            ) from e


    def _build_dashboard_html(
        self,
        dataset: Dataset,
        stats: Dict[str, Any],
        charts_html: str,
        total_charts: int
    ) -> str:
        """Build the complete dashboard HTML structure."""
    
        return f"""<!DOCTYPE html>
        <html lang="en">
            <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Dataset Dashboard - {dataset.file_name}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }}
            .container {{ 
                max-width: 1400px; 
                margin: 0 auto; 
            }}
            .header {{ 
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                color: white; 
                padding: 30px; 
                border-radius: 12px; 
                margin-bottom: 25px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }}
            .header h1 {{ 
                margin: 0 0 10px 0; 
                font-size: 32px;
                font-weight: 600;
            }}
            .header-subtitle {{
                font-size: 14px;
                opacity: 0.9;
                margin-top: 8px;
            }}
            .stats {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); 
                gap: 20px; 
                margin-bottom: 25px; 
            }}
            .stat-card {{ 
                background: white; 
                padding: 25px; 
                border-radius: 12px; 
                box-shadow: 0 4px 15px rgba(0,0,0,0.15);
                border-left: 5px solid #3498db;
                transition: transform 0.2s, box-shadow 0.2s;
            }}
            .stat-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.2);
            }}
            .stat-label {{ 
                color: #7f8c8d; 
                font-size: 13px; 
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 8px;
            }}
            .stat-value {{ 
                font-size: 28px; 
                font-weight: bold; 
                color: #2c3e50;
                margin: 5px 0;
            }}
            .stat-subvalue {{
                font-size: 12px;
                color: #95a5a6;
                margin-top: 5px;
            }}
            .quality-bar {{
                width: 100%;
                height: 8px;
                background: #ecf0f1;
                border-radius: 4px;
                margin-top: 10px;
                overflow: hidden;
            }}
            .quality-fill {{
                height: 100%;
                background: linear-gradient(90deg, #3498db 0%, #2ecc71 100%);
                border-radius: 4px;
                transition: width 0.5s ease;
            }}
            .grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); 
                gap: 25px; 
            }}
            .chart {{ 
                background: white; 
                padding: 25px; 
                border-radius: 12px; 
                box-shadow: 0 4px 15px rgba(0,0,0,0.15);
                transition: transform 0.2s, box-shadow 0.2s;
            }}
            .chart:hover {{
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.2);
            }}
            .chart-title {{
                font-size: 18px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 2px solid #ecf0f1;
            }}
            .chart-container {{
                width: 100%;
                overflow: hidden;
            }}
            .chart-container iframe {{
                width: 100%;
                border: none;
                min-height: 400px;
            }}
            .no-charts {{
                background: white;
                padding: 40px;
                border-radius: 12px;
                text-align: center;
                box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            }}
            .loading-spinner {{
                display: inline-block;
                width: 40px;
                height: 40px;
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            .footer {{
                margin-top: 30px;
                padding: 20px;
                background: rgba(255,255,255,0.1);
                border-radius: 12px;
                text-align: center;
                color: white;
                font-size: 13px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 {dataset.file_name} Dashboard</h1>
                <div class="header-subtitle">
                    Dataset ID: {dataset.id} | Status: {dataset.status.value if hasattr(dataset.status, 'value') else dataset.status} | 
                    Charts: {total_charts}
                </div>
            </div>
        
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-label">📋 Total Rows</div>
                    <div class="stat-value">{stats['total_rows']:,}</div>
                    <div class="stat-subvalue">Data records</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">📊 Total Columns</div>
                    <div class="stat-value">{stats['total_columns']}</div>
                    <div class="stat-subvalue">{stats['numerical_cols']} numerical, {stats['categorical_cols']} categorical</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">✅ Data Quality</div>
                    <div class="stat-value">{stats['quality_score']:.1f}%</div>
                    <div class="quality-bar">
                        <div class="quality-fill" style="width: {stats['quality_score']}%"></div>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">💾 Memory Size</div>
                    <div class="stat-value">{stats['memory_size']}</div>
                    <div class="stat-subvalue">Dataset footprint</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">🕒 Last Updated</div>
                    <div class="stat-value" style="font-size: 18px;">{stats['last_updated']}</div>
                    <div class="stat-subvalue">Processing completed</div>
                </div>
            </div>
        
            <div class="grid">
                {charts_html}
            </div>
        
            <div class="footer">
                Generated by DataSense Visualization Service | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
    </body>
    </html>
    """


# ============================================================
# HELPER - CREATE EXPORT FIGURE
# ============================================================

    def _create_export_figure(
        self,
        df: pd.DataFrame,
        chart_id: int,
        width: int,
        height: int
    ) -> go.Figure:
        """Create a Plotly figure for export."""
        # Example: Create a sample histogram
        fig = go.Figure()
    
        # Get first numerical column
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            col = numerical_cols[0]
            fig.add_trace(go.Histogram(
                x=df[col].dropna(),
                nbinsx=50,
                marker_color='rgba(55, 126, 184, 0.7)',
                name='Distribution',
            ))
        
            fig.update_layout(
                title=f'Chart Export - ID: {chart_id}',
                xaxis_title=col,
                yaxis_title='Frequency',
                width=width,
                height=height,
                template='plotly_white',
                hovermode='x unified',
            )
    
        return fig
    def _embed_charts_in_dashboard(
        self,
        visualizations: List[DatasetVisualization],
        dataset_id: int
    ) -> str:
        """
        Embed saved chart HTML files into dashboard grid.
    
        Args:
            visualizations: List of DatasetVisualization objects
            dataset_id: Dataset ID for error handling
        
        Returns:
            HTML string with all embedded charts
        """
        if not visualizations:
            return """
            <div class="no-charts">
                <div class="loading-spinner"></div>
                <h3 style="margin-top: 20px; color: #7f8c8d;">No Visualizations Available</h3>
                <p style="color: #95a5a6;">Generate visualizations from the API to populate this dashboard.</p>
            </div>
            """
    
        charts_html = []
    
        for viz in visualizations:
            try:
                chart_path = Path(viz.chart_url)
            
                if not chart_path.exists():
                    logger.warning(f"Chart file not found: {chart_path}")
                    charts_html.append(f"""
                    <div class="chart">
                        <div class="chart-title">{viz.title}</div>
                        <div class="chart-container">
                            <p style="color: #e74c3c; text-align: center; padding: 40px;">
                                ⚠️ Chart file not found
                            </p>
                        </div>
                    </div>
                    """)
                    continue
            
                # Read chart HTML content
                with open(chart_path, 'r', encoding='utf-8') as f:
                    chart_html = f.read()
            
                # Extract only the Plotly div and script (remove full HTML wrapper)
                # This prevents duplicate HTML structure
                chart_content = self._extract_plotly_content(chart_html)
            
                charts_html.append(f"""
                <div class="chart">
                    <div class="chart-title">{viz.title}</div>
                    <div class="chart-container">
                        {chart_content}
                    </div>
                </div>
                """)
            
            except Exception as e:
                logger.error(f"Failed to embed chart {viz.id}: {str(e)}")
                charts_html.append(f"""
                <div class="chart">
                    <div class="chart-title">{viz.title}</div>
                    <div class="chart-container">
                        <p style="color: #e74c3c; text-align: center; padding: 40px;">
                            ❌ Error loading chart: {str(e)}
                        </p>
                    </div>
                </div>
                """)
    
        return "\n".join(charts_html)


    def _extract_plotly_content(self, full_html: str) -> str:
        """
        Extract only the Plotly div and script from full HTML.
    
        Args:
            full_html: Complete HTML string with Plotly chart
        
        Returns:
            Cleaned HTML with just the chart div and script
        """
        try:
            # Try to extract just the plot div
            import re
        
            # Find the Plotly div
            div_match = re.search(r'<div id=["\']([^"\']+)["\'][^>]*>.*?</div>', full_html, re.DOTALL)
        
            # Find the Plotly script
            script_match = re.search(
                r'<script[^>]*>.*?Plotly\.newPlot.*?</script>',
                full_html,
                re.DOTALL
            )
        
            if div_match and script_match:
                return div_match.group(0) + "\n" + script_match.group(0)
            else:
                # Fallback: use iframe embedding
                return full_html
            
        except Exception as e:
            logger.warning(f"Could not extract Plotly content: {str(e)}, using full HTML")
            return full_html


# ============================================================
# HELPER - GET EXPORT STATUS
# ============================================================

    async def get_export_status(
        self,
        export_id: str
    ) -> Dict[str, Any]:
        """Get status of an export operation."""
        try:
            export_dir = Path(settings.UPLOAD_DIR) / "exports"
        
            # List all exports
            if export_dir.exists():
                exports = []
                for file in export_dir.rglob("*"):
                    if file.is_file():
                        exports.append({
                            "filename": file.name,
                            "path": str(file),
                            "size": file.stat().st_size,
                            "created": datetime.fromtimestamp(file.stat().st_ctime).isoformat(),
                        })
            
                return {
                    "success": True,
                    "total_exports": len(exports),
                    "exports": sorted(exports, key=lambda x: x['created'], reverse=True)[:10],
                }
            else:
                return {
                    "success": True,
                    "total_exports": 0,
                    "exports": [],
                }
            
        except Exception as e:
            logger.error(f"Failed to get export status: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get export status: {str(e)}"
            ) from e

# ============================================================
# HELPER - DOWNLOAD EXPORT
# ============================================================

    async def download_export(
        self,
        filename: str
    ) -> Dict[str, Any]:
        """
        Download exported file.
    
        Args:
            filename: Name of exported file
    
        Returns:
            File path for download
        """
        try:
            export_dir = Path(settings.UPLOAD_DIR) / "exports"
        
            # Security: Prevent directory traversal
            if ".." in filename or "/" in filename or "\\" in filename:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid filename"
                )
        
            filepath = export_dir / filename
        
            if not filepath.exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Export file not found"
                )
        
            return {
                "success": True,
                "filename": filename,
                "path": str(filepath),
                "size": filepath.stat().st_size,
                "mime_type": self._get_mime_type(filepath.suffix),
            }
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Download failed: {str(e)}"
            ) from e

# ============================================================
# HELPER - GET MIME TYPE
# ============================================================

    def _get_mime_type(self, file_extension: str) -> str:
        """Get MIME type for file extension."""
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".pdf": "application/pdf",
            ".html": "text/html",
        }
        return mime_types.get(file_extension.lower(), "application/octet-stream")

    # ============================================================
    # DATABASE OPERATIONS
    # ============================================================
    
    def _save_visualizations(
        self,
        charts: List[Dict[str, Any]]
    ) -> List[DatasetVisualization]:
        """Save visualizations to database."""
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
    """Dependency for injecting VisualizationService."""
    return VisualizationService(db)
