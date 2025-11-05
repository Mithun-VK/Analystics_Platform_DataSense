"""
Exploratory Data Analysis (EDA) Service.

Handles automated EDA generation including statistical analysis,
data profiling, correlation analysis, distribution visualization,
and comprehensive HTML report generation.
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats  # ✅ FIXED: Renamed import
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.dataset import Dataset, DatasetStatus, DatasetStatistics
from app.database import get_db

logger = logging.getLogger(__name__)


class EDAService:
    """
    EDA service for automated exploratory data analysis.
    
    Implements:
    - Automated profiling with ydata-profiling
    - Statistical summary generation
    - Correlation analysis
    - Distribution analysis
    - Data quality assessment
    - Interactive HTML report generation
    - Custom statistical tests
    """
    
    def __init__(self, db: Session):
        """
        Initialize EDA service.
        
        Args:
            db: SQLAlchemy database session
        """
        self.db = db
    
    # ============================================================
    # MAIN EDA GENERATION (ASYNC)
    # ============================================================
    
    async def generate_eda_report(
        self,
        dataset_id: int,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive EDA report for dataset (async).
        
        Args:
            dataset_id: Dataset ID
            config: EDA configuration dict with keys:
                - minimal_report: bool (default: False)
                - sample_size: Optional[int] (default: None)
                - generate_correlations: bool (default: True)
                - generate_distributions: bool (default: True)
            
        Returns:
            Dictionary with EDA results and report URL
            
        Raises:
            HTTPException: If dataset not found or generation fails
        """
        # Extract config values with defaults
        minimal_report = config.get('minimal_report', False)
        sample_size = config.get('sample_size', None)
        generate_correlations = config.get('generate_correlations', True)
        generate_distributions = config.get('generate_distributions', True)
        
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
                detail=f"Dataset is not ready. Current status: {dataset.status}",
            )
        
        # Update status
        dataset.status = DatasetStatus.ANALYZING
        self.db.commit()
        
        try:
            logger.info(
                f"Starting EDA generation for dataset {dataset_id} "
                f"(minimal={minimal_report}, sample_size={sample_size})"
            )
            
            # Simulate async processing
            await asyncio.sleep(0.1)
            
            # Load data (run in executor to avoid blocking)
            df = await asyncio.to_thread(
                self._read_dataframe,
                dataset.file_path,
                dataset.file_type
            )
            
            logger.info(f"Loaded dataset with shape: {df.shape}")
            
            # Apply sampling for large datasets
            actual_sample_size = len(df)
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
                actual_sample_size = sample_size
                logger.info(f"Sampled {sample_size} rows from {dataset.row_count}")
            
            # Generate statistical summary (run in executor)
            statistics = await asyncio.to_thread(
                self._generate_statistics,
                df,
                generate_correlations,
                generate_distributions
            )
            
            logger.info(f"Statistics generated for dataset {dataset_id}")
            
            # Generate profile report if not minimal
            report_url = None
            if not minimal_report:
                try:
                    report_url = await asyncio.to_thread(
                        self._generate_profile_report,
                        df,
                        dataset,
                        minimal_report
                    )
                except Exception as e:
                    logger.warning(f"Profile report generation failed: {e}")
                    # Continue without profile report
            
            # Save statistics to database
            await asyncio.to_thread(
                self._save_statistics,
                dataset_id,
                statistics
            )
            
            logger.info(f"Statistics saved to database for dataset {dataset_id}")
            
            # Update dataset
            dataset.eda_report_url = report_url
            dataset.eda_report_generated_at = datetime.now(timezone.utc)
            dataset.status = DatasetStatus.COMPLETED
            self.db.commit()
            
            logger.info(f"EDA generation completed for dataset {dataset_id}")
            
            # Build response
            response = {
                "dataset_id": dataset_id,
                "status": "completed",
                "message": "EDA report generated successfully",
                "config_used": {
                    "minimal_report": minimal_report,
                    "sample_size": actual_sample_size,
                    "correlations_included": generate_correlations,
                    "distributions_included": generate_distributions,
                },
                "report_url": report_url,
                "statistics": statistics,
                "generated_at": dataset.eda_report_generated_at.isoformat(),
            }
            
            # Add recommendations
            response["recommendations"] = self._generate_recommendations(statistics)
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            dataset.status = DatasetStatus.FAILED
            dataset.processing_error = f"EDA generation failed: {str(e)}"
            self.db.commit()
            
            logger.error(f"EDA generation failed for dataset {dataset_id}: {str(e)}", exc_info=True)
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"EDA generation failed: {str(e)}",
            ) from e
    
    # ============================================================
    # PROFILE REPORT GENERATION
    # ============================================================
    
    def _generate_profile_report(
        self,
        df: pd.DataFrame,
        dataset: Dataset,
        minimal: bool = False
    ) -> str:
        """
        Generate ydata-profiling HTML report.
        
        Args:
            df: DataFrame to profile
            dataset: Dataset instance
            minimal: Generate minimal report for faster processing
            
        Returns:
            URL/path to generated HTML report
        """
        try:
            from ydata_profiling import ProfileReport
            
            # Configure profiling
            profile_config = {
                "title": f"EDA Report: {dataset.name}",
                "minimal": minimal,
                "explorative": not minimal,
                "dark_mode": False,
            }
            
            # Additional configuration for faster processing
            if minimal:
                profile_config.update({
                    "correlations": None,
                    "missing_diagrams": None,
                    "duplicates": None,
                    "interactions": None,
                })
            
            # Generate profile report
            logger.info(f"Generating {'minimal' if minimal else 'full'} profile report")
            
            profile = ProfileReport(df, **profile_config)
            
            # Save report
            report_path = self._get_report_path(dataset)
            profile.to_file(report_path)
            
            logger.info(f"Profile report saved to {report_path}")
            
            # In production, upload to S3 and return URL
            if hasattr(settings, 'USE_S3') and settings.USE_S3:
                report_url = self._upload_report_to_s3(report_path, dataset)
                return report_url
            
            return str(report_path)
            
        except ImportError:
            logger.warning("ydata-profiling not installed, skipping HTML report")
            return None
        except Exception as e:
            logger.error(f"Profile report generation failed: {str(e)}")
            raise
    
    def _get_report_path(self, dataset: Dataset) -> Path:
        """
        Generate path for EDA report.
        
        Args:
            dataset: Dataset instance
            
        Returns:
            Path to report file
        """
        reports_dir = Path(settings.UPLOAD_DIR) / f"user_{dataset.owner_id}" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"eda_report_{dataset.id}_{timestamp}.html"
        
        return reports_dir / report_filename
    
    def _upload_report_to_s3(self, report_path: Path, dataset: Dataset) -> str:
        """
        Upload report to S3 (placeholder for production).
        
        Args:
            report_path: Local report path
            dataset: Dataset instance
            
        Returns:
            S3 URL
        """
        # TODO: Implement S3 upload
        # from app.utils.s3_utils import upload_file_to_s3
        # s3_key = f"reports/user_{dataset.owner_id}/eda_report_{dataset.id}.html"
        # return upload_file_to_s3(report_path, s3_key)
        
        return str(report_path)
    
    # ============================================================
    # STATISTICAL ANALYSIS
    # ============================================================
    
    def _generate_statistics(
        self,
        df: pd.DataFrame,
        generate_correlations: bool = True,
        generate_distributions: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive statistical summary.
        
        Args:
            df: DataFrame
            generate_correlations: Whether to calculate correlations
            generate_distributions: Whether to analyze distributions
            
        Returns:
            Dictionary with statistics
        """
        statistics = {
            "overview": self._get_overview_statistics(df),
            "numerical": self._get_numerical_statistics(df),
            "categorical": self._get_categorical_statistics(df),
        }
        
        # Add correlation analysis if requested
        if generate_correlations:
            try:
                statistics["correlations"] = self._calculate_correlations(df)
            except Exception as e:
                logger.warning(f"Correlation calculation failed: {e}")
                statistics["correlations"] = {}
        
        # Add distribution analysis if requested
        if generate_distributions:
            try:
                statistics["distributions"] = self._analyze_distributions(df)
            except Exception as e:
                logger.warning(f"Distribution analysis failed: {e}")
                statistics["distributions"] = {}
        
        # Add data quality score
        statistics["data_quality"] = self._calculate_data_quality_score(statistics)
        
        return statistics
    
    def _get_overview_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get overview statistics for dataset."""
        return {
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            "total_missing": int(df.isnull().sum().sum()),
            "missing_percentage": round((df.isnull().sum().sum() / df.size) * 100, 2) if df.size > 0 else 0,
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_percentage": round((df.duplicated().sum() / len(df)) * 100, 2) if len(df) > 0 else 0,
            "column_types": {str(k): int(v) for k, v in df.dtypes.value_counts().to_dict().items()},
            "numerical_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns),
        }
    
    def _get_numerical_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics for numerical columns."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return {}
        
        stats_dict = {}  # ✅ FIXED: Renamed to avoid collision with scipy.stats
        
        for col in numerical_cols:
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            try:
                # Basic statistics
                stats_dict[col] = {
                    "count": int(col_data.count()),
                    "missing": int(df[col].isnull().sum()),
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "median": float(col_data.median()),
                    "q1": float(col_data.quantile(0.25)),
                    "q3": float(col_data.quantile(0.75)),
                    "iqr": float(col_data.quantile(0.75) - col_data.quantile(0.25)),
                    "skewness": float(scipy_stats.skew(col_data)),  # ✅ FIXED: Using scipy_stats
                    "kurtosis": float(scipy_stats.kurtosis(col_data)),  # ✅ FIXED: Using scipy_stats
                    "variance": float(col_data.var()),
                    "range": float(col_data.max() - col_data.min()),
                    "coefficient_variation": float((col_data.std() / col_data.mean()) * 100) if col_data.mean() != 0 else 0,
                }
                
                # Zero values
                stats_dict[col]["zeros"] = int((col_data == 0).sum())
                stats_dict[col]["zeros_percentage"] = round((stats_dict[col]["zeros"] / len(col_data)) * 100, 2) if len(col_data) > 0 else 0
                
                # Outlier detection using IQR method
                Q1 = stats_dict[col]["q1"]
                Q3 = stats_dict[col]["q3"]
                IQR = stats_dict[col]["iqr"]
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                
                stats_dict[col]["outliers"] = int(len(outliers))
                stats_dict[col]["outliers_percentage"] = round((len(outliers) / len(col_data)) * 100, 2) if len(col_data) > 0 else 0
            
            except Exception as e:
                logger.warning(f"Error calculating stats for column {col}: {str(e)}")
                stats_dict[col] = {"error": str(e)}
        
        return stats_dict
    
    def _get_categorical_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics for categorical columns."""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return {}
        
        stats_dict = {}
        
        for col in categorical_cols:
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            try:
                # Value counts
                value_counts = col_data.value_counts()
                
                stats_dict[col] = {
                    "count": int(col_data.count()),
                    "missing": int(df[col].isnull().sum()),
                    "unique": int(col_data.nunique()),
                    "unique_percentage": round((col_data.nunique() / len(col_data)) * 100, 2) if len(col_data) > 0 else 0,
                    "top": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    "top_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "top_percentage": round((value_counts.iloc[0] / len(col_data)) * 100, 2) if len(value_counts) > 0 and len(col_data) > 0 else 0,
                }
                
                # Top 10 most frequent values
                if len(value_counts) > 0:
                    top_10 = value_counts.head(10)
                    stats_dict[col]["top_values"] = {
                        str(k): int(v) for k, v in top_10.to_dict().items()
                    }
                
                # Entropy (measure of uncertainty)
                if len(col_data) > 0:
                    probabilities = value_counts / len(col_data)
                    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                    stats_dict[col]["entropy"] = float(entropy)
            
            except Exception as e:
                logger.warning(f"Error calculating stats for column {col}: {str(e)}")
                stats_dict[col] = {"error": str(e)}
        
        return stats_dict
    
    # ============================================================
    # CORRELATION ANALYSIS
    # ============================================================
    
    def _calculate_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlation matrices for numerical columns."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            return {}
        
        correlations = {}
        
        try:
            # Pearson correlation
            pearson = df[numerical_cols].corr(method='pearson')
            correlations["pearson"] = pearson.to_dict()
            
            # Spearman correlation (rank-based)
            spearman = df[numerical_cols].corr(method='spearman')
            correlations["spearman"] = spearman.to_dict()
            
            # Find strong correlations (|r| > 0.7)
            strong_correlations = []
            for i in range(len(numerical_cols)):
                for j in range(i + 1, len(numerical_cols)):
                    col1 = numerical_cols[i]
                    col2 = numerical_cols[j]
                    corr_value = pearson.loc[col1, col2]
                    
                    if abs(corr_value) > 0.7:
                        strong_correlations.append({
                            "column1": str(col1),
                            "column2": str(col2),
                            "correlation": float(corr_value),
                            "strength": "strong positive" if corr_value > 0 else "strong negative",
                        })
            
            correlations["strong_correlations"] = strong_correlations
            correlations["strong_correlations_count"] = len(strong_correlations)
        
        except Exception as e:
            logger.warning(f"Correlation calculation error: {str(e)}")
        
        return correlations
    
    # ============================================================
    # DISTRIBUTION ANALYSIS
    # ============================================================
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions of numerical columns."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return {}
        
        distributions = {}
        
        for col in numerical_cols:
            col_data = df[col].dropna()
            
            if len(col_data) < 2:
                continue
            
            try:
                distributions[col] = {
                    "distribution_type": self._detect_distribution_type(col_data),
                    "normality_test": self._test_normality(col_data),
                }
            except Exception as e:
                logger.warning(f"Distribution analysis error for {col}: {str(e)}")
                distributions[col] = {"error": str(e)}
        
        return distributions
    
    def _detect_distribution_type(self, data: pd.Series) -> str:
        """Detect likely distribution type based on skewness and kurtosis."""
        skewness = scipy_stats.skew(data)
        kurtosis_val = scipy_stats.kurtosis(data)
        
        # Simple heuristic classification
        if abs(skewness) < 0.5 and abs(kurtosis_val) < 0.5:
            return "normal"
        elif skewness > 1:
            return "right_skewed"
        elif skewness < -1:
            return "left_skewed"
        elif kurtosis_val > 3:
            return "heavy_tailed"
        elif kurtosis_val < -1:
            return "light_tailed"
        else:
            return "unknown"
    
    def _test_normality(self, data: pd.Series) -> Dict[str, Any]:
        """Test if data follows normal distribution (Shapiro-Wilk test)."""
        # Limit sample size for performance
        sample_size = min(5000, len(data))
        sample = data.sample(n=sample_size, random_state=42)
        
        try:
            statistic, p_value = scipy_stats.shapiro(sample)
            
            return {
                "test": "shapiro-wilk",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "is_normal": bool(p_value > 0.05),
                "confidence": "95%",
            }
        except Exception as e:
            logger.warning(f"Normality test failed: {str(e)}")
            return {"test": "shapiro-wilk", "error": str(e)}
    
    # ============================================================
    # DATA QUALITY
    # ============================================================
    
    def _calculate_data_quality_score(self, statistics: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate overall data quality score.
        
        Args:
            statistics: Statistics dictionary
            
        Returns:
            Dictionary with quality metrics
        """
        overview = statistics.get("overview", {})
        
        # Completeness score (100 - missing_percentage)
        completeness = 100 - overview.get("missing_percentage", 0)
        
        # Uniqueness score (100 - duplicate_percentage)
        uniqueness = 100 - overview.get("duplicate_percentage", 0)
        
        # Validity score (based on outliers in numerical columns)
        numerical_stats = statistics.get("numerical", {})
        if numerical_stats:
            avg_outliers = np.mean([
                col_stats.get("outliers_percentage", 0)
                for col_stats in numerical_stats.values()
                if "error" not in col_stats
            ])
            validity = 100 - min(avg_outliers, 100)
        else:
            validity = 100
        
        # Overall score (weighted average)
        overall = (completeness * 0.4 + uniqueness * 0.3 + validity * 0.3)
        
        return {
            "overall_score": round(overall, 2),
            "completeness": round(completeness, 2),
            "uniqueness": round(uniqueness, 2),
            "validity": round(validity, 2),
        }
    
    def _generate_recommendations(self, statistics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on statistics."""
        recommendations = []
        
        overview = statistics.get("overview", {})
        numerical = statistics.get("numerical", {})
        correlations = statistics.get("correlations", {})
        quality = statistics.get("data_quality", {})
        
        # Missing values
        if overview.get("missing_percentage", 0) > 5:
            recommendations.append(
                f"High percentage of missing values ({overview['missing_percentage']:.1f}%) - "
                "consider imputation or removal"
            )
        
        # Duplicates
        if overview.get("duplicate_percentage", 0) > 1:
            recommendations.append(
                f"Found {overview['duplicate_rows']} duplicate rows - "
                "consider removing duplicates"
            )
        
        # Outliers
        high_outlier_cols = [
            col for col, col_stats in numerical.items()
            if col_stats.get("outliers_percentage", 0) > 5
        ]
        if high_outlier_cols:
            recommendations.append(
                f"High outlier percentage in columns: {', '.join(high_outlier_cols[:3])} - "
                "investigate and handle appropriately"
            )
        
        # Strong correlations
        if correlations.get("strong_correlations_count", 0) > 0:
            recommendations.append(
                f"Found {correlations['strong_correlations_count']} strong correlations - "
                "consider feature engineering or dimensionality reduction"
            )
        
        # Overall quality
        if quality.get("overall_score", 100) < 70:
            recommendations.append(
                f"Overall data quality score is {quality['overall_score']:.1f}% - "
                "comprehensive cleaning recommended"
            )
        elif quality.get("overall_score", 100) > 90:
            recommendations.append(
                f"Excellent data quality score ({quality['overall_score']:.1f}%) - "
                "dataset is ready for analysis"
            )
        
        return recommendations if recommendations else [
            "Data quality is good - no major issues detected"
        ]
    
    # ============================================================
    # DATABASE OPERATIONS
    # ============================================================
    
    def _save_statistics(
        self,
        dataset_id: int,
        statistics: Dict[str, Any]
    ) -> None:
        """Save statistics to database."""
        # Check if statistics already exist
        existing = self.db.query(DatasetStatistics).filter(
            DatasetStatistics.dataset_id == dataset_id
        ).first()
        
        if existing:
            # Update existing
            existing.numerical_stats = statistics.get("numerical")
            existing.categorical_stats = statistics.get("categorical")
            existing.correlation_matrix = statistics.get("correlations")
            existing.distributions = statistics.get("distributions")
        else:
            # Create new
            stats_record = DatasetStatistics(
                dataset_id=dataset_id,
                numerical_stats=statistics.get("numerical"),
                categorical_stats=statistics.get("categorical"),
                correlation_matrix=statistics.get("correlations"),
                distributions=statistics.get("distributions"),
            )
            self.db.add(stats_record)
        
        self.db.commit()
    
    # ============================================================
    # HELPER METHODS
    # ============================================================
    
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
# DEPENDENCY INJECTION HELPER
# ============================================================

def get_eda_service(db: Session = Depends(get_db)) -> EDAService:
    """
    Dependency for injecting EDAService.
    
    Args:
        db: Database session
        
    Returns:
        EDAService instance
    """
    return EDAService(db)
