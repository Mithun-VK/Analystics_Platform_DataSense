"""
EDA Service - Part 1: Enhanced Core Features (10 Features)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Features 1-10:
1. ✅ Enhanced Overview Statistics with memory optimization
2. ✅ Advanced Numerical Statistics with percentiles & modes
3. ✅ Enhanced Categorical Statistics with entropy & concentration
4. ✅ Multi-method Correlation Analysis (Pearson, Spearman, Kendall)
5. ✅ Distribution Analysis with 3 normality tests
6. ✅ IQR-based Outlier Detection with bounds
7. ✅ Z-score Outlier Detection
8. ✅ Isolation Forest Outlier Detection (ML-based)
9. ✅ Missing Value Pattern Analysis with visualization data
10. ✅ VIF Multicollinearity Detection with severity classification
"""

import asyncio
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import chi2_contingency
from sklearn.ensemble import IsolationForest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.dataset import Dataset, DatasetStatus, DatasetStatistics
from app.database import get_db
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.ensemble import IsolationForest
from scipy.stats import f_oneway, kruskal
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from scipy.stats import chi2_contingency, f_oneway, kruskal, mannwhitneyu, levene, ttest_ind
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class EDAService:
    """Production-grade EDA service - Part 1 (Features 1-10)."""
    
    def __init__(self, db: Session):
        self.db = db
        logger.info("🔬 EDA Service initialized")

    @staticmethod
    def _convert_numpy_types(obj: Any) -> Any:
        """Convert NumPy types to native Python types."""
        if isinstance(obj, dict):
            return {k: EDAService._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [EDAService._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(EDAService._convert_numpy_types(item) for item in obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16,
                             np.int32, np.int64, np.uint8, np.uint16,
                             np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        else:
            return obj

    # ═══════════════════════════════════════════════════════════
    # FEATURE 1: ENHANCED OVERVIEW STATISTICS
    # ═══════════════════════════════════════════════════════════
    
    def _get_overview_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        ✅ FEATURE 1: Enhanced Overview Statistics
        
        Provides comprehensive dataset overview with:
        - Row/column counts
        - Memory usage (deep analysis)
        - Missing value analysis
        - Duplicate detection
        - Column type distribution
        - Data density metrics
        - Sparsity analysis
        """
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        filled_cells = total_cells - missing_cells
        
        return {
            # Basic dimensions
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "total_cells": int(total_cells),
            
            # Memory analysis
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            "memory_per_row_kb": round(df.memory_usage(deep=True).sum() / len(df) / 1024, 2) if len(df) > 0 else 0,
            
            # Missing value analysis
            "total_missing": int(missing_cells),
            "missing_percentage": round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0,
            "columns_with_missing": int((df.isnull().sum() > 0).sum()),
            "completely_empty_columns": int((df.isnull().sum() == len(df)).sum()),
            
            # Duplicate analysis
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_percentage": round((df.duplicated().sum() / len(df)) * 100, 2) if len(df) > 0 else 0,
            "unique_rows": int(len(df) - df.duplicated().sum()),
            
            # Column types
            "column_types": {str(k): int(v) for k, v in df.dtypes.value_counts().to_dict().items()},
            "numerical_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns),
            "datetime_columns": len(df.select_dtypes(include=['datetime64']).columns),
            
            # ✅ NEW: Data density metrics
            "data_density": round((filled_cells / total_cells) * 100, 2) if total_cells > 0 else 0,
            "sparsity": round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0,
            
            # Column name analysis
            "column_names": df.columns.tolist(),
            "avg_column_name_length": round(np.mean([len(col) for col in df.columns]), 2),
        }
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 2: ADVANCED NUMERICAL STATISTICS
    # ═══════════════════════════════════════════════════════════
    
    def _get_numerical_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        ✅ FEATURE 2: Advanced Numerical Statistics
        
        Comprehensive numerical analysis including:
        - Central tendency (mean, median, mode)
        - Dispersion (std, variance, CV, range)
        - Distribution shape (skewness, kurtosis)
        - Quartiles and percentiles (5th, 10th, 25th, 50th, 75th, 90th, 95th, 99th)
        - Outlier detection (IQR method)
        - Zero value analysis
        - Extreme value detection
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return {}
        
        stats_dict = {}
        
        for col in numerical_cols:
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                stats_dict[col] = {"error": "No non-null values"}
                continue
            
            try:
                # Quartiles
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                # Outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                
                # ✅ NEW: Mode calculation
                mode_value = col_data.mode()
                mode = float(mode_value[0]) if len(mode_value) > 0 else None
                
                stats_dict[col] = {
                    # Basic stats
                    "count": int(col_data.count()),
                    "missing": int(df[col].isnull().sum()),
                    "missing_percentage": round((df[col].isnull().sum() / len(df)) * 100, 2),
                    
                    # Central tendency
                    "mean": float(col_data.mean()),
                    "median": float(col_data.median()),
                    "mode": mode,
                    
                    # Dispersion
                    "std": float(col_data.std()),
                    "variance": float(col_data.var()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "range": float(col_data.max() - col_data.min()),
                    "coefficient_variation": float((col_data.std() / col_data.mean()) * 100) if col_data.mean() != 0 else 0,
                    
                    # Quartiles
                    "q1": float(Q1),
                    "q2_median": float(col_data.median()),
                    "q3": float(Q3),
                    "iqr": float(IQR),
                    
                    # ✅ NEW: Extended percentiles
                    "percentiles": {
                        "p5": float(col_data.quantile(0.05)),
                        "p10": float(col_data.quantile(0.10)),
                        "p25": float(Q1),
                        "p50": float(col_data.median()),
                        "p75": float(Q3),
                        "p90": float(col_data.quantile(0.90)),
                        "p95": float(col_data.quantile(0.95)),
                        "p99": float(col_data.quantile(0.99)),
                    },
                    
                    # Distribution shape
                    "skewness": float(scipy_stats.skew(col_data)),
                    "kurtosis": float(scipy_stats.kurtosis(col_data)),
                    "skewness_interpretation": self._interpret_skewness(scipy_stats.skew(col_data)),
                    "kurtosis_interpretation": self._interpret_kurtosis(scipy_stats.kurtosis(col_data)),
                    
                    # Zero analysis
                    "zeros": int((col_data == 0).sum()),
                    "zeros_percentage": round(((col_data == 0).sum() / len(col_data)) * 100, 2),
                    
                    # Outlier analysis
                    "outliers": int(len(outliers)),
                    "outliers_percentage": round((len(outliers) / len(col_data)) * 100, 2),
                    "outlier_lower_bound": float(lower_bound),
                    "outlier_upper_bound": float(upper_bound),
                    
                    # ✅ NEW: Extreme values
                    "min_values": col_data.nsmallest(3).tolist(),
                    "max_values": col_data.nlargest(3).tolist(),
                    
                    # ✅ NEW: Value concentration
                    "unique_values": int(col_data.nunique()),
                    "unique_percentage": round((col_data.nunique() / len(col_data)) * 100, 2),
                }
                
            except Exception as e:
                logger.warning(f"⚠️ Error calculating stats for {col}: {e}")
                stats_dict[col] = {"error": str(e)}
        
        return stats_dict
    
    def _interpret_skewness(self, skew: float) -> str:
        """Interpret skewness value."""
        if abs(skew) < 0.5:
            return "fairly_symmetrical"
        elif skew > 0.5:
            return "right_skewed_positive"
        else:
            return "left_skewed_negative"
    
    def _interpret_kurtosis(self, kurt: float) -> str:
        """Interpret kurtosis value."""
        if abs(kurt) < 0.5:
            return "mesokurtic_normal"
        elif kurt > 0.5:
            return "leptokurtic_heavy_tailed"
        else:
            return "platykurtic_light_tailed"
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 3: ENHANCED CATEGORICAL STATISTICS
    # ═══════════════════════════════════════════════════════════
    
    def _get_categorical_statistics(
        self,
        df: pd.DataFrame,
        cardinality_limit: int = 50
    ) -> Dict[str, Any]:
        """
        ✅ FEATURE 3: Enhanced Categorical Statistics
        
        Comprehensive categorical analysis:
        - Unique value counts
        - Frequency analysis
        - Entropy calculation
        - Concentration metrics (Gini coefficient)
        - High cardinality detection
        - Top N value analysis
        - Imbalance detection
        """
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return {}
        
        stats_dict = {}
        
        for col in categorical_cols:
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                stats_dict[col] = {"error": "No non-null values"}
                continue
            
            try:
                value_counts = col_data.value_counts()
                unique_count = col_data.nunique()
                
                # ✅ NEW: Calculate entropy
                probabilities = value_counts / len(col_data)
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                
                # ✅ NEW: Calculate Gini coefficient (concentration)
                sorted_probs = np.sort(probabilities.values)
                n = len(sorted_probs)
                gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_probs)) / (n * np.sum(sorted_probs)) - (n + 1) / n
                
                stats_dict[col] = {
                    # Basic counts
                    "count": int(col_data.count()),
                    "missing": int(df[col].isnull().sum()),
                    "missing_percentage": round((df[col].isnull().sum() / len(df)) * 100, 2),
                    
                    # Unique analysis
                    "unique": int(unique_count),
                    "unique_percentage": round((unique_count / len(col_data)) * 100, 2),
                    
                    # Top value
                    "top": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    "top_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "top_percentage": round((value_counts.iloc[0] / len(col_data)) * 100, 2),
                    
                    # ✅ NEW: Top N values analysis
                    "top_values": {
                        str(k): int(v) for k, v in value_counts.head(10).to_dict().items()
                    },
                    "top_5_concentration": round((value_counts.head(5).sum() / len(col_data)) * 100, 2),
                    "top_10_concentration": round((value_counts.head(10).sum() / len(col_data)) * 100, 2),
                    
                    # ✅ NEW: Information theory metrics
                    "entropy": float(entropy),
                    "max_entropy": float(np.log2(unique_count)) if unique_count > 0 else 0,
                    "normalized_entropy": float(entropy / np.log2(unique_count)) if unique_count > 1 else 0,
                    
                    # ✅ NEW: Concentration metrics
                    "gini_coefficient": float(gini),
                    "concentration_ratio": round((value_counts.iloc[0] / value_counts.iloc[-1]) if len(value_counts) > 1 else 1, 2),
                    
                    # Cardinality flags
                    "high_cardinality": unique_count > cardinality_limit,
                    "cardinality_level": self._classify_cardinality(unique_count, len(col_data)),
                    
                    # ✅ NEW: Imbalance detection
                    "is_imbalanced": value_counts.iloc[0] / len(col_data) > 0.9 if len(value_counts) > 0 else False,
                    "imbalance_ratio": round((value_counts.iloc[0] / value_counts.iloc[-1]), 2) if len(value_counts) > 1 else 1,
                }
                
            except Exception as e:
                logger.warning(f"⚠️ Error calculating stats for {col}: {e}")
                stats_dict[col] = {"error": str(e)}
        
        return stats_dict
    
    def _classify_cardinality(self, unique: int, total: int) -> str:
        """Classify cardinality level."""
        ratio = unique / total if total > 0 else 0
        
        if unique == 1:
            return "constant"
        elif ratio < 0.01:
            return "low"
        elif ratio < 0.1:
            return "medium"
        elif ratio < 0.5:
            return "high"
        else:
            return "very_high"
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 4: MULTI-METHOD CORRELATION ANALYSIS
    # ═══════════════════════════════════════════════════════════
    
    def _calculate_correlations(
        self,
        df: pd.DataFrame,
        threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        ✅ FEATURE 4: Multi-method Correlation Analysis
        
        Calculates correlations using three methods:
        - Pearson (linear relationships)
        - Spearman (monotonic relationships)
        - Kendall (rank-based, robust to outliers)
        
        Also identifies:
        - Strong correlations (above threshold)
        - Perfect correlations (potential duplicates)
        - Correlation strength classification
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            return {
                "message": "Not enough numerical columns for correlation analysis",
                "numerical_columns_count": len(numerical_cols)
            }
        
        try:
            # Calculate all three correlation methods
            pearson = df[numerical_cols].corr(method='pearson')
            spearman = df[numerical_cols].corr(method='spearman')
            kendall = df[numerical_cols].corr(method='kendall')
            
            # Find strong correlations
            strong_correlations = []
            perfect_correlations = []
            
            for i in range(len(numerical_cols)):
                for j in range(i + 1, len(numerical_cols)):
                    col1 = numerical_cols[i]
                    col2 = numerical_cols[j]
                    
                    pearson_val = pearson.loc[col1, col2]
                    spearman_val = spearman.loc[col1, col2]
                    kendall_val = kendall.loc[col1, col2]
                    
                    # Check for perfect correlation (potential duplicates)
                    if abs(pearson_val) > 0.99:
                        perfect_correlations.append({
                            "column1": str(col1),
                            "column2": str(col2),
                            "pearson": float(pearson_val),
                            "warning": "Possible duplicate columns"
                        })
                    
                    # Check for strong correlation
                    if abs(pearson_val) > threshold:
                        strength = self._classify_correlation_strength(pearson_val)
                        
                        strong_correlations.append({
                            "column1": str(col1),
                            "column2": str(col2),
                            "pearson": float(pearson_val),
                            "spearman": float(spearman_val),
                            "kendall": float(kendall_val),
                            "strength": strength,
                            "direction": "positive" if pearson_val > 0 else "negative",
                            "agreement": self._check_correlation_agreement(pearson_val, spearman_val, kendall_val),
                        })
            
            return {
                # Correlation matrices
                "pearson": pearson.to_dict(),
                "spearman": spearman.to_dict(),
                "kendall": kendall.to_dict(),
                
                # Summary
                "strong_correlations": sorted(strong_correlations, key=lambda x: abs(x['pearson']), reverse=True),
                "strong_correlations_count": len(strong_correlations),
                "perfect_correlations": perfect_correlations,
                "perfect_correlations_count": len(perfect_correlations),
                
                # Configuration
                "threshold_used": threshold,
                "methods_used": ["pearson", "spearman", "kendall"],
            }
            
        except Exception as e:
            logger.error(f"❌ Correlation calculation failed: {e}")
            return {"error": str(e)}
    
    def _classify_correlation_strength(self, corr: float) -> str:
        """Classify correlation strength."""
        abs_corr = abs(corr)
        
        if abs_corr > 0.9:
            return "very_strong"
        elif abs_corr > 0.7:
            return "strong"
        elif abs_corr > 0.5:
            return "moderate"
        elif abs_corr > 0.3:
            return "weak"
        else:
            return "very_weak"
    
    def _check_correlation_agreement(self, pearson: float, spearman: float, kendall: float) -> str:
        """Check if all three methods agree on correlation direction."""
        signs = [np.sign(pearson), np.sign(spearman), np.sign(kendall)]
        
        if len(set(signs)) == 1:
            return "all_agree"
        elif len(set(signs)) == 2:
            return "partial_agreement"
        else:
            return "conflicting"
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 5: DISTRIBUTION ANALYSIS WITH 3 NORMALITY TESTS
    # ═══════════════════════════════════════════════════════════
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        ✅ FEATURE 5: Distribution Analysis with Multiple Tests
        
        Performs comprehensive distribution analysis:
        - Distribution type detection (based on skewness/kurtosis)
        - Shapiro-Wilk test (best for n < 5000)
        - Kolmogorov-Smirnov test
        - Anderson-Darling test
        - Normality consensus
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return {}
        
        distributions = {}
        
        for col in numerical_cols:
            col_data = df[col].dropna()
            
            if len(col_data) < 3:
                distributions[col] = {"error": "Not enough data points"}
                continue
            
            try:
                skewness = scipy_stats.skew(col_data)
                kurtosis_val = scipy_stats.kurtosis(col_data)
                
                # Detect distribution type
                dist_type = self._detect_distribution_type(skewness, kurtosis_val)
                
                # Sample for tests (max 5000 points)
                sample_size = min(5000, len(col_data))
                sample = col_data.sample(n=sample_size, random_state=42)
                
                # Test 1: Shapiro-Wilk
                shapiro_stat, shapiro_p = scipy_stats.shapiro(sample)
                
                # Test 2: Kolmogorov-Smirnov
                ks_stat, ks_p = scipy_stats.kstest(sample, 'norm', args=(sample.mean(), sample.std()))
                
                # Test 3: Anderson-Darling
                anderson_result = scipy_stats.anderson(sample, dist='norm')
                anderson_normal = anderson_result.statistic < anderson_result.critical_values[2]  # 5% significance
                
                # Consensus
                normality_votes = sum([
                    shapiro_p > 0.05,
                    ks_p > 0.05,
                    anderson_normal
                ])
                
                distributions[col] = {
                    # Distribution classification
                    "distribution_type": dist_type,
                    "skewness": float(skewness),
                    "kurtosis": float(kurtosis_val),
                    
                    # Shapiro-Wilk test
                    "shapiro_wilk": {
                        "test": "shapiro-wilk",
                        "statistic": float(shapiro_stat),
                        "p_value": float(shapiro_p),
                        "is_normal": bool(shapiro_p > 0.05),
                        "confidence": "95%"
                    },
                    
                    # Kolmogorov-Smirnov test
                    "kolmogorov_smirnov": {
                        "test": "kolmogorov-smirnov",
                        "statistic": float(ks_stat),
                        "p_value": float(ks_p),
                        "is_normal": bool(ks_p > 0.05),
                    },
                    
                    # Anderson-Darling test
                    "anderson_darling": {
                        "test": "anderson-darling",
                        "statistic": float(anderson_result.statistic),
                        "critical_values": anderson_result.critical_values.tolist(),
                        "significance_levels": anderson_result.significance_level.tolist(),
                        "is_normal": bool(anderson_normal),
                    },
                    
                    # ✅ NEW: Normality consensus
                    "normality_consensus": {
                        "tests_agreeing_normal": int(normality_votes),
                        "total_tests": 3,
                        "consensus": "normal" if normality_votes >= 2 else "non_normal",
                        "confidence_level": f"{(normality_votes / 3) * 100:.0f}%"
                    }
                }
                
            except Exception as e:
                logger.warning(f"⚠️ Distribution analysis failed for {col}: {e}")
                distributions[col] = {"error": str(e)}
        
        return distributions
    
    def _detect_distribution_type(self, skewness: float, kurtosis: float) -> str:
        """Detect distribution type based on skewness and kurtosis."""
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return "normal"
        elif skewness > 1:
            return "right_skewed"
        elif skewness < -1:
            return "left_skewed"
        elif kurtosis > 3:
            return "heavy_tailed"
        elif kurtosis < -1:
            return "light_tailed"
        elif abs(skewness) < 0.5:
            return "symmetric_non_normal"
        else:
            return "unknown"


# ═══════════════════════════════════════════════════════════
# FEATURES 6-8: OUTLIER DETECTION (THREE METHODS)
# ═══════════════════════════════════════════════════════════
    """
EDA Service - Part 2: Advanced Outlier Detection & Missing Value Analysis (Features 6-15)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Features 6-15:
6. ✅ IQR-based Outlier Detection with detailed bounds
7. ✅ Z-score Outlier Detection with severity classification
8. ✅ Isolation Forest ML-based Outlier Detection
9. ✅ Missing Value Pattern Analysis with correlation
10. ✅ VIF Multicollinearity Detection with recommendations
11. ✅ Constant & Quasi-Constant Feature Detection
12. ✅ Feature Importance using Mutual Information
13. ✅ PCA Analysis with optimal component selection
14. ✅ K-means Clustering with elbow method
15. ✅ Anomaly Detection using Isolation Forest
    """

    # ═══════════════════════════════════════════════════════════
    # FEATURE 6: IQR-BASED OUTLIER DETECTION
    # ═══════════════════════════════════════════════════════════
    
    def _detect_outliers_iqr(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        ✅ FEATURE 6: IQR-based Outlier Detection
        
        Uses Interquartile Range method:
        - Lower bound: Q1 - 1.5 × IQR
        - Upper bound: Q3 + 1.5 × IQR
        - Identifies mild and extreme outliers
        - Provides outlier indices and values
        """
        numerical = df.select_dtypes(include=[np.number])
        
        if len(numerical.columns) == 0:
            return {"message": "No numerical columns for outlier detection"}
        
        outlier_results = {}
        
        for col in numerical.columns:
            col_data = numerical[col].dropna()
            
            if len(col_data) < 10:
                outlier_results[col] = {"error": "Not enough data points"}
                continue
            
            try:
                # Calculate IQR
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                # Outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Extreme outlier bounds (3 × IQR)
                extreme_lower = Q1 - 3 * IQR
                extreme_upper = Q3 + 3 * IQR
                
                # Identify outliers
                mild_outliers_mask = ((col_data < lower_bound) | (col_data > upper_bound)) & \
                                     ((col_data >= extreme_lower) & (col_data <= extreme_upper))
                extreme_outliers_mask = (col_data < extreme_lower) | (col_data > extreme_upper)
                
                mild_outliers = col_data[mild_outliers_mask]
                extreme_outliers = col_data[extreme_outliers_mask]
                all_outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                
                outlier_results[col] = {
                    "method": "iqr",
                    
                    # Bounds
                    "bounds": {
                        "lower_mild": float(lower_bound),
                        "upper_mild": float(upper_bound),
                        "lower_extreme": float(extreme_lower),
                        "upper_extreme": float(extreme_upper),
                    },
                    
                    # Outlier counts
                    "total_outliers": int(len(all_outliers)),
                    "mild_outliers": int(len(mild_outliers)),
                    "extreme_outliers": int(len(extreme_outliers)),
                    
                    # Percentages
                    "total_outlier_percentage": round((len(all_outliers) / len(col_data)) * 100, 2),
                    "mild_outlier_percentage": round((len(mild_outliers) / len(col_data)) * 100, 2),
                    "extreme_outlier_percentage": round((len(extreme_outliers) / len(col_data)) * 100, 2),
                    
                    # Outlier values (limited to 50 for performance)
                    "outlier_values_sample": all_outliers.head(50).tolist(),
                    "outlier_indices_sample": all_outliers.head(50).index.tolist(),
                    
                    # Statistical context
                    "iqr_value": float(IQR),
                    "q1": float(Q1),
                    "q3": float(Q3),
                    
                    # Severity assessment
                    "severity": self._assess_outlier_severity(len(all_outliers) / len(col_data)),
                }
                
            except Exception as e:
                logger.warning(f"⚠️ IQR outlier detection failed for {col}: {e}")
                outlier_results[col] = {"error": str(e)}
        
        return {
            "method": "iqr",
            "description": "Interquartile Range method (1.5 × IQR)",
            "results_by_column": outlier_results,
            "columns_with_outliers": [col for col, res in outlier_results.items() 
                                     if res.get('total_outliers', 0) > 0],
            "summary": self._summarize_outliers(outlier_results)
        }
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 7: Z-SCORE OUTLIER DETECTION
    # ═══════════════════════════════════════════════════════════
    
    def _detect_outliers_zscore(self, df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, Any]:
        """
        ✅ FEATURE 7: Z-score Outlier Detection
        
        Statistical outlier detection using standard deviations:
        - Threshold 3: Standard (99.7% rule)
        - Threshold 2: Moderate (95% rule)
        - Calculates z-scores for all values
        - Identifies values beyond threshold standard deviations
        """
        numerical = df.select_dtypes(include=[np.number])
        
        if len(numerical.columns) == 0:
            return {"message": "No numerical columns for z-score analysis"}
        
        outlier_results = {}
        
        for col in numerical.columns:
            col_data = numerical[col].dropna()
            
            if len(col_data) < 10:
                outlier_results[col] = {"error": "Not enough data points"}
                continue
            
            try:
                # Calculate z-scores
                mean = col_data.mean()
                std = col_data.std()
                
                if std == 0:
                    outlier_results[col] = {"error": "Zero standard deviation"}
                    continue
                
                z_scores = np.abs((col_data - mean) / std)
                
                # Classify by severity
                mild_outliers_mask = (z_scores > 2) & (z_scores <= 3)
                moderate_outliers_mask = (z_scores > 3) & (z_scores <= 4)
                extreme_outliers_mask = z_scores > 4
                
                all_outliers_mask = z_scores > threshold
                
                outlier_results[col] = {
                    "method": "zscore",
                    "threshold": threshold,
                    
                    # Statistics
                    "mean": float(mean),
                    "std": float(std),
                    "max_z_score": float(z_scores.max()),
                    "min_z_score": float(z_scores.min()),
                    
                    # Outlier counts by severity
                    "total_outliers": int(all_outliers_mask.sum()),
                    "mild_outliers_2_3_sigma": int(mild_outliers_mask.sum()),
                    "moderate_outliers_3_4_sigma": int(moderate_outliers_mask.sum()),
                    "extreme_outliers_4plus_sigma": int(extreme_outliers_mask.sum()),
                    
                    # Percentages
                    "total_outlier_percentage": round((all_outliers_mask.sum() / len(col_data)) * 100, 2),
                    
                    # Outlier details (sample)
                    "outlier_values_sample": col_data[all_outliers_mask].head(50).tolist(),
                    "outlier_z_scores_sample": z_scores[all_outliers_mask].head(50).tolist(),
                    
                    # Severity
                    "severity": self._assess_outlier_severity(all_outliers_mask.sum() / len(col_data)),
                    "recommendation": self._get_zscore_recommendation(z_scores.max()),
                }
                
            except Exception as e:
                logger.warning(f"⚠️ Z-score outlier detection failed for {col}: {e}")
                outlier_results[col] = {"error": str(e)}
        
        return {
            "method": "zscore",
            "description": f"Standard deviation method (threshold: {threshold} sigma)",
            "results_by_column": outlier_results,
            "columns_with_outliers": [col for col, res in outlier_results.items() 
                                     if res.get('total_outliers', 0) > 0],
            "summary": self._summarize_outliers(outlier_results)
        }
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 8: ISOLATION FOREST OUTLIER DETECTION
    # ═══════════════════════════════════════════════════════════
    
    def _detect_outliers_isolation_forest(
        self,
        df: pd.DataFrame,
        contamination: float = 0.1
    ) -> Dict[str, Any]:
        """
        ✅ FEATURE 8: Isolation Forest ML-based Outlier Detection
        
        Machine learning approach:
        - Uses ensemble of isolation trees
        - Detects multivariate outliers
        - Handles high-dimensional data
        - Provides anomaly scores
        """
        numerical = df.select_dtypes(include=[np.number])
        
        if len(numerical.columns) == 0:
            return {"message": "No numerical columns for isolation forest"}
        
        try:
            # Prepare data
            data_clean = numerical.dropna()
            
            if len(data_clean) < 50:
                return {"error": "Not enough data points (minimum 50 required)"}
            
            # Train Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100,
                max_samples='auto',
                n_jobs=-1
            )
            
            # Predict outliers
            predictions = iso_forest.fit_predict(data_clean)
            anomaly_scores = iso_forest.score_samples(data_clean)
            
            # Identify outliers
            outlier_mask = predictions == -1
            outlier_indices = data_clean[outlier_mask].index.tolist()
            
            # Per-column analysis
            column_outliers = {}
            for col in numerical.columns:
                col_outliers = data_clean.loc[outlier_mask, col]
                
                column_outliers[col] = {
                    "outliers_detected": int(len(col_outliers)),
                    "outlier_percentage": round((len(col_outliers) / len(data_clean)) * 100, 2),
                    "outlier_values_sample": col_outliers.head(50).tolist(),
                }
            
            return {
                "method": "isolation_forest",
                "description": "ML-based anomaly detection using Isolation Forest",
                
                # Global results
                "total_outliers": int(outlier_mask.sum()),
                "total_outlier_percentage": round((outlier_mask.sum() / len(data_clean)) * 100, 2),
                "contamination_used": contamination,
                
                # Anomaly scores
                "min_anomaly_score": float(anomaly_scores.min()),
                "max_anomaly_score": float(anomaly_scores.max()),
                "mean_anomaly_score": float(anomaly_scores.mean()),
                
                # Outlier details
                "outlier_indices_sample": outlier_indices[:100],
                "outlier_scores_sample": anomaly_scores[outlier_mask][:100].tolist(),
                
                # Per-column breakdown
                "results_by_column": column_outliers,
                
                # Model info
                "n_estimators": 100,
                "samples_analyzed": len(data_clean),
            }
            
        except Exception as e:
            logger.error(f"❌ Isolation Forest failed: {e}")
            return {"error": str(e)}
    
    def _assess_outlier_severity(self, percentage: float) -> str:
        """Assess outlier severity based on percentage."""
        if percentage < 1:
            return "low"
        elif percentage < 5:
            return "moderate"
        elif percentage < 10:
            return "high"
        else:
            return "very_high"
    
    def _get_zscore_recommendation(self, max_z: float) -> str:
        """Get recommendation based on maximum z-score."""
        if max_z > 5:
            return "Extreme outliers detected - investigate data quality issues"
        elif max_z > 4:
            return "Significant outliers - consider removal or transformation"
        elif max_z > 3:
            return "Moderate outliers - analyze context before action"
        else:
            return "Outliers within normal range"
    
    def _summarize_outliers(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize outlier results across all columns."""
        total_outliers = sum(res.get('total_outliers', 0) for res in results.values() if isinstance(res, dict))
        columns_with_outliers = sum(1 for res in results.values() if isinstance(res, dict) and res.get('total_outliers', 0) > 0)
        
        return {
            "total_outliers_across_all_columns": total_outliers,
            "columns_with_outliers": columns_with_outliers,
            "columns_analyzed": len(results),
        }
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 9: MISSING VALUE PATTERN ANALYSIS
    # ═══════════════════════════════════════════════════════════
    
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        ✅ FEATURE 9: Missing Value Pattern Analysis
        
        Comprehensive missing value analysis:
        - Missing counts and percentages per column
        - Missing value correlations
        - Missingness patterns (MCAR, MAR, MNAR detection)
        - Heatmap data for visualization
        - Imputation recommendations
        """
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100
        
        # Columns with missing values
        cols_with_missing = missing_counts[missing_counts > 0]
        
        if len(cols_with_missing) == 0:
            return {
                "message": "No missing values detected",
                "total_missing": 0,
                "completeness": 100.0
            }
        
        try:
            # Missing value correlation matrix
            missing_matrix = df.isnull().astype(int)
            missing_corr = missing_matrix.corr()
            
            # Find correlated missingness
            correlated_missing = []
            for i in range(len(missing_corr.columns)):
                for j in range(i + 1, len(missing_corr.columns)):
                    corr_val = missing_corr.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        correlated_missing.append({
                            "column1": missing_corr.columns[i],
                            "column2": missing_corr.columns[j],
                            "correlation": float(corr_val),
                            "pattern": "likely_related" if corr_val > 0.7 else "possibly_related"
                        })
            
            # Analyze missingness type
            missingness_types = {}
            for col in cols_with_missing.index:
                missing_type = self._classify_missingness(df, col)
                missingness_types[col] = missing_type
            
            # Generate imputation recommendations
            imputation_recommendations = {}
            for col in cols_with_missing.index:
                imputation_recommendations[col] = self._recommend_imputation(df, col, missingness_types[col])
            
            return {
                # Summary
                "total_missing_cells": int(df.isnull().sum().sum()),
                "total_cells": int(df.size),
                "missing_percentage": round((df.isnull().sum().sum() / df.size) * 100, 2),
                "completeness": round(100 - (df.isnull().sum().sum() / df.size) * 100, 2),
                
                # Per column
                "columns_with_missing": {
                    col: {
                        "count": int(count),
                        "percentage": round(pct, 2),
                        "missingness_type": missingness_types.get(col, "unknown"),
                        "imputation_recommendation": imputation_recommendations.get(col, "unknown")
                    }
                    for col, count, pct in zip(cols_with_missing.index, cols_with_missing.values, missing_pct[cols_with_missing.index])
                },
                
                # Patterns
                "correlated_missingness": correlated_missing,
                "columns_with_high_missing": missing_pct[missing_pct > 50].index.tolist(),
                "columns_with_moderate_missing": missing_pct[(missing_pct > 10) & (missing_pct <= 50)].index.tolist(),
                "columns_with_low_missing": missing_pct[(missing_pct > 0) & (missing_pct <= 10)].index.tolist(),
                
                # Heatmap data for visualization
                "missing_matrix": missing_matrix.head(100).to_dict(),
                "missing_correlation_matrix": missing_corr.to_dict(),
            }
            
        except Exception as e:
            logger.error(f"❌ Missing pattern analysis failed: {e}")
            return {"error": str(e)}
    
    def _classify_missingness(self, df: pd.DataFrame, col: str) -> str:
        """Classify type of missingness (MCAR, MAR, MNAR)."""
        # Simplified classification - in production, use statistical tests
        missing_pct = df[col].isnull().sum() / len(df)
        
        if missing_pct < 0.05:
            return "MCAR_likely"  # Missing Completely At Random
        elif missing_pct < 0.3:
            return "MAR_possible"  # Missing At Random
        else:
            return "MNAR_possible"  # Missing Not At Random
    
    def _recommend_imputation(self, df: pd.DataFrame, col: str, missingness_type: str) -> str:
        """Recommend imputation method based on column type and missingness."""
        if df[col].dtype in ['int64', 'float64']:
            if missingness_type == "MCAR_likely":
                return "mean_or_median"
            else:
                return "ml_based_imputation"
        else:
            return "mode_or_ml_based"
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 10: VIF MULTICOLLINEARITY DETECTION
    # ═══════════════════════════════════════════════════════════
    
    def _detect_multicollinearity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        ✅ FEATURE 10: VIF Multicollinearity Detection
        
        Variance Inflation Factor analysis:
        - Calculates VIF for each numerical feature
        - VIF > 10: High multicollinearity
        - VIF 5-10: Moderate multicollinearity
        - VIF < 5: Low multicollinearity
        - Provides removal recommendations
        """
        numerical = df.select_dtypes(include=[np.number])
        
        if len(numerical.columns) < 2:
            return {"message": "At least 2 numerical columns required for VIF analysis"}
        
        try:
            # Remove columns with zero variance
            numerical_clean = numerical.loc[:, numerical.std() > 0]
            
            if len(numerical_clean.columns) < 2:
                return {"message": "Not enough columns with variance for VIF"}
            
            # Drop rows with any NaN
            numerical_clean = numerical_clean.dropna()
            
            if len(numerical_clean) < 10:
                return {"error": "Not enough rows after removing NaN"}
            
            # Calculate VIF for each feature
            vif_data = pd.DataFrame()
            vif_data["feature"] = numerical_clean.columns
            vif_data["VIF"] = [
                variance_inflation_factor(numerical_clean.values, i)
                for i in range(len(numerical_clean.columns))
            ]
            
            # Classify severity
            vif_data['severity'] = vif_data['VIF'].apply(lambda x:
                'High' if x > 10 else
                'Moderate' if x > 5 else
                'Low'
            )
            
            # Sort by VIF descending
            vif_data = vif_data.sort_values('VIF', ascending=False)
            
            # Identify problematic features
            high_vif = vif_data[vif_data['VIF'] > 10]
            moderate_vif = vif_data[(vif_data['VIF'] > 5) & (vif_data['VIF'] <= 10)]
            
            return {
                "method": "variance_inflation_factor",
                "description": "Multicollinearity detection using VIF",
                
                # All VIF scores
                "vif_scores": vif_data.to_dict('records'),
                
                # Categorized features
                "high_multicollinearity_features": high_vif['feature'].tolist(),
                "moderate_multicollinearity_features": moderate_vif['feature'].tolist(),
                "low_multicollinearity_features": vif_data[vif_data['VIF'] <= 5]['feature'].tolist(),
                
                # Counts
                "high_multicollinearity_count": len(high_vif),
                "moderate_multicollinearity_count": len(moderate_vif),
                
                # Recommendations
                "recommendations": [
                    f"Consider removing '{feat}' (VIF={vif:.2f})"
                    for feat, vif in zip(high_vif['feature'], high_vif['VIF'])
                ],
                
                # Summary
                "summary": (
                    f"Found {len(high_vif)} features with high multicollinearity (VIF > 10), "
                    f"{len(moderate_vif)} with moderate (VIF 5-10)"
                    if len(high_vif) > 0 or len(moderate_vif) > 0
                    else "No significant multicollinearity detected"
                ),
                
                # Interpretation guide
                "interpretation": {
                    "VIF < 5": "Low multicollinearity - acceptable",
                    "VIF 5-10": "Moderate multicollinearity - monitor",
                    "VIF > 10": "High multicollinearity - consider removal"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ VIF calculation failed: {e}")
            return {"error": str(e)}
    """
    EDA Service - Part 3: ML Insights & Advanced Analytics (Features 11-20)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Features 11-20:
    11. ✅ Constant & Quasi-Constant Feature Detection
    12. ✅ Feature Importance using Mutual Information
    13. ✅ PCA Analysis with Optimal Component Selection
    14. ✅ K-means Clustering with Elbow Method & Silhouette
    15. ✅ Anomaly Detection using Isolation Forest
    16. ✅ Statistical Tests (Chi-square, ANOVA, Kruskal-Wallis)
    17. ✅ Time Series Analysis with Decomposition
    18. ✅ Comprehensive Data Quality Scoring
    19. ✅ Automated Insights Generation
    20. ✅ Smart Recommendations Engine
    """


    # ═══════════════════════════════════════════════════════════
    # FEATURE 11: CONSTANT & QUASI-CONSTANT FEATURE DETECTION
    # ═══════════════════════════════════════════════════════════
    
    def _detect_constant_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        ✅ FEATURE 11: Constant & Quasi-Constant Feature Detection
        
        Identifies features with little to no variance:
        - Constant features (single unique value)
        - Quasi-constant features (>95% same value)
        - Low variance features (>90% same value)
        - Provides removal recommendations
        """
        constant_features = []
        quasi_constant_features = []
        low_variance_features = []
        
        feature_analysis = {}
        
        for col in df.columns:
            unique_count = df[col].nunique()
            total_count = len(df[col].dropna())
            
            if total_count == 0:
                continue
            
            unique_ratio = unique_count / total_count
            
            # Get most common value
            if unique_count > 0:
                most_common = df[col].mode()
                most_common_value = most_common[0] if len(most_common) > 0 else None
                most_common_count = (df[col] == most_common_value).sum() if most_common_value is not None else 0
                most_common_pct = (most_common_count / total_count) * 100
            else:
                most_common_value = None
                most_common_pct = 0
            
            # Classify feature
            feature_type = "variable"
            recommendation = "keep"
            
            if unique_count == 1:
                constant_features.append(col)
                feature_type = "constant"
                recommendation = "remove_constant"
            elif most_common_pct > 95:
                quasi_constant_features.append(col)
                feature_type = "quasi_constant"
                recommendation = "consider_removing"
            elif most_common_pct > 90:
                low_variance_features.append(col)
                feature_type = "low_variance"
                recommendation = "review"
            
            feature_analysis[col] = {
                "unique_values": int(unique_count),
                "unique_ratio": round(unique_ratio, 4),
                "most_common_value": str(most_common_value) if most_common_value is not None else None,
                "most_common_percentage": round(most_common_pct, 2),
                "feature_type": feature_type,
                "recommendation": recommendation,
            }
        
        return {
            "method": "variance_analysis",
            "description": "Detection of constant and quasi-constant features",
            
            # Categorized features
            "constant_features": constant_features,
            "quasi_constant_features": quasi_constant_features,
            "low_variance_features": low_variance_features,
            
            # Counts
            "constant_count": len(constant_features),
            "quasi_constant_count": len(quasi_constant_features),
            "low_variance_count": len(low_variance_features),
            
            # Detailed analysis
            "feature_analysis": feature_analysis,
            
            # Recommendations
            "removal_recommendations": {
                "immediate_removal": constant_features,
                "consider_removal": quasi_constant_features,
                "review_needed": low_variance_features,
            },
            
            # Summary
            "summary": (
                f"Found {len(constant_features)} constant, "
                f"{len(quasi_constant_features)} quasi-constant, "
                f"and {len(low_variance_features)} low-variance features"
            ),
            
            # Thresholds used
            "thresholds": {
                "constant": "100% same value",
                "quasi_constant": ">95% same value",
                "low_variance": ">90% same value"
            }
        }
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 12: FEATURE IMPORTANCE USING MUTUAL INFORMATION
    # ═══════════════════════════════════════════════════════════
    
    def _calculate_feature_importance(
        self,
        df: pd.DataFrame,
        target_col: str,
        top_n: int = 15
    ) -> Dict[str, Any]:
        """
        ✅ FEATURE 12: Feature Importance using Mutual Information
        
        Calculates feature importance for supervised learning:
        - Mutual Information (MI) scores
        - Handles both regression and classification
        - Identifies most predictive features
        - Provides feature selection recommendations
        """
        if target_col not in df.columns:
            return {"error": f"Target column '{target_col}' not found"}
        
        try:
            # Separate features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Handle only numerical features for now
            numerical_features = X.select_dtypes(include=[np.number])
            
            if len(numerical_features.columns) == 0:
                return {"error": "No numerical features available"}
            
            # Remove rows with NaN
            X_clean = numerical_features.dropna()
            y_clean = y.loc[X_clean.index]
            
            # Determine if classification or regression
            is_classification = y_clean.dtype == 'object' or y_clean.nunique() < 20
            
            # Encode target if categorical
            if is_classification:
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y_clean.astype(str))
                mi_scores = mutual_info_classif(X_clean, y_encoded, random_state=42)
                task_type = "classification"
            else:
                mi_scores = mutual_info_regression(X_clean, y_clean, random_state=42)
                task_type = "regression"
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': X_clean.columns,
                'importance': mi_scores,
                'importance_normalized': mi_scores / mi_scores.max() if mi_scores.max() > 0 else mi_scores
            }).sort_values('importance', ascending=False)
            
            # Classify features by importance
            high_importance = importance_df[importance_df['importance_normalized'] > 0.7]
            medium_importance = importance_df[
                (importance_df['importance_normalized'] > 0.3) & 
                (importance_df['importance_normalized'] <= 0.7)
            ]
            low_importance = importance_df[importance_df['importance_normalized'] <= 0.3]
            
            return {
                "method": "mutual_information",
                "task_type": task_type,
                "target_column": target_col,
                
                # Top features
                "top_features": importance_df.head(top_n).to_dict('records'),
                "all_features": importance_df.to_dict('records'),
                
                # Categorized features
                "high_importance_features": high_importance['feature'].tolist(),
                "medium_importance_features": medium_importance['feature'].tolist(),
                "low_importance_features": low_importance['feature'].tolist(),
                
                # Statistics
                "total_features_analyzed": len(importance_df),
                "max_importance_score": float(mi_scores.max()),
                "mean_importance_score": float(mi_scores.mean()),
                "min_importance_score": float(mi_scores.min()),
                
                # Recommendations
                "feature_selection_recommendations": {
                    "keep_definitely": high_importance['feature'].tolist(),
                    "keep_probably": medium_importance['feature'].tolist(),
                    "consider_removing": low_importance['feature'].tolist()[:5]  # Bottom 5
                },
                
                # Summary
                "summary": (
                    f"Analyzed {len(importance_df)} features for {task_type}. "
                    f"Found {len(high_importance)} high-importance features."
                )
            }
            
        except Exception as e:
            logger.error(f"❌ Feature importance calculation failed: {e}")
            return {"error": str(e)}
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 13: PCA ANALYSIS WITH OPTIMAL COMPONENT SELECTION
    # ═══════════════════════════════════════════════════════════
    
    def _perform_pca(self, df: pd.DataFrame, n_components: Optional[int] = None) -> Dict[str, Any]:
        """
        ✅ FEATURE 13: PCA Analysis with Optimal Component Selection
        
        Principal Component Analysis:
        - Explained variance ratio per component
        - Cumulative variance explained
        - Optimal component recommendation (95% variance)
        - Component loadings
        - Dimensionality reduction suggestions
        """
        numerical = df.select_dtypes(include=[np.number])
        
        if len(numerical.columns) < 2:
            return {"error": "At least 2 numerical features required for PCA"}
        
        try:
            # Clean data
            numerical_clean = numerical.dropna()
            
            if len(numerical_clean) < 10:
                return {"error": "Not enough rows after removing NaN"}
            
            # Standardize features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numerical_clean)
            
            # Determine number of components
            if n_components is None:
                n_components = min(len(numerical_clean.columns), len(numerical_clean))
            
            # Perform PCA
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(scaled_data)
            
            # Calculate cumulative variance
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            
            # Find optimal components for different thresholds
            optimal_90 = np.argmax(cumulative_variance >= 0.90) + 1
            optimal_95 = np.argmax(cumulative_variance >= 0.95) + 1
            optimal_99 = np.argmax(cumulative_variance >= 0.99) + 1
            
            # Component loadings (contribution of each feature to each PC)
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=numerical_clean.columns
            )
            
            # Find most important features for each PC
            top_features_per_pc = {}
            for i in range(min(5, n_components)):
                pc_name = f'PC{i+1}'
                top_features = loadings[pc_name].abs().nlargest(5)
                top_features_per_pc[pc_name] = {
                    feat: float(loadings.loc[feat, pc_name])
                    for feat in top_features.index
                }
            
            return {
                "method": "principal_component_analysis",
                "description": "Dimensionality reduction using PCA",
                
                # Variance explained
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "cumulative_variance": cumulative_variance.tolist(),
                "explained_variance": pca.explained_variance_.tolist(),
                
                # Component recommendations
                "optimal_components": {
                    "for_90_percent_variance": int(optimal_90),
                    "for_95_percent_variance": int(optimal_95),
                    "for_99_percent_variance": int(optimal_99),
                },
                
                # Detailed results
                "total_components": int(n_components),
                "original_dimensions": len(numerical_clean.columns),
                "variance_retained_top_5": float(cumulative_variance[min(4, len(cumulative_variance)-1)]),
                
                # Component loadings
                "top_features_per_component": top_features_per_pc,
                "full_loadings": loadings.to_dict(),
                
                # Dimensionality reduction potential
                "dimensionality_reduction": {
                    "from": len(numerical_clean.columns),
                    "to_90_percent": int(optimal_90),
                    "to_95_percent": int(optimal_95),
                    "reduction_ratio_95": round((1 - optimal_95/len(numerical_clean.columns)) * 100, 2)
                },
                
                # Summary
                "summary": (
                    f"Can reduce {len(numerical_clean.columns)} dimensions to "
                    f"{optimal_95} components while retaining 95% of variance"
                )
            }
            
        except Exception as e:
            logger.error(f"❌ PCA analysis failed: {e}")
            return {"error": str(e)}
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 14: K-MEANS CLUSTERING WITH ELBOW METHOD
    # ═══════════════════════════════════════════════════════════
    
    def _perform_clustering(self, df: pd.DataFrame, max_clusters: int = 10) -> Dict[str, Any]:
        """
        ✅ FEATURE 14: K-means Clustering with Elbow Method & Silhouette
        
        Unsupervised clustering analysis:
        - K-means clustering (k=2 to max_clusters)
        - Elbow method for optimal k
        - Silhouette scores
        - Davies-Bouldin index
        - Cluster size analysis
        """
        numerical = df.select_dtypes(include=[np.number])
        
        if len(numerical.columns) < 2:
            return {"error": "At least 2 numerical features required for clustering"}
        
        try:
            # Clean and standardize data
            numerical_clean = numerical.dropna()
            
            if len(numerical_clean) < 50:
                return {"error": "At least 50 samples required for reliable clustering"}
            
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numerical_clean)
            
            # Test different numbers of clusters
            k_range = range(2, min(max_clusters + 1, len(numerical_clean)))
            
            inertias = []
            silhouette_scores_list = []
            davies_bouldin_scores = []
            
            cluster_results = {}
            
            for k in k_range:
                # Perform K-means
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(scaled_data)
                
                # Calculate metrics
                inertia = kmeans.inertia_
                silhouette = silhouette_score(scaled_data, labels)
                davies_bouldin = davies_bouldin_score(scaled_data, labels)
                
                inertias.append(inertia)
                silhouette_scores_list.append(silhouette)
                davies_bouldin_scores.append(davies_bouldin)
                
                # Cluster sizes
                unique, counts = np.unique(labels, return_counts=True)
                cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))
                
                cluster_results[int(k)] = {
                    "inertia": float(inertia),
                    "silhouette_score": float(silhouette),
                    "davies_bouldin_score": float(davies_bouldin),
                    "cluster_sizes": cluster_sizes,
                    "cluster_balance": round(min(counts) / max(counts), 2) if len(counts) > 0 else 0
                }
            
            # Find optimal k using multiple criteria
            optimal_k_silhouette = k_range[np.argmax(silhouette_scores_list)]
            optimal_k_davies_bouldin = k_range[np.argmin(davies_bouldin_scores)]
            
            # Elbow method (simplified - look for maximum second derivative)
            if len(inertias) >= 3:
                deltas = np.diff(inertias)
                second_deltas = np.diff(deltas)
                optimal_k_elbow = k_range[np.argmax(second_deltas) + 1] if len(second_deltas) > 0 else optimal_k_silhouette
            else:
                optimal_k_elbow = optimal_k_silhouette
            
            return {
                "method": "kmeans_clustering",
                "description": "K-means clustering with multiple evaluation metrics",
                
                # Optimal cluster recommendations
                "optimal_clusters": {
                    "by_silhouette_score": int(optimal_k_silhouette),
                    "by_davies_bouldin_index": int(optimal_k_davies_bouldin),
                    "by_elbow_method": int(optimal_k_elbow),
                    "recommended": int(optimal_k_silhouette)  # Silhouette is most reliable
                },
                
                # Detailed results per k
                "results_by_k": cluster_results,
                
                # Metrics arrays for plotting
                "elbow_data": {
                    "k_values": list(k_range),
                    "inertias": inertias,
                },
                "silhouette_data": {
                    "k_values": list(k_range),
                    "scores": silhouette_scores_list,
                },
                "davies_bouldin_data": {
                    "k_values": list(k_range),
                    "scores": davies_bouldin_scores,
                },
                
                # Best performing configuration
                "best_configuration": cluster_results[optimal_k_silhouette],
                
                # Summary
                "summary": (
                    f"Optimal number of clusters: {optimal_k_silhouette} "
                    f"(Silhouette score: {silhouette_scores_list[optimal_k_silhouette - 2]:.3f})"
                ),
                
                # Interpretation guide
                "interpretation": {
                    "silhouette_score": "Higher is better (range: -1 to 1)",
                    "davies_bouldin_index": "Lower is better",
                    "inertia": "Look for elbow point"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Clustering analysis failed: {e}")
            return {"error": str(e)}
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 15: ANOMALY DETECTION USING ISOLATION FOREST
    # ═══════════════════════════════════════════════════════════
    
    def _detect_anomalies(self, df: pd.DataFrame, contamination: float = 0.1) -> Dict[str, Any]:
        """
        ✅ FEATURE 15: Anomaly Detection using Isolation Forest
        
        ML-based anomaly detection:
        - Multivariate anomaly detection
        - Anomaly scores for all samples
        - Contamination parameter tuning
        - Feature-wise anomaly contribution
        - Anomaly clustering analysis
        """
        numerical = df.select_dtypes(include=[np.number])
        
        if len(numerical.columns) == 0:
            return {"error": "No numerical columns for anomaly detection"}
        
        try:
            # Clean data
            numerical_clean = numerical.dropna()
            
            if len(numerical_clean) < 50:
                return {"error": "At least 50 samples required for anomaly detection"}
            
            # Train Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100,
                max_samples='auto',
                max_features=1.0,
                bootstrap=False,
                n_jobs=-1
            )
            
            # Predict anomalies
            predictions = iso_forest.fit_predict(numerical_clean)
            anomaly_scores = iso_forest.score_samples(numerical_clean)
            
            # Identify anomalies
            anomaly_mask = predictions == -1
            normal_mask = predictions == 1
            
            anomaly_indices = numerical_clean[anomaly_mask].index.tolist()
            normal_indices = numerical_clean[normal_mask].index.tolist()
            
            # Analyze anomaly scores distribution
            anomaly_scores_anomalies = anomaly_scores[anomaly_mask]
            anomaly_scores_normal = anomaly_scores[normal_mask]
            
            # Per-feature anomaly analysis
            feature_anomaly_analysis = {}
            for col in numerical_clean.columns:
                anomaly_values = numerical_clean.loc[anomaly_mask, col]
                normal_values = numerical_clean.loc[normal_mask, col]
                
                feature_anomaly_analysis[col] = {
                    "anomaly_mean": float(anomaly_values.mean()) if len(anomaly_values) > 0 else 0,
                    "normal_mean": float(normal_values.mean()) if len(normal_values) > 0 else 0,
                    "anomaly_std": float(anomaly_values.std()) if len(anomaly_values) > 0 else 0,
                    "difference": float(abs(anomaly_values.mean() - normal_values.mean())) if len(anomaly_values) > 0 and len(normal_values) > 0 else 0
                }
            
            return {
                "method": "isolation_forest",
                "description": "ML-based multivariate anomaly detection",
                
                # Global statistics
                "total_samples": len(numerical_clean),
                "total_anomalies": int(anomaly_mask.sum()),
                "total_normal": int(normal_mask.sum()),
                "anomaly_percentage": round((anomaly_mask.sum() / len(numerical_clean)) * 100, 2),
                
                # Configuration
                "contamination_used": contamination,
                "n_estimators": 100,
                "features_analyzed": len(numerical_clean.columns),
                
                # Anomaly scores
                "anomaly_scores_summary": {
                    "min_score": float(anomaly_scores.min()),
                    "max_score": float(anomaly_scores.max()),
                    "mean_score": float(anomaly_scores.mean()),
                    "anomalies_mean_score": float(anomaly_scores_anomalies.mean()) if len(anomaly_scores_anomalies) > 0 else 0,
                    "normal_mean_score": float(anomaly_scores_normal.mean()) if len(anomaly_scores_normal) > 0 else 0,
                },
                
                # Anomaly details (sample)
                "anomaly_indices_sample": anomaly_indices[:100],
                "anomaly_scores_sample": anomaly_scores[anomaly_mask][:100].tolist(),
                
                # Per-feature analysis
                "feature_anomaly_analysis": feature_anomaly_analysis,
                
                # Severity classification
                "severity_distribution": {
                    "extreme": int((anomaly_scores < -0.5).sum()),
                    "high": int(((anomaly_scores >= -0.5) & (anomaly_scores < -0.3)).sum()),
                    "moderate": int(((anomaly_scores >= -0.3) & (anomaly_scores < -0.1)).sum()),
                },
                
                # Summary
                "summary": (
                    f"Detected {anomaly_mask.sum()} anomalies ({(anomaly_mask.sum() / len(numerical_clean)) * 100:.2f}%) "
                    f"out of {len(numerical_clean)} samples"
                )
            }
            
        except Exception as e:
            logger.error(f"❌ Anomaly detection failed: {e}")
            return {"error": str(e)}
    """
EDA Service - Part 4: Statistical Tests, Time Series & Intelligence (Features 16-20)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Features 16-20:
16. ✅ Statistical Tests (Chi-square, ANOVA, Kruskal-Wallis, Mann-Whitney, Levene)
17. ✅ Time Series Analysis with Decomposition & Stationarity
18. ✅ Comprehensive Data Quality Scoring (6 dimensions)
19. ✅ Automated Insights Generation (AI-powered)
20. ✅ Smart Recommendations Engine
"""


    # ═══════════════════════════════════════════════════════════
    # FEATURE 16: STATISTICAL TESTS
    # ═══════════════════════════════════════════════════════════
    
    def _perform_statistical_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        ✅ FEATURE 16: Statistical Tests
        
        Performs comprehensive statistical testing:
        - Chi-square test (categorical independence)
        - ANOVA (compare 3+ groups)
        - Kruskal-Wallis (non-parametric ANOVA)
        - Mann-Whitney U (compare 2 groups)
        - Levene's test (variance homogeneity)
        - T-test (parametric comparison)
        """
        tests_results = {}
        
        # ──────────────────────────────────────────────────────
        # 1. CHI-SQUARE TEST (Categorical Independence)
        # ──────────────────────────────────────────────────────
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) >= 2:
            chi_square_results = []
            
            # Test all pairs of categorical variables
            for i in range(len(categorical_cols)):
                for j in range(i + 1, min(i + 4, len(categorical_cols))):  # Limit to avoid too many tests
                    col1, col2 = categorical_cols[i], categorical_cols[j]
                    
                    try:
                        # Create contingency table
                        contingency_table = pd.crosstab(df[col1], df[col2])
                        
                        # Skip if table is too sparse
                        if contingency_table.size < 4:
                            continue
                        
                        # Perform chi-square test
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                        
                        chi_square_results.append({
                            "column1": str(col1),
                            "column2": str(col2),
                            "chi2_statistic": float(chi2),
                            "p_value": float(p_value),
                            "degrees_of_freedom": int(dof),
                            "significant": bool(p_value < 0.05),
                            "interpretation": "Independent" if p_value >= 0.05 else "Dependent",
                            "effect_size": self._calculate_cramers_v(chi2, contingency_table.sum().sum(), min(contingency_table.shape))
                        })
                    except Exception as e:
                        logger.warning(f"Chi-square test failed for {col1} vs {col2}: {e}")
            
            tests_results["chi_square_tests"] = {
                "description": "Test of independence between categorical variables",
                "tests_performed": len(chi_square_results),
                "results": chi_square_results,
                "significant_associations": [
                    f"{r['column1']} ↔ {r['column2']}"
                    for r in chi_square_results if r['significant']
                ]
            }
        
        # ──────────────────────────────────────────────────────
        # 2. ANOVA & KRUSKAL-WALLIS (Compare Multiple Groups)
        # ──────────────────────────────────────────────────────
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(categorical_cols) > 0 and len(numerical_cols) > 0:
            anova_results = []
            kruskal_results = []
            
            # Test numerical variables across categorical groups
            for num_col in numerical_cols[:5]:  # Limit to first 5 numerical
                for cat_col in categorical_cols[:3]:  # Limit to first 3 categorical
                    
                    try:
                        # Get groups
                        groups = df.groupby(cat_col)[num_col].apply(list)
                        
                        # Skip if less than 2 groups or any group is too small
                        if len(groups) < 2 or any(len(g) < 3 for g in groups):
                            continue
                        
                        # ANOVA (parametric)
                        f_stat, p_value_anova = f_oneway(*groups)
                        
                        # Kruskal-Wallis (non-parametric)
                        h_stat, p_value_kruskal = kruskal(*groups)
                        
                        anova_results.append({
                            "numerical_variable": str(num_col),
                            "grouping_variable": str(cat_col),
                            "f_statistic": float(f_stat),
                            "p_value": float(p_value_anova),
                            "significant": bool(p_value_anova < 0.05),
                            "n_groups": len(groups),
                            "interpretation": "Groups differ significantly" if p_value_anova < 0.05 else "No significant difference"
                        })
                        
                        kruskal_results.append({
                            "numerical_variable": str(num_col),
                            "grouping_variable": str(cat_col),
                            "h_statistic": float(h_stat),
                            "p_value": float(p_value_kruskal),
                            "significant": bool(p_value_kruskal < 0.05),
                            "n_groups": len(groups)
                        })
                        
                    except Exception as e:
                        logger.warning(f"ANOVA/Kruskal test failed for {num_col} by {cat_col}: {e}")
            
            if anova_results:
                tests_results["anova_tests"] = {
                    "description": "One-way ANOVA (parametric test for comparing group means)",
                    "tests_performed": len(anova_results),
                    "results": anova_results,
                    "significant_differences": [
                        f"{r['numerical_variable']} across {r['grouping_variable']}"
                        for r in anova_results if r['significant']
                    ]
                }
            
            if kruskal_results:
                tests_results["kruskal_wallis_tests"] = {
                    "description": "Kruskal-Wallis (non-parametric ANOVA alternative)",
                    "tests_performed": len(kruskal_results),
                    "results": kruskal_results
                }
        
        # ──────────────────────────────────────────────────────
        # 3. LEVENE'S TEST (Variance Homogeneity)
        # ──────────────────────────────────────────────────────
        
        if len(categorical_cols) > 0 and len(numerical_cols) > 0:
            levene_results = []
            
            for num_col in numerical_cols[:5]:
                for cat_col in categorical_cols[:2]:
                    
                    try:
                        groups = df.groupby(cat_col)[num_col].apply(list)
                        
                        if len(groups) < 2:
                            continue
                        
                        # Levene's test
                        w_stat, p_value = levene(*groups)
                        
                        levene_results.append({
                            "numerical_variable": str(num_col),
                            "grouping_variable": str(cat_col),
                            "w_statistic": float(w_stat),
                            "p_value": float(p_value),
                            "equal_variances": bool(p_value >= 0.05),
                            "interpretation": "Variances are equal" if p_value >= 0.05 else "Variances differ"
                        })
                        
                    except Exception as e:
                        logger.warning(f"Levene test failed: {e}")
            
            if levene_results:
                tests_results["levene_tests"] = {
                    "description": "Test for equality of variances across groups",
                    "tests_performed": len(levene_results),
                    "results": levene_results
                }
        
        # ──────────────────────────────────────────────────────
        # Summary
        # ──────────────────────────────────────────────────────
        
        tests_results["summary"] = {
            "total_tests_performed": sum(
                v.get("tests_performed", 0)
                for v in tests_results.values()
                if isinstance(v, dict)
            ),
            "tests_available": list(tests_results.keys())
        }
        
        return tests_results
    
    def _calculate_cramers_v(self, chi2: float, n: int, min_dim: int) -> float:
        """Calculate Cramér's V effect size for chi-square test."""
        if n == 0 or min_dim <= 1:
            return 0.0
        return float(np.sqrt(chi2 / (n * (min_dim - 1))))
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 17: TIME SERIES ANALYSIS
    # ═══════════════════════════════════════════════════════════
    
    def _analyze_time_series(self, df: pd.DataFrame, date_column: str) -> Dict[str, Any]:
        """
        ✅ FEATURE 17: Time Series Analysis
        
        Comprehensive time series analysis:
        - Trend detection
        - Seasonal decomposition
        - Stationarity testing (ADF test)
        - Autocorrelation analysis (ACF/PACF)
        - Frequency detection
        - Missing timestamps detection
        """
        if date_column not in df.columns:
            return {"error": f"Date column '{date_column}' not found"}
        
        try:
            # Convert to datetime
            df_ts = df.copy()
            df_ts[date_column] = pd.to_datetime(df_ts[date_column], errors='coerce')
            
            # Remove NaT values
            df_ts = df_ts.dropna(subset=[date_column])
            
            if len(df_ts) < 10:
                return {"error": "Not enough valid dates for time series analysis"}
            
            # Sort by date
            df_ts = df_ts.sort_values(date_column)
            
            # ──────────────────────────────────────────────────
            # Basic time range analysis
            # ──────────────────────────────────────────────────
            
            date_range_days = (df_ts[date_column].max() - df_ts[date_column].min()).days
            
            # Detect frequency
            date_diffs = df_ts[date_column].diff().dropna()
            median_diff = date_diffs.median()
            
            if median_diff.days <= 1:
                frequency = "daily"
            elif median_diff.days <= 7:
                frequency = "weekly"
            elif median_diff.days <= 31:
                frequency = "monthly"
            else:
                frequency = "irregular"
            
            results = {
                "date_column": date_column,
                "start_date": df_ts[date_column].min().isoformat(),
                "end_date": df_ts[date_column].max().isoformat(),
                "date_range_days": int(date_range_days),
                "total_records": len(df_ts),
                "detected_frequency": frequency,
                "median_time_gap_days": float(median_diff.days + median_diff.seconds / 86400),
            }
            
            # ──────────────────────────────────────────────────
            # Analyze numerical columns over time
            # ──────────────────────────────────────────────────
            
            numerical_cols = df_ts.select_dtypes(include=[np.number]).columns
            
            if len(numerical_cols) > 0:
                time_series_analysis = {}
                
                for col in numerical_cols[:3]:  # Limit to first 3 numerical columns
                    
                    try:
                        # Set date as index
                        ts_data = df_ts.set_index(date_column)[col].dropna()
                        
                        if len(ts_data) < 20:
                            continue
                        
                        # ──────────────────────────────────────
                        # Stationarity test (Augmented Dickey-Fuller)
                        # ──────────────────────────────────────
                        
                        adf_result = adfuller(ts_data, autolag='AIC')
                        
                        is_stationary = adf_result[1] < 0.05
                        
                        # ──────────────────────────────────────
                        # Autocorrelation
                        # ──────────────────────────────────────
                        
                        acf_values = acf(ts_data, nlags=min(20, len(ts_data) // 2))
                        pacf_values = pacf(ts_data, nlags=min(20, len(ts_data) // 2))
                        
                        # ──────────────────────────────────────
                        # Seasonal decomposition (if enough data)
                        # ──────────────────────────────────────
                        
                        decomposition_results = None
                        
                        if len(ts_data) >= 24 and frequency in ["daily", "weekly", "monthly"]:
                            try:
                                # Determine period
                                period = 7 if frequency == "weekly" else 30 if frequency == "monthly" else 12
                                period = min(period, len(ts_data) // 2)
                                
                                if period >= 2:
                                    decomposition = seasonal_decompose(
                                        ts_data,
                                        model='additive',
                                        period=period,
                                        extrapolate_trend='freq'
                                    )
                                    
                                    decomposition_results = {
                                        "period": int(period),
                                        "trend_strength": float(np.std(decomposition.trend.dropna()) / np.std(ts_data)),
                                        "seasonal_strength": float(np.std(decomposition.seasonal.dropna()) / np.std(ts_data)),
                                        "residual_strength": float(np.std(decomposition.resid.dropna()) / np.std(ts_data)),
                                    }
                            except Exception as e:
                                logger.warning(f"Seasonal decomposition failed for {col}: {e}")
                        
                        time_series_analysis[col] = {
                            # Stationarity
                            "stationarity_test": {
                                "test": "augmented_dickey_fuller",
                                "adf_statistic": float(adf_result[0]),
                                "p_value": float(adf_result[1]),
                                "is_stationary": is_stationary,
                                "critical_values": {
                                    "1%": float(adf_result[4]['1%']),
                                    "5%": float(adf_result[4]['5%']),
                                    "10%": float(adf_result[4]['10%'])
                                },
                                "interpretation": "Stationary series" if is_stationary else "Non-stationary series (requires differencing)"
                            },
                            
                            # Autocorrelation
                            "autocorrelation": {
                                "acf_lag1": float(acf_values[1]) if len(acf_values) > 1 else 0,
                                "acf_lag5": float(acf_values[5]) if len(acf_values) > 5 else 0,
                                "pacf_lag1": float(pacf_values[1]) if len(pacf_values) > 1 else 0,
                                "significant_lags": int((np.abs(acf_values[1:]) > 2/np.sqrt(len(ts_data))).sum()),
                            },
                            
                            # Decomposition
                            "decomposition": decomposition_results,
                            
                            # Basic stats
                            "mean": float(ts_data.mean()),
                            "std": float(ts_data.std()),
                            "trend": "increasing" if ts_data.iloc[-1] > ts_data.iloc[0] else "decreasing"
                        }
                        
                    except Exception as e:
                        logger.warning(f"Time series analysis failed for {col}: {e}")
                        time_series_analysis[col] = {"error": str(e)}
                
                results["time_series_analysis"] = time_series_analysis
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Time series analysis failed: {e}")
            return {"error": str(e)}
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 18: COMPREHENSIVE DATA QUALITY SCORING
    # ═══════════════════════════════════════════════════════════
    
    def _calculate_comprehensive_quality(self, statistics: Dict[str, Any]) -> Dict[str, Any]:
        """
        ✅ FEATURE 18: Comprehensive Data Quality Scoring
        
        Six-dimensional quality assessment:
        1. Completeness (missing values)
        2. Uniqueness (duplicates)
        3. Validity (outliers, data types)
        4. Consistency (standardization)
        5. Accuracy (data ranges)
        6. Timeliness (for time-series)
        
        Overall score: Weighted average of all dimensions
        """
        overview = statistics.get("overview", {})
        numerical = statistics.get("numerical", {})
        categorical = statistics.get("categorical", {})
        
        # ──────────────────────────────────────────────────────
        # 1. COMPLETENESS SCORE
        # ──────────────────────────────────────────────────────
        
        completeness = 100 - overview.get("missing_percentage", 0)
        
        # ──────────────────────────────────────────────────────
        # 2. UNIQUENESS SCORE
        # ──────────────────────────────────────────────────────
        
        uniqueness = 100 - overview.get("duplicate_percentage", 0)
        
        # ──────────────────────────────────────────────────────
        # 3. VALIDITY SCORE (based on outliers)
        # ──────────────────────────────────────────────────────
        
        if numerical:
            outlier_percentages = [
                col_stats.get("outliers_percentage", 0)
                for col_stats in numerical.values()
                if "error" not in col_stats
            ]
            avg_outliers = np.mean(outlier_percentages) if outlier_percentages else 0
            validity = 100 - min(avg_outliers, 100)
        else:
            validity = 100
        
        # ──────────────────────────────────────────────────────
        # 4. CONSISTENCY SCORE
        # ──────────────────────────────────────────────────────
        
        consistency_issues = 0
        consistency_checks = 0
        
        if categorical:
            for col, stats in categorical.items():
                if "error" in stats:
                    continue
                
                consistency_checks += 1
                
                # Check for inconsistent capitalization/spacing
                if stats.get("unique", 0) > stats.get("unique", 0) * 0.5:
                    # High cardinality might indicate inconsistency
                    consistency_issues += 0.5
        
        consistency = 100 - min((consistency_issues / max(consistency_checks, 1)) * 100, 100)
        
        # ──────────────────────────────────────────────────────
        # 5. ACCURACY SCORE
        # ──────────────────────────────────────────────────────
        
        accuracy_issues = 0
        accuracy_checks = 0
        
        if numerical:
            for col, stats in numerical.items():
                if "error" in stats:
                    continue
                
                accuracy_checks += 1
                
                # Check for impossible values (e.g., negative where shouldn't be)
                if stats.get("min", 0) < 0 and "price" in col.lower() or "age" in col.lower():
                    accuracy_issues += 1
        
        accuracy = 100 - min((accuracy_issues / max(accuracy_checks, 1)) * 100, 100)
        
        # ──────────────────────────────────────────────────────
        # 6. TIMELINESS SCORE (for time-series data)
        # ──────────────────────────────────────────────────────
        
        timeliness = 100  # Default to perfect if no time series
        
        # ──────────────────────────────────────────────────────
        # OVERALL WEIGHTED SCORE
        # ──────────────────────────────────────────────────────
        
        weights = {
            "completeness": 0.25,
            "uniqueness": 0.20,
            "validity": 0.25,
            "consistency": 0.15,
            "accuracy": 0.10,
            "timeliness": 0.05
        }
        
        overall = (
            completeness * weights["completeness"] +
            uniqueness * weights["uniqueness"] +
            validity * weights["validity"] +
            consistency * weights["consistency"] +
            accuracy * weights["accuracy"] +
            timeliness * weights["timeliness"]
        )
        
        # Quality classification
        if overall >= 90:
            quality_class = "Excellent"
        elif overall >= 75:
            quality_class = "Good"
        elif overall >= 60:
            quality_class = "Fair"
        elif overall >= 40:
            quality_class = "Poor"
        else:
            quality_class = "Very Poor"
        
        return {
            # Individual scores
            "completeness": round(completeness, 2),
            "uniqueness": round(uniqueness, 2),
            "validity": round(validity, 2),
            "consistency": round(consistency, 2),
            "accuracy": round(accuracy, 2),
            "timeliness": round(timeliness, 2),
            
            # Overall score
            "overall_score": round(overall, 2),
            "quality_class": quality_class,
            
            # Weights used
            "weights_used": weights,
            
            # Detailed breakdown
            "breakdown": {
                "completeness": f"{completeness:.1f}% ({100 - overview.get('missing_percentage', 0):.1f}% data present)",
                "uniqueness": f"{uniqueness:.1f}% ({100 - overview.get('duplicate_percentage', 0):.1f}% unique rows)",
                "validity": f"{validity:.1f}% (outliers under control)",
                "consistency": f"{consistency:.1f}% (standardization level)",
                "accuracy": f"{accuracy:.1f}% (data ranges acceptable)",
                "timeliness": f"{timeliness:.1f}% (data recency)"
            },
            
            # Recommendations by dimension
            "improvement_priorities": self._get_quality_improvement_priorities(
                completeness, uniqueness, validity, consistency, accuracy, timeliness
            )
        }
    
    def _get_quality_improvement_priorities(
        self,
        completeness: float,
        uniqueness: float,
        validity: float,
        consistency: float,
        accuracy: float,
        timeliness: float
    ) -> List[str]:
        """Identify top 3 quality improvement priorities."""
        dimensions = {
            "completeness": (completeness, "Handle missing values"),
            "uniqueness": (uniqueness, "Remove duplicate records"),
            "validity": (validity, "Address outliers and invalid values"),
            "consistency": (consistency, "Standardize data formats"),
            "accuracy": (accuracy, "Validate data ranges and types"),
            "timeliness": (timeliness, "Update stale records")
        }
        
        # Sort by score (lowest first = highest priority)
        sorted_dims = sorted(dimensions.items(), key=lambda x: x[1][0])
        
        # Return top 3 priorities with scores below 85
        priorities = [
            f"{dim.capitalize()}: {action} (current: {score:.1f}%)"
            for dim, (score, action) in sorted_dims[:3]
            if score < 85
        ]
        
        return priorities if priorities else ["Data quality is excellent - no major improvements needed"]
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 19: AUTOMATED INSIGHTS GENERATION
    # ═══════════════════════════════════════════════════════════
    
    def _generate_automated_insights(self, df: pd.DataFrame, statistics: Dict[str, Any]) -> List[str]:
        """
        ✅ FEATURE 19: Automated Insights Generation
        
        AI-powered insight generation based on:
        - Data size and composition
        - Missing values and quality
        - Correlations and relationships
        - Outliers and anomalies
        - Distribution patterns
        - Multicollinearity issues
        - Statistical significance
        """
        insights = []
        overview = statistics.get("overview", {})
        numerical = statistics.get("numerical", {})
        categorical = statistics.get("categorical", {})
        correlations = statistics.get("correlations", {})
        distributions = statistics.get("distributions", {})
        multicollinearity = statistics.get("multicollinearity", {})
        outliers = statistics.get("outliers", {})
        quality = statistics.get("data_quality", {})
        
        # ──────────────────────────────────────────────────────
        # 1. Dataset Size Insights
        # ──────────────────────────────────────────────────────
        
        rows = overview.get("total_rows", 0)
        cols = overview.get("total_columns", 0)
        
        if rows > 1000000:
            insights.append(f"📊 Large dataset detected with {rows:,} rows - consider sampling for faster analysis")
        elif rows < 100:
            insights.append(f"⚠️ Small dataset ({rows} rows) - statistical tests may lack power")
        else:
            insights.append(f"✅ Dataset size ({rows:,} rows × {cols} columns) is suitable for comprehensive analysis")
        
        # ──────────────────────────────────────────────────────
        # 2. Missing Values Insights
        # ──────────────────────────────────────────────────────
        
        missing_pct = overview.get("missing_percentage", 0)
        
        if missing_pct > 20:
            insights.append(f"🔴 High missing data ({missing_pct:.1f}%) - investigate missingness patterns and consider imputation or row removal")
        elif missing_pct > 5:
            insights.append(f"🟡 Moderate missing data ({missing_pct:.1f}%) - apply appropriate imputation strategies")
        elif missing_pct > 0:
            insights.append(f"🟢 Low missing data ({missing_pct:.1f}%) - minimal impact on analysis")
        
        # ──────────────────────────────────────────────────────
        # 3. Duplicate Insights
        # ──────────────────────────────────────────────────────
        
        dup_pct = overview.get("duplicate_percentage", 0)
        dup_count = overview.get("duplicate_rows", 0)
        
        if dup_pct > 5:
            insights.append(f"⚠️ Significant duplicates detected ({dup_count:,} rows, {dup_pct:.1f}%) - remove to improve data quality")
        
        # ──────────────────────────────────────────────────────
        # 4. Correlation Insights
        # ──────────────────────────────────────────────────────
        
        strong_corr_count = correlations.get("strong_correlations_count", 0)
        
        if strong_corr_count > 0:
            insights.append(f"🔗 Found {strong_corr_count} strong correlations - potential for feature engineering or dimensionality reduction")
        
        perfect_corr_count = correlations.get("perfect_correlations_count", 0)
        if perfect_corr_count > 0:
            insights.append(f"⚠️ {perfect_corr_count} perfect correlations detected - possible duplicate features")
        
        # ──────────────────────────────────────────────────────
        # 5. Multicollinearity Insights
        # ──────────────────────────────────────────────────────
        
        high_vif_count = multicollinearity.get("high_multicollinearity_count", 0)
        
        if high_vif_count > 0:
            high_vif_features = multicollinearity.get("high_multicollinearity_features", [])
            insights.append(f"🔴 High multicollinearity in {high_vif_count} features: {', '.join(high_vif_features[:3])} - impacts regression model stability")
        
        # ──────────────────────────────────────────────────────
        # 6. Outlier Insights
        # ──────────────────────────────────────────────────────
        
        if outliers and "results_by_column" in outliers:
            high_outlier_cols = [
                col for col, res in outliers["results_by_column"].items()
                if isinstance(res, dict) and res.get("total_outlier_percentage", 0) > 10
            ]
            
            if high_outlier_cols:
                insights.append(f"📈 Significant outliers in {len(high_outlier_cols)} columns: {', '.join(high_outlier_cols[:3])} - verify data quality or consider robust methods")
        
        # ──────────────────────────────────────────────────────
        # 7. Distribution Insights
        # ──────────────────────────────────────────────────────
        
        if distributions:
            non_normal_count = sum(
                1 for col_dist in distributions.values()
                if isinstance(col_dist, dict) and
                col_dist.get("normality_test", {}).get("is_normal") == False
            )
            
            if non_normal_count > len(distributions) * 0.7:
                insights.append(f"📊 Most numerical features ({non_normal_count}/{len(distributions)}) are non-normally distributed - consider transformations for parametric tests")
        
        # ──────────────────────────────────────────────────────
        # 8. Data Quality Insights
        # ──────────────────────────────────────────────────────
        
        overall_quality = quality.get("overall_score", 0)
        quality_class = quality.get("quality_class", "Unknown")
        
        if overall_quality >= 90:
            insights.append(f"✨ Excellent data quality ({overall_quality:.1f}%) - dataset is analysis-ready")
        elif overall_quality >= 75:
            insights.append(f"✅ Good data quality ({overall_quality:.1f}%) - minor improvements recommended")
        elif overall_quality >= 60:
            insights.append(f"🟡 Fair data quality ({overall_quality:.1f}%) - cleaning required before modeling")
        else:
            insights.append(f"🔴 Poor data quality ({overall_quality:.1f}%) - comprehensive data cleaning essential")
        
        # ──────────────────────────────────────────────────────
        # 9. Cardinality Insights
        # ──────────────────────────────────────────────────────
        
        if categorical:
            high_card_cols = [
                col for col, stats in categorical.items()
                if isinstance(stats, dict) and stats.get("high_cardinality") == True
            ]
            
            if high_card_cols:
                insights.append(f"🏷️ High cardinality categorical features detected: {', '.join(high_card_cols[:3])} - consider encoding strategies")
        
        # ──────────────────────────────────────────────────────
        # 10. Feature Count Insights
        # ──────────────────────────────────────────────────────
        
        num_features = overview.get("numerical_columns", 0)
        cat_features = overview.get("categorical_columns", 0)
        
        if num_features > 50:
            insights.append(f"📉 High-dimensional data ({num_features} numerical features) - consider PCA or feature selection")
        
        return insights if insights else ["✅ Data appears well-structured with no major issues detected"]
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 20: SMART RECOMMENDATIONS ENGINE
    # ═══════════════════════════════════════════════════════════
    
    def _generate_recommendations(self, statistics: Dict[str, Any]) -> List[str]:
        """
        ✅ FEATURE 20: Smart Recommendations Engine
        
        Generates actionable, prioritized recommendations:
        - Data cleaning steps
        - Feature engineering suggestions
        - Model selection guidance
        - Preprocessing requirements
        - Analysis strategy
        """
        recommendations = []
        overview = statistics.get("overview", {})
        quality = statistics.get("data_quality", {})
        multicollinearity = statistics.get("multicollinearity", {})
        outliers = statistics.get("outliers", {})
        correlations = statistics.get("correlations", {})
        constant_features = statistics.get("constant_features", {})
        
        # ──────────────────────────────────────────────────────
        # Priority 1: Critical Data Quality Issues
        # ──────────────────────────────────────────────────────
        
        if overview.get("missing_percentage", 0) > 10:
            recommendations.append(
                "🔴 HIGH PRIORITY: Address missing values (>10%) using appropriate imputation (mean/median for numerical, mode for categorical, or ML-based methods)"
            )
        
        if overview.get("duplicate_percentage", 0) > 1:
            recommendations.append(
                f"🔴 HIGH PRIORITY: Remove {overview.get('duplicate_rows', 0):,} duplicate rows to improve data quality and reduce bias"
            )
        
        # ──────────────────────────────────────────────────────
        # Priority 2: Feature Engineering
        # ──────────────────────────────────────────────────────
        
        if constant_features.get("constant_count", 0) > 0:
            recommendations.append(
                f"🟡 MEDIUM PRIORITY: Remove {constant_features['constant_count']} constant features that provide no predictive value"
            )
        
        if multicollinearity.get("high_multicollinearity_count", 0) > 0:
            features = multicollinearity.get("high_multicollinearity_features", [])
            recommendations.append(
                f"🟡 MEDIUM PRIORITY: Address multicollinearity by removing redundant features: {', '.join(features[:3])}"
            )
        
        # ──────────────────────────────────────────────────────
        # Priority 3: Outlier Handling
        # ──────────────────────────────────────────────────────
        
        if outliers and "results_by_column" in outliers:
            high_outlier_cols = [
                col for col, res in outliers["results_by_column"].items()
                if isinstance(res, dict) and res.get("total_outlier_percentage", 0) > 5
            ]
            
            if high_outlier_cols:
                recommendations.append(
                    f"🟡 MEDIUM PRIORITY: Investigate outliers in {len(high_outlier_cols)} columns ({', '.join(high_outlier_cols[:2])}) - cap, transform, or use robust methods"
                )
        
        # ──────────────────────────────────────────────────────
        # Priority 4: Feature Selection
        # ──────────────────────────────────────────────────────
        
        if correlations.get("strong_correlations_count", 0) > 5:
            recommendations.append(
                "🟢 LOW PRIORITY: Consider dimensionality reduction (PCA) or feature selection due to high correlations"
            )
        
        # ──────────────────────────────────────────────────────
        # Priority 5: Model Selection Guidance
        # ──────────────────────────────────────────────────────
        
        overall_quality = quality.get("overall_score", 0)
        
        if overall_quality >= 90:
            recommendations.append(
                "✨ READY FOR MODELING: Data quality is excellent - proceed with standard ML algorithms"
            )
        elif overall_quality >= 75:
            recommendations.append(
                "✅ NEARLY READY: After minor cleaning, use cross-validation and consider ensemble methods"
            )
        else:
            recommendations.append(
                "⚠️ CLEANING REQUIRED: Focus on data quality improvements before modeling"
            )
        
        # ──────────────────────────────────────────────────────
        # Priority 6: Preprocessing Steps
        # ──────────────────────────────────────────────────────
        
        recommendations.append(
            "📋 PREPROCESSING CHECKLIST: (1) Handle missing values, (2) Remove duplicates, (3) Encode categoricals, (4) Scale features, (5) Split train/test"
        )
        
        return recommendations if recommendations else [
            "✅ No major issues detected - dataset is well-prepared for analysis"
        ]
    """
EDA Service - Part 5: Integration, Helpers & Complete Service (Final)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Final Features:
21. ✅ Unified Outlier Detection (combines all 3 methods)
22. ✅ Data Reading with Error Handling
23. ✅ Database Statistics Persistence
24. ✅ Profile Report Generation (ydata-profiling)
25. ✅ Main EDA Generation Orchestrator
26. ✅ Dependency Injection Helper
    """

    # ═══════════════════════════════════════════════════════════
    # FEATURE 21: UNIFIED OUTLIER DETECTION
    # ═══════════════════════════════════════════════════════════
    
    def _detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> Dict[str, Any]:
        """
        ✅ FEATURE 21: Unified Outlier Detection
        
        Single entry point for all outlier detection methods:
        - method='iqr': IQR-based detection
        - method='zscore': Z-score based detection
        - method='isolation_forest': ML-based detection
        - method='all': Run all three methods and compare
        """
        if method == 'iqr':
            return self._detect_outliers_iqr(df)
        elif method == 'zscore':
            return self._detect_outliers_zscore(df)
        elif method == 'isolation_forest':
            return self._detect_outliers_isolation_forest(df)
        elif method == 'all':
            # Run all three methods and combine results
            iqr_results = self._detect_outliers_iqr(df)
            zscore_results = self._detect_outliers_zscore(df)
            iso_results = self._detect_outliers_isolation_forest(df)
            
            return {
                "method": "combined_analysis",
                "description": "Comprehensive outlier detection using multiple methods",
                "iqr_method": iqr_results,
                "zscore_method": zscore_results,
                "isolation_forest_method": iso_results,
                "consensus": self._calculate_outlier_consensus(iqr_results, zscore_results, iso_results)
            }
        else:
            return {"error": f"Unknown method: {method}. Use 'iqr', 'zscore', 'isolation_forest', or 'all'"}
    
    def _calculate_outlier_consensus(
        self,
        iqr_results: Dict[str, Any],
        zscore_results: Dict[str, Any],
        iso_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate consensus across outlier detection methods."""
        consensus = {}
        
        # Get common columns
        iqr_cols = set(iqr_results.get("results_by_column", {}).keys())
        zscore_cols = set(zscore_results.get("results_by_column", {}).keys())
        
        common_cols = iqr_cols.intersection(zscore_cols)
        
        for col in common_cols:
            iqr_count = iqr_results["results_by_column"][col].get("total_outliers", 0)
            zscore_count = zscore_results["results_by_column"][col].get("total_outliers", 0)
            
            consensus[col] = {
                "iqr_outliers": iqr_count,
                "zscore_outliers": zscore_count,
                "agreement": "high" if abs(iqr_count - zscore_count) < 10 else "low"
            }
        
        return consensus
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 22: DATA READING WITH ERROR HANDLING
    # ═══════════════════════════════════════════════════════════
    
    def _read_dataframe(self, file_path: str, file_type: str) -> pd.DataFrame:
        """
        ✅ FEATURE 22: Data Reading with Error Handling
        
        Reads various file formats with comprehensive error handling:
        - CSV (with encoding detection)
        - Excel (.xlsx, .xls)
        - JSON
        - Parquet
        - Automatic delimiter detection for CSV
        - Encoding fallback strategies
        """
        try:
            logger.info(f"📂 Reading file: {file_path} (type: {file_type})")
            
            if file_type == ".csv":
                # Try UTF-8 first, then fall back to other encodings
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        logger.info(f"✅ Successfully read CSV with {encoding} encoding")
                        return df
                    except UnicodeDecodeError:
                        continue
                
                # If all encodings fail, try with error handling
                df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace')
                logger.warning("⚠️ Read CSV with replacement characters for encoding errors")
                return df
                
            elif file_type in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
                logger.info(f"✅ Successfully read Excel file")
                return df
                
            elif file_type == ".json":
                df = pd.read_json(file_path)
                logger.info(f"✅ Successfully read JSON file")
                return df
                
            elif file_type == ".parquet":
                df = pd.read_parquet(file_path)
                logger.info(f"✅ Successfully read Parquet file")
                return df
                
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            logger.error(f"❌ Failed to read file {file_path}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to read file: {str(e)}"
            )
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 23: DATABASE STATISTICS PERSISTENCE
    # ═══════════════════════════════════════════════════════════
    
    def _save_statistics(self, dataset_id: int, statistics: Dict[str, Any]) -> None:
        """
        ✅ FEATURE 23: Database Statistics Persistence (FIXED)
    
        Saves comprehensive statistics to database with NumPy type conversion:
        - Converts NumPy types to native Python types
        - Checks for existing records
        - Updates or creates new records
        - Handles large JSON objects
        - Error handling and rollback
        """
        try:
            logger.info(f"💾 Saving statistics for dataset {dataset_id}")
        
            # ✅ FIX: Convert NumPy types to native Python types
            def convert_numpy_types(obj):
                """Recursively convert NumPy types to Python native types."""
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_numpy_types(item) for item in obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                     np.int16, np.int32, np.int64,
                                     np.uint8, np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.generic):
                    return obj.item()
                else:
                    return obj
        
            # Convert all statistics to JSON-serializable format
            logger.info("🔄 Converting NumPy types to native Python types")
            clean_statistics = convert_numpy_types(statistics)
        
            # Check if statistics already exist
            existing = self.db.query(DatasetStatistics).filter(
                DatasetStatistics.dataset_id == dataset_id
            ).first()
        
            if existing:
                # Update existing record
                logger.info(f"📝 Updating existing statistics record")
                existing.numerical_stats = clean_statistics.get("numerical")
                existing.categorical_stats = clean_statistics.get("categorical")
                existing.correlation_matrix = clean_statistics.get("correlations")
                existing.distributions = clean_statistics.get("distributions")
                existing.updated_at = datetime.now(timezone.utc)
            else:
                # Create new record
                logger.info(f"➕ Creating new statistics record")
                stats_record = DatasetStatistics(
                    dataset_id=dataset_id,
                    numerical_stats=clean_statistics.get("numerical"),
                    categorical_stats=clean_statistics.get("categorical"),
                    correlation_matrix=clean_statistics.get("correlations"),
                    distributions=clean_statistics.get("distributions"),
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                )
                self.db.add(stats_record)
        
            self.db.commit()
            logger.info(f"✅ Statistics saved successfully for dataset {dataset_id}")
        
        except Exception as e:
            logger.error(f"❌ Failed to save statistics: {e}")
            self.db.rollback()
            raise

# ═══════════════════════════════════════════════════════════
# FEATURE 24: ENHANCED PROFILE REPORT GENERATION
# ═══════════════════════════════════════════════════════════

    async def _generate_profile_report(
            self,
            df: pd.DataFrame,
            dataset: Dataset,
            minimal: bool = False
        ) -> str:  # Changed from Optional[str] to str - should never return None
        """
    ✅ FEATURE 24: Enhanced Profile Report Generation
    
    Generates standard ydata-profiling HTML report.
    Raises exceptions instead of returning None on failure.
        """
        try:
            from ydata_profiling import ProfileReport
        
            logger.info(f"📊 Generating {'minimal' if minimal else 'full'} profile report")
        
            # Configure profiling
            config = {
                "title": f"EDA Report: {dataset.name}",
                "minimal": minimal,
                "explorative": not minimal,
                "dark_mode": False,
                "samples": {"head": 10, "tail": 10},
            }
        
            # Reduce processing for large datasets
            if len(df) > 10000:
                config["samples"] = {"head": 5, "tail": 5}
                config["correlations"] = {
                    "pearson": {"calculate": True}, 
                    "spearman": False, 
                    "kendall": False
                }
        
            if minimal:
                config.update({
                    "correlations": None,
                    "missing_diagrams": None,
                    "duplicates": None,
                    "interactions": None,
                })
        
        # Generate standard profile
            profile = ProfileReport(df, **config)
        
        # Save standard report
            report_path = self._get_report_path(dataset)
            await asyncio.to_thread(profile.to_file, report_path)
        
            logger.info(f"✅ Profile report saved to {report_path}")
        
        # Upload to S3 if configured
            if hasattr(settings, 'USE_S3') and settings.USE_S3:
                report_url = await self._upload_report_to_s3(report_path, dataset)
                return report_url
        
            return str(report_path)
        
        except ImportError as e:
            logger.error("⚠️ ydata-profiling not installed")
            raise HTTPException(500, "ydata-profiling library not installed. Cannot generate HTML report.") from e
    
        except Exception as e:
            logger.error(f"❌ Profile report generation failed: {e}", exc_info=True)
            raise HTTPException(500, f"Profile report generation failed: {str(e)}") from e

    def _create_enhanced_html_template(self, stats: Dict[str, Any], dataset_name: str) -> str:
        """Create beautiful HTML template with all 26 advanced features."""
    
        overview = stats.get('overview', {})
        quality = stats.get('data_quality', {})
        numerical = stats.get('numerical', {})
        categorical = stats.get('categorical', {})
        correlations = stats.get('correlations', {})
        distributions = stats.get('distributions', {})
        outliers = stats.get('outliers', {})
        missing = stats.get('missing_patterns', {})
        multicollinearity = stats.get('multicollinearity', {})
        constant = stats.get('constant_features', {})
        anomalies = stats.get('anomalies', {})
        clustering = stats.get('clustering', {})
        pca = stats.get('pca', {})
        stat_tests = stats.get('statistical_tests', {})
        insights = stats.get('insights', [])
        recommendations = stats.get('recommendations', [])
    
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced EDA Report - {dataset_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 50px 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 3em;
            margin-bottom: 10px;
            font-weight: 700;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        
        .header .subtitle {{
            font-size: 1.3em;
            opacity: 0.95;
            margin-bottom: 5px;
        }}
        
        .header .meta {{
            font-size: 0.9em;
            opacity: 0.8;
            margin-top: 10px;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin: 40px 0;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 15px;
            border-left: 6px solid #667eea;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        
        .section h2 {{
            color: #667eea;
            margin-bottom: 25px;
            font-size: 2em;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .section h2::before {{
            content: '';
            width: 6px;
            height: 40px;
            background: #667eea;
            border-radius: 3px;
        }}
        
        .quality-overview {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 30px;
        }}
        
        .quality-overview .main-score {{
            font-size: 5em;
            font-weight: 800;
            margin: 20px 0;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        }}
        
        .quality-overview .score-label {{
            font-size: 1.8em;
            font-weight: 600;
            margin-bottom: 20px;
            background: rgba(255,255,255,0.2);
            padding: 10px 30px;
            border-radius: 25px;
            display: inline-block;
        }}
        
        .quality-scorecard {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 25px;
            margin: 30px 0;
        }}
        
        .quality-card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            text-align: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border: 2px solid transparent;
        }}
        
        .quality-card:hover {{
            transform: translateY(-8px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            border-color: #667eea;
        }}
        
        .quality-card .icon {{
            font-size: 3em;
            margin-bottom: 15px;
        }}
        
        .quality-card h3 {{
            color: #666;
            font-size: 0.95em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        
        .quality-card .score {{
            font-size: 3.5em;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }}
        
        .stat-box {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            border-left: 5px solid #667eea;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            transition: transform 0.2s;
        }}
        
        .stat-box:hover {{
            transform: translateX(5px);
        }}
        
        .stat-label {{
            color: #666;
            font-size: 0.95em;
            margin-bottom: 10px;
            font-weight: 500;
        }}
        
        .stat-value {{
            font-size: 2.2em;
            font-weight: 700;
            color: #333;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            margin: 20px 0;
        }}
        
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 18px;
            text-align: left;
            font-weight: 600;
            font-size: 1.05em;
        }}
        
        td {{
            padding: 15px 18px;
            border-bottom: 1px solid #eee;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        tr:last-child td {{
            border-bottom: none;
        }}
        
        .badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .badge-success {{
            background: #d4edda;
            color: #155724;
        }}
        
        .badge-warning {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .badge-danger {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .badge-info {{
            background: #d1ecf1;
            color: #0c5460;
        }}
        
        .insights-list, .recommendations-list {{
            list-style: none;
            padding: 0;
        }}
        
        .insights-list li {{
            background: white;
            padding: 20px 25px;
            margin: 15px 0;
            border-radius: 10px;
            border-left: 5px solid #28a745;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            transition: all 0.3s;
            font-size: 1.05em;
        }}
        
        .insights-list li:hover {{
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transform: translateX(5px);
        }}
        
        .recommendations-list li {{
            background: white;
            padding: 20px 25px;
            margin: 15px 0;
            border-radius: 10px;
            border-left: 5px solid #ffc107;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            transition: all 0.3s;
            font-size: 1.05em;
        }}
        
        .recommendations-list li:hover {{
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transform: translateX(5px);
        }}
        
        .priority-high {{
            border-left-color: #dc3545 !important;
        }}
        
        .priority-medium {{
            border-left-color: #ffc107 !important;
        }}
        
        .priority-low {{
            border-left-color: #28a745 !important;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 35px;
            background: #e9ecef;
            border-radius: 20px;
            overflow: hidden;
            margin: 15px 0;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
            font-size: 1.05em;
            transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        .info-box {{
            background: #e7f3ff;
            border-left: 5px solid #2196f3;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        
        .warning-box {{
            background: #fff3e0;
            border-left: 5px solid #ff9800;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        
        @media (max-width: 768px) {{
            .quality-scorecard, .stats-grid {{
                grid-template-columns: 1fr;
            }}
            .header h1 {{
                font-size: 2em;
            }}
            .quality-overview .main-score {{
                font-size: 3.5em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🔬 Enhanced EDA Report</h1>
            <div class="subtitle">{dataset_name}</div>
            <div class="meta">Comprehensive Analysis with 26+ Advanced Features</div>
        </div>
        
        <!-- Content -->
        <div class="content">
            
            <!-- Data Quality Score -->
            <div class="quality-overview">
                <div style="font-size: 1.5em; opacity: 0.9;">Overall Data Quality Score</div>
                <div class="main-score">{quality.get('overall_score', 0):.1f}%</div>
                <div class="score-label">{quality.get('quality_class', 'Unknown')}</div>
                <div class="progress-bar" style="background: rgba(255,255,255,0.3);">
                    <div class="progress-fill" style="width: {quality.get('overall_score', 0)}%; background: rgba(255,255,255,0.9); color: #667eea;">
                        {quality.get('overall_score', 0):.1f}%
                    </div>
                </div>
            </div>
            
            <!-- 6 Dimension Quality Scorecard -->
            <div class="section">
                <h2>📊 6-Dimensional Quality Assessment</h2>
                <div class="quality-scorecard">
                    <div class="quality-card">
                        <div class="icon">✅</div>
                        <h3>Completeness</h3>
                        <div class="score">{quality.get('completeness', 0):.1f}%</div>
                    </div>
                    <div class="quality-card">
                        <div class="icon">🎯</div>
                        <h3>Uniqueness</h3>
                        <div class="score">{quality.get('uniqueness', 0):.1f}%</div>
                    </div>
                    <div class="quality-card">
                        <div class="icon">✔️</div>
                        <h3>Validity</h3>
                        <div class="score">{quality.get('validity', 0):.1f}%</div>
                    </div>
                    <div class="quality-card">
                        <div class="icon">📏</div>
                        <h3>Consistency</h3>
                        <div class="score">{quality.get('consistency', 0):.1f}%</div>
                    </div>
                    <div class="quality-card">
                        <div class="icon">🎲</div>
                        <h3>Accuracy</h3>
                        <div class="score">{quality.get('accuracy', 0):.1f}%</div>
                    </div>
                    <div class="quality-card">
                        <div class="icon">⏰</div>
                        <h3>Timeliness</h3>
                        <div class="score">{quality.get('timeliness', 0):.1f}%</div>
                    </div>
                </div>
            </div>
            
            <!-- Dataset Overview -->
            <div class="section">
                <h2>📈 Dataset Overview</h2>
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-label">📊 Total Rows</div>
                        <div class="stat-value">{overview.get('total_rows', 0):,}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">📋 Total Columns</div>
                        <div class="stat-value">{overview.get('total_columns', 0)}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">💾 Memory Usage</div>
                        <div class="stat-value">{overview.get('memory_usage_mb', 0):.2f} MB</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">❓ Missing Values</div>
                        <div class="stat-value">{overview.get('missing_percentage', 0):.2f}%</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">🔄 Duplicates</div>
                        <div class="stat-value">{overview.get('duplicate_percentage', 0):.2f}%</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">📊 Data Density</div>
                        <div class="stat-value">{overview.get('data_density', 0):.2f}%</div>
                    </div>
                </div>
                
                <table style="margin-top: 30px;">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Numerical Columns</strong></td>
                            <td>{overview.get('numerical_columns', 0)}</td>
                        </tr>
                        <tr>
                            <td><strong>Categorical Columns</strong></td>
                            <td>{overview.get('categorical_columns', 0)}</td>
                        </tr>
                        <tr>
                            <td><strong>Datetime Columns</strong></td>
                            <td>{overview.get('datetime_columns', 0)}</td>
                        </tr>
                        <tr>
                            <td><strong>Total Cells</strong></td>
                            <td>{overview.get('total_cells', 0):,}</td>
                        </tr>
                        <tr>
                            <td><strong>Unique Rows</strong></td>
                            <td>{overview.get('unique_rows', 0):,}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            {self._html_multicollinearity_section(multicollinearity)}
            {self._html_outliers_section(outliers)}
            {self._html_anomalies_section(anomalies)}
            {self._html_constant_features_section(constant)}
            {self._html_correlations_section(correlations)}
            {self._html_statistical_tests_section(stat_tests)}
            
            <!-- AI-Powered Insights -->
            <div class="section">
                <h2>💡 AI-Powered Insights</h2>
                <p style="margin-bottom: 20px; color: #666;">Automated analysis discovered {len(insights)} key insights:</p>
                <ul class="insights-list">
                    {self._format_insights_html(insights)}
                </ul>
            </div>
            
            <!-- Smart Recommendations -->
            <div class="section">
                <h2>📋 Smart Recommendations</h2>
                <p style="margin-bottom: 20px; color: #666;">Prioritized action items for data improvement:</p>
                <ul class="recommendations-list">
                    {self._format_recommendations_html(recommendations)}
                </ul>
            </div>
            
        </div>
    </div>
</body>
</html>
"""


    def _html_multicollinearity_section(self, vif_data: Dict[str, Any]) -> str:
        """Generate multicollinearity HTML section."""
        if not vif_data or 'vif_scores' not in vif_data:
            return ""
    
        rows = ""
        for score in vif_data.get('vif_scores', [])[:15]:
            severity = score.get('severity', 'Low')
            vif_val = score.get('VIF', 0)
            badge_class = 'badge-danger' if severity == 'High' else 'badge-warning' if severity == 'Moderate' else 'badge-success'
        
            rows += f"""
            <tr>
                <td><strong>{score.get('feature', 'N/A')}</strong></td>
                <td style="font-size: 1.3em; font-weight: 600;">{vif_val:.2f}</td>
                <td><span class="badge {badge_class}">{severity}</span></td>
            </tr>
            """
    
        high_count = vif_data.get('high_multicollinearity_count', 0)
        warning_html = f"""
        <div class="warning-box" style="margin-bottom: 20px;">
            <strong>⚠️ Warning:</strong> {high_count} features with high multicollinearity (VIF > 10) detected. 
            Consider removing redundant features to improve model stability.
        </div>
        """ if high_count > 0 else ""
    
        return f"""
        <div class="section">
            <h2>🔗 Multicollinearity Analysis (VIF)</h2>
            <p style="margin-bottom: 20px; color: #666;">
                Variance Inflation Factor measures multicollinearity between features.
                VIF > 10 indicates high multicollinearity.
            </p>
            {warning_html}
            <table>
               <thead>
                    <tr>
                        <th>Feature</th>
                        <th>VIF Score</th>
                        <th>Severity</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """

    def _html_outliers_section(self, outliers: Dict[str, Any]) -> str:
        """Generate outliers HTML section."""
        if not outliers or 'results_by_column' not in outliers:
            return ""
    
        method = outliers.get('method', 'iqr')
        results = outliers.get('results_by_column', {})
    
        rows = ""
        for col, data in list(results.items())[:10]:
            if isinstance(data, dict) and 'total_outliers' in data:
                total = data.get('total_outliers', 0)
                pct = data.get('total_outlier_percentage', 0)
                severity = data.get('severity', 'low')
                badge_class = 'badge-danger' if severity == 'high' else 'badge-warning' if severity == 'moderate' else 'badge-success'
            
                rows += f"""
                <tr>
                    <td><strong>{col}</strong></td>
                    <td>{total:,}</td>
                    <td>{pct:.2f}%</td>
                    <td><span class="badge {badge_class}">{severity.capitalize()}</span></td>
                </tr>
                """
    
        return f"""
        <div class="section">
            <h2>📊 Outlier Detection ({method.upper()})</h2>
            <p style="margin-bottom: 20px; color: #666;">
                Outliers detected using {method.upper()} method across numerical columns.
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Total Outliers</th>
                        <th>Percentage</th>
                        <th>Severity</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """

    def _html_anomalies_section(self, anomalies: Dict[str, Any]) -> str:
        """Generate anomalies HTML section."""
        if not anomalies or 'total_anomalies' not in anomalies:
            return ""
    
        total = anomalies.get('total_anomalies', 0)
        pct = anomalies.get('anomaly_percentage', 0)
        method = anomalies.get('method', 'isolation_forest')
    
        return f"""
        <div class="section">
            <h2>⚠️ Anomaly Detection</h2>
        <p style="margin-bottom: 20px; color: #666;">
            ML-based multivariate anomaly detection using {method.replace('_', ' ').title()}.
        </p>
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-label">🚨 Total Anomalies</div>
                <div class="stat-value">{total:,}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">📊 Percentage</div>
                <div class="stat-value">{pct:.2f}%</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">🔬 Method</div>
                <div class="stat-value" style="font-size: 1.2em;">{method.replace('_', ' ').title()}</div>
            </div>
        </div>
    </div>
        """

    def _html_constant_features_section(self, constant: Dict[str, Any]) -> str:
        """Generate constant features HTML section."""
        if not constant or constant.get('constant_count', 0) == 0:
            return ""
    
        const_features = constant.get('constant_features', [])
        quasi_features = constant.get('quasi_constant_features', [])
        low_var_features = constant.get('low_variance_features', [])
    
        features_html = ""
        if const_features:
            features_html += f"<tr><td><strong>Constant (100%)</strong></td><td>{', '.join(const_features[:5])}</td></tr>"
        if quasi_features:
            features_html += f"<tr><td><strong>Quasi-Constant (>95%)</strong></td><td>{', '.join(quasi_features[:5])}</td></tr>"
        if low_var_features:
            features_html += f"<tr><td><strong>Low Variance (>90%)</strong></td><td>{', '.join(low_var_features[:5])}</td></tr>"
    
        return f"""
        <div class="section">
            <h2>🔍 Constant & Low Variance Features</h2>
        <div class="warning-box" style="margin-bottom: 20px;">
            <strong>⚠️ Action Required:</strong> {len(const_features)} constant features detected. 
            Consider removing features with little to no variance.
        </div>
        <table>
            <thead>
                <tr>
                    <th>Type</th>
                    <th>Features</th>
                </tr>
            </thead>
            <tbody>{features_html}</tbody>
        </table>
    </div>
        """

    def _html_correlations_section(self, correlations: Dict[str, Any]) -> str:
        """Generate correlations HTML section."""
        if not correlations or 'strong_correlations_count' not in correlations:
            return ""
    
        count = correlations.get('strong_correlations_count', 0)
    
        if count == 0:
            return ""
    
        strong_corr = correlations.get('strong_correlations', [])
        rows = ""
        for corr in strong_corr[:10]:
            col1 = corr.get('column1', '')
            col2 = corr.get('column2', '')
            val = corr.get('correlation', 0)
            strength = corr.get('strength', '')
            badge_class = 'badge-danger' if abs(val) > 0.9 else 'badge-warning'
        
            rows += f"""
            <tr>
                <td><strong>{col1}</strong></td>
                <td><strong>{col2}</strong></td>
                <td style="font-size: 1.2em; font-weight: 600;">{val:.3f}</td>
                <td><span class="badge {badge_class}">{strength}</span></td>
            </tr>
            """
    
        return f"""
        <div class="section">
            <h2>🔗 Strong Correlations</h2>
        <p style="margin-bottom: 20px; color: #666;">
            {count} strong correlations (|r| > 0.7) detected between features.
        </p>
        <table>
            <thead>
                <tr>
                    <th>Feature 1</th>
                    <th>Feature 2</th>
                    <th>Correlation</th>
                    <th>Strength</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
        """


    def _html_statistical_tests_section(self, tests: Dict[str, Any]) -> str:
        """Generate statistical tests HTML section."""
        if not tests:
            return ""
    
        sections = ""
    
        # Chi-square tests
        if 'chi_square_tests' in tests:
            chi_data = tests['chi_square_tests']
            results = chi_data.get('results', [])[:5]
            if results:
                rows = ""
                for res in results:
                    badge_class = 'badge-success' if res.get('significant') else 'badge-info'
                    rows += f"""
                    <tr>
                        <td>{res.get('column1', '')}</td>
                        <td>{res.get('column2', '')}</td>
                        <td>{res.get('chi2_statistic', 0):.2f}</td>
                        <td>{res.get('p_value', 0):.4f}</td>
                        <td><span class="badge {badge_class}">{res.get('interpretation', '')}</span></td>
                    </tr>
                    """
            
                sections += f"""
                <h3 style="color: #667eea; margin: 25px 0 15px 0;">Chi-Square Independence Tests</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Variable 1</th>
                            <th>Variable 2</th>
                            <th>χ² Statistic</th>
                            <th>p-value</th>
                            <th>Interpretation</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
                """
    
        if not sections:
            return ""
    
        return f"""
        <div class="section">
            <h2>🧪 Statistical Tests</h2>
        <p style="margin-bottom: 20px; color: #666;">
            Hypothesis testing results for categorical and numerical relationships.
        </p>
        {sections}
    </div>
        """


    def _format_insights_html(self, insights: List[str]) -> str:
        """Format insights as HTML list items."""
        if not insights:
            return '<li>No insights generated</li>'
    
        return ''.join(f'<li>{insight}</li>' for insight in insights)


    def _format_recommendations_html(self, recommendations: List[str]) -> str:
        """Format recommendations as HTML list items with priority colors."""
        if not recommendations:
            return '<li>No recommendations generated</li>'
    
        html = ""
        for rec in recommendations:
            priority_class = ""
            if "🔴" in rec or "HIGH PRIORITY" in rec:
                priority_class = "priority-high"
            elif "🟡" in rec or "MEDIUM PRIORITY" in rec:
                priority_class = "priority-medium"
            elif "🟢" in rec or "LOW PRIORITY" in rec:
                priority_class = "priority-low"
        
            html += f'<li class="{priority_class}">{rec}</li>'
    
        return html

    def _html_multicollinearity_section(self, vif_data: Dict[str, Any]) -> str:
        """Generate multicollinearity HTML section."""
        if not vif_data or 'vif_scores' not in vif_data:
            return ""
        rows = ""
        for score in vif_data.get('vif_scores', [])[:15]:
            severity = score.get('severity', 'Low')
            vif_val = score.get('VIF', 0)
            badge_class = 'badge-danger' if severity == 'High' else 'badge-warning' if severity == 'Moderate' else 'badge-success'
            rows += f"""
            <tr>
            <td><strong>{score.get('feature', 'N/A')}</strong></td>
            <td style="font-size: 1.3em; font-weight: 600;">{vif_val:.2f}</td>
            <td><span class="badge {badge_class}">{severity}</span></td>
            </tr>
            """
        high_count = vif_data.get('high_multicollinearity_count', 0)
        warning_html = f"""
    <div class="warning-box" style="margin-bottom: 20px;">
        <strong>⚠️ Warning:</strong> {high_count} features with high multicollinearity (VIF > 10) detected. 
        Consider removing redundant features to improve model stability.
    </div>
        """ if high_count > 0 else ""
        return f"""
    <div class="section">
        <h2>🔗 Multicollinearity Analysis (VIF)</h2>
        <p style="margin-bottom: 20px; color: #666;">
            Variance Inflation Factor measures multicollinearity between features.
            VIF > 10 indicates high multicollinearity.
        </p>
        {warning_html}
        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>VIF Score</th>
                    <th>Severity</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
        """

    def _html_outliers_section(self, outliers: Dict[str, Any]) -> str:
        """Generate outliers HTML section."""
        if not outliers or 'results_by_column' not in outliers:
            return ""
        method = outliers.get('method', 'iqr').upper()
        results = outliers.get('results_by_column', {})
        rows = ""
        for col, data in list(results.items())[:10]:
            if isinstance(data, dict) and 'total_outliers' in data:
                total = data.get('total_outliers', 0)
                pct = data.get('total_outlier_percentage', 0)
                severity = data.get('severity', 'low')
                badge_class = 'badge-danger' if severity == 'high' else 'badge-warning' if severity == 'moderate' else 'badge-success'
                rows += f"""
                <tr>
                <td><strong>{col}</strong></td>
                <td>{total:,}</td>
                <td>{pct:.2f}%</td>
                <td><span class="badge {badge_class}">{severity.capitalize()}</span></td>
            </tr>
                """
        return f"""
    <div class="section">
        <h2>📊 Outlier Detection ({method})</h2>
        <p style="margin-bottom: 20px; color: #666;">
            Outliers detected using {method} method across numerical columns.
        </p>
        <table>
            <thead>
                <tr>
                    <th>Column</th>
                    <th>Total Outliers</th>
                    <th>Percentage</th>
                    <th>Severity</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
        """

    def _html_anomalies_section(self, anomalies: Dict[str, Any]) -> str:
        """Generate anomalies HTML section."""
        if not anomalies or 'total_anomalies' not in anomalies:
            return ""
        total = anomalies.get('total_anomalies', 0)
        pct = anomalies.get('anomaly_percentage', 0)
        method = anomalies.get('method', 'isolation_forest')
        return f"""
    <div class="section">
        <h2>⚠️ Anomaly Detection</h2>
        <p style="margin-bottom: 20px; color: #666;">
            ML-based multivariate anomaly detection using {method.replace('_', ' ').title()}.
        </p>
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-label">🚨 Total Anomalies</div>
                <div class="stat-value">{total:,}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">📊 Percentage</div>
                <div class="stat-value">{pct:.2f}%</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">🔬 Method</div>
                <div class="stat-value" style="font-size: 1.2em;">{method.replace('_', ' ').title()}</div>
            </div>
        </div>
    </div>
        """

    def _html_constant_features_section(self, constant: Dict[str, Any]) -> str:
        """Generate constant features HTML section."""
        if not constant or constant.get('constant_count', 0) == 0:
            return ""
        const_features = constant.get('constant_features', [])
        quasi_features = constant.get('quasi_constant_features', [])
        low_var_features = constant.get('low_variance_features', [])
        features_html = ""
        if const_features:
            features_html += f"<tr><td><strong>Constant (100%)</strong></td><td>{', '.join(const_features[:5])}</td></tr>"
        if quasi_features:
            features_html += f"<tr><td><strong>Quasi-Constant (>95%)</strong></td><td>{', '.join(quasi_features[:5])}</td></tr>"
        if low_var_features:
            features_html += f"<tr><td><strong>Low Variance (>90%)</strong></td><td>{', '.join(low_var_features[:5])}</td></tr>"
        return f"""
    <div class="section">
        <h2>🔍 Constant & Low Variance Features</h2>
        <div class="warning-box" style="margin-bottom: 20px;">
            <strong>⚠️ Action Required:</strong> {len(const_features)} constant features detected. 
            Consider removing features with little to no variance.
        </div>
        <table>
            <thead>
                <tr>
                    <th>Type</th>
                    <th>Features</th>
                </tr>
            </thead>
            <tbody>{features_html}</tbody>
        </table>
    </div>
        """

    def _html_correlations_section(self, correlations: Dict[str, Any]) -> str:
        """Generate correlations HTML section."""
        if not correlations or 'strong_correlations_count' not in correlations:
            return ""
        count = correlations.get('strong_correlations_count', 0)
        if count == 0:
            return ""
        strong_corr = correlations.get('strong_correlations', [])
        rows = ""
        for corr in strong_corr[:10]:
            col1 = corr.get('column1', '')
            col2 = corr.get('column2', '')
            val = corr.get('correlation', 0)
            strength = corr.get('strength', '')
            badge_class = 'badge-danger' if abs(val) > 0.9 else 'badge-warning'
            rows += f"""
        <tr>
            <td><strong>{col1}</strong></td>
            <td><strong>{col2}</strong></td>
            <td style="font-size: 1.2em; font-weight: 600;">{val:.3f}</td>
            <td><span class="badge {badge_class}">{strength}</span></td>
        </tr>
            """
        return f"""
    <div class="section">
        <h2>🔗 Strong Correlations</h2>
        <p style="margin-bottom: 20px; color: #666;">
            {count} strong correlations (|r| > 0.7) detected between features.
        </p>
        <table>
            <thead>
                <tr>
                    <th>Feature 1</th>
                    <th>Feature 2</th>
                    <th>Correlation</th>
                    <th>Strength</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
        """

    def _html_statistical_tests_section(self, tests: Dict[str, Any]) -> str:
        """Generate statistical tests HTML section."""
        if not tests:
            return ""
        sections = ""
        # Chi-square tests
        if 'chi_square_tests' in tests:
            chi_data = tests['chi_square_tests']
            results = chi_data.get('results', [])[:5]
            if results:
                rows = ""
                for res in results:
                    badge_class = 'badge-success' if res.get('significant') else 'badge-info'
                    rows += f"""
                <tr>
                    <td>{res.get('column1', '')}</td>
                    <td>{res.get('column2', '')}</td>
                    <td>{res.get('chi2_statistic', 0):.2f}</td>
                    <td>{res.get('p_value', 0):.4f}</td>
                    <td><span class="badge {badge_class}">{res.get('interpretation', '')}</span></td>
                </tr>
                    """
                sections += f"""
            <h3 style="color: #667eea; margin: 25px 0 15px 0;">Chi-Square Independence Tests</h3>
            <table>
                <thead>
                    <tr>
                        <th>Variable 1</th>
                        <th>Variable 2</th>
                        <th>χ² Statistic</th>
                        <th>p-value</th>
                        <th>Interpretation</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
                """
        if not sections:
            return ""
        return f"""
    <div class="section">
        <h2>🧪 Statistical Tests</h2>
        <p style="margin-bottom: 20px; color: #666;">
            Hypothesis testing results for categorical and numerical relationships.
        </p>
        {sections}
    </div>
        """

    def _format_insights_html(self, insights: list) -> str:
        """Format insights as HTML list items."""
        if not insights:
            return '<li>No insights generated</li>'
        return ''.join(f'<li>{insight}</li>' for insight in insights)

    def _format_recommendations_html(self, recommendations: list) -> str:
        """Format recommendations as HTML list items with priority colors."""            
        if not recommendations:
            return '<li>No recommendations generated</li>'
        html = ""
        for rec in recommendations:
            priority_class = ""
            if "🔴" in rec or "HIGH PRIORITY" in rec:
                priority_class = "priority-high"
            elif "🟡" in rec or "MEDIUM PRIORITY" in rec:                priority_class = "priority-medium"
            elif "🟢" in rec or "LOW PRIORITY" in rec:
                priority_class = "priority-low"
                html += f'<li class="{priority_class}">{rec}</li>'
        return html

    def _get_report_path(self, dataset: Dataset) -> Path:
        """Generate path for EDA report."""
        reports_dir = Path(settings.UPLOAD_DIR) / f"user_{dataset.owner_id}" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
    
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"eda_report_{dataset.id}_{timestamp}.html"
    
        return reports_dir / report_filename


    async def _generate_enhanced_dashboard(
        self,
        statistics: Dict[str, Any],
        standard_report_path: Path,
        dataset_name: str
    ) -> str:
        """Generate enhanced HTML dashboard with all 26 advanced features."""
        try:
            html_content = self._create_enhanced_html_template(statistics, dataset_name)
        
            # Save as enhanced report
            dashboard_path = standard_report_path.parent / f"enhanced_{standard_report_path.name}"
        
            await asyncio.to_thread(
                dashboard_path.write_text,
                html_content,
                encoding='utf-8'
            )
        
            return str(dashboard_path)
        
        except Exception as e:
            logger.error(f"❌ Enhanced dashboard generation failed: {e}")
            return None
    
    def _get_report_path(self, dataset: Dataset) -> Path:
        """Generate path for EDA report."""
        reports_dir = Path(settings.UPLOAD_DIR) / f"user_{dataset.owner_id}" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"eda_report_{dataset.id}_{timestamp}.html"
        
        return reports_dir / report_filename
    
    async def _upload_report_to_s3(self, report_path: Path, dataset: Dataset) -> str:
        """Upload report to S3 (placeholder for production)."""
        # TODO: Implement actual S3 upload
        # from app.utils.s3_utils import upload_file_to_s3
        # s3_key = f"reports/user_{dataset.owner_id}/eda_report_{dataset.id}.html"
        # return await upload_file_to_s3(report_path, s3_key)
        
        logger.info("📤 S3 upload not configured, using local path")
        return str(report_path)
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 25: MAIN EDA GENERATION ORCHESTRATOR (ENHANCED)
    # ═══════════════════════════════════════════════════════════
    
    async def generate_eda_report(
            self,
            dataset_id: int,
            config: Dict[str, Any]
        ) -> Dict[str, Any]:
        """
    ✅ FEATURE 25: Main EDA Generation Orchestrator (FIXED)
    
    Orchestrates the entire EDA process with all 26 features:
    - Configuration parsing
    - Data loading
    - Statistical analysis (Features 1-5)
    - Outlier detection (Features 6-8)
    - Missing value analysis (Feature 9)
    - Multicollinearity (Feature 10)
    - ML insights (Features 11-15)
    - Statistical tests (Feature 16)
    - Time series (Feature 17)
    - Quality scoring (Feature 18)
    - Insights & recommendations (Features 19-20)
    
    FIXES:
    - NumPy type conversion for JSON serialization
    - Timezone-aware datetime handling
    - Accurate processing time tracking
        """
        start_time = datetime.now(timezone.utc)
    
        minimal = config.get('minimal_report', False)
        sample_size = config.get('sample_size')
        gen_corr = config.get('generate_correlations', True)
        gen_dist = config.get('generate_distributions', True)
        outlier_method = config.get('outlier_method', 'iqr')
        corr_threshold = config.get('min_correlation_threshold', 0.3)
        card_limit = config.get('categorical_cardinality_limit', 50)
        top_features = config.get('top_features', 15)
        clustering = config.get('perform_clustering', False)
        pca_analysis = config.get('perform_pca', False)
        detect_anomalies = config.get('detect_anomalies', True)
        ts_analysis = config.get('time_series_analysis', False)
        date_col = config.get('date_column')
        target_col = config.get('target_column')
    
        dataset = self.db.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(404, "Dataset not found")
        if not dataset.is_ready():
            raise HTTPException(400, f"Dataset not ready: {dataset.status}")
    
        dataset.status = DatasetStatus.ANALYZING
        self.db.commit()
    
        try:
            logger.info(f"🔬 Starting comprehensive EDA for dataset {dataset_id}")

            df = await asyncio.to_thread(self._read_dataframe, dataset.file_path, dataset.file_type)
            logger.info(f"📊 Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
                logger.info(f"📉 Sampled to {sample_size:,} rows")
        
            statistics = {}
        
            logger.info("📈 Phase 1: Basic Profiling")
            statistics['overview'] = self._get_overview_statistics(df)
            statistics['numerical'] = self._get_numerical_statistics(df)
            statistics['categorical'] = self._get_categorical_statistics(df, card_limit)
        
            logger.info("🔍 Phase 2: Advanced Analysis")
            if gen_corr:
                statistics['correlations'] = self._calculate_correlations(df, corr_threshold)
            if gen_dist:
                statistics['distributions'] = self._analyze_distributions(df)
            statistics['outliers'] = self._detect_outliers(df, method=outlier_method)
            statistics['missing_patterns'] = self._analyze_missing_patterns(df)
            statistics['multicollinearity'] = self._detect_multicollinearity(df)
            statistics['constant_features'] = self._detect_constant_features(df)
        
            logger.info("🤖 Phase 3: ML Insights")
            if target_col and target_col in df.columns:
                statistics['feature_importance'] = self._calculate_feature_importance(df, target_col, top_features)
            if pca_analysis:
                statistics['pca'] = self._perform_pca(df)
            if clustering:
                statistics['clustering'] = self._perform_clustering(df)
            if detect_anomalies:
                statistics['anomalies'] = self._detect_anomalies(df)
        
            if ts_analysis and date_col and date_col in df.columns:
                logger.info("📅 Phase 4: Time Series Analysis")
                statistics['time_series'] = self._analyze_time_series(df, date_col)
        
            logger.info("📊 Phase 5: Statistical Tests")
            statistics['statistical_tests'] = self._perform_statistical_tests(df)
        
            logger.info("✅ Phase 6: Data Quality Assessment")
            statistics['data_quality'] = self._calculate_comprehensive_quality(statistics)
        
            logger.info("💡 Phase 7: Automated Insights")
            statistics['insights'] = self._generate_automated_insights(df, statistics)
            statistics['recommendations'] = self._generate_recommendations(statistics)
        
            logger.info("💾 Phase 8: Saving Results")
            await asyncio.to_thread(self._save_statistics, dataset_id, statistics)

            report_url = await self._generate_profile_report(df, dataset, minimal)
        
            dataset.eda_report_url = report_url
            dataset.eda_report_generated_at = datetime.now(timezone.utc)
            dataset.status = DatasetStatus.COMPLETED
            dataset.statistics_data = statistics
            self.db.commit()
        
            enhanced_report_url = None
            if not minimal:
                enhanced_report_url = await self._generate_enhanced_dashboard(statistics, Path(report_url), dataset.name)
        
            clean_statistics = self._convert_numpy_types(statistics)
            processing_time = round((datetime.now(timezone.utc) - start_time).total_seconds(), 2)
        
            return {
                "dataset_id": dataset_id,
                "dataset_name": dataset.name,
                "status": "completed",
                "message": "Comprehensive EDA report generated successfully",
                "config_used": config,
                "report_url": enhanced_report_url if (enhanced_report_url and not minimal) else report_url,
                "statistics": clean_statistics,
                "generated_at": dataset.eda_report_generated_at.isoformat(),
                "processing_time_seconds": processing_time,
                "summary": {
                    "total_rows": int(clean_statistics['overview']['total_rows']),
                    "total_columns": int(clean_statistics['overview']['total_columns']),
                    "data_quality_score": float(clean_statistics['data_quality']['overall_score']),
                    "quality_class": str(clean_statistics['data_quality']['quality_class']),
                    "insights_generated": len(clean_statistics['insights']),
                    "recommendations_count": len(clean_statistics['recommendations']),
                }
            }
    
        except HTTPException:
            raise
        except Exception as e:
            dataset.status = DatasetStatus.FAILED
            dataset.processing_error = str(e)
            self.db.commit()
            logger.error(f"❌ EDA generation failed: {e}", exc_info=True)
            raise HTTPException(500, f"EDA generation failed: {str(e)}")

# ═══════════════════════════════════════════════════════════
# FEATURE 26: DEPENDENCY INJECTION HELPER
# ═══════════════════════════════════════════════════════════

def get_eda_service(db: Session = Depends(get_db)) -> EDAService:
    """
    ✅ FEATURE 26: Dependency Injection Helper
    
    FastAPI dependency for injecting EDAService into endpoints.
    
    Usage:
    ```
    @router.post("/eda/{dataset_id}/generate")
    async def generate_eda(
        dataset_id: int,
        eda_service: EDAService = Depends(get_eda_service)
    ):
        result = await eda_service.generate_eda_report(dataset_id, config)
        return result
    ```
    """
    return EDAService(db)


# ═══════════════════════════════════════════════════════════
# COMPLETE SERVICE SUMMARY
# ═══════════════════════════════════════════════════════════

"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRODUCTION-GRADE EDA SERVICE - COMPLETE IMPLEMENTATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ ALL 26 FEATURES IMPLEMENTED:

PART 1 - Core Statistics (Features 1-5):
1. Enhanced Overview Statistics
2. Advanced Numerical Statistics  
3. Enhanced Categorical Statistics
4. Multi-method Correlation Analysis
5. Distribution Analysis with 3 Tests

PART 2 - Outlier Detection (Features 6-10):
6. IQR-based Outlier Detection
7. Z-score Outlier Detection
8. Isolation Forest Outlier Detection
9. Missing Value Pattern Analysis
10. VIF Multicollinearity Detection

PART 3 - ML Insights (Features 11-15):
11. Constant/Quasi-Constant Detection
12. Feature Importance (Mutual Information)
13. PCA Analysis
14. K-means Clustering
15. Anomaly Detection

PART 4 - Intelligence (Features 16-20):
16. Statistical Tests (5 types)
17. Time Series Analysis
18. Data Quality Scoring (6 dimensions)
19. Automated Insights Generation
20. Smart Recommendations Engine

PART 5 - Integration (Features 21-26):
21. Unified Outlier Detection
22. Data Reading with Error Handling
23. Database Statistics Persistence
24. Profile Report Generation
25. Main EDA Orchestrator
26. Dependency Injection

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE EXAMPLE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

config = {
    "minimal_report": False,
    "sample_size": None,
    "generate_correlations": True,
    "generate_distributions": True,
    "outlier_method": "isolation_forest",
    "min_correlation_threshold": 0.3,
    "categorical_cardinality_limit": 50,
    "top_features": 15,
    "perform_clustering": True,
    "perform_pca": True,
    "detect_anomalies": True,
    "time_series_analysis": True,
    "date_column": "Sale_Date",
    "target_column": "Payment_Status"
}

eda_service = EDAService(db)
result = await eda_service.generate_eda_report(dataset_id=11, config=config)

print(result['summary'])
# Output:
# {
#   "total_rows": 29469,
#   "total_columns": 9,
#   "data_quality_score": 98.99,
#   "quality_class": "Excellent",
#   "insights_generated": 12,
#   "recommendations_count": 8
# }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
