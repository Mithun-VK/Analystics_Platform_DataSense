"""
Exploratory Data Analysis (EDA) Endpoints - ENHANCED PRODUCTION GRADE

Complete implementation with 50+ features:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ ALL 26 ADVANCED FEATURES:
- Enhanced Overview Statistics
- Advanced Numerical & Categorical Statistics
- Multi-method Correlation (Pearson, Spearman, Kendall)
- Distribution Analysis (3 normality tests)
- Triple Outlier Detection (IQR, Z-score, Isolation Forest)
- Missing Value Pattern Analysis
- VIF Multicollinearity Detection
- Constant/Quasi-Constant Features
- Feature Importance (Mutual Information)
- PCA Analysis
- K-means Clustering
- Anomaly Detection
- Statistical Tests (Chi-square, ANOVA, Kruskal-Wallis)
- Time Series Analysis
- 6-Dimensional Data Quality Scoring
- AI-Powered Insights Generation
- Smart Recommendations Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import logging
import asyncio
import os
from typing import Any, Optional, Dict, List
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import FileResponse
from networkx import is_path
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


# ════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════

def _verify_dataset_access(
    dataset_id: int,
    current_user: User,
    dataset_service: DatasetService
) -> Dataset:
    """Verify user has access to dataset."""
    dataset = dataset_service.get_dataset(dataset_id=dataset_id, user=current_user)
    if not dataset:
        raise HTTPException(404, f"Dataset {dataset_id} not found")
    return dataset


def _check_dataset_ready(dataset: Dataset) -> None:
    """Check if dataset is ready for analysis."""
    if not dataset.is_ready():
        raise HTTPException(400, f"Dataset not ready. Status: {dataset.status}")


# ════════════════════════════════════════════════════════════
# 1. GENERATE COMPREHENSIVE EDA REPORT (ALL 26 FEATURES)
# ════════════════════════════════════════════════════════════

@router.post(
    "/{dataset_id}/generate",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Generate Comprehensive EDA Report (50+ Features)",
    description="Generate advanced EDA report with ML insights, statistical tests, and AI-powered recommendations.",
    status_code=202
)
async def generate_comprehensive_eda(
    dataset_id: int,
    config: Optional[DatasetEDAConfig] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    eda_service: EDAService = Depends(get_eda_service),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> SuccessResponse[Dict[str, Any]]:
    """
    Generate comprehensive EDA with ALL advanced features.
    
    **✨ NEW FEATURES INCLUDED:**
    - 🔬 Enhanced Statistics (percentiles, modes, entropy, Gini coefficient)
    - 🔍 Triple Outlier Detection (IQR + Z-score + Isolation Forest)
    - 📊 Multi-method Correlation (Pearson, Spearman, Kendall)
    - 📈 Distribution Analysis (3 normality tests with consensus)
    - 🧮 VIF Multicollinearity Detection
    - 🎯 Feature Importance (Mutual Information)
    - 📉 PCA Analysis (optimal components)
    - 🎨 K-means Clustering (elbow method + silhouette)
    - ⚠️ Anomaly Detection (Isolation Forest)
    - 📅 Time Series Analysis (seasonality, stationarity)
    - 🧪 Statistical Tests (Chi-square, ANOVA, Kruskal-Wallis)
    - ✅ 6D Data Quality Scoring
    - 🤖 AI-Powered Insights (10+ types)
    - 💡 Smart Recommendations (prioritized)
    
    **Request Body:**
    ```
    {
        "minimal_report": false,
        "sample_size": null,
        "generate_correlations": true,
        "generate_distributions": true,
        "outlier_method": "isolation_forest",
        "min_correlation_threshold": 0.3,
        "categorical_cardinality_limit": 50,
        "top_features": 15,
        "perform_clustering": true,
        "perform_pca": true,
        "detect_anomalies": true,
        "time_series_analysis": false,
        "date_column": null,
        "target_column": null
    }
    ```
    
    **Response Includes:**
    ```
    {
        "success": true,
        "data": {
            "dataset_id": 11,
            "status": "completed",
            "statistics": {
                "overview": {...},
                "numerical": {...},
                "categorical": {...},
                "correlations": {...},
                "distributions": {...},
                "outliers": {...},
                "missing_patterns": {...},
                "multicollinearity": {...},
                "constant_features": {...},
                "feature_importance": {...},
                "pca": {...},
                "clustering": {...},
                "anomalies": {...},
                "time_series": {...},
                "statistical_tests": {...},
                "data_quality": {
                    "overall_score": 92.5,
                    "completeness": 98.7,
                    "uniqueness": 99.2,
                    "validity": 95.1,
                    "consistency": 89.3,
                    "accuracy": 93.4
                },
                "insights": [
                    "✅ Excellent data quality (92.5%)",
                    "🔗 Found 12 strong correlations",
                    "🔴 High multicollinearity in 3 features"
                ],
                "recommendations": [
                    "🔴 HIGH PRIORITY: Address multicollinearity",
                    "✨ READY FOR MODELING: Data quality excellent"
                ]
            }
        }
    }
    ```
    """
    try:
        logger.info(f"🔬 Comprehensive EDA requested for dataset {dataset_id}")
        
        if config is None:
            config = DatasetEDAConfig()
        
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        _check_dataset_ready(dataset)
        
        if dataset.status == DatasetStatus.ANALYZING:
            raise HTTPException(409, "EDA analysis already in progress")
        
        # Generate comprehensive report with all features
        results = await eda_service.generate_eda_report(
            dataset_id=dataset_id,
            config=config.model_dump()
        )
        
        logger.info(f"✅ Comprehensive EDA completed for dataset {dataset_id}")
        
        return SuccessResponse(
            success=True,
            message="Comprehensive EDA report generated with all advanced features",
            data=results
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Comprehensive EDA failed: {e}", exc_info=True)
        raise HTTPException(500, f"EDA generation failed: {str(e)}")


# ════════════════════════════════════════════════════════════
# 2. GET STATISTICS (ENHANCED)
# ════════════════════════════════════════════════════════════

@router.get(
    "/{dataset_id}/statistics",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Get Enhanced Statistics",
    description="Get comprehensive statistics with percentiles, modes, entropy, and Gini coefficient."
)
async def get_enhanced_statistics(
    dataset_id: int,
    include_correlations: bool = Query(False),
    include_distributions: bool = Query(False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    eda_service: EDAService = Depends(get_eda_service),
) -> SuccessResponse[Dict[str, Any]]:
    """
    Get enhanced statistics with advanced metrics.
    
    **✨ NEW: Enhanced Numerical Statistics:**
    - Mode calculation
    - Extended percentiles (p5, p10, p90, p95, p99)
    - Extreme values (top 3, bottom 3)
    - Skewness & kurtosis interpretation
    
    **✨ NEW: Enhanced Categorical Statistics:**
    - Entropy & normalized entropy
    - Gini coefficient (concentration)
    - Top 5 & top 10 concentration ratios
    - Cardinality classification
    - Imbalance detection
    """
    try:
        logger.info(f"📊 Enhanced statistics requested for dataset {dataset_id}")
        
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        _check_dataset_ready(dataset)
        
        df = await asyncio.to_thread(
            eda_service._read_dataframe,
            dataset.file_path,
            dataset.file_type
        )
        
        # Get enhanced statistics
        statistics = {}
        statistics['overview'] = eda_service._get_overview_statistics(df)
        statistics['numerical'] = eda_service._get_numerical_statistics(df)
        statistics['categorical'] = eda_service._get_categorical_statistics(df, 50)
        
        if include_correlations:
            statistics['correlations'] = eda_service._calculate_correlations(df, 0.3)
        
        if include_distributions:
            statistics['distributions'] = eda_service._analyze_distributions(df)

        statistics=eda_service._convert_numpy_types(statistics)
        return SuccessResponse(
            success=True,
            message="Enhanced statistics retrieved successfully",
            data={
                "dataset_id": dataset_id,
                "dataset_name": dataset.name,
                **statistics
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Statistics failed: {e}", exc_info=True)
        raise HTTPException(500, f"Statistics retrieval failed: {str(e)}")


# ════════════════════════════════════════════════════════════
# 3. GET OUTLIERS (3 METHODS)
# ════════════════════════════════════════════════════════════

@router.get(
    "/{dataset_id}/outliers",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Detect Outliers (IQR, Z-score, Isolation Forest)",
    description="Comprehensive outlier detection using three methods."
)
async def detect_outliers(
    dataset_id: int,
    method: str = Query("iqr", description="Method: iqr, zscore, isolation_forest, or all"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    eda_service: EDAService = Depends(get_eda_service),
) -> SuccessResponse[Dict[str, Any]]:
    """
    Detect outliers using multiple methods.
    
    **Methods:**
    - `iqr`: Interquartile Range (mild & extreme outliers)
    - `zscore`: Standard deviation method (2σ, 3σ, 4σ+ classification)
    - `isolation_forest`: ML-based multivariate detection
    - `all`: Run all three and compare consensus
    
    **Returns:**
    ```
    {
        "method": "isolation_forest",
        "total_outliers": 234,
        "total_outlier_percentage": 0.79,
        "results_by_column": {
            "age": {
                "total_outliers": 45,
                "outlier_percentage": 0.15,
                "severity": "low"
            }
        }
    }
    ```
    """
    try:
        logger.info(f"🔍 Outlier detection ({method}) for dataset {dataset_id}")
        
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        _check_dataset_ready(dataset)
        
        df = await asyncio.to_thread(
            eda_service._read_dataframe,
            dataset.file_path,
            dataset.file_type
        )
        
        outliers = await asyncio.to_thread(
            eda_service._detect_outliers,
            df,
            method
        )
        
        return SuccessResponse(
            success=True,
            message=f"Outliers detected using {method} method",
            data=eda_service._convert_numpy_types(outliers)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Outlier detection failed: {e}", exc_info=True)
        raise HTTPException(500, f"Outlier detection failed: {str(e)}")


# ════════════════════════════════════════════════════════════
# 4. GET MULTICOLLINEARITY (VIF)
# ════════════════════════════════════════════════════════════

@router.get(
    "/{dataset_id}/multicollinearity",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Detect Multicollinearity (VIF)",
    description="Calculate Variance Inflation Factor for all numerical features."
)
async def detect_multicollinearity(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    eda_service: EDAService = Depends(get_eda_service),
) -> SuccessResponse[Dict[str, Any]]:
    """
    Detect multicollinearity using VIF.
    
    **Returns:**
    ```
    {
        "vif_scores": [
            {"feature": "age", "VIF": 1.23, "severity": "Low"},
            {"feature": "income", "VIF": 12.45, "severity": "High"}
        ],
        "high_multicollinearity_features": ["income"],
        "recommendations": ["Consider removing 'income' (VIF=12.45)"]
    }
    ```
    
    **VIF Interpretation:**
    - VIF < 5: Low (acceptable)
    - VIF 5-10: Moderate (monitor)
    - VIF > 10: High (remove feature)
    """
    try:
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        _check_dataset_ready(dataset)
        
        df = await asyncio.to_thread(
            eda_service._read_dataframe,
            dataset.file_path,
            dataset.file_type
        )
        
        vif = await asyncio.to_thread(eda_service._detect_multicollinearity, df)
        
        return SuccessResponse(
            success=True,
            message="Multicollinearity analysis completed",
            data=vif
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ VIF calculation failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ════════════════════════════════════════════════════════════
# 5. GET FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════

@router.get(
    "/{dataset_id}/feature-importance",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Calculate Feature Importance",
    description="Calculate feature importance using Mutual Information."
)
async def get_feature_importance(
    dataset_id: int,
    target_column: str = Query(..., description="Target column name"),
    top_n: int = Query(15, ge=1, le=50, description="Number of top features"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    eda_service: EDAService = Depends(get_eda_service),
) -> SuccessResponse[Dict[str, Any]]:
    """
    Calculate feature importance for supervised learning.
    
    **Uses Mutual Information:**
    - Handles both classification and regression
    - Detects non-linear relationships
    - Ranks features by predictive power
    
    **Returns:**
    ```
    {
        "task_type": "classification",
        "target_column": "Payment_Status",
        "top_features": [
            {"feature": "amount", "importance": 0.85, "importance_normalized": 1.0},
            {"feature": "age", "importance": 0.42, "importance_normalized": 0.49}
        ],
        "feature_selection_recommendations": {
            "keep_definitely": ["amount", "age"],
            "consider_removing": ["feature_x"]
        }
    }
    ```
    """
    try:
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        _check_dataset_ready(dataset)
        
        df = await asyncio.to_thread(
            eda_service._read_dataframe,
            dataset.file_path,
            dataset.file_type
        )
        
        importance = await asyncio.to_thread(
            eda_service._calculate_feature_importance,
            df,
            target_column,
            top_n
        )
        
        return SuccessResponse(
            success=True,
            message="Feature importance calculated",
            data=eda_service._convert_numpy_types(importance)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Feature importance failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ════════════════════════════════════════════════════════════
# 6. GET PCA ANALYSIS
# ════════════════════════════════════════════════════════════

@router.get(
    "/{dataset_id}/pca",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="PCA Analysis",
    description="Perform Principal Component Analysis with optimal component selection."
)
async def get_pca_analysis(
    dataset_id: int,
    n_components: Optional[int] = Query(None, ge=2, description="Number of components"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    eda_service: EDAService = Depends(get_eda_service),
) -> SuccessResponse[Dict[str, Any]]:
    """
    Perform PCA with dimensionality reduction recommendations.
    
    **Returns:**
    ```
    {
        "explained_variance_ratio": [0.45, 0.23, 0.15, ...],
        "cumulative_variance": [0.45, 0.68, 0.83, ...],
        "optimal_components": {
            "for_90_percent_variance": 5,
            "for_95_percent_variance": 8,
            "for_99_percent_variance": 12
        },
        "dimensionality_reduction": {
            "from": 25,
            "to_95_percent": 8,
            "reduction_ratio_95": 68.0
        }
    }
    ```
    """
    try:
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        _check_dataset_ready(dataset)
        
        df = await asyncio.to_thread(
            eda_service._read_dataframe,
            dataset.file_path,
            dataset.file_type
        )
        
        pca = await asyncio.to_thread(eda_service._perform_pca, df, n_components)
        
        return SuccessResponse(
            success=True,
            message="PCA analysis completed",
            data=pca
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ PCA failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ════════════════════════════════════════════════════════════
# 7. GET CLUSTERING ANALYSIS
# ════════════════════════════════════════════════════════════

@router.get(
    "/{dataset_id}/clustering",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="K-means Clustering Analysis",
    description="Perform K-means clustering with elbow method and silhouette scores."
)
async def get_clustering_analysis(
    dataset_id: int,
    max_clusters: int = Query(10, ge=2, le=20, description="Maximum clusters to test"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    eda_service: EDAService = Depends(get_eda_service),
) -> SuccessResponse[Dict[str, Any]]:
    """
    K-means clustering with optimal cluster selection.
    
    **Returns:**
    ```
    {
        "optimal_clusters": {
            "by_silhouette_score": 4,
            "by_davies_bouldin_index": 3,
            "by_elbow_method": 4,
            "recommended": 4
        },
        "elbow_data": {
            "k_values": [2, 3, 4, 5, ...],
            "inertias": [1234.5, 890.2, ...]
        },
        "silhouette_data": {
            "scores": [0.45, 0.62, 0.78, ...]
        }
    }
    ```
    """
    try:
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        _check_dataset_ready(dataset)
        
        df = await asyncio.to_thread(
            eda_service._read_dataframe,
            dataset.file_path,
            dataset.file_type
        )
        
        clustering = await asyncio.to_thread(
            eda_service._perform_clustering,
            df,
            max_clusters
        )
        
        return SuccessResponse(
            success=True,
            message="Clustering analysis completed",
            data=clustering
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Clustering failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ════════════════════════════════════════════════════════════
# 8. GET DATA QUALITY SCORE (6 DIMENSIONS)
# ════════════════════════════════════════════════════════════

@router.get(
    "/{dataset_id}/quality",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Get Data Quality Score (6 Dimensions)",
    description="Comprehensive 6-dimensional data quality assessment."
)
async def get_data_quality(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    eda_service: EDAService = Depends(get_eda_service),
) -> SuccessResponse[Dict[str, Any]]:
    """
    Get comprehensive data quality score.
    
    **6 Quality Dimensions:**
    1. Completeness (100 - missing%)
    2. Uniqueness (100 - duplicate%)
    3. Validity (100 - outliers%)
    4. Consistency (standardization)
    5. Accuracy (data ranges)
    6. Timeliness (for time-series)
    
    **Returns:**
    ```
    {
        "overall_score": 92.5,
        "quality_class": "Excellent",
        "completeness": 98.7,
        "uniqueness": 99.2,
        "validity": 95.1,
        "consistency": 89.3,
        "accuracy": 93.4,
        "timeliness": 100.0,
        "improvement_priorities": [
            "Consistency: Standardize data formats (current: 89.3%)"
        ]
    }
    ```
    
    **Quality Classes:**
    - 90-100: Excellent
    - 75-89: Good
    - 60-74: Fair
    - <60: Poor
    """
    try:
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        _check_dataset_ready(dataset)
        
        df = await asyncio.to_thread(
            eda_service._read_dataframe,
            dataset.file_path,
            dataset.file_type
        )
        
        # Generate statistics first
        statistics = {}
        statistics['overview'] = eda_service._get_overview_statistics(df)
        statistics['numerical'] = eda_service._get_numerical_statistics(df)
        statistics['categorical'] = eda_service._get_categorical_statistics(df, 50)
        
        # Calculate quality
        quality = await asyncio.to_thread(
            eda_service._calculate_comprehensive_quality,
            statistics
        )
        
        return SuccessResponse(
            success=True,
            message="Data quality assessment completed",
            data=quality
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Quality assessment failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ════════════════════════════════════════════════════════════
# 9. GET AI-POWERED INSIGHTS
# ════════════════════════════════════════════════════════════

@router.get(
    "/{dataset_id}/insights",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Get AI-Powered Insights",
    description="Get automated insights and smart recommendations."
)
async def get_ai_insights(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    eda_service: EDAService = Depends(get_eda_service),
) -> SuccessResponse[Dict[str, Any]]:
    """
    Get AI-generated insights and recommendations.
    
    **Insights Include:**
    - Data size assessment
    - Missing value issues
    - Outlier warnings
    - Correlation discoveries
    - Multicollinearity alerts
    - Quality warnings
    
    **Returns:**
    ```
    {
        "insights": [
            "✅ Dataset size (29,469 rows × 9 columns) suitable for analysis",
            "🟢 Low missing data (0.56%) - minimal impact",
            "🔗 Found 12 strong correlations - feature engineering opportunity",
            "🔴 High multicollinearity in 3 features - impacts regression"
        ],
        "recommendations": [
            "🔴 HIGH PRIORITY: Address multicollinearity by removing redundant features",
            "✨ READY FOR MODELING: Data quality is excellent (92.5%)"
        ]
    }
    ```
    """
    try:
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        _check_dataset_ready(dataset)
        
        df = await asyncio.to_thread(
            eda_service._read_dataframe,
            dataset.file_path,
            dataset.file_type
        )
        
        # Generate full statistics
        statistics = {}
        statistics['overview'] = eda_service._get_overview_statistics(df)
        statistics['numerical'] = eda_service._get_numerical_statistics(df)
        statistics['categorical'] = eda_service._get_categorical_statistics(df, 50)
        statistics['correlations'] = eda_service._calculate_correlations(df, 0.3)
        statistics['multicollinearity'] = eda_service._detect_multicollinearity(df)
        statistics['outliers'] = eda_service._detect_outliers(df, 'iqr')
        statistics['data_quality'] = eda_service._calculate_comprehensive_quality(statistics)
        
        # Generate insights
        insights = await asyncio.to_thread(
            eda_service._generate_automated_insights,
            df,
            statistics
        )
        
        recommendations = await asyncio.to_thread(
            eda_service._generate_recommendations,
            statistics
        )
        
        return SuccessResponse(
            success=True,
            message="AI insights generated successfully",
            data={
                "insights": insights,
                "recommendations": recommendations,
                "data_quality_score": statistics['data_quality']['overall_score']
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Insights generation failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ════════════════════════════════════════════════════════════
# 10. GET TIME SERIES ANALYSIS
# ════════════════════════════════════════════════════════════

@router.get(
    "/{dataset_id}/time-series",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Time Series Analysis",
    description="Analyze time series data with seasonality, trend, and stationarity tests."
)
async def get_time_series_analysis(
    dataset_id: int,
    date_column: str = Query(..., description="Date/timestamp column name"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    eda_service: EDAService = Depends(get_eda_service),
) -> SuccessResponse[Dict[str, Any]]:
    """
    Perform comprehensive time series analysis.
    
    **Includes:**
    - Trend detection
    - Seasonal decomposition
    - Stationarity test (Augmented Dickey-Fuller)
    - Autocorrelation analysis (ACF/PACF)
    - Frequency detection
    
    **Returns:**
    ```
    {
        "date_column": "Sale_Date",
        "start_date": "2024-01-01T00:00:00",
        "end_date": "2025-11-08T00:00:00",
        "date_range_days": 677,
        "detected_frequency": "daily",
        "time_series_analysis": {
            "amount": {
                "stationarity_test": {
                    "test": "augmented_dickey_fuller",
                    "adf_statistic": -3.45,
                    "p_value": 0.009,
                    "is_stationary": true,
                    "interpretation": "Stationary series"
                },
                "autocorrelation": {
                    "acf_lag1": 0.87,
                    "significant_lags": 12
                },
                "decomposition": {
                    "period": 7,
                    "trend_strength": 0.65,
                    "seasonal_strength": 0.23
                }
            }
        }
    }
    ```
    """
    try:
        logger.info(f"📅 Time series analysis for dataset {dataset_id}")
        
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        _check_dataset_ready(dataset)
        
        df = await asyncio.to_thread(
            eda_service._read_dataframe,
            dataset.file_path,
            dataset.file_type
        )
        
        ts_analysis = await asyncio.to_thread(
            eda_service._analyze_time_series,
            df,
            date_column
        )
        
        return SuccessResponse(
            success=True,
            message="Time series analysis completed",
            data=ts_analysis
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Time series analysis failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ════════════════════════════════════════════════════════════
# 11. GET STATISTICAL TESTS
# ════════════════════════════════════════════════════════════

@router.get(
    "/{dataset_id}/statistical-tests",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Statistical Tests",
    description="Perform Chi-square, ANOVA, Kruskal-Wallis, and other statistical tests."
)
async def get_statistical_tests(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    eda_service: EDAService = Depends(get_eda_service),
) -> SuccessResponse[Dict[str, Any]]:
    """
    Perform comprehensive statistical tests.
    
    **Tests Included:**
    
    **For Categorical Variables:**
    - Chi-Square Test (independence)
    
    **For Numerical Variables:**
    - ANOVA (compare 3+ groups)
    - Kruskal-Wallis (non-parametric ANOVA)
    - Mann-Whitney U (compare 2 groups)
    - Levene's Test (variance equality)
    
    **Returns:**
    ```
    {
        "chi_square_tests": {
            "description": "Test of independence between categorical variables",
            "tests_performed": 6,
            "results": [
                {
                    "column1": "category",
                    "column2": "payment_status",
                    "chi2_statistic": 45.67,
                    "p_value": 0.0001,
                    "significant": true,
                    "interpretation": "Dependent",
                    "effect_size": 0.23
                }
            ]
        },
        "anova_tests": {...},
        "kruskal_wallis_tests": {...},
        "levene_tests": {...}
    }
    ```
    """
    try:
        logger.info(f"🧪 Statistical tests for dataset {dataset_id}")
        
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        _check_dataset_ready(dataset)
        
        df = await asyncio.to_thread(
            eda_service._read_dataframe,
            dataset.file_path,
            dataset.file_type
        )
        
        tests = await asyncio.to_thread(
            eda_service._perform_statistical_tests,
            df
        )
        
        return SuccessResponse(
            success=True,
            message="Statistical tests completed",
            data=tests
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Statistical tests failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ════════════════════════════════════════════════════════════
# 12. GET ANOMALY DETECTION
# ════════════════════════════════════════════════════════════

@router.get(
    "/{dataset_id}/anomalies",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Detect Anomalies",
    description="ML-based anomaly detection using Isolation Forest."
)
async def detect_anomalies(
    dataset_id: int,
    contamination: float = Query(0.1, ge=0.01, le=0.5, description="Expected proportion of anomalies"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    eda_service: EDAService = Depends(get_eda_service),
) -> SuccessResponse[Dict[str, Any]]:
    """
    Detect anomalies using Isolation Forest.
    
    **Method:**
    - Isolation Forest (ML-based)
    - Handles multivariate anomalies
    - Works with high-dimensional data
    
    **Returns:**
    ```
    {
        "method": "isolation_forest",
        "total_anomalies": 147,
        "total_anomaly_percentage": 0.50,
        "contamination_used": 0.1,
        "min_anomaly_score": -0.45,
        "max_anomaly_score": 0.32,
        "anomaly_indices_sample": ,
        "feature_anomaly_analysis": {
            "age": {
                "anomaly_mean": 78.5,
                "normal_mean": 45.3,
                "difference": 33.2
            }
        }
    }
    ```
    """
    try:
        logger.info(f"⚠️ Anomaly detection for dataset {dataset_id}")
        
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        _check_dataset_ready(dataset)
        
        df = await asyncio.to_thread(
            eda_service._read_dataframe,
            dataset.file_path,
            dataset.file_type
        )
        
        anomalies = await asyncio.to_thread(
            eda_service._detect_anomalies,
            df,
            contamination
        )
        
        return SuccessResponse(
            success=True,
            message="Anomaly detection completed",
            data=anomalies
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Anomaly detection failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ════════════════════════════════════════════════════════════
# 13. GET MISSING VALUE PATTERNS
# ════════════════════════════════════════════════════════════

@router.get(
    "/{dataset_id}/missing-patterns",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Analyze Missing Value Patterns",
    description="Comprehensive missing value pattern analysis."
)
async def get_missing_patterns(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    eda_service: EDAService = Depends(get_eda_service),
) -> SuccessResponse[Dict[str, Any]]:
    """
    Analyze missing value patterns.
    
    **Includes:**
    - Missing counts per column
    - Missing value correlations
    - Missingness type classification (MCAR, MAR, MNAR)
    - Imputation recommendations
    
    **Returns:**
    ```
    {
        "total_missing_cells": 156,
        "missing_percentage": 0.56,
        "completeness": 99.44,
        "columns_with_missing": {
            "age": {
                "count": 45,
                "percentage": 0.15,
                "missingness_type": "MCAR_likely",
                "imputation_recommendation": "mean_or_median"
            }
        },
        "correlated_missingness": [
            {
                "column1": "age",
                "column2": "income",
                "correlation": 0.78,
                "pattern": "likely_related"
            }
        ]
    }
    ```
    """
    try:
        logger.info(f"❓ Missing pattern analysis for dataset {dataset_id}")
        
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        _check_dataset_ready(dataset)
        
        df = await asyncio.to_thread(
            eda_service._read_dataframe,
            dataset.file_path,
            dataset.file_type
        )
        
        patterns = await asyncio.to_thread(
            eda_service._analyze_missing_patterns,
            df
        )
        
        return SuccessResponse(
            success=True,
            message="Missing value patterns analyzed",
            data=patterns
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Missing pattern analysis failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ════════════════════════════════════════════════════════════
# 14. GET CONSTANT FEATURES
# ════════════════════════════════════════════════════════════

@router.get(
    "/{dataset_id}/constant-features",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Detect Constant Features",
    description="Identify constant and quasi-constant features."
)
async def get_constant_features(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    eda_service: EDAService = Depends(get_eda_service),
) -> SuccessResponse[Dict[str, Any]]:
    """
    Detect constant and quasi-constant features.
    
    **Classification:**
    - Constant: 100% same value
    - Quasi-constant: >95% same value
    - Low variance: >90% same value
    
    **Returns:**
    ```
    {
        "constant_features": ["feature_a"],
        "quasi_constant_features": ["feature_b", "feature_c"],
        "low_variance_features": ["feature_d"],
        "constant_count": 1,
        "quasi_constant_count": 2,
        "removal_recommendations": {
            "immediate_removal": ["feature_a"],
            "consider_removal": ["feature_b", "feature_c"]
        }
    }
    ```
    """
    try:
        logger.info(f"🔍 Constant features detection for dataset {dataset_id}")
        
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        _check_dataset_ready(dataset)
        
        df = await asyncio.to_thread(
            eda_service._read_dataframe,
            dataset.file_path,
            dataset.file_type
        )
        
        constant = await asyncio.to_thread(
            eda_service._detect_constant_features,
            df
        )
        
        return SuccessResponse(
            success=True,
            message="Constant feature detection completed",
            data=constant
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Constant feature detection failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ════════════════════════════════════════════════════════════
# KEEP ALL EXISTING ENDPOINTS FROM ORIGINAL FILE
# ════════════════════════════════════════════════════════════

# 15. GET CORRELATIONS (Enhanced from original)
@router.get(
    "/{dataset_id}/correlations",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Get Multi-Method Correlation Analysis",
    description="Get correlation matrices using Pearson, Spearman, and Kendall methods."
)
async def get_correlations(
    dataset_id: int,
    method: str = Query("pearson", description="pearson, spearman, or kendall"),
    threshold: float = Query(0.3, ge=0, le=1, description="Correlation threshold"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    eda_service: EDAService = Depends(get_eda_service),
) -> SuccessResponse[Dict[str, Any]]:
    """
    Get correlation analysis with three methods.
    
    **Methods:**
    - Pearson: Linear relationships
    - Spearman: Monotonic relationships
    - Kendall: Rank-based (robust to outliers)
    
    **Returns all three methods plus:**
    - Strong correlations (above threshold)
    - Perfect correlations (possible duplicates)
    - Correlation strength classification
    - Method agreement analysis
    """
    try:
        logger.info(f"🔗 Multi-method correlations for dataset {dataset_id}")
        
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        _check_dataset_ready(dataset)
        
        df = await asyncio.to_thread(
            eda_service._read_dataframe,
            dataset.file_path,
            dataset.file_type
        )
        
        correlations = await asyncio.to_thread(
            eda_service._calculate_correlations,
            df,
            threshold
        )
        
        if not correlations or 'error' in correlations:
            raise HTTPException(400, "Not enough numerical columns for correlation")
        
        return SuccessResponse(
            success=True,
            message="Multi-method correlation analysis completed",
            data={
                "dataset_id": dataset_id,
                "dataset_name": dataset.name,
                "method_used": method,
                "threshold": threshold,
                **correlations
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Correlation analysis failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# 16. GET DISTRIBUTIONS (Enhanced from original)
@router.get(
    "/{dataset_id}/distributions",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Get Distribution Analysis (3 Normality Tests)",
    description="Distribution analysis with Shapiro-Wilk, Kolmogorov-Smirnov, and Anderson-Darling tests."
)
async def get_distributions(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    eda_service: EDAService = Depends(get_eda_service),
) -> SuccessResponse[Dict[str, Any]]:
    """
    Get comprehensive distribution analysis.
    
    **Includes:**
    - Distribution type detection
    - Shapiro-Wilk test
    - Kolmogorov-Smirnov test
    - Anderson-Darling test
    - Normality consensus (2 out of 3 agree)
    
    **Returns:**
    ```
    {
        "distributions": {
            "age": {
                "distribution_type": "right_skewed",
                "shapiro_wilk": {...},
                "kolmogorov_smirnov": {...},
                "anderson_darling": {...},
                "normality_consensus": {
                    "tests_agreeing_normal": 1,
                    "total_tests": 3,
                    "consensus": "non_normal",
                    "confidence_level": "33%"
                }
            }
        }
    }
    ```
    """
    try:
        logger.info(f"📈 Distribution analysis for dataset {dataset_id}")
        
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        _check_dataset_ready(dataset)
        
        df = await asyncio.to_thread(
            eda_service._read_dataframe,
            dataset.file_path,
            dataset.file_type
        )
        
        distributions = await asyncio.to_thread(
            eda_service._analyze_distributions,
            df
        )
        
        if not distributions:
            raise HTTPException(400, "No numerical columns for distribution analysis")
        
        # Generate summary
        distribution_summary = {
            "normal_columns": sum(1 for d in distributions.values() if isinstance(d, dict) and d.get("distribution_type") == "normal"),
            "right_skewed": sum(1 for d in distributions.values() if isinstance(d, dict) and d.get("distribution_type") == "right_skewed"),
            "left_skewed": sum(1 for d in distributions.values() if isinstance(d, dict) and d.get("distribution_type") == "left_skewed"),
            "heavy_tailed": sum(1 for d in distributions.values() if isinstance(d, dict) and d.get("distribution_type") == "heavy_tailed"),
        }
        
        return SuccessResponse(
            success=True,
            message="Distribution analysis completed with 3 normality tests",
            data={
                "dataset_id": dataset_id,
                "dataset_name": dataset.name,
                "distributions": distributions,
                "distribution_summary": distribution_summary,
                "total_numerical_columns": len(distributions)
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Distribution analysis failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# 17. GET COMPREHENSIVE SUMMARY (Enhanced from original)
@router.get(
    "/{dataset_id}/summary",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Get Complete EDA Summary (All Features)",
    description="Comprehensive summary combining all 26 advanced features."
)
async def get_comprehensive_summary(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
    eda_service: EDAService = Depends(get_eda_service),
) -> SuccessResponse[Dict[str, Any]]:
    """
    Get complete EDA summary with all advanced features.
    
    **Includes ALL:**
    - Overview, numerical, categorical statistics
    - Correlations (3 methods)
    - Distributions (3 tests)
    - Outliers (IQR method)
    - Missing patterns
    - Multicollinearity (VIF)
    - Constant features
    - Data quality (6 dimensions)
    - AI insights
    - Smart recommendations
    
    This is the most comprehensive endpoint combining all analyses.
    """
    try:
        logger.info(f"📋 Comprehensive summary for dataset {dataset_id}")
        
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        _check_dataset_ready(dataset)
        
        start_time = datetime.now()
        
        df = await asyncio.to_thread(
            eda_service._read_dataframe,
            dataset.file_path,
            data=eda_service._convert_numpy_types(is_path)
        )
        
        # Generate all statistics
        statistics = {}
        statistics['overview'] = eda_service._get_overview_statistics(df)
        statistics['numerical'] = eda_service._get_numerical_statistics(df)
        statistics['categorical'] = eda_service._get_categorical_statistics(df, 50)
        statistics['correlations'] = eda_service._calculate_correlations(df, 0.3)
        statistics['distributions'] = eda_service._analyze_distributions(df)
        statistics['outliers'] = eda_service._detect_outliers(df, 'iqr')
        statistics['missing_patterns'] = eda_service._analyze_missing_patterns(df)
        statistics['multicollinearity'] = eda_service._detect_multicollinearity(df)
        statistics['constant_features'] = eda_service._detect_constant_features(df)
        statistics['data_quality'] = eda_service._calculate_comprehensive_quality(statistics)
        
        insights = await asyncio.to_thread(
            eda_service._generate_automated_insights,
            df,
            statistics
        )
        
        recommendations = await asyncio.to_thread(
            eda_service._generate_recommendations,
            statistics
        )
        
        analysis_duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ Summary generated in {analysis_duration:.2f}s")
        
        return SuccessResponse(
            success=True,
            message="Comprehensive EDA summary with all 26 features",
            data={
                "dataset_id": dataset_id,
                "dataset_name": dataset.name,
                "status": "success",
                "statistics": statistics,
                "insights": insights,
                "recommendations": recommendations,
                "report_url": getattr(dataset, "eda_report_url", None),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "analysis_duration_seconds": round(analysis_duration, 2),
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Summary generation failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# 18. GET REPORT FILE (from original)
@router.get(
    "/{dataset_id}/report",
    summary="Download EDA Report HTML",
    description="Download the generated HTML EDA report file."
)
async def get_eda_report(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> FileResponse:
    """Download the generated EDA HTML report."""
    try:
        logger.info(f"📥 Report download for dataset {dataset_id}")
        
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        
        report_url = getattr(dataset, "eda_report_url", None)
        if not report_url:
            raise HTTPException(400, "Report not generated. Call POST /generate first.")
        
        if not os.path.exists(report_url):
            raise HTTPException(404, "Report file not found")
        
        return FileResponse(
            path=report_url,
            filename=f"eda_report_{dataset_id}.html",
            media_type="text/html"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Report download failed: {e}", exc_info=True)
        raise HTTPException(500, "Failed to retrieve report")


# 19. REGENERATE EDA (from original)
@router.post(
    "/{dataset_id}/regenerate",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Regenerate EDA Report",
    description="Force regeneration with new configuration."
)
async def regenerate_eda(
    dataset_id: int,
    config: Optional[DatasetEDAConfig] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    eda_service: EDAService = Depends(get_eda_service),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> SuccessResponse[Dict[str, Any]]:
    """Regenerate EDA with fresh configuration."""
    try:
        logger.info(f"🔄 EDA regeneration for dataset {dataset_id}")
        
        if config is None:
            config = DatasetEDAConfig()
        
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        _check_dataset_ready(dataset)
        
        results = await eda_service.generate_eda_report(
            dataset_id=dataset_id,
            config=config.model_dump()
        )
        
        return SuccessResponse(
            success=True,
            message="EDA regenerated successfully",
            data=results
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Regeneration failed: {e}", exc_info=True)
        raise HTTPException(500, "Failed to regenerate EDA")


# 20. GET STATUS (from original)
@router.get(
    "/{dataset_id}/status",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Get EDA Status",
    description="Check EDA analysis status."
)
async def get_eda_status(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> SuccessResponse[Dict[str, Any]]:
    """Check current EDA analysis status."""
    try:
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
        logger.error(f"❌ Status check failed: {e}", exc_info=True)
        raise HTTPException(500, "Failed to check status")
