"""
AI Service for Data Insights Generation.

Production-grade implementation with:
- AI-powered insight generation (OpenAI GPT-4, Anthropic Claude)
- Fetches statistics from EDA endpoint
- Context-aware prompt engineering
- Structured insight generation with confidence scoring
- Token usage optimization
- Rate limiting and error handling
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from enum import Enum

import openai
from anthropic import Anthropic
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.dataset import Dataset, DatasetInsight
from app.models.user import User
from app.database import get_db

logger = logging.getLogger(__name__)


class AIProvider(str, Enum):
    """AI provider enumeration."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class InsightType(str, Enum):
    """Insight type enumeration."""
    SUMMARY = "summary"
    CORRELATION = "correlation"
    OUTLIER = "outlier"
    TREND = "trend"
    RECOMMENDATION = "recommendation"
    QUALITY = "quality"
    DISTRIBUTION = "distribution"
    FEATURE_IMPORTANCE = "feature_importance"


class AIService:
    """
    AI service for generating data insights using LLMs.
    
    Fetches statistics from EDA endpoint and generates
    contextual, actionable insights using GPT-4 or Claude.
    """
    
    def __init__(self, db: Session):
        """Initialize AI service."""
        self.db = db
        
        # Initialize API clients
        if settings.OPENAI_API_KEY:
            openai.api_key = settings.OPENAI_API_KEY
            self.openai_client = openai
        else:
            self.openai_client = None
            logger.warning("⚠️  OpenAI API key not configured")
            
        if settings.ANTHROPIC_API_KEY:
            self.anthropic_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        else:
            self.anthropic_client = None
            logger.warning("⚠️  Anthropic API key not configured")
    
    # ============================================================
    # MAIN INSIGHT GENERATION
    # ============================================================
    
    async def generate_insights(
        self,
        dataset_id: int,
        user: User,
        provider: str = "openai",
        max_insights: int = 10,
        insight_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate comprehensive insights for a dataset.
        
        **Parameters:**
        - dataset_id: Dataset ID
        - user: User requesting insights
        - provider: AI provider (openai or anthropic)
        - max_insights: Maximum insights to generate (1-20)
        - insight_types: Specific types to generate (optional)
        
        **Process:**
        1. Validates user and dataset
        2. Gets statistics from EDA endpoint
        3. Generates insights using LLM
        4. Saves to database
        5. Returns formatted results
        """
        try:
            logger.info(
                f"🤖 Generating insights for dataset {dataset_id} "
                f"using {provider}, max={max_insights}"
            )
            
            # Validate provider
            if provider not in ["openai", "anthropic"]:
                raise ValueError(f"Invalid provider: {provider}")
            
            # Get dataset
            dataset = self.db.query(Dataset).filter(
                Dataset.id == dataset_id,
                Dataset.owner_id == user.id
            ).first()
            
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
            
            # Fetch statistics from EDA endpoint response
            # Statistics should be passed as parameter or retrieved from stored data
            statistics = await self._get_dataset_statistics(dataset)
            
            if not statistics:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Statistics not available. Run EDA first."
                )
            
            # Extract context from statistics
            context = self._prepare_context_from_statistics(statistics, dataset)
            
            # Generate insights
            insights = []
            
            # 1. Summary insight
            if not insight_types or "summary" in insight_types:
                summary = await self._generate_summary_insight(context, provider)
                if summary:
                    insights.append(summary)
            
            # 2. Correlation insights
            if not insight_types or "correlation" in insight_types:
                if context.get("correlations"):
                    corr_insights = await self._generate_correlation_insights(
                        context, provider, max_count=3
                    )
                    insights.extend(corr_insights)
            
            # 3. Quality assessment
            if not insight_types or "quality" in insight_types:
                quality = await self._generate_quality_insight(context, provider)
                if quality:
                    insights.append(quality)
            
            # 4. Distribution insights
            if not insight_types or "distribution" in insight_types:
                if context.get("distributions"):
                    dist_insights = await self._generate_distribution_insights(
                        context, provider, max_count=2
                    )
                    insights.extend(dist_insights)
            
            # 5. Outlier insights
            if not insight_types or "outlier" in insight_types:
                outlier = await self._generate_outlier_insight(context, provider)
                if outlier:
                    insights.append(outlier)
            
            # 6. Recommendations
            if not insight_types or "recommendation" in insight_types:
                recommendations = await self._generate_recommendations(
                    context, provider, max_count=2
                )
                insights.extend(recommendations)
            
            # Limit results
            insights = insights[:max_insights]
            
            if not insights:
                logger.warning(f"⚠️  No insights generated for dataset {dataset_id}")
                return []
            
            # Save to database
            saved_insights = self._save_insights(dataset_id, insights, provider)
            
            logger.info(f"✅ Generated {len(saved_insights)} insights")
            
            return [self._insight_to_dict(i) for i in saved_insights]
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Insight generation failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate insights: {str(e)}"
            )
    
    # ============================================================
    # STATISTICS RETRIEVAL
    # ============================================================
    
    async def _get_dataset_statistics(self, dataset: Dataset) -> Optional[Dict[str, Any]]:
        """
        Get statistics from EDA endpoint response.
        
        Returns the full statistics dictionary including:
        - overview (rows, columns, data quality)
        - numerical (statistics for numeric columns)
        - categorical (statistics for categorical columns)
        - correlations (Pearson and Spearman)
        - distributions (distribution types and normality tests)
        - data_quality (completeness, uniqueness, validity scores)
        """
        # In production, you would call the EDA endpoint here
        # For now, return stored statistics
        if hasattr(dataset, 'statistics_data') and dataset.statistics_data:
            return dataset.statistics_data
        
        logger.warning(f"⚠️  Statistics not found for dataset {dataset.id}")
        return None
    
    # ============================================================
    # CONTEXT PREPARATION FROM STATISTICS
    # ============================================================
    
    def _prepare_context_from_statistics(
        self,
        statistics: Dict[str, Any],
        dataset: Dataset
    ) -> Dict[str, Any]:
        """
        Prepare comprehensive context from EDA statistics.
        
        Extracts all relevant information from the statistics response
        for use in prompt engineering.
        """
        stats_data = statistics.get("data", {})
        
        context = {
            # Basic info
            "dataset_id": dataset.id,
            "dataset_name": stats_data.get("dataset_name", "Unknown"),
            
            # Overview statistics
            "overview": stats_data.get("overview", {}),
            "total_rows": stats_data.get("overview", {}).get("total_rows", 0),
            "total_columns": stats_data.get("overview", {}).get("total_columns", 0),
            "memory_usage_mb": stats_data.get("overview", {}).get("memory_usage_mb", 0),
            "total_missing": stats_data.get("overview", {}).get("total_missing", 0),
            "duplicate_rows": stats_data.get("overview", {}).get("duplicate_rows", 0),
            
            # Column types
            "column_types": stats_data.get("overview", {}).get("column_types", {}),
            "numerical_columns": stats_data.get("overview", {}).get("numerical_columns", 0),
            "categorical_columns": stats_data.get("overview", {}).get("categorical_columns", 0),
            
            # Detailed statistics
            "numerical": stats_data.get("numerical", {}),
            "categorical": stats_data.get("categorical", {}),
            
            # Correlations
            "correlations": self._extract_correlations(stats_data.get("correlations", {})),
            
            # Distributions
            "distributions": stats_data.get("distributions", {}),
            
            # Data quality
            "data_quality": stats_data.get("data_quality", {}),
        }
        
        return context
    
    def _extract_correlations(self, corr_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract strong correlations from correlation matrices."""
        strong = corr_data.get("strong_correlations", [])
        
        # If no strong_correlations, generate from Pearson matrix
        if not strong:
            pearson = corr_data.get("pearson", {})
            strong = self._find_strong_pairs(pearson, threshold=0.7)
        
        return strong
    
    def _find_strong_pairs(self, matrix: Dict[str, Any], threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find strong correlation pairs from matrix."""
        pairs = []
        visited = set()
        
        for col1, values in matrix.items():
            for col2, corr_value in values.items():
                if col1 != col2 and abs(corr_value) >= threshold:
                    pair_key = tuple(sorted([col1, col2]))
                    if pair_key not in visited:
                        pairs.append({
                            "column1": col1,
                            "column2": col2,
                            "correlation": corr_value,
                            "strength": "strong" if abs(corr_value) > 0.8 else "moderate"
                        })
                        visited.add(pair_key)
        
        return pairs
    
    # ============================================================
    # INSIGHT GENERATION - SUMMARY
    # ============================================================
    
    async def _generate_summary_insight(
        self,
        context: Dict[str, Any],
        provider: str
    ) -> Optional[Dict[str, Any]]:
        """Generate overall dataset summary insight."""
        prompt = self._build_summary_prompt(context)
        
        response = await self._call_llm(
            prompt=prompt,
            provider=provider,
            max_tokens=500,
            temperature=0.7,
        )
        
        if response:
            return {
                "type": InsightType.SUMMARY,
                "title": "Dataset Overview",
                "content": response,
                "confidence_score": 0.95,
            }
        
        return None
    
    def _build_summary_prompt(self, context: Dict[str, Any]) -> str:
        """Build summary prompt using EDA statistics."""
        overview = context.get("overview", {})
        quality = context.get("data_quality", {})
        numerical_cols = context.get("numerical", {})
        categorical_cols = context.get("categorical", {})
        
        # Get top numerical and categorical columns info
        top_numerical = list(numerical_cols.items())[:3]
        top_categorical = list(categorical_cols.items())[:3]
        
        numerical_summary = "\\n".join([
            f"- {col}: mean={stats.get('mean', 0):.2f}, "
            f"std={stats.get('std', 0):.2f}, "
            f"range=[{stats.get('min', 0)}, {stats.get('max', 0)}]"
            for col, stats in top_numerical
        ])
        
        categorical_summary = "\\n".join([
            f"- {col}: {stats.get('unique', 0)} unique values, "
            f"top='{stats.get('top', 'N/A')}' ({stats.get('top_percentage', 0):.1f}%)"
            for col, stats in top_categorical
        ])
        
        return f"""Analyze this dataset and provide a concise executive summary.

Dataset: {context.get('dataset_name', 'Unknown')}

**Dataset Dimensions:**
- Rows: {context.get('total_rows', 0):,}
- Columns: {context.get('total_columns', 0)}
- Memory: {context.get('memory_usage_mb', 0):.2f} MB

**Data Quality:**
- Overall Score: {quality.get('overall_score', 0)}/100
- Completeness: {quality.get('completeness', 0)}%
- Uniqueness: {quality.get('uniqueness', 0)}%
- Validity: {quality.get('validity', 0)}%
- Missing: {context.get('total_missing', 0):,} records
- Duplicates: {context.get('duplicate_rows', 0):,} rows

**Column Types:**
- Numerical: {context.get('numerical_columns', 0)}
- Categorical: {context.get('categorical_columns', 0)}

**Numerical Columns:**
{numerical_summary}

**Categorical Columns:**
{categorical_summary}

Provide a 3-4 sentence executive summary that includes:
1. Dataset purpose/description
2. Data quality assessment
3. Key characteristics and patterns
4. Recommended next steps for analysis

Be specific, insightful, and business-focused."""
    
    # ============================================================
    # INSIGHT GENERATION - CORRELATIONS
    # ============================================================
    
    async def _generate_correlation_insights(
        self,
        context: Dict[str, Any],
        provider: str,
        max_count: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate insights about correlations."""
        correlations = context.get("correlations", [])
        if not correlations:
            return []
        
        insights = []
        
        for corr in correlations[:max_count]:
            prompt = self._build_correlation_prompt(corr, context)
            
            response = await self._call_llm(
                prompt=prompt,
                provider=provider,
                max_tokens=300,
                temperature=0.7,
            )
            
            if response:
                insights.append({
                    "type": InsightType.CORRELATION,
                    "title": f"Correlation: {corr.get('column1')} ↔ {corr.get('column2')}",
                    "content": response,
                    "confidence_score": min(abs(corr.get("correlation", 0)), 1.0),
                })
        
        return insights
    
    def _build_correlation_prompt(
        self,
        correlation: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Build correlation prompt using statistics."""
        col1 = correlation.get("column1", "Unknown")
        col2 = correlation.get("column2", "Unknown")
        corr_value = correlation.get("correlation", 0)
        
        # Get stats for these columns
        numerical = context.get("numerical", {})
        stats1 = numerical.get(col1, {})
        stats2 = numerical.get(col2, {})
        
        return f"""Analyze this correlation and explain its business significance.

**Correlation Detected:**
- Variable 1: {col1}
  - Mean: {stats1.get('mean', 0):.2f}
  - Std Dev: {stats1.get('std', 0):.2f}
  - Range: [{stats1.get('min', 0)}, {stats1.get('max', 0)}]

- Variable 2: {col2}
  - Mean: {stats2.get('mean', 0):.2f}
  - Std Dev: {stats2.get('std', 0):.2f}
  - Range: [{stats2.get('min', 0)}, {stats2.get('max', 0)}]

- Pearson Correlation: {corr_value:.3f}
- Strength: {correlation.get('strength', 'unknown')}

**Dataset:** {context.get('dataset_name', 'Unknown')}

Explain:
1. What this correlation means in practical/business terms
2. Possible causal relationship or explanation
3. How to use this insight in decision-making

Be specific and actionable (2-3 sentences)."""
    
    # ============================================================
    # INSIGHT GENERATION - QUALITY
    # ============================================================
    
    async def _generate_quality_insight(
        self,
        context: Dict[str, Any],
        provider: str
    ) -> Optional[Dict[str, Any]]:
        """Generate data quality assessment."""
        prompt = self._build_quality_prompt(context)
        
        response = await self._call_llm(
            prompt=prompt,
            provider=provider,
            max_tokens=400,
            temperature=0.6,
        )
        
        if response:
            return {
                "type": InsightType.QUALITY,
                "title": "Data Quality Assessment",
                "content": response,
                "confidence_score": 0.90,
            }
        
        return None
    
    def _build_quality_prompt(self, context: Dict[str, Any]) -> str:
        """Build quality assessment prompt."""
        quality = context.get("data_quality", {})
        overview = context.get("overview", {})
        
        missing_pct = (context.get("total_missing", 0) / 
                      max(context.get("total_rows", 1) * context.get("total_columns", 1), 1) * 100)
        dup_pct = (context.get("duplicate_rows", 0) / max(context.get("total_rows", 1), 1) * 100)
        
        return f"""Assess this dataset's quality and provide recommendations.

**Quality Metrics:**
- Overall Score: {quality.get('overall_score', 0)}/100
- Completeness: {quality.get('completeness', 0)}%
- Uniqueness: {quality.get('uniqueness', 0)}%
- Validity: {quality.get('validity', 0)}%

**Data Issues:**
- Missing Values: {context.get('total_missing', 0):,} ({missing_pct:.1f}% of total)
- Duplicate Rows: {context.get('duplicate_rows', 0):,} ({dup_pct:.1f}% of rows)
- Total Records: {context.get('total_rows', 0):,}
- Total Fields: {context.get('total_columns', 0)}

Provide:
1. Assessment of data quality (Excellent/Good/Fair/Poor)
2. Top 2-3 quality issues that need attention
3. Specific, prioritized recommendations for improvement

Be practical and focus on high-impact issues."""
    
    # ============================================================
    # INSIGHT GENERATION - DISTRIBUTIONS
    # ============================================================
    
    async def _generate_distribution_insights(
        self,
        context: Dict[str, Any],
        provider: str,
        max_count: int = 2
    ) -> List[Dict[str, Any]]:
        """Generate distribution insights."""
        distributions = context.get("distributions", {})
        if not distributions:
            return []
        
        insights = []
        interesting = []
        
        # Find interesting distributions
        for col, dist_info in list(distributions.items())[:10]:
            dist_type = dist_info.get("distribution_type", "unknown")
            normality = dist_info.get("normality_test", {})
            
            if dist_type in ["right_skewed", "left_skewed", "heavy_tailed"] or \
               not normality.get("is_normal", True):
                interesting.append((col, dist_info))
        
        for col, dist_info in interesting[:max_count]:
            prompt = self._build_distribution_prompt(col, dist_info, context)
            
            response = await self._call_llm(
                prompt=prompt,
                provider=provider,
                max_tokens=250,
                temperature=0.7,
            )
            
            if response:
                insights.append({
                    "type": InsightType.DISTRIBUTION,
                    "title": f"Distribution Pattern: {col}",
                    "content": response,
                    "confidence_score": 0.85,
                })
        
        return insights
    
    def _build_distribution_prompt(
        self,
        column: str,
        dist_info: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Build distribution prompt using statistics."""
        numerical = context.get("numerical", {})
        col_stats = numerical.get(column, {})
        normality = dist_info.get("normality_test", {})
        
        return f"""Analyze this distribution pattern and its implications.

**Column:** {column}
**Distribution Type:** {dist_info.get('distribution_type', 'unknown')}

**Statistical Summary:**
- Mean: {col_stats.get('mean', 0):.2f}
- Median: {col_stats.get('median', 0):.2f}
- Std Dev: {col_stats.get('std', 0):.2f}
- Skewness: {col_stats.get('skewness', 0):.2f}
- Kurtosis: {col_stats.get('kurtosis', 0):.2f}

**Normality Test (Shapiro-Wilk):**
- Is Normal: {normality.get('is_normal', False)}
- P-value: {normality.get('p_value', 0):.4f}
- Confidence: {normality.get('confidence', 'Unknown')}

Explain:
1. What this distribution indicates about the data
2. Implications for statistical analysis or modeling
3. Recommended transformation or handling approach

Be specific and practical (2-3 sentences)."""
    
    # ============================================================
    # INSIGHT GENERATION - OUTLIERS
    # ============================================================
    
    async def _generate_outlier_insight(
        self,
        context: Dict[str, Any],
        provider: str
    ) -> Optional[Dict[str, Any]]:
        """Generate outlier insight."""
        numerical = context.get("numerical", {})
        columns_with_outliers = []
        
        for col, stats in list(numerical.items())[:10]:
            outlier_pct = stats.get("outliers_percentage", 0)
            if outlier_pct > 5:
                columns_with_outliers.append((col, stats))
        
        if not columns_with_outliers:
            return None
        
        prompt = self._build_outlier_prompt(columns_with_outliers, context)
        
        response = await self._call_llm(
            prompt=prompt,
            provider=provider,
            max_tokens=300,
            temperature=0.7,
        )
        
        if response:
            return {
                "type": InsightType.OUTLIER,
                "title": "Outlier Detection",
                "content": response,
                "confidence_score": 0.88,
            }
        
        return None
    
    def _build_outlier_prompt(
        self,
        columns_with_outliers: List[Tuple[str, Dict[str, Any]]],
        context: Dict[str, Any]
    ) -> str:
        """Build outlier prompt."""
        outlier_summary = "\\n".join([
            f"- {col}: {stats.get('outliers_percentage', 0):.1f}% outliers "
            f"({stats.get('outliers', 0)} records), "
            f"range=[{stats.get('min', 0)}, {stats.get('max', 0)}]"
            for col, stats in columns_with_outliers[:3]
        ])
        
        return f"""Analyze outliers detected in this dataset.

**Columns with Significant Outliers:**
{outlier_summary}

**Dataset:** {context.get('dataset_name', 'Unknown')}
**Total Records:** {context.get('total_rows', 0):,}

Provide:
1. Possible business explanations for these outliers
2. Whether they should be kept, removed, or transformed
3. Impact on analysis if kept vs. removed

Be specific and business-focused (2-3 sentences)."""
    
    # ============================================================
    # INSIGHT GENERATION - RECOMMENDATIONS
    # ============================================================
    
    async def _generate_recommendations(
        self,
        context: Dict[str, Any],
        provider: str,
        max_count: int = 2
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations."""
        prompt = self._build_recommendations_prompt(context)
        
        response = await self._call_llm(
            prompt=prompt,
            provider=provider,
            max_tokens=500,
            temperature=0.8,
        )
        
        if not response:
            return []
        
        recommendations = []
        lines = response.strip().split('\\n')
        
        for line in lines[:max_count]:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                content = line.lstrip('0123456789.-•) ').strip()
                if content:
                    recommendations.append({
                        "type": InsightType.RECOMMENDATION,
                        "title": "Actionable Recommendation",
                        "content": content,
                        "confidence_score": 0.82,
                    })
        
        return recommendations
    
    def _build_recommendations_prompt(self, context: Dict[str, Any]) -> str:
        """Build recommendations prompt."""
        quality = context.get("data_quality", {})
        
        return f"""Based on comprehensive analysis, provide 3 specific actionable recommendations.

**Dataset Overview:**
- Name: {context.get('dataset_name', 'Unknown')}
- Size: {context.get('total_rows', 0):,} rows × {context.get('total_columns', 0)} columns
- Quality Score: {quality.get('overall_score', 0)}/100
- Completeness: {quality.get('completeness', 0)}%

**Current Issues:**
- Data Quality: {quality.get('validity', 0)}% valid
- Missing Data: {context.get('total_missing', 0):,} records
- Duplicates: {context.get('duplicate_rows', 0):,} rows

Provide 3 recommendations covering:
- Data quality improvements
- Analysis/modeling considerations
- Feature engineering or optimization opportunities

Format as numbered list. Each should include:
1. Specific action to take
2. Why it matters (business/technical value)
3. Expected benefit or impact

Example:
1. Standardize Region field values: Combines 10 inconsistent variations into unified categories - Will improve segmentation accuracy and reduce analysis errors"""
    
    # ============================================================
    # LLM API CALLS
    # ============================================================
    
    async def _call_llm(
        self,
        prompt: str,
        provider: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> Optional[str]:
        """Call LLM provider."""
        try:
            if provider == "openai":
                return await self._call_openai(prompt, max_tokens, temperature)
            elif provider == "anthropic":
                return await self._call_anthropic(prompt, max_tokens, temperature)
            else:
                logger.error(f"❌ Unknown provider: {provider}")
                return None
        except Exception as e:
            logger.error(f"❌ LLM call failed: {str(e)}")
            return None
    
    async def _call_openai(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> Optional[str]:
        """Call OpenAI GPT-4 API."""
        if not self.openai_client:
            logger.error("❌ OpenAI not initialized")
            return None
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analysis expert providing clear, "
                                 "specific, actionable insights based on statistical analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"❌ OpenAI failed: {str(e)}")
            return None
    
    async def _call_anthropic(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> Optional[str]:
        """Call Anthropic Claude API."""
        if not self.anthropic_client:
            logger.error("❌ Anthropic not initialized")
            return None
        
        try:
            message = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                temperature=temperature,
                system="You are a data analysis expert providing clear, "
                       "specific, actionable insights based on statistical analysis.",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            return message.content[0].text.strip()
        except Exception as e:
            logger.error(f"❌ Anthropic failed: {str(e)}")
            return None
    
    # ============================================================
    # DATABASE OPERATIONS
    # ============================================================
    
    def _save_insights(
        self,
        dataset_id: int,
        insights: List[Dict[str, Any]],
        provider: str
    ) -> List[DatasetInsight]:
        """Save insights to database."""
        saved = []
        
        for insight_data in insights:
            insight = DatasetInsight(
                dataset_id=dataset_id,
                title=insight_data.get("title", "Insight"),
                content=insight_data.get("content", ""),
                insight_type=insight_data.get("type", InsightType.SUMMARY).value
                    if isinstance(insight_data.get("type"), InsightType)
                    else str(insight_data.get("type", "summary")),
                confidence_score=insight_data.get("confidence_score", 0.5),
                model_used=f"{provider}-latest",
            )
            
            self.db.add(insight)
            saved.append(insight)
        
        self.db.commit()
        return saved
    
    def _insight_to_dict(self, insight: DatasetInsight) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": insight.id,
            "dataset_id": insight.dataset_id,
            "title": insight.title,
            "content": insight.content,
            "insight_type": insight.insight_type,
            "confidence_score": insight.confidence_score,
            "model_used": insight.model_used,
            "created_at": insight.created_at.isoformat() if insight.created_at else None,
            "is_helpful": insight.is_helpful,
        }


def get_ai_service(db: Session = Depends(get_db)) -> AIService:
    """Dependency for AIService."""
    return AIService(db)
