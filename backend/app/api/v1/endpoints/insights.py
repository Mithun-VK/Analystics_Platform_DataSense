"""
AI Insights Endpoints.

Production-grade implementation with:
- AI-powered insight generation (OpenAI GPT-4, Anthropic Claude)
- Proper enum subscription validation
- Email verification checking
- Comprehensive error handling
- Enterprise-level logging
- All features implemented and tested
"""

import logging
from typing import Any, List, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User, UserSubscription
from app.models.dataset import DatasetInsight
from app.schemas.response import SuccessResponse, MessageResponse
from app.services.ai_service import AIService, get_ai_service
from app.services.dataset_service import DatasetService, get_dataset_service
from app.core.deps import get_current_verified_user

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _verify_premium_access(current_user: User) -> None:
    """
    Verify user has premium subscription and email verified.
    
    Checks:
    1. Subscription is PREMIUM or ENTERPRISE (enum comparison)
    2. Email is verified
    3. Account is active
    """
    logger.info(
        f"Verifying premium access for user {current_user.id}: "
        f"subscription={current_user.subscription}, verified={current_user.is_verified}"
    )
    
    # 1. Check subscription type (ENUM comparison)
    valid_subscriptions = [
        UserSubscription.PREMIUM,
        UserSubscription.ENTERPRISE,
    ]
    
    if current_user.subscription not in valid_subscriptions:
        logger.warning(
            f"❌ User {current_user.id} has invalid subscription: {current_user.subscription}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium subscription required. Upgrade to access AI insights."
        )
    
    # 2. Check email verification
    if not current_user.is_verified:
        logger.warning(f"❌ User {current_user.id} email not verified")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required. Please verify your email to access premium features."
        )
    
    # 3. Check account active
    if not current_user.is_active:
        logger.warning(f"❌ User {current_user.id} account inactive")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account inactive. Please contact support."
        )
    
    logger.info(f"✅ Premium access verified for user {current_user.id}")


def _verify_dataset_access(
    dataset_id: int,
    current_user: User,
    dataset_service: DatasetService
):
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


def _check_dataset_ready(dataset) -> None:
    """Check if dataset is ready for analysis."""
    if not dataset.is_ready():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset not ready. Status: {dataset.status}"
        )


def _check_statistics_exist(dataset) -> None:
    """Check if dataset statistics exist."""
    if not dataset.statistics:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset statistics not available. Run GET /api/v1/eda/{id}/statistics first."
        )


# ============================================================
# 1. GENERATE INSIGHTS
# ============================================================

@router.post(
    "/{dataset_id}/generate",
    response_model=SuccessResponse[List[dict]],
    summary="Generate AI Insights",
    status_code=202
)
async def generate_insights(
    dataset_id: int,
    provider: str = Query("openai", description="AI provider: openai or anthropic"),
    max_insights: int = Query(10, ge=1, le=20, description="Maximum insights (1-20)"),
    insight_types: Optional[str] = Query(None, description="Comma-separated insight types to focus on"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    ai_service: AIService = Depends(get_ai_service),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> SuccessResponse[List[dict]]:
    """
    Generate AI-powered insights from dataset analysis (Premium Feature).
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Query Parameters:**
    - provider: AI provider (openai or anthropic, default: openai)
    - max_insights: Number of insights to generate (1-20, default: 10)
    - insight_types: Comma-separated types (summary,correlation,outlier,trend,recommendation,quality,distribution)
    
    **AI Providers:**
    - openai: GPT-4o (recommended for data analysis)
    - anthropic: Claude 3.5 Sonnet (excellent reasoning)
    
    **Generated Insights Include:**
    - Dataset summary and characteristics
    - Correlation analysis and business implications
    - Data quality assessment with recommendations
    - Distribution patterns and transformation suggestions
    - Outlier analysis and handling strategies
    - Actionable recommendations for improvement
    - Feature engineering ideas
    
    **Response (202 Accepted):**
    ```json
    {
        "success": true,
        "message": "Generated 10 insights successfully",
        "data": [
            {
                "id": 1,
                "title": "Strong correlation detected",
                "content": "Age and salary show strong positive correlation...",
                "insight_type": "correlation",
                "confidence_score": 0.95,
                "model_used": "gpt-4o",
                "is_helpful": null,
                "created_at": "2025-11-05T20:45:00Z"
            }
        ]
    }
    ```
    
    **Premium Access Requirements:**
    - ✅ Premium or Enterprise subscription (enum validated)
    - ✅ Email verified (is_verified: true)
    - ✅ Account active (is_active: true)
    
    **Errors:**
    - 403: Not premium or email not verified
    - 404: Dataset not found
    - 400: Dataset not ready or statistics missing
    - 422: Invalid provider or parameters
    - 500: AI service error
    """
    try:
        logger.info(
            f"🤖 AI insights generation requested for dataset {dataset_id} "
            f"by user {current_user.id} using {provider}"
        )
        
        # Verify premium access
        _verify_premium_access(current_user)
        
        # Verify dataset access
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        _check_dataset_ready(dataset)
        _check_statistics_exist(dataset)
        
        # Validate provider
        if provider not in ["openai", "anthropic"]:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Provider must be 'openai' or 'anthropic'"
            )
        
        # Parse insight types
        requested_types = None
        if insight_types:
            requested_types = [t.strip() for t in insight_types.split(",")]
            valid_types = ["summary", "correlation", "outlier", "trend", "recommendation", "quality", "distribution"]
            for itype in requested_types:
                if itype not in valid_types:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Invalid insight type: {itype}. Valid types: {', '.join(valid_types)}"
                    )
        
        # Generate insights
        insights = await ai_service.generate_insights(
            dataset_id=dataset_id,
            user=current_user,
            provider=provider,
            max_insights=max_insights,
            statistics=dataset.statistics,
            insight_types=requested_types
        )
        
        logger.info(
            f"✅ Generated {len(insights)} insights for dataset {dataset_id} "
            f"using {provider}"
        )
        
        return SuccessResponse(
            success=True,
            message=f"Generated {len(insights)} insights successfully",
            data=insights
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"❌ Insight generation failed for dataset {dataset_id}: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Insight generation failed: {str(e)}"
        )


# ============================================================
# 2. GET INSIGHTS
# ============================================================

@router.get(
    "/{dataset_id}",
    response_model=SuccessResponse[List[dict]],
    summary="Get Dataset Insights"
)
async def get_insights(
    dataset_id: int,
    insight_type: Optional[str] = Query(None, description="Filter by type"),
    min_confidence: float = Query(0.0, ge=0, le=1, description="Minimum confidence score"),
    only_helpful: bool = Query(False, description="Only show helpful insights"),
    limit: int = Query(50, ge=1, le=100, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> SuccessResponse[List[dict]]:
    """
    Get previously generated insights for a dataset.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Query Parameters:**
    - insight_type: Filter by type (summary, correlation, outlier, etc.)
    - min_confidence: Minimum confidence score (0-1)
    - only_helpful: Only show insights marked as helpful
    - limit: Maximum results (1-100, default: 50)
    - offset: Pagination offset (default: 0)
    
    **Insight Types:**
    - summary: Dataset overview
    - correlation: Relationship analysis
    - outlier: Outlier analysis
    - trend: Trend identification
    - recommendation: Actionable suggestions
    - quality: Data quality assessment
    - distribution: Distribution patterns
    
    **Returns:**
    ```json
    {
        "success": true,
        "message": "Retrieved 5 insights",
        "data": [
            {
                "id": 1,
                "title": "Insight title",
                "content": "Detailed content...",
                "insight_type": "correlation",
                "confidence_score": 0.95,
                "model_used": "gpt-4o",
                "is_helpful": true,
                "created_at": "2025-11-05T20:45:00Z"
            }
        ]
    }
    ```
    
    **Errors:**
    - 403: Premium access required
    - 404: Dataset not found or no insights
    """
    try:
        logger.info(f"📊 Fetching insights for dataset {dataset_id}")
        
        # Verify premium access
        _verify_premium_access(current_user)
        
        # Get dataset
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        
        # Query insights
        query = db.query(DatasetInsight).filter(
            DatasetInsight.dataset_id == dataset_id
        )
        
        # Filter by type
        if insight_type:
            query = query.filter(DatasetInsight.insight_type == insight_type)
        
        # Filter by confidence
        query = query.filter(DatasetInsight.confidence_score >= min_confidence)
        
        # Filter by helpfulness
        if only_helpful:
            query = query.filter(DatasetInsight.is_helpful == True)
        
        # Get total count before pagination
        total_count = query.count()
        
        # Apply pagination and ordering
        insights = query.order_by(
            DatasetInsight.confidence_score.desc(),
            DatasetInsight.created_at.desc()
        ).limit(limit).offset(offset).all()
        
        if not insights:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No insights found. Generate insights first using POST /generate"
            )
        
        # Format response
        insight_list = [
            {
                "id": i.id,
                "title": i.title,
                "content": i.content,
                "insight_type": i.insight_type,
                "confidence_score": i.confidence_score,
                "model_used": i.model_used,
                "is_helpful": i.is_helpful,
                "created_at": i.created_at.isoformat(),
            }
            for i in insights
        ]
        
        logger.info(f"✅ Retrieved {len(insights)} insights for dataset {dataset_id}")
        
        return SuccessResponse(
            success=True,
            message=f"Retrieved {len(insights)} of {total_count} insights",
            data=insight_list
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to fetch insights: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve insights"
        )


# ============================================================
# 3. RATE INSIGHT
# ============================================================

@router.post(
    "/{insight_id}/rate",
    response_model=SuccessResponse[dict],
    summary="Rate Insight"
)
async def rate_insight(
    insight_id: int,
    is_helpful: bool = Query(..., description="true if helpful, false otherwise"),
    feedback: Optional[str] = Query(None, description="Optional feedback text", max_length=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
) -> SuccessResponse[dict]:
    """
    Rate an insight as helpful or not helpful.
    
    **Path Parameters:**
    - insight_id: Insight ID
    
    **Query Parameters:**
    - is_helpful: true or false
    - feedback: Optional feedback text (max 500 chars)
    
    **Returns:**
    ```json
    {
        "success": true,
        "message": "Thank you for your feedback!",
        "data": {"insight_id": 1, "rating": true}
    }
    ```
    
    **Note:**
    - Feedback improves AI model training
    - Used to enhance future insights
    
    **Errors:**
    - 403: Premium access required
    - 404: Insight not found
    - 403: Access denied (not your insight)
    """
    try:
        logger.info(f"⭐ Rating insight {insight_id} as helpful={is_helpful}")
        
        # Verify premium access
        _verify_premium_access(current_user)
        
        # Get insight
        insight = db.query(DatasetInsight).filter(
            DatasetInsight.id == insight_id
        ).first()
        
        if not insight:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Insight not found"
            )
        
        # Verify ownership
        if insight.dataset.owner_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Update rating
        insight.is_helpful = is_helpful
        if hasattr(insight, 'user_feedback'):
            insight.user_feedback = feedback
        db.commit()
        
        logger.info(f"✅ Insight {insight_id} rated successfully")
        
        return SuccessResponse(
            success=True,
            message="Thank you for your feedback!",
            data={
                "insight_id": insight_id,
                "rating": is_helpful,
                "feedback_recorded": feedback is not None
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to rate insight: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to rate insight"
        )


# ============================================================
# 4. DELETE INSIGHTS
# ============================================================

@router.delete(
    "/{dataset_id}",
    response_model=SuccessResponse[dict],
    summary="Delete Insights"
)
async def delete_insights(
    dataset_id: int,
    insight_type: Optional[str] = Query(None, description="Delete only specific type"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> SuccessResponse[dict]:
    """
    Delete insights for a dataset.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Query Parameters:**
    - insight_type: Delete only specific type (optional, deletes all if not specified)
    
    **Returns:**
    ```json
    {
        "success": true,
        "message": "10 insights deleted successfully",
        "data": {"dataset_id": 11, "deleted_count": 10}
    }
    ```
    
    **Note:**
    - Cannot be undone
    - Use before regenerating with different settings
    
    **Errors:**
    - 403: Premium access required
    - 404: Dataset not found
    """
    try:
        logger.info(f"🗑️ Deleting insights for dataset {dataset_id}")
        
        # Verify premium access
        _verify_premium_access(current_user)
        
        # Verify dataset access
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        
        # Build delete query
        query = db.query(DatasetInsight).filter(
            DatasetInsight.dataset_id == dataset_id
        )
        
        # Filter by type if specified
        if insight_type:
            query = query.filter(DatasetInsight.insight_type == insight_type)
        
        # Delete
        deleted_count = query.delete()
        db.commit()
        
        logger.info(f"✅ {deleted_count} insights deleted for dataset {dataset_id}")
        
        return SuccessResponse(
            success=True,
            message=f"{deleted_count} insights deleted successfully",
            data={
                "dataset_id": dataset_id,
                "deleted_count": deleted_count
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete insights: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete insights"
        )


# ============================================================
# 5. GET INSIGHTS SUMMARY
# ============================================================

@router.get(
    "/{dataset_id}/summary",
    response_model=SuccessResponse[dict],
    summary="Get Insights Summary"
)
async def get_insights_summary(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> SuccessResponse[dict]:
    """
    Get summary statistics of insights for a dataset.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Returns:**
    ```json
    {
        "success": true,
        "message": "Insights summary retrieved",
        "data": {
            "dataset_id": 11,
            "total_insights": 10,
            "insights_by_type": {
                "correlation": 3,
                "recommendation": 4,
                "quality": 3
            },
            "average_confidence": 0.923,
            "confidence_distribution": {
                "high": 8,
                "medium": 1,
                "low": 1
            },
            "ratings": {
                "helpful": 8,
                "not_helpful": 1,
                "unrated": 1
            },
            "latest_generation": "2025-11-05T20:45:00Z",
            "models_used": ["gpt-4o", "claude-3.5-sonnet"]
        }
    }
    ```
    
    **Errors:**
    - 403: Premium access required
    - 404: Dataset not found or no insights
    """
    try:
        logger.info(f"📈 Fetching insights summary for dataset {dataset_id}")
        
        # Verify premium access
        _verify_premium_access(current_user)
        
        # Get dataset
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        
        # Query insights
        insights = db.query(DatasetInsight).filter(
            DatasetInsight.dataset_id == dataset_id
        ).all()
        
        if not insights:
            return SuccessResponse(
                success=True,
                message="No insights generated yet",
                data={
                    "dataset_id": dataset_id,
                    "total_insights": 0,
                    "message": "Generate insights first using POST /generate"
                }
            )
        
        # Calculate statistics
        total = len(insights)
        
        # Count by type
        by_type = {}
        for insight in insights:
            insight_type = insight.insight_type
            by_type[insight_type] = by_type.get(insight_type, 0) + 1
        
        # Average confidence
        avg_confidence = sum(i.confidence_score for i in insights) / total
        
        # Confidence distribution
        high_conf = sum(1 for i in insights if i.confidence_score >= 0.8)
        medium_conf = sum(1 for i in insights if 0.5 <= i.confidence_score < 0.8)
        low_conf = sum(1 for i in insights if i.confidence_score < 0.5)
        
        # Count ratings
        helpful_count = sum(1 for i in insights if i.is_helpful is True)
        not_helpful_count = sum(1 for i in insights if i.is_helpful is False)
        unrated_count = sum(1 for i in insights if i.is_helpful is None)
        
        # Unique models used
        models_used = list(set(i.model_used for i in insights if i.model_used))
        
        # Latest timestamp
        latest = max(i.created_at for i in insights)
        
        logger.info(f"✅ Generated summary for {total} insights")
        
        return SuccessResponse(
            success=True,
            message="Insights summary retrieved",
            data={
                "dataset_id": dataset_id,
                "total_insights": total,
                "insights_by_type": by_type,
                "average_confidence": round(avg_confidence, 3),
                "confidence_distribution": {
                    "high": high_conf,
                    "medium": medium_conf,
                    "low": low_conf
                },
                "ratings": {
                    "helpful": helpful_count,
                    "not_helpful": not_helpful_count,
                    "unrated": unrated_count
                },
                "latest_generation": latest.isoformat(),
                "models_used": models_used
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get summary: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve summary"
        )


# ============================================================
# 6. EXPORT INSIGHTS
# ============================================================

@router.get(
    "/{dataset_id}/export",
    summary="Export Insights",
    description="Export insights as JSON or CSV"
)
async def export_insights(
    dataset_id: int,
    format: str = Query("json", description="Export format: json or csv"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
):
    """
    Export insights in JSON or CSV format.
    
    **Path Parameters:**
    - dataset_id: Dataset ID
    
    **Query Parameters:**
    - format: json or csv (default: json)
    
    **Returns:**
    - File download (JSON or CSV)
    
    **Errors:**
    - 403: Premium access required
    - 404: Dataset not found or no insights
    """
    try:
        logger.info(f"📥 Exporting insights for dataset {dataset_id} as {format}")
        
        # Verify premium access
        _verify_premium_access(current_user)
        
        # Get dataset
        dataset = _verify_dataset_access(dataset_id, current_user, dataset_service)
        
        # Get insights
        insights = db.query(DatasetInsight).filter(
            DatasetInsight.dataset_id == dataset_id
        ).order_by(DatasetInsight.created_at.desc()).all()
        
        if not insights:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No insights to export"
            )
        
        if format not in ["json", "csv"]:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Format must be 'json' or 'csv'"
            )
        
        # Format data
        insight_data = [
            {
                "id": i.id,
                "title": i.title,
                "content": i.content,
                "type": i.insight_type,
                "confidence": i.confidence_score,
                "model": i.model_used,
                "helpful": i.is_helpful,
                "created": i.created_at.isoformat(),
            }
            for i in insights
        ]
        
        if format == "json":
            import json
            from fastapi.responses import Response
            
            content = json.dumps(insight_data, indent=2)
            return Response(
                content=content,
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=insights_{dataset_id}.json"}
            )
        
        else:  # csv
            import csv
            from io import StringIO
            from fastapi.responses import Response
            
            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=[
                "id", "title", "content", "type", "confidence", "model", "helpful", "created"
            ])
            writer.writeheader()
            writer.writerows(insight_data)
            
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=insights_{dataset_id}.csv"}
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Export failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export insights"
        )
