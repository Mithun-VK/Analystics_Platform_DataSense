"""
Enhanced Data Cleaning Endpoints - Integrated with 10-Phase Service
"""

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.schemas.response import SuccessResponse
from app.core.deps import get_current_verified_user
from app.services.dataset_service import DatasetService, get_dataset_service
from app.services.cleaning_service import DataCleaningService, CleaningConfig, CleaningPreset

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/{dataset_id}/clean",
    response_model=SuccessResponse[dict],
    summary="Clean Dataset",
    description="Apply comprehensive data cleaning pipeline to a dataset."
)
async def clean_dataset(
    dataset_id: int,
    preset: Optional[str] = "standard",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> Any:
    try:
        logger.info(f"🧹 Cleaning dataset {dataset_id} for user {current_user.id} with preset '{preset}'")

        dataset = dataset_service.get_dataset(dataset_id=dataset_id, user=current_user)
        if not dataset.is_ready():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dataset not ready. Current status: {dataset.status}"
            )

        service = DataCleaningService(db)

        # Load config from presets or fallback to standard
        try:
            preset_enum = CleaningPreset(preset.lower())
            config_dict = {
                **CleaningConfig.schema()['properties'],  # default empty schema keys
                **CleaningPreset.__members__.get(preset_enum.name.upper(), {})
            }
            config = CleaningConfig(**config_dict)
        except Exception:
            config = CleaningConfig(
                remove_duplicates=True,
                missing_strategy="median",
                outlier_detection=True,
                user_id=current_user.id,
                save_audit_log=True
            )

        # Execute cleaning
        result = service.clean_dataset(dataset_id, config)
        logger.debug(f"Cleaning result: {result}")

        if not result or not isinstance(result, dict):
            logger.error(f"Invalid cleaning result: {result}")
            raise ValueError("Cleaning service returned invalid result")

        logger.info(f"✅ Dataset {dataset_id} cleaned successfully with quality score: {result.get('quality_metrics', {}).get('overall_score', 'N/A')}")

        return SuccessResponse(
            success=True,
            message="Dataset cleaned successfully",
            data=result
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Cleaning failed for dataset {dataset_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data cleaning failed: {str(e)}"
        )


@router.get(
    "/presets",
    response_model=SuccessResponse[list],
    summary="List Cleaning Presets",
    description="Get all available cleaning configuration presets."
)
async def list_cleaning_presets() -> Any:
    try:
        presets = [{
            "name": preset.value,
            "description": f"{preset.value.replace('_', ' ').title()} cleaning preset"
        } for preset in CleaningPreset if preset != CleaningPreset.CUSTOM]

        return SuccessResponse(
            success=True,
            message=f"Found {len(presets)} cleaning presets",
            data=presets
        )
    except Exception as e:
        logger.error(f"Failed to list presets: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve presets")


@router.get(
    "/{dataset_id}/suggestions",
    response_model=SuccessResponse[dict],
    summary="Get Cleaning Suggestions",
    description="Get AI-powered suggestions for data cleaning."
)
async def get_cleaning_suggestions(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> Any:
    try:
        logger.info(f"Fetching cleaning suggestions for dataset {dataset_id}")

        dataset = dataset_service.get_dataset(dataset_id=dataset_id, user=current_user)

        suggestions = {
            "dataset_id": dataset_id,
            "data_quality_score": dataset.data_quality_score,
            "issues": [],
            "recommendations": []
        }

        if dataset.duplicate_rows_count > 0:
            dup_pct = (dataset.duplicate_rows_count / dataset.row_count) * 100
            suggestions["issues"].append({
                "type": "duplicates",
                "severity": "high" if dup_pct > 5 else "medium",
                "description": f"{dataset.duplicate_rows_count} duplicate rows ({dup_pct:.1f}%)"
            })
            suggestions["recommendations"].append({
                "action": "remove_duplicates",
                "reason": "Duplicate rows affect data quality",
                "preset": "standard",
                "config": {"remove_duplicates": True}
            })

        if dataset.missing_values_count > 0:
            total_cells = dataset.row_count * dataset.column_count
            missing_pct = (dataset.missing_values_count / total_cells) * 100
            suggestions["issues"].append({
                "type": "missing_values",
                "severity": "high" if missing_pct > 10 else "medium",
                "description": f"{dataset.missing_values_count} missing values ({missing_pct:.1f}%)"
            })
            strat = "knn" if missing_pct > 15 else "median" if missing_pct > 5 else "drop"
            preset = "aggressive" if missing_pct > 15 else "standard" if missing_pct > 5 else "minimal"
            suggestions["recommendations"].append({
                "action": "handle_missing",
                "strategy": strat,
                "preset": preset,
                "reason": "Suggested missing value handling based on missing data percentage",
                "config": {"missing_strategy": strat}
            })

        if dataset.data_quality_score < 70:
            suggestions["recommendations"].append({
                "action": "comprehensive_cleaning",
                "preset": "production",
                "reason": "Low overall quality score"
            })
        else:
            suggestions["recommendations"].append({
                "action": "ml_preparation",
                "preset": "ml_ready",
                "reason": "Dataset suitable for ML pipeline"
            })

        return SuccessResponse(
            success=True,
            message="Cleaning suggestions generated",
            data=suggestions
        )
    except Exception as e:
        logger.error(f"Failed to get cleaning suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate suggestions")
