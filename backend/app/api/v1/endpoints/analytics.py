from fastapi import APIRouter, Query, Depends, HTTPException, status
from typing import Optional, Any, AnyStr
import pandas as pd
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.core.deps import get_current_verified_user
from app.services.dataset_service import DatasetService, get_dataset_service
from app.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/api/v1/analytics", tags=["Analytics"])

# Utility to verify dataset ownership/access and readiness
def get_and_check_dataset(dataset_id: int, current_user: User, dataset_service: DatasetService):
    dataset = dataset_service.get_dataset(dataset_id=dataset_id, user=current_user)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    if not dataset.is_ready():
        raise HTTPException(status_code=400, detail=f"Dataset not ready. Current status: {dataset.status}")
    # Validate that all required analytics paths exist
    missing_paths = []
    for attr in ['orders_path', 'customers_path', 'products_path', 'marketing_path']:
        if not getattr(dataset, attr):
            missing_paths.append(attr)
    if missing_paths:
        raise HTTPException(status_code=400, detail=f"Dataset missing analytics file paths: {missing_paths}")
    return dataset


def read_csv_strict(path: AnyStr) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load file {path}: {str(e)}")


# Analytics service dependency loading all four datasets
def get_analytics_service_for_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_verified_user),
    dataset_service: DatasetService = Depends(get_dataset_service)
) -> AnalyticsService:
    dataset = get_and_check_dataset(dataset_id, current_user, dataset_service)
    
    orders_df = read_csv_strict(dataset.orders_path)
    customers_df = read_csv_strict(dataset.customers_path)
    products_df = read_csv_strict(dataset.products_path)
    marketing_df = read_csv_strict(dataset.marketing_path)
    
    return AnalyticsService(
        orders_df=orders_df,
        customers_df=customers_df,
        products_df=products_df,
        marketing_df=marketing_df
    )


# --- Performance Dashboard Endpoints ---

@router.get("/{dataset_id}/revenue")
def get_revenue(
    dataset_id: int,
    period: str = Query("monthly", enum=["daily", "weekly", "monthly"]),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    service: AnalyticsService = Depends(get_analytics_service_for_dataset)
):
    return service.revenue(period=period, start_date=start_date, end_date=end_date).to_dict(orient="records")


@router.get("/{dataset_id}/orders")
def get_orders(
    dataset_id: int,
    period: str = Query("monthly", enum=["daily", "weekly", "monthly"]),
    service: AnalyticsService = Depends(get_analytics_service_for_dataset)
):
    return service.orders_and_aov(period=period).to_dict(orient="records")


@router.get("/{dataset_id}/aov")
def get_aov(
    dataset_id: int,
    period: str = Query("monthly", enum=["daily", "weekly", "monthly"]),
    service: AnalyticsService = Depends(get_analytics_service_for_dataset)
):
    df = service.orders_and_aov(period=period)
    aov = df[["period", "aov"]]
    return aov.to_dict(orient="records")


@router.get("/{dataset_id}/customers/segmentation")
def get_customers_segmentation(
    dataset_id: int,
    service: AnalyticsService = Depends(get_analytics_service_for_dataset)
):
    return service.new_vs_returning_customers()


@router.get("/{dataset_id}/products/top-revenue")
def get_top_products_by_revenue(
    dataset_id: int,
    top_n: int = Query(10, ge=1),
    service: AnalyticsService = Depends(get_analytics_service_for_dataset)
):
    return service.top_products_by_revenue(top_n=top_n).to_dict(orient="records")

# --- Customer Acquisition Dashboard Endpoints ---

@router.get("/{dataset_id}/cac")
def get_cac_by_channel(
    dataset_id: int,
    service: AnalyticsService = Depends(get_analytics_service_for_dataset)
):
    return service.cac_by_channel().to_dict(orient="records")


@router.get("/{dataset_id}/cohorts")
def get_customer_cohorts(
    dataset_id: int,
    service: AnalyticsService = Depends(get_analytics_service_for_dataset)
):
    return service.customer_cohorts().to_dict(orient="records")


@router.get("/{dataset_id}/attribution")
def get_channel_attribution(
    dataset_id: int,
    model: str = Query("first_touch", enum=["first_touch", "last_touch", "multi_touch"]),
    service: AnalyticsService = Depends(get_analytics_service_for_dataset)
):
    # Adjust section for real touchpoints source if available
    # Here simply load a dedicated touchpoints file if it exists or fallback
    touchpoints_path = f"uploads/touchpoints_{dataset_id}.csv"
    try:
        touchpoints_df = pd.read_csv(touchpoints_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load touchpoints file: {str(e)}")
    
    return service.channel_attribution(touchpoints_df=touchpoints_df, model=model).to_dict(orient="records")


@router.get("/{dataset_id}/roas")
def get_roas_by_channel(
    dataset_id: int,
    service: AnalyticsService = Depends(get_analytics_service_for_dataset)
):
    return service.roas_by_channel().to_dict(orient="records")


# --- Customer Lifetime Value Dashboard Endpoints ---

@router.get("/{dataset_id}/ltv")
def get_ltv_by_cohort(
    dataset_id: int,
    service: AnalyticsService = Depends(get_analytics_service_for_dataset)
):
    return service.ltv_by_cohort().to_dict(orient="records")


@router.get("/{dataset_id}/repeat-purchase-rate")
def get_repeat_purchase_rate(
    dataset_id: int,
    service: AnalyticsService = Depends(get_analytics_service_for_dataset)
):
    return {"repeat_purchase_rate": service.repeat_purchase_rate()}


@router.get("/{dataset_id}/order-frequency")
def get_average_order_frequency(
    dataset_id: int,
    service: AnalyticsService = Depends(get_analytics_service_for_dataset)
):
    return {"average_order_frequency": service.average_order_frequency()}


@router.get("/{dataset_id}/churn-rate")
def get_churn_rate(
    dataset_id: int,
    period_days: int = Query(90, ge=1),
    service: AnalyticsService = Depends(get_analytics_service_for_dataset)
):
    return {"churn_rate": service.churn_rate(period_days=period_days)}


@router.get("/{dataset_id}/retention-curves")
def get_retention_curves(
    dataset_id: int,
    period_days: int = Query(30, ge=1),
    service: AnalyticsService = Depends(get_analytics_service_for_dataset)
):
    return service.retention_curves(period_days=period_days).to_dict(orient="records")


# --- Marketing Performance Dashboard Endpoints ---

@router.get("/{dataset_id}/ad-spend")
def get_ad_spend_by_channel(
    dataset_id: int,
    service: AnalyticsService = Depends(get_analytics_service_for_dataset)
):
    return service.ad_spend_by_channel().to_dict(orient="records")


@router.get("/{dataset_id}/cost-per-acquisition")
def get_cost_per_acquisition(
    dataset_id: int,
    service: AnalyticsService = Depends(get_analytics_service_for_dataset)
):
    return service.cost_per_acquisition().to_dict(orient="records")


@router.get("/{dataset_id}/conversion-rate")
def get_conversion_rate_by_channel(
    dataset_id: int,
    service: AnalyticsService = Depends(get_analytics_service_for_dataset)
):
    # If visits data is separate, load similarly as channel attribution
    visits_path = f"uploads/visits_{dataset_id}.csv"
    try:
        visits_df = pd.read_csv(visits_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load visits file: {str(e)}")
    return service.conversion_rate_by_channel(visits_df).to_dict(orient="records")


@router.get("/{dataset_id}/customer-journey")
def get_customer_journey(
    dataset_id: int,
    customer_id: Any,
    service: AnalyticsService = Depends(get_analytics_service_for_dataset)
):
    touchpoints_path = f"uploads/touchpoints_{dataset_id}.csv"
    try:
        touchpoints_df = pd.read_csv(touchpoints_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load touchpoints file: {str(e)}")
    return service.customer_journey(customer_id=customer_id, touchpoints_df=touchpoints_df)


# --- Profitability Dashboard Endpoints ---

@router.get("/{dataset_id}/gross-margin")
def get_gross_margin_by_product(
    dataset_id: int,
    service: AnalyticsService = Depends(get_analytics_service_for_dataset)
):
    return service.gross_margin_by_product().to_dict(orient="records")


@router.get("/{dataset_id}/net-margin")
def get_net_margin_after_marketing(
    dataset_id: int,
    service: AnalyticsService = Depends(get_analytics_service_for_dataset)
):
    return {"net_margin_after_marketing": service.net_margin_after_marketing()}


@router.get("/{dataset_id}/unit-economics")
def get_unit_economics(
    dataset_id: int,
    service: AnalyticsService = Depends(get_analytics_service_for_dataset)
):
    return service.unit_economics()


@router.get("/{dataset_id}/break-even")
def get_break_even_analysis(
    dataset_id: int,
    product_id: Any = Query(...),
    service: AnalyticsService = Depends(get_analytics_service_for_dataset)
):
    return service.break_even_analysis(product_id=product_id)
