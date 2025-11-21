# app/services/analytics_service.py

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import logging

logger = logging.getLogger("AnalyticsService")

class AnalyticsService:
    """
    Production-grade e-commerce analytics service.
    Covers all dashboard metrics with scalable, robust, and efficient methods.
    """

    def __init__(
        self,
        orders_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        products_df: pd.DataFrame,
        marketing_df: pd.DataFrame
    ):
        """
        Initialize service with all relevant datasets (as pandas dataframes).
        """
        required_orders_fields = {"order_id", "customer_id", "order_date", "total_amount"}
        required_customers_fields = {"customer_id", "acquisition_date", "acquisition_channel"}
        required_products_fields = {"product_id", "cost", "price"}
        required_marketing_fields = {"channel", "spend_amount"}

        # Validate columns for critical fields, raise error if missing
        for field_group, df, required in [
            ("orders", orders_df, required_orders_fields),
            ("customers", customers_df, required_customers_fields),
            ("products", products_df, required_products_fields),
            ("marketing", marketing_df, required_marketing_fields),
        ]:
            missing = required - set(df.columns)
            if missing:
                logger.error(f"{field_group} missing columns: {missing}")
                raise ValueError(f"Missing fields in {field_group} dataset: {missing}")

        self.orders = orders_df.copy()
        # Ensure order_date is datetime
        self.orders["order_date"] = pd.to_datetime(self.orders["order_date"])
        self.customers = customers_df.copy()
        self.customers["acquisition_date"] = pd.to_datetime(self.customers["acquisition_date"])
        self.products = products_df.copy()
        self.marketing = marketing_df.copy()

    # --- Performance Dashboard ---

    def revenue(
        self,
        period: str = "daily",
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """
        Revenue by day/week/month.
        Returns a DataFrame with columns: period, revenue
        """
        df = self.orders

        # Filter by start/end date if provided
        if start_date:
            df = df[df["order_date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["order_date"] <= pd.to_datetime(end_date)]

        # Group by requested period
        if period == "daily":
            group_col = df["order_date"].dt.date
        elif period == "weekly":
            group_col = df["order_date"].dt.to_period("W")
        elif period == "monthly":
            group_col = df["order_date"].dt.to_period("M")
        else:
            logger.error(f"Invalid grouping period argument: {period}")
            raise ValueError("Period must be daily, weekly, or monthly.")

        grouped = df.groupby(group_col)["total_amount"].sum().reset_index()
        grouped.columns = ["period", "revenue"]
        logger.debug(f"Revenue grouped by {period}: {grouped.shape[0]} rows")
        return grouped

    def orders_and_aov(self, period: str = "monthly") -> pd.DataFrame:
        """
        Orders count and Average Order Value by chosen period.
        Returns a DataFrame: period, orders, aov
        """
        df = self.orders
        if period == "daily":
            group_col = df["order_date"].dt.date
        elif period == "weekly":
            group_col = df["order_date"].dt.to_period("W")
        elif period == "monthly":
            group_col = df["order_date"].dt.to_period("M")
        else:
            logger.error(f"Invalid period for orders/AOV: {period}")
            raise ValueError("Period must be daily, weekly, or monthly.")

        grouped = df.groupby(group_col).agg(
            orders=("order_id", "nunique"),
            aov=("total_amount", "mean"),
        ).reset_index()
        grouped.columns = ["period", "orders", "aov"]
        return grouped

    def new_vs_returning_customers(self) -> Dict[str, int]:
        """
        Number of new vs. returning customers.
        - New: first purchase in orders table
        - Returning: repeat transactions
        Returns dict.
        """
        # Find each customer's first order date
        first_orders = self.orders.sort_values("order_date").drop_duplicates("customer_id", keep="first")
        new_customer_count = first_orders["customer_id"].nunique()
        returning_customer_count = self.orders["customer_id"].nunique() - new_customer_count
        logger.info(f"New customers: {new_customer_count}, Returning customers: {returning_customer_count}")
        return {
            "new_customers": int(new_customer_count),
            "returning_customers": int(returning_customer_count)
        }

    def top_products_by_revenue(self, top_n: int = 10) -> pd.DataFrame:
        """
        Top N products by total revenue.
        Assumes orders['products'] is a list-of-dicts (product_id, quantity).
        Returns DataFrame: product_id, total_revenue, category
        """
        df = self.orders.copy()
        if "products" not in df.columns:
            logger.error("orders dataset missing column: products")
            raise ValueError("orders dataset must contain 'products' column")

        # Flatten products
        df = df.explode("products")
        products_expanded = pd.DataFrame(df["products"].tolist())
        df = pd.concat([df.reset_index(drop=True), products_expanded], axis=1)
        df = df.merge(self.products, on="product_id")
        df["revenue"] = df["quantity"].astype(float) * df["price"].astype(float)
        grouped = df.groupby("product_id")["revenue"].sum().nlargest(top_n).reset_index()
        grouped = grouped.merge(self.products[["product_id", "category"]], on="product_id", how="left")
        grouped = grouped.rename(columns={"revenue": "total_revenue"})
        logger.info(f"Top {top_n} products by revenue calculated")
        return grouped

    # --- Customer Acquisition Dashboard ---

    def cac_by_channel(self) -> pd.DataFrame:
        """
        Customer Acquisition Cost by channel.
        Returns DataFrame: channel, cac (may include NaNs if no acquisitions)
        """
        acq = self.customers.groupby("acquisition_channel")["customer_id"].count().reset_index()
        marketing = self.marketing.groupby("channel")["spend_amount"].sum().reset_index()
        merged = marketing.merge(acq, left_on="channel", right_on="acquisition_channel", how="left")
        merged["cac"] = np.where(merged["customer_id"] > 0, merged["spend_amount"] / merged["customer_id"], np.nan)
        merged = merged.fillna(0)
        logger.debug(f"CAC by channel: {merged.shape[0]} channels calculated")
        return merged[["channel", "cac"]]

    def customer_cohorts(self) -> pd.DataFrame:
        """
        Group customers by acquisition month.
        Returns DataFrame: cohort, num_customers
        """
        self.customers["acquisition_month"] = self.customers["acquisition_date"].dt.to_period("M")
        cohort_counts = self.customers.groupby("acquisition_month")["customer_id"].nunique().reset_index()
        cohort_counts.columns = ["cohort", "num_customers"]
        return cohort_counts

    def channel_attribution(self, touchpoints_df: pd.DataFrame, model: str = "first_touch") -> pd.DataFrame:
        """
        Attribution: first-touch, last-touch, or multi-touch based on touchpoints.
        Returns DataFrame: customer_id, channel (with attribution logic).
        """
        # Example logic for first-touch attribution
        touchpoints_df["date"] = pd.to_datetime(touchpoints_df["date"])
        if model == "first_touch":
            first_touch = touchpoints_df.sort_values("date").drop_duplicates("customer_id", keep="first")
            out = first_touch[["customer_id", "channel"]]
        elif model == "last_touch":
            last_touch = touchpoints_df.sort_values("date").drop_duplicates("customer_id", keep="last")
            out = last_touch[["customer_id", "channel"]]
        else:
            logger.info("Multi-touch attribution requires fraction or weighted logic (extend as needed)")
            out = touchpoints_df[["customer_id", "channel"]].drop_duplicates()
        return out

    def roas_by_channel(self) -> pd.DataFrame:
        """
        Return on Ad Spend by channel.
        Returns DataFrame: channel, roas
        """
        orders_grouped = self.orders.groupby("channel")["total_amount"].sum().reset_index()
        marketing_grouped = self.marketing.groupby("channel")["spend_amount"].sum().reset_index()
        merged = orders_grouped.merge(marketing_grouped, on="channel", how="left")
        merged["roas"] = np.where(merged["spend_amount"] > 0, merged["total_amount"] / merged["spend_amount"], np.nan)
        logger.info("ROAS by channel calculation done")
        return merged[["channel", "roas"]]

    # --- Customer Lifetime Value Dashboard ---

    def ltv_by_cohort(self) -> pd.DataFrame:
        """
        Lifetime Value by cohort (mean revenue per cohort).
        Returns DataFrame: cohort, ltv
        """
        orders = self.orders.copy()
        customers = self.customers.copy()
        customers["cohort"] = customers["acquisition_date"].dt.to_period("M")
        merged = orders.merge(customers[["customer_id", "cohort"]], on="customer_id", how="left")
        grouped = merged.groupby("cohort").agg(
            revenue=("total_amount", "sum"),
            customers=("customer_id", "nunique")
        ).reset_index()
        grouped["ltv"] = np.where(grouped["customers"] > 0, grouped["revenue"] / grouped["customers"], 0)
        logger.info("LTV by cohort calculated")
        return grouped[["cohort", "ltv"]]

    def repeat_purchase_rate(self) -> float:
        """
        Proportion of customers with >1 order.
        """
        freq = self.orders.groupby("customer_id")["order_id"].count()
        rate = (freq > 1).mean()
        logger.info(f"Repeat purchase rate: {rate:.2%}")
        return rate

    def average_order_frequency(self) -> float:
        """
        Average orders per customer.
        """
        avg = self.orders.groupby("customer_id")["order_id"].count().mean()
        return avg

    def churn_rate(self, period_days: int = 90, reference_date: Optional[datetime] = None) -> float:
        """
        Churn rate: Fraction of customers with last order before cutoff.
        """
        if reference_date is None:
            reference_date = datetime.now()
        cutoff = reference_date - pd.Timedelta(days=period_days)
        last_order = self.orders.groupby("customer_id")["order_date"].max()
        churned = (last_order < cutoff).sum()
        total = last_order.shape[0]
        rate = churned / total if total > 0 else 0
        logger.info(f"Churned: {churned}, Total: {total}, Rate: {rate:.2%}")
        return rate

    def retention_curves(self, period_days: int = 30) -> pd.DataFrame:
        """
        Retention curves: customers active at each period since acquisition.
        Returns DataFrame: cohort, period_num, retained_customers
        """
        customers = self.customers.copy()
        orders = self.orders.copy()
        orders = orders.merge(customers, on="customer_id")
        orders["days_since_acq"] = (orders["order_date"] - orders["acquisition_date"]).dt.days
        orders["period_num"] = (orders["days_since_acq"] // period_days).astype(int)
        orders["cohort"] = orders["acquisition_date"].dt.to_period("M")
        retained = orders.groupby(["cohort", "period_num"])["customer_id"].nunique().reset_index()
        retained = retained.rename(columns={"customer_id": "retained_customers"})
        return retained

    # --- Marketing Performance Dashboard ---

    def ad_spend_by_channel(self) -> pd.DataFrame:
        """
        Ad spend aggregated by channel.
        Returns DataFrame: channel, spend_amount
        """
        return self.marketing.groupby("channel")["spend_amount"].sum().reset_index()

    def cost_per_acquisition(self) -> pd.DataFrame:
        """
        Cost per acquisition per channel (may include zeros/NaNs where data missing).
        """
        acq = self.customers.groupby("acquisition_channel")["customer_id"].count().reset_index()
        marketing = self.marketing.groupby("channel")["spend_amount"].sum().reset_index()
        merged = marketing.merge(acq, left_on="channel", right_on="acquisition_channel", how="left")
        merged["cpa"] = np.where(merged["customer_id"] > 0, merged["spend_amount"] / merged["customer_id"], np.nan)
        merged = merged.fillna(0)
        return merged[["channel", "cpa"]]

    def conversion_rate_by_channel(self, visits_df: pd.DataFrame) -> pd.DataFrame:
        """
        Conversion rate per channel, requires visits dataset.
        visits_df should have: channel, visit_id, customer_id, session_id
        Returns DataFrame: channel, conversion_rate
        """
        orders = self.orders.groupby("channel")["order_id"].nunique().reset_index()
        visits = visits_df.groupby("channel")["visit_id"].nunique().reset_index()
        merged = visits.merge(orders, on="channel", how="left")
        merged["conversion_rate"] = np.where(merged["visit_id"] > 0, merged["order_id"] / merged["visit_id"], 0)
        return merged[["channel", "conversion_rate"]]

    def customer_journey(self, customer_id: Any, touchpoints_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Returns ordered list of customer touchpoints for visualization.
        touchpoints_df should have: customer_id, date, channel, interaction_type, etc.
        """
        journey = touchpoints_df[touchpoints_df["customer_id"] == customer_id]
        journey = journey.sort_values("date")
        return journey.to_dict("records")

    # --- Profitability Dashboard ---

    def gross_margin_by_product(self) -> pd.DataFrame:
        """
        Gross margin per product (revenue - COGS).
        Requires product-level sales and cost.
        Returns DataFrame: product_id, gross_margin
        """
        df = self.orders.copy()
        if "products" not in df.columns:
            logger.error("orders dataset missing column: products")
            raise ValueError("orders dataset must contain 'products' column")
        df = df.explode("products")
        products_expanded = pd.DataFrame(df["products"].tolist())
        df = pd.concat([df.reset_index(drop=True), products_expanded], axis=1)
        df = df.merge(self.products, on="product_id")
        df["revenue"] = df["quantity"] * df["price"]
        df["cogs"] = df["quantity"] * df["cost"]
        margin = df.groupby("product_id").agg(
            gross_margin=("revenue", "sum"),
            total_cogs=("cogs", "sum")
        ).reset_index()
        margin["gross_margin"] = margin["gross_margin"] - margin["total_cogs"]
        return margin[["product_id", "gross_margin"]]

    def net_margin_after_marketing(self) -> float:
        """
        Net margin after marketing spend.
        Returns float.
        """
        gross_margin = self.orders["total_amount"].sum() - self.products["cost"].sum()
        net_margin = gross_margin - self.marketing["spend_amount"].sum()
        logger.info(f"Net margin calculation: {net_margin:.2f}")
        return net_margin

    def unit_economics(self) -> Dict[str, float]:
        """
        Returns avg_cac, avg_ltv, cac_ltv_ratio
        """
        cac = self.cac_by_channel()["cac"].replace(np.nan, 0).mean()
        ltv = self.ltv_by_cohort()["ltv"].replace(np.nan, 0).mean()
        ratio = ltv / cac if cac > 0 else None
        return {"avg_cac": float(cac), "avg_ltv": float(ltv), "cac_ltv_ratio": float(ratio) if ratio else None}

    def break_even_analysis(self, product_id: Any) -> Dict[str, Any]:
        """
        Returns break-even units for a product.
        Assumes products_df includes "fixed_costs" per product, otherwise uses zero.
        """
        product = self.products[self.products["product_id"] == product_id]
        if product.empty:
            logger.warning(f"Product {product_id} not found for break-even analysis")
            return {"product_id": product_id, "break_even_units": None}

        fixed_costs = float(product.get("fixed_costs", 0))
        price = float(product["price"].iloc[0])
        cost = float(product["cost"].iloc[0])
        if price - cost == 0:
            logger.warning(f"Product {product_id} has zero margin, infinite or undefined break-even")
            return {"product_id": product_id, "break_even_units": None}
        break_even_units = fixed_costs / (price - cost)
        return {"product_id": product_id, "break_even_units": break_even_units}
