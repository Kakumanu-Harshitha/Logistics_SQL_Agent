"""
backend/analytics/dashboard.py
Calculates global dashboard metrics from the database.
"""
from sqlalchemy import text
from backend.database.db_connection import get_engine
import logging

logger = logging.getLogger(__name__)

def get_dashboard_metrics():
    """
    Fetch global metrics for the dashboard.
    Returns: Dict with count, avg_delivery, late_rate, and revenue.
    """
    engine = get_engine()
    metrics = {
        "total_orders": 0,
        "avg_delivery_days": 0.0,
        "late_delivery_rate": 0.0,
        "total_revenue": 0.0,
        "revenue_str": "$0",
        "trends": {
            "orders": "+0%",
            "delivery": "0 days",
            "late_rate": "+0%",
            "revenue": "+0%"
        }
    }

    try:
        with engine.connect() as conn:
            # 1. Total Orders
            res_orders = conn.execute(text("SELECT COUNT(*) FROM orders")).scalar()
            metrics["total_orders"] = int(res_orders or 0)

            # 2. Avg Delivery Days
            res_delivery = conn.execute(text("SELECT AVG(days_for_shipping_real) FROM deliveries")).scalar()
            metrics["avg_delivery_days"] = round(float(res_delivery or 0), 1)

            # 3. Late Delivery Rate
            res_late = conn.execute(text("SELECT (COUNT(*) FILTER (WHERE late_delivery_risk = 1) * 100.0 / NULLIF(COUNT(*), 0)) FROM deliveries")).scalar()
            metrics["late_delivery_rate"] = round(float(res_late or 0), 1)

            # 4. Total Revenue
            res_rev = conn.execute(text("SELECT SUM(order_value) FROM orders")).scalar()
            rev = float(res_rev or 0)
            metrics["total_revenue"] = rev
            
            # Formatted revenue
            if rev >= 1_000_000:
                metrics["revenue_str"] = f"${rev/1_000_000:.1f}M"
            elif rev >= 1_000:
                metrics["revenue_str"] = f"${rev/1_000:.1f}K"
            else:
                metrics["revenue_str"] = f"${rev:.0f}"

            # Mock trends for now (could be calculated based on date ranges)
            metrics["trends"] = {
                "orders": "+12.3%", # Mocked trend based on reference UI
                "delivery": "-0.3 days",
                "late_rate": "+2.1%",
                "revenue": "+8.7%"
            }

        return metrics
    except Exception as e:
        logger.error(f"Error fetching dashboard metrics: {e}")
        return metrics
