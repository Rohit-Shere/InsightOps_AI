
from langchain.tools import tool

from app.analytics.marketplace import (
    get_marketplace_kpis,
    get_monthly_marketplace_trends,
    get_cancellation_trend,
)


@tool
def get_marketplace_overview() -> dict:
    """
    Retrieve overall marketplace KPIs.

    Use for questions about total orders, delivered GMV,
    AOV, late delivery, cancellations, and overall business health.
    """
    return get_marketplace_kpis()


@tool
def get_monthly_marketplace_performance() -> list:
    """
    Retrieve monthly marketplace KPI trends.

    Use for questions about monthly GMV, orders,
    delivery performance, and marketplace growth.
    """
    result = get_monthly_marketplace_trends()
    return result.to_dict(orient="records")


@tool
def get_marketplace_cancellation_trend() -> list:
    """
    Retrieve monthly cancellation trends.

    Use for questions about cancellation patterns
    and changes in cancellation rates over time.
    """
    result = get_cancellation_trend()
    return result.to_dict(orient="records")


marketplace_tools = [
    get_marketplace_overview,
    get_monthly_marketplace_performance,
    get_marketplace_cancellation_trend,
]
