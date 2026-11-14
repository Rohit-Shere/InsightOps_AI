
from langchain.tools import tool

from app.analytics.sellers import (
    get_seller_overview,
    get_top_sellers,
    get_seller_delivery_performance,
    get_seller_contribution,
    get_seller_performance_summary,
)


@tool
def get_seller_overview_tool() -> list:
    """
    Get an overall summary of marketplace seller performance.

    Includes seller count, items sold, delivered items, seller-order
    associations, delivered GMV, average freight, and late deliveries.

    Use this for broad questions about the seller marketplace.
    """
    result = get_seller_overview()
    return result.to_dict(orient="records")


@tool
def get_top_sellers_tool(
    metric: str = "delivered_gmv",
    limit: int = 10
) -> list:
    """
    Retrieve the highest-performing sellers for a selected metric.

    Supported metrics include:
    delivered_gmv, delivered_items, total_items_sold,
    total_orders, avg_selling_price, unique_customers.

    Use this to identify top sellers by business performance.
    Limit must be between 1 and 100.
    """
    result = get_top_sellers(metric=metric, limit=limit)
    return result.to_dict(orient="records")


@tool
def get_seller_delivery_performance_tool(limit: int = 20) -> list:
    """
    Retrieve seller-level delivery performance, including eligible
    delivered orders, late orders, late delivery rate, and on-time rate.

    Use this to investigate seller delivery performance.

    Important: sellers with very few eligible orders can have unstable
    delivery rates. Avoid drawing strong conclusions from small samples.
    A late order associated with a seller does not prove that the seller
    caused the delay.
    """
    result = get_seller_delivery_performance(limit=limit)
    return result.to_dict(orient="records")


@tool
def get_seller_contribution_tool(limit: int = 20) -> list:
    """
    Retrieve sellers ranked by their contribution to delivered GMV.

    Includes delivered GMV, percentage contribution, and cumulative
    GMV contribution.

    Use this to understand seller concentration and contribution
    to marketplace delivered GMV.

    Sellers with NULL delivered GMV are excluded from the contribution
    calculation. Delivered GMV is not profit.
    """
    result = get_seller_contribution(limit=limit)
    return result.to_dict(orient="records")


@tool
def get_seller_performance_summary_tool() -> list:
    """
    Get a consolidated seller performance summary.

    Includes marketplace seller totals, delivered GMV, delivery
    performance, top seller, and top-10 seller contribution.

    Use this for high-level seller performance questions.
    """
    result = get_seller_performance_summary()
    return result.to_dict(orient="records")


seller_tools = [
    get_seller_overview_tool,
    get_top_sellers_tool,
    get_seller_delivery_performance_tool,
    get_seller_contribution_tool,
    get_seller_performance_summary_tool,
]
