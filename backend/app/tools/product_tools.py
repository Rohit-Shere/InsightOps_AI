
from langchain.tools import tool

from app.analytics.products import (
    get_product_overview,
    get_top_products,
    get_category_performance,
    get_category_contribution,
    get_product_price_distribution,
    get_product_performance_summary,
)


@tool
def get_product_overview_tool() -> dict:
    """
    Retrieve overall product and inventory performance KPIs.

    Use for questions about:
    - Total products
    - Products with sales
    - Items sold and delivered
    - Delivered GMV
    - Product selling prices
    - Product-level marketplace overview
    """
    return get_product_overview()


@tool
def get_top_products_tool(
    metric: str = "delivered_gmv",
    limit: int = 10
) -> list:
    """
    Retrieve top-performing products ranked by a selected metric.

    Supported metrics:
    - delivered_gmv
    - delivered_items
    - total_items_sold
    - total_orders
    - avg_selling_price
    - unique_customers

    Use for questions about:
    - Best-selling products
    - Highest GMV products
    - Products with the most customers
    - Product performance rankings

    Limit must be between 1 and 100.
    """
    result = get_top_products(metric=metric, limit=limit)

    return result.to_dict(orient="records")


@tool
def get_category_performance_tool() -> list:
    """
    Retrieve product category performance metrics.

    Use for questions about:
    - Category-level sales performance
    - Category GMV
    - Number of products per category
    - Delivered items by category
    - Average selling price by category
    """
    result = get_category_performance()

    return result.to_dict(orient="records")


@tool
def get_category_contribution_tool() -> list:
    """
    Retrieve category contribution to overall delivered GMV
    and delivered item volume.

    Use for questions about:
    - Which categories contribute most to GMV
    - Category revenue contribution
    - Category share comparisons
    - Category concentration
    """
    result = get_category_contribution()

    return result.to_dict(orient="records")


@tool
def get_product_price_distribution_tool() -> list:
    """
    Retrieve product distribution across price bands.

    Use for questions about:
    - Product price ranges
    - Low-price versus high-price product distribution
    - Average product prices by price band
    - GMV contribution by price band
    """
    result = get_product_price_distribution()

    return result.to_dict(orient="records")


@tool
def get_product_performance_summary_tool() -> dict:
    """
    Retrieve a concise summary of overall product performance.

    Use for questions requiring:
    - Key product performance highlights
    - Top product identification
    - Top category identification
    - Overall product GMV summary
    """
    return get_product_performance_summary()


product_tools = [
    get_product_overview_tool,
    get_top_products_tool,
    get_category_performance_tool,
    get_category_contribution_tool,
    get_product_price_distribution_tool,
    get_product_performance_summary_tool,
]
