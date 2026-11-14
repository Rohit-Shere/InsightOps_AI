
from langchain.tools import tool

from app.analytics.customers import (
    get_customer_overview,
    get_customer_frequency_distribution,
    get_customer_segments,
    get_rfm_distribution,
)


@tool
def get_customer_overview_tool() -> dict:
    """
    Retrieve overall customer KPIs.

    Use when the user asks about:
    - Total unique customers
    - Customer purchasing behavior
    - Customer spending
    - Customer activity and retention overview
    """
    return get_customer_overview()


@tool
def get_customer_frequency_distribution_tool() -> list:
    """
    Retrieve the distribution of customers by purchase frequency.

    Use when the user asks about:
    - One-time versus repeat customers
    - Customer purchase frequency
    - Repeat purchase behavior
    - Customer loyalty based on order frequency
    """
    result = get_customer_frequency_distribution()

    return result.to_dict(orient="records")


@tool
def get_customer_segments_tool() -> list:
    """
    Retrieve customer segments based on RFM analysis.

    Use when the user asks about:
    - High-value customers
    - Champions and loyal customers
    - At-risk customers
    - Customer segment contribution to delivered GMV
    - Customer targeting opportunities
    """
    result = get_customer_segments()

    return result.to_dict(orient="records")


@tool
def get_rfm_distribution_tool() -> list:
    """
    Retrieve the distribution of customers across RFM segments.

    Use when the user asks about:
    - Recency, frequency, and monetary analysis
    - RFM customer segmentation
    - Customer value distribution
    - RFM segment comparisons
    """
    result = get_rfm_distribution()

    return result.to_dict(orient="records")


customer_tools = [
    get_customer_overview_tool,
    get_customer_frequency_distribution_tool,
    get_customer_segments_tool,
    get_rfm_distribution_tool,
]
