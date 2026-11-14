
"""
Central Tool Registry for InsightOps AI.

This module combines business analytics and statistical
tools into one registry for future LangGraph agents.
"""

from app.tools.marketplace_tools import marketplace_tools
from app.tools.customer_tools import customer_tools
from app.tools.product_tools import product_tools
from app.tools.seller_tools import seller_tools
from app.tools.payment_tools import payment_tools
from app.tools.review_tools import review_tools
from app.tools.statistical_tools import statistical_tools


# --------------------------------------------------
# 1. CATEGORY-WISE TOOL REGISTRY
# --------------------------------------------------

tool_categories = {
    "marketplace": marketplace_tools,
    "customers": customer_tools,
    "products": product_tools,
    "sellers": seller_tools,
    "payments": payment_tools,
    "reviews": review_tools,
    "statistics": statistical_tools,
}


# --------------------------------------------------
# 2. COMBINED TOOL REGISTRY
# --------------------------------------------------

all_tools = [
    tool
    for category_tools in tool_categories.values()
    for tool in category_tools
]


# --------------------------------------------------
# 3. TOOL LOOKUP
# --------------------------------------------------

def get_tool_by_name(tool_name: str):
    """
    Retrieve a registered tool using its exact name.

    Returns:
        Tool object if found, otherwise None.
    """
    for tool in all_tools:
        if tool.name == tool_name:
            return tool

    return None


# --------------------------------------------------
# 4. CATEGORY LOOKUP
# --------------------------------------------------

def get_tools_by_category(category: str):
    """
    Retrieve all tools belonging to a category.
    """
    category = category.strip().lower()

    if category not in tool_categories:
        raise ValueError(
            f"Unknown category: {category}. "
            f"Available categories: {list(tool_categories.keys())}"
        )

    return tool_categories[category]


# --------------------------------------------------
# 5. TOOL METADATA
# --------------------------------------------------

def get_tool_catalog():
    """
    Return a lightweight catalog of registered tools.

    Useful for debugging and future agent configuration.
    """
    catalog = {}

    for category, tools in tool_categories.items():
        catalog[category] = [
            {
                "name": tool.name,
                "description": tool.description,
            }
            for tool in tools
        ]

    return catalog


# --------------------------------------------------
# 6. REGISTRY VALIDATION
# --------------------------------------------------

def validate_tool_registry():
    """
    Validate that:
    - The registry is not empty.
    - Tool names are unique.
    - Every tool has a name and description.
    """

    if not all_tools:
        raise ValueError("Tool registry is empty.")

    tool_names = [tool.name for tool in all_tools]

    duplicate_names = sorted({
        name for name in tool_names
        if tool_names.count(name) > 1
    })

    if duplicate_names:
        raise ValueError(
            f"Duplicate tool names detected: {duplicate_names}"
        )

    invalid_tools = [
        tool for tool in all_tools
        if not tool.name or not tool.description
    ]

    if invalid_tools:
        raise ValueError(
            "Some registered tools have missing names or descriptions."
        )

    return {
        "success": True,
        "total_tools": len(all_tools),
        "total_categories": len(tool_categories),
        "category_counts": {
            category: len(tools)
            for category, tools in tool_categories.items()
        },
        "tool_names": tool_names,
    }
