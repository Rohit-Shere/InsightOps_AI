
import json
import pandas as pd

from typing import Literal
from fastapi import APIRouter, HTTPException, Query

from app.analytics.products import (
    get_product_overview,
    get_top_products,
    get_category_performance,
    get_category_contribution,
    get_product_price_distribution,
    get_product_performance_summary,
)


router = APIRouter(
    prefix="/api/analytics/products",
    tags=["Product & Category Analytics"]
)


def serialize_records(dataframe: pd.DataFrame) -> list[dict]:
    """Convert DataFrame results into JSON-compatible records."""

    return json.loads(
        dataframe.to_json(
            orient="records",
            date_format="iso"
        )
    )


def serialize_record(record: dict) -> dict:
    """Convert a single result into a JSON-compatible object."""

    dataframe = pd.DataFrame([record])
    return serialize_records(dataframe)[0]


@router.get("/overview")
def product_overview():
    try:
        result = get_product_overview()
        return serialize_record(result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve product overview"
        ) from exc


@router.get("/top")
def top_products(
    metric: Literal[
        "delivered_gmv",
        "delivered_items",
        "total_items_sold",
        "total_orders",
        "avg_selling_price",
        "unique_customers"
    ] = "delivered_gmv",

    limit: int = Query(default=10, ge=1, le=100)
):
    try:
        result = get_top_products(
            metric=metric,
            limit=limit
        )

        return serialize_records(result)

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve top products"
        ) from exc


@router.get("/categories/performance")
def category_performance():
    try:
        result = get_category_performance()
        return serialize_records(result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve category performance"
        ) from exc


@router.get("/categories/contribution")
def category_contribution():
    try:
        result = get_category_contribution()
        return serialize_records(result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve category contribution"
        ) from exc


@router.get("/price-distribution")
def product_price_distribution():
    try:
        result = get_product_price_distribution()
        return serialize_records(result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve product price distribution"
        ) from exc


@router.get("/performance-summary")
def product_performance_summary():
    try:
        result = get_product_performance_summary()
        return serialize_record(result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve product performance summary"
        ) from exc
