
import json
import pandas as pd

from typing import Literal
from fastapi import APIRouter, HTTPException, Query

from app.analytics.sellers import (
    get_seller_overview,
    get_top_sellers,
    get_seller_delivery_performance,
    get_seller_contribution,
    get_seller_performance_summary,
)


router = APIRouter(
    prefix="/api/analytics/sellers",
    tags=["Seller Analytics"]
)


def serialize_records(dataframe: pd.DataFrame) -> list[dict]:
    """Convert DataFrame results into JSON-compatible records."""

    return json.loads(
        dataframe.to_json(
            orient="records",
            date_format="iso"
        )
    )


def serialize_record(dataframe: pd.DataFrame) -> dict:
    """Convert a single-row DataFrame into a JSON object."""

    records = serialize_records(dataframe)

    return records[0] if records else {}


@router.get("/overview")
def seller_overview():
    try:
        result = get_seller_overview()
        return serialize_record(result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve seller overview"
        ) from exc


@router.get("/top")
def top_sellers(
    metric: Literal[
        "delivered_gmv",
        "delivered_orders",
        "delivered_items",
        "total_items_sold",
        "total_orders",
        "unique_customers",
        "avg_selling_price"
    ] = "delivered_gmv",

    limit: int = Query(default=10, ge=1, le=100)
):
    try:
        result = get_top_sellers(
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
            detail="Failed to retrieve top sellers"
        ) from exc


@router.get("/delivery-performance")
def seller_delivery_performance(
    limit: int = Query(default=20, ge=1, le=100)
):
    try:
        result = get_seller_delivery_performance(limit=limit)
        return serialize_records(result)

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve seller delivery performance"
        ) from exc


@router.get("/contribution")
def seller_contribution(
    limit: int = Query(default=20, ge=1, le=100)
):
    try:
        result = get_seller_contribution(limit=limit)
        return serialize_records(result)

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve seller contribution"
        ) from exc


@router.get("/performance-summary")
def seller_performance_summary():
    try:
        result = get_seller_performance_summary()
        return serialize_record(result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve seller performance summary"
        ) from exc
