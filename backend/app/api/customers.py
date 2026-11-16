
import json
import pandas as pd

from fastapi import APIRouter, HTTPException

from app.analytics.customers import (
    get_customer_overview,
    get_customer_frequency_distribution,
    get_customer_segments,
    get_rfm_distribution,
)


router = APIRouter(
    prefix="/api/analytics/customers",
    tags=["Customer Analytics"]
)


def serialize_records(dataframe: pd.DataFrame) -> list[dict]:
    """
    Convert DataFrame results into JSON-compatible records.
    Handles timestamps, NumPy values and missing values.
    """
    return json.loads(
        dataframe.to_json(
            orient="records",
            date_format="iso"
        )
    )


def serialize_record(record: dict) -> dict:
    """Convert a single aggregated result into a JSON object."""
    dataframe = pd.DataFrame([record])
    return serialize_records(dataframe)[0]


@router.get("/overview")
def customer_overview():
    try:
        result = get_customer_overview()
        return serialize_record(result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve customer overview"
        ) from exc


@router.get("/frequency")
def customer_frequency():
    try:
        result = get_customer_frequency_distribution()
        return serialize_records(result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve customer frequency distribution"
        ) from exc


@router.get("/segments")
def customer_segments():
    try:
        result = get_customer_segments()
        return serialize_records(result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve customer segments"
        ) from exc


@router.get("/rfm")
def customer_rfm_distribution():
    try:
        result = get_rfm_distribution()
        return serialize_records(result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve RFM distribution"
        ) from exc
