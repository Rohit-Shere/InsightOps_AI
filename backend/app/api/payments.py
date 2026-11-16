
import json
import pandas as pd

from fastapi import APIRouter, HTTPException

from app.analytics.payments import (
    get_payment_overview,
    get_payment_method_distribution,
    get_payment_installment_analysis,
    get_payment_value_distribution,
    get_payment_anomalies,
    get_payment_analytics_summary,
)


router = APIRouter(
    prefix="/api/analytics/payments",
    tags=["Payment Analytics"]
)


def serialize_records(dataframe: pd.DataFrame) -> list[dict]:
    """Convert DataFrame results into JSON-compatible records."""

    return json.loads(
        dataframe.to_json(
            orient="records",
            date_format="iso"
        )
    )


def serialize_single_record(dataframe: pd.DataFrame) -> dict:
    """Convert a single-row DataFrame into a JSON object."""

    records = serialize_records(dataframe)

    return records[0] if records else {}


@router.get("/overview")
def payment_overview():
    try:
        result = get_payment_overview()
        return serialize_single_record(result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve payment overview"
        ) from exc


@router.get("/methods")
def payment_method_distribution():
    try:
        result = get_payment_method_distribution()
        return serialize_records(result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve payment method distribution"
        ) from exc


@router.get("/installments")
def payment_installment_analysis():
    try:
        result = get_payment_installment_analysis()
        return serialize_records(result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve payment installment analysis"
        ) from exc


@router.get("/value-distribution")
def payment_value_distribution():
    try:
        result = get_payment_value_distribution()
        return serialize_records(result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve payment value distribution"
        ) from exc


@router.get("/anomalies")
def payment_anomalies():
    try:
        result = get_payment_anomalies()
        return serialize_single_record(result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve payment anomalies"
        ) from exc


@router.get("/summary")
def payment_analytics_summary():
    try:
        result = get_payment_analytics_summary()
        return serialize_single_record(result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve payment analytics summary"
        ) from exc
