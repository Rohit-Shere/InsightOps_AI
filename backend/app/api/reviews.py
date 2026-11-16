
import json
import pandas as pd

from datetime import date
from fastapi import APIRouter, HTTPException

from app.analytics.reviews import (
    get_review_overview,
    get_review_score_distribution,
    get_review_sentiment_analysis,
    get_review_delivery_relationship,
    get_review_monthly_trend,
    get_review_analytics_summary,
)


router = APIRouter(
    prefix="/api/analytics/reviews",
    tags=["Review Analytics"]
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
def review_overview():
    try:
        result = get_review_overview()
        return serialize_single_record(result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve review overview"
        ) from exc


@router.get("/distribution")
def review_score_distribution():
    try:
        result = get_review_score_distribution()
        return serialize_records(result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve review score distribution"
        ) from exc


@router.get("/sentiment")
def review_sentiment_analysis():
    try:
        result = get_review_sentiment_analysis()
        return serialize_records(result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve review sentiment analysis"
        ) from exc


@router.get("/delivery-relationship")
def review_delivery_relationship():
    try:
        result = get_review_delivery_relationship()
        return serialize_records(result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve review delivery relationship"
        ) from exc


@router.get("/trends")
def review_monthly_trends(
    start_date: date | None = None,
    end_date: date | None = None
):
    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=422,
            detail="start_date cannot be after end_date"
        )

    try:
        result = get_review_monthly_trend(
            start_date=start_date,
            end_date=end_date
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
            detail="Failed to retrieve monthly review trends"
        ) from exc


@router.get("/summary")
def review_analytics_summary():
    try:
        result = get_review_analytics_summary()
        return serialize_single_record(result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve review analytics summary"
        ) from exc
