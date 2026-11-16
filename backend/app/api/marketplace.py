
import pandas as pd

from fastapi import APIRouter, HTTPException, Query

from app.analytics.marketplace import (
    get_marketplace_kpis,
    get_monthly_marketplace_trends,
    get_cancellation_trend,
)


router = APIRouter(
    prefix="/api/analytics/marketplace",
    tags=["Marketplace Analytics"]
)


def serialize_record(record: dict) -> dict:
    """Convert Pandas/NumPy values into JSON-compatible values."""

    return pd.Series(record).to_json(date_format="iso", date_unit="ms") | {}


def serialize_dataframe(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame into JSON-compatible records."""

    import json

    return json.loads(
        df.to_json(
            orient="records",
            date_format="iso",
            date_unit="ms"
        )
    )


def validate_dates(start_date, end_date):
    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=422,
            detail="start_date cannot be after end_date"
        )


@router.get("/kpis")
def marketplace_kpis(
    start_date: str | None = Query(default=None),
    end_date: str | None = Query(default=None)
):
    from datetime import date

    parsed_start = date.fromisoformat(start_date) if start_date else None
    parsed_end = date.fromisoformat(end_date) if end_date else None

    validate_dates(parsed_start, parsed_end)

    try:
        result = get_marketplace_kpis(parsed_start, parsed_end)

        import json
        return json.loads(pd.Series(result).to_json(date_format="iso"))

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve marketplace KPIs"
        ) from exc


@router.get("/trends")
def marketplace_trends(
    start_date: str | None = Query(default=None),
    end_date: str | None = Query(default=None)
):
    from datetime import date

    parsed_start = date.fromisoformat(start_date) if start_date else None
    parsed_end = date.fromisoformat(end_date) if end_date else None

    validate_dates(parsed_start, parsed_end)

    try:
        result = get_monthly_marketplace_trends(
            parsed_start,
            parsed_end
        )

        return serialize_dataframe(result)

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve marketplace trends"
        ) from exc


@router.get("/cancellations")
def marketplace_cancellations(
    start_date: str | None = Query(default=None),
    end_date: str | None = Query(default=None)
):
    from datetime import date

    parsed_start = date.fromisoformat(start_date) if start_date else None
    parsed_end = date.fromisoformat(end_date) if end_date else None

    validate_dates(parsed_start, parsed_end)

    try:
        result = get_cancellation_trend(
            parsed_start,
            parsed_end
        )

        return serialize_dataframe(result)

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve cancellation trends"
        ) from exc
