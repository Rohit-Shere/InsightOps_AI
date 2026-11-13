from datetime import date
from typing import Optional

from app.database.query_executor import fetch_dataframe


def _validate_date_range(
    start_date: Optional[date],
    end_date: Optional[date]
):
    """Validate the requested analysis date range."""

    if start_date and end_date and start_date > end_date:
        raise ValueError("start_date cannot be after end_date")


def get_marketplace_kpis(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> dict:
    """
    Retrieve overall marketplace KPIs.

    Grain: One aggregated result for the selected period.
    Source: olist.vw_order_analytics
    """

    _validate_date_range(start_date, end_date)

    query = """
        SELECT
            COUNT(*) AS total_orders,

            COUNT(*) FILTER (
                WHERE order_status = 'delivered'
            ) AS delivered_orders,

            COUNT(*) FILTER (
                WHERE order_status = 'canceled'
            ) AS canceled_orders,

            COUNT(DISTINCT customer_unique_id) AS unique_customers,

            COALESCE(
                SUM(product_value), 0
            ) AS ordered_product_value,

            COALESCE(
                SUM(product_value) FILTER (
                    WHERE order_status = 'delivered'
                ), 0
            ) AS delivered_gmv,

            ROUND(
                (
                    SUM(product_value) FILTER (
                        WHERE order_status = 'delivered'
                    ) / NULLIF(
                        COUNT(*) FILTER (
                            WHERE order_status = 'delivered'
                        ), 0
                    )
                )::numeric, 2
            ) AS delivered_aov,

            COUNT(*) FILTER (
                WHERE order_status = 'delivered'
                AND is_late_delivery IS NOT NULL
            ) AS eligible_delivered_orders,

            COUNT(*) FILTER (
                WHERE order_status = 'delivered'
                AND is_late_delivery IS TRUE
            ) AS late_orders,

            ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE order_status = 'delivered'
                    AND is_late_delivery IS TRUE
                ) / NULLIF(
                    COUNT(*) FILTER (
                        WHERE order_status = 'delivered'
                        AND is_late_delivery IS NOT NULL
                    ), 0
                ), 2
            ) AS late_delivery_rate,

            ROUND(
                AVG(delivery_days) FILTER (
                    WHERE order_status = 'delivered'
                )::numeric, 2
            ) AS avg_delivery_days,

            ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE order_status = 'canceled'
                ) / NULLIF(COUNT(*), 0), 2
            ) AS cancellation_rate

        FROM olist.vw_order_analytics

        WHERE
            (
                CAST(:start_date AS date) IS NULL
                OR order_purchase_timestamp::date >= CAST(:start_date AS date)
            )
            AND
            (
                CAST(:end_date AS date) IS NULL
                OR order_purchase_timestamp::date <= CAST(:end_date AS date)
            );
    """

    result = fetch_dataframe(
        query,
        {
            "start_date": start_date,
            "end_date": end_date
        }
    )

    return result.iloc[0].to_dict()


def get_monthly_marketplace_trends(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
):
    """
    Retrieve monthly marketplace performance.

    Grain: One row per purchase month.
    Source: olist.vw_monthly_kpis
    """

    _validate_date_range(start_date, end_date)

    query = """
        SELECT
            month,
            total_orders,
            delivered_orders,
            canceled_orders,
            unique_customers,
            ordered_product_value,
            delivered_gmv,
            delivered_aov,
            eligible_delivered_orders,
            late_orders,
            late_delivery_rate,
            avg_delivery_days,
            cancellation_rate

        FROM olist.vw_monthly_kpis

        WHERE
            (
                CAST(:start_date AS date) IS NULL
                OR month >= DATE_TRUNC(
                    'month', CAST(:start_date AS date)
                )
            )
            AND
            (
                CAST(:end_date AS date) IS NULL
                OR month <= DATE_TRUNC(
                    'month', CAST(:end_date AS date)
                )
            )

        ORDER BY month;
    """

    return fetch_dataframe(
        query,
        {
            "start_date": start_date,
            "end_date": end_date
        }
    )


def get_cancellation_trend(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
):
    """
    Retrieve monthly cancellation metrics.

    Source: olist.vw_monthly_kpis
    """

    _validate_date_range(start_date, end_date)

    query = """
        SELECT
            month,
            total_orders,
            canceled_orders,
            cancellation_rate

        FROM olist.vw_monthly_kpis

        WHERE
            (
                CAST(:start_date AS date) IS NULL
                OR month >= DATE_TRUNC(
                    'month', CAST(:start_date AS date)
                )
            )
            AND
            (
                CAST(:end_date AS date) IS NULL
                OR month <= DATE_TRUNC(
                    'month', CAST(:end_date AS date)
                )
            )

        ORDER BY month;
    """

    return fetch_dataframe(
        query,
        {
            "start_date": start_date,
            "end_date": end_date
        }
    )
