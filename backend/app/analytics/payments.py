
from app.database.query_executor import fetch_dataframe


def get_payment_overview():
    """
    Overall payment metrics.

    Grain: one row per payment record.
    Distinct order counts are calculated separately.
    """
    query = """
        SELECT
            COUNT(*) AS total_payment_records,
            COUNT(DISTINCT order_id) AS unique_orders,
            COUNT(DISTINCT customer_unique_id) AS unique_customers,

            COUNT(*) FILTER (
                WHERE is_delivered = TRUE
            ) AS delivered_payment_records,

            COUNT(DISTINCT order_id) FILTER (
                WHERE is_delivered = TRUE
            ) AS delivered_orders,

            COUNT(DISTINCT order_id) FILTER (
                WHERE order_status = 'canceled'
            ) AS canceled_orders,

            ROUND(SUM(payment_value), 2)
                AS total_recorded_payment_value,

            ROUND(
                SUM(payment_value) FILTER (
                    WHERE is_delivered = TRUE
                ),
                2
            ) AS delivered_order_recorded_payment_value,

            ROUND(AVG(payment_value), 2)
                AS avg_payment_record_value,

            ROUND(
                SUM(payment_value) /
                NULLIF(COUNT(DISTINCT order_id), 0),
                2
            ) AS avg_recorded_payment_per_order

        FROM olist.vw_payment_analytics;
    """
    return fetch_dataframe(query)


def get_payment_method_distribution():
    """
    Payment distribution by payment method.

    Percentages are based on total recorded payment value.
    """
    query = """
        WITH payment_methods AS (
            SELECT
                payment_type,
                COUNT(*) AS payment_records,
                COUNT(DISTINCT order_id) AS unique_orders,
                COUNT(DISTINCT customer_unique_id)
                    AS unique_customers,
                SUM(payment_value) AS total_payment_value
            FROM olist.vw_payment_analytics
            GROUP BY payment_type
        )
        SELECT
            payment_type,
            payment_records,
            unique_orders,
            unique_customers,
            ROUND(total_payment_value, 2)
                AS total_payment_value,

            ROUND(
                100.0 * payment_records /
                NULLIF(SUM(payment_records) OVER (), 0),
                2
            ) AS record_percentage,

            ROUND(
                100.0 * total_payment_value /
                NULLIF(SUM(total_payment_value) OVER (), 0),
                2
            ) AS payment_value_percentage

        FROM payment_methods
        ORDER BY total_payment_value DESC NULLS LAST;
    """
    return fetch_dataframe(query)


def get_payment_installment_analysis():
    """
    Analyses payment records by installment count.
    """
    query = """
        SELECT
            payment_installments,
            COUNT(*) AS payment_records,
            COUNT(DISTINCT order_id) AS unique_orders,
            ROUND(SUM(payment_value), 2)
                AS total_payment_value,
            ROUND(AVG(payment_value), 2)
                AS avg_payment_value,

            ROUND(
                100.0 * COUNT(*) /
                NULLIF(SUM(COUNT(*)) OVER (), 0),
                2
            ) AS record_percentage

        FROM olist.vw_payment_analytics
        GROUP BY payment_installments
        ORDER BY payment_installments NULLS FIRST;
    """
    return fetch_dataframe(query)


def get_payment_value_distribution():
    """
    Groups individual payment records into value bands.

    These are payment-record bands, not order-level bands.
    """
    query = """
        WITH payment_bands AS (
            SELECT
                CASE
                    WHEN payment_value IS NULL THEN 'Missing'
                    WHEN payment_value < 50 THEN 'Below 50'
                    WHEN payment_value < 100 THEN '50-99'
                    WHEN payment_value < 250 THEN '100-249'
                    WHEN payment_value < 500 THEN '250-499'
                    ELSE '500+'
                END AS payment_value_band,

                CASE
                    WHEN payment_value IS NULL THEN 0
                    WHEN payment_value < 50 THEN 1
                    WHEN payment_value < 100 THEN 2
                    WHEN payment_value < 250 THEN 3
                    WHEN payment_value < 500 THEN 4
                    ELSE 5
                END AS band_order,

                payment_value,
                order_id

            FROM olist.vw_payment_analytics
        )
        SELECT
            payment_value_band,
            COUNT(*) AS payment_records,
            COUNT(DISTINCT order_id) AS unique_orders,
            ROUND(SUM(payment_value), 2)
                AS total_payment_value,
            ROUND(AVG(payment_value), 2)
                AS avg_payment_value,

            ROUND(
                100.0 * COUNT(*) /
                NULLIF(SUM(COUNT(*)) OVER (), 0),
                2
            ) AS record_percentage

        FROM payment_bands
        GROUP BY payment_value_band, band_order
        ORDER BY band_order;
    """
    return fetch_dataframe(query)


def get_payment_anomalies():
    """
    Checks known payment-data quality exceptions.

    Missing payment orders are checked against the raw orders table.
    """
    query = """
        SELECT
            COUNT(*) FILTER (
                WHERE payment_value = 0
            ) AS zero_value_records,

            COUNT(*) FILTER (
                WHERE payment_type IS NULL
                OR LOWER(TRIM(payment_type)) IN ('', 'not_defined', 'unknown')
            ) AS unspecified_payment_type_records,

            COUNT(*) FILTER (
                WHERE payment_value < 0
            ) AS negative_payment_records,

            COUNT(*) FILTER (
                WHERE payment_type = 'credit_card'
                  AND payment_installments = 0
            ) AS credit_card_zero_installment_records,

            (
                SELECT COUNT(*)
                FROM olist.orders o
                LEFT JOIN (
                    SELECT DISTINCT order_id
                    FROM olist.vw_payment_analytics
                ) p ON o.order_id = p.order_id
                WHERE p.order_id IS NULL
            ) AS orders_without_payment_records,

            COUNT(*) FILTER (
                WHERE payment_type IS NULL
            ) AS missing_payment_type_records

        FROM olist.vw_payment_analytics;
    """
    return fetch_dataframe(query)


def get_payment_analytics_summary():
    """
    Consolidated payment analytics summary.
    """
    query = """
        WITH overall AS (
            SELECT
                COUNT(*) AS total_payment_records,
                COUNT(DISTINCT order_id) AS unique_orders,
                ROUND(SUM(payment_value), 2)
                    AS total_recorded_payment_value
            FROM olist.vw_payment_analytics
        ),

        method_ranking AS (
            SELECT
                payment_type,
                SUM(payment_value) AS method_payment_value,
                ROW_NUMBER() OVER (
                    ORDER BY SUM(payment_value) DESC NULLS LAST
                ) AS method_rank
            FROM olist.vw_payment_analytics
            GROUP BY payment_type
        ),

        installment_ranking AS (
            SELECT
                payment_installments,
                COUNT(*) AS installment_records,
                ROW_NUMBER() OVER (
                    ORDER BY COUNT(*) DESC
                ) AS installment_rank
            FROM olist.vw_payment_analytics
            GROUP BY payment_installments
        )

        SELECT
            o.total_payment_records,
            o.unique_orders,
            o.total_recorded_payment_value,

            m.payment_type AS top_payment_method,
            ROUND(m.method_payment_value, 2)
                AS top_payment_method_value,

            i.payment_installments AS most_common_installment_count,
            i.installment_records AS most_common_installment_records

        FROM overall o
        LEFT JOIN method_ranking m
            ON m.method_rank = 1
        LEFT JOIN installment_ranking i
            ON i.installment_rank = 1;
    """
    return fetch_dataframe(query)
