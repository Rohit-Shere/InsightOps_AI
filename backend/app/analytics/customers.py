
import pandas as pd

from app.database.query_executor import fetch_dataframe


# ---------------------------------------------------------
# 1. CUSTOMER OVERVIEW
# ---------------------------------------------------------

def get_customer_overview():
    """
    Returns overall customer purchasing metrics.

    Customer grain: customer_unique_id
    Repeat customer: At least 2 delivered orders.
    """

    query = """
        SELECT
            COUNT(*) AS total_customers,

            COUNT(*) FILTER (
                WHERE delivered_orders > 0
            ) AS customers_with_delivered_orders,

            COUNT(*) FILTER (
                WHERE delivered_orders >= 2
            ) AS repeat_customers,

            COALESCE(SUM(total_orders), 0) AS total_orders,

            COALESCE(SUM(delivered_orders), 0)
                AS total_delivered_orders,

            COALESCE(SUM(canceled_orders), 0)
                AS total_canceled_orders,

            COALESCE(SUM(delivered_product_value), 0)
                AS delivered_gmv,

            ROUND(
                AVG(delivered_product_value)
                FILTER (WHERE delivered_orders > 0),
                2
            ) AS avg_delivered_spend_per_customer,

            ROUND(
                AVG(delivered_aov)
                FILTER (WHERE delivered_orders > 0),
                2
            ) AS avg_customer_aov,

            ROUND(
                AVG(delivered_orders::numeric)
                FILTER (WHERE delivered_orders > 0),
                2
            ) AS avg_delivered_orders_per_active_customer,

            ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE delivered_orders >= 2
                )
                / NULLIF(
                    COUNT(*) FILTER (
                        WHERE delivered_orders > 0
                    ), 0
                ),
                2
            ) AS repeat_customer_rate

        FROM olist.vw_customer_kpis;
    """

    result = fetch_dataframe(query)

    return result.iloc[0].to_dict()


# ---------------------------------------------------------
# 2. CUSTOMER FREQUENCY DISTRIBUTION
# ---------------------------------------------------------

def get_customer_frequency_distribution():
    """
    Returns customer distribution by simple frequency segment.

    Segments:
    - One-time
    - Occasional
    - Frequent
    - Loyal

    Uses vw_customer_kpis.
    """

    query = """
        SELECT
            frequency_segment,

            COUNT(*) AS total_customers,

            ROUND(
                100.0 * COUNT(*)
                / NULLIF(SUM(COUNT(*)) OVER (), 0),
                2
            ) AS customer_percentage,

            ROUND(
                AVG(total_orders::numeric),
                2
            ) AS avg_total_orders,

            ROUND(
                AVG(delivered_product_value),
                2
            ) AS avg_delivered_spend

        FROM olist.vw_customer_kpis

        GROUP BY frequency_segment

        ORDER BY total_customers DESC;
    """

    return fetch_dataframe(query)


# ---------------------------------------------------------
# 3. RFM BUSINESS SEGMENT ANALYSIS
# ---------------------------------------------------------

def get_customer_segments():
    """
    Returns customer distribution and delivered GMV
    by business-oriented RFM customer segment.

    Uses vw_customer_segments.

    Note:
    - Monetary represents delivered product value.
    - It does not represent profit.
    """

    query = """
        WITH segment_metrics AS (
            SELECT
                customer_segment,

                COUNT(*) AS total_customers,

                SUM(monetary) AS delivered_gmv,

                AVG(recency) AS avg_recency_days,

                AVG(frequency) AS avg_frequency,

                AVG(monetary) AS avg_monetary

            FROM olist.vw_customer_segments

            GROUP BY customer_segment
        )

        SELECT
            customer_segment,

            total_customers,

            ROUND(
                100.0 * total_customers
                / NULLIF(SUM(total_customers) OVER (), 0),
                2
            ) AS customer_percentage,

            ROUND(delivered_gmv, 2) AS delivered_gmv,

            ROUND(
                100.0 * delivered_gmv
                / NULLIF(SUM(delivered_gmv) OVER (), 0),
                2
            ) AS gmv_percentage,

            ROUND(avg_recency_days, 2)
                AS avg_recency_days,

            ROUND(avg_frequency, 2)
                AS avg_frequency,

            ROUND(avg_monetary, 2)
                AS avg_monetary

        FROM segment_metrics

        ORDER BY delivered_gmv DESC;
    """

    return fetch_dataframe(query)


# ---------------------------------------------------------
# 4. RFM SCORE SEGMENT DISTRIBUTION
# ---------------------------------------------------------

def get_rfm_distribution():
    """
    Returns customer distribution by RFM score segment.

    Uses vw_customer_rfm.

    Includes:
    - Customer count
    - Average recency
    - Average frequency
    - Average monetary value
    - Average RFM total score
    """

    query = """
        SELECT
            rfm_segment,

            COUNT(*) AS total_customers,

            ROUND(
                100.0 * COUNT(*)
                / NULLIF(SUM(COUNT(*)) OVER (), 0),
                2
            ) AS customer_percentage,

            ROUND(AVG(recency::numeric), 2)
                AS avg_recency_days,

            ROUND(AVG(frequency::numeric), 2)
                AS avg_frequency,

            ROUND(AVG(monetary), 2)
                AS avg_monetary,

            ROUND(AVG(rfm_total_score::numeric), 2)
                AS avg_rfm_score

        FROM olist.vw_customer_rfm

        GROUP BY rfm_segment

        ORDER BY total_customers DESC;
    """

    return fetch_dataframe(query)
