
from app.database.query_executor import fetch_dataframe


# ---------------------------------------------------------
# 1. SELLER OVERVIEW
# ---------------------------------------------------------

def get_seller_overview():
    """
    Returns overall seller marketplace metrics.

    Note:
    - total_orders is the sum of seller-associated orders.
    - It is not necessarily the number of distinct marketplace orders.
    - unique_customers summed across sellers represents
      seller-customer relationships, not distinct marketplace customers.
    """

    query = """
        SELECT
            COUNT(*) AS total_sellers,

            SUM(total_items_sold) AS total_items_sold,
            SUM(delivered_items) AS delivered_items,

            SUM(total_orders) AS seller_order_associations,
            SUM(delivered_orders) AS delivered_seller_order_associations,

            SUM(total_product_value) AS total_product_value,
            SUM(delivered_gmv) AS delivered_gmv,
            ROUND(AVG(avg_freight_value), 2) AS avg_seller_freight_value,

            SUM(unique_customers) AS seller_customer_relationships,

            SUM(eligible_delivered_orders) AS eligible_seller_order_associations,
            SUM(late_orders) AS late_seller_order_associations,

            ROUND(
                SUM(delivered_gmv) /
                NULLIF(SUM(delivered_items), 0),
                2
            ) AS weighted_avg_delivered_item_price,

            ROUND(
                AVG(avg_selling_price),
                2
            ) AS avg_seller_avg_selling_price

        FROM olist.vw_seller_kpis;
    """

    return fetch_dataframe(query)


# ---------------------------------------------------------
# 2. TOP SELLERS
# ---------------------------------------------------------

def get_top_sellers(metric="delivered_gmv", limit=10):
    """
    Returns top sellers based on a selected metric.

    Supported metrics:
    - delivered_gmv
    - delivered_orders
    - delivered_items
    - total_items_sold
    - total_orders
    - unique_customers
    - avg_selling_price
    """

    allowed_metrics = {
        "delivered_gmv",
        "delivered_orders",
        "delivered_items",
        "total_items_sold",
        "total_orders",
        "unique_customers",
        "avg_selling_price",
    }

    if metric not in allowed_metrics:
        raise ValueError(
            f"Invalid metric: {metric}. "
            f"Choose from {sorted(allowed_metrics)}"
        )

    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ValueError("limit must be an integer")

    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")

    query = f"""
        SELECT
            seller_id,
            total_items_sold,
            total_orders,
            delivered_items,
            delivered_orders,
            total_product_value,
            delivered_gmv,
            avg_selling_price,
            avg_freight_value,
            unique_customers,
            eligible_delivered_orders,
            late_orders

        FROM olist.vw_seller_kpis

        ORDER BY {metric} DESC NULLS LAST

        LIMIT :limit;
    """

    return fetch_dataframe(query, {"limit": limit})


# ---------------------------------------------------------
# 3. SELLER DELIVERY PERFORMANCE
# ---------------------------------------------------------

def get_seller_delivery_performance(limit=20):
    """
    Returns sellers ranked by late delivery rate.

    Late delivery rate:
        late_orders / eligible_delivered_orders * 100

    Only sellers with at least one eligible delivered order
    are included.
    """

    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ValueError("limit must be an integer")

    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")

    query = """
        SELECT
            seller_id,
            delivered_orders,
            eligible_delivered_orders,
            late_orders,

            ROUND(
                100.0 * late_orders /
                NULLIF(eligible_delivered_orders, 0),
                2
            ) AS late_delivery_rate,

            ROUND(
                100.0 * (
                    eligible_delivered_orders - late_orders
                ) /
                NULLIF(eligible_delivered_orders, 0),
                2
            ) AS on_time_delivery_rate

        FROM olist.vw_seller_kpis

        WHERE eligible_delivered_orders > 0

        ORDER BY late_delivery_rate DESC, late_orders DESC

        LIMIT :limit;
    """

    return fetch_dataframe(query, {"limit": limit})


# ---------------------------------------------------------
# 4. SELLER GMV CONTRIBUTION
# ---------------------------------------------------------


def get_seller_contribution(limit=20):
    """
    Calculates seller contribution to total delivered GMV.

    Sellers with NULL delivered GMV are excluded because
    they have no recorded delivered GMV to contribute.
    """

    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ValueError("limit must be an integer")

    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")

    query = """
        WITH seller_contribution AS (
            SELECT
                seller_id,
                delivered_gmv,

                SUM(delivered_gmv) OVER ()
                    AS marketplace_delivered_gmv,

                SUM(delivered_gmv) OVER (
                    ORDER BY delivered_gmv DESC NULLS LAST,
                             seller_id
                    ROWS BETWEEN UNBOUNDED PRECEDING
                    AND CURRENT ROW
                ) AS cumulative_gmv

            FROM olist.vw_seller_kpis

            WHERE delivered_gmv IS NOT NULL
        )

        SELECT
            seller_id,
            delivered_gmv,

            ROUND(
                100.0 * delivered_gmv /
                NULLIF(marketplace_delivered_gmv, 0),
                2
            ) AS gmv_percentage,

            ROUND(
                100.0 * cumulative_gmv /
                NULLIF(marketplace_delivered_gmv, 0),
                2
            ) AS cumulative_gmv_percentage

        FROM seller_contribution

        ORDER BY delivered_gmv DESC NULLS LAST,
                 seller_id

        LIMIT :limit;
    """

    return fetch_dataframe(query, {"limit": limit})

# ---------------------------------------------------------
# 5. SELLER PERFORMANCE SUMMARY
# ---------------------------------------------------------

def get_seller_performance_summary():
    """
    Returns a consolidated seller performance summary,
    including top seller and seller concentration.
    """

    query = """
        WITH ranked_sellers AS (
            SELECT
                seller_id,
                delivered_gmv,

                ROW_NUMBER() OVER (
                    ORDER BY delivered_gmv DESC NULLS LAST, seller_id
                ) AS seller_rank,

                SUM(delivered_gmv) OVER () AS total_marketplace_gmv,

                SUM(delivered_gmv) OVER (
                    ORDER BY delivered_gmv DESC, seller_id
                    ROWS BETWEEN UNBOUNDED PRECEDING
                    AND CURRENT ROW
                ) AS cumulative_gmv

            FROM olist.vw_seller_kpis
        ),

        seller_summary AS (
            SELECT
                COUNT(*) AS total_sellers,
                SUM(total_items_sold) AS total_items_sold,
                SUM(delivered_items) AS delivered_items,
                SUM(total_orders) AS seller_order_associations,
                SUM(delivered_orders) AS delivered_seller_order_associations,
                SUM(delivered_gmv) AS delivered_gmv,
                ROUND(AVG(avg_freight_value), 2) AS avg_seller_freight_value,
                SUM(unique_customers) AS seller_customer_relationships,
                SUM(eligible_delivered_orders) AS eligible_seller_order_associations,
                SUM(late_orders) AS late_seller_order_associations
            FROM olist.vw_seller_kpis
        )

        SELECT
            s.*,

            ROUND(
                100.0 * s.late_seller_order_associations /
                NULLIF(s.eligible_seller_order_associations, 0),
                2
            ) AS late_delivery_rate,

            MAX(
                CASE WHEN r.seller_rank = 1
                    THEN r.seller_id
                END
            ) AS top_seller_id,

            MAX(
                CASE WHEN r.seller_rank = 1
                    THEN r.delivered_gmv
                END
            ) AS top_seller_gmv,

            ROUND(
                100.0 * SUM(
                    CASE WHEN r.seller_rank <= 10
                        THEN r.delivered_gmv
                        ELSE 0
                    END
                ) / NULLIF(MAX(r.total_marketplace_gmv), 0),
                2
            ) AS top_10_sellers_gmv_percentage

        FROM seller_summary s
        CROSS JOIN ranked_sellers r

        GROUP BY
            s.total_sellers,
            s.total_items_sold,
            s.delivered_items,
            s.seller_order_associations,
            s.delivered_seller_order_associations,
            s.delivered_gmv,
            s.avg_seller_freight_value,
            s.seller_customer_relationships,
            s.eligible_seller_order_associations,
            s.late_seller_order_associations;
    """

    return fetch_dataframe(query)
