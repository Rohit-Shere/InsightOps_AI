
from app.database.query_executor import fetch_dataframe


def get_product_overview():
    """
    Overall product-level marketplace metrics.

    Note:
    - Product counts and item counts are additive.
    - Product-order combinations are not distinct marketplace orders.
    - Customer and seller counts are not summed across products.
    """
    query = """
        SELECT
            COUNT(*) AS total_products,
            COUNT(*) FILTER (
                WHERE total_items_sold > 0
            ) AS products_with_sales,
            COUNT(*) FILTER (
                WHERE delivered_items > 0
            ) AS products_with_deliveries,
            COALESCE(SUM(total_items_sold), 0)
                AS total_items_sold,
            COALESCE(SUM(delivered_items), 0)
                AS delivered_items,
            COALESCE(SUM(canceled_items), 0)
                AS canceled_items,
            COALESCE(SUM(delivered_gmv), 0)
                AS delivered_gmv,
            COALESCE(SUM(total_product_value), 0)
                AS total_product_value,
            COALESCE(SUM(total_freight_value), 0)
                AS total_freight_value,
            ROUND(AVG(avg_selling_price), 2)
                AS avg_product_selling_price,
            ROUND(
                SUM(delivered_gmv)
                / NULLIF(SUM(delivered_items), 0),
                2
            ) AS avg_delivered_item_price,
            COALESCE(SUM(total_orders), 0)
                AS product_order_combinations
        FROM olist.vw_product_kpis;
    """
    result = fetch_dataframe(query)
    return result.iloc[0].to_dict()


def get_top_products(metric="delivered_gmv", limit=10):
    """
    Rank products by an approved metric.

    Supported metrics:
    - delivered_gmv
    - delivered_items
    - total_items_sold
    - total_orders
    - avg_selling_price
    - unique_customers
    """
    allowed_metrics = {
        "delivered_gmv",
        "delivered_items",
        "total_items_sold",
        "total_orders",
        "avg_selling_price",
        "unique_customers",
    }

    if metric not in allowed_metrics:
        raise ValueError(
            f"Unsupported metric: {metric}. "
            f"Choose from {sorted(allowed_metrics)}."
        )

    if not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer between 1 and 100.")

    query = f"""
        SELECT
            product_id,
            product_category,
            total_items_sold,
            total_orders,
            unique_customers,
            unique_sellers,
            delivered_items,
            delivered_orders,
            delivered_gmv,
            avg_selling_price,
            canceled_items
        FROM olist.vw_product_kpis
        ORDER BY {metric} DESC NULLS LAST
        LIMIT :limit;
    """

    return fetch_dataframe(query, params={"limit": limit})


def get_category_performance():
    """
    Returns category-level product, volume, value and pricing metrics.
    """
    query = """
        SELECT
            product_category,
            total_products,
            total_items_sold,
            product_order_combinations,
            delivered_items,
            delivered_gmv,
            total_freight_value,
            avg_delivered_item_price,
            ROUND(
                100.0 * delivered_items
                / NULLIF(total_items_sold, 0),
                2
            ) AS delivered_item_share_of_sold
        FROM olist.vw_category_kpis
        ORDER BY delivered_gmv DESC NULLS LAST;
    """
    return fetch_dataframe(query)


def get_category_contribution():
    """
    Calculates each category's contribution to delivered marketplace GMV.

    GMV share is based on delivered_gmv, not total product value or profit.
    """
    query = """
        SELECT
            product_category,
            total_products,
            delivered_items,
            delivered_gmv,
            ROUND(
                100.0 * delivered_gmv
                / NULLIF(SUM(delivered_gmv) OVER (), 0),
                2
            ) AS gmv_percentage,
            ROUND(
                100.0 * delivered_items
                / NULLIF(SUM(delivered_items) OVER (), 0),
                2
            ) AS delivered_item_percentage,
            avg_delivered_item_price
        FROM olist.vw_category_kpis
        ORDER BY delivered_gmv DESC;
    """
    return fetch_dataframe(query)


def get_product_price_distribution():
    """
    Groups products into price bands using each product's average
    selling price.

    This is a distribution of product-level average prices,
    not individual transaction prices.
    """
    query = """
        WITH product_price_bands AS (
            SELECT
                product_id,
                avg_selling_price,
                delivered_gmv,
                CASE
                    WHEN avg_selling_price < 50
                        THEN 'Below R$50'
                    WHEN avg_selling_price < 100
                        THEN 'R$50-R$99'
                    WHEN avg_selling_price < 250
                        THEN 'R$100-R$249'
                    WHEN avg_selling_price < 500
                        THEN 'R$250-R$499'
                    ELSE 'R$500 and above'
                END AS price_band,
                CASE
                    WHEN avg_selling_price < 50 THEN 1
                    WHEN avg_selling_price < 100 THEN 2
                    WHEN avg_selling_price < 250 THEN 3
                    WHEN avg_selling_price < 500 THEN 4
                    ELSE 5
                END AS band_order
            FROM olist.vw_product_kpis
            WHERE avg_selling_price IS NOT NULL
        )
        SELECT
            price_band,
            COUNT(*) AS total_products,
            ROUND(AVG(avg_selling_price), 2)
                AS avg_product_price,
            ROUND(SUM(delivered_gmv), 2)
                AS delivered_gmv,
            ROUND(
                100.0 * COUNT(*)
                / NULLIF(SUM(COUNT(*)) OVER (), 0),
                2
            ) AS product_percentage
        FROM product_price_bands
        GROUP BY price_band, band_order
        ORDER BY band_order;
    """
    return fetch_dataframe(query)


def get_product_performance_summary():
    """
    Returns a compact summary of product and category performance,
    including the top category and top product by delivered GMV.
    """
    query = """
        WITH product_summary AS (
            SELECT
                COUNT(*) AS total_products,
                SUM(total_items_sold) AS total_items_sold,
                SUM(delivered_items) AS delivered_items,
                SUM(delivered_gmv) AS delivered_gmv
            FROM olist.vw_product_kpis
        ),
        top_product AS (
            SELECT
                product_id,
                product_category,
                delivered_gmv
            FROM olist.vw_product_kpis
            ORDER BY delivered_gmv DESC NULLS LAST
            LIMIT 1
        ),
        top_category AS (
            SELECT
                product_category,
                delivered_gmv
            FROM olist.vw_category_kpis
            ORDER BY delivered_gmv DESC NULLS LAST
            LIMIT 1
        )
        SELECT
            ps.total_products,
            ps.total_items_sold,
            ps.delivered_items,
            ROUND(ps.delivered_gmv, 2) AS delivered_gmv,
            tp.product_id AS top_product_id,
            tp.product_category AS top_product_category,
            ROUND(tp.delivered_gmv, 2) AS top_product_gmv,
            tc.product_category AS top_category,
            ROUND(tc.delivered_gmv, 2) AS top_category_gmv
        FROM product_summary ps
        CROSS JOIN top_product tp
        CROSS JOIN top_category tc;
    """
    result = fetch_dataframe(query)
    return result.iloc[0].to_dict()
