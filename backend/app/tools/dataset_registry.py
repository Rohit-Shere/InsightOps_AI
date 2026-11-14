
from app.database.query_executor import fetch_dataframe


DATASET_REGISTRY = {
    "review_delivery": {
        "view": "olist.vw_review_analytics",
        "grain": "One row per review record",
        "columns": [
            "review_id", "order_id", "review_score",
            "review_comment_title", "review_comment_message",
            "review_creation_date", "review_answer_timestamp",
            "order_status", "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
            "customer_unique_id", "delivery_days",
            "is_late_delivery",
        ],
        "numeric_columns": ["review_score", "delivery_days"],
        "categorical_columns": ["order_status"],
        "boolean_columns": ["is_late_delivery"],
        "date_columns": [
            "review_creation_date", "review_answer_timestamp",
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
        "identifier_columns": [
            "review_id", "order_id", "customer_unique_id",
        ],
        "caveats": [
            "Review IDs are not unique; investigate duplicate records.",
            "Multiple review records may relate to the same order.",
            "Review scores are ordinal ratings.",
        ],
    },

    "order_performance": {
        "view": "olist.vw_order_analytics",
        "grain": "One row per order",
        "columns": [
            "order_id", "order_status", "order_purchase_timestamp",
            "order_approved_at", "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date", "customer_id",
            "customer_unique_id", "customer_city", "customer_state",
            "total_items", "unique_products", "unique_sellers",
            "unique_categories", "product_categories",
            "product_value", "freight_value", "total_item_value",
            "total_payment_value", "payment_records",
            "zero_value_payments", "payment_methods",
            "total_reviews", "average_review_score",
            "delivery_days", "is_late_delivery",
        ],
        "numeric_columns": [
            "total_items", "unique_products", "unique_sellers",
            "unique_categories", "product_value", "freight_value",
            "total_item_value", "total_payment_value",
            "payment_records", "zero_value_payments",
            "total_reviews", "average_review_score", "delivery_days",
        ],
        "categorical_columns": [
            "order_status", "customer_city", "customer_state",
            "product_categories", "payment_methods",
        ],
        "boolean_columns": ["is_late_delivery"],
        "date_columns": [
            "order_purchase_timestamp", "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
        "identifier_columns": [
            "order_id", "customer_id", "customer_unique_id",
        ],
        "caveats": [
            "One row represents one order.",
            "Missing delivery dates must be handled before delivery analyses.",
            "Product categories and payment methods can contain multiple values.",
        ],
    },

    "customer_rfm": {
        "view": "olist.vw_customer_rfm",
        "grain": "One row per delivered customer",
        "columns": [
            "customer_unique_id", "last_purchase_date",
            "reference_date", "recency", "frequency", "monetary",
            "r_score", "f_score", "m_score", "rfm_segment",
            "rfm_total_score",
        ],
        "numeric_columns": [
            "recency", "frequency", "monetary",
            "r_score", "f_score", "m_score", "rfm_total_score",
        ],
        "categorical_columns": ["rfm_segment"],
        "boolean_columns": [],
        "date_columns": ["last_purchase_date", "reference_date"],
        "identifier_columns": ["customer_unique_id"],
        "caveats": [
            "Contains customers with delivered orders.",
            "RFM scores are derived from the underlying recency, frequency and monetary values.",
            "RFM scores are ordinal and should not automatically be treated as continuous measurements.",
        ],
    },

    "product_performance": {
        "view": "olist.vw_product_kpis",
        "grain": "One row per product",
        "columns": [
            "product_id", "product_category", "total_items_sold",
            "total_orders", "unique_customers", "unique_sellers",
            "delivered_items", "delivered_orders",
            "total_product_value", "delivered_gmv",
            "total_freight_value", "avg_selling_price",
            "avg_freight_per_item", "canceled_items",
        ],
        "numeric_columns": [
            "total_items_sold", "total_orders", "unique_customers",
            "unique_sellers", "delivered_items", "delivered_orders",
            "total_product_value", "delivered_gmv",
            "total_freight_value", "avg_selling_price",
            "avg_freight_per_item", "canceled_items",
        ],
        "categorical_columns": ["product_category"],
        "boolean_columns": [],
        "date_columns": [],
        "identifier_columns": ["product_id"],
        "caveats": [
            "Product-level order and customer counts overlap across products.",
            "Delivered GMV is not profit.",
            "No product cost data is available.",
        ],
    },

    "seller_performance": {
        "view": "olist.vw_seller_kpis",
        "grain": "One row per seller",
        "columns": [
            "seller_id", "total_items_sold", "total_orders",
            "delivered_items", "delivered_orders",
            "total_product_value", "delivered_gmv",
            "avg_selling_price", "avg_freight_value",
            "unique_customers", "eligible_delivered_orders",
            "late_orders",
        ],
        "numeric_columns": [
            "total_items_sold", "total_orders", "delivered_items",
            "delivered_orders", "total_product_value",
            "delivered_gmv", "avg_selling_price",
            "avg_freight_value", "unique_customers",
            "eligible_delivered_orders", "late_orders",
        ],
        "categorical_columns": [],
        "boolean_columns": [],
        "date_columns": [],
        "identifier_columns": ["seller_id"],
        "caveats": [
            "Seller-level customer counts overlap across sellers.",
            "Seller delivery rates can be unstable for low-volume sellers.",
            "A late order association does not prove the seller caused the delay.",
            "NULL delivered GMV values must not be interpreted as zero.",
        ],
    },

    "payment_performance": {
        "view": "olist.vw_payment_analytics",
        "grain": "One row per payment record",
        "columns": [
            "order_id", "payment_sequential", "payment_type",
            "payment_installments", "payment_value", "order_status",
            "order_purchase_timestamp", "purchase_month",
            "customer_unique_id", "is_delivered",
        ],
        "numeric_columns": [
            "payment_sequential", "payment_installments",
            "payment_value",
        ],
        "categorical_columns": ["payment_type", "order_status"],
        "boolean_columns": ["is_delivered"],
        "date_columns": ["order_purchase_timestamp", "purchase_month"],
        "identifier_columns": ["order_id", "customer_unique_id"],
        "caveats": [
            "One order can have multiple payment records.",
            "Payment records are not independent orders.",
            "Payment value is not the same as delivered GMV or profit.",
        ],
    },
}


def get_available_statistical_datasets():
    """Return metadata for all approved statistical datasets."""

    datasets = []

    for name, metadata in DATASET_REGISTRY.items():
        datasets.append({
            "dataset": name,
            "source_view": metadata["view"],
            "grain": metadata["grain"],
            "row_level_caveats": metadata["caveats"],
        })

    return datasets


def get_dataset_metadata(dataset_name: str):
    """Retrieve metadata for one approved dataset."""

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available datasets: {list(DATASET_REGISTRY)}"
        )

    return DATASET_REGISTRY[dataset_name]


def get_dataset_columns(dataset_name: str):
    """Return approved columns and their analytical roles."""

    metadata = get_dataset_metadata(dataset_name)

    return {
        "dataset": dataset_name,
        "grain": metadata["grain"],
        "numeric_columns": metadata["numeric_columns"],
        "categorical_columns": metadata["categorical_columns"],
        "boolean_columns": metadata["boolean_columns"],
        "date_columns": metadata["date_columns"],
        "identifier_columns": metadata["identifier_columns"],
        "caveats": metadata["caveats"],
    }


def load_statistical_dataset(dataset_name: str):
    """
    Load an approved statistical dataset.

    SQL identifiers are taken only from the fixed registry.
    No user- or agent-generated SQL is accepted.
    """

    metadata = get_dataset_metadata(dataset_name)

    view_name = metadata["view"]
    columns = metadata["columns"]

    # Both view and column names come from our fixed registry.
    column_sql = ", ".join(columns)

    query = f"""
        SELECT {column_sql}
        FROM {view_name}
    """

    df = fetch_dataframe(query)

    if df.empty:
        raise ValueError(
            f"Dataset '{dataset_name}' returned no records."
        )

    missing_columns = set(columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset schema mismatch. Missing columns: "
            f"{sorted(missing_columns)}"
        )

    return df
