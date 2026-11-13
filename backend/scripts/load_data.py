
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATABASE_URL = (
    "postgresql+psycopg://insightops_user:"
    "change_this_password@localhost:5432/insightops"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Load parent tables before dependent tables.
TABLES = [
    ("olist_customers_dataset.csv", "customers"),
    ("olist_sellers_dataset.csv", "sellers"),
    ("olist_products_dataset.csv", "products"),
    ("product_category_name_translation.csv", "category_translation"),
    ("olist_orders_dataset.csv", "orders"),
    ("olist_order_items_dataset.csv", "order_items"),
    ("olist_order_payments_dataset.csv", "order_payments"),
    ("olist_order_reviews_dataset.csv", "order_reviews"),
    ("olist_geolocation_dataset.csv", "geolocation"),
]

DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": ["shipping_limit_date"],
    "order_reviews": [
        "review_creation_date",
        "review_answer_timestamp",
    ],
}


def load_table(filename, table_name):
    path = DATA_DIR / filename
    print(f"\nLoading {table_name}...")

    df = pd.read_csv(path)

    for column in DATE_COLUMNS.get(table_name, []):
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce"
        )

    df.to_sql(
        name=table_name,
        con=engine,
        schema="olist",
        if_exists="append",
        index=False,
        chunksize=10_000,
        method="multi"
    )

    print(f"Inserted {len(df):,} rows into {table_name}")


def main():
    with engine.connect() as connection:
        for _, table_name in TABLES:
            count = connection.execute(
                text(f"SELECT COUNT(*) FROM olist.{table_name}")
            ).scalar_one()

            if count > 0:
                raise RuntimeError(
                    f"Table {table_name} is not empty. "
                    "Refusing to append duplicate data."
                )

    for filename, table_name in TABLES:
        load_table(filename, table_name)

    print("\nData ingestion completed.")


if __name__ == "__main__":
    main()
