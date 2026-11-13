
from pathlib import Path
import pandas as pd

# Project root is two levels above this script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

FILES = [
    "olist_orders_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "product_category_name_translation.csv",
]


def inspect_file(file_path: Path):
    print("\n" + "=" * 65)
    print(f"FILE: {file_path.name}")
    print("=" * 65)

    df = pd.read_csv(file_path)

    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn types:")
    print(df.dtypes)

    print("\nMissing values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(missing if not missing.empty else "No missing values")

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nSample records:")
    print(df.head(3).to_string(index=False))

    return df


def main():
    print("InsightOps AI - Olist Data Inspection")

    missing_files = [
        name for name in FILES
        if not (DATA_DIR / name).exists()
    ]

    if missing_files:
        print("\nMissing files:")
        for name in missing_files:
            print(f"- {name}")
        raise FileNotFoundError(
            "Download and extract all Olist CSV files first."
        )

    for name in FILES:
        inspect_file(DATA_DIR / name)


if __name__ == "__main__":
    main()
