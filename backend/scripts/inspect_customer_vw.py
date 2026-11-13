from sqlalchemy import text
from backend.app.database.connection import engine

VIEWS = [
    "vw_customer_kpis",
    "vw_customer_rfm",
    "vw_customer_segments"
]

def inspect_customer_views():
    query = text("""
        SELECT
            table_name,
            column_name,
            data_type,
            ordinal_position
        FROM information_schema.columns
        WHERE table_schema = 'olist'
          AND table_name = ANY(:view_names)
        ORDER BY table_name, ordinal_position;
    """)

    with engine.connect() as connection:
        result = connection.execute(
            query,
            {"view_names": VIEWS}
        )

        rows = result.fetchall()

    for view in VIEWS:
        print(f"\n{'=' * 65}")
        print(f"VIEW: olist.{view}")
        print("=" * 65)

        view_columns = [
            row for row in rows if row[0] == view
        ]

        if not view_columns:
            print("No columns found. Check that the view exists.")
            continue

        for row in view_columns:
            print(f"{row[1]:35} {row[2]}")


if __name__ == "__main__":
    inspect_customer_views()