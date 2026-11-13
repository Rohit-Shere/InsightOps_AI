
import pandas as pd
import numpy as np


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str]
) -> dict:

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return {
        "is_valid": True,
        "row_count": len(df),
        "column_count": len(df.columns),
        "required_columns": required_columns
    }


def clean_numeric_data(
    df: pd.DataFrame,
    columns: list[str]
) -> pd.DataFrame:

    result = df.copy()

    for column in columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        )

        result[column] = result[column].replace(
            [np.inf, -np.inf],
            np.nan
        )

    return result


def get_data_quality_report(
    df: pd.DataFrame
) -> pd.DataFrame:

    report = pd.DataFrame({
        "column": df.columns,
        "data_type": df.dtypes.astype(str).values,
        "missing_count": df.isna().sum().values,
        "missing_percentage": (
            df.isna().mean().values * 100
        ).round(2),
        "unique_values": df.nunique().values
    })

    return report


def prepare_order_level_data(
    df: pd.DataFrame,
    order_column: str = "order_id",
    score_column: str = "review_score",
    delivery_group_column: str = "is_late_delivery",
    duration_column: str = "delivery_days"
) -> tuple[pd.DataFrame, dict]:

    required_columns = [
        order_column,
        score_column,
        delivery_group_column,
        duration_column
    ]

    validate_dataframe(df, required_columns)

    data = df[required_columns].copy()

    # Convert numeric columns safely
    data = clean_numeric_data(
        data,
        [score_column, duration_column]
    )

    # Remove rows without an order identifier
    data = data.dropna(subset=[order_column])

    # Validate review scores
    data.loc[
        ~data[score_column].between(1, 5),
        score_column
    ] = float("nan")

    # Aggregate reviews at order level
    grouped = data.groupby(order_column).agg(
        average_review_score=(score_column, "mean"),
        review_count=(score_column, "count"),
        delivery_group_count=(delivery_group_column, "nunique"),
        delivery_classification=(delivery_group_column, "first"),
        delivery_days_count=(duration_column, "nunique"),
        delivery_days=(duration_column, "first")
    ).reset_index()

    # Identify conflicting delivery classifications
    conflicting_groups = (
        grouped["delivery_group_count"] > 1
    )

    # Exclude missing or conflicting delivery classifications
    grouped.loc[
        conflicting_groups,
        "delivery_classification"
    ] = None

    grouped["delivery_classification"] = (
        grouped["delivery_classification"]
        .astype("boolean")
    )

    # Data-quality summary
    report = {
        "original_records": len(df),
        "unique_orders": data[order_column].nunique(),
        "prepared_orders": len(grouped),
        "orders_with_multiple_reviews": int(
            (grouped["review_count"] > 1).sum()
        ),
        "orders_with_missing_scores": int(
            grouped["average_review_score"].isna().sum()
        ),
        "orders_with_missing_delivery_classification": int(
            grouped["delivery_classification"].isna().sum()
        ),
        "orders_with_conflicting_delivery_classification": int(
            conflicting_groups.sum()
        ),
        "orders_with_conflicting_delivery_duration": int(
            (grouped["delivery_days_count"] > 1).sum()
        )
    }

    return grouped, report
