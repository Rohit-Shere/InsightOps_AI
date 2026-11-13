
import numpy as np
import pandas as pd


def get_descriptive_statistics(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    percentiles: tuple = (0.25, 0.50, 0.75)
) -> pd.DataFrame:
    """
    Calculate descriptive statistics for numeric columns.

    Returns one row per numeric variable.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if not all(0 < p < 1 for p in percentiles):
        raise ValueError("Percentiles must be between 0 and 1.")

    if len(set(percentiles)) != len(percentiles):
        raise ValueError("Percentiles must be unique.")

    if columns is None:
        columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

    if not columns:
        raise ValueError("No numeric columns selected.")

    missing_columns = [
        col for col in columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Columns not found: {missing_columns}"
        )

    non_numeric = [
        col for col in columns
        if not pd.api.types.is_numeric_dtype(df[col])
        or pd.api.types.is_bool_dtype(df[col])
    ]

    if non_numeric:
        raise TypeError(
            f"Non-numeric columns selected: {non_numeric}"
        )

    results = []

    for column in columns:

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        valid = series.dropna()

        row = {
            "variable": column,
            "count": int(valid.count()),
            "missing": int(series.isna().sum()),
            "mean": valid.mean(),
            "median": valid.median(),
            "std": valid.std(),
            "variance": valid.var(),
            "min": valid.min(),
            "max": valid.max(),
            "range": valid.max() - valid.min(),
            "skewness": valid.skew(),
            "kurtosis": valid.kurt()
        }

        quantiles = valid.quantile(percentiles)

        for percentile, value in quantiles.items():
            row[f"p{int(percentile * 100)}"] = value

        if 0.25 in percentiles and 0.75 in percentiles:
            row["iqr"] = (
                quantiles.loc[0.75]
                - quantiles.loc[0.25]
            )

        results.append(row)

    return pd.DataFrame(results)
