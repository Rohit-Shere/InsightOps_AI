
import numpy as np
import pandas as pd

from scipy.stats import t


def _clean_numeric_series(data):
    """Convert input to numeric and remove missing/infinite values."""

    series = pd.Series(data)
    series = pd.to_numeric(series, errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan).dropna()

    return series


def calculate_mean_confidence_interval(
    data,
    confidence_level: float = 0.95
) -> dict:
    """
    Calculate a t-based confidence interval for a population mean.
    """

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    series = _clean_numeric_series(data)

    n = len(series)

    if n < 2:
        raise ValueError("At least 2 valid observations are required.")

    mean = float(series.mean())
    std = float(series.std(ddof=1))
    standard_error = std / np.sqrt(n)

    degrees_of_freedom = n - 1
    alpha = 1 - confidence_level

    critical_value = t.ppf(
        1 - alpha / 2,
        degrees_of_freedom
    )

    margin_of_error = critical_value * standard_error

    return {
        "sample_size": n,
        "mean": mean,
        "standard_deviation": std,
        "standard_error": float(standard_error),
        "confidence_level": confidence_level,
        "lower_bound": float(mean - margin_of_error),
        "upper_bound": float(mean + margin_of_error),
        "margin_of_error": float(margin_of_error)
    }


def calculate_difference_confidence_interval(
    df: pd.DataFrame,
    group_column: str,
    target_column: str,
    group1_value,
    group2_value,
    confidence_level: float = 0.95
) -> dict:
    """
    Calculate a Welch confidence interval for the difference
    between two independent group means.

    Difference = mean(group1) - mean(group2)
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("DataFrame cannot be empty.")

    required_columns = [group_column, target_column]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if group1_value == group2_value:
        raise ValueError("Group values must be different.")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    data = df[required_columns].copy()

    data[target_column] = pd.to_numeric(
        data[target_column], errors="coerce"
    )

    data[target_column] = data[target_column].replace(
        [np.inf, -np.inf], np.nan
    )

    group1 = data.loc[
        data[group_column] == group1_value,
        target_column
    ].dropna()

    group2 = data.loc[
        data[group_column] == group2_value,
        target_column
    ].dropna()

    n1 = len(group1)
    n2 = len(group2)

    if n1 < 2 or n2 < 2:
        raise ValueError("Each group requires at least 2 observations.")

    mean1 = float(group1.mean())
    mean2 = float(group2.mean())

    var1 = float(group1.var(ddof=1))
    var2 = float(group2.var(ddof=1))

    difference = mean1 - mean2

    variance_component1 = var1 / n1
    variance_component2 = var2 / n2

    standard_error = np.sqrt(
        variance_component1 + variance_component2
    )

    if standard_error == 0:
        raise ValueError(
            "Cannot calculate interval when both groups have zero variance."
        )

    numerator = (
        variance_component1 + variance_component2
    ) ** 2

    denominator = (
        variance_component1 ** 2 / (n1 - 1)
        + variance_component2 ** 2 / (n2 - 1)
    )

    degrees_of_freedom = numerator / denominator

    alpha = 1 - confidence_level

    critical_value = t.ppf(
        1 - alpha / 2,
        degrees_of_freedom
    )

    margin_of_error = critical_value * standard_error

    return {
        "group1": group1_value,
        "group2": group2_value,
        "n_group1": n1,
        "n_group2": n2,
        "mean_group1": mean1,
        "mean_group2": mean2,
        "mean_difference": float(difference),
        "standard_error": float(standard_error),
        "degrees_of_freedom": float(degrees_of_freedom),
        "confidence_level": confidence_level,
        "lower_bound": float(difference - margin_of_error),
        "upper_bound": float(difference + margin_of_error)
    }
