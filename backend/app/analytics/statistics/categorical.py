
import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency


def _prepare_contingency_table(
    df: pd.DataFrame,
    column1: str,
    column2: str
) -> pd.DataFrame:
    """Build a contingency table after removing missing pairs."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("DataFrame cannot be empty.")

    missing_columns = [
        col for col in [column1, column2]
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if column1 == column2:
        raise ValueError("Select two different categorical columns.")

    data = df[[column1, column2]].dropna()

    if data.empty:
        raise ValueError("No valid observations remain after removing missing values.")

    table = pd.crosstab(data[column1], data[column2])

    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError(
            "Both categorical variables must have at least 2 categories."
        )

    return table


def chi_square_test(
    df: pd.DataFrame,
    column1: str,
    column2: str,
    alpha: float = 0.05,
    correction: bool = True
) -> dict:
    """
    Perform a Chi-square test of independence.

    H0: The two categorical variables are independent.
    H1: The two categorical variables are associated.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    table = _prepare_contingency_table(
        df, column1, column2
    )

    chi2, p_value, degrees_of_freedom, expected = (
        chi2_contingency(
            table,
            correction=correction
        )
    )

    expected_df = pd.DataFrame(
        expected,
        index=table.index,
        columns=table.columns
    )

    return {
        "variable_1": column1,
        "variable_2": column2,
        "sample_size": int(table.to_numpy().sum()),
        "chi_square_statistic": float(chi2),
        "degrees_of_freedom": int(degrees_of_freedom),
        "p_value": float(p_value),
        "alpha": alpha,
        "reject_null": bool(p_value < alpha),
        "minimum_expected_frequency": float(expected.min()),
        "cells_with_expected_frequency_below_5": int(
            (expected < 5).sum()
        ),
        "contingency_table": table,
        "chi_square_assumptions_satisfied": bool(
    (expected >= 5).all()
        ),
        "assumption_warning": (
            "Some expected frequencies are below 5. "
            "Consider category consolidation or an exact/Monte Carlo test."
            if (expected < 5).any()
            else None
        ),
        "expected_frequencies": expected_df
    }


def calculate_cramers_v(
    contingency_table
) -> float:
    """
    Calculate Cramer's V, a measure of association strength
    between two categorical variables.

    Range: 0 to 1.
    """

    table = np.asarray(contingency_table, dtype=float)

    if table.ndim != 2 or min(table.shape) < 2:
        raise ValueError("Contingency table must have at least 2 rows and columns.")

    if not np.isfinite(table).all():
        raise ValueError("Contingency table must contain finite values.")

    if (table < 0).any():
        raise ValueError("Contingency table cannot contain negative values.")

    n = table.sum()

    if n <= 0:
        raise ValueError("Contingency table must have a positive total.")

    chi2, _, _, _ = chi2_contingency(
        table,
        correction=False
    )

    denominator = n * min(table.shape[0] - 1, table.shape[1] - 1)

    return float(np.sqrt(chi2 / denominator))


def analyze_categorical_relationship(
    df: pd.DataFrame,
    column1: str,
    column2: str,
    alpha: float = 0.05,
    correction: bool = True
) -> dict:
    """
    Combine Chi-square testing and Cramer's V into one analysis.
    """

    result = chi_square_test(
        df=df,
        column1=column1,
        column2=column2,
        alpha=alpha,
        correction=correction
    )

    table = result["contingency_table"]

    result["cramers_v"] = calculate_cramers_v(table.to_numpy())

    return result
