
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr, kendalltau


SUPPORTED_METHODS = {
    "pearson": pearsonr,
    "spearman": spearmanr,
    "kendall": kendalltau,
}


def calculate_pairwise_correlation(
    df: pd.DataFrame,
    column1: str,
    column2: str,
    method: str = "pearson",
    alpha: float = 0.05
) -> dict:
    """
    Calculate correlation between two numeric variables.

    Supports Pearson, Spearman and Kendall correlation.
    Missing and infinite values are excluded pairwise.
    """

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
        raise ValueError("Select two different columns.")

    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method. Choose from: {list(SUPPORTED_METHODS)}"
        )

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    data = df[[column1, column2]].copy()

    for column in [column1, column2]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    x = data[column1].to_numpy()
    y = data[column2].to_numpy()

    n = len(data)

    if n < 3:
        raise ValueError("At least 3 valid paired observations are required.")

    if np.unique(x).size < 2 or np.unique(y).size < 2:
        raise ValueError("Both variables must contain at least 2 unique values.")

    correlation, p_value = SUPPORTED_METHODS[method](x, y)

    return {
        "variable_1": column1,
        "variable_2": column2,
        "method": method,
        "sample_size": n,
        "correlation": float(correlation),
        "p_value": float(p_value),
        "alpha": alpha,
        "statistically_significant": bool(p_value < alpha)
    }


def get_correlation_matrix(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "pearson"
) -> pd.DataFrame:
    """
    Generate a correlation matrix for selected numeric columns.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("DataFrame cannot be empty.")

    if method not in ["pearson", "spearman", "kendall"]:
        raise ValueError("Unsupported correlation method.")

    if columns is None:
        numeric_df = df.select_dtypes(include="number").copy()
    else:
        missing_columns = [
            col for col in columns if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")

        numeric_df = df[columns].copy()

        for column in columns:
            numeric_df[column] = pd.to_numeric(
                numeric_df[column], errors="coerce"
            )

    numeric_df = numeric_df.replace(
        [np.inf, -np.inf], np.nan
    )

    if numeric_df.shape[1] < 2:
        raise ValueError("At least two numeric columns are required.")

    return numeric_df.corr(method=method)
