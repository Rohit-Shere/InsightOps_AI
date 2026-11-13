
import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu, ttest_ind


def compare_independent_groups(
    df: pd.DataFrame,
    group_column: str,
    target_column: str,
    group1_value,
    group2_value,
    method: str = "mannwhitney",
    alternative: str = "two-sided",
    alpha: float = 0.05
) -> dict:
    """
    Compare a numeric outcome between two independent groups.

    Supported methods:
        - mannwhitney
        - welch_t

    Effect sizes:
        - Mann-Whitney: Cliff's delta
        - Welch t-test: Cohen's d (pooled SD)

    Positive effect size means group 1 tends to
    have higher values than group 2.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    required = [group_column, target_column]

    missing_columns = [
        col for col in required
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if group1_value == group2_value:
        raise ValueError("Groups must be different.")

    if method not in ["mannwhitney", "welch_t"]:
        raise ValueError("Unsupported statistical test.")

    if alternative not in [
        "two-sided", "less", "greater"
    ]:
        raise ValueError("Invalid alternative hypothesis.")

    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")

    data = df[[group_column, target_column]].copy()

    data[target_column] = pd.to_numeric(
        data[target_column],
        errors="coerce"
    )

    data[target_column] = data[target_column].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Keep only the two requested groups
    data = data[
        data[group_column].isin(
            [group1_value, group2_value]
        )
    ]

    # Remove missing values
    data = data.dropna(
        subset=[group_column, target_column]
    )

    group1 = data.loc[
        data[group_column] == group1_value,
        target_column
    ].to_numpy()

    group2 = data.loc[
        data[group_column] == group2_value,
        target_column
    ].to_numpy()

    n1 = len(group1)
    n2 = len(group2)

    if n1 < 2 or n2 < 2:
        raise ValueError(
            "Each group must contain at least two observations."
        )

    if np.unique(group1).size < 2 or np.unique(group2).size < 2:
        raise ValueError(
            "Both groups must contain variation for this analysis."
        )

    if method == "mannwhitney":

        test_result = mannwhitneyu(
            group1,
            group2,
            alternative=alternative,
            method="auto"
        )

        statistic = float(test_result.statistic)
        p_value = float(test_result.pvalue)

        # Cliff's delta
        effect_size = (
            (2 * statistic) / (n1 * n2)
        ) - 1

        effect_name = "Cliff's delta"

    else:

        test_result = ttest_ind(
            group1,
            group2,
            equal_var=False,
            alternative=alternative
        )

        statistic = float(test_result.statistic)
        p_value = float(test_result.pvalue)

        pooled_std = np.sqrt(
            (
                (n1 - 1) * np.var(group1, ddof=1)
                + (n2 - 1) * np.var(group2, ddof=1)
            ) / (n1 + n2 - 2)
        )

        effect_size = (
            np.mean(group1) - np.mean(group2)
        ) / pooled_std

        effect_name = "Cohen's d (pooled SD)"

    return {
        "test": method,
        "alternative": alternative,
        "group1": str(group1_value),
        "group2": str(group2_value),
        "n_group1": n1,
        "n_group2": n2,
        "group1_mean": float(np.mean(group1)),
        "group2_mean": float(np.mean(group2)),
        "group1_median": float(np.median(group1)),
        "group2_median": float(np.median(group2)),
        "test_statistic": statistic,
        "p_value": p_value,
        "alpha": alpha,
        "reject_null": bool(p_value < alpha),
        "effect_size": float(effect_size),
        "effect_size_name": effect_name
    }
