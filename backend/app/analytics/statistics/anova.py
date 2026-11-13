
import numpy as np
import pandas as pd

from scipy.stats import f_oneway, kruskal
from statsmodels.stats.oneway import anova_oneway


SUPPORTED_METHODS = ["standard", "welch", "kruskal"]


def one_way_group_comparison(
    df: pd.DataFrame,
    group_column: str,
    target_column: str,
    method: str = "welch",
    alpha: float = 0.05
) -> dict:
    """
    Compare a numeric variable across multiple independent groups.

    Methods:
    - standard: Classical one-way ANOVA
    - welch: Welch's one-way ANOVA
    - kruskal: Kruskal-Wallis H test

    Returns test results, group summaries and effect sizes.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("DataFrame cannot be empty.")

    if group_column not in df.columns:
        raise ValueError(f"Missing group column: {group_column}")

    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")

    if group_column == target_column:
        raise ValueError("Group and target columns must be different.")

    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method. Choose from {SUPPORTED_METHODS}"
        )

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    data = df[[group_column, target_column]].copy()

    data[target_column] = pd.to_numeric(
        data[target_column], errors="coerce"
    )

    data[target_column] = data[target_column].replace(
        [np.inf, -np.inf], np.nan
    )

    data = data.dropna()

    if data.empty:
        raise ValueError("No valid observations remain.")

    groups = {
        group: values[target_column].to_numpy()
        for group, values in data.groupby(group_column)
    }

    if len(groups) < 2:
        raise ValueError("At least two groups are required.")

    group_names = list(groups.keys())
    group_values = list(groups.values())

    for name, values in groups.items():
        if len(values) < 2:
            raise ValueError(
                f"Group {name} requires at least 2 observations."
            )

        if np.unique(values).size < 2:
            raise ValueError(
                f"Group {name} must contain at least 2 unique values."
            )

    group_summaries = []

    for name, values in groups.items():
        group_summaries.append({
            "group": name,
            "sample_size": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "standard_deviation": float(np.std(values, ddof=1))
        })

    total_n = len(data)
    number_of_groups = len(groups)

    overall_mean = float(data[target_column].mean())

    ss_between = sum(
        len(values) * (np.mean(values) - overall_mean) ** 2
        for values in group_values
    )

    ss_total = sum(
        (value - overall_mean) ** 2
        for value in data[target_column].to_numpy()
    )

    eta_squared = (
        float(ss_between / ss_total)
        if ss_total > 0 else None
    )

    if method == "standard":
        statistic, p_value = f_oneway(*group_values)

        test_name = "One-way ANOVA"
        effect_size = eta_squared
        effect_size_name = "Eta-squared"

    elif method == "welch":
        result = anova_oneway(
            data[target_column].to_numpy(),
            groups=data[group_column].to_numpy(),
            use_var="unequal",
            welch_correction=True
        )

        statistic = result.statistic
        p_value = result.pvalue

        test_name = "Welch ANOVA"

        # Eta-squared is reported as a descriptive
        # variance-explained measure, not as the
        # Welch test's own test statistic.
        effect_size = eta_squared
        effect_size_name = "Eta-squared (descriptive)"

    else:
        statistic, p_value = kruskal(*group_values)

        test_name = "Kruskal-Wallis"

        # Epsilon-squared effect size.
        effect_size = max(
            0.0,
            float(
                (statistic - number_of_groups + 1)
                / (total_n - number_of_groups)
            )
        )

        effect_size_name = "Epsilon-squared"

    return {
        "test": test_name,
        "method": method,
        "group_column": group_column,
        "target_column": target_column,
        "number_of_groups": number_of_groups,
        "sample_size": total_n,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": alpha,
        "reject_null": bool(p_value < alpha),
        "overall_mean": overall_mean,
        "effect_size": effect_size,
        "effect_size_name": effect_size_name,
        "eta_squared": eta_squared,
        "group_summaries": group_summaries
    }
