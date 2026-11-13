
import numpy as np
import pandas as pd

from itertools import combinations

from scipy.stats import (
    rankdata,
    norm,
    studentized_range
)

from statsmodels.stats.multicomp import pairwise_tukeyhsd

from app.analytics.statistics.preparation import validate_dataframe


def _prepare_groups(df, group_column, target_column):
    """Clean data and prepare observations by group."""

    validate_dataframe(df, [group_column, target_column])

    data = df[[group_column, target_column]].copy()

    data[target_column] = pd.to_numeric(
        data[target_column], errors="coerce"
    )

    data[target_column] = data[target_column].replace(
        [np.inf, -np.inf], np.nan
    )

    data = data.dropna()

    groups = {
        group: values[target_column].to_numpy(dtype=float)
        for group, values in data.groupby(group_column)
    }

    if len(groups) < 2:
        raise ValueError("At least two groups are required.")

    for group, values in groups.items():
        if len(values) < 2:
            raise ValueError(
                f"Group '{group}' must contain at least two observations."
            )

    return data, groups


def _holm_adjust(p_values):
    """Apply Holm-Bonferroni adjustment to p-values."""

    p_values = np.asarray(p_values, dtype=float)

    m = len(p_values)
    order = np.argsort(p_values)

    adjusted = np.empty(m, dtype=float)
    running_max = 0.0

    for rank, index in enumerate(order):
        value = (m - rank) * p_values[index]
        running_max = max(running_max, value)
        adjusted[index] = min(running_max, 1.0)

    return adjusted


def tukey_hsd(
    df,
    group_column,
    target_column,
    alpha=0.05
):
    """Pairwise comparisons using Tukey HSD."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    data, groups = _prepare_groups(
        df, group_column, target_column
    )

    result = pairwise_tukeyhsd(
        endog=data[target_column].to_numpy(dtype=float),
        groups=data[group_column].to_numpy(),
        alpha=alpha
    )

    table = pd.DataFrame(
        result._results_table.data[1:],
        columns=result._results_table.data[0]
    )

    table = table.rename(columns={
        "group1": "group_1",
        "group2": "group_2",
        "meandiff": "mean_difference",
        "p-adj": "adjusted_p_value",
        "lower": "ci_lower",
        "upper": "ci_upper",
        "reject": "reject_null"
    })
    table["mean_difference"] *= -1

    old_lower = table["ci_lower"].copy()

    table["ci_lower"] = -table["ci_upper"]
    table["ci_upper"] = -old_lower

    table["method"] = "Tukey HSD"

    return table


def games_howell(
    df,
    group_column,
    target_column,
    alpha=0.05
):
    """
    Games-Howell pairwise comparisons.

    Does not assume equal group variances or equal sample sizes.
    Uses the studentized range distribution.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    _, groups = _prepare_groups(
        df, group_column, target_column
    )

    k = len(groups)
    results = []

    for group_1, group_2 in combinations(groups.keys(), 2):

        x = groups[group_1]
        y = groups[group_2]

        n1, n2 = len(x), len(y)
        mean1, mean2 = np.mean(x), np.mean(y)

        var1 = np.var(x, ddof=1)
        var2 = np.var(y, ddof=1)

        variance_sum = var1 / n1 + var2 / n2

        if variance_sum <= 0:
            raise ValueError(
                "Games-Howell cannot compare groups with zero "
                "combined sampling variance."
            )

        mean_difference = mean1 - mean2

        standard_error = np.sqrt(variance_sum)

        degrees_of_freedom = (
            variance_sum ** 2
            / (
                (var1 / n1) ** 2 / (n1 - 1)
                + (var2 / n2) ** 2 / (n2 - 1)
            )
        )

        q_statistic = (
            abs(mean_difference)
            / np.sqrt(variance_sum / 2)
        )

        p_value = studentized_range.sf(
            q_statistic,
            k,
            degrees_of_freedom
        )

        critical_q = studentized_range.ppf(
            1 - alpha,
            k,
            degrees_of_freedom
        )

        margin = critical_q * np.sqrt(variance_sum / 2)

        results.append({
            "group_1": group_1,
            "group_2": group_2,
            "mean_difference": mean_difference,
            "standard_error": standard_error,
            "degrees_of_freedom": degrees_of_freedom,
            "q_statistic": q_statistic,
            "p_value": p_value,
            "ci_lower": mean_difference - margin,
            "ci_upper": mean_difference + margin,
            "reject_null": bool(p_value < alpha),
            "method": "Games-Howell"
        })

    return pd.DataFrame(results)


def dunn_test(
    df,
    group_column,
    target_column,
    alpha=0.05
):
    """
    Dunn's pairwise rank test with Holm correction.

    Suitable as a post-hoc procedure following Kruskal-Wallis.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    data, groups = _prepare_groups(
        df, group_column, target_column
    )

    values = data[target_column].to_numpy(dtype=float)
    labels = data[group_column].to_numpy()

    ranks = rankdata(values, method="average")

    data = data.copy()
    data["_rank"] = ranks

    total_n = len(data)

    rank_sums = data.groupby(group_column)["_rank"].sum()
    group_sizes = data.groupby(group_column).size()
    mean_ranks = rank_sums / group_sizes

    # Tie correction for tied observations.
    _, tie_counts = np.unique(values, return_counts=True)

    tie_sum = np.sum(tie_counts ** 3 - tie_counts)

    tie_correction = (
        1 - tie_sum / (total_n ** 3 - total_n)
    )

    if tie_correction <= 0:
        raise ValueError(
            "All observations are tied; Dunn's test cannot be computed."
        )

    results = []

    for group_1, group_2 in combinations(groups.keys(), 2):

        n1 = group_sizes[group_1]
        n2 = group_sizes[group_2]

        rank_difference = (
            mean_ranks[group_1] - mean_ranks[group_2]
        )

        variance = (
            total_n * (total_n + 1) / 12
            * tie_correction
            * (1 / n1 + 1 / n2)
        )

        z_statistic = rank_difference / np.sqrt(variance)

        p_value = 2 * norm.sf(abs(z_statistic))

        results.append({
            "group_1": group_1,
            "group_2": group_2,
            "mean_rank_difference": rank_difference,
            "z_statistic": z_statistic,
            "p_value": p_value
        })

    result_df = pd.DataFrame(results)

    result_df["adjusted_p_value"] = _holm_adjust(
        result_df["p_value"].to_numpy()
    )

    result_df["reject_null"] = (
        result_df["adjusted_p_value"] < alpha
    )

    result_df["method"] = "Dunn with Holm correction"

    return result_df


def run_posthoc_test(
    df,
    group_column,
    target_column,
    method="dunn",
    alpha=0.05
):
    """Generic post-hoc test dispatcher."""

    methods = {
        "tukey": tukey_hsd,
        "games_howell": games_howell,
        "dunn": dunn_test
    }

    if method not in methods:
        raise ValueError(
            f"Unsupported method: {method}. "
            f"Choose from {list(methods.keys())}."
        )

    return methods[method](
        df=df,
        group_column=group_column,
        target_column=target_column,
        alpha=alpha
    )
