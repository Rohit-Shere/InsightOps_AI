
import numpy as np
import pandas as pd

from scipy.stats import levene, normaltest, skew, kurtosis

from app.analytics.statistics.preparation import validate_dataframe


def _get_groups(df, group_column, target_column):
    """Prepare valid numeric observations for each group."""

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


def check_group_variance(
    df,
    group_column,
    target_column,
    alpha=0.05
):
    """
    Test whether group variances are equal.

    Uses Levene's test with median centering
    (also known as the Brown-Forsythe version).
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    _, groups = _get_groups(
        df, group_column, target_column
    )

    statistic, p_value = levene(
        *groups.values(),
        center="median"
    )

    return {
        "test": "Brown-Forsythe (Levene with median center)",
        "group_column": group_column,
        "target_column": target_column,
        "number_of_groups": len(groups),
        "sample_size": sum(len(x) for x in groups.values()),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": alpha,
        "equal_variances": bool(p_value >= alpha),
        "interpretation": (
            "No statistically significant evidence of unequal variances."
            if p_value >= alpha
            else "Evidence suggests that group variances are unequal."
        )
    }


def check_residual_normality(
    df,
    group_column,
    target_column,
    alpha=0.05
):
    """
    Assess normality of residuals from a one-way group-means model.

    Residual = observation - its group's mean.
    Uses D'Agostino-Pearson's normality test.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    data, groups = _get_groups(
        df, group_column, target_column
    )

    group_means = data.groupby(group_column)[
        target_column
    ].transform("mean")

    residuals = (
        data[target_column] - group_means
    ).to_numpy(dtype=float)

    if len(residuals) < 8:
        raise ValueError(
            "At least 8 residual observations are required."
        )

    statistic, p_value = normaltest(residuals)

    return {
        "test": "D'Agostino-Pearson normality test",
        "number_of_groups": len(groups),
        "sample_size": len(residuals),
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals, ddof=1)),
        "residual_skewness": float(skew(residuals)),
        "residual_kurtosis": float(kurtosis(residuals)),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": alpha,
        "normality_not_rejected": bool(p_value >= alpha),
        "interpretation": (
            "Residual normality was not rejected."
            if p_value >= alpha
            else "Residuals show evidence of deviation from normality."
        ),
        "warning": (
            "With large samples, normality tests can detect very small "
            "departures. Interpret alongside residual plots and robustness."
        )
    }


def check_anova_assumptions(
    df,
    group_column,
    target_column,
    alpha=0.05
):
    """Run the main numerical diagnostics for one-way ANOVA."""

    variance_result = check_group_variance(
        df, group_column, target_column, alpha
    )

    normality_result = check_residual_normality(
        df, group_column, target_column, alpha
    )

    return {
        "variance_test": variance_result,
        "residual_normality_test": normality_result,
        "standard_anova_variance_assumption": (
            variance_result["equal_variances"]
        ),
        "residual_normality_not_rejected": (
            normality_result["normality_not_rejected"]
        ),
        "note": (
            "These diagnostics do not establish independence. "
            "Independence must be assessed from the study design."
        )
    }



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import probplot

from app.analytics.statistics.preparation import validate_dataframe


def plot_anova_diagnostics(
    df,
    group_column,
    target_column
):
    """
    Generate four diagnostic plots for a one-way group-means model.

    Returns:
        fig, axes
    """

    validate_dataframe(df, [group_column, target_column])

    data = df[[group_column, target_column]].copy()

    data[target_column] = pd.to_numeric(
        data[target_column], errors="coerce"
    )

    data[target_column] = data[target_column].replace(
        [np.inf, -np.inf], np.nan
    )

    data = data.dropna()

    if data[group_column].nunique() < 2:
        raise ValueError("At least two groups are required.")

    # Group means act as fitted values in a one-way ANOVA.
    data["_fitted"] = data.groupby(group_column)[
        target_column
    ].transform("mean")

    data["_residual"] = (
        data[target_column] - data["_fitted"]
    )

    residuals = data["_residual"].to_numpy()
    fitted = data["_fitted"].to_numpy()

    # Standardize residuals for diagnostic visualization.
    residual_std = np.std(residuals, ddof=1)

    if residual_std == 0:
        raise ValueError("Residual variance is zero.")

    standardized_residuals = residuals / residual_std

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 9)
    )

    # 1. Residual distribution
    axes[0, 0].hist(
        residuals,
        bins=40,
        edgecolor="black"
    )

    axes[0, 0].set_title("Residual Distribution")
    axes[0, 0].set_xlabel("Residual")
    axes[0, 0].set_ylabel("Frequency")

    # 2. Normal Q-Q plot
    probplot(
        residuals,
        dist="norm",
        plot=axes[0, 1]
    )

    axes[0, 1].set_title("Normal Q-Q Plot")

    # 3. Residuals versus fitted values
    axes[1, 0].scatter(
        fitted,
        residuals,
        alpha=0.15
    )

    axes[1, 0].axhline(
        y=0,
        linestyle="--"
    )

    axes[1, 0].set_title("Residuals vs Fitted")
    axes[1, 0].set_xlabel("Fitted Values")
    axes[1, 0].set_ylabel("Residuals")

    # 4. Scale-location plot
    axes[1, 1].scatter(
        fitted,
        np.sqrt(np.abs(standardized_residuals)),
        alpha=0.15
    )

    axes[1, 1].set_title("Scale-Location Plot")
    axes[1, 1].set_xlabel("Fitted Values")
    axes[1, 1].set_ylabel(
        "Sqrt(|Standardized Residuals|)"
    )

    fig.suptitle(
        "One-Way ANOVA Diagnostic Plots",
        fontsize=15
    )

    fig.tight_layout()

    return fig, axes
