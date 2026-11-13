
from app.analytics.statistics.descriptive import (
    get_descriptive_statistics
)

from app.analytics.statistics.hypothesis_testing import (
    compare_independent_groups
)

from app.analytics.statistics.correlation import (
    calculate_pairwise_correlation,
    get_correlation_matrix
)

from app.analytics.statistics.confidence_intervals import (
    calculate_mean_confidence_interval,
    calculate_difference_confidence_interval
)

from app.analytics.statistics.categorical import (
    analyze_categorical_relationship
)

from app.analytics.statistics.anova import (
    one_way_group_comparison
)

from app.analytics.statistics.diagnostics import (
    check_anova_assumptions,
    plot_anova_diagnostics
)

from app.analytics.statistics.posthoc import (
    run_posthoc_test
)

from app.analytics.statistics.outliers import (
    detect_outliers
)


SUPPORTED_ANALYSES = {
    "descriptive",
    "independent_groups",
    "correlation",
    "correlation_matrix",
    "mean_confidence_interval",
    "difference_confidence_interval",
    "categorical",
    "anova",
    "anova_diagnostics",
    "anova_plot",
    "posthoc",
    "outliers"
}


def run_statistical_analysis(
    df,
    analysis_type,
    **kwargs
):
    """
    Central dispatcher for statistical analyses.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    analysis_type : str
        Requested statistical analysis.

    kwargs : dict
        Parameters required by the selected analysis.

    Returns
    -------
    dict
        Structured result with analysis metadata.
    """

    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    if not hasattr(df, "columns"):
        raise TypeError("Input must be a Pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if analysis_type not in SUPPORTED_ANALYSES:
        raise ValueError(
            f"Unsupported analysis: {analysis_type}. "
            f"Choose from {sorted(SUPPORTED_ANALYSES)}."
        )

    analysis_functions = {
        "descriptive": get_descriptive_statistics,

        "independent_groups": compare_independent_groups,

        "correlation": calculate_pairwise_correlation,

        "correlation_matrix": get_correlation_matrix,

        "mean_confidence_interval":
            calculate_mean_confidence_interval,

        "difference_confidence_interval":
            calculate_difference_confidence_interval,

        "categorical": analyze_categorical_relationship,

        "anova": one_way_group_comparison,

        "anova_diagnostics": check_anova_assumptions,

        "anova_plot": plot_anova_diagnostics,

        "posthoc": run_posthoc_test,

        "outliers": detect_outliers
    }

    if analysis_type == "mean_confidence_interval":
        column = kwargs.pop("column")

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")

        result = calculate_mean_confidence_interval(
            data=df[column],
            **kwargs
        )

    else:
        function = analysis_functions[analysis_type]

        result = function(
            df=df,
            **kwargs
        )

    return {
        "success": True,
        "analysis_type": analysis_type,
        "input_rows": int(len(df)),
        "input_columns": list(df.columns),
        "result": result
    }
