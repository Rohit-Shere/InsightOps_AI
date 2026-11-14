
from datetime import date, datetime
import math

import numpy as np
import pandas as pd
from langchain.tools import tool

from app.tools.dataset_registry import (
    get_available_statistical_datasets,
    get_dataset_columns,
    load_statistical_dataset,
)
from app.analytics.statistics.engine import run_statistical_analysis


# ---------------------------------------------------------
# 1. JSON NORMALIZATION
# ---------------------------------------------------------

def to_json_safe(value):
    """Convert Pandas, NumPy and date values into JSON-safe Python values."""

    if isinstance(value, pd.DataFrame):
        return [
            to_json_safe(row)
            for row in value.to_dict(orient="records")
        ]

    if isinstance(value, pd.Series):
        return [to_json_safe(item) for item in value.tolist()]

    if isinstance(value, dict):
        return {
            str(key): to_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())

    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()

    if isinstance(value, np.generic):
        return to_json_safe(value.item())

    if value is None or value is pd.NaT or value is pd.NA:
        return None

    if isinstance(value, float) and not math.isfinite(value):
        return None

    if pd.api.types.is_scalar(value):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

    return value


# ---------------------------------------------------------
# 2. DATASET AND COLUMN VALIDATION
# ---------------------------------------------------------

def _get_dataset_context(dataset_name: str):
    """Load an approved dataset and its metadata."""

    metadata = get_dataset_columns(dataset_name)
    df = load_statistical_dataset(dataset_name)

    return df, metadata


def _validate_column(
    column: str,
    metadata: dict,
    allowed_types: tuple[str, ...],
):
    """Validate that a column exists and has an allowed type."""

    available_columns = (
        metadata["numeric_columns"]
        + metadata["categorical_columns"]
        + metadata["boolean_columns"]
        + metadata["date_columns"]
        + metadata["identifier_columns"]
    )

    if column not in available_columns:
        raise ValueError(
            f"Column '{column}' is not available in this dataset."
        )

    column_type = None

    for type_name in (
        "numeric",
        "categorical",
        "boolean",
        "date",
        "identifier",
    ):
        if column in metadata[f"{type_name}_columns"]:
            column_type = type_name
            break

    if column_type not in allowed_types:
        raise ValueError(
            f"Column '{column}' has type '{column_type}'. "
            f"Allowed types: {allowed_types}."
        )


def _execute_analysis(
    dataset_name: str,
    analysis_type: str,
    **kwargs,
):
    """Load, validate context, run the existing engine and normalize output."""

    try:
        df, metadata = _get_dataset_context(dataset_name)

        result = run_statistical_analysis(
            df,
            analysis_type,
            **kwargs,
        )

        return to_json_safe({
            "success": True,
            "dataset": dataset_name,
            "grain": metadata["grain"],
            "row_count": len(df),
            "analysis": result,
            "data_caveats": metadata["caveats"],
        })

    except Exception as error:
        return {
            "success": False,
            "dataset": dataset_name,
            "error_type": type(error).__name__,
            "error": str(error),
        }


# ---------------------------------------------------------
# 3. DATASET DISCOVERY TOOLS
# ---------------------------------------------------------

@tool
def list_statistical_datasets() -> list:
    """
    List approved datasets available for statistical analysis.

    Use this before selecting a dataset when the user's
    question does not clearly identify the required data.
    """
    return to_json_safe(get_available_statistical_datasets())


@tool
def inspect_statistical_dataset(dataset_name: str) -> dict:
    """
    Inspect the columns, data types, grain and caveats of an
    approved statistical dataset before choosing an analysis.
    """
    try:
        metadata = get_dataset_columns(dataset_name)

        return {
            "success": True,
            **to_json_safe(metadata),
        }

    except Exception as error:
        return {
            "success": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }


# ---------------------------------------------------------
# 4. DESCRIPTIVE STATISTICS
# ---------------------------------------------------------

@tool
def run_descriptive_analysis(
    dataset_name: str,
    columns: list[str] | None = None,
) -> dict:
    """
    Calculate descriptive statistics for numeric columns.

    Use for means, medians, standard deviations, quartiles,
    ranges and distribution summaries.

    If columns are omitted, all registered numeric columns
    in the selected dataset are analyzed.
    """
    try:
        metadata = get_dataset_columns(dataset_name)

        selected_columns = (
            columns
            if columns is not None
            else metadata["numeric_columns"]
        )

        if not selected_columns:
            raise ValueError("No numeric columns are available.")

        for column in selected_columns:
            _validate_column(column, metadata, ("numeric",))

        return _execute_analysis(
            dataset_name,
            "descriptive",
            columns=selected_columns,
        )

    except Exception as error:
        return {
            "success": False,
            "dataset": dataset_name,
            "error_type": type(error).__name__,
            "error": str(error),
        }


# ---------------------------------------------------------
# 5. INDEPENDENT GROUP COMPARISON
# ---------------------------------------------------------

@tool
def run_group_comparison(
    dataset_name: str,
    group_column: str,
    target_column: str,
    group1_value: str,
    group2_value: str,
    method: str = "mannwhitney",
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> dict:
    """
    Compare a numeric outcome between two independent groups.

    Supports Mann-Whitney U and Welch's independent t-test.

    Use when asking whether a numeric measure differs
    between two groups. Check dataset grain and caveats
    before assuming observations are independent.
    """
    try:
        metadata = get_dataset_columns(dataset_name)

        _validate_column(
            group_column,
            metadata,
            ("categorical", "boolean"),
        )
        _validate_column(
            target_column,
            metadata,
            ("numeric",),
        )

        if method not in ("mannwhitney", "welch"):
            raise ValueError(
                "method must be 'mannwhitney' or 'welch'."
            )

        if alternative not in (
            "two-sided",
            "less",
            "greater",
        ):
            raise ValueError("Invalid alternative hypothesis.")

        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")

        return _execute_analysis(
            dataset_name,
            "independent_groups",
            group_column=group_column,
            target_column=target_column,
            group1_value=group1_value,
            group2_value=group2_value,
            method=method,
            alternative=alternative,
            alpha=alpha,
        )

    except Exception as error:
        return {
            "success": False,
            "dataset": dataset_name,
            "error_type": type(error).__name__,
            "error": str(error),
        }


# ---------------------------------------------------------
# 6. CORRELATION
# ---------------------------------------------------------

@tool
def run_correlation_analysis(
    dataset_name: str,
    column1: str,
    column2: str,
    method: str = "spearman",
    alpha: float = 0.05,
) -> dict:
    """
    Calculate the correlation between two numeric variables.

    Supports Pearson, Spearman and Kendall correlation.
    Use Spearman for ordinal or monotonic relationships.
    Correlation does not establish causation.
    """
    try:
        metadata = get_dataset_columns(dataset_name)

        _validate_column(column1, metadata, ("numeric",))
        _validate_column(column2, metadata, ("numeric",))

        if method not in ("pearson", "spearman", "kendall"):
            raise ValueError(
                "method must be pearson, spearman or kendall."
            )

        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")

        return _execute_analysis(
            dataset_name,
            "correlation",
            column1=column1,
            column2=column2,
            method=method,
            alpha=alpha,
        )

    except Exception as error:
        return {
            "success": False,
            "dataset": dataset_name,
            "error_type": type(error).__name__,
            "error": str(error),
        }


# ---------------------------------------------------------
# 7. CONFIDENCE INTERVALS
# ---------------------------------------------------------

@tool
def run_mean_confidence_interval(
    dataset_name: str,
    column: str,
    confidence_level: float = 0.95,
) -> dict:
    """
    Calculate a confidence interval for a numeric variable's mean.
    """
    try:
        metadata = get_dataset_columns(dataset_name)

        _validate_column(column, metadata, ("numeric",))

        if not 0 < confidence_level < 1:
            raise ValueError(
                "confidence_level must be between 0 and 1."
            )

        return _execute_analysis(
            dataset_name,
            "mean_confidence_interval",
            column=column,
            confidence_level=confidence_level,
        )

    except Exception as error:
        return {
            "success": False,
            "dataset": dataset_name,
            "error_type": type(error).__name__,
            "error": str(error),
        }


@tool
def run_difference_confidence_interval(
    dataset_name: str,
    group_column: str,
    target_column: str,
    group1_value: str,
    group2_value: str,
    confidence_level: float = 0.95,
) -> dict:
    """
    Calculate a Welch confidence interval for the difference
    between two independent group means.
    """
    try:
        metadata = get_dataset_columns(dataset_name)

        _validate_column(
            group_column,
            metadata,
            ("categorical", "boolean"),
        )
        _validate_column(target_column, metadata, ("numeric",))

        if not 0 < confidence_level < 1:
            raise ValueError(
                "confidence_level must be between 0 and 1."
            )

        return _execute_analysis(
            dataset_name,
            "difference_confidence_interval",
            group_column=group_column,
            target_column=target_column,
            group1_value=group1_value,
            group2_value=group2_value,
            confidence_level=confidence_level,
        )

    except Exception as error:
        return {
            "success": False,
            "dataset": dataset_name,
            "error_type": type(error).__name__,
            "error": str(error),
        }


# ---------------------------------------------------------
# 8. CATEGORICAL ANALYSIS
# ---------------------------------------------------------

@tool
def run_categorical_analysis(
    dataset_name: str,
    column1: str,
    column2: str,
    alpha: float = 0.05,
    correction: bool = True,
) -> dict:
    """
    Test the association between two categorical variables.

    Returns chi-square test results and effect-size information
    such as Cramer's V through the statistical engine.
    """
    try:
        metadata = get_dataset_columns(dataset_name)

        allowed = ("categorical", "boolean")

        _validate_column(column1, metadata, allowed)
        _validate_column(column2, metadata, allowed)

        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")

        return _execute_analysis(
            dataset_name,
            "categorical",
            column1=column1,
            column2=column2,
            alpha=alpha,
            correction=correction,
        )

    except Exception as error:
        return {
            "success": False,
            "dataset": dataset_name,
            "error_type": type(error).__name__,
            "error": str(error),
        }


# ---------------------------------------------------------
# 9. ANOVA
# ---------------------------------------------------------

@tool
def run_anova_analysis(
    dataset_name: str,
    group_column: str,
    target_column: str,
    method: str = "welch",
    alpha: float = 0.05,
) -> dict:
    """
    Compare a numeric outcome across three or more groups.

    Supports standard ANOVA, Welch ANOVA and Kruskal-Wallis.
    Select the method based on the data and assumptions.
    """
    try:
        metadata = get_dataset_columns(dataset_name)

        _validate_column(
            group_column,
            metadata,
            ("categorical", "boolean"),
        )
        _validate_column(target_column, metadata, ("numeric",))

        if method not in ("standard", "welch", "kruskal"):
            raise ValueError(
                "method must be standard, welch or kruskal."
            )

        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")

        return _execute_analysis(
            dataset_name,
            "anova",
            group_column=group_column,
            target_column=target_column,
            method=method,
            alpha=alpha,
        )

    except Exception as error:
        return {
            "success": False,
            "dataset": dataset_name,
            "error_type": type(error).__name__,
            "error": str(error),
        }


@tool
def run_anova_diagnostics(
    dataset_name: str,
    group_column: str,
    target_column: str,
    alpha: float = 0.05,
) -> dict:
    """
    Check ANOVA assumptions and diagnostics for a grouped
    numeric outcome, including variance and residual normality.
    """
    try:
        metadata = get_dataset_columns(dataset_name)

        _validate_column(
            group_column,
            metadata,
            ("categorical", "boolean"),
        )
        _validate_column(target_column, metadata, ("numeric",))

        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")

        return _execute_analysis(
            dataset_name,
            "anova_diagnostics",
            group_column=group_column,
            target_column=target_column,
            alpha=alpha,
        )

    except Exception as error:
        return {
            "success": False,
            "dataset": dataset_name,
            "error_type": type(error).__name__,
            "error": str(error),
        }


# ---------------------------------------------------------
# 10. POST-HOC TESTS
# ---------------------------------------------------------

@tool
def run_posthoc_analysis(
    dataset_name: str,
    group_column: str,
    target_column: str,
    method: str = "dunn",
    alpha: float = 0.05,
) -> dict:
    """
    Perform post-hoc pairwise comparisons between groups.

    Supports Dunn-Holm, Games-Howell and Tukey.
    Normally use after a suitable overall group comparison.
    """
    try:
        metadata = get_dataset_columns(dataset_name)

        _validate_column(
            group_column,
            metadata,
            ("categorical", "boolean"),
        )
        _validate_column(target_column, metadata, ("numeric",))

        if method not in ("dunn", "games_howell", "tukey"):
            raise ValueError(
                "method must be dunn, games_howell or tukey."
            )

        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")

        return _execute_analysis(
            dataset_name,
            "posthoc",
            group_column=group_column,
            target_column=target_column,
            method=method,
            alpha=alpha,
        )

    except Exception as error:
        return {
            "success": False,
            "dataset": dataset_name,
            "error_type": type(error).__name__,
            "error": str(error),
        }


# ---------------------------------------------------------
# 11. OUTLIER DETECTION
# ---------------------------------------------------------

@tool
def run_outlier_analysis(
    dataset_name: str,
    method: str = "iqr",
    column: str | None = None,
    columns: list[str] | None = None,
) -> dict:
    """
    Detect potential outliers using IQR, Z-score,
    modified Z-score or Isolation Forest.

    Specify either one column or a list of columns.
    Outlier flags require business context before action.
    """
    try:
        metadata = get_dataset_columns(dataset_name)

        if column is not None and columns is not None:
            raise ValueError(
                "Provide either column or columns, not both."
            )

        selected_columns = (
            [column] if column is not None
            else columns
        )

        if selected_columns is not None:
            for selected_column in selected_columns:
                _validate_column(
                    selected_column,
                    metadata,
                    ("numeric",),
                )

        if method not in (
            "iqr",
            "zscore",
            "modified_zscore",
            "isolation_forest",
        ):
            raise ValueError("Unsupported outlier method.")

        return _execute_analysis(
            dataset_name,
            "outliers",
            method=method,
            column=column,
            columns=columns,
        )

    except Exception as error:
        return {
            "success": False,
            "dataset": dataset_name,
            "error_type": type(error).__name__,
            "error": str(error),
        }


# ---------------------------------------------------------
# 12. TOOL COLLECTION
# ---------------------------------------------------------

statistical_tools = [
    list_statistical_datasets,
    inspect_statistical_dataset,
    run_descriptive_analysis,
    run_group_comparison,
    run_correlation_analysis,
    run_mean_confidence_interval,
    run_difference_confidence_interval,
    run_categorical_analysis,
    run_anova_analysis,
    run_anova_diagnostics,
    run_posthoc_analysis,
    run_outlier_analysis,
]


def _coerce_group_value(value: str, df: pd.DataFrame, column: str):
    if pd.api.types.is_bool_dtype(df[column]):
        normalized = value.strip().lower()

        if normalized == "true":
            return True
        if normalized == "false":
            return False

        raise ValueError(
            f"'{value}' is not a valid Boolean value for {column}."
        )

    return value