
import numpy as np
import pandas as pd

from scipy.stats import median_abs_deviation
from sklearn.ensemble import IsolationForest

from app.analytics.statistics.preparation import validate_dataframe


def _clean_numeric_data(df, columns):
    """Validate and prepare numeric columns."""

    validate_dataframe(df, columns)

    data = df[columns].copy()

    for column in columns:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.replace(
        [np.inf, -np.inf], np.nan
    )

    return data


def detect_iqr_outliers(
    df,
    column,
    multiplier=1.5
):
    """Detect univariate outliers using IQR boundaries."""

    if multiplier <= 0:
        raise ValueError("multiplier must be positive.")

    data = _clean_numeric_data(df, [column])

    values = data[column].dropna()

    if len(values) < 4:
        raise ValueError(
            "At least four valid observations are required."
        )

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    mask = (
        (data[column] < lower_bound)
        | (data[column] > upper_bound)
    )

    return {
        "method": "IQR",
        "column": column,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "valid_observations": int(values.size),
        "outlier_count": int(mask.sum()),
        "outlier_percentage": float(
            mask.sum() / values.size * 100
        ),
        "outlier_indices": data.index[mask].tolist(),
        "outlier_values": data.loc[mask, column].tolist()
    }


def detect_zscore_outliers(
    df,
    column,
    threshold=3.0
):
    """Detect outliers using standard Z-scores."""

    if threshold <= 0:
        raise ValueError("threshold must be positive.")

    data = _clean_numeric_data(df, [column])

    values = data[column].dropna()

    if len(values) < 2:
        raise ValueError(
            "At least two valid observations are required."
        )

    mean = values.mean()
    std = values.std(ddof=1)

    if std == 0 or not np.isfinite(std):
        raise ValueError(
            "Z-score cannot be calculated for constant data."
        )

    z_scores = (data[column] - mean) / std

    mask = z_scores.abs() > threshold

    return {
        "method": "Z-score",
        "column": column,
        "threshold": threshold,
        "mean": float(mean),
        "standard_deviation": float(std),
        "valid_observations": int(values.size),
        "outlier_count": int(mask.sum()),
        "outlier_percentage": float(
            mask.sum() / values.size * 100
        ),
        "outlier_indices": data.index[mask].tolist(),
        "outlier_values": data.loc[mask, column].tolist()
    }


def detect_modified_zscore_outliers(
    df,
    column,
    threshold=3.5
):
    """Detect outliers using median and MAD."""

    if threshold <= 0:
        raise ValueError("threshold must be positive.")

    data = _clean_numeric_data(df, [column])

    values = data[column].dropna()

    if len(values) < 2:
        raise ValueError(
            "At least two valid observations are required."
        )

    median = values.median()

    mad = median_abs_deviation(
        values,
        scale=1.0
    )

    if mad == 0:
        raise ValueError(
            "MAD is zero. Modified Z-score is undefined "
            "for this dataset; consider IQR or another method."
        )

    modified_z = (
        0.6745 * (data[column] - median) / mad
    )

    mask = modified_z.abs() > threshold

    return {
        "method": "Modified Z-score",
        "column": column,
        "threshold": threshold,
        "median": float(median),
        "mad": float(mad),
        "valid_observations": int(values.size),
        "outlier_count": int(mask.sum()),
        "outlier_percentage": float(
            mask.sum() / values.size * 100
        ),
        "outlier_indices": data.index[mask].tolist(),
        "outlier_values": data.loc[mask, column].tolist()
    }


def detect_isolation_forest_outliers(
    df,
    columns,
    contamination="auto",
    random_state=42
):
    """Detect multivariate outliers using Isolation Forest."""

    if not columns:
        raise ValueError("At least one column is required.")

    data = _clean_numeric_data(df, columns)

    valid_data = data.dropna()

    if len(valid_data) < 10:
        raise ValueError(
            "At least ten complete observations are required."
        )

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state
    )

    predictions = model.fit_predict(valid_data)

    scores = model.decision_function(valid_data)

    mask = predictions == -1

    result = valid_data.copy()

    result["anomaly_score"] = scores
    result["is_outlier"] = mask

    return {
        "method": "Isolation Forest",
        "columns": columns,
        "valid_observations": int(len(valid_data)),
        "outlier_count": int(mask.sum()),
        "outlier_percentage": float(
            mask.sum() / len(valid_data) * 100
        ),
        "results": result
    }


def detect_outliers(
    df,
    method="iqr",
    column=None,
    columns=None,
    **kwargs
):
    """Generic outlier detection dispatcher."""

    methods = {
        "iqr": detect_iqr_outliers,
        "zscore": detect_zscore_outliers,
        "modified_zscore": detect_modified_zscore_outliers,
        "isolation_forest": detect_isolation_forest_outliers
    }

    if method not in methods:
        raise ValueError(
            f"Unsupported method: {method}. "
            f"Choose from {list(methods.keys())}."
        )

    if method == "isolation_forest":
        if columns is None:
            raise ValueError(
                "columns must be provided for Isolation Forest."
            )

        return methods[method](
            df=df,
            columns=columns,
            **kwargs
        )

    if column is None:
        raise ValueError(
            "column must be provided for univariate methods."
        )

    return methods[method](
        df=df,
        column=column,
        **kwargs
    )
