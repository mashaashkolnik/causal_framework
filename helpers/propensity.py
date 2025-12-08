from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# from variables import index
# from helpers import plot_error_bars
index = ["RegistrationCode", "id"]

###################################    PREPROCESSING    ###################################


def export_dataframe(
    file: str | Path,
    sleep_targets: Iterable[str],
    features_id: Iterable[str],
) -> pd.DataFrame:
    """Load, clean, and preprocess a CSV

    Args:
        file: Path to the CSV file
        sleep_targets: Sleep outcome columns; rows with NaNs in these are dropped
        features_id: Feature columns; rows with NaNs in these are dropped
    Returns:
        A cleaned and preprocessed dataframe.
    """
    df = pd.read_csv(file)
    df = df.dropna(subset=sleep_targets)
    df = df.dropna(subset=features_id)
    df.rename(columns={"Unnamed: 0": "id"}, inplace=True)  # TODO
    df = df.set_index(index)
    return df


###################################    ASSIGNING TREATMENT VALUES    ###################################


def treated(value: Optional[float], median: float) -> Optional[bool]:
    """Assign treatment based on a global median threshold
    Returns:
        True if value >= median, False if less, or None if value is None.
    """
    if value is None:
        return None
    return value >= median


def treated_gb(
    value: Optional[float],
    gender: int,
    median: Sequence[float],
) -> Optional[bool]:
    """Assign treatment using gender-specific medians
    Returns:
        True if value >= gender_specific_median, False if less, or None if value is None
    """
    if value is None:
        return None
    gender_median = median[0] if gender == 0 else median[1]
    return value >= gender_median


def assign_treatment_values(
    df: pd.DataFrame,
    exposure: str,
    target: str,
    alpha: float = 0.01,
    method: str = "median",
    cutoff_values: Optional[Tuple[float, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Assign binary treatment based on exposure and trimming rules

    Args:
        df: Input dataframe. Must contain ``exposure`` and baseline column
        exposure: Name of day-specific exposure
        target: Name of the binary treatment column to create
        alpha: Two-sided trimming level on the baseline exposure. If 0, no trimming
        method: Treatment definition; either ``"median"`` or ``"gb"``
        cutoff_values: Optional (low, high) bounds for the day-specific exposure

    Returns:
        A tuple of:
        - Updated dataframe with a treatment column
        - Dictionary with metadata (exposure name, baseline bounds, medians, alpha)
    """
    exposure_baseline = exposure.split("_target_day")[0]
    low, high = np.min(df[exposure_baseline]), np.max(df[exposure_baseline])

    if alpha:
        if df[exposure_baseline].isna().sum() > 0:
            print(f"{df[exposure_baseline].isna().sum()} nulls in exposure baseline")
            df = df.dropna(subset=[exposure_baseline])
        low, high = np.quantile(df[exposure_baseline], alpha / 2), np.quantile(
            df[exposure_baseline], 1 - alpha / 2
        )
        df = df[(df[exposure_baseline] >= low) & (df[exposure_baseline] <= high)]
    baseline_median = df[exposure_baseline].median()

    if method == "median":
        if cutoff_values:
            low, high = cutoff_values
            df = df[(df[exposure] >= low) & (df[exposure] <= high)]
        baseline_median = df[exposure_baseline].median()
        df[target] = [treated(value, baseline_median) for value in df[exposure]]
    elif method == "gb":
        if cutoff_values:
            low, high = cutoff_values
            df = df[(df[exposure] >= low) & (df[exposure] <= high)]
        baseline_median = df.groupby("gender").median()[exposure_baseline]
        f_median, m_median = baseline_median[0], baseline_median[1]
        df[target] = [
            treated_gb(df[exposure][idx], df["gender"][idx], [f_median, m_median])
            for idx in df.index
        ]

    df = df.dropna(subset=[target])
    kwargs = {
        "exposure": exposure,
        "exposure_baseline": exposure_baseline,
        "low": np.min(df[exposure]),
        "high": np.max(df[exposure]),
        "baseline_median": baseline_median,
        #'custom_median' : custom_median,
        "alpha": alpha,
    }
    return df, kwargs


###################################    PROPENSITY MODEL TRAINING    ###################################


def create_pool(
    df: pd.DataFrame,
    categorical: Sequence[str],
    numerical: Sequence[str],
    target: str,
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """Create subject-level train/test splits for propensity modeling

    Splits are done on unique RegistrationCode so an individual does not
    appear in both train and test sets

    Args:
        df: Input dataframe with a MultiIndex including RegistrationCode
        categorical: Names of categorical feature columns
        numerical: Names of numerical feature columns
        target: Name of the binary treatment column

    Returns:
        A tuple with:
        - X: Full feature dataframe
        - y: Full target series
        - X_train: Training features
        - y_train: Training labels
        - X_test: Test features
        - y_test: Test labels
    """

    X = df[numerical + categorical]
    y = df[target]

    for col in categorical:
        X[col] = X[col].astype(str)
    for col in numerical:
        X[col] = X[col].astype(float)

    # (!) Splitting one individual to either test or train
    unique_codes = df.index.get_level_values("RegistrationCode").unique()
    train_codes, test_codes = train_test_split(
        unique_codes, test_size=0.3, random_state=42
    )

    # Create train and test DataFrames
    X_train = X[X.index.get_level_values("RegistrationCode").isin(train_codes)]
    y_train = y[y.index.get_level_values("RegistrationCode").isin(train_codes)]

    X_test = X[X.index.get_level_values("RegistrationCode").isin(test_codes)]
    y_test = y[y.index.get_level_values("RegistrationCode").isin(test_codes)]

    return X, y, X_train, y_train, X_test, y_test


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    categorical: Sequence[str],
    verbose: bool = False,
) -> CatBoostClassifier:
    """Train a CatBoost classifier with class balancing and report accuracy.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        categorical: Names of categorical feature columns.
        verbose: Whether to print time-stamped progress and classification report.

    Returns:
        The fitted CatBoostClassifier.
    """
    pool_train = Pool(X_train, y_train, cat_features=categorical)
    pool_test = Pool(X_test, cat_features=categorical)

    if verbose:
        t = datetime.now()
        print(f"{t.hour}:{t.minute}:{t.second} Create Training and Test Pools...")

    # Balancing classes
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    if verbose:
        t = datetime.now()
        print(
            f"{t.hour}:{t.minute}:{t.second} Compute Class Weights: {class_weights}..."
        )

    # Train
    cb = CatBoostClassifier(class_weights=class_weights, verbose=False)
    cb.fit(pool_train)

    if verbose:
        t = datetime.now()
        print(f"{t.hour}:{t.minute}:{t.second} Fit the CatBoostClassifier..")

    y_predicted = cb.predict(pool_test)

    if verbose:
        t = datetime.now()
        print(f"{t.hour}:{t.minute}:{t.second} Generate Accuracy Report...\n")
        print(classification_report(list(y_predicted), list(y_test)))

    print(f"Accuracy Score={round(accuracy_score(list(y_predicted), list(y_test)), 3)}")

    return cb


def get_shap_summary(
    n_features: int,
    X_temp: pd.DataFrame,
    cb: CatBoostClassifier,
    target_title: str,
    treated_title: str,
    target_name: str,
    save: bool,
) -> Any:
    """Compute SHAP values for a fitted CatBoost model.

    Note:
        Only ``X_temp`` and ``cb`` are used; the other arguments are retained for
        compatibility with plotting or legacy code.

    Args:
        n_features: Number of features (unused).
        X_temp: Feature dataframe for SHAP computation.
        cb: Fitted CatBoost classifier.
        target_title: Unused placeholder for plotting title.
        treated_title: Unused placeholder for plotting title.
        target_name: Unused placeholder.
        save: Unused flag.

    Returns:
        SHAP values as returned by :class:`shap.TreeExplainer`.
    """
    explainer = shap.TreeExplainer(cb)
    shap_values = explainer.shap_values(X_temp)
    return shap_values


def calculate_propensity_scores(
    X: pd.DataFrame,
    y: pd.Series,
    numerical: Sequence[str],
    categorical: Sequence[str],
    cb: CatBoostClassifier,
) -> pd.DataFrame:
    """Compute and Platt-calibrate propensity scores.

    Adds three columns to X:

    * ``'treated'`` – binary treatment indicator from ``y``.
    * ``'score'`` – raw propensity from CatBoost.
    * ``'calibrated_scores_logreg'`` – Platt-calibrated propensity.

    Args:
        X: Feature dataframe.
        y: Binary treatment series.
        numerical: Names of numerical feature columns.
        categorical: Names of categorical feature columns.
        cb: Fitted CatBoost classifier.

    Returns:
        A copy of X with propensity-related columns.
    """
    X["treated"] = y.astype(int)
    propensity_scores = cb.predict_proba(X[numerical + categorical])
    propensity_scores = [score[1] for score in propensity_scores]

    X["score"] = propensity_scores

    # Reshape scores for sklearn
    propensity_scores = np.array(propensity_scores).reshape(-1, 1)
    treatment = np.array(X["treated"])

    # Fit logistic regression for Platt scaling
    platt_model = LogisticRegression()
    platt_model.fit(propensity_scores, treatment)

    # Predict recalibrated probabilities
    calibrated_scores = platt_model.predict_proba(propensity_scores)[:, 1]
    X["calibrated_scores_logreg"] = calibrated_scores
    return X


def train_ps(
    df: pd.DataFrame,
    categorical: Sequence[str],
    numerical: Sequence[str],
    target: str,
    config: Mapping[str, Any],
    kwargs: Mapping[str, Any],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    CatBoostClassifier,
    Any,
]:
    """Train propensity model and compute SHAP and calibrated scores.

    Args:
        df: Input dataframe.
        categorical: Names of categorical features.
        numerical: Names of numerical features.
        target: Name of the treatment column.
        config: Configuration dict; must contain ``'n_features'``.
        kwargs: Metadata dict; must contain ``'exposure_baseline'``.

    Returns:
        A tuple with:
        - df: Original dataframe with propensity-related columns joined.
        - X: Feature dataframe with scores.
        - y: Treatment series.
        - X_train, y_train, X_test, y_test: Train/test splits.
        - cb: Fitted CatBoost model.
        - shap_values: SHAP values.
    """
    X, y, X_train, y_train, X_test, y_test = create_pool(
        df, categorical, numerical, target
    )
    cb = train_and_evaluate(X_train, y_train, X_test, y_test, categorical)

    shap_values = get_shap_summary(
        config["n_features"],
        X[numerical + categorical],
        cb,
        kwargs["exposure_baseline"],
        kwargs["exposure_baseline"],
        "",
        False,
    )

    X = calculate_propensity_scores(X, y, numerical, categorical, cb)
    new_columns = ["score", "treated", "calibrated_scores_logreg"]
    df = df.join(X[new_columns])

    return df, X, y, X_train, y_train, X_test, y_test, cb, shap_values


def get_propensity_scores(
    exposure: str,
    config: Mapping[str, Any],
    variables: Mapping[str, Any],
    file: str | Path,
    method: str = "median",
    cutoff_values: Optional[Tuple[float, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, Any]:
    """Full pipeline: load data, assign treatment, train PS model, compute SHAP.

    Args:
        exposure: Day-specific exposure name
        config: Global configuration
        variables: Configuration dict
        file: CSV file path
        method: Treatment assignment method ("median" or "gb")
        cutoff_values: Optional (low, high) exposure trimming on target day

    Returns:
        A tuple of:
        - df: Dataframe with treatment and PS-related columns
        - kwargs: Metadata from :func:`assign_treatment_values`
        - X: Feature dataframe used for propensity modeling (with scores)
        - shap_values: SHAP values from :func:`get_shap_summary`
    """
    df = export_dataframe(file, variables["sleep_targets"], variables["features_id"])
    df, kwargs = assign_treatment_values(
        df,
        exposure,
        variables["target"],
        alpha=config["alpha"],
        method=method,
        cutoff_values=cutoff_values,
    )
    df, X, y, X_train, y_train, X_test, y_test, cb, shap_values = train_ps(
        df,
        variables["categorical"],
        variables["numerical"],
        variables["target"],
        config,
        kwargs,
    )
    return df, kwargs, X, shap_values
