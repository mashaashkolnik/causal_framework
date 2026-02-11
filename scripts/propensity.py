from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

index = ["RegistrationCode", "id"]


def export_dataframe(
    file: str | Path,
    sleep_targets: Iterable[str],
    features_id: Iterable[str],
) -> pd.DataFrame:
    df = pd.read_csv(file)
    df = df.dropna(subset=sleep_targets)
    df = df.dropna(subset=features_id)
    df.rename(columns={"Unnamed: 0": "id"}, inplace=True)
    df = df.set_index(index)
    return df


def treated(value: Optional[float], median: float) -> Optional[bool]:
    if value is None:
        return None
    return value >= median


def treated_gb(
    value: Optional[float],
    gender: int,
    median: Sequence[float],
) -> Optional[bool]:
    if value is None:
        return None
    gender_median = median[0] if gender == 0 else median[1]
    return value >= gender_median


def treated_extreme(
    value: Optional[float],
    upper_left: float,
    lower_right: float,
) -> Optional[bool]:
    if value is None:
        return None
    elif value <= upper_left:
        return False
    elif value >= lower_right:
        return True
    return None


def assign_treatment_values(
    df: pd.DataFrame,
    exposure: str,
    target: str,
    alpha: float = 0.00,
    method: str = "median",
    limit=None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    exposure_baseline = exposure.split("_target_day")[0]
    low, high = np.min(df[exposure]), np.max(df[exposure])
    baseline_median = df[exposure].median()
    if limit:
        low, high = limit
        df = df[(df[exposure] >= low) & (df[exposure] <= high)]
    if method == "median":
        baseline_median = df[exposure].median()
        baseline_median = max(baseline_median, 0.05)
        df[target] = [treated(value, baseline_median) for value in df[exposure]]
    elif method == "gb":
        baseline_median = df[["gender", exposure]].groupby("gender").median()[exposure]
        f_median, m_median = baseline_median[0], baseline_median[1]
        df[target] = [
            treated_gb(df[exposure][idx], df["gender"][idx], [f_median, m_median])
            for idx in df.index
        ]
    elif method == "quantile":
        q = kwargs["q"]
        upper_left, lower_right = np.quantile(df[exposure], [q, 1 - q])
        df[target] = [
            treated_extreme(df[exposure][idx], upper_left, lower_right)
            for idx in df.index
        ]
    output = {
        "exposure": exposure,
        "exposure_baseline": exposure_baseline,
        "low": np.min(df[exposure]),
        "high": np.max(df[exposure]),
        "baseline_median": baseline_median,
        "alpha": alpha,
    }
    return df, output


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
    X = df[numerical + categorical].copy()
    y = df[target]
    for col in categorical:
        X[col] = X[col].astype(str)
    for col in numerical:
        X[col] = X[col].astype(float)
    unique_codes = df.index.get_level_values("RegistrationCode").unique()
    train_codes, test_codes = train_test_split(
        unique_codes, test_size=0.3, random_state=42
    )
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
    pool_train = Pool(X_train, y_train, cat_features=categorical)
    pool_test = Pool(X_test, cat_features=categorical)
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    cb = CatBoostClassifier(class_weights=class_weights, verbose=False)
    cb.fit(pool_train)
    y_predicted = cb.predict(pool_test)
    if verbose:
        print(classification_report(list(y_predicted), list(y_test)))
    return cb


def get_shap_summary(
    X_temp: pd.DataFrame,
    cb: CatBoostClassifier,
) -> Any:
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
    X = X.copy()
    X["treated"] = y.astype(int)
    propensity_scores = cb.predict_proba(X[numerical + categorical])
    propensity_scores = [score[1] for score in propensity_scores]
    X["score"] = propensity_scores
    propensity_scores = np.array(propensity_scores).reshape(-1, 1)
    treatment = np.array(X["treated"])
    platt_model = LogisticRegression()
    platt_model.fit(propensity_scores, treatment)
    calibrated_scores = platt_model.predict_proba(propensity_scores)[:, 1]
    X["calibrated_scores_logreg"] = calibrated_scores
    return X


def train_ps(
    df: pd.DataFrame,
    variables: Mapping[str, Any],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Any,
]:
    categorical, numerical, target = (
        variables["categorical"],
        variables["numerical"],
        variables["target"],
    )
    X, y, X_train, y_train, X_test, y_test = create_pool(
        df, categorical, numerical, target
    )
    cb = train_and_evaluate(X_train, y_train, X_test, y_test, categorical)
    shap_values = get_shap_summary(X[numerical + categorical], cb)
    X = calculate_propensity_scores(X, y, numerical, categorical, cb)
    new_columns = ["score", "treated", "calibrated_scores_logreg"]
    df = df.join(X[new_columns])
    return df, X, shap_values


def get_propensity_scores(
    exposure: str,
    config: Mapping[str, Any],
    variables: Mapping[str, Any],
    file: str | Path,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, Any]:
    df = export_dataframe(file, variables["sleep_targets"], variables["features_id"])
    df, kwargs = assign_treatment_values(
        df,
        exposure,
        variables["target"],
        alpha=config["alpha"],
        method=config["method"],
        limit=config["limit"],
        kwargs={"q": config["quantile"]},
    )
    df = df.dropna(subset=[variables["target"]])
    df, X, shap_values = train_ps(df, variables)
    return df, kwargs, X, shap_values
