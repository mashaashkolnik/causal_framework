from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.ipw import bootstrap_ipw, prepare_ipw_dataset
from scripts.plot import plot_error_bars
from variables.variables import (
    df_folder_path,
    errorbar_folder_path,
    log_csv_path,
    result_plot_folder_path,
)


def _next_experiment_id(csv_path: Path) -> int:
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return 1
    try:
        existing = pd.read_csv(csv_path, usecols=["experiment_id"])
        ids = pd.to_numeric(existing["experiment_id"], errors="coerce").dropna()
        return int(ids.max()) + 1 if not ids.empty else 1
    except (ValueError, KeyError, pd.errors.EmptyDataError):
        return 1


def summarize_experiment(
    bal_after: pd.DataFrame,
    ate_table: pd.DataFrame,
    structural_confounders: list[str],
    direct_confounders: list[str],
    partial_confounders: list[str],
    negative_targets: list[str],
    csv_path: str | Path = log_csv_path,
    experiment_id: str | int | None = None,
    exposure: str | None = None,
    extra_meta: dict[str, Any] | None = None,
):
    csv_path = Path(csv_path)
    extra_meta = extra_meta or {}
    required_bal = {"feature", "balanced_0.10_med", "balanced_0.05_med"}
    required_ate = {"outcome", "is_significant_abs"}
    if not required_bal.issubset(bal_after.columns):
        raise ValueError(f"bal_after missing: {required_bal - set(bal_after.columns)}")
    if not required_ate.issubset(ate_table.columns):
        raise ValueError(f"ate_table missing: {required_ate - set(ate_table.columns)}")

    struct_set = set(structural_confounders or [])
    direct_set = set(direct_confounders or [])
    partial_set = set(partial_confounders or [])
    neg_set = set(negative_targets or [])
    
    experiment_id = experiment_id or _next_experiment_id(csv_path)

    def get_pct_and_failing(df: pd.DataFrame, features: set[str], col: str) -> tuple[float, list[str]]:
        subset = df[df["feature"].isin(features)]
        if subset.empty:
            return float(np.nan), []
        mask_pass = subset[col] == True
        pct = 100.0 * mask_pass.sum() / len(subset)
        failing = subset.loc[~mask_pass, "feature"].astype(str).sort_values().tolist()
        return pct, failing

    pct_all_bal10, failing_all = get_pct_and_failing(bal_after, set(bal_after["feature"]), "balanced_0.10_med")
    pct_struct_bal05, failing_struct = get_pct_and_failing(bal_after, struct_set, "balanced_0.05_med")
    pct_direct_bal05, failing_direct = get_pct_and_failing(bal_after, direct_set, "balanced_0.05_med")
    pct_partial_bal10, failing_partial = get_pct_and_failing(bal_after, partial_set, "balanced_0.10_med")

    ate_neg = ate_table[ate_table["outcome"].isin(neg_set)]
    mask_neg_sig = ate_neg["is_significant_abs"] == True
    sig_neg_list = sorted(ate_neg.loc[mask_neg_sig, "outcome"].astype(str).tolist())
    pct_neg_not_sig = 100.0 * (~mask_neg_sig).sum() / len(ate_neg) if not ate_neg.empty else float(np.nan)

    pass_score = (pct_all_bal10 + pct_neg_not_sig == 200) and (pct_struct_bal05 >= 50) and (pct_direct_bal05 >= 50)
    strict_score = (pct_all_bal10 + pct_neg_not_sig + pct_struct_bal05 == 300) and (pct_direct_bal05 >= 70)
    strict_coffee = (pct_all_bal10 + pct_neg_not_sig + pct_struct_bal05 >= 285) and (pct_direct_bal05 >= 100)

    message_pass = 'FAIL'
    if pass_score:
        message_pass = 'PASS'
    if strict_score or strict_coffee:
        message_pass = 'PASS_STRICT'

    log_msg = (
        f"{message_pass} | alpha={extra_meta.get('alpha')}, q={extra_meta.get('q')}, "
        f"clip=({extra_meta.get('clip_low')}, {extra_meta.get('clip_high')}), method={extra_meta.get('method')}"
    )

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment_id": experiment_id,
        "exposure": exposure,
        "log_message": log_msg,
        "pct_all_features_balanced_0_10_med": pct_all_bal10,
        "pct_structural_balanced_0_05_med": pct_struct_bal05,
        "pct_direct_balanced_0_05_med": pct_direct_bal05,
        "pct_partial_balanced_0_10_med": pct_partial_bal10,
        "pct_negative_targets_not_significant": pct_neg_not_sig,
        "n_features_all": len(bal_after),
        "failing_features": ";".join(failing_all),
        "failing_structural_features": ";".join(failing_struct),
        "failing_direct_features": ";".join(failing_direct),
        "failing_partial_features": ";".join(failing_partial),
        "significant_negative_targets": ";".join(sig_neg_list),
        "config": extra_meta,
    }

    out_df = pd.DataFrame([row])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(csv_path, mode="a", index=False, header=not csv_path.exists() or csv_path.stat().st_size == 0)

    return out_df, message_pass, log_msg


def generate_features(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    additional_features: Sequence[str],
    n_features: int = 25,
) -> list[str]:
    shap_importance = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(shap_importance)[::-1][:n_features]
    feature_names = list(X.columns[top_indices])

    for feature in additional_features:
        if feature not in feature_names:
            feature_names.append(feature)
    return feature_names


def run_experiment(
    config: Mapping[str, Any],
    variable_config: Mapping[str, Any],
    df: pd.DataFrame,
    kwargs: Mapping[str, Any],
    X: pd.DataFrame,
    shap_values: np.ndarray,
    save_results: bool = False,
    log_experiment: bool = True,
    errorbar_folder_path: str = errorbar_folder_path,
    log_csv_path: str = log_csv_path,
    df_folder_path: str = df_folder_path,
    result_plot_folder_path: str = result_plot_folder_path,
) -> str | pd.DataFrame | None:
    prop_df = prepare_ipw_dataset(
        df=df,
        treat_col=config["t"],
        e_col=config["calibration"],
        sleep_targets=variable_config["sleep_targets"],
        q=config["q"],
        stabilize=False,
        clip=None,
        dropna_targets=True,
    )
    important_confounders = generate_features(
        X, shap_values, variable_config["confounders"], n_features=config["n_features"]
    )
    all_outcomes = (
        list(variable_config["sleep_targets"])
        + important_confounders
        + list(variable_config["negative_targets"])
    )
    bal_after, ate_table = bootstrap_ipw(
        prop_df,
        treat_col=config["t"],
        e_col=config["calibration"],
        features_to_check=important_confounders,
        outcomes=all_outcomes,
        B=config["n_iter"],
        stabilize=config.get("stabilize", False),
        clip=config.get("clip"),
    )
    sleep_targets = variable_config["sleep_targets"]
    df_bootstrap = ate_table[ate_table["outcome"].isin(sleep_targets)].copy()
    exposure_name = str(kwargs.get("exposure_baseline", "unknown"))
    if save_results and not log_experiment:
        plot_error_bars(
            df_bootstrap,
            treated_title=exposure_name,
            experiment_id=None,
            target_title="Treatment",
            dir=result_plot_folder_path,
        )
        Path(df_folder_path).mkdir(parents=True, exist_ok=True)
        bal_after.to_csv(Path(df_folder_path) / f"{exposure_name}_asmd.csv")
        ate_table.to_csv(Path(df_folder_path) / f"{exposure_name}_ate.csv")
        return "SUCCESS"

    if log_experiment:
        clips = config.get("clip") or (None, None)
        extra_meta = {
            "exposure_baseline": exposure_name,
            "alpha": config.get("alpha"),
            "q": config.get("q"),
            "clip_low": clips[0],
            "clip_high": clips[1],
            "method": config.get("method"),
        }
        summary_df, message_pass, log_message = summarize_experiment(
            bal_after=bal_after.dropna(),
            ate_table=ate_table,
            structural_confounders=variable_config["structural_confounders"],
            direct_confounders=variable_config["direct_confounders"],
            partial_confounders=variable_config["partial_confounders"],
            negative_targets=variable_config["negative_targets"],
            csv_path=log_csv_path,
            exposure=exposure_name,
            extra_meta=extra_meta,
        )
        if save_results:
            Path(df_folder_path).mkdir(parents=True, exist_ok=True)
            bal_after.to_csv(Path(df_folder_path) / f"{exposure_name}_asmd.csv")
            ate_table.to_csv(Path(df_folder_path) / f"{exposure_name}_ate.csv")
        
        exp_id = summary_df.iloc[0]["experiment_id"]
        if message_pass in ('PASS', 'PASS_STRICT'):
            print(f"\nexp_id={exp_id} | {log_message}", end='\t')
            plot_error_bars(
                df_bootstrap,
                treated_title=exposure_name,
                experiment_id=exp_id,
                target_title="Treatment",
                dir=errorbar_folder_path,
            )
        else:
            print(f"exp_id={exp_id} (FAIL)", end='\t')
        return message_pass
    
    return None
