from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.ipw import *
from scripts.plot import plot_error_bars
from variables.variables import *


def _next_experiment_id(csv_path: Path) -> int:
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return 1
    try:
        existing = pd.read_csv(csv_path, usecols=["experiment_id"])
    except Exception:
        return 1
    ids = pd.to_numeric(existing["experiment_id"], errors="coerce").dropna()
    if ids.empty:
        return 1
    return int(ids.max()) + 1


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
    required_bal_cols = {"feature", "balanced_0.10_med", "balanced_0.05_med"}
    required_ate_cols = {"outcome", "is_significant_abs"}
    missing_bal = required_bal_cols - set(bal_after.columns)
    missing_ate = required_ate_cols - set(ate_table.columns)
    if missing_bal:
        raise ValueError(f"bal_after missing columns: {missing_bal}")
    if missing_ate:
        raise ValueError(f"ate_table missing columns: {missing_ate}")
    structural_set = set(structural_confounders or [])
    direct_set = set(direct_confounders or [])
    partial_set = set(partial_confounders or [])
    neg_set = set(negative_targets or [])
    union_conf = structural_set | direct_set | partial_set
    if experiment_id is None:
        experiment_id = _next_experiment_id(csv_path)
    def pct(numer: int, denom: int) -> float:
        """Compute percentage or NaN if denom is 0."""
        return float(np.nan) if denom == 0 else 100.0 * numer / denom
    n_all = len(bal_after)
    n_all_bal10 = int((bal_after["balanced_0.10_med"] == True).sum())  # noqa: E712
    pct_all_bal10 = pct(n_all_bal10, n_all)
    def group_pct(
        df_bal: pd.DataFrame, names: set[str], col_name: str
    ) -> tuple[float, pd.DataFrame, int, int]:
        group_df = df_bal[df_bal["feature"].isin(names)]
        denom = len(group_df)
        numer = int((group_df[col_name] == True).sum())  # noqa: E712
        return pct(numer, denom), group_df, numer, denom
    pct_struct_bal05, df_struct, n_struct_bal, n_struct_total = group_pct(
        bal_after, structural_set, "balanced_0.05_med"
    )
    pct_direct_bal05, df_direct, n_direct_bal, n_direct_total = group_pct(
        bal_after, direct_set, "balanced_0.05_med"
    )
    pct_partial_bal10, df_partial, n_partial_bal, n_partial_total = group_pct(
        bal_after, partial_set, "balanced_0.10_med"
    )
    def failing_features(group_df: pd.DataFrame, threshold_col: str) -> list[str]:
        if group_df.empty:
            return []
        mask_fail = group_df[threshold_col] != True  # noqa: E712
        return group_df.loc[mask_fail, "feature"].astype(str).sort_values().tolist()

    failing_all = failing_features(bal_after, "balanced_0.10_med")
    failing_struct = failing_features(df_struct, "balanced_0.05_med")
    failing_direct = failing_features(df_direct, "balanced_0.05_med")
    failing_partial = failing_features(df_partial, "balanced_0.10_med")

    n_struct_unbal = len(failing_struct)
    n_direct_unbal = len(failing_direct)
    n_partial_unbal = len(failing_partial)

    ate_conf = ate_table[ate_table["outcome"].isin(union_conf)].copy()
    ate_neg = ate_table[ate_table["outcome"].isin(neg_set)].copy()

    #sig_conf = ate_conf.loc[
    #    ate_conf["is_significant_abs"] == True, "outcome"  # noqa: E712
    #].astype(str)
    #sig_conf_list = sorted(sig_conf.unique().tolist())

    #denom_conf = len(ate_conf)
    #numer_conf_not_sig = int(
    #    (ate_conf["is_significant_abs"] == False).sum()  # noqa: E712
    #)
    #pct_conf_not_sig = pct(numer_conf_not_sig, denom_conf)

    sig_neg = ate_neg.loc[
        ate_neg["is_significant_abs"] == True, "outcome"  # noqa: E712
    ].astype(str)
    sig_neg_list = sorted(sig_neg.unique().tolist())

    denom_neg = len(ate_neg)
    numer_neg_not_sig = int(
        (ate_neg["is_significant_abs"] == False).sum()  # noqa: E712
    )
    pct_neg_not_sig = pct(numer_neg_not_sig, denom_neg)
    #flag_struct_unbalanced = n_struct_unbal > 0
    #flag_direct_unbalanced = n_direct_unbal > 0
    #flag_partial_unbalanced = n_partial_unbal > 0
    #flag_neg_controls_issue = len(sig_neg_list) > 0
    
    quality_score_pass = bool(
        (pct_all_bal10 + pct_neg_not_sig == 200) and (pct_struct_bal05 >= 50) and (pct_direct_bal05 >= 50)
    )
    quality_score_pass_strict = bool(
        (pct_all_bal10 + pct_neg_not_sig + pct_struct_bal05 == 300) and (pct_direct_bal05 >= 70)
    ) 
    quality_score_pass_strict_coffee = bool(
        (pct_all_bal10 + pct_neg_not_sig + pct_struct_bal05 >= 285) and (pct_direct_bal05 >= 100)
    ) 
    message_pass = 'FAIL'
    if quality_score_pass:
        message_pass = 'PASS'
    if quality_score_pass_strict or quality_score_pass_strict_coffee:
        message_pass = 'PASS_STRICT'
    log_message = (
        f"{message_pass} | with alpha={extra_meta['alpha']}, q={extra_meta['q']}, clip={extra_meta['clip_low'], extra_meta['clip_high']}, cutoff={extra_meta['cutoff_values']}, method={extra_meta['method']}\t"
    )
    row: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment_id": experiment_id,
        "exposure": exposure,
        "log_message" : log_message,
        "pct_all_features_balanced_0_10_med": pct_all_bal10,
        "pct_structural_balanced_0_05_med": pct_struct_bal05,
        "pct_direct_balanced_0_05_med": pct_direct_bal05,
        "pct_partial_balanced_0_10_med": pct_partial_bal10,
        "pct_negative_targets_not_significant": pct_neg_not_sig,
        "n_features_all": n_all,
        "failing_features": ";".join(failing_all),
        "failing_structural_features": ";".join(failing_struct),
        "failing_direct_features": ";".join(failing_direct),
        "failing_partial_features": ";".join(failing_partial),
        "significant_negative_targets": ";".join(sig_neg_list),
        "config" : extra_meta,
    }
    out_df = pd.DataFrame([row])

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = not csv_path.exists() or csv_path.stat().st_size == 0
    out_df.to_csv(csv_path, mode="a", index=False, header=header)

    return out_df, message_pass, log_message


def generate_features(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    additional_features: Sequence[str],
    n_features: int = 25,
) -> list[str]:
    important_features = list(
        X.columns[np.argsort(np.abs(shap_values).mean(axis=0))]
    )
    important_features.reverse()
    feature_names: list[str] = important_features[:n_features]
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
) -> int | pd.DataFrame | None:
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
    confounders_for_asmd_report = generate_features(
        X,
        shap_values,
        variable_config["confounders"],
        n_features=config["n_features"],
    )
    outcomes = (
        list(variable_config["sleep_targets"])
        + confounders_for_asmd_report
        + list(variable_config["negative_targets"])
    )
    bal_after, ate_table = bootstrap_ipw(
        prop_df,
        treat_col=config["t"],
        e_col=config["calibration"],
        features_to_check=confounders_for_asmd_report,
        outcomes=outcomes,
        B=config["n_iter"],
        stabilize=config["stabilize"],
        clip=config["clip"],
    )
    df_bootstrap = (
        ate_table.set_index("outcome").loc[variable_config["sleep_targets"]].copy()
    )
    df_bootstrap["outcome"] = df_bootstrap.index
    if save_results and not log_experiment:
        plot_error_bars(
            df_bootstrap,
            treated_title=str(kwargs["exposure_baseline"]),
            experiment_id=None,  # no experiment_id prefix in this mode
            target_title="Treatment",
            dir=result_plot_folder_path,
        )
        filename = str(kwargs["exposure_baseline"])
        Path(df_folder_path).mkdir(parents=True, exist_ok=True)
        bal_after.to_csv(f"{df_folder_path}/{filename}_asmd.csv")
        ate_table.to_csv(f"{df_folder_path}/{filename}_ate.csv")
        return 1

    if log_experiment:
        experiment_id = _next_experiment_id(log_csv_path)
        extra_meta = {
            "exposure_baseline": kwargs.get("exposure_baseline"),
            "alpha": config.get("alpha"),
            "q": config.get("q"),
            "clip_low": (config.get("clip") or (None, None))[0],
            "clip_high": (config.get("clip") or (None, None))[1],
            "stabilize": config.get("stabilize"),
            "n_iter": config.get("n_iter"),
            "quantile_cut": config.get("quantile_cut"),
            "cutoff_values": config.get("cutoff_values"),
            "rdi_values": config.get("rdi_values"),
            "method": config.get("method"),
        }

        summary, message_pass, log_message = summarize_experiment(
            bal_after=bal_after.dropna(),
            ate_table=ate_table,
            structural_confounders=list(variable_config["structural_confounders"]),
            direct_confounders=list(variable_config["direct_confounders"]),
            partial_confounders=list(variable_config["partial_confounders"]),
            negative_targets=list(variable_config["negative_targets"]),
            csv_path=log_csv_path,
            experiment_id=experiment_id,
            exposure=str(kwargs["exposure_baseline"]),
            extra_meta=extra_meta,
        )
        if save_results:
            filename = str(kwargs["exposure_baseline"])
            Path(df_folder_path).mkdir(parents=True, exist_ok=True)
            bal_after.to_csv(f"{df_folder_path}/{filename}_ate.csv")
            ate_table.to_csv(f"{df_folder_path}/{filename}_asmd.csv")

        s = summary.iloc[0]
        if message_pass == 'PASS' or message_pass == 'PASS_STRICT':
            print(f"\nexp_id={s['experiment_id']} | {log_message}", end='\t')
            plot_error_bars(
                df_bootstrap,
                treated_title=str(kwargs["exposure_baseline"]),
                experiment_id=experiment_id,
                target_title="Treatment",
                dir=errorbar_folder_path,
            )
        else:
            print(f"exp_id={s['experiment_id']}", end='\t')
        return message_pass
    return None
