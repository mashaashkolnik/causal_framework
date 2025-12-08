from __future__ import annotations
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from helpers.ipw import *


###################################    PLOT    ###################################


def plot_error_bars(
    df_bootstrap: pd.DataFrame,
    treated_title: str,
    experiment_id: int | str | None = None,
    target_title: str = "Treatment",
    dir: str = "experiment/results/charts",
) -> Path:
    """Plot ATE bootstrap error bars for each outcome.

    If experiment_id is provided, it is used as a prefix in the filename.
    This is intended to match the experiment_id used in experiment_summaries.csv.
    """
    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")

    for _, row in df_bootstrap.iterrows():
        color = "red" if row["is_significant_abs"] else "black"
        effect = row["ATE_pct_point"]
        lower = row["CI_pct_2.5"]
        upper = row["CI_pct_97.5"]
        ci_low = effect - lower
        ci_high = upper - effect
        feature = row["outcome"]
        p_value = row["p_value_boot_abs"]

        # Plot error bar
        plt.errorbar(
            effect,
            feature,
            xerr=[[ci_low], [ci_high]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2.5,
            capsize=4,
            alpha=0.7,
        )

        # Annotate p-value a bit above the error bar
        plt.annotate(
            text=f"p = {p_value:.3f}" if p_value >= 0.001 else "p < 0.001",
            xy=(effect, feature),
            xytext=(0, 7),  # Shift upward in points
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="black",
            fontstyle="italic",
        )

    # Aesthetics
    plt.axvline(x=0, color="blue", linestyle="dotted", alpha=0.5)
    plt.title(f"{target_title}\nEffect of {treated_title}")
    plt.xlabel("% Effect (Test vs. Control)")
    plt.tight_layout()
    plt.grid(False)

    out_dir = Path(dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = "" if experiment_id is None else f"{experiment_id}_"
    # avoid slashes in filename
    safe_title = str(treated_title).replace(os.sep, "_")
    out_path = out_dir / f"{prefix}{safe_title}.png"

    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    return out_path


################################### EXPERIMENT SUMMARY ###################################


def _next_experiment_id(csv_path: Path) -> int:
    """Return next experiment_id based on existing CSV contents.

    If the file does not exist, is empty, or has no numeric experiment_id,
    returns 1.
    """
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return 1

    try:
        existing = pd.read_csv(csv_path, usecols=["experiment_id"])
    except Exception:
        # If file is malformed or missing the column, start fresh.
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
    csv_path: str | Path = "experiment/results/experiment_summaries.csv",
    experiment_id: str | int | None = None,
    exposure: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Summarize balance and ATEs for a single experiment and append to CSV.

    This CSV is the single source of logging for experiments. If experiment_id
    is None, it is automatically assigned as max(existing) + 1.

    The log includes:
      * Balance percentages per group.
      * Flags for structural imbalance and negative control issues.
      * Compact string fields listing failing features and significant neg. targets.
    """
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

    # Auto-assign experiment_id if not provided.
    if experiment_id is None:
        experiment_id = _next_experiment_id(csv_path)

    def pct(numer: int, denom: int) -> float:
        """Compute percentage or NaN if denom is 0."""
        return float(np.nan) if denom == 0 else 100.0 * numer / denom

    # (A) Balance metrics
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

    # Handy counts
    n_struct_unbal = len(failing_struct)
    n_direct_unbal = len(failing_direct)
    n_partial_unbal = len(failing_partial)

    # (B) ATE / significance
    ate_conf = ate_table[ate_table["outcome"].isin(union_conf)].copy()
    ate_neg = ate_table[ate_table["outcome"].isin(neg_set)].copy()

    sig_conf = ate_conf.loc[
        ate_conf["is_significant_abs"] == True, "outcome"  # noqa: E712
    ].astype(str)
    sig_conf_list = sorted(sig_conf.unique().tolist())

    denom_conf = len(ate_conf)
    numer_conf_not_sig = int(
        (ate_conf["is_significant_abs"] == False).sum()  # noqa: E712
    )
    pct_conf_not_sig = pct(numer_conf_not_sig, denom_conf)

    sig_neg = ate_neg.loc[
        ate_neg["is_significant_abs"] == True, "outcome"  # noqa: E712
    ].astype(str)
    sig_neg_list = sorted(sig_neg.unique().tolist())

    denom_neg = len(ate_neg)
    numer_neg_not_sig = int(
        (ate_neg["is_significant_abs"] == False).sum()  # noqa: E712
    )
    pct_neg_not_sig = pct(numer_neg_not_sig, denom_neg)

    # High-level flags
    flag_struct_unbalanced = n_struct_unbal > 0
    flag_direct_unbalanced = n_direct_unbal > 0
    flag_partial_unbalanced = n_partial_unbal > 0
    flag_neg_controls_issue = len(sig_neg_list) > 0

    # Compact human-readable summary message (for quick scanning of the CSV)
    log_message = (
        f"exp_id={experiment_id}, exposure={exposure}, "
        f"all_bal_0.10={pct_all_bal10:.1f}%, "
        f"struct_bal_0.05={pct_struct_bal05:.1f}% (unbal={n_struct_unbal}), "
        f"direct_bal_0.05={pct_direct_bal05:.1f}% (unbal={n_direct_unbal}), "
        f"partial_bal_0.10={pct_partial_bal10:.1f}% (unbal={n_partial_unbal}), "
        f"neg_controls_sig={len(sig_neg_list)}"
    )

    row: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment_id": experiment_id,
        "exposure": exposure,
        # High-level text summary:
        "log_message": log_message,
        # Balance percentages:
        "pct_all_features_balanced_0_10_med": pct_all_bal10,
        "pct_structural_balanced_0_05_med": pct_struct_bal05,
        "pct_direct_balanced_0_05_med": pct_direct_bal05,
        "pct_partial_balanced_0_10_med": pct_partial_bal10,
        # Counts:
        "n_features_all": n_all,
        "n_structural_features": n_struct_total,
        "n_structural_unbalanced_0_05": n_struct_unbal,
        "n_direct_features": n_direct_total,
        "n_direct_unbalanced_0_05": n_direct_unbal,
        "n_partial_features": n_partial_total,
        "n_partial_unbalanced_0_10": n_partial_unbal,
        # Failing feature lists:
        "failing_features": ";".join(failing_all),
        "failing_structural_features": ";".join(failing_struct),
        "failing_direct_features": ";".join(failing_direct),
        "failing_partial_features": ";".join(failing_partial),
        # ATE / neg controls:
        "significant_confounder_outcomes": ";".join(sig_conf_list),
        "pct_confounder_outcomes_not_significant": pct_conf_not_sig,
        "significant_negative_targets": ";".join(sig_neg_list),
        "pct_negative_targets_not_significant": pct_neg_not_sig,
        # Flags:
        "flag_structural_unbalanced": flag_struct_unbalanced,
        "flag_direct_unbalanced": flag_direct_unbalanced,
        "flag_partial_unbalanced": flag_partial_unbalanced,
        "flag_negative_controls_issue": flag_neg_controls_issue,
    }

    # Merge in extra meta (hyperparameters etc.)
    for k, v in extra_meta.items():
        if k not in row:
            row[k] = v

    out_df = pd.DataFrame([row])

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = not csv_path.exists() or csv_path.stat().st_size == 0
    out_df.to_csv(csv_path, mode="a", index=False, header=header)

    return out_df


################################### FEATURE SELECTION ###################################


def generate_features(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    additional_features: Sequence[str],
    n_features: int = 25,
) -> list[str]:
    """Select top features by SHAP importance plus forced extras."""
    important_features = list(
        X.columns[np.argsort(np.abs(shap_values).mean(axis=0))]
    )
    important_features.reverse()
    feature_names: list[str] = important_features[:n_features]

    for feature in additional_features:
        if feature not in feature_names:
            feature_names.append(feature)

    return feature_names


################################### MAIN EXPERIMENT RUNNER ###################################


def run_experiment(
    config: Mapping[str, Any],
    variable_config: Mapping[str, Any],
    df: pd.DataFrame,
    kwargs: Mapping[str, Any],
    X: pd.DataFrame,
    shap_values: np.ndarray,
    save_results: bool = False,
    log_experiment: bool = True,
    df_folder_path: str = "experiment/results/dataframes",
    errorbar_folder_path: str = "experiment/results/final",
) -> int | pd.DataFrame | None:
    """Run a single IPW + bootstrap experiment.

    Logging is done via experiment_summaries.csv; experiment_id is
    auto-assigned from that CSV and reused for chart filenames.
    """
    prop_df = prepare_ipw_dataset(
        df=df,
        treat_col=config["t"],
        e_col=config["calibration"],
        sleep_targets=variable_config["sleep_targets"],
        q=config["q"],
        stabilize=False,
        clip=None,
        dropna_targets=True,
        verbose=False,
    )

    confounders_for_asmd_report = generate_features(
        X,
        shap_values,
        variable_config["confounders"],
        n_features=10,
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

    # Case 1: only save ATE/ASMD + chart, no logging row
    if save_results and not log_experiment:
        plot_error_bars(
            df_bootstrap,
            treated_title=str(kwargs["exposure_baseline"]),
            experiment_id=None,  # no experiment_id prefix in this mode
            target_title="Treatment",
            dir=errorbar_folder_path,
        )
        filename = str(kwargs["exposure_baseline"])
        Path(df_folder_path).mkdir(parents=True, exist_ok=True)
        bal_after.to_csv(f"{df_folder_path}/{filename}_ate.csv")
        ate_table.to_csv(f"{df_folder_path}/{filename}_asmd.csv")
        return 1

    # Case 2: log_experiment (with or without save_results)
    if log_experiment:
        log_csv_path = Path("experiment/results/experiment_summaries.csv")
        experiment_id = _next_experiment_id(log_csv_path)

        # Chart with the same experiment_id as the log row:
        plot_error_bars(
            df_bootstrap,
            treated_title=str(kwargs["exposure_baseline"]),
            experiment_id=experiment_id,
            target_title="Treatment",
            dir=errorbar_folder_path,
        )

        extra_meta = {
            "exposure_baseline": kwargs.get("exposure_baseline"),
            "alpha": config.get("alpha"),
            "q": config.get("q"),
            "clip_low": (config.get("clip") or (None, None))[0],
            "clip_high": (config.get("clip") or (None, None))[1],
            "stabilize": config.get("stabilize"),
            "n_iter": config.get("n_iter"),
        }

        summary = summarize_experiment(
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

        # optionally still save ASMD/ATE tables when save_results=True
        if save_results:
            filename = str(kwargs["exposure_baseline"])
            Path(df_folder_path).mkdir(parents=True, exist_ok=True)
            bal_after.to_csv(f"{df_folder_path}/{filename}_ate.csv")
            ate_table.to_csv(f"{df_folder_path}/{filename}_asmd.csv")

        # quick console summary
        s = summary.iloc[0]
        print(
            f"[exp_id={s['experiment_id']}] exposure={s['exposure']} | "
            f"all_bal_0.10={s['pct_all_features_balanced_0_10_med']:.1f}% | "
            f"struct_bal_0.05={s['pct_structural_balanced_0_05_med']:.1f}% "
            f"(unbal={s['n_structural_unbalanced_0_05']}) | "
            f"neg_controls_sig={len(s['significant_negative_targets'].split(';')) if s['significant_negative_targets'] else 0}"
        )

        return summary

    # Case 3: neither save_results nor log_experiment -> just compute, no side effects
    return None
