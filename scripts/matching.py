from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests

# ============================================================
# Utilities
# ============================================================

def _logit(p, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _smd(x_t, x_c) -> float:
    x_t = np.asarray(x_t, dtype=float)
    x_c = np.asarray(x_c, dtype=float)

    mt, mc = np.nanmean(x_t), np.nanmean(x_c)
    vt, vc = np.nanvar(x_t, ddof=1), np.nanvar(x_c, ddof=1)

    denom = np.sqrt((vt + vc) / 2.0)
    return (mt - mc) / denom if denom > 0 else np.nan


# ============================================================
# Matching
# ============================================================

def match_ps_plus_covariates(
    df: pd.DataFrame,
    treat_col: str = "treated",
    ps_col: str = "calibrated_scores_logreg",
    match_covariates: list[str] | None = None,
    caliper: float = 0.2,
    replace: bool = False,
    random_state: int = 0,
) -> pd.DataFrame:

    match_covariates = match_covariates or []
    rng = np.random.default_rng(random_state)

    d = df.dropna(subset=[treat_col, ps_col, *match_covariates]).copy()
    d["_logit_ps"] = _logit(d[ps_col])

    treated = d[d[treat_col] == 1]
    control = d[d[treat_col] == 0]

    if treated.empty or control.empty:
        raise ValueError("Need both treated==1 and treated==0 rows.")

    ps_sd = d["_logit_ps"].std(ddof=1)
    caliper_abs = caliper * ps_sd if caliper is not None else None

    feat_cols = ["_logit_ps", *match_covariates]
    X = d[feat_cols].astype(float).to_numpy()

    scaler = StandardScaler().fit(X)
    Xt = scaler.transform(treated[feat_cols])
    Xc = scaler.transform(control[feat_cols])

    nn = NearestNeighbors(n_neighbors=1).fit(Xc)

    treated_idx = treated.index.to_numpy()
    control_idx = control.index.to_numpy()

    order = rng.permutation(len(treated_idx))

    used_controls = set()
    matches = []

    for pos in order:
        ti = treated_idx[pos]
        _, idx = nn.kneighbors(Xt[pos:pos + 1])
        ci = control_idx[int(idx[0, 0])]

        if caliper_abs is not None:
            if abs(d.loc[ti, "_logit_ps"] - d.loc[ci, "_logit_ps"]) > caliper_abs:
                continue

        if not replace and ci in used_controls:
            continue

        used_controls.add(ci)
        matches.append((ti, ci))

    if not matches:
        raise ValueError("No matches found. Try increasing caliper or replace=True.")

    t_rows = d.loc[[t for t, _ in matches]].assign(
        pair_id=range(len(matches)),
        _role="treated",
    )
    c_rows = d.loc[[c for _, c in matches]].assign(
        pair_id=range(len(matches)),
        _role="control",
    )

    return pd.concat([t_rows, c_rows], ignore_index=True)


# ============================================================
# Balance diagnostics
# ============================================================

def balance_table(
    df: pd.DataFrame,
    matched_df: pd.DataFrame,
    treat_col: str,
    confounders: list[str],
    *,
    asmd_thresh: float = 0.1,
) -> tuple[pd.DataFrame, list[str]]:

    pre = df.dropna(subset=[treat_col])
    pre_t = pre[pre[treat_col] == 1]
    pre_c = pre[pre[treat_col] == 0]

    post_t = matched_df[matched_df["_role"] == "treated"]
    post_c = matched_df[matched_df["_role"] == "control"]

    rows = []
    for col in confounders:
        if col not in df.columns:
            continue

        smd_pre = _smd(pre_t[col], pre_c[col])
        smd_post = _smd(post_t[col], post_c[col])

        rows.append({
            "covariate": col,
            "SMD_pre": smd_pre,
            "SMD_post": smd_post,
            "abs_SMD_pre": abs(smd_pre) if np.isfinite(smd_pre) else np.nan,
            "abs_SMD_post": abs(smd_post) if np.isfinite(smd_post) else np.nan,
        })

    bal = pd.DataFrame(rows).sort_values("abs_SMD_post", ascending=False)

    fails = bal.loc[
        (bal["abs_SMD_post"] > asmd_thresh) & np.isfinite(bal["abs_SMD_post"]),
        "covariate"
    ].tolist()

    return bal, fails


def balance_summary(bal: pd.DataFrame, *, asmd_thresh: float = 0.1) -> dict:
    s = bal["abs_SMD_post"].dropna()
    return {
        "mean_abs_SMD_post": float(s.mean()) if len(s) else np.nan,
        "p75_abs_SMD_post": float(s.quantile(0.75)) if len(s) else np.nan,
        "max_abs_SMD_post": float(s.max()) if len(s) else np.nan,
        "median_abs_SMD_post_below_thresh": bool(len(s) and s.median() < asmd_thresh),
        "pct_below_thresh": float((s <= asmd_thresh).mean() * 100) if len(s) else np.nan,
    }


# ============================================================
# Effect estimation (paired CUPED + t-test)
# ============================================================

def outcome_effect_cuped_ttest(
    matched_df: pd.DataFrame,
    outcome: str,
    alpha: float = 0.05,
    min_pairs_for_cuped: int = 10,
) -> dict:

    baseline = outcome.split("_target_day")[0]

    dy, dx, yc = [], [], []

    for _, g in matched_df.groupby("pair_id"):
        try:
            yt = g.loc[g["_role"] == "treated", outcome].values[0]
            yc_ = g.loc[g["_role"] == "control", outcome].values[0]
        except IndexError:
            continue

        if not (np.isfinite(yt) and np.isfinite(yc_)):
            continue

        dy.append(yt - yc_)
        yc.append(yc_)

        if baseline in g.columns:
            xt = g.loc[g["_role"] == "treated", baseline].values
            xc = g.loc[g["_role"] == "control", baseline].values
            dx.append(xt[0] - xc[0] if len(xt) == len(xc) == 1 else np.nan)
        else:
            dx.append(np.nan)

    dy = np.asarray(dy, float)
    dx = np.asarray(dx, float)
    yc = np.asarray(yc, float)

    n = dy.size
    if n < 2:
        return {
            "n_pairs_used": n,
            "ATE": np.nan,
            "CI_low": np.nan,
            "CI_high": np.nan,
            "p_value": np.nan,
            "ATE_pct": np.nan,
            "CI_low_pct": np.nan,
            "CI_high_pct": np.nan,
            "p_value_pct": np.nan,
            "ref_control_mean": np.nan,
            "used_cuped": False,
            "cuped_theta": np.nan,
            "baseline_col": baseline,
        }

    dy_adj = dy.copy()
    used_cuped = False
    theta = np.nan

    ok = np.isfinite(dx)
    if ok.sum() >= min_pairs_for_cuped and np.var(dx[ok], ddof=1) > 1e-12:
        theta = np.cov(dy[ok], dx[ok], ddof=1)[0, 1] / np.var(dx[ok], ddof=1)
        dy_adj[ok] -= theta * dx[ok]
        used_cuped = True

    ate = dy_adj.mean()
    sd = dy_adj.std(ddof=1)
    se = sd / np.sqrt(n)

    tcrit = stats.t.ppf(1 - alpha / 2, n - 1)
    ci_low = ate - tcrit * se
    ci_high = ate + tcrit * se

    tstat = ate / se if se > 0 else np.nan
    p = 2 * stats.t.sf(abs(tstat), n - 1) if np.isfinite(tstat) else np.nan

    ref = yc.mean()
    ate_pct = 100 * ate / ref if ref != 0 else np.nan

    return {
        "n_pairs_used": n,
        "ATE": float(ate),
        "CI_low": float(ci_low),
        "CI_high": float(ci_high),
        "p_value": float(p),
        "ATE_pct": float(ate_pct),
        "CI_low_pct": 100 * ci_low / ref if ref != 0 else np.nan,
        "CI_high_pct": 100 * ci_high / ref if ref != 0 else np.nan,
        "p_value_pct": float(p),
        "ref_control_mean": float(ref),
        "used_cuped": used_cuped,
        "cuped_theta": float(theta) if np.isfinite(theta) else np.nan,
        "baseline_col": baseline,
    }


# ============================================================
# Full pipeline
# ============================================================

def estimate_ate_matching_pipeline_cuped(
    df: pd.DataFrame,
    variable_config: dict,
    treat_col: str = "treated",
    ps_col: str = "calibrated_scores_logreg",
    match_covariates: list[str] | None = None,
    caliper: float = 0.2,
    replace: bool = False,
    alpha: float = 0.05,
    seed: int = 0,
    smd_threshold: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:

    match_covariates = match_covariates or ["age", "gender", "bmi"]

    confounders = list(variable_config["confounders"])
    outcomes = list(variable_config["sleep_targets"])

    matched_df = match_ps_plus_covariates(
        df=df,
        treat_col=treat_col,
        ps_col=ps_col,
        match_covariates=match_covariates,
        caliper=caliper,
        replace=replace,
        random_state=seed,
    )

    bal_df, asmd_fails = balance_table(
        df=df,
        matched_df=matched_df,
        treat_col=treat_col,
        confounders=confounders,
        asmd_thresh=smd_threshold,
    )
    bal_sum = balance_summary(bal_df, asmd_thresh=smd_threshold)

    rows = []
    for out in outcomes:
        eff = outcome_effect_cuped_ttest(
            matched_df.dropna(subset=[out]),
            outcome=out,
            alpha=alpha,
        )
        rows.append({"outcome": out, **eff})

    results_df = pd.DataFrame(rows)

    for col in ["p_value", "p_value_pct"]:
        p = results_df[col].to_numpy(float)
        ok = np.isfinite(p)
        results_df[f"{col}_fdr_bh"] = np.nan
        if ok.sum():
            results_df.loc[ok, f"{col}_fdr_bh"] = multipletests(p[ok], method="fdr_bh")[1]

    n_pairs = matched_df["pair_id"].nunique()
    n_t = (matched_df["_role"] == "treated").sum()
    n_c = (matched_df["_role"] == "control").sum()
    unique_controls = matched_df.loc[matched_df["_role"] == "control"].drop_duplicates().shape[0]

    meta = {
        "n_pairs": int(n_pairs),
        "n_treated_matched": int(n_t),
        "n_controls_matched": int(n_c),
        "unique_controls_used": int(unique_controls),
        "controls_reused": bool(unique_controls < n_c),
        **bal_sum,
        "median_abs_SMD_post_OK": bal_sum["median_abs_SMD_post_below_thresh"],
        "asmd_fail_count": len(asmd_fails),
        "asmd_fail_features": "; ".join(asmd_fails),
        "caliper": caliper,
        "replace": replace,
        "match_covariates": "; ".join(match_covariates),
        "smd_threshold": smd_threshold,
        "seed": seed,
    }

    return results_df, matched_df, bal_df, meta
