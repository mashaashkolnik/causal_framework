from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests
from scipy import stats


# -----------------------------
# Utilities
# -----------------------------
def _logit(p, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))

def _smd(x_t, x_c) -> float:
    """Standardized mean difference (continuous or 0/1)."""
    x_t = np.asarray(x_t, dtype=float)
    x_c = np.asarray(x_c, dtype=float)
    mt, mc = np.nanmean(x_t), np.nanmean(x_c)
    vt, vc = np.nanvar(x_t, ddof=1), np.nanvar(x_c, ddof=1)
    denom = np.sqrt((vt + vc) / 2.0)
    return (mt - mc) / denom if denom > 0 else np.nan


# -----------------------------
# Matching: logit(PS) + key covariates, with PS caliper
# -----------------------------
def match_ps_plus_covariates(
    df: pd.DataFrame,
    treat_col: str = "treated",
    ps_col: str = "calibrated_scores_logreg",
    match_covariates: list[str] | None = None,  # e.g. ["age", "gender", "bmi"]
    caliper: float = 0.2,                       # SD units of logit(PS)
    replace: bool = False,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    1:1 NN matching on standardized [logit(PS) + match_covariates],
    subject to a caliper on |logit(PS_t) - logit(PS_c)|.
    Returns matched_df with: pair_id, _role, _logit_ps.
    """
    if match_covariates is None:
        match_covariates = []

    rng = np.random.default_rng(random_state)

    needed = [treat_col, ps_col] + match_covariates
    d = df.dropna(subset=needed).copy()

    d["_logit_ps"] = _logit(d[ps_col].values)

    treated = d[d[treat_col] == 1].copy()
    control = d[d[treat_col] == 0].copy()
    if treated.empty or control.empty:
        raise ValueError("Need both treated==1 and treated==0 rows.")

    ps_sd = d["_logit_ps"].std(ddof=1)
    caliper_abs = (caliper * ps_sd) if (caliper is not None) else None

    feat_cols = ["_logit_ps"] + match_covariates
    X_t = treated[feat_cols].astype(float).to_numpy()
    X_c = control[feat_cols].astype(float).to_numpy()

    scaler = StandardScaler()
    scaler.fit(np.vstack([X_t, X_c]))
    X_tz = scaler.transform(X_t)
    X_cz = scaler.transform(X_c)

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(X_cz)

    treated_idx = treated.index.to_numpy()
    control_idx = control.index.to_numpy()

    order = np.arange(len(treated_idx))
    rng.shuffle(order)

    used_controls: set[int] = set()
    matches: list[tuple[int, int]] = []

    for pos in order:
        ti = treated_idx[pos]
        dist, idx = nn.kneighbors(X_tz[pos:pos + 1])
        ci = control_idx[int(idx[0][0])]

        # PS caliper check
        if caliper_abs is not None:
            if abs(d.loc[ti, "_logit_ps"] - d.loc[ci, "_logit_ps"]) > caliper_abs:
                continue

        if (not replace) and (ci in used_controls):
            continue

        used_controls.add(ci)
        matches.append((ti, ci))

    if not matches:
        raise ValueError("No matches found. Try caliper=0.3/0.5 or replace=True.")

    t_rows = d.loc[[ti for ti, _ in matches]].copy()
    c_rows = d.loc[[ci for _, ci in matches]].copy()

    t_rows["pair_id"] = np.arange(len(matches))
    c_rows["pair_id"] = np.arange(len(matches))
    t_rows["_role"] = "treated"
    c_rows["_role"] = "control"

    matched_df = pd.concat([t_rows, c_rows], ignore_index=True)
    return matched_df


# -----------------------------
# Balance diagnostics (SMD) + summary (median |SMD|)
# -----------------------------
import numpy as np
import pandas as pd


def balance_table(
    df: pd.DataFrame,
    matched_df: pd.DataFrame,
    treat_col: str,
    confounders: list[str],
    *,
    asmd_thresh: float = 0.1,
) -> tuple[pd.DataFrame, list[str]]:
    pre = df.dropna(subset=[treat_col]).copy()
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
        abs_pre = abs(smd_pre) if np.isfinite(smd_pre) else np.nan
        abs_post = abs(smd_post) if np.isfinite(smd_post) else np.nan

        rows.append(
            {
                "covariate": col,
                "SMD_pre": smd_pre,
                "SMD_post": smd_post,
                "abs_SMD_pre": abs_pre,
                "abs_SMD_post": abs_post,
            }
        )

    bal = pd.DataFrame(rows).sort_values("abs_SMD_post", ascending=False)

    s_post = bal["abs_SMD_post"]
    fails = bal.loc[np.isfinite(s_post) & (s_post > asmd_thresh), "covariate"].tolist()

    return bal, fails


def balance_summary(bal: pd.DataFrame, *, asmd_thresh: float = 0.1) -> dict:
    s = bal["abs_SMD_post"].dropna()
    return {
        "mean_abs_SMD_post": float(s.mean()) if len(s) else np.nan,
        "p75_abs_SMD_post": float(s.quantile(0.75)) if len(s) else np.nan,
        "max_abs_SMD_post": float(s.max()) if len(s) else np.nan,
        "median_abs_SMD_post_below_thresh": bool(len(s) and (s.median() < asmd_thresh)),
        "pct_below_thresh": float((s <= asmd_thresh).mean() * 100.0) if len(s) else np.nan,
    }



# -----------------------------
# Effect estimation: paired CUPED + one-sample t-test
# Baseline column rule: outcome.split("_target_day")[0]
# -----------------------------
def outcome_effect_cuped_ttest(
    matched_df: pd.DataFrame,
    outcome: str,
    alpha: float = 0.05,
    min_pairs_for_cuped: int = 10,
) -> dict:
    baseline = outcome.split("_target_day")[0]

    diffs_y = []
    diffs_x = []
    ctrl_y = []

    for _, g in matched_df.groupby("pair_id"):
        yt = g.loc[g["_role"] == "treated", outcome].to_numpy()
        yc = g.loc[g["_role"] == "control", outcome].to_numpy()
        if len(yt) != 1 or len(yc) != 1:
            continue
        if not (np.isfinite(yt[0]) and np.isfinite(yc[0])):
            continue

        dy = float(yt[0] - yc[0])
        diffs_y.append(dy)
        ctrl_y.append(float(yc[0]))

        # baseline diffs (optional per pair)
        if baseline in matched_df.columns:
            xt = g.loc[g["_role"] == "treated", baseline].to_numpy()
            xc = g.loc[g["_role"] == "control", baseline].to_numpy()
            if len(xt) == 1 and len(xc) == 1 and np.isfinite(xt[0]) and np.isfinite(xc[0]):
                diffs_x.append(float(xt[0] - xc[0]))
            else:
                diffs_x.append(np.nan)
        else:
            diffs_x.append(np.nan)

    dy = np.asarray(diffs_y, dtype=float)
    dx = np.asarray(diffs_x, dtype=float)
    ctrl_y = np.asarray(ctrl_y, dtype=float)

    n = dy.size
    if n < 2:
        return {
            "n_pairs_used": int(n),
            "ATE": np.nan, "CI_low": np.nan, "CI_high": np.nan, "p_value": np.nan,
            "ATE_pct": np.nan, "CI_low_pct": np.nan, "CI_high_pct": np.nan, "p_value_pct": np.nan,
            "ref_control_mean": np.nan,
            "used_cuped": False,
            "cuped_theta": np.nan,
            "baseline_col": baseline,
        }

    # CUPED on available baseline diffs
    ok = np.isfinite(dx)
    used_cuped = False
    theta = np.nan
    dy_adj = dy.copy()

    if ok.sum() >= min_pairs_for_cuped and np.nanvar(dx[ok]) > 1e-12:
        cov = np.cov(dy[ok], dx[ok], ddof=1)[0, 1]
        var = np.var(dx[ok], ddof=1)
        theta = float(cov / var)
        dy_adj[ok] = dy[ok] - theta * dx[ok]
        used_cuped = True

    ate = float(np.mean(dy_adj))

    sd = float(np.std(dy_adj, ddof=1))
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)

    ci_low = ate - tcrit * se
    ci_high = ate + tcrit * se

    tstat = ate / se if se > 0 else np.nan
    p = float(2 * stats.t.sf(np.abs(tstat), df=n - 1)) if np.isfinite(tstat) else np.nan

    # % effect vs matched controls mean
    ref = float(np.mean(ctrl_y))
    if np.isfinite(ref) and ref != 0:
        ate_pct = 100.0 * ate / ref
        ci_low_pct = 100.0 * ci_low / ref
        ci_high_pct = 100.0 * ci_high / ref
    else:
        ate_pct = ci_low_pct = ci_high_pct = np.nan

    return {
        "n_pairs_used": int(n),
        "ATE": float(ate),
        "CI_low": float(ci_low),
        "CI_high": float(ci_high),
        "p_value": float(p),
        "ATE_pct": float(ate_pct),
        "CI_low_pct": float(ci_low_pct),
        "CI_high_pct": float(ci_high_pct),
        "p_value_pct": float(p),
        "ref_control_mean": float(ref),
        "used_cuped": bool(used_cuped),
        "cuped_theta": float(theta) if np.isfinite(theta) else np.nan,
        "baseline_col": baseline,
    }


# -----------------------------
# Full pipeline (matching + balance + CUPED p-values + BH correction)
# -----------------------------
def estimate_ate_matching_pipeline_cuped(
    df: pd.DataFrame,
    variable_config: dict,
    treat_col: str = "treated",
    ps_col: str = "calibrated_scores_logreg",
    match_covariates: list[str] | None = None,  # e.g. ["age", "gender", "bmi"]
    caliper: float = 0.2,
    replace: bool = False,
    alpha: float = 0.05,
    seed: int = 0,
    smd_threshold: float = 0.1,  # median |SMD| threshold you care about
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:

    if match_covariates is None:
        match_covariates = ["age", "gender", "bmi"]  # change if needed

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
    
    # Balance (UPDATED)
    bal_df, asmd_fails = balance_table(
        df=df,
        matched_df=matched_df,
        treat_col=treat_col,
        confounders=confounders,
        asmd_thresh=smd_threshold,
    )
    bal_sum = balance_summary(bal_df, asmd_thresh=smd_threshold)

    # Balance
    #bal_df = balance_table(df=df, matched_df=matched_df, treat_col=treat_col, confounders=confounders)
    #al_sum = balance_summary(bal_df)

    # Effects per outcome
    rows = []
    for out in outcomes:
        md = matched_df.dropna(subset=[out]).copy()
        eff = outcome_effect_cuped_ttest(md, outcome=out, alpha=alpha)
        rows.append({"outcome": out, **eff})

    results_df = pd.DataFrame(rows)

    # BH correction on raw p-values (absolute scale)
    p = results_df["p_value"].to_numpy(dtype=float)
    ok = np.isfinite(p)
    results_df["p_value_fdr_bh"] = np.nan
    if ok.sum() > 0:
        results_df.loc[ok, "p_value_fdr_bh"] = multipletests(p[ok], method="fdr_bh")[1]

    # BH correction on % p-values (same ordering typically, but keep explicit)
    p2 = results_df["p_value_pct"].to_numpy(dtype=float)
    ok2 = np.isfinite(p2)
    results_df["p_value_pct_fdr_bh"] = np.nan
    if ok2.sum() > 0:
        results_df.loc[ok2, "p_value_pct_fdr_bh"] = multipletests(p2[ok2], method="fdr_bh")[1]

    # Meta / sanity checks
    n_pairs = int(matched_df["pair_id"].nunique())
    n_t = int((matched_df["_role"] == "treated").sum())
    n_c = int((matched_df["_role"] == "control").sum())
    unique_controls = int(matched_df.loc[matched_df["_role"] == "control"].drop_duplicates().shape[0])

    meta = {
        "n_pairs": n_pairs,
        "n_treated_matched": n_t,
        "n_controls_matched": n_c,
        "unique_controls_used": unique_controls,
        "controls_reused": bool(unique_controls < n_c),
        **bal_sum,
        "median_abs_SMD_post_OK": bool(
            np.isfinite(bal_sum.get("median_abs_SMD_post", np.nan))
            and bal_sum["median_abs_SMD_post"] < smd_threshold
        ),
        "asmd_fail_count": int(len(asmd_fails)),
        "asmd_fail_features": "; ".join(asmd_fails) if asmd_fails else "",
        "caliper": caliper,
        "replace": replace,
        "match_covariates": "; ".join(match_covariates),
        "smd_threshold": smd_threshold,
        "seed": seed,
    }

    return results_df, matched_df, bal_df, meta


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

def matching_plot_error_bars(
    df_bootstrap,  # results_df from matching pipeline
    treated_title: str,
    dir: str,
    experiment_id: int | str | None = None,
    target_title: str = "Treatment",
    alpha: float = 0.05,
    out_dir = None,
) -> Path | None:

    df_bootstrap = df_bootstrap.copy()

    # ------------------ pick which columns to plot ------------------
    use_pct = all(c in df_bootstrap.columns for c in ["ATE_pct", "CI_low_pct", "CI_high_pct", "p_value_pct"])
    use_abs = all(c in df_bootstrap.columns for c in ["ATE", "CI_low", "CI_high", "p_value"])

    if not (use_pct or use_abs):
        raise KeyError(
            "df_bootstrap must contain either "
            "['ATE_pct','CI_low_pct','CI_high_pct','p_value_pct'] "
            "or ['ATE','CI_low','CI_high','p_value']."
        )

    if use_pct:
        eff_col, lo_col, hi_col, p_col = "ATE_pct", "CI_low_pct", "CI_high_pct", "p_value_pct"
        x_label = "Effect (% difference vs matched controls)"
        plot_mode = "pct"
    else:
        eff_col, lo_col, hi_col, p_col = "ATE", "CI_low", "CI_high", "p_value"
        x_label = "Effect (absolute difference)"
        plot_mode = "abs"

    # ------------------ try to find a control mean (to compute both abs + %) ------------------
    control_mean_candidates = [
        "control_mean", "mean_control", "matched_control_mean", "control_mean_matched",
        "y0_mean", "outcome_mean_control"
    ]
    control_mean_col = next((c for c in control_mean_candidates if c in df_bootstrap.columns), None)

    # Create unified annotation columns: ATE_abs and Effect_pct
    # df_bootstrap["ATE"] = np.nan
    df_bootstrap["Effect_pct_for_annot"] = np.nan

    if plot_mode == "abs":
        # We already have abs
        df_bootstrap["ATE"] = df_bootstrap["ATE"].astype(float)
        # If control mean exists, compute %
        if control_mean_col is not None:
            denom = df_bootstrap[control_mean_col].astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                df_bootstrap["Effect_pct_for_annot"] = 100.0 * df_bootstrap["ATE_abs_for_annot"] / denom
    else:
        # We already have %
        df_bootstrap["Effect_pct_for_annot"] = df_bootstrap["ATE_pct"].astype(float)
        # If control mean exists, compute abs
        if control_mean_col is not None:
            denom = df_bootstrap[control_mean_col].astype(float)
            df_bootstrap["ATE_abs_for_annot"] = (df_bootstrap["Effect_pct_for_annot"] / 100.0) * denom

    # ------------------ BH correction ------------------
    raw_pvals = df_bootstrap[p_col].to_numpy(dtype=float)

    ok = np.isfinite(raw_pvals)
    adj = np.full_like(raw_pvals, np.nan, dtype=float)
    if ok.sum() > 0:
        adj[ok] = multipletests(
            raw_pvals[ok],
            alpha=alpha,
            method="fdr_bh",
            maxiter=1,
            is_sorted=False,
            returnsorted=False,
        )[1]

    df_bootstrap["p_value_adj_bh"] = adj
    df_bootstrap["is_significant_bh"] = df_bootstrap["p_value_adj_bh"] < alpha

    # ------------------ style params ------------------
    point_size: float = 32.0
    ci_line_width: float = 5.0
    cap_height: float = 0.09
    font_size: int = 9
    y_offset: float = 0.12
    zero_line = True

    def _color_for_sig(is_sig: bool) -> str:
        return "#E31A1C" if is_sig else "#9E9E9E"

    def _fmt_p(p: float) -> str:
        if not np.isfinite(p):
            return "p = NA"
        if p < 0.001:
            return "p < 0.001"
        return f"p = {p:.3f}"

    def _fmt_abs(x: float) -> str:
        if not np.isfinite(x):
            return "ATE = NA"
        # tweak precision if you want
        return f"ATE = {x:+.3g}"

    def _fmt_pct(x: float) -> str:
        if not np.isfinite(x):
            return "Effect = NA"
        return f"Effect = {x:+.1f}%"

    # ------------------ prep y axis ------------------
    outcomes = df_bootstrap["outcome"].astype(str).tolist()
    m = len(outcomes)
    y = np.arange(m)

    # ------------------ figure ------------------
    fig, ax = plt.subplots(figsize=(10, 8))

    if zero_line:
        ax.axvline(0.0, linestyle="--", linewidth=1.25, color="#8a8a8a", alpha=0.85, zorder=0)

    ax.grid(False)

    eff = df_bootstrap[eff_col].to_numpy(dtype=float)
    lo = df_bootstrap[lo_col].to_numpy(dtype=float)
    hi = df_bootstrap[hi_col].to_numpy(dtype=float)
    p_adj = df_bootstrap["p_value_adj_bh"].to_numpy(dtype=float)
    is_sig = df_bootstrap["is_significant_bh"].to_numpy(dtype=bool)

    # ------------------ draw CIs + caps ------------------
    for j in range(m):
        if not (np.isfinite(lo[j]) and np.isfinite(hi[j]) and np.isfinite(eff[j])):
            continue
        c = _color_for_sig(bool(is_sig[j]))
        ax.hlines(y[j], lo[j], hi[j], lw=ci_line_width, color=c, zorder=2)
        ax.plot([lo[j], lo[j]], [y[j] - cap_height, y[j] + cap_height], color=c, lw=ci_line_width, zorder=2)
        ax.plot([hi[j], hi[j]], [y[j] - cap_height, y[j] + cap_height], color=c, lw=ci_line_width, zorder=2)

    # point estimate (always black)
    ax.scatter(eff, y, s=point_size, color="black", zorder=3)

    # ------------------ annotations (BH-only p, plus ATE abs + effect %) ------------------
    abs_ate = df_bootstrap["ATE"].to_numpy(dtype=float)
    pct_eff = df_bootstrap["Effect_pct_for_annot"].to_numpy(dtype=float)

    for j in range(m):
        if not (np.isfinite(eff[j]) and np.isfinite(p_adj[j])):
            continue

        label = f"{_fmt_p(p_adj[j])} | {_fmt_abs(abs_ate[j])} | {_fmt_pct(pct_eff[j])}"

        ax.text(
            eff[j],
            y[j] - y_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=font_size,
            color="#333",
            zorder=4,
            clip_on=False,
        )

    # ------------------ axes cosmetics ------------------
    ax.set_title(f"{target_title}\nEffect of {treated_title}", fontsize=14, pad=10)
    ax.set_xlabel(x_label, fontsize=12)

    ax.set_yticks(y)
    ax.set_yticklabels(outcomes, fontsize=12)

    ax.set_ylim(-0.5, m - 1 + 0.5)
    ax.margins(y=0.02)

    # (keeping your manual xlim; feel free to switch to dynamic)
    ax.set_xlim(-25, 25)
    xmin, xmax = ax.get_xlim()
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:g}" if (xmin <= t <= xmax) else "" for t in xticks])

    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("gray")
    ax.spines["bottom"].set_color("gray")

    fig.tight_layout()
    #plt.show()
    
    # ------------------ save ------------------
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    #prefix = "" if experiment_id is None else f"{experiment_id}_"
    safe_title = str(treated_title).replace(os.sep, "_").replace(" ", "_")
    out_path = out_dir / f"{safe_title}.png"

    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return out_path


def matching_plot_error_bars_old(
    df_bootstrap,  # results_df from matching pipeline
    treated_title: str,
    dir: str,
    experiment_id: int | str | None = None,
    target_title: str = "Treatment",
    alpha: float = 0.05,
) -> Path:

    df_bootstrap = df_bootstrap.copy()

    # ------------------ pick which columns to plot ------------------
    use_pct = all(c in df_bootstrap.columns for c in ["ATE_pct", "CI_low_pct", "CI_high_pct", "p_value_pct"])
    use_abs = all(c in df_bootstrap.columns for c in ["ATE", "CI_low", "CI_high", "p_value"])

    if not (use_pct or use_abs):
        raise KeyError(
            "df_bootstrap must contain either "
            "['ATE_pct','CI_low_pct','CI_high_pct','p_value_pct'] "
            "or ['ATE','CI_low','CI_high','p_value']."
        )

    if use_pct:
        eff_col, lo_col, hi_col, p_col = "ATE_pct", "CI_low_pct", "CI_high_pct", "p_value_pct"
        x_label = "Effect (% difference vs matched controls)"
    else:
        eff_col, lo_col, hi_col, p_col = "ATE", "CI_low", "CI_high", "p_value"
        x_label = "Effect (absolute difference)"

    # ------------------ BH correction (unchanged idea) ------------------
    raw_pvals = df_bootstrap[p_col].to_numpy(dtype=float)

    # keep NaNs out of BH, then reinsert
    ok = np.isfinite(raw_pvals)
    adj = np.full_like(raw_pvals, np.nan, dtype=float)
    if ok.sum() > 0:
        adj[ok] = multipletests(
            raw_pvals[ok],
            alpha=0.05,
            method="fdr_bh",
            maxiter=1,
            is_sorted=False,
            returnsorted=False,
        )[1]

    df_bootstrap["p_value_adj_bh"] = adj
    df_bootstrap["is_significant_bh"] = df_bootstrap["p_value_adj_bh"] < alpha

    # ------------------ style params (match panels) ------------------
    point_size: float = 32.0
    ci_line_width: float = 5.0
    cap_height: float = 0.09
    p_fontsize: int = 9
    p_y_offset: float = 0.12
    zero_line = True

    def _color_for_sig(is_sig: bool) -> str:
        return "#E31A1C" if is_sig else "#9E9E9E"  # red if significant, grey otherwise

    # ------------------ prep y axis ------------------
    outcomes = df_bootstrap["outcome"].astype(str).tolist()
    m = len(outcomes)
    y = np.arange(m)

    # ------------------ figure ------------------
    fig, ax = plt.subplots(figsize=(10, 8))

    if zero_line:
        ax.axvline(
            0.0,
            linestyle="--",
            linewidth=1.25,
            color="#8a8a8a",
            alpha=0.85,
            zorder=0,
        )

    ax.grid(False)

    eff = df_bootstrap[eff_col].to_numpy(dtype=float)
    lo = df_bootstrap[lo_col].to_numpy(dtype=float)
    hi = df_bootstrap[hi_col].to_numpy(dtype=float)
    p_raw = df_bootstrap[p_col].to_numpy(dtype=float)
    p_adj = df_bootstrap["p_value_adj_bh"].to_numpy(dtype=float)
    is_sig = df_bootstrap["is_significant_bh"].to_numpy(dtype=bool)

    # ------------------ draw CIs + caps ------------------
    for j in range(m):
        if np.isnan(lo[j]) or np.isnan(hi[j]) or np.isnan(eff[j]):
            continue

        c = _color_for_sig(bool(is_sig[j]))

        ax.hlines(y[j], lo[j], hi[j], lw=ci_line_width, color=c, zorder=2)
        ax.plot([lo[j], lo[j]], [y[j] - cap_height, y[j] + cap_height], color=c, lw=ci_line_width, zorder=2)
        ax.plot([hi[j], hi[j]], [y[j] - cap_height, y[j] + cap_height], color=c, lw=ci_line_width, zorder=2)

    # point estimate (always black)
    ax.scatter(eff, y, s=point_size, color="black", zorder=3)

    # ------------------ annotations ------------------
    for j in range(m):
        if np.isnan(eff[j]) or np.isnan(p_raw[j]) or np.isnan(p_adj[j]):
            continue

        if p_raw[j] < 0.001:
            p_text = "p < 0.001"
        else:
            p_text = f"p = {p_raw[j]:.3f}; after adj p={p_adj[j]:.3f}"

        if use_pct:
            effect_text = f"{eff[j]:+.1f}%"
        else:
            effect_text = f"{eff[j]:+.3g}"

        label = f"{p_text}, Δ = {effect_text}" if bool(is_sig[j]) else p_text

        ax.text(
            eff[j],
            y[j] - p_y_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=p_fontsize,
            color="#333",
            zorder=4,
            clip_on=False,
        )

    # ------------------ axes cosmetics ------------------
    ax.set_title(f"{target_title}\nEffect of {treated_title}", fontsize=14, pad=10)
    ax.set_xlabel(x_label, fontsize=12)

    ax.set_yticks(y)
    ax.set_yticklabels(outcomes, fontsize=12)

    ax.set_ylim(-0.5, m - 1 + 0.5)
    ax.margins(y=0.02)
    
    ax.set_xlim(-11, 11)
    xmin, xmax = ax.get_xlim()
    xticks = ax.get_xticks()
    xticklabels = []
    for t in xticks:
        if t < xmin or t > xmax:
            xticklabels.append("")
        else:
            xticklabels.append(f"{t:g}")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    # dynamic xlim based on CI range (keeps things simple + robust)
    #finite = np.isfinite(lo) & np.isfinite(hi)
    #if finite.any():
    #    xmin = float(np.nanmin(lo[finite]))
    #    xmax = float(np.nanmax(hi[finite]))
    #    pad = 0.08 * (xmax - xmin) if xmax > xmin else (1.0 if use_pct else 0.1)
    #    ax.set_xlim(xmin - pad, xmax + pad)

    ax.invert_yaxis()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("gray")
    ax.spines["bottom"].set_color("gray")

    fig.tight_layout()
    plt.show()

    # ------------------ save ------------------
    #out_dir = Path(dir)
    #out_dir.mkdir(parents=True, exist_ok=True)

    #prefix = "" if experiment_id is None else f"{experiment_id}_"
    #safe_title = str(treated_title).replace(os.sep, "_")
    #out_path = out_dir / f"{prefix}{safe_title}.png"

    #fig.savefig(out_path, bbox_inches="tight")
    #plt.close(fig)

    return None #out_path
