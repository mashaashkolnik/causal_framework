from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from statsmodels.stats.multitest import multipletests

import numpy as np
import pandas as pd


################################### IPW PREPARATION ###################################


def compute_overlap_and_trim(
    df: pd.DataFrame,
    treat_col: str,
    e_col: str,
    q: float = 0.01,
) -> pd.DataFrame:
    """
    Restrict the dataset to a region of good treated–control overlap in the
    propensity score, then perform one-sided quantile trimming within this region.

    This implements the overlap + trimming step described in the Methods:

    1. **Common support (overlap) restriction**
        - For each treatment arm (A=0, A=1), compute the min and max of the propensity score `e_col`
        Keep only observations whose propensity lies in the intersection of
        these two ranges. This avoids extrapolating to regions where only
        treated or only control observations appear.

    2. **One-sided quantile trimming within arms**
        - Among the treated, drop the lowest `q` fraction of propensity scores.
        - Among the controls, drop the highest `q` fraction of propensity scores.
        - This reduces the influence of extremely unlikely treatment assignments.

    Parameters
    ----------
    df :
        Full analysis dataframe containing treatment indicator and propensity.
    treat_col :
        Column name of the binary treatment indicator (0 = control, 1 = treated).
    e_col :
        Column name of the (calibrated) propensity score P(A=1 | X).
    q :
        One-sided trimming quantile in each arm (default 0.01 = 1%).

    Returns
    -------
    pd.DataFrame
        A copy of `df` restricted to overlap and trimmed by quantiles.
    """
    # Work with only the columns we actually need and drop rows with missing data
    d0 = df[[treat_col, e_col]].dropna().copy()
    # For each arm, compute min/max propensity score
    arm_stats = d0.groupby(treat_col)[e_col].agg(["min", "max"])
    # Intersection of treated and control support
    lo = max(float(arm_stats.loc[0, "min"]), float(arm_stats.loc[1, "min"]))
    hi = min(float(arm_stats.loc[0, "max"]), float(arm_stats.loc[1, "max"]))
    # Keep only observations whose propensity falls in the overlap interval [lo, hi]
    d1 = df.loc[df[e_col].between(lo, hi)].copy()
    # Within this overlapped sample, compute arm-specific quantiles
    # q_stats is a 2x2 table: index = arm, columns = {q, 1-q}
    q_stats = d1.groupby(treat_col)[e_col].quantile([q, 1 - q]).unstack()
    # For treated: drop extremely low propensities (below q-quantile)
    treated_lower = float(q_stats.loc[1, q])
    # For controls: drop extremely high propensities (above (1-q)-quantile)
    control_upper = float(q_stats.loc[0, 1 - q])
    # Keep treated with e >= treated_lower and controls with e <= control_upper
    mask = ((d1[treat_col] == 1) & (d1[e_col] >= treated_lower)) | (
        (d1[treat_col] == 0) & (d1[e_col] <= control_upper)
    )
    return d1.loc[mask].copy()


def make_ipw_weights(
    df: pd.DataFrame,
    treat_col: str,
    e_col: str,
    stabilize: bool = True,
    clip: tuple[float, float] | None = None,
) -> pd.Series:
    """
    Construct stabilized, Hájek-normalized inverse-probability weights (IPW).

    Steps
    -----
    1. Read treatment indicator A and propensity score e.
    2. Optionally add a *stabilization* factor equal to the marginal treatment
        prevalence P(A=1), which reduces variance.
    3. Compute inverse probability weights:
           w_raw = numer / [ A * e + (1-A) * (1-e) ].
    4. Optionally clip weights at chosen percentiles to down-weight extremes.
    5. Apply Hájek normalization **within each arm**:
        sum_treated(w) = n_treated
        sum_control(w) = n_control
        so that weighted means are on the same scale as unweighted means.

    Parameters
    ----------
    df :
        Dataframe containing treatment and propensity score.
    treat_col :
        Binary treatment indicator column (0/1).
    e_col :
        Propensity score column.
    stabilize :
        If True, use stabilized weights with marginal prevalence in numerator.
    clip :
        Optional (low, high) percentiles (e.g. (0.5, 99.5)) used to clip weights.

    Returns
    -------
    pd.Series
        A weight vector named `"w"` aligned with `df`.
    """
    # Treatment indicator as int array (0/1)
    A = df[treat_col].astype(int)
    e = df[e_col].astype(float).clip(1e-6, 1 - 1e-6)
    # Marginal treatment prevalence P(A=1)
    if stabilize:
        p_t = A.mean()
        # Stabilized numerator: P(A=a) for each individual
        numer = A * p_t + (1 - A) * (1 - p_t)
    else:
        # Unstabilized IPW
        numer = 1.0
    w = numer / (A * e + (1 - A) * (1 - e))
    if clip is not None:
        lo, hi = clip
        lo_v, hi_v = np.percentile(w, [lo, hi])
        w = w.clip(lo_v, hi_v)
        
    # Hájek normalization per arm 
    # Treat weights in each arm so that weighted sums equal the arm sample sizes
    # This keeps weighted means on the natural scale and improves stability
    w_t = w[A == 1]
    w_c = w[A == 0]
    # Rescale treated weights
    w.loc[A == 1] = w_t * (len(w_t) / w_t.sum())
    # Rescale control weights
    w.loc[A == 0] = w_c * (len(w_c) / w_c.sum())
    w.name = "w"
    return w


def prepare_ipw_dataset(
    df: pd.DataFrame,
    treat_col: str,
    e_col: str,
    sleep_targets: Sequence[str] | None = None,
    q: float = 0.01,
    stabilize: bool = True,
    clip: tuple[float, float] | None = (0.5, 99.5),
    dropna_targets: bool = True,
) -> pd.DataFrame:
    """
    Prepare a dataset for IPW estimation by applying overlap/trim rules and
    attaching a fully processed weight column.

    Pipeline
    --------
    1. Call :func:`compute_overlap_and_trim` to:
        - restrict to propensity-score overlap region, and
        - remove extreme treated/control observations via one-sided trimming.
    2. Compute stabilized, Hájek-normalized IPW weights via
        :func:`make_ipw_weights` and store them as column `"w"`.
    3. Optionally drop rows that have missing values in specified sleep outcomes.

    Parameters
    ----------
    df :
        Original analysis dataframe.
    treat_col :
        Binary treatment indicator column.
    e_col :
        Propensity score column.
    sleep_targets :
        Sleep outcome columns that must be non-missing (if `dropna_targets`).
    q :
        Trimming quantile used in `compute_overlap_and_trim`.
    stabilize :
        Passed through to :func:`make_ipw_weights`.
    clip :
        Percentile clipping bounds passed through to :func:`make_ipw_weights`.
    dropna_targets :
        If True, drop observations with NaNs in any of `sleep_targets`.

    Returns
    -------
    pd.DataFrame
        A trimmed, weighted dataset ready for causal analysis.
        Contains all original columns plus a `"w"` column with IPW weights.
    """
    # 1. Overlap + quantile trimming step
    trimmed = compute_overlap_and_trim(df, treat_col, e_col, q=q)
    trimmed = trimmed.copy()
    # 2. Construct and attach IPW weights
    trimmed["w"] = make_ipw_weights(
        trimmed, treat_col, e_col, stabilize=stabilize, clip=clip
    )
    # 3. Optionally enforce non-missing sleep outcomes
    if dropna_targets and sleep_targets:
        trimmed = trimmed.dropna(subset=list(sleep_targets)).copy()
    return trimmed


################################### ASMD ###################################


def _asmd_weighted_np(X: np.ndarray, A: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute weighted Absolute Standardized Mean Differences (ASMDs)
    for a matrix of continuous covariates.

    This is the vectorized, NumPy implementation used throughout the paper
    to assess balance before/after weighting.

    ASMD for feature j is:

        | m1_j - m0_j | / sqrt( 0.5 * (v1_j + v0_j) )

    where m1_j and v1_j are the weighted mean and variance of feature j in
    the treated group, and m0_j, v0_j are the same for the control group.

    Parameters
    ----------
    X :
        2D array of shape (n_samples, n_features) with covariate values.
    A :
        1D array of length n_samples with binary treatment indicators (0/1).
    w :
        1D array of length n_samples with IPW weights (already normalized).

    Returns
    -------
    np.ndarray
        1D array of ASMD values (length = n_features).
    """
    # Boolean masks for treated and control
    mask1 = A == 1
    mask0 = ~mask1
    w1 = w[mask1][:, None]
    w0 = w[mask0][:, None]
    # Split X into treated and control submatrices
    X1 = X[mask1]
    X0 = X[mask0]
    # Sum of weights in each arm (per feature axis)
    sumw1 = w1.sum(axis=0)
    sumw0 = w0.sum(axis=0)
    # Weighted means per feature in each arm
    m1 = (w1 * X1).sum(axis=0) / (sumw1 + 1e-12)
    m0 = (w0 * X0).sum(axis=0) / (sumw0 + 1e-12)
    # Weighted variances per feature in each arm
    v1 = (w1 * (X1 - m1) ** 2).sum(axis=0) / (sumw1 + 1e-12)
    v0 = (w0 * (X0 - m0) ** 2).sum(axis=0) / (sumw0 + 1e-12)
    # Pooled standard deviation used in ASMD denominator
    sd_pooled = np.sqrt(0.5 * (v1 + v0) + 1e-12)
    return np.abs(m1 - m0) / sd_pooled


################################### IPW ATE + BOOTSTRAP ###################################


def ate_ipw_means(
    df: pd.DataFrame,
    outcome: str,
    treat_col: str,
    w: pd.Series | np.ndarray,
    eps: float = 1e-12,
) -> tuple[float, float, float, float]:
    """
    Compute weighted arm means and ATE for a single outcome using IPW.

    The function uses Hájek-style weighted means for treated and control arms and
    reports both absolute and percent differences.

    Parameters
    ----------
    df :
        IPW-ready dataframe containing outcome and treatment indicator.
    outcome :
        Name of the outcome variable (column in `df`).
    treat_col :
        Binary treatment indicator column.
    w :
        IPW weights (Series or array) aligned with `df`.
    eps :
        Small constant to avoid division by zero.

    Returns
    -------
    (mu_t, mu_c, tau_abs, tau_pct) :
        - mu_t : weighted treated mean
        - mu_c : weighted control mean
        - tau_abs : absolute ATE (mu_t - mu_c)
        - tau_pct : percent ATE relative to control mean
    """
    # Treatment indicator
    A = df[treat_col].astype(int).to_numpy()
    # Outcome values
    Y = df[outcome].astype(float).to_numpy()
    # Coerce weights into a NumPy array
    w_arr = np.asarray(w, dtype=float)
    # Weighted sums in treated and control arms
    num_t = float(np.sum(w_arr * A * Y))
    den_t = float(np.sum(w_arr * A))
    num_c = float(np.sum(w_arr * (1 - A) * Y))
    den_c = float(np.sum(w_arr * (1 - A)))
    # Hájek weighted means
    mu_t = num_t / (den_t + eps)
    mu_c = num_c / (den_c + eps)
    # Absolute and relative treatment effects
    tau_abs = mu_t - mu_c
    tau_pct = 100.0 * tau_abs / (mu_c + eps)
    return mu_t, mu_c, tau_abs, tau_pct


def bootstrap_ipw(
    df: pd.DataFrame,
    treat_col: str,
    e_col: str,
    features_to_check: Sequence[str],
    outcomes: Sequence[str],
    B: int = 1000,
    stabilize: bool = True,
    clip: tuple[float, float] | None = (0.5, 99.5),
    random_state: int = 42,
    m: int | None = None,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the full IPW bootstrap procedure for balance diagnostics and ATEs.

    For a given exposure configuration, this function:

    1. Computes IPW weights on the full dataset.
    2. Computes weighted ASMDs on all `features_to_check` (once on full data,
        plus their bootstrap distribution).
    3. For each outcome in `outcomes`, computes:
        - point estimates of IPW ATEs (absolute and percent),
        - bootstrap confidence intervals,
        - bootstrap p-values,
        - significance flags.

    Parameters
    ----------
    df :
        Trimmed dataset (after overlap/trimming), still *unweighted*.
    treat_col :
        Binary treatment indicator column.
    e_col :
        Propensity score column used to build weights.
    features_to_check :
        Confounder columns for ASMD balance assessment.
    outcomes :
        Outcomes for which IPW ATEs will be estimated.
    B :
        Number of bootstrap replications.
    stabilize :
        If True, use stabilized IPW in all bootstrap replicates.
    clip :
        Percentiles for clipping weights in each replicate.
    random_state :
        Seed for NumPy random generator to make results reproducible.
    m :
        Size of each bootstrap sample. If None, use `n = len(df)` (standard bootstrap).
    alpha :
        Significance level for two-sided tests (default 0.05).

    Returns
    -------
    balance_after : pd.DataFrame
        Summary of ASMD diagnostics for each feature.
    ate_df : pd.DataFrame
        Summary of ATE estimates, CIs, p-values, and significance for each outcome.
    """
    rng = np.random.default_rng(random_state)
    n = len(df)
    if m is None:
        m = n
        
    # Pre-extract reusable arrays for efficiency
    A_full = df[treat_col].to_numpy(dtype=int)
    X_full = df[list(features_to_check)].to_numpy(dtype=float)

    # Compute IPW weights on the full dataset
    w_full = make_ipw_weights(df, treat_col, e_col, stabilize=stabilize, clip=clip)
    
    # Single ASMD vector on the full data (used as point estimate)
    asmd_once = _asmd_weighted_np(X_full, A_full, w_full.to_numpy(dtype=float))

    # ASMD matrix: B x (#features) for bootstrapped ASMDs
    asmd_mat = np.empty((B, len(features_to_check)), dtype=float)
    # For each outcome, store bootstrap samples of absolute and percent ATEs
    abs_samples: dict[str, np.ndarray] = {y: np.empty(B, dtype=float) for y in outcomes}
    pct_samples: dict[str, np.ndarray] = {y: np.empty(B, dtype=float) for y in outcomes}

    idx = np.arange(n)

    # Point estimates on full data
    point: dict[str, dict[str, float]] = {}
    for y in outcomes:
        mu_t, mu_c, tau_abs, tau_pct = ate_ipw_means(df, y, treat_col, w_full)
        point[y] = {"mu_t": mu_t, "mu_c": mu_c, "abs": tau_abs, "pct": tau_pct}

    # Bootstrap loop
    for b in range(B):
        # Sample row indices with replacement
        boot_idx = rng.choice(idx, size=m, replace=True)

        d_b = df.iloc[boot_idx]
        A_b = A_full[boot_idx]
        X_b = X_full[boot_idx, :]
        
        # Recompute weights within the bootstrap sample
        w_b = make_ipw_weights(
            d_b, treat_col, e_col, stabilize=stabilize, clip=clip
        ).to_numpy(dtype=float)

        # Store ASMD vector for this replicate
        asmd_mat[b, :] = _asmd_weighted_np(X_b, A_b, w_b)

        # Compute and store bootstrap ATEs for all outcomes
        for y in outcomes:
            _, _, tau_abs_b, tau_pct_b = ate_ipw_means(d_b, y, treat_col, w_b)
            abs_samples[y][b] = tau_abs_b
            pct_samples[y][b] = tau_pct_b

    # Summarize balance diagnostics
    balance_after = (
        pd.DataFrame(
            {
                "feature": list(features_to_check),
                "ASMD": asmd_once,
                "ASMD_med_boot": np.mean(asmd_mat, axis=0), #np.median(asmd_mat, axis=0),
                "ASMD_p90_boot": np.percentile(asmd_mat, 90, axis=0),
                "ASMD_p95_boot": np.percentile(asmd_mat, 95, axis=0),
            }
        )
        .sort_values("ASMD_med_boot", ascending=False)
        .reset_index(drop=True)
    )
    #print(f"Yes I confirm that we used mean instead of median")

    balance_after["balanced_0.10_med"] = balance_after["ASMD_med_boot"] <= 0.10
    balance_after["balanced_0.05_med"] = balance_after["ASMD_med_boot"] <= 0.05

    # Summarize ATE estimates
    rows: list[dict[str, Any]] = []
    for y in outcomes:
        abs_s = np.asarray(abs_samples[y], float)
        pct_s = np.asarray(pct_samples[y], float)

        abs_lo, abs_hi = np.percentile(abs_s, [2.5, 97.5])
        pct_lo, pct_hi = np.percentile(pct_s, [2.5, 97.5])

        p_abs = 2 * min((abs_s > 0).mean(), (abs_s < 0).mean())
        is_sig = (abs_lo * abs_hi) > 0

        rows.append(
            {
                "outcome": y,
                "ATE_abs_point": point[y]["abs"],
                "CI_abs_2.5": abs_lo,
                "CI_abs_97.5": abs_hi,
                "p_value_boot_abs": p_abs,
                "is_significant_abs": bool(is_sig and (p_abs < alpha)),
                "ATE_pct_point": point[y]["pct"],
                "CI_pct_2.5": pct_lo,
                "CI_pct_97.5": pct_hi,
                "mu_t_point": point[y]["mu_t"],
                "mu_c_point": point[y]["mu_c"],
            }
        )

    ate_df = pd.DataFrame(rows).sort_values("outcome").reset_index(drop=True)
    return balance_after, ate_df
