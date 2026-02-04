from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def compute_overlap_and_trim(
    df: pd.DataFrame,
    treat_col: str,
    e_col: str,
    q: float = 0.01,
) -> pd.DataFrame:
    d0 = df[[treat_col, e_col]].dropna().copy()
    arm_stats = d0.groupby(treat_col)[e_col].agg(["min", "max"])
    lo = max(float(arm_stats.loc[0, "min"]), float(arm_stats.loc[1, "min"]))
    hi = min(float(arm_stats.loc[0, "max"]), float(arm_stats.loc[1, "max"]))
    d1 = df.loc[df[e_col].between(lo, hi)].copy()
    q_stats = d1.groupby(treat_col)[e_col].quantile([q, 1 - q]).unstack()
    treated_lower = float(q_stats.loc[1, q])
    control_upper = float(q_stats.loc[0, 1 - q])
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
    A = df[treat_col].astype(int)
    e = df[e_col].astype(float).clip(1e-6, 1 - 1e-6)
    if stabilize:
        p_t = A.mean()
        numer = A * p_t + (1 - A) * (1 - p_t)
    else:
        numer = 1.0
    w = numer / (A * e + (1 - A) * (1 - e))
    if clip is not None:
        lo, hi = clip
        lo_v, hi_v = np.percentile(w, [lo, hi])
        w = w.clip(lo_v, hi_v)
    w_t = w[A == 1]
    w_c = w[A == 0]
    w.loc[A == 1] = w_t * (len(w_t) / w_t.sum())
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
    trimmed = compute_overlap_and_trim(df, treat_col, e_col, q=q)
    trimmed = trimmed.copy()
    trimmed["w"] = make_ipw_weights(
        trimmed, treat_col, e_col, stabilize=stabilize, clip=clip
    )
    if dropna_targets and sleep_targets:
        trimmed = trimmed.dropna(subset=list(sleep_targets)).copy()
    return trimmed


def _asmd_weighted_np(X: np.ndarray, A: np.ndarray, w: np.ndarray) -> np.ndarray:
    mask1 = A == 1
    mask0 = ~mask1
    w1 = w[mask1][:, None]
    w0 = w[mask0][:, None]
    X1 = X[mask1]
    X0 = X[mask0]
    sumw1 = w1.sum(axis=0)
    sumw0 = w0.sum(axis=0)
    m1 = (w1 * X1).sum(axis=0) / (sumw1 + 1e-12)
    m0 = (w0 * X0).sum(axis=0) / (sumw0 + 1e-12)
    v1 = (w1 * (X1 - m1) ** 2).sum(axis=0) / (sumw1 + 1e-12)
    v0 = (w0 * (X0 - m0) ** 2).sum(axis=0) / (sumw0 + 1e-12)
    sd_pooled = np.sqrt(0.5 * (v1 + v0) + 1e-12)
    return np.abs(m1 - m0) / sd_pooled


def ate_ipw_means(
    df: pd.DataFrame,
    outcome: str,
    treat_col: str,
    w: pd.Series | np.ndarray,
    eps: float = 1e-12,
) -> tuple[float, float, float, float]:
    A = df[treat_col].astype(int).to_numpy()
    Y = df[outcome].astype(float).to_numpy()
    w_arr = np.asarray(w, dtype=float)
    num_t = float(np.sum(w_arr * A * Y))
    den_t = float(np.sum(w_arr * A))
    num_c = float(np.sum(w_arr * (1 - A) * Y))
    den_c = float(np.sum(w_arr * (1 - A)))
    mu_t = num_t / (den_t + eps)
    mu_c = num_c / (den_c + eps)
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
    rng = np.random.default_rng(random_state)
    n = len(df)
    if m is None:
        m = n
    A_full = df[treat_col].to_numpy(dtype=int)
    X_full = df[list(features_to_check)].to_numpy(dtype=float)
    w_full = make_ipw_weights(df, treat_col, e_col, stabilize=stabilize, clip=clip)
    asmd_once = _asmd_weighted_np(X_full, A_full, w_full.to_numpy(dtype=float))
    asmd_mat = np.empty((B, len(features_to_check)), dtype=float)
    abs_samples: dict[str, np.ndarray] = {y: np.empty(B, dtype=float) for y in outcomes}
    pct_samples: dict[str, np.ndarray] = {y: np.empty(B, dtype=float) for y in outcomes}
    idx = np.arange(n)
    point: dict[str, dict[str, float]] = {}
    for y in outcomes:
        mu_t, mu_c, tau_abs, tau_pct = ate_ipw_means(df, y, treat_col, w_full)
        point[y] = {"mu_t": mu_t, "mu_c": mu_c, "abs": tau_abs, "pct": tau_pct}
    for b in range(B):
        boot_idx = rng.choice(idx, size=m, replace=True)
        d_b = df.iloc[boot_idx]
        A_b = A_full[boot_idx]
        X_b = X_full[boot_idx, :]
        w_b = make_ipw_weights(
            d_b, treat_col, e_col, stabilize=stabilize, clip=clip
        ).to_numpy(dtype=float)
        asmd_mat[b, :] = _asmd_weighted_np(X_b, A_b, w_b)
        for y in outcomes:
            _, _, tau_abs_b, tau_pct_b = ate_ipw_means(d_b, y, treat_col, w_b)
            abs_samples[y][b] = tau_abs_b
            pct_samples[y][b] = tau_pct_b
    balance_after = (
        pd.DataFrame(
            {
                "feature": list(features_to_check),
                "ASMD": asmd_once,
                "ASMD_med_boot": np.mean(
                    asmd_mat, axis=0
                ),  # np.median(asmd_mat, axis=0),
                "ASMD_p90_boot": np.percentile(asmd_mat, 90, axis=0),
                "ASMD_p95_boot": np.percentile(asmd_mat, 95, axis=0),
            }
        )
        .sort_values("ASMD_med_boot", ascending=False)
        .reset_index(drop=True)
    )
    balance_after["balanced_0.10_med"] = balance_after["ASMD_med_boot"] <= 0.10
    balance_after["balanced_0.05_med"] = balance_after["ASMD_med_boot"] <= 0.05
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
