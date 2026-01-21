from __future__ import annotations
from typing import Dict, Optional
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from typing import List, Optional, Dict, Sequence
import matplotlib.pyplot as plt
import seaborn as sns
import math
from typing import Mapping, Sequence, Optional, Dict, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def summarize_ps_dataframes_6cols(
    ps_dataframes: Dict[str, pd.DataFrame],
    labels_dict: Optional[Dict[str, str]] = None,
    gender_col: str = "gender",
    treated_col: str = "treated",
    male_values: tuple = ("male", "m", 1, True),
    female_values: tuple = ("female", "f", 0, False),
    ddof: int = 1,
) -> pd.DataFrame:
    """
    For each exposure dataframe, summarize mean ± std of:
        [exposure, exposure_target_day] pooled together

    Stratified into exactly 6 columns:
        treated
        untreated
        treated_male
        untreated_male
        treated_female
        untreated_female

    If labels_dict is provided, rename index values using labels_dict.get(exposure, exposure).
    Index name is set to "Exposure".
    """

    def is_male(s: pd.Series) -> pd.Series:
        return s.isin(male_values) | s.astype(str).str.lower().isin(
            [str(x).lower() for x in male_values]
        )

    def is_female(s: pd.Series) -> pd.Series:
        return s.isin(female_values) | s.astype(str).str.lower().isin(
            [str(x).lower() for x in female_values]
        )

    def fmt(x: pd.Series) -> str:
        x = pd.to_numeric(x, errors="coerce").dropna()
        if x.empty:
            return ""
        mean = float(x.mean())
        std = float(x.std(ddof=ddof))
        return f"{mean:.2f} ± {std:.2f}"

    rows = {}

    for exposure, df in ps_dataframes.items():
        cols = [exposure, f"{exposure}_target_day"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{exposure}: missing columns {missing}")

        treated = df[treated_col].astype(bool)
        male = is_male(df[gender_col])
        female = is_female(df[gender_col])

        rows[exposure] = {
            "Treated Mean ± Std": fmt(pd.concat([df.loc[treated, c] for c in cols], axis=0)),
            "Control Mean ± Std": fmt(pd.concat([df.loc[~treated, c] for c in cols], axis=0)),
            "Treated Male Mean ± Std": fmt(pd.concat([df.loc[treated & male, c] for c in cols], axis=0)),
            "Control Male Mean ± Std": fmt(pd.concat([df.loc[(~treated) & male, c] for c in cols], axis=0)),
            "Treated Female Mean ± Std": fmt(pd.concat([df.loc[treated & female, c] for c in cols], axis=0)),
            "Control Female Mean ± Std": fmt(pd.concat([df.loc[(~treated) & female, c] for c in cols], axis=0)),
        }

    res = pd.DataFrame.from_dict(rows, orient="index")

    # Rename exposures if labels_dict provided
    if labels_dict:
        res.index = [labels_dict.get(k, k) for k in res.index]

    res.index.name = "Exposure"

    # enforce exact column order
    res = res[
        [
            "Treated Mean ± Std",
            "Control Mean ± Std",
            "Treated Male Mean ± Std",
            "Control Male Mean ± Std",
            "Treated Female Mean ± Std",
            "Control Female Mean ± Std",
        ]
    ]

    return res

import math
from typing import Any, Mapping, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_test_control_distributions(
    feature_values: Mapping[str, Mapping[str, Sequence[float] | np.ndarray]],
    rdis: Mapping[str, float] | None = None,
    labels: Mapping[str, str] | None = None,
    ncols: int = 4,
    figsize_per_col: float = 4,
    figsize_per_row: float = 4,
    show_mean: bool = True,
    save_path: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    if rdis is None:
        rdis = {}
    if labels is None:
        labels = {}

    exposures = list(feature_values.keys())
    n = len(exposures)
    if n == 0:
        raise ValueError("feature_values is empty.")

    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / ncols))

    fig_w = figsize_per_col * ncols
    fig_h = figsize_per_row * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)

    def _to_clean_array(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=float).ravel()
        return arr[np.isfinite(arr)]

    def _safe_mean(a: np.ndarray) -> float | None:
        return float(np.mean(a)) if a.size else None

    def _safe_median(a: np.ndarray) -> float | None:
        return float(np.median(a)) if a.size else None

    box_face = (0.55, 0.72, 0.95, 0.45)
    edge_col = (0.25, 0.25, 0.25, 1.0)
    rdi_col = (0.15, 0.65, 0.15, 0.9)

    for i, exp in enumerate(exposures):
        ax = axes[i // ncols, i % ncols]

        treated = _to_clean_array(feature_values[exp].get("treated", []))
        control = _to_clean_array(feature_values[exp].get("control", []))

        data = [treated, control]
        positions = [1, 2]

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showmeans=show_mean,
            meanline=True,
            showfliers=False,
            medianprops=dict(color="red", linewidth=2),
            meanprops=dict(color=edge_col, linestyle="--", linewidth=1.5),
            boxprops=dict(edgecolor=edge_col, linewidth=1.0),
            whiskerprops=dict(color=edge_col, linewidth=1.0),
            capprops=dict(color=edge_col, linewidth=1.0),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(box_face)

        ax.set_xticks(positions, ["Test", "Control"])

        title = labels.get(exp, exp)
        ax.set_title(title, fontsize=11)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("gray")
        ax.spines["bottom"].set_color("gray")
        ax.grid(False)

        # --- median numeric labels (keep as in your original) ---
        med_t = _safe_median(treated)
        med_c = _safe_median(control)

        y_min, y_max = ax.get_ylim()
        y_span = max(1e-12, y_max - y_min)

        #if med_t is not None:
        #    ax.text(1, med_t + 0.03 * y_span, f"{med_t:.3f}", ha="center", va="bottom", fontsize=8)
        #if med_c is not None:
        #    ax.text(2, med_c + 0.03 * y_span, f"{med_c:.3f}", ha="center", va="bottom", fontsize=8)

        # --- RDI line (if available) ---
        if exp in rdis and rdis[exp] is not None:
            rdi_val = float(rdis[exp])
            ax.axhline(rdi_val, color=rdi_col, linestyle="--", linewidth=1.2, zorder=0)
            ax.text(
                0.98,
                rdi_val,
                f"RDI: {rdi_val:g}",
                color=rdi_col,
                fontsize=11,
                ha="right",
                va="bottom",
                transform=ax.get_yaxis_transform(),
            )

        # --- NEW: per-axes legend showing means in upper-right ---
        mean_t = _safe_mean(treated)
        mean_c = _safe_mean(control)

        # Use text-only "legend" entries (invisible handles) so it sits like a legend
        mean_handles = [
            Line2D([], [], linestyle="none", marker=None, label=f"Test mean = {mean_t:.1f}" if mean_t is not None else "Test mean = NA"),
            Line2D([], [], linestyle="none", marker=None, label=f"Control mean = {mean_c:.1f}" if mean_c is not None else "Control mean = NA"),
        ]
        ax.legend(
            handles=mean_handles,
            #loc="upper right",
            loc='lower left',
            frameon=False,
            fontsize=11,
            handlelength=0,
            handletextpad=0,
            borderpad=0.2,
            labelspacing=0.25,
        )

    # Hide any unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    legend_elements = [
        Line2D([0], [0], color="red", lw=2, label="Median"),
        Line2D([0], [0], color="black", lw=1.5, linestyle="--", label="Mean"),
    ]
    
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
        fontsize=12,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes

import pandas as pd
import numpy as np


def summarize_runs(
    control,
    asmd,
    variable_config,
    *,
    labels=None,
    outcome_col=None,
    p_col="p_value_boot_abs",
    asmd_col="ASMD",
    index_name="dietary exposure",
):
    labels = {} if labels is None else dict(labels)

    confounders = list(variable_config.get("confounders", []))
    negative_targets = list(variable_config.get("negative_targets", []))
    structural_confounders = list(variable_config.get("structural_confounders", []))
    direct_confounders = list(variable_config.get("direct_confounders", []))

    def _series(df, value_col):
        if outcome_col and outcome_col in df.columns:
            s = df.set_index(outcome_col)[value_col]
        else:
            if value_col not in df.columns:
                raise KeyError(f"Missing column '{value_col}' in dataframe.")
            s = df[value_col]
            if s.index.name is None:
                s.index.name = "variable"

        if not s.index.is_unique:
            s = s[~s.index.duplicated(keep="first")]
        return s

    def _pct(names, series, thresh):
        if not names:
            return np.nan
        idx = [n for n in names if n in series.index]
        if not idx:
            return np.nan
        vals = pd.to_numeric(series.loc[idx], errors="coerce").dropna()
        if vals.empty:
            return np.nan
        return 100.0 * float((vals < thresh).sum()) / float(len(vals))

    def _subset_vals(names, series):
        if not names:
            return pd.Series(dtype=float)
        idx = [n for n in names if n in series.index]
        if not idx:
            return pd.Series(dtype=float)
        return pd.to_numeric(series.loc[idx], errors="coerce").dropna()

    rows = []
    index_vals = []

    for k in sorted(control.keys()):
        index_vals.append(labels.get(k, k))

        s_p = _series(control[k], p_col)
        pct_neg_sig = _pct(negative_targets, s_p, 0.05)

        if k in asmd:
            s_a = _series(asmd[k], asmd_col)

            pct_struct = _pct(structural_confounders, s_a, 0.05)
            pct_direct = _pct(direct_confounders, s_a, 0.05)
            pct_conf_a = _pct(confounders, s_a, 0.10)

            conf_vals = _subset_vals(confounders, s_a)
            max_asmd = float(conf_vals.max()) if not conf_vals.empty else np.nan
            mean_asmd = float(conf_vals.mean()) if not conf_vals.empty else np.nan
        else:
            pct_struct = pct_direct = pct_conf_a = np.nan
            max_asmd = mean_asmd = np.nan

        rows.append(
            {
                "% confounders with ASMD<0.1": pct_conf_a,
                "% structural confounders with ASMD<0.05": pct_struct,
                "% direct confounders with ASMD<0.05": pct_direct,
                "% negative targets significant": pct_neg_sig,
                "max ASMD (confounders)": max_asmd,
                "mean ASMD (confounders)": mean_asmd,
            }
        )

    cols = [
        "% confounders with ASMD<0.1",
        "% direct confounders with ASMD<0.05",
        "% structural confounders with ASMD<0.05",
        "% negative targets significant",
        "max ASMD (confounders)",
        "mean ASMD (confounders)",
    ]

    out = pd.DataFrame(rows, index=pd.Index(index_vals, name=index_name))[cols].astype(float).sort_values(
        by=[
            "% confounders with ASMD<0.1",
            "% direct confounders with ASMD<0.05",
            "% structural confounders with ASMD<0.05",
            "max ASMD (confounders)",
        ],
        ascending=[False, False, False, True],
    )

    return out


def heatmap_effects_generic(
    dfs: List[pd.DataFrame],
    treatment_names: Sequence[str],
    labels_dict: Optional[Dict[str, str]] = None,
    x_labels: Optional[Dict[str, str]] = None,
    outcome_col: str = "outcome",
    figsize=(15, 12),
    title="Relative Effect of Different Nutrition Factors on Sleep",
    fontsize_title=18,
    fontsize_ticks=13,          # was 11
    p_fontsize=12,              # was 10
    p_format="p = {:.3g}",
    show_effect=True,
    effect_format="{:.1f}%",
    sig_thresh: float = 0.05,
    hatch_pattern: str = "///",
    hatch_color: str = "lightgray",
):
    if len(dfs) != len(treatment_names):
        raise ValueError("dfs and treatment_names must have the same length.")

    # --- normalize inputs ---
    normed = []
    for i, df in enumerate(dfs):
        need = {"ATE_pct_point", "p_value_boot_abs"}
        if not need.issubset(df.columns):
            missing = need - set(df.columns)
            raise ValueError(f"DataFrame {i} missing columns: {missing}")
        d = df.copy()
        if outcome_col in d.columns:
            d = d.set_index(outcome_col)
        d.index = d.index.astype(str)
        normed.append(d[["ATE_pct_point", "p_value_boot_abs"]])

    # Build matrices
    all_outcomes = sorted(set().union(*[d.index for d in normed]))
    effect_df = pd.DataFrame(index=all_outcomes, columns=treatment_names, dtype=float)
    pval_df   = pd.DataFrame(index=all_outcomes, columns=treatment_names, dtype=float)
    for name, d in zip(treatment_names, normed):
        effect_df.loc[d.index, name] = d["ATE_pct_point"].astype(float)
        pval_df.loc[d.index, name]   = d["p_value_boot_abs"].astype(float)

    # --- reorder x-axis based on x_labels dict keys ---
    if x_labels is not None:
        ordered_cols = [col for col in x_labels.keys() if col in effect_df.columns]
        remaining = [col for col in effect_df.columns if col not in ordered_cols]
        new_order = ordered_cols + remaining
        effect_df = effect_df[new_order]
        pval_df   = pval_df[new_order]
        treatment_names = new_order

    # Optional row order / labels
    if labels_dict is not None:
        ordered = [k for k in labels_dict.keys() if k in effect_df.index]
        effect_df = effect_df.reindex(ordered)
        pval_df   = pval_df.reindex(ordered)

    # Color scale range
    arr = effect_df.to_numpy(dtype=float)
    finite_vals = arr[np.isfinite(arr)]
    vmax = float(np.nanmax(np.abs(finite_vals))) if finite_vals.size else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    colors = ["#ff5e8c", "#ffffff", "#6eb9ff"]
    cmap = LinearSegmentedColormap.from_list("soft_bwr_r", colors)

    sig = (pval_df < sig_thresh) & np.isfinite(pval_df) & np.isfinite(effect_df)

    # -----------------------------------------------------------
    # Annotation function without p-values — only effect
    # -----------------------------------------------------------
    def _fmt_cell(i, j):
        if not sig.iat[i, j]:
            return ""
        eff = effect_df.iat[i, j]
        if pd.isna(eff):
            return ""
        return effect_format.format(eff).strip()

    annot = effect_df.copy().astype(object)
    for i in range(effect_df.shape[0]):
        for j in range(effect_df.shape[1]):
            annot.iat[i, j] = _fmt_cell(i, j)

    # --- PLOT ---
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        effect_df.astype(float),
        cmap=cmap,
        vmin=-vmax, vmax=vmax, center=0,
        mask=~sig,
        annot=annot,
        fmt="",
        annot_kws={"fontsize": p_fontsize, "color": "black", "ha": "center", "va": "center"},
        linewidths=3.5,
        linecolor="white",
        cbar=True,
        square=False
    )

    # hatched non-significant tiles
    nrows, ncols = effect_df.shape
    for i in range(nrows):
        for j in range(ncols):
            if not sig.iat[i, j]:
                ax.add_patch(
                    Rectangle(
                        (j, i), 1, 1,
                        fill=True,
                        facecolor="white",
                        hatch=hatch_pattern,
                        edgecolor=hatch_color,
                        linewidth=0,
                        zorder=6
                    )
                )

    # grid borders
    for x in range(effect_df.shape[1] + 1):
        ax.axvline(x, color="white", linewidth=3.5, alpha=1, zorder=7)
    for y in range(effect_df.shape[0] + 1):
        ax.axhline(y, color="white", linewidth=3.5, alpha=1, zorder=7)

    # No title for journal figure
    # ax.set_title(title, fontsize=fontsize_title, pad=20)

    # x labels using x_labels
    xticks = ax.get_xticklabels()
    new_x = []
    for tick in xticks:
        key = tick.get_text()
        if x_labels and key in x_labels:
            new_x.append(x_labels[key])
        else:
            new_x.append(key)

    ax.set_xticklabels(new_x, rotation=0, fontsize=fontsize_ticks)

    # put x-axis ticks at the TOP
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", which="both", pad=8)

    # y labels
    ylabels = [labels_dict.get(idx, idx) for idx in effect_df.index] if labels_dict else list(effect_df.index)
    ax.set_yticklabels(ylabels, fontsize=fontsize_ticks)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsize_ticks)
    cbar.outline.set_visible(False)
    cbar.set_label("Effect (%)", fontsize=fontsize_ticks)

    plt.tight_layout()
    plt.show()
    
    
def plot_outcome_effects_panels_significant(
    dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    df_labels: Optional[Sequence[str]] = None,
    annotation_dict=None,
    outcome_col: str = "outcome",
    effect_col: str = "ATE_pct_point",
    ci_low_col: str = "CI_pct_2.5",
    ci_high_col: str = "CI_pct_97.5",
    p_col: str = "p_value_boot_abs",
    ate_abs_column = "ATE_abs_point",
    labels_dict: Optional[Dict[str, str]] = None,
    figsize_per_panel=(8, 8),
    x_label: str = "Effect (% point difference)",
    sig_thresh: float = 0.05,
    point_size: float = 32.0,
    ci_line_width: float = 5.0,
    cap_height: float = 0.09,
    annotate_p: bool = True,
    p_fontsize: int = 9,
    p_y_offset: float = 0.12,
    sort_by: Optional[str] = None,
    zero_line: bool = True,
    panels_per_row: int = 3,
    p_adjust: Optional[str] = "fdr_bh",   # NEW: None disables adjustment
    annotate_adj: bool = True,            # NEW: whether to show adjusted p in text
):
    """
    Panels in a grid. Shares a single outcome order across panels.
    """

    # ---------- normalize inputs ----------
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
    n = len(dfs)
    if n == 0:
        raise ValueError("No dataframes provided.")
    if df_labels is None:
        df_labels = [f"Set {i+1}" for i in range(n)]
    if len(df_labels) != n:
        raise ValueError("df_labels length must match number of dataframes.")

    # ---------- standardize columns ----------
    prepped = []
    for df in dfs:
        d = df.copy()
        if outcome_col in d.columns:
            d = d.set_index(outcome_col)
        for col in (effect_col, ci_low_col, ci_high_col, p_col):
            if col not in d.columns:
                raise KeyError(f"Missing column '{col}' in one dataframe.")
        d = d[[effect_col, ci_low_col, ci_high_col, p_col, ate_abs_column]].rename(
            columns={effect_col: "effect", ci_low_col: "ci_low", ci_high_col: "ci_high", p_col: "p_raw"}
        )
        prepped.append(d)

    # ---------- outcome order based on first df ----------
    base = prepped[0]
    if sort_by == "abs":
        order = base["effect"].abs().sort_values(ascending=True).index.tolist()
    elif sort_by == "signed":
        order = base["effect"].sort_values(ascending=True).index.tolist()
    else:
        order = list(base.index)

    union = set(order)
    for d in prepped[1:]:
        union |= set(d.index)
    outcomes = order + [o for o in union if o not in order]

    if labels_dict:
        ylabels = [labels_dict.get(o) for o in outcomes]
    else:
        ylabels = list(outcomes)

    m = len(outcomes)
    y = np.arange(m)

    # ---------- figure and axes (grid) ----------
    cols = min(panels_per_row, n)
    rows = math.ceil(n / panels_per_row)
    fig_w = figsize_per_panel[0] * cols
    fig_h = figsize_per_panel[1] * rows

    fig, axes = plt.subplots(
        rows, cols, sharey=True, figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.05, "hspace": 0.15},
        constrained_layout=False,
    )

    if rows == 1 and cols == 1:
        axes_2d = np.array([[axes]])
    elif rows == 1:
        axes_2d = np.array([axes])
    elif cols == 1:
        axes_2d = axes.reshape(rows, 1)
    else:
        axes_2d = axes

    axes_flat = axes_2d.flatten()

    # ---------- color rule now depends on significance flag ----------
    def _color_for_sig(is_sig: bool) -> str:
        return "#E31A1C" if is_sig else "#9E9E9E"

    # ---------- helper: per-panel p-value adjustment ----------
    def _adjust_pvals(p_raw: np.ndarray) -> np.ndarray:
        """
        Adjust p-values within a panel.
        Leaves NaNs as NaN, and adjusts only finite entries.
        """
        p_raw = np.asarray(p_raw, dtype=float)
        p_adj = np.full_like(p_raw, np.nan, dtype=float)

        mask = np.isfinite(p_raw)
        if mask.sum() == 0 or p_adjust is None:
            return p_raw.copy()  # no adjustment

        p_adj_masked = multipletests(
            p_raw[mask],
            alpha=sig_thresh,
            method=p_adjust,
            is_sorted=False,
            returnsorted=False,
        )[1]

        p_adj[mask] = p_adj_masked
        return p_adj

    # ---------- draw panels ----------
    for idx, (ax, d, raw_label) in enumerate(zip(axes_flat, prepped, df_labels)):
        d = d.reindex(outcomes)

        eff = d["effect"].to_numpy()
        lo = d["ci_low"].to_numpy()
        hi = d["ci_high"].to_numpy()
        p_raw = d["p_raw"].to_numpy()
        ate_abs_point = d["ATE_abs_point"].to_numpy()

        # NEW: per-panel adjustment computed after reindexing (so it matches plotted outcomes)
        #p_adj = _adjust_pvals(p_raw)
        #is_sig = np.isfinite(p_adj) & (p_adj < sig_thresh)

        is_sig = p_raw < sig_thresh
        if zero_line:
            ax.axvline(0.0, linestyle="--", linewidth=1.25, color="#8a8a8a", alpha=0.85, zorder=0)

        ax.grid(False)

        # CIs + caps
        for j in range(m):
            if np.isnan(lo[j]) or np.isnan(hi[j]):
                continue
            c = _color_for_sig(bool(is_sig[j]))
            ax.hlines(y[j], lo[j], hi[j], lw=ci_line_width, color=c, zorder=2)
            ax.plot([lo[j], lo[j]], [y[j] - cap_height, y[j] + cap_height], color=c, lw=ci_line_width, zorder=2)
            ax.plot([hi[j], hi[j]], [y[j] - cap_height, y[j] + cap_height], color=c, lw=ci_line_width, zorder=2)

        # point estimate
        ax.scatter(eff, y, s=point_size, color="black", zorder=3)

        # annotations
        if annotate_p:
            for j in range(m):
                if np.isnan(eff[j]) or np.isnan(p_raw[j]):
                    continue

                # 2) remove annotation when NOT significant
                if not bool(is_sig[j]):
                    continue

                # 1) significant: "ATE={ATE_abs_point} (±{effect}%); p-value={p-value}"
                # ATE_abs_point is assumed to be the absolute effect in original units.
                # If your df does NOT have it, we fall back to eff[j] (percent points).
                ate_abs = None
                if "ATE_abs_point" in d.columns:
                    ate_abs = d["ATE_abs_point"].to_numpy()[j]
                else:
                    ate_abs = np.nan  # will fallback below

                # p-value formatting
                if p_raw[j] < 0.001:
                    p_txt = "<0.001"
                else:
                    p_txt = f"{p_raw[j]:.3f}"

                #print(outcomes)
                #measure_units = annotation_dict[outcomes[d]]
                
                outcome_key = outcomes[j]  # this is the raw outcome name
                measure_units = annotation_dict.get(outcome_key, "") if annotation_dict else "idk"
                
                sig = '+' if eff[j] > 0 else '-'
                txt = f"{sig}{abs(ate_abs_point[j]):.1f} {measure_units} ({eff[j]:+.0f}%)" # ; p-value={p_txt}

                ax.text(
                    eff[j],
                    y[j] - p_y_offset,
                    txt,
                    ha="center",
                    va="bottom",
                    fontsize=p_fontsize,
                    color="#333",
                    zorder=4,
                    clip_on=False,
                )

        # cosmetics
        ax.set_title(raw_label, fontsize=14, pad=10)
        ax.set_xlabel(x_label, fontsize=12)

        ax.set_yticks(y)
        ax.set_yticklabels(ylabels, fontsize=12)

        ax.set_ylim(-0.5, m - 1 + 0.5)
        ax.margins(y=0.02)

        ax.set_xlim(-11, 11)
        xmin, xmax = ax.get_xlim()
        xticks = ax.get_xticks()
        ax.set_xticks(xticks)
        ax.set_xticklabels([("" if (t < xmin or t > xmax) else f"{t:g}") for t in xticks])

        ax.invert_yaxis()

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("gray")
        ax.spines["bottom"].set_color("gray")

        col = idx % cols
        if col != 0:
            ax.tick_params(axis="y", labelleft=False, left=False)

    for k in range(n, rows * cols):
        axes_flat[k].set_visible(False)

    return fig, axes_2d


def plot_outcome_effects_panels(
    results: Mapping[str, pd.DataFrame],
    to_plot: Sequence[str],
    *,
    exposure_labels: Optional[Mapping[str, str]] = None,   # e.g., diet_full_names_mapping
    outcome_col: str = "outcome",
    effect_col: str = "ATE_pct_point",
    ci_low_col: str = "CI_pct_2.5",
    ci_high_col: str = "CI_pct_97.5",
    p_col: str = "p_value_boot_abs",
    labels_dict: Optional[Dict[str, str]] = None,          # outcome label prettifier
    figsize_per_panel=(8, 8),
    x_label: str = "Effect (%)",
    sig_thresh: float = 0.05,
    point_size: float = 50.0,
    ci_line_width: float = 5.0,
    cap_height: float = 0.09,
    annotate_p: bool = False,
    p_fontsize: int = 11,
    p_y_offset: float = 0.0,
    sort_by: Optional[str] = None,
    zero_line: bool = True,
    factor: float = 1.0,
):
    """
    Single-panel forest plot where multiple exposures (selected by `to_plot`) are shown together.

    """
    exposure_labels = {} if exposure_labels is None else dict(exposure_labels)

    # ---------- build dfs + labels in the order of to_plot ----------
    missing = [k for k in to_plot if k not in results]
    if missing:
        raise KeyError(f"These keys from `to_plot` are missing in `results`: {missing}")

    dfs = [results[k] for k in to_plot]
    df_labels = [exposure_labels.get(k, k) for k in to_plot]
    n = len(dfs)
    if n == 0:
        raise ValueError("`to_plot` is empty.")

    # ---------- standardize columns ----------
    prepped = []
    for df in dfs:
        d = df.copy()
        if outcome_col in d.columns:
            d = d.set_index(outcome_col)

        for col in (effect_col, ci_low_col, ci_high_col, p_col):
            if col not in d.columns:
                raise KeyError(f"Missing column '{col}' in one dataframe.")

        d = d[[effect_col, ci_low_col, ci_high_col, p_col]].rename(
            columns={effect_col: "effect", ci_low_col: "ci_low", ci_high_col: "ci_high", p_col: "p"}
        )
        prepped.append(d)

    # ---------- outcome order based on first df ----------
    base = prepped[0]
    if sort_by == "abs":
        order = base["effect"].abs().sort_values(ascending=True).index.tolist()
    elif sort_by == "signed":
        order = base["effect"].sort_values(ascending=True).index.tolist()
    else:
        order = list(base.index)

    union = set(order)
    for d in prepped[1:]:
        union |= set(d.index)
    outcomes = order + [o for o in union if o not in order]

    # prettified y-labels (outcomes)
    if labels_dict:
        ylabels = [labels_dict.get(o, o) for o in outcomes]
    else:
        ylabels = list(outcomes)

    m = len(outcomes)

    # OUTCOME spacing > treatment spacing
    outcome_gap = 2.25
    y_base = np.arange(m) * outcome_gap

    # ---------- figure & single axes ----------
    fig_w, fig_h = figsize_per_panel
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ---------- colors & offsets ----------
    pastel_palette = ["#777C6D", "#B7B89F", "#CBCBCB", "#57595B", "#452829", "#A18D6D"]
    colors = [pastel_palette[i % len(pastel_palette)] for i in range(n)]

    if n == 1:
        offsets = [0.0]
    else:
        offsets = np.linspace(-0.5, 0.5, n)

    # ---------- global x-range for p-label positioning ----------
    all_los, all_his = [], []
    for d in prepped:
        dd = d.reindex(outcomes)
        all_los.append(dd["ci_low"].values)
        all_his.append(dd["ci_high"].values)
    all_los = np.concatenate(all_los)
    all_his = np.concatenate(all_his)

    xmin = np.nanmin(all_los)
    xmax = np.nanmax(all_his)
    x_span = xmax - xmin if np.isfinite(xmax - xmin) and (xmax - xmin) > 0 else 1.0
    p_dx = 0.02 * x_span

    # ---------- draw everything ----------
    if zero_line:
        ax.axvline(0.0, linestyle="--", linewidth=1.5, color="#8a8a8a", alpha=0.85, zorder=0)

    handles = []

    for d, label, color, offset in zip(prepped, df_labels, colors, offsets):
        d = d.reindex(outcomes)

        eff = d["effect"].values * factor
        lo  = d["ci_low"].values * factor
        hi  = d["ci_high"].values * factor
        pvl = d["p"].values

        y = y_base + offset

        # CIs + caps
        for j in range(m):
            if np.isnan(lo[j]) or np.isnan(hi[j]):
                continue
            ax.hlines(y[j], lo[j], hi[j], lw=ci_line_width, color=color, zorder=2)
            ax.plot([lo[j], lo[j]], [y[j] - cap_height, y[j] + cap_height], color=color, lw=ci_line_width, zorder=2)
            ax.plot([hi[j], hi[j]], [y[j] - cap_height, y[j] + cap_height], color=color, lw=ci_line_width, zorder=2)

        # points
        ax.scatter(
            eff, y,
            s=point_size,
            marker="o",
            facecolor=color,
            edgecolor=color,
            linewidth=1,
            zorder=4.5,
        )

        # p-values
        if annotate_p:
            for j in range(m):
                if np.isnan(hi[j]) or np.isnan(pvl[j]):
                    continue
                ax.text(
                    hi[j] + p_dx,
                    y[j] - p_y_offset,
                    f"p={pvl[j]:.3f}",
                    ha="left",
                    va="center",
                    fontsize=p_fontsize,
                    color=color,
                    zorder=5,
                    clip_on=False,
                )

        # legend handle
        h = plt.Line2D(
            [0], [0],
            color=color,
            lw=ci_line_width,
            marker="o",
            markeredgecolor=color,
            markerfacecolor=color,
            markersize=math.sqrt(point_size),
            label=label,
        )
        handles.append(h)

    # ---------- axis cosmetics ----------
    ax.grid(False)
    ax.set_xlabel(x_label, fontsize=14)

    ax.set_yticks(y_base)
    ax.set_yticklabels(ylabels, fontsize=12)
    ax.tick_params(axis="x", labelsize=12)

    ax.set_ylim(-outcome_gap, (m - 1) * outcome_gap + outcome_gap)
    ax.set_xlim(-11, 11)

    xmin2, xmax2 = ax.get_xlim()
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([("" if (t < xmin2 or t > xmax2) else f"{t:g}") for t in xticks])

    ax.invert_yaxis()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("gray")
    ax.spines["bottom"].set_color("gray")

    ax.legend(
        handles=handles,
        fontsize=12,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=1,
        frameon=False,
    )

    fig.tight_layout()
    return fig, ax
