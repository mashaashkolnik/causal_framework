from __future__ import annotations

import math
import os
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from statsmodels.stats.multitest import multipletests


def summarize_ps_dataframes_6cols(
    ps_dataframes: Dict[str, pd.DataFrame],
    labels_dict: Optional[Dict[str, str]] = None,
    gender_col: str = "gender",
    treated_col: str = "treated",
    male_values: tuple = ("male", "m", 1, True),
    female_values: tuple = ("female", "f", 0, False),
    ddof: int = 1,
) -> pd.DataFrame:
    
    def get_mask(series: pd.Series, values: tuple) -> pd.Series:
        return series.isin(values) | series.astype(str).str.lower().isin(
            [str(v).lower() for v in values]
        )

    def format_stats(data: pd.Series) -> str:
        clean_data = pd.to_numeric(data, errors="coerce").dropna()
        if clean_data.empty:
            return ""
        return f"{clean_data.mean():.2f} ± {clean_data.std(ddof=ddof):.2f}"

    summary_rows = {}
    strata_labels = [
        ("Treated Mean ± Std", lambda t, m, f: t),
        ("Control Mean ± Std", lambda t, m, f: ~t),
        ("Treated Male Mean ± Std", lambda t, m, f: t & m),
        ("Control Male Mean ± Std", lambda t, m, f: ~t & m),
        ("Treated Female Mean ± Std", lambda t, m, f: t & f),
        ("Control Female Mean ± Std", lambda t, m, f: ~t & f),
    ]

    for exposure, df in ps_dataframes.items():
        val_cols = [exposure, f"{exposure}_target_day"]
        if not all(c in df.columns for c in val_cols):
            missing = [c for c in val_cols if c not in df.columns]
            raise ValueError(f"{exposure}: missing columns {missing}")
        
        is_treated = df[treated_col].astype(bool)
        is_m = get_mask(df[gender_col], male_values)
        is_f = get_mask(df[gender_col], female_values)
        row_stats = {}
        for col_name, mask_func in strata_labels:
            mask = mask_func(is_treated, is_m, is_f)
            pooled_data = df.loc[mask, val_cols].stack()
            row_stats[col_name] = format_stats(pooled_data)
        idx_label = labels_dict.get(exposure, exposure) if labels_dict else exposure
        summary_rows[idx_label] = row_stats

    res = pd.DataFrame.from_dict(summary_rows, orient="index")
    res.index.name = "Exposure"
    return res[[label for label, _ in strata_labels]]


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
    
    rdis = rdis or {}
    labels = labels or {}
    exposures = list(feature_values.keys())
    n = len(exposures)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, 
        figsize=(figsize_per_col * ncols, figsize_per_row * nrows), 
        squeeze=False
    )
    STYLE = {
        "box_face": (0.55, 0.72, 0.95, 0.45),
        "edge_col": (0.25, 0.25, 0.25, 1.0),
        "rdi_col": (0.15, 0.65, 0.15, 0.9),
        "median_props": dict(color="red", linewidth=2),
        "mean_props": dict(color=(0.25, 0.25, 0.25, 1.0), linestyle="--", linewidth=1.5)
    }
    def _clean(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=float).ravel()
        return arr[np.isfinite(arr)]

    for i, exp in enumerate(exposures):
        ax = axes[i // ncols, i % ncols]
        treated = _clean(feature_values[exp].get("treated", []))
        control = _clean(feature_values[exp].get("control", []))
        bp = ax.boxplot(
            [treated, control],
            positions=[1, 2],
            widths=0.55,
            patch_artist=True,
            showmeans=show_mean,
            meanline=True,
            showfliers=False,
            medianprops=STYLE["median_props"],
            meanprops=STYLE["mean_props"],
            boxprops=dict(edgecolor=STYLE["edge_col"], linewidth=1.0),
            whiskerprops=dict(color=STYLE["edge_col"], linewidth=1.0),
            capprops=dict(color=STYLE["edge_col"], linewidth=1.0),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(STYLE["box_face"])
        ax.set_xticks([1, 2], ["Test", "Control"])
        ax.set_title(labels.get(exp, exp), fontsize=11)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("gray")
        ax.spines["bottom"].set_color("gray")

        if rdis.get(exp) is not None:
            rdi_val = float(rdis[exp])
            ax.axhline(rdi_val, color=STYLE["rdi_col"], linestyle="--", linewidth=1.2, zorder=0)
            ax.text(0.98, rdi_val, f"RDI: {rdi_val:g}", color=STYLE["rdi_col"],
                    fontsize=11, ha="right", va="bottom", transform=ax.get_yaxis_transform())
        m_t = np.mean(treated) if treated.size else None
        m_c = np.mean(control) if control.size else None     
        ax.legend(
            handles=[
                Line2D([], [], linestyle="none", label=f"Test mean = {m_t:.1f}" if m_t is not None else "Test mean = NA"),
                Line2D([], [], linestyle="none", label=f"Control mean = {m_c:.1f}" if m_c is not None else "Control mean = NA")
            ],
            loc='lower left', frameon=False, fontsize=11, handlelength=0, handletextpad=0
        )
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.legend(
        handles=[
            Line2D([0], [0], color="red", lw=2, label="Median"),
            Line2D([0], [0], color=STYLE["edge_col"], lw=1.5, linestyle="--", label="Mean"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False, fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes


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
    labels = labels or {}
    cfg = {
        "confounders": variable_config.get("confounders", []),
        "negative_targets": variable_config.get("negative_targets", []),
        "structural": variable_config.get("structural_confounders", []),
        "direct": variable_config.get("direct_confounders", [])
    }
    def _get_clean_series(df, val_col):
        if outcome_col and outcome_col in df.columns:
            df = df.set_index(outcome_col)
        if val_col not in df.columns:
            raise KeyError(f"Missing column '{val_col}'")
        series = pd.to_numeric(df[val_col], errors="coerce")
        return series[~series.index.duplicated(keep="first")].dropna()

    def _calc_pct(series, subset_names, threshold):
        subset = series.reindex(subset_names).dropna()
        if subset.empty:
            return np.nan
        return (subset < threshold).mean() * 100.0
    summary_data = {}
    for k in sorted(control.keys()):
        p_series = _get_clean_series(control[k], p_col)
        a_series = _get_clean_series(asmd[k], asmd_col) if k in asmd else pd.Series(dtype=float)
        conf_vals = a_series.reindex(cfg["confounders"]).dropna()        
        row_label = labels.get(k, k)
        summary_data[row_label] = {
            "% confounders with ASMD<0.1": _calc_pct(a_series, cfg["confounders"], 0.10),
            "% direct confounders with ASMD<0.05": _calc_pct(a_series, cfg["direct"], 0.05),
            "% structural confounders with ASMD<0.05": _calc_pct(a_series, cfg["structural"], 0.05),
            "% negative targets significant": _calc_pct(p_series, cfg["negative_targets"], 0.05),
            "max ASMD (confounders)": conf_vals.max() if not conf_vals.empty else np.nan,
            "mean ASMD (confounders)": conf_vals.mean() if not conf_vals.empty else np.nan,
        }
    cols = [
        "% confounders with ASMD<0.1",
        "% direct confounders with ASMD<0.05",
        "% structural confounders with ASMD<0.05",
        "% negative targets significant",
        "max ASMD (confounders)",
        "mean ASMD (confounders)",
    ]
    return (
        pd.DataFrame.from_dict(summary_data, orient="index")[cols]
        .rename_axis(index_name)
        .sort_values(
            by=cols[:3] + ["max ASMD (confounders)"],
            ascending=[False, False, False, True]
        )
    )


def heatmap_effects_generic(
    dfs: list[pd.DataFrame],
    treatment_names: list[str],
    labels_dict: dict[str, str] = None,
    x_labels: dict[str, str] = None,
    outcome_col: str = "",
    figsize=(15, 12),
    fontsize_ticks=13,
    p_fontsize=12,
    effect_format="{:.1f}%",
    sig_thresh: float = 0.05,
    hatch_pattern: str = "///",
    hatch_color: str = "lightgray",
):
    effect_map, pval_map = {}, {}
    for name, df in zip(treatment_names, dfs):
        d = df.set_index(outcome_col) if outcome_col in df.columns else df.copy()
        d.index = d.index.astype(str)   
        effect_map[name] = d["ATE_pct_point"]
        pval_map[name] = d["pvalue_fdr_bh"]
    effect_df = pd.DataFrame(effect_map)
    pval_df = pd.DataFrame(pval_map)
    if x_labels:
        ordered_cols = [c for c in x_labels if c in effect_df.columns]
        other_cols = [c for c in effect_df.columns if c not in ordered_cols]
        effect_df = effect_df[ordered_cols + other_cols]
        pval_df = pval_df[ordered_cols + other_cols]

    if labels_dict:
        ordered_rows = [r for r in labels_dict.keys() if r in effect_df.index]
        effect_df = effect_df.reindex(ordered_rows)
        pval_df = pval_df.reindex(ordered_rows)
        
    is_sig = (pval_df < sig_thresh) & pval_df.notna() & effect_df.notna()
    annot = effect_df.copy().astype(object)
    annot = effect_df.applymap(lambda v: effect_format.format(v) if not pd.isna(v) else "")
    annot = annot.where(is_sig, "")
    vmax = np.nanmax(np.abs(effect_df.to_numpy())) or 1.0
    cmap = LinearSegmentedColormap.from_list("soft_bwr_r", ["#ff5e8c", "#ffffff", "#6eb9ff"])

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        effect_df,
        cmap=cmap, vmin=-vmax, vmax=vmax, center=0,
        mask=~is_sig,
        annot=annot,
        fmt="",
        annot_kws={"fontsize": p_fontsize, "color": "black"},
        linewidths=3.5, linecolor="white", cbar=True
    )
    for i in range(len(effect_df.index)):
        for j in range(len(effect_df.columns)):
            if not is_sig.iat[i, j]:
                ax.add_patch(Rectangle(
                    (j, i), 1, 1, fill=True, facecolor="white",
                    hatch=hatch_pattern, edgecolor=hatch_color, 
                    linewidth=0, zorder=1
                ))
                
    x_display_labels = [x_labels.get(c, c) for c in effect_df.columns] if x_labels else effect_df.columns
    ax.set_xticklabels(x_display_labels, rotation=0, fontsize=fontsize_ticks)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")    
    y_display_labels = [labels_dict[r] for r in effect_df.index]
    ax.set_yticklabels(y_display_labels, fontsize=fontsize_ticks)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    sns.despine(left=True, bottom=True, top=True, right=True)
    cbar = ax.collections[0].colorbar
    cbar.set_label("Effect (%)", fontsize=fontsize_ticks)
    cbar.ax.tick_params(labelsize=fontsize_ticks)
    plt.tight_layout()
    return ax


def plot_outcome_effects_panels_significant(
    dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    df_labels: Optional[Sequence[str]] = None,
    annotation_dict: Optional[Dict[str, str]] = None,
    outcome_col: str = "outcome",
    effect_col: str = "ATE_pct_point",
    ci_low_col: str = "CI_pct_2.5",
    ci_high_col: str = "CI_pct_97.5",
    p_col: str = "pvalue_fdr_bh",
    ate_abs_column: str = "ATE_abs_point",
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
    p_adjust: Optional[str] = "fdr_bh",
):
    dfs = [dfs] if isinstance(dfs, pd.DataFrame) else dfs
    n = len(dfs)
    df_labels = df_labels or [f"Set {i+1}" for i in range(n)]
    annotation_dict = annotation_dict or {}

    prepped = []
    for df in dfs:
        d = df.copy()
        if outcome_col in d.columns: d = d.set_index(outcome_col)        
        cols_map = {effect_col: "eff", ci_low_col: "lo", ci_high_col: "hi", 
                    p_col: "p_raw", ate_abs_column: "abs_val"}
        prepped.append(d[list(cols_map.keys())].rename(columns=cols_map))
    base = prepped[0]
    if sort_by == "abs":
        order = base["eff"].abs().sort_values().index.tolist()
    elif sort_by == "signed":
        order = base["eff"].sort_values().index.tolist()
    else:
        order = list(base.index)
    all_outcomes = order + [o for d in prepped for o in d.index if o not in order]
    ylabels = [labels_dict.get(o, o) for o in all_outcomes] if labels_dict else all_outcomes
    m = len(all_outcomes)
    y_pos = np.arange(m)
    cols = min(panels_per_row, n)
    rows = math.ceil(n / panels_per_row)
    fig, axes = plt.subplots(
        rows, cols, sharey=True, 
        figsize=(figsize_per_panel[0] * cols, figsize_per_panel[1] * rows),
        squeeze=False
    )
    axes_flat = axes.flatten()
    for idx, (ax, d, label) in enumerate(zip(axes_flat, prepped, df_labels)):
        d = d.reindex(all_outcomes)
        p_vals = d["p_raw"].to_numpy()
        is_sig = p_vals < sig_thresh
        colors = ["#E31A1C" if s else "#9E9E9E" for s in is_sig]
        if zero_line:
            ax.axvline(0, linestyle="--", lw=1.25, color="#8a8a8a", alpha=0.85, zorder=0)
        for i in range(m):
            if pd.isna(d["lo"].iat[i]): continue
            color = colors[i]
            ax.hlines(y_pos[i], d["lo"].iat[i], d["hi"].iat[i], lw=ci_line_width, color=color, zorder=2)
            ax.vlines([d["lo"].iat[i], d["hi"].iat[i]], y_pos[i]-cap_height, y_pos[i]+cap_height, color=color, lw=ci_line_width, zorder=2)
        ax.scatter(d["eff"], y_pos, s=point_size, color="black", zorder=3)
        if annotate_p:
            for i in range(m):
                if not is_sig[i] or pd.isna(d["eff"].iat[i]): continue  
                unit = annotation_dict.get(all_outcomes[i], "")
                sign = '+' if d["eff"].iat[i] > 0 else '-'
                txt = f"{sign}{abs(d['abs_val'].iat[i]):.1f} {unit} ({d['eff'].iat[i]:+.0f}%)"
                
                ax.text(d["eff"].iat[i], y_pos[i] - p_y_offset, txt, 
                        ha="center", va="bottom", fontsize=p_fontsize, color="#333", zorder=4)
        ax.set_title(label, fontsize=p_fontsize, pad=10)
        ax.set_xlabel(x_label, fontsize=p_fontsize)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ylabels, fontsize=p_fontsize)
        ax.set_ylim(-0.5, m - 0.5)
        ax.set_xlim(-11, 11)
        ax.invert_yaxis()
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("gray")
        ax.spines["bottom"].set_color("gray")
        if idx % cols != 0:
            ax.tick_params(axis="y", labelleft=False, left=False)
    for k in range(n, len(axes_flat)): axes_flat[k].set_visible(False)
    return fig, axes


def plot_outcome_effects_panels(
    results: Mapping[str, pd.DataFrame],
    to_plot: Sequence[str],
    *,
    exposure_labels: Optional[Mapping[str, str]] = None,
    outcome_col: str = "outcome",
    effect_col: str = "ATE_pct_point",
    ci_low_col: str = "CI_pct_2.5",
    ci_high_col: str = "CI_pct_97.5",
    p_col: str = "pvalue_fdr_bh",
    labels_dict: Optional[Dict[str, str]] = None,
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
    exposure_labels = exposure_labels or {}
    prepped_dfs = []    
    for key in to_plot:
        if key not in results:
            raise KeyError(f"Key '{key}' missing in results.")
        df = results[key].copy()
        if outcome_col in df.columns:
            df = df.set_index(outcome_col)            
        cols = {effect_col: "eff", ci_low_col: "lo", ci_high_col: "hi", p_col: "p"}
        prepped_dfs.append(df[list(cols.keys())].rename(columns=cols))

    base = prepped_dfs[0]
    if sort_by == "abs":
        order = base["eff"].abs().sort_values().index.tolist()
    elif sort_by == "signed":
        order = base["eff"].sort_values().index.tolist()
    else:
        order = list(base.index)

    outcomes = order + [o for d in prepped_dfs for o in d.index if o not in order]
    ylabels = [labels_dict.get(o, o) for o in outcomes] if labels_dict else outcomes
    m_outcomes = len(outcomes)
    outcome_gap = 2.25
    y_base = np.arange(m_outcomes) * outcome_gap
    offsets = np.linspace(-0.5, 0.5, len(to_plot)) if len(to_plot) > 1 else [0.0]
    colors = ["#777C6D", "#B7B89F", "#CBCBCB", "#57595B", "#452829", "#A18D6D"]
    fig, ax = plt.subplots(figsize=figsize_per_panel)
    if zero_line:
        ax.axvline(0, linestyle="--", lw=1.5, color="#8a8a8a", alpha=0.85, zorder=0)
    handles = []
    for i, (df, label) in enumerate(zip(prepped_dfs, to_plot)):
        d = df.reindex(outcomes)
        color = colors[i % len(colors)]
        y_coords = y_base + offsets[i]
        for j in range(m_outcomes):
            if pd.isna(d["lo"].iat[j]): continue
            ax.hlines(y_coords[j], d["lo"].iat[j] * factor, d["hi"].iat[j] * factor, lw=ci_line_width, color=color, zorder=2)
            for x_val in [d["lo"].iat[j], d["hi"].iat[j]]:
                ax.vlines(x_val * factor, y_coords[j] - cap_height, y_coords[j] + cap_height, color=color, lw=ci_line_width, zorder=2)
        ax.scatter(d["eff"] * factor, y_coords, s=point_size, color=color, zorder=4.5)
        if annotate_p:
            for j in range(m_outcomes):
                if pd.isna(d["hi"].iat[j]) or pd.isna(d["p"].iat[j]): continue
                ax.text(d["hi"].iat[j] * factor + 0.2, y_coords[j] - p_y_offset, 
                        f"p={d['p'].iat[j]:.3f}", ha="left", va="center", 
                        fontsize=p_fontsize, color=color, zorder=5)
        display_name = exposure_labels.get(label, label)
        handles.append(plt.Line2D([0], [0], color=color, lw=ci_line_width, marker="o", label=display_name))
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_yticks(y_base)
    ax.set_yticklabels(ylabels, fontsize=12)
    ax.set_xlim(-15, 15)
    ax.invert_yaxis() 
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("gray")
    ax.spines["bottom"].set_color("gray")
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.tight_layout()
    return fig, ax


def matching_plot_error_bars(
    df: pd.DataFrame,
    treated_title: str,
    dir: str,
    experiment_id: str = None,
    alpha: float = 0.05,
    figsize=(8, 8),
    out_dir: str = None,
    *,
    labels_dict: dict = None,
    diet_short_names_mapping: dict = None,
    outcome_col: str = "outcome",
    xlim: list = None,
    show_annotations: bool = True,
    text_above_offset: float = 0.28,
):
    labels_dict = labels_dict or {}
    mapping = diet_short_names_mapping or {}
    is_pct_mode = "ATE_pct" in df.columns
    cols = {
        "eff": "ATE_pct" if is_pct_mode else "ATE",
        "lo": "CI_low_pct" if is_pct_mode else "CI_low",
        "hi": "CI_high_pct" if is_pct_mode else "CI_high",
        "p": "p_value_pct" if is_pct_mode else "p_value"
    }    
    x_label = f"Effect ({'% difference vs matched controls' if is_pct_mode else 'absolute difference'})"
    raw_p = df[cols["p"]].to_numpy(dtype=float)
    adj_p = np.full_like(raw_p, np.nan)
    is_sig = (adj_p < alpha) & np.isfinite(adj_p)
    c_mean_col = df.columns[df.columns.str.contains('control_mean|mean_control|y0_mean')].tolist()
    c_mean = df[c_mean_col[0]] if c_mean_col else None
    outcomes = [labels_dict.get(str(o), o) for o in df[outcome_col]]
    m = len(outcomes)
    y_pos = np.arange(m)[::-1]
    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0, linestyle="--", lw=1.2, color="#7a7a7a", alpha=0.75, zorder=1)
    ax.grid(axis="x", lw=0.6, alpha=0.25, zorder=0)
    
    for i in range(m):
        eff_val = df[cols["eff"]].iat[i]
        if not np.isfinite(eff_val): continue
        color = "#C62828" if is_sig[i] else "#9E9E9E"
        lo, hi = df[cols["lo"]].iat[i], df[cols["hi"]].iat[i]
        if np.isfinite(lo) and np.isfinite(hi):
            ax.hlines(y_pos[i], lo, hi, lw=3.2, color=color, zorder=2)
            ax.vlines([lo, hi], y_pos[i]-0.12, y_pos[i]+0.12, color=color, lw=3.2, zorder=2)
        ax.scatter(eff_val, y_pos[i], s=42, color="black", zorder=3)
        if show_annotations and is_sig[i]:
            p_txt = f"p<{0.001 if adj_p[i] < 0.001 else f'{adj_p[i]:.3f}'}"

            ate_val = eff_val if not is_pct_mode else (eff_val/100 * c_mean.iat[i] if c_mean is not None else np.nan)
            pct_val = eff_val if is_pct_mode else (eff_val/c_mean.iat[i] * 100 if c_mean is not None else np.nan)
            
            annot_text = f"BH {p_txt} • ATE={ate_val:+.2g} • Δ={pct_val:+.1f}%"
            ax.text((lo+hi)/2 if np.isfinite(lo) else eff_val, y_pos[i] + text_above_offset, 
                    annot_text, ha="center", va="bottom", fontsize=10, color="#2b2b2b")

    ax.set_title(mapping.get(treated_title, treated_title), fontsize=14, pad=12)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(outcomes, fontsize=11)
    if xlim: ax.set_xlim(*xlim)
    
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    plt.tight_layout()

    save_dir = Path(out_dir or dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{f'{experiment_id}_' if experiment_id else ''}{treated_title}.png"
    save_path = save_dir / fname
    
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_combined_matching_ipw_results(
    feature: str,
    labels_dict: Dict[str, str],
    annotation_dict: Dict[str, str],
    diet_full_names_mapping: Dict[str, str],
    sig_thresh: float = 0.05,
    figsize_per_panel=(8, 8),
    point_size: float = 60.0,
    ci_line_width: float = 4.0,
    cap_height: float = 0.15,
) -> Tuple[plt.Figure, plt.Axes]:

    def load_and_prep(path: str, col_map: dict) -> pd.DataFrame:
        df = pd.read_csv(path)
        d = df[df["outcome"].isin(labels_dict.keys())].copy()
        d = d.rename(columns=col_map).set_index("outcome")
        p_vals = d["p"].to_numpy(dtype=float)
        #mask = np.isfinite(p_vals)
        #if mask.any():
        #    _, adj_p, _, _ = multipletests(p_vals[mask], alpha=sig_thresh, method="fdr_bh")
        #    d.loc[mask, "is_sig"] = adj_p < sig_thresh
        #else:
        d["is_sig"] =  p_vals < sig_thresh
        return d

    map_matching = {
        "ATE_pct": "effect", "CI_low_pct": "ci_low", "CI_high_pct": "ci_high",
        "p_value_pct_fdr_bh": "p", "ATE": "ate_abs"
    }
    map_ipw = {
        "ATE_pct_point": "effect", "CI_pct_2.5": "ci_low", "CI_pct_97.5": "ci_high",
        "pvalue_fdr_bh": "p", "ATE_abs_point": "ate_abs"
    }
    df_m = load_and_prep(f"results_matching/dataframes/{feature}_results.csv", map_matching)
    df_i = load_and_prep(f"results/dataframes/{feature}_ate.csv", map_ipw)
    outcomes = list(labels_dict.keys())
    m_count = len(outcomes)
    y_base = np.arange(m_count) * 3.5 
    methods = [
        ("Matching", df_m, "#ab001a", "#454545", -0.5), # Name, Data, SigColor, NonSigColor, Offset
        ("IPW",      df_i, "#ff5c75", "#8a8a8a",  0.5)
    ]
    fig, ax = plt.subplots(figsize=figsize_per_panel)
    ax.axvline(0.0, linestyle="--", lw=1.5, color="#8a8a8a", alpha=0.8, zorder=1)

    for name, data, sig_color, nonsig_color, offset in methods:
        d = data.reindex(outcomes)
        y_coords = y_base + offset

        for j in range(m_count):
            eff, lo, hi = d["effect"].iat[j], d["ci_low"].iat[j], d["ci_high"].iat[j]
            if pd.isna(eff): continue

            color = sig_color if d["is_sig"].iat[j] else nonsig_color
            ax.hlines(y_coords[j], lo, hi, lw=ci_line_width, color=color, zorder=2)
            ax.vlines([lo, hi], y_coords[j] - cap_height, y_coords[j] + cap_height, lw=ci_line_width, color=color, zorder=2)
            ax.scatter(eff, y_coords[j], s=point_size, color="black", edgecolors="white", zorder=3)

            if d["is_sig"].iat[j]:
                unit = annotation_dict.get(outcomes[j], "")
                abs_val = d["ate_abs"].iat[j]
                txt = f"{abs_val:+.1f}{f' {unit}' if unit else ''} ({eff:+.1f}%)"
                ax.text(14, y_coords[j], txt, ha="left", va="center", 
                        fontsize=14, fontweight="bold", color=color)

    title = diet_full_names_mapping.get(feature, feature).replace("\n", " ")
    ax.set_title(title, fontsize=14, pad=60)
    ax.set_yticks(y_base)
    ax.set_yticklabels([labels_dict[o] for o in outcomes], fontsize=14)
    ax.set_xlim(-25, 25)
    ax.invert_yaxis()    
    handles = [plt.Line2D([0], [0], color=c, lw=3, label=f"{n} ({s})") for n, _, sc, nsc, _ in methods for c, s in [(sc, "BH significant"), (nsc, "BH non-significant")]]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize=13)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    plt.tight_layout()
    return fig, ax


def plot_error_bars(
    df_bootstrap: pd.DataFrame,
    treated_title: str,
    dir: str,
    experiment_id: int | str | None = None,
    #target_title: str = "Treatment",
    #alpha: float = 0.05,
) -> Path:
    #raw_pvals = df_bootstrap["pvalue_fdr_bh"].to_numpy()
    #_, adj_pvals, _, _ = multipletests(raw_pvals, alpha=alpha, method="fdr_bh")    
    df = df_bootstrap#.assign(
    #    p_adj=adj_pvals,
    #    is_sig=adj_pvals < alpha
    #)
    outcomes = df["outcome"].tolist()
    m = len(outcomes)
    y_pos = np.arange(m)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axvline(0, linestyle="--", lw=1.25, color="#8a8a8a", alpha=0.85, zorder=0)
    eff = df["ATE_pct_point"].to_numpy()
    lo, hi = df["CI_pct_2.5"].to_numpy(), df["CI_pct_97.5"].to_numpy()
    is_sig = df["fdr_bh_significant"].to_numpy()
    for i in range(m):
        if pd.isna([eff[i], lo[i], hi[i]]).any():
            continue            
        color = "#E31A1C" if is_sig[i] else "#9E9E9E"
        ax.hlines(y_pos[i], lo[i], hi[i], lw=5.0, color=color, zorder=2)
        ax.vlines([lo[i], hi[i]], y_pos[i] - 0.09, y_pos[i] + 0.09, color=color, lw=5.0, zorder=2)
        p_text = "p < 0.001" if df["p_adj"].iat[i] < 0.001 else f"p={df['p_adj'].iat[i]:.3f}"
        label = f"{p_text}, Δ = {eff[i]:+.1f}%" if is_sig[i] else p_text
        ax.text(eff[i], y_pos[i] - 0.12, label, ha="center", va="bottom", 
                fontsize=9, color="#333", zorder=4)

    ax.scatter(eff, y_pos, s=32.0, color="black", zorder=3)
    ax.set_title(f"Effect of {treated_title}", fontsize=14, pad=10)
    ax.set_xlabel("Effect (% point difference)", fontsize=12)
    labels = [o.split('_target_day')[0].replace('_', ' ').capitalize() for o in outcomes]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlim(-21, 21)
    ax.set_ylim(-0.5, m - 0.5)
    ax.invert_yaxis()
    
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("gray")
    ax.spines["bottom"].set_color("gray")
    out_dir = Path(dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{experiment_id}_" if experiment_id is not None else ""
    safe_title = str(treated_title).replace(os.sep, "_")
    out_path = out_dir / f"{prefix}{safe_title}.png"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
