from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import math
import re
import warnings

import numpy as np
import pandas as pd

try:
    from .config import *
except ImportError:  # allow running a file directly from this folder
    from config import *

try:
    from .core_base import *
    from .core_neutralization import *
except ImportError:
    from core_base import *
    from core_neutralization import *

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from matplotlib.patches import Rectangle

CORR_OUTDIR = OUTDIR / "neutral_factor_redundancy"
CORR_OUTDIR.mkdir(parents=True, exist_ok=True)

CORR_THRESHOLD = 0.80
CORR_MIN_OBS = MIN_OBS

SAVE_DAILY_PAIR_CORR = True

FACTOR_CORR_EXCEL = CORR_OUTDIR / "neutral_factor_redundancy_cluster.xlsx"
FACTOR_CORR_HEATMAP = CORR_OUTDIR / "neutral_factor_corr_clustermap.png"
HIGH_PAIR_PATH = CORR_OUTDIR / "neutral_factor_high_corr_pairs_abs_gt_0p8.csv"


# =============================================================================
# Readable factor labels
# =============================================================================

def redundancy_factor_label(factor: str) -> str:
    label_map = {
        "close_ma5_gap": "Close / MA5 - 1",
        "close_ma10_gap": "Close / MA10 - 1",
        "close_ma20_gap": "Close / MA20 - 1",
        "close_ma60_gap": "Close / MA60 - 1",
        "ma5_ma20_gap": "MA5 / MA20 - 1",
        "ma10_ma20_gap": "MA10 / MA20 - 1",

        "close_ema5_gap": "Close / EMA5 - 1",
        "close_ema10_gap": "Close / EMA10 - 1",
        "close_ema20_gap": "Close / EMA20 - 1",
        "close_ema60_gap": "Close / EMA60 - 1",
        "ema5_ema20_gap": "EMA5 / EMA20 - 1",
        "ema10_ema20_gap": "EMA10 / EMA20 - 1",

        "rsi14": "RSI 14",
        "rsi14_centered": "RSI 14 - 50",

        "macd_dif_pct": "MACD DIF / Price",
        "macd_dea_pct": "MACD DEA / Price",
        "macd_hist_pct": "MACD Hist / Price",
        "ema12_ema26_gap": "EMA12 / EMA26 - 1",
    }
    return label_map.get(str(factor), str(factor))


# =============================================================================
# 1. Build factor panel and exposure matrix
# =============================================================================

def build_factor_exposure_panel_for_redundancy() -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Rebuild stock-level technical factor panel and add exposure matrix.

    Exposure matrix:
        industry dummy 0/1 matrix + log_mkt_cap

    This function does NOT calculate IC.
    It only prepares data for neutralized factor correlation analysis.
    """
    files = find_input_files(
        DATA_DIR,
        years=YEARS,
        file_pattern=FILE_PATTERN,
        allow_missing=ALLOW_MISSING_FILES,
    )

    print(f"Found {len(files)} parquet files for redundancy check.")
    print(f"First file: {files[0]}")
    print(f"Last file : {files[-1]}")

    df_raw = read_parquet_files(files)
    print(f"Raw rows: {len(df_raw):,}")

    df = prepare_data(
        df_raw,
        price_mode=PRICE_MODE,
        filter_a_share=FILTER_A_SHARE,
    )
    del df_raw

    print(f"Prepared rows: {len(df):,}")
    print(f"Codes: {df['code'].nunique():,}; days: {df['trade_day'].nunique():,}")
    print(f"Date range: {df['trade_day'].min().date()} to {df['trade_day'].max().date()}")

    df, factor_cols = add_technical_factors(df, price_col="price")
    print(f"Factor columns: {factor_cols}")

    if DO_INDUSTRY_SIZE_NEUTRAL_IC:
        df, exposure_cols, industry_cols = add_industry_size_exposures(
            df,
            industry_xlsx=INDUSTRY_XLSX,
            market_cap_xlsx=MARKET_CAP_XLSX,
            base_date=MARKET_CAP_BASE_DATE,
            price_col="price",
        )
    else:
        raise ValueError(
            "DO_INDUSTRY_SIZE_NEUTRAL_IC must be True for neutralized factor redundancy check."
        )

    keep_cols = ["code", "trade_day"] + factor_cols + exposure_cols

    if "industry_valid" in df.columns:
        keep_cols.append("industry_valid")

    if "volume" in df.columns:
        keep_cols.append("volume")

    keep_cols = list(dict.fromkeys([c for c in keep_cols if c in df.columns]))
    df = df[keep_cols].copy()

    if not INCLUDE_SUSPENDED and "volume" in df.columns:
        before = len(df)
        df = df[df["volume"].fillna(0) > 0].copy()
        print(f"Tradable filter volume > 0: {before:,} rows -> {len(df):,} rows")

    return df, factor_cols, exposure_cols


# =============================================================================
# 2. Neutralize factor values by day
# =============================================================================

def neutralize_factors_for_one_day(
    x: pd.DataFrame,
    factor_cols: list[str],
    exposure_cols: list[str],
    min_obs: int = CORR_MIN_OBS,
) -> pd.DataFrame:
    """
    For one trading day, neutralize each factor by OLS:

        factor = industry dummies + log_mkt_cap + residual

    Return residualized factor values.
    Rows are stocks, columns are factors.
    """
    x = x.copy()

    if "industry_valid" in x.columns:
        base_valid = x["industry_valid"].astype(bool).to_numpy()
    else:
        base_valid = np.ones(len(x), dtype=bool)

    X_all = x[exposure_cols].to_numpy(dtype=float, copy=False)
    base_valid = base_valid & np.isfinite(X_all).all(axis=1)

    resid_data = {}

    for factor in factor_cols:
        y_all = x[factor].to_numpy(dtype=float, copy=False)

        valid = base_valid & np.isfinite(y_all)

        if valid.sum() < min_obs:
            resid_data[factor] = np.full(len(x), np.nan)
            continue

        y = y_all[valid]
        X = X_all[valid, :]

        if np.nanstd(y) == 0:
            resid_data[factor] = np.full(len(x), np.nan)
            continue

        resid_valid = _ols_residual_1d(y, X)

        resid_full = np.full(len(x), np.nan)
        resid_full[valid] = resid_valid

        resid_data[factor] = resid_full

    resid_df = pd.DataFrame(resid_data, index=x.index)
    return resid_df


# =============================================================================
# 3. Daily Spearman correlation matrix, then time-series average
# =============================================================================

def compute_mean_neutral_factor_corr(
    df: pd.DataFrame,
    factor_cols: list[str],
    exposure_cols: list[str],
    min_obs: int = CORR_MIN_OBS,
    save_daily_pair_corr: bool = SAVE_DAILY_PAIR_CORR,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    For each day:
        1. Neutralize factor values cross-sectionally.
        2. Compute Spearman correlation matrix across neutralized factors.

    Then average daily correlation matrices over time.
    """
    factor_cols = list(factor_cols)

    corr_sum = pd.DataFrame(0.0, index=factor_cols, columns=factor_cols)
    corr_count = pd.DataFrame(0, index=factor_cols, columns=factor_cols)

    daily_pair_records = []

    grouped = df.groupby("trade_day", sort=True)
    total_days = df["trade_day"].nunique()

    for idx, (day, x) in enumerate(grouped, start=1):
        if idx % 100 == 0:
            print(f"Processing factor correlation day {idx:,} / {total_days:,}: {pd.to_datetime(day).date()}")

        if len(x) < min_obs:
            continue

        resid_df = neutralize_factors_for_one_day(
            x=x,
            factor_cols=factor_cols,
            exposure_cols=exposure_cols,
            min_obs=min_obs,
        )

        corr_day = resid_df.corr(method="spearman", min_periods=min_obs)

        valid = corr_day.notna()

        corr_sum = corr_sum.add(corr_day.fillna(0.0), fill_value=0.0)
        corr_count = corr_count.add(valid.astype(int), fill_value=0)

        if save_daily_pair_corr:
            for i, f1 in enumerate(factor_cols):
                for j in range(i + 1, len(factor_cols)):
                    f2 = factor_cols[j]
                    val = corr_day.loc[f1, f2]

                    if pd.notna(val):
                        daily_pair_records.append(
                            {
                                "trade_day": day,
                                "factor_1": f1,
                                "factor_1_label": redundancy_factor_label(f1),
                                "factor_2": f2,
                                "factor_2_label": redundancy_factor_label(f2),
                                "corr": float(val),
                                "abs_corr": float(abs(val)),
                            }
                        )

    mean_corr = corr_sum / corr_count.replace(0, np.nan)

    for f in factor_cols:
        mean_corr.loc[f, f] = 1.0

    daily_pair_corr = pd.DataFrame(daily_pair_records) if save_daily_pair_corr else None

    return mean_corr, daily_pair_corr


# =============================================================================
# 4. Distance matrix and Ward clustering
# =============================================================================

def cluster_factor_corr_matrix(
    mean_corr: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], np.ndarray]:
    """
    Distance matrix:
        distance = 1 - abs(mean_corr)

    Use Ward hierarchical clustering and return reordered matrices.
    """
    corr_for_cluster = mean_corr.copy()

    corr_for_cluster = corr_for_cluster.replace([np.inf, -np.inf], np.nan)
    corr_for_cluster = corr_for_cluster.fillna(0.0)

    for f in corr_for_cluster.index:
        corr_for_cluster.loc[f, f] = 1.0

    distance = 1.0 - corr_for_cluster.abs()
    distance = distance.clip(lower=0.0, upper=1.0)

    np.fill_diagonal(distance.values, 0.0)

    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance.values, 0.0)

    condensed_distance = squareform(distance.values, checks=False)

    Z = linkage(condensed_distance, method="ward", optimal_ordering=True)

    order_idx = leaves_list(Z)
    ordered_factors = corr_for_cluster.index[order_idx].tolist()

    corr_ordered = corr_for_cluster.loc[ordered_factors, ordered_factors]
    distance_ordered = distance.loc[ordered_factors, ordered_factors]

    return corr_ordered, distance_ordered, ordered_factors, Z


# =============================================================================
# 5. High correlation pairs
# =============================================================================

def extract_high_corr_pairs(
    mean_corr: pd.DataFrame,
    threshold: float = CORR_THRESHOLD,
) -> pd.DataFrame:
    """
    Extract factor pairs with abs(mean Spearman corr) > threshold.
    """
    records = []
    factors = list(mean_corr.index)

    for i, f1 in enumerate(factors):
        for j in range(i + 1, len(factors)):
            f2 = factors[j]
            corr = mean_corr.loc[f1, f2]

            if pd.notna(corr) and abs(corr) > threshold:
                records.append(
                    {
                        "factor_1": f1,
                        "factor_1_label": redundancy_factor_label(f1),
                        "factor_2": f2,
                        "factor_2_label": redundancy_factor_label(f2),
                        "mean_spearman_corr": float(corr),
                        "abs_corr": float(abs(corr)),
                    }
                )

    high_pairs = pd.DataFrame(records)

    if not high_pairs.empty:
        high_pairs = high_pairs.sort_values("abs_corr", ascending=False).reset_index(drop=True)

    return high_pairs


# =============================================================================
# 6. Run redundancy check
# =============================================================================

def run_neutral_factor_redundancy_check():
    """
    Full pipeline:
    1. Rebuild factor panel and exposure matrix.
    2. Neutralize factors by industry + log market cap.
    3. Compute daily Spearman factor correlation.
    4. Average correlation over time.
    5. Cluster by distance = 1 - abs(corr).
    6. Export Excel, CSV, and seaborn clustermap.
    """
    df, factor_cols, exposure_cols = build_factor_exposure_panel_for_redundancy()

    print("Computing mean neutralized factor correlation matrix ...")
    mean_corr, daily_pair_corr = compute_mean_neutral_factor_corr(
        df=df,
        factor_cols=factor_cols,
        exposure_cols=exposure_cols,
        min_obs=CORR_MIN_OBS,
        save_daily_pair_corr=SAVE_DAILY_PAIR_CORR,
    )

    print("Running Ward hierarchical clustering ...")
    corr_ordered, distance_ordered, ordered_factors, linkage_matrix = cluster_factor_corr_matrix(mean_corr)

    high_pairs = extract_high_corr_pairs(
        mean_corr=mean_corr,
        threshold=CORR_THRESHOLD,
    )

    order_df = pd.DataFrame(
        {
            "cluster_order": np.arange(1, len(ordered_factors) + 1),
            "factor": ordered_factors,
            "factor_label": [redundancy_factor_label(f) for f in ordered_factors],
        }
    )

    distance = 1.0 - mean_corr.fillna(0.0).abs()
    distance = distance.clip(lower=0.0, upper=1.0)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance.values, 0.0)

    # -------------------------------------------------------------------------
    # CSV outputs
    # -------------------------------------------------------------------------
    mean_corr.to_csv(CORR_OUTDIR / "neutral_factor_corr_mean.csv", encoding="utf-8-sig")
    corr_ordered.to_csv(CORR_OUTDIR / "neutral_factor_corr_mean_clustered.csv", encoding="utf-8-sig")
    distance.to_csv(CORR_OUTDIR / "neutral_factor_distance_1_minus_abs_corr.csv", encoding="utf-8-sig")
    distance_ordered.to_csv(CORR_OUTDIR / "neutral_factor_distance_clustered.csv", encoding="utf-8-sig")
    order_df.to_csv(CORR_OUTDIR / "neutral_factor_cluster_order.csv", index=False, encoding="utf-8-sig")
    high_pairs.to_csv(HIGH_PAIR_PATH, index=False, encoding="utf-8-sig")

    if SAVE_DAILY_PAIR_CORR and daily_pair_corr is not None:
        daily_pair_corr.to_csv(
            CORR_OUTDIR / "neutral_factor_daily_pair_corr_long.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # -------------------------------------------------------------------------
    # Excel output
    # -------------------------------------------------------------------------
    with pd.ExcelWriter(FACTOR_CORR_EXCEL, engine="openpyxl") as writer:
        mean_corr.to_excel(writer, sheet_name="Mean_Corr")
        corr_ordered.to_excel(writer, sheet_name="Mean_Corr_Clustered")
        distance.to_excel(writer, sheet_name="Distance_1_minus_abs")
        distance_ordered.to_excel(writer, sheet_name="Distance_Clustered")
        order_df.to_excel(writer, sheet_name="Cluster_Order", index=False)
        high_pairs.to_excel(writer, sheet_name="High_Corr_Pairs", index=False)

        if SAVE_DAILY_PAIR_CORR and daily_pair_corr is not None:
            if len(daily_pair_corr) <= 1_048_000:
                daily_pair_corr.to_excel(writer, sheet_name="Daily_Pair_Corr", index=False)
            else:
                print(
                    f"Daily_Pair_Corr has {len(daily_pair_corr):,} rows, "
                    "larger than Excel limit. Saved as CSV only."
                )

    print(f"Saved redundancy Excel: {FACTOR_CORR_EXCEL}")

    # -------------------------------------------------------------------------
    # Better seaborn clustermap
    # -------------------------------------------------------------------------
    corr = mean_corr.copy()
    corr = corr.replace([np.inf, -np.inf], np.nan)
    corr = corr.dropna(axis=0, how="all").dropna(axis=1, how="all")

    common_factors = corr.index.intersection(corr.columns)
    corr = corr.loc[common_factors, common_factors]

    corr = corr.fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)

    distance_plot = 1.0 - corr.abs()
    distance_plot = distance_plot.clip(lower=0.0, upper=1.0)
    np.fill_diagonal(distance_plot.values, 0.0)

    distance_plot = (distance_plot + distance_plot.T) / 2.0
    np.fill_diagonal(distance_plot.values, 0.0)

    condensed_distance_plot = squareform(distance_plot.values, checks=False)

    Z_plot = linkage(condensed_distance_plot, method="ward", optimal_ordering=True)

    label_map = {f: redundancy_factor_label(f) for f in corr.index}

    corr_plot = corr.rename(index=label_map, columns=label_map)

    sns.set_theme(style="white", font_scale=0.85)

    g = sns.clustermap(
        corr_plot,
        row_linkage=Z_plot,
        col_linkage=Z_plot,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.3,
        linecolor="white",
        figsize=(14, 14),
        dendrogram_ratio=(0.18, 0.12),
        cbar_pos=(0.02, 0.78, 0.03, 0.16),
    )

    g.fig.suptitle(
        "Neutralized Factor Spearman Correlation\n"
        "Mean Daily Cross-Sectional Correlation, Ward Clustering by 1 - |corr|",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )

    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, ha="center", fontsize=9)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=9)

    g.ax_cbar.set_ylabel("Mean Spearman Corr", fontsize=10)

    row_order = g.dendrogram_row.reordered_ind
    ordered_factor_names = list(corr.index[row_order])

    ordered_corr = corr.loc[ordered_factor_names, ordered_factor_names]
    n = ordered_corr.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            value = ordered_corr.iloc[i, j]

            if np.isfinite(value) and abs(value) > CORR_THRESHOLD:
                rect = Rectangle(
                    (j, i),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=1.4,
                )
                g.ax_heatmap.add_patch(rect)

    g.fig.savefig(FACTOR_CORR_HEATMAP, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved seaborn clustermap: {FACTOR_CORR_HEATMAP}")
    print(f"Saved high-correlation pairs: {HIGH_PAIR_PATH}")

    print("\nHigh correlation factor pairs:")
    if high_pairs.empty:
        print(f"No factor pairs with abs(corr) > {CORR_THRESHOLD}.")
    else:
        print(high_pairs.to_string(index=False))

    return {
        "mean_corr": mean_corr,
        "corr_ordered": corr_ordered,
        "distance": distance,
        "distance_ordered": distance_ordered,
        "cluster_order": order_df,
        "high_pairs": high_pairs,
        "daily_pair_corr": daily_pair_corr,
        "linkage_matrix": linkage_matrix,
    }


if __name__ == "__main__":
    redundancy_result = run_neutral_factor_redundancy_check()