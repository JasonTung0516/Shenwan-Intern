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
    from .factor_redundancy import *
except ImportError:
    from core_base import *
    from core_neutralization import *
    from factor_redundancy import *

# Select non-redundant factors by WARD clustering
# Requirement:
# 1. Use neutralized factor mean Spearman correlation matrix.
# 2. Distance = 1 - abs(corr).
# 3. Use Ward hierarchical clustering.
# 4. Cut tree at distance threshold = 1 - 0.9 = 0.1.
# 5. In each Ward cluster, keep the factor with the highest abs(NW-corrected Rank ICIR).
# 6. Single-factor clusters are kept directly.

def selection_factor_label(factor: str) -> str:
    """
    Unified readable factor label function.

    Some later chunks call selection_factor_label().
    This function prevents 'undefined name: selection_factor_label' errors.
    """
    factor = str(factor)

    label_map = {
        # MA
        "close_ma5_gap": "Close / MA5 - 1",
        "close_ma10_gap": "Close / MA10 - 1",
        "close_ma20_gap": "Close / MA20 - 1",
        "close_ma60_gap": "Close / MA60 - 1",
        "ma5_ma20_gap": "MA5 / MA20 - 1",
        "ma10_ma20_gap": "MA10 / MA20 - 1",

        # EMA
        "close_ema5_gap": "Close / EMA5 - 1",
        "close_ema10_gap": "Close / EMA10 - 1",
        "close_ema20_gap": "Close / EMA20 - 1",
        "close_ema60_gap": "Close / EMA60 - 1",
        "ema5_ema20_gap": "EMA5 / EMA20 - 1",
        "ema10_ema20_gap": "EMA10 / EMA20 - 1",

        # RSI
        "rsi14": "RSI 14",
        "rsi14_centered": "RSI 14 - 50",

        # MACD
        "macd_dif_pct": "MACD DIF / Price",
        "macd_dea_pct": "MACD DEA / Price",
        "macd_hist_pct": "MACD Hist / Price",
        "ema12_ema26_gap": "EMA12 / EMA26 - 1",

        # Composite factors
        "composite_selected_mad_z": "Equal-weight composite, MAD-z",
        "composite_equal_mad_z": "Equal-weight composite, MAD-z",
        "composite_weighted_mad_z": "Abs(NW Rank ICIR)-weighted composite, MAD-z",
    }

    return label_map.get(factor, factor)


def selection_factor_group(factor: str) -> str:
    """
    Unified factor group function.

    Some later chunks may call selection_factor_group().
    """
    factor = str(factor)

    if factor in [
        "composite_selected_mad_z",
        "composite_equal_mad_z",
        "composite_weighted_mad_z",
    ]:
        return "Composite"

    if factor.startswith("close_ma"):
        return "MA"

    if factor.startswith("ma") and "_ma" in factor:
        return "MA_Cross"

    if factor.startswith("close_ema"):
        return "EMA"

    if factor in ["ema5_ema20_gap", "ema10_ema20_gap"]:
        return "EMA_Cross"

    if factor.startswith("rsi"):
        return "RSI"

    if factor.startswith("macd") or factor == "ema12_ema26_gap":
        return "MACD"

    return "Other"


# Optional aliases for consistency.
selection_indicator_group = selection_factor_group

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


# =============================================================================
# CONFIG
# =============================================================================

REDUNDANCY_GROUP_THRESHOLD = 0.90
WARD_DISTANCE_THRESHOLD = 1.0 - REDUNDANCY_GROUP_THRESHOLD

SELECTION_SCORE_COL = "rank_ic_nw_icir"

SELECTION_OUTDIR = OUTDIR / "neutral_factor_redundancy"
SELECTION_OUTDIR.mkdir(parents=True, exist_ok=True)

MEAN_CORR_CSV = SELECTION_OUTDIR / "neutral_factor_corr_mean.csv"

SUMMARY_CSV_FOR_SELECTION = (
    OUTDIR / "technical_factor_ic_summary_winsor_industry_size_neutral_2005_2026.csv"
)

DAILY_CSV_FOR_SELECTION = (
    OUTDIR / "technical_factor_daily_ic_winsor_industry_size_neutral_2005_2026.csv"
)

# Ward-specific outputs
WARD_SELECTION_EXCEL = (
    SELECTION_OUTDIR / "selected_factors_by_ward_corr_gt_0p9_rank_ic_nw_icir.xlsx"
)

WARD_SELECTED_FACTORS_CSV = (
    SELECTION_OUTDIR / "selected_factors_by_ward_corr_gt_0p9_rank_ic_nw_icir.csv"
)

WARD_GROUP_DETAIL_CSV = (
    SELECTION_OUTDIR / "ward_redundancy_groups_corr_gt_0p9_detail.csv"
)

WARD_GROUP_SUMMARY_CSV = (
    SELECTION_OUTDIR / "ward_redundancy_groups_corr_gt_0p9_summary.csv"
)

# Compatibility outputs:
# Keep these True so your later composite code can keep reading the old file names.
OVERWRITE_COMPATIBILITY_FILES = True

COMPAT_SELECTED_FACTORS_CSV = (
    SELECTION_OUTDIR / "selected_factors_by_corr_gt_0p9_rank_ic_nw_icir.csv"
)

COMPAT_SELECTION_EXCEL = (
    SELECTION_OUTDIR / "selected_factors_by_corr_gt_0p9_rank_ic_nw_icir.xlsx"
)

COMPAT_GROUP_DETAIL_CSV = (
    SELECTION_OUTDIR / "redundancy_groups_corr_gt_0p9_detail.csv"
)


# =============================================================================
# Helper: factor labels
# =============================================================================

def ward_selection_factor_label(factor: str) -> str:
    if "selection_factor_label" in globals():
        return selection_factor_label(factor)

    if "redundancy_factor_label" in globals():
        return redundancy_factor_label(factor)

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


def ward_selection_indicator_group(factor: str) -> str:
    factor = str(factor)

    if factor.startswith("close_ma"):
        return "MA"

    if factor.startswith("ma") and "_ma" in factor:
        return "MA_Cross"

    if factor.startswith("close_ema"):
        return "EMA"

    if factor in ["ema5_ema20_gap", "ema10_ema20_gap"]:
        return "EMA_Cross"

    if factor.startswith("rsi"):
        return "RSI"

    if factor.startswith("macd") or factor == "ema12_ema26_gap":
        return "MACD"

    return "Other"


# =============================================================================
# 1. Load mean neutralized factor correlation matrix
# =============================================================================

def load_mean_corr_for_ward_selection() -> pd.DataFrame:
    """
    Load neutralized factor mean Spearman correlation matrix.

    Priority:
    1. Use redundancy_result["mean_corr"] if it exists in memory.
    2. Otherwise read neutral_factor_corr_mean.csv.
    """
    if (
        "redundancy_result" in globals()
        and isinstance(redundancy_result, dict)
        and "mean_corr" in redundancy_result
    ):
        mean_corr = redundancy_result["mean_corr"].copy()
    else:
        if not MEAN_CORR_CSV.exists():
            raise FileNotFoundError(
                f"Cannot find mean correlation matrix: {MEAN_CORR_CSV}. "
                "Please run the neutral factor redundancy check first."
            )
        mean_corr = pd.read_csv(MEAN_CORR_CSV, index_col=0)

    mean_corr.index = mean_corr.index.astype(str)
    mean_corr.columns = mean_corr.columns.astype(str)

    common = mean_corr.index.intersection(mean_corr.columns)
    mean_corr = mean_corr.loc[common, common].copy()

    mean_corr = mean_corr.replace([np.inf, -np.inf], np.nan)
    mean_corr = mean_corr.fillna(0.0)

    # Symmetrize for safety
    mean_corr = (mean_corr + mean_corr.T) / 2.0

    np.fill_diagonal(mean_corr.values, 1.0)

    return mean_corr


# =============================================================================
# 2. Load IC summary and ensure rank_ic_nw_icir exists
# =============================================================================

def load_summary_for_ward_selection() -> pd.DataFrame:
    """
    Load factor IC summary.

    Need rank_ic_nw_icir for selecting the representative factor in each cluster.
    """
    if "summary" in globals() and isinstance(summary, pd.DataFrame):
        summary_df = summary.copy()
    else:
        if not SUMMARY_CSV_FOR_SELECTION.exists():
            raise FileNotFoundError(
                f"Cannot find summary CSV: {SUMMARY_CSV_FOR_SELECTION}"
            )
        summary_df = pd.read_csv(SUMMARY_CSV_FOR_SELECTION)

    if SELECTION_SCORE_COL not in summary_df.columns:
        if (
            "daily" in globals()
            and isinstance(daily, pd.DataFrame)
            and "compute_newey_west_icir_table" in globals()
        ):
            print(f"{SELECTION_SCORE_COL} not found. Computing from daily DataFrame ...")
            nw_df = compute_newey_west_icir_table(daily)
            summary_df = summary_df.merge(nw_df, on=["factor", "horizon"], how="left")

        elif DAILY_CSV_FOR_SELECTION.exists() and "compute_newey_west_icir_table" in globals():
            print(f"{SELECTION_SCORE_COL} not found. Computing from daily CSV ...")
            daily_df = pd.read_csv(DAILY_CSV_FOR_SELECTION)
            nw_df = compute_newey_west_icir_table(daily_df)
            summary_df = summary_df.merge(nw_df, on=["factor", "horizon"], how="left")

        else:
            raise ValueError(
                f"{SELECTION_SCORE_COL} not found in summary table. "
                "Please run the IC script with Newey-West Rank ICIR first."
            )

    return summary_df


def build_factor_best_score_for_ward(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each factor, choose the horizon where abs(rank_ic_nw_icir) is highest.
    """
    work = summary_df.copy()

    if "factor" not in work.columns or "horizon" not in work.columns:
        raise ValueError("summary table must contain factor and horizon columns.")

    if SELECTION_SCORE_COL not in work.columns:
        raise ValueError(f"summary table must contain {SELECTION_SCORE_COL}.")

    work[SELECTION_SCORE_COL] = pd.to_numeric(work[SELECTION_SCORE_COL], errors="coerce")
    work["selection_score_abs"] = work[SELECTION_SCORE_COL].abs()

    valid = work.dropna(subset=["factor", "horizon", SELECTION_SCORE_COL]).copy()

    if valid.empty:
        raise ValueError(f"No valid values found for {SELECTION_SCORE_COL}.")

    idx = valid.groupby("factor")["selection_score_abs"].idxmax()
    factor_best = valid.loc[idx].copy()

    keep_cols = [
        "factor",
        "horizon",
        SELECTION_SCORE_COL,
        "selection_score_abs",
        "rank_ic_mean",
        "rank_icir",
        "rank_ic_nw_std",
        "rank_ic_nw_icir",
        "sample_days",
        "avg_obs_per_day",
        "start_day",
        "end_day",
    ]

    factor_best = factor_best[[c for c in keep_cols if c in factor_best.columns]].copy()

    factor_best = factor_best.rename(
        columns={
            "horizon": "best_horizon",
            SELECTION_SCORE_COL: "best_rank_ic_nw_icir",
        }
    )

    factor_best["factor"] = factor_best["factor"].astype(str)
    factor_best["factor_label"] = factor_best["factor"].map(ward_selection_factor_label)
    factor_best["indicator_group"] = factor_best["factor"].map(ward_selection_indicator_group)

    return factor_best.reset_index(drop=True)


# =============================================================================
# 3. Ward clustering groups
# =============================================================================

def build_ward_redundancy_groups_from_corr(
    mean_corr: pd.DataFrame,
    corr_threshold: float = REDUNDANCY_GROUP_THRESHOLD,
) -> tuple[list[list[str]], pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Ward clustering using distance = 1 - abs(corr).

    Cut tree at:
        distance threshold = 1 - corr_threshold

    This is different from the previous connected-component method.
    Previous method has single-link chain effect:
        A-B high corr, B-C high corr -> A/B/C same group.
    Ward clustering is more conservative and usually keeps more factors.
    """
    factors = list(mean_corr.index)

    corr = mean_corr.copy()
    corr = corr.replace([np.inf, -np.inf], np.nan)
    corr = corr.fillna(0.0)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr.values, 1.0)

    distance = 1.0 - corr.abs()
    distance = distance.clip(lower=0.0, upper=1.0)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance.values, 0.0)

    condensed_distance = squareform(distance.values, checks=False)

    linkage_matrix = linkage(
        condensed_distance,
        method="ward",
        optimal_ordering=True,
    )

    distance_threshold = 1.0 - corr_threshold

    cluster_labels = fcluster(
        linkage_matrix,
        t=distance_threshold,
        criterion="distance",
    )

    cluster_df = pd.DataFrame(
        {
            "factor": factors,
            "ward_cluster_raw": cluster_labels,
        }
    )

    # Re-label clusters as G01, G02, ... sorted by group size desc
    tmp = (
        cluster_df
        .groupby("ward_cluster_raw")
        .agg(
            group_size=("factor", "count"),
            first_factor=("factor", "min"),
        )
        .reset_index()
        .sort_values(["group_size", "first_factor"], ascending=[False, True])
        .reset_index(drop=True)
    )

    cluster_id_map = {
        raw: f"G{i + 1:02d}"
        for i, raw in enumerate(tmp["ward_cluster_raw"].tolist())
    }

    cluster_df["group_id"] = cluster_df["ward_cluster_raw"].map(cluster_id_map)

    groups = (
        cluster_df
        .groupby("group_id")["factor"]
        .apply(lambda s: sorted(s.astype(str).tolist()))
        .tolist()
    )

    # High corr pairs are still useful for inspection, but not used to form groups
    high_pair_records = []

    for i, f1 in enumerate(factors):
        for j in range(i + 1, len(factors)):
            f2 = factors[j]
            corr_val = corr.loc[f1, f2]
            abs_corr = abs(corr_val)

            if pd.notna(corr_val) and abs_corr > corr_threshold:
                high_pair_records.append(
                    {
                        "factor_1": f1,
                        "factor_1_label": ward_selection_factor_label(f1),
                        "factor_2": f2,
                        "factor_2_label": ward_selection_factor_label(f2),
                        "mean_spearman_corr": float(corr_val),
                        "abs_corr": float(abs_corr),
                    }
                )

    high_pairs = pd.DataFrame(high_pair_records)

    if not high_pairs.empty:
        high_pairs = high_pairs.sort_values("abs_corr", ascending=False).reset_index(drop=True)

    linkage_df = pd.DataFrame(
        linkage_matrix,
        columns=["cluster_1", "cluster_2", "ward_distance", "sample_count"],
    )

    return groups, high_pairs, linkage_df, linkage_matrix


# =============================================================================
# 4. Select one factor per Ward cluster
# =============================================================================

def select_best_factor_per_ward_group(
    groups: list[list[str]],
    factor_best: pd.DataFrame,
    mean_corr: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each Ward cluster, keep factor with highest abs(NW Rank ICIR).
    """
    score_map = factor_best.set_index("factor")

    group_records = []
    selected_records = []
    group_summary_records = []

    for group_idx, factors in enumerate(groups, start=1):
        group_id = f"G{group_idx:02d}"
        group_size = len(factors)

        rows = []

        for f in factors:
            if f in score_map.index:
                row = score_map.loc[f].to_dict()

                if isinstance(row.get("best_rank_ic_nw_icir"), pd.Series):
                    row = score_map.loc[[f]].iloc[0].to_dict()
            else:
                row = {
                    "factor": f,
                    "factor_label": ward_selection_factor_label(f),
                    "indicator_group": ward_selection_indicator_group(f),
                    "best_horizon": np.nan,
                    "best_rank_ic_nw_icir": np.nan,
                    "selection_score_abs": np.nan,
                    "rank_ic_mean": np.nan,
                    "rank_icir": np.nan,
                    "sample_days": np.nan,
                    "avg_obs_per_day": np.nan,
                    "start_day": np.nan,
                    "end_day": np.nan,
                }

            row["factor"] = f
            row["factor_label"] = ward_selection_factor_label(f)
            row["indicator_group"] = ward_selection_indicator_group(f)
            rows.append(row)

        group_df = pd.DataFrame(rows)

        valid_score = group_df.dropna(subset=["selection_score_abs"]).copy()

        if valid_score.empty:
            selected_factor = factors[0]
        else:
            selected_factor = (
                valid_score
                .sort_values(["selection_score_abs", "factor"], ascending=[False, True])
                .iloc[0]["factor"]
            )

        selected_row = group_df[group_df["factor"] == selected_factor].iloc[0].to_dict()

        # Pairwise corr stats inside group
        pair_abs_corr_values = []

        if group_size >= 2:
            for i, f1 in enumerate(factors):
                for j in range(i + 1, len(factors)):
                    f2 = factors[j]

                    if f1 in mean_corr.index and f2 in mean_corr.columns:
                        pair_abs_corr_values.append(abs(float(mean_corr.loc[f1, f2])))

        max_abs_corr_in_group = (
            max(pair_abs_corr_values)
            if pair_abs_corr_values
            else np.nan
        )

        mean_abs_corr_in_group = (
            float(np.mean(pair_abs_corr_values))
            if pair_abs_corr_values
            else np.nan
        )

        selected_records.append(
            {
                "group_id": group_id,
                "group_size": group_size,
                "selected_factor": selected_factor,
                "selected_factor_label": ward_selection_factor_label(selected_factor),
                "selected_indicator_group": ward_selection_indicator_group(selected_factor),
                "selected_best_horizon": selected_row.get("best_horizon", np.nan),
                "selected_rank_ic_nw_icir": selected_row.get("best_rank_ic_nw_icir", np.nan),
                "selected_abs_rank_ic_nw_icir": selected_row.get("selection_score_abs", np.nan),
                "max_abs_corr_in_group": max_abs_corr_in_group,
                "mean_abs_corr_in_group": mean_abs_corr_in_group,
                "group_factors": ", ".join(factors),
                "group_factor_labels": ", ".join([ward_selection_factor_label(f) for f in factors]),
                "selection_method": (
                    f"Ward clustering on distance=1-|corr|, "
                    f"cut distance={WARD_DISTANCE_THRESHOLD:.4f}, "
                    f"keep max abs({SELECTION_SCORE_COL})"
                ),
            }
        )

        group_summary_records.append(
            {
                "group_id": group_id,
                "group_size": group_size,
                "selected_factor": selected_factor,
                "selected_factor_label": ward_selection_factor_label(selected_factor),
                "max_abs_corr_in_group": max_abs_corr_in_group,
                "mean_abs_corr_in_group": mean_abs_corr_in_group,
                "group_factors": ", ".join(factors),
                "group_factor_labels": ", ".join([ward_selection_factor_label(f) for f in factors]),
            }
        )

        for row in rows:
            group_records.append(
                {
                    "group_id": group_id,
                    "group_size": group_size,
                    "factor": row.get("factor"),
                    "factor_label": row.get("factor_label"),
                    "indicator_group": row.get("indicator_group"),
                    "is_selected": row.get("factor") == selected_factor,
                    "best_horizon": row.get("best_horizon", np.nan),
                    "rank_ic_nw_icir_at_best_horizon": row.get("best_rank_ic_nw_icir", np.nan),
                    "abs_rank_ic_nw_icir_at_best_horizon": row.get("selection_score_abs", np.nan),
                    "rank_ic_mean": row.get("rank_ic_mean", np.nan),
                    "rank_icir": row.get("rank_icir", np.nan),
                    "sample_days": row.get("sample_days", np.nan),
                    "avg_obs_per_day": row.get("avg_obs_per_day", np.nan),
                    "selected_factor": selected_factor,
                    "selected_factor_label": ward_selection_factor_label(selected_factor),
                }
            )

    selected_factors = pd.DataFrame(selected_records)
    group_detail = pd.DataFrame(group_records)
    group_summary = pd.DataFrame(group_summary_records)

    selected_factors = selected_factors.sort_values(
        ["group_size", "selected_abs_rank_ic_nw_icir"],
        ascending=[False, False],
    ).reset_index(drop=True)

    group_detail = group_detail.sort_values(
        ["group_id", "is_selected", "abs_rank_ic_nw_icir_at_best_horizon"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    group_summary = group_summary.sort_values(
        ["group_size", "group_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return selected_factors, group_detail, group_summary


# =============================================================================
# 5. Optional: select by horizon using same Ward clusters
# =============================================================================

def select_best_factor_per_ward_group_by_horizon(
    groups: list[list[str]],
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each horizon and each Ward cluster, select the best factor
    using abs(rank_ic_nw_icir) at that horizon.
    """
    work = summary_df.copy()

    work[SELECTION_SCORE_COL] = pd.to_numeric(work[SELECTION_SCORE_COL], errors="coerce")
    work["selection_score_abs"] = work[SELECTION_SCORE_COL].abs()

    records = []

    horizons = sorted(work["horizon"].dropna().unique())

    for h in horizons:
        h_df = work[work["horizon"] == h].copy()
        h_map = h_df.set_index("factor")

        for group_idx, factors in enumerate(groups, start=1):
            group_id = f"G{group_idx:02d}"

            rows = []

            for f in factors:
                if f in h_map.index:
                    row = h_map.loc[f]

                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]

                    rows.append(
                        {
                            "factor": f,
                            "factor_label": ward_selection_factor_label(f),
                            "rank_ic_nw_icir": row.get(SELECTION_SCORE_COL, np.nan),
                            "abs_rank_ic_nw_icir": abs(row.get(SELECTION_SCORE_COL, np.nan)),
                            "rank_ic_mean": row.get("rank_ic_mean", np.nan),
                            "rank_icir": row.get("rank_icir", np.nan),
                            "sample_days": row.get("sample_days", np.nan),
                        }
                    )
                else:
                    rows.append(
                        {
                            "factor": f,
                            "factor_label": ward_selection_factor_label(f),
                            "rank_ic_nw_icir": np.nan,
                            "abs_rank_ic_nw_icir": np.nan,
                            "rank_ic_mean": np.nan,
                            "rank_icir": np.nan,
                            "sample_days": np.nan,
                        }
                    )

            gdf = pd.DataFrame(rows)
            valid = gdf.dropna(subset=["abs_rank_ic_nw_icir"])

            if valid.empty:
                selected_factor = factors[0]
                selected_row = gdf[gdf["factor"] == selected_factor].iloc[0]
            else:
                selected_row = (
                    valid
                    .sort_values(["abs_rank_ic_nw_icir", "factor"], ascending=[False, True])
                    .iloc[0]
                )
                selected_factor = selected_row["factor"]

            records.append(
                {
                    "horizon": h,
                    "group_id": group_id,
                    "group_size": len(factors),
                    "selected_factor": selected_factor,
                    "selected_factor_label": ward_selection_factor_label(selected_factor),
                    "selected_rank_ic_nw_icir": selected_row["rank_ic_nw_icir"],
                    "selected_abs_rank_ic_nw_icir": selected_row["abs_rank_ic_nw_icir"],
                    "group_factors": ", ".join(factors),
                    "group_factor_labels": ", ".join([ward_selection_factor_label(f) for f in factors]),
                }
            )

    selected_by_horizon = pd.DataFrame(records)

    selected_by_horizon = selected_by_horizon.sort_values(
        ["horizon", "group_size", "selected_abs_rank_ic_nw_icir"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    return selected_by_horizon


# =============================================================================
# 6. Run Ward selection
# =============================================================================

def run_ward_factor_selection_by_nw_rank_icir():
    mean_corr = load_mean_corr_for_ward_selection()
    summary_df = load_summary_for_ward_selection()

    factor_best = build_factor_best_score_for_ward(summary_df)

    groups, high_pairs_0p9, ward_linkage_df, ward_linkage_matrix = build_ward_redundancy_groups_from_corr(
        mean_corr=mean_corr,
        corr_threshold=REDUNDANCY_GROUP_THRESHOLD,
    )

    selected_factors, group_detail, group_summary = select_best_factor_per_ward_group(
        groups=groups,
        factor_best=factor_best,
        mean_corr=mean_corr,
    )

    selected_by_horizon = select_best_factor_per_ward_group_by_horizon(
        groups=groups,
        summary_df=summary_df,
    )

    # Save Ward-specific outputs
    selected_factors.to_csv(WARD_SELECTED_FACTORS_CSV, index=False, encoding="utf-8-sig")
    group_detail.to_csv(WARD_GROUP_DETAIL_CSV, index=False, encoding="utf-8-sig")
    group_summary.to_csv(WARD_GROUP_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    high_pairs_0p9.to_csv(
        SELECTION_OUTDIR / "ward_neutral_factor_high_corr_pairs_abs_gt_0p9.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selected_by_horizon.to_csv(
        SELECTION_OUTDIR / "ward_selected_factors_by_horizon_corr_gt_0p9_rank_ic_nw_icir.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with pd.ExcelWriter(WARD_SELECTION_EXCEL, engine="openpyxl") as writer:
        selected_factors.to_excel(writer, sheet_name="Selected_Factors_Overall", index=False)
        group_detail.to_excel(writer, sheet_name="Group_Detail", index=False)
        group_summary.to_excel(writer, sheet_name="Group_Summary", index=False)
        selected_by_horizon.to_excel(writer, sheet_name="Selected_By_Horizon", index=False)
        high_pairs_0p9.to_excel(writer, sheet_name="High_Corr_Pairs_0p9", index=False)
        factor_best.to_excel(writer, sheet_name="Factor_Best_Score", index=False)
        ward_linkage_df.to_excel(writer, sheet_name="Ward_Linkage", index=False)
        mean_corr.to_excel(writer, sheet_name="Mean_Corr")

    # Save compatibility files so later composite code can read the same filenames
    if OVERWRITE_COMPATIBILITY_FILES:
        selected_factors.to_csv(COMPAT_SELECTED_FACTORS_CSV, index=False, encoding="utf-8-sig")
        group_detail.to_csv(COMPAT_GROUP_DETAIL_CSV, index=False, encoding="utf-8-sig")

        with pd.ExcelWriter(COMPAT_SELECTION_EXCEL, engine="openpyxl") as writer:
            selected_factors.to_excel(writer, sheet_name="Selected_Factors_Overall", index=False)
            group_detail.to_excel(writer, sheet_name="Group_Detail", index=False)
            group_summary.to_excel(writer, sheet_name="Group_Summary", index=False)
            selected_by_horizon.to_excel(writer, sheet_name="Selected_By_Horizon", index=False)
            high_pairs_0p9.to_excel(writer, sheet_name="High_Corr_Pairs_0p9", index=False)
            factor_best.to_excel(writer, sheet_name="Factor_Best_Score", index=False)
            ward_linkage_df.to_excel(writer, sheet_name="Ward_Linkage", index=False)
            mean_corr.to_excel(writer, sheet_name="Mean_Corr")

    print(f"Saved Ward selected factors CSV: {WARD_SELECTED_FACTORS_CSV}")
    print(f"Saved Ward group detail CSV: {WARD_GROUP_DETAIL_CSV}")
    print(f"Saved Ward selection Excel: {WARD_SELECTION_EXCEL}")

    if OVERWRITE_COMPATIBILITY_FILES:
        print("\nCompatibility files overwritten for later composite code:")
        print(f"  {COMPAT_SELECTED_FACTORS_CSV}")
        print(f"  {COMPAT_SELECTION_EXCEL}")
        print(f"  {COMPAT_GROUP_DETAIL_CSV}")

    print("\nWard selected factors to keep:")
    print(
        selected_factors[
            [
                "group_id",
                "group_size",
                "selected_factor",
                "selected_factor_label",
                "selected_best_horizon",
                "selected_rank_ic_nw_icir",
                "selected_abs_rank_ic_nw_icir",
            ]
        ].to_string(index=False)
    )

    print("\nNumber of selected factors:")
    print(len(selected_factors))

    return {
        "selected_factors": selected_factors,
        "group_detail": group_detail,
        "group_summary": group_summary,
        "selected_by_horizon": selected_by_horizon,
        "high_pairs_0p9": high_pairs_0p9,
        "factor_best": factor_best,
        "mean_corr": mean_corr,
        "ward_linkage_df": ward_linkage_df,
        "ward_linkage_matrix": ward_linkage_matrix,
        "groups": groups,
    }


if __name__ == "__main__":
    ward_selection_result = run_ward_factor_selection_by_nw_rank_icir()
# For compatibility with your later composite code:
selection_result = ward_selection_result
