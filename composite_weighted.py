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
    from .factor_selection_ward import *
    from .composite_equal import *
except ImportError:
    from core_base import *
    from core_neutralization import *
    from factor_selection_ward import *
    from composite_equal import *

#Build ICIR-weighted composite factor and compare with equal-weight composite
# Requirement:
# After selecting non-redundant factors, use abs(NW-corrected Rank ICIR)
# as weights instead of equal weight, then run IC again and compare.

from pathlib import Path
import numpy as np
import pandas as pd



WEIGHTED_COMPOSITE_OUTDIR = OUTDIR / "composite_factor_weighted"
WEIGHTED_COMPOSITE_OUTDIR.mkdir(parents=True, exist_ok=True)

SELECTION_OUTDIR = OUTDIR / "neutral_factor_redundancy"

SELECTED_FACTORS_CSV_FOR_WEIGHTED = (
    SELECTION_OUTDIR / "selected_factors_by_corr_gt_0p9_rank_ic_nw_icir.csv"
)

PREVIOUS_SUMMARY_CSV_FOR_WEIGHTED = (
    OUTDIR / "technical_factor_ic_summary_winsor_industry_size_neutral_2005_2026.csv"
)

PREVIOUS_DAILY_CSV_FOR_WEIGHTED = (
    OUTDIR / "technical_factor_daily_ic_winsor_industry_size_neutral_2005_2026.csv"
)

EQUAL_COMPOSITE_COL = "composite_equal_mad_z"
WEIGHTED_COMPOSITE_COL = "composite_weighted_mad_z"

WEIGHTED_COMPOSITE_SUMMARY_CSV = (
    WEIGHTED_COMPOSITE_OUTDIR / "weighted_composite_factor_ic_summary.csv"
)

WEIGHTED_COMPOSITE_DAILY_CSV = (
    WEIGHTED_COMPOSITE_OUTDIR / "weighted_composite_factor_daily_ic.csv"
)

WEIGHTED_COMPOSITE_COMPARE_EXCEL = (
    WEIGHTED_COMPOSITE_OUTDIR / "weighted_composite_factor_ic_comparison.xlsx"
)

# Median / MAD z-score settings
MAD_SCALE = 1.4826
MAD_Z_CAP = None

# If None, require all selected factors to be valid.
# If you want more coverage, set e.g. MIN_VALID_SELECTED_FACTOR_COUNT = 3
MIN_VALID_SELECTED_FACTOR_COUNT = None

# Weight rule:
# For each selected factor, find the horizon where abs(rank_ic_nw_icir) is largest.
# Use abs(rank_ic_nw_icir) as weight.
# Use sign(rank_ic_nw_icir) as direction.
WEIGHT_SCORE_COL = "rank_ic_nw_icir"


# =============================================================================
# Helper labels
# =============================================================================

def weighted_composite_factor_label(factor: str) -> str:
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


def weighted_composite_indicator_group(factor: str) -> str:
    factor = str(factor)

    if factor in [EQUAL_COMPOSITE_COL, WEIGHTED_COMPOSITE_COL]:
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


# =============================================================================
# 1. Load selected factors
# =============================================================================

def load_selected_factors_for_weighted_composite() -> pd.DataFrame:
    """
    Load selected non-redundant factors.

    Expected file:
        selected_factors_by_corr_gt_0p9_rank_ic_nw_icir.csv

    Expected column:
        selected_factor
    """
    if (
        "selection_result" in globals()
        and isinstance(selection_result, dict)
        and "selected_factors" in selection_result
    ):
        selected_df = selection_result["selected_factors"].copy()
    else:
        if not SELECTED_FACTORS_CSV_FOR_WEIGHTED.exists():
            raise FileNotFoundError(
                f"Cannot find selected factor file: {SELECTED_FACTORS_CSV_FOR_WEIGHTED}. "
                "Please run the redundancy-selection chunk first."
            )
        selected_df = pd.read_csv(SELECTED_FACTORS_CSV_FOR_WEIGHTED)

    if "selected_factor" in selected_df.columns:
        col = "selected_factor"
    elif "factor" in selected_df.columns:
        col = "factor"
    else:
        raise ValueError("Selected factor file must contain selected_factor or factor column.")

    selected_factors = (
        selected_df[col]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    out = pd.DataFrame({"factor": selected_factors})
    out["factor_label"] = out["factor"].map(weighted_composite_factor_label)
    out["indicator_group"] = out["factor"].map(weighted_composite_indicator_group)

    return out


# =============================================================================
# 2. Load previous summary and build weights
# =============================================================================

def load_previous_summary_for_weighted_composite() -> pd.DataFrame:
    """
    Load previous single-factor IC summary.

    Need rank_ic_nw_icir to determine:
    - direction = sign(rank_ic_nw_icir)
    - weight = abs(rank_ic_nw_icir)
    """
    if "summary" in globals() and isinstance(summary, pd.DataFrame):
        prev_summary = summary.copy()
    else:
        if not PREVIOUS_SUMMARY_CSV_FOR_WEIGHTED.exists():
            raise FileNotFoundError(
                f"Cannot find previous summary CSV: {PREVIOUS_SUMMARY_CSV_FOR_WEIGHTED}"
            )
        prev_summary = pd.read_csv(PREVIOUS_SUMMARY_CSV_FOR_WEIGHTED)

    if WEIGHT_SCORE_COL not in prev_summary.columns:
        print(f"{WEIGHT_SCORE_COL} not found. Computing from previous daily IC ...")

        if "daily" in globals() and isinstance(daily, pd.DataFrame):
            prev_daily = daily.copy()
        else:
            if not PREVIOUS_DAILY_CSV_FOR_WEIGHTED.exists():
                raise FileNotFoundError(
                    f"Cannot find previous daily IC CSV: {PREVIOUS_DAILY_CSV_FOR_WEIGHTED}"
                )
            prev_daily = pd.read_csv(PREVIOUS_DAILY_CSV_FOR_WEIGHTED)

        if "compute_newey_west_icir_table" not in globals():
            raise NameError(
                "compute_newey_west_icir_table() is not defined. "
                "Please run the Newey-West function chunk first."
            )

        nw_df = compute_newey_west_icir_table(prev_daily)

        prev_summary = prev_summary.merge(
            nw_df,
            on=["factor", "horizon"],
            how="left",
        )

    return prev_summary


def build_weight_table(
    selected_factors_df: pd.DataFrame,
    prev_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each selected factor:
    1. Find the horizon where abs(rank_ic_nw_icir) is largest.
    2. direction = sign(rank_ic_nw_icir)
    3. raw_weight = abs(rank_ic_nw_icir)
    4. normalized_weight = raw_weight / sum(raw_weight)
    """
    selected = selected_factors_df["factor"].astype(str).tolist()

    work = prev_summary[prev_summary["factor"].astype(str).isin(selected)].copy()
    work[WEIGHT_SCORE_COL] = pd.to_numeric(work[WEIGHT_SCORE_COL], errors="coerce")
    work["abs_weight_score"] = work[WEIGHT_SCORE_COL].abs()

    records = []

    for factor in selected:
        tmp = work[work["factor"].astype(str) == factor].copy()

        if tmp.empty or tmp["abs_weight_score"].dropna().empty:
            best_horizon = np.nan
            score = np.nan
            abs_score = np.nan
            direction = 1.0
        else:
            best = tmp.sort_values("abs_weight_score", ascending=False).iloc[0]
            best_horizon = best["horizon"]
            score = best[WEIGHT_SCORE_COL]
            abs_score = abs(score)
            direction = np.sign(score)

            if not np.isfinite(direction) or direction == 0:
                direction = 1.0

        records.append(
            {
                "factor": factor,
                "factor_label": weighted_composite_factor_label(factor),
                "indicator_group": weighted_composite_indicator_group(factor),
                "best_horizon_for_weight": best_horizon,
                "rank_ic_nw_icir_for_weight": score,
                "abs_rank_ic_nw_icir_for_weight": abs_score,
                "direction": direction,
            }
        )

    weights = pd.DataFrame(records)

    # If any factor has missing score, set its raw weight to 0
    weights["raw_weight"] = pd.to_numeric(
        weights["abs_rank_ic_nw_icir_for_weight"],
        errors="coerce",
    ).fillna(0.0)

    weight_sum = weights["raw_weight"].sum()

    if weight_sum <= 0:
        print("Warning: all raw weights are zero or missing. Falling back to equal weights.")
        weights["normalized_weight"] = 1.0 / len(weights)
    else:
        weights["normalized_weight"] = weights["raw_weight"] / weight_sum

    # Signed weight is used in weighted composite.
    weights["signed_weight"] = weights["direction"] * weights["normalized_weight"]

    # Equal signed weight is used in equal-weight comparison.
    weights["equal_signed_weight"] = weights["direction"] / len(weights)

    return weights


# =============================================================================
# 3. Median / MAD z-score and composite construction
# =============================================================================

def _mad_zscore_by_day(
    df: pd.DataFrame,
    col: str,
    valid_universe: pd.Series,
    mad_scale: float = MAD_SCALE,
    z_cap: float | None = MAD_Z_CAP,
) -> pd.Series:
    """
    Cross-sectional robust z-score by trade_day:

        z = (x - median) / (1.4826 * MAD)

    MAD = median(|x - median|)
    """
    x = pd.to_numeric(df[col], errors="coerce").where(valid_universe)

    med = x.groupby(df["trade_day"]).transform("median")
    mad = (x - med).abs().groupby(df["trade_day"]).transform("median")

    denom = mad_scale * mad
    z = (x - med) / denom

    z = z.where(np.isfinite(z) & np.isfinite(denom) & (denom > 0))

    if z_cap is not None:
        z = z.clip(lower=-z_cap, upper=z_cap)

    return z


def add_equal_and_weighted_composites(
    df: pd.DataFrame,
    weights: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """
    Add two composite factors:

    1. Equal-weight composite:
        composite_equal = mean(direction_j * z_j)

    2. ICIR-weighted composite:
        composite_weighted = sum(normalized_abs_icir_j * direction_j * z_j)

    z_j is median/MAD z-score of factor j on each day.
    """
    df = df.copy()

    selected_factors = weights["factor"].astype(str).tolist()

    missing = [f for f in selected_factors if f not in df.columns]
    if missing:
        raise ValueError(f"Selected factors not found in factor panel: {missing}")

    if not INCLUDE_SUSPENDED and "volume" in df.columns:
        valid_universe = df["volume"].fillna(0) > 0
    else:
        valid_universe = pd.Series(True, index=df.index)

    z_cols = []
    signed_z_cols = []

    direction_map = weights.set_index("factor")["direction"].to_dict()
    equal_weight_map = weights.set_index("factor")["equal_signed_weight"].to_dict()
    signed_weight_map = weights.set_index("factor")["signed_weight"].to_dict()

    weighted_component_cols = []
    equal_component_cols = []

    for factor in selected_factors:
        z_col = f"z_mad__{factor}"
        signed_z_col = f"signed_z__{factor}"
        weighted_component_col = f"weighted_component__{factor}"
        equal_component_col = f"equal_component__{factor}"

        df[z_col] = _mad_zscore_by_day(
            df=df,
            col=factor,
            valid_universe=valid_universe,
            mad_scale=MAD_SCALE,
            z_cap=MAD_Z_CAP,
        )

        direction = direction_map.get(factor, 1.0)

        # Direction-adjusted z-score
        df[signed_z_col] = df[z_col] * direction

        # Equal contribution = direction * z / N
        df[equal_component_col] = df[z_col] * equal_weight_map.get(factor, 0.0)

        # Weighted contribution = direction * normalized_abs_icir_weight * z
        df[weighted_component_col] = df[z_col] * signed_weight_map.get(factor, 0.0)

        z_cols.append(z_col)
        signed_z_cols.append(signed_z_col)
        equal_component_cols.append(equal_component_col)
        weighted_component_cols.append(weighted_component_col)

    if MIN_VALID_SELECTED_FACTOR_COUNT is None:
        min_valid_count = len(selected_factors)
    else:
        min_valid_count = int(MIN_VALID_SELECTED_FACTOR_COUNT)

    valid_count = df[z_cols].notna().sum(axis=1)

    df[EQUAL_COMPOSITE_COL] = df[equal_component_cols].sum(axis=1)
    df[WEIGHTED_COMPOSITE_COL] = df[weighted_component_cols].sum(axis=1)

    df.loc[valid_count < min_valid_count, EQUAL_COMPOSITE_COL] = np.nan
    df.loc[valid_count < min_valid_count, WEIGHTED_COMPOSITE_COL] = np.nan

    composite_stats = (
        pd.DataFrame(
            {
                "trade_day": df["trade_day"],
                "valid_selected_factor_count": valid_count,
                "equal_composite_notna": df[EQUAL_COMPOSITE_COL].notna(),
                "weighted_composite_notna": df[WEIGHTED_COMPOSITE_COL].notna(),
            }
        )
        .groupby("trade_day", as_index=False)
        .agg(
            avg_valid_factor_count=("valid_selected_factor_count", "mean"),
            min_valid_factor_count=("valid_selected_factor_count", "min"),
            max_valid_factor_count=("valid_selected_factor_count", "max"),
            equal_composite_obs=("equal_composite_notna", "sum"),
            weighted_composite_obs=("weighted_composite_notna", "sum"),
        )
    )

    aux_cols = z_cols + signed_z_cols + equal_component_cols + weighted_component_cols

    return df, aux_cols, composite_stats


# =============================================================================
# 4. Comparison table
# =============================================================================

def build_weighted_composite_comparison(
    composite_summary: pd.DataFrame,
    selected_single_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare:
    1. Equal-weight composite
    2. ICIR-weighted composite
    3. Best selected single factor
    4. Average selected single factor
    """
    comp = composite_summary.copy()
    selected = selected_single_summary.copy()

    if "rank_ic_nw_icir" not in comp.columns:
        comp["rank_ic_nw_icir"] = np.nan
    if "rank_ic_nw_icir" not in selected.columns:
        selected["rank_ic_nw_icir"] = np.nan

    metrics = [
        "rank_ic_mean",
        "rank_icir",
        "rank_ic_nw_icir",
        "neutral_rank_ic_mean",
        "neutral_rank_icir",
        "rank_ic_positive_ratio",
        "sample_days",
        "avg_obs_per_day",
    ]

    records = []

    horizons = sorted(comp["horizon"].dropna().unique())

    for h in horizons:
        comp_h = comp[comp["horizon"] == h].copy()
        selected_h = selected[selected["horizon"] == h].copy()

        equal_row = comp_h[comp_h["factor"] == EQUAL_COMPOSITE_COL]
        weighted_row = comp_h[comp_h["factor"] == WEIGHTED_COMPOSITE_COL]

        row = {"horizon": h}

        if not equal_row.empty:
            equal_dict = equal_row.iloc[0].to_dict()
            for m in metrics:
                row[f"equal_{m}"] = equal_dict.get(m, np.nan)

        if not weighted_row.empty:
            weighted_dict = weighted_row.iloc[0].to_dict()
            for m in metrics:
                row[f"weighted_{m}"] = weighted_dict.get(m, np.nan)

        if not selected_h.empty:
            selected_h["_abs_score"] = selected_h["rank_ic_nw_icir"].abs()

            if selected_h["_abs_score"].dropna().empty:
                best = selected_h.iloc[0]
            else:
                best = selected_h.sort_values("_abs_score", ascending=False).iloc[0]

            row["best_single_factor"] = best.get("factor", np.nan)
            row["best_single_factor_label"] = best.get("factor_label", np.nan)

            for m in metrics:
                row[f"best_single_{m}"] = best.get(m, np.nan)

            for m in metrics:
                row[f"avg_selected_single_{m}"] = (
                    selected_h[m].mean() if m in selected_h.columns else np.nan
                )

        row["diff_weighted_minus_equal_rank_ic_nw_icir"] = (
            row.get("weighted_rank_ic_nw_icir", np.nan)
            - row.get("equal_rank_ic_nw_icir", np.nan)
        )

        row["diff_weighted_minus_best_single_rank_ic_nw_icir"] = (
            row.get("weighted_rank_ic_nw_icir", np.nan)
            - row.get("best_single_rank_ic_nw_icir", np.nan)
        )

        row["diff_weighted_minus_avg_single_rank_ic_nw_icir"] = (
            row.get("weighted_rank_ic_nw_icir", np.nan)
            - row.get("avg_selected_single_rank_ic_nw_icir", np.nan)
        )

        records.append(row)

    return pd.DataFrame(records)


# =============================================================================
# 5. Run weighted composite IC
# =============================================================================

def run_weighted_composite_factor_ic():
    selected_factors_df = load_selected_factors_for_weighted_composite()
    prev_summary = load_previous_summary_for_weighted_composite()

    weights = build_weight_table(
        selected_factors_df=selected_factors_df,
        prev_summary=prev_summary,
    )

    selected_factors = weights["factor"].astype(str).tolist()

    print("\nSelected factor weights:")
    print(weights.to_string(index=False))

    # -------------------------------------------------------------------------
    # Build stock-level factor panel
    # -------------------------------------------------------------------------
    files = find_input_files(
        DATA_DIR,
        years=YEARS,
        file_pattern=FILE_PATTERN,
        allow_missing=ALLOW_MISSING_FILES,
    )

    print(f"\nFound {len(files)} parquet files.")
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

    df, all_factor_cols = add_technical_factors(df, price_col="price")

    missing_selected = [f for f in selected_factors if f not in all_factor_cols]
    if missing_selected:
        raise ValueError(f"These selected factors were not generated: {missing_selected}")

    # -------------------------------------------------------------------------
    # Add equal-weight and ICIR-weighted composite factors
    # -------------------------------------------------------------------------
    df, aux_cols, composite_stats = add_equal_and_weighted_composites(
        df=df,
        weights=weights,
    )

    # -------------------------------------------------------------------------
    # Add industry + size exposures for neutral IC
    # -------------------------------------------------------------------------
    exposure_cols = []

    if DO_INDUSTRY_SIZE_NEUTRAL_IC:
        df, exposure_cols, industry_cols = add_industry_size_exposures(
            df,
            industry_xlsx=INDUSTRY_XLSX,
            market_cap_xlsx=MARKET_CAP_XLSX,
            base_date=MARKET_CAP_BASE_DATE,
            price_col="price",
        )
    else:
        industry_cols = []

    # -------------------------------------------------------------------------
    # Reduce memory and calculate forward returns
    # -------------------------------------------------------------------------
    keep_cols = [
        "code",
        "trade_day",
        "price",
        EQUAL_COMPOSITE_COL,
        WEIGHTED_COMPOSITE_COL,
    ]

    if USE_NEXT_DAY_VWAP_RETURN and "vwap_price" in df.columns:
        keep_cols.append("vwap_price")

    if "volume" in df.columns:
        keep_cols.append("volume")

    if DO_INDUSTRY_SIZE_NEUTRAL_IC:
        keep_cols += exposure_cols
        if "industry_valid" in df.columns:
            keep_cols.append("industry_valid")

    keep_cols = list(dict.fromkeys([c for c in keep_cols if c in df.columns]))

    df_ic = df[keep_cols].copy()

    df_ic = add_forward_returns(
        df_ic,
        price_col="price",
        return_price_col="vwap_price",
        horizons=HORIZONS,
        use_next_day_vwap=USE_NEXT_DAY_VWAP_RETURN,
    )

    # -------------------------------------------------------------------------
    # Run IC for equal and weighted composite factors
    # -------------------------------------------------------------------------
    composite_factor_cols = [EQUAL_COMPOSITE_COL, WEIGHTED_COMPOSITE_COL]

    composite_summary, composite_daily = compute_ic_summary_fast(
        df_ic,
        factor_cols=composite_factor_cols,
        horizons=HORIZONS,
        min_obs=MIN_OBS,
        tradable_only=not INCLUDE_SUSPENDED,
        exposure_cols=exposure_cols if DO_INDUSTRY_SIZE_NEUTRAL_IC else None,
        neutralize_return_too=NEUTRALIZE_RETURN_TOO,
    )

    # -------------------------------------------------------------------------
    # Add Newey-West correction
    # -------------------------------------------------------------------------
    if "compute_newey_west_icir_table" in globals():
        composite_nw = compute_newey_west_icir_table(composite_daily)

        composite_summary = composite_summary.merge(
            composite_nw,
            on=["factor", "horizon"],
            how="left",
        )

        composite_summary["icir_nw_diff"] = (
            composite_summary["ic_nw_icir"] - composite_summary["icir"]
        )
        composite_summary["rank_icir_nw_diff"] = (
            composite_summary["rank_ic_nw_icir"] - composite_summary["rank_icir"]
        )

        composite_summary["icir_nw_change_pct"] = np.where(
            composite_summary["icir"].abs() > 1e-12,
            composite_summary["ic_nw_icir"] / composite_summary["icir"] - 1,
            np.nan,
        )

        composite_summary["rank_icir_nw_change_pct"] = np.where(
            composite_summary["rank_icir"].abs() > 1e-12,
            composite_summary["rank_ic_nw_icir"] / composite_summary["rank_icir"] - 1,
            np.nan,
        )

    composite_summary["factor_label"] = composite_summary["factor"].map(
        {
            EQUAL_COMPOSITE_COL: "Equal-weight composite, MAD-z",
            WEIGHTED_COMPOSITE_COL: "Abs(NW Rank ICIR)-weighted composite, MAD-z",
        }
    )
    composite_summary["indicator_group"] = "Composite"

    # -------------------------------------------------------------------------
    # Compare with selected single factors
    # -------------------------------------------------------------------------
    selected_single_summary = prev_summary[
        prev_summary["factor"].astype(str).isin(selected_factors)
    ].copy()

    selected_single_summary["factor_label"] = selected_single_summary["factor"].map(
        weighted_composite_factor_label
    )
    selected_single_summary["indicator_group"] = selected_single_summary["factor"].map(
        weighted_composite_indicator_group
    )

    comparison_by_horizon = build_weighted_composite_comparison(
        composite_summary=composite_summary,
        selected_single_summary=selected_single_summary,
    )

    comparison_long = pd.concat(
        [
            selected_single_summary.assign(factor_type="selected_single_factor"),
            composite_summary.assign(factor_type="composite_factor"),
        ],
        ignore_index=True,
        sort=False,
    )

    # -------------------------------------------------------------------------
    # Save output
    # -------------------------------------------------------------------------
    composite_summary.to_csv(
        WEIGHTED_COMPOSITE_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    composite_daily.to_csv(
        WEIGHTED_COMPOSITE_DAILY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    with pd.ExcelWriter(WEIGHTED_COMPOSITE_COMPARE_EXCEL, engine="openpyxl") as writer:
        composite_summary.to_excel(writer, sheet_name="Composite_Summary", index=False)
        comparison_by_horizon.to_excel(writer, sheet_name="Comparison_By_Horizon", index=False)
        comparison_long.to_excel(writer, sheet_name="Comparison_Long", index=False)
        weights.to_excel(writer, sheet_name="Factor_Weights", index=False)
        selected_single_summary.to_excel(writer, sheet_name="Selected_Single_Factors", index=False)
        composite_stats.to_excel(writer, sheet_name="Composite_Stats_By_Day", index=False)
        composite_daily.to_excel(writer, sheet_name="Composite_Daily_IC", index=False)

    print(f"\nSaved weighted composite summary CSV: {WEIGHTED_COMPOSITE_SUMMARY_CSV}")
    print(f"Saved weighted composite daily IC CSV: {WEIGHTED_COMPOSITE_DAILY_CSV}")
    print(f"Saved weighted composite comparison Excel: {WEIGHTED_COMPOSITE_COMPARE_EXCEL}")

    print("\nComposite summary:")
    print(composite_summary.to_string(index=False))

    print("\nComparison by horizon:")
    print(comparison_by_horizon.to_string(index=False))

    return {
        "weights": weights,
        "composite_summary": composite_summary,
        "composite_daily": composite_daily,
        "comparison_by_horizon": comparison_by_horizon,
        "comparison_long": comparison_long,
        "selected_single_summary": selected_single_summary,
        "composite_stats": composite_stats,
    }



if __name__ == "__main__":
    weighted_composite_result = run_weighted_composite_factor_ic()