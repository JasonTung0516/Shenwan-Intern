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
except ImportError:
    from core_base import *
    from core_neutralization import *
    from factor_selection_ward import *

# Requirement:
# 1. Use selected non-redundant factors.
# 2. Cross-sectionally standardize each selected factor using median and MAD.
# 3. Equal-weight combine them into one composite factor.
# 4. Run IC using the same previous logic.
# 5. Compare composite factor with previous selected single factors.

from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

COMPOSITE_OUTDIR = OUTDIR / "composite_factor"
COMPOSITE_OUTDIR.mkdir(parents=True, exist_ok=True)

SELECTION_OUTDIR = OUTDIR / "neutral_factor_redundancy"

SELECTED_FACTORS_CSV_FOR_COMPOSITE = (
    SELECTION_OUTDIR / "selected_factors_by_corr_gt_0p9_rank_ic_nw_icir.csv"
)

PREVIOUS_SUMMARY_CSV_FOR_COMPOSITE = (
    OUTDIR / "technical_factor_ic_summary_winsor_industry_size_neutral_2005_2026.csv"
)

PREVIOUS_DAILY_CSV_FOR_COMPOSITE = (
    OUTDIR / "technical_factor_daily_ic_winsor_industry_size_neutral_2005_2026.csv"
)

COMPOSITE_FACTOR_COL = "composite_selected_mad_z"

COMPOSITE_SUMMARY_CSV = (
    COMPOSITE_OUTDIR / "composite_factor_ic_summary.csv"
)

COMPOSITE_DAILY_CSV = (
    COMPOSITE_OUTDIR / "composite_factor_daily_ic.csv"
)

COMPOSITE_COMPARE_EXCEL = (
    COMPOSITE_OUTDIR / "composite_factor_ic_comparison.xlsx"
)

# Median / MAD z-score config
MAD_SCALE = 1.4826

# If True, multiply each selected factor by sign(rank_ic_nw_icir),
# so factors with negative IC are reversed before equal-weight combination.
ALIGN_SELECTED_FACTORS_BY_NW_RANK_ICIR = True

# If None, require all selected factors to be valid for composite factor.
# If you want more coverage, set it to a number, e.g. 3.
MIN_VALID_SELECTED_FACTOR_COUNT = None

# Optional cap for robust z-score. Set None if no cap.
MAD_Z_CAP = None


# =============================================================================
# Helper labels
# =============================================================================

def composite_factor_label(factor: str) -> str:
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


def composite_indicator_group(factor: str) -> str:
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

    if factor == COMPOSITE_FACTOR_COL:
        return "Composite"

    return "Other"


# =============================================================================
# 1. Load selected factors
# =============================================================================

def load_selected_factors_for_composite() -> pd.DataFrame:
    """
    Load selected factors from the previous redundancy-selection step.

    Expected selected file:
        selected_factors_by_corr_gt_0p9_rank_ic_nw_icir.csv

    Main column:
        selected_factor
    """
    if (
        "selection_result" in globals()
        and isinstance(selection_result, dict)
        and "selected_factors" in selection_result
    ):
        selected_df = selection_result["selected_factors"].copy()
    else:
        if not SELECTED_FACTORS_CSV_FOR_COMPOSITE.exists():
            raise FileNotFoundError(
                f"Cannot find selected factors file: {SELECTED_FACTORS_CSV_FOR_COMPOSITE}. "
                "Please run the factor redundancy selection chunk first."
            )
        selected_df = pd.read_csv(SELECTED_FACTORS_CSV_FOR_COMPOSITE)

    if "selected_factor" in selected_df.columns:
        factor_col = "selected_factor"
    elif "factor" in selected_df.columns:
        factor_col = "factor"
    else:
        raise ValueError(
            "Selected factor file must contain either 'selected_factor' or 'factor' column."
        )

    selected_factors = (
        selected_df[factor_col]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    out = pd.DataFrame({"factor": selected_factors})
    out["factor_label"] = out["factor"].map(composite_factor_label)
    out["indicator_group"] = out["factor"].map(composite_indicator_group)

    return out


# =============================================================================
# 2. Load previous IC summary and ensure NW Rank ICIR exists
# =============================================================================

def load_previous_summary_for_composite() -> pd.DataFrame:
    """
    Load previous single-factor IC summary.

    Need rank_ic_nw_icir for factor direction and comparison.
    """
    if "summary" in globals() and isinstance(summary, pd.DataFrame):
        prev_summary = summary.copy()
    else:
        if not PREVIOUS_SUMMARY_CSV_FOR_COMPOSITE.exists():
            raise FileNotFoundError(
                f"Cannot find previous summary CSV: {PREVIOUS_SUMMARY_CSV_FOR_COMPOSITE}"
            )
        prev_summary = pd.read_csv(PREVIOUS_SUMMARY_CSV_FOR_COMPOSITE)

    if "rank_ic_nw_icir" not in prev_summary.columns:
        print("rank_ic_nw_icir not found in previous summary. Computing from daily IC ...")

        if "daily" in globals() and isinstance(daily, pd.DataFrame):
            prev_daily = daily.copy()
        else:
            if not PREVIOUS_DAILY_CSV_FOR_COMPOSITE.exists():
                raise FileNotFoundError(
                    f"Cannot find previous daily IC CSV: {PREVIOUS_DAILY_CSV_FOR_COMPOSITE}"
                )
            prev_daily = pd.read_csv(PREVIOUS_DAILY_CSV_FOR_COMPOSITE)

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

        if "rank_icir" in prev_summary.columns and "rank_ic_nw_icir" in prev_summary.columns:
            prev_summary["rank_icir_nw_diff"] = (
                prev_summary["rank_ic_nw_icir"] - prev_summary["rank_icir"]
            )
            prev_summary["rank_icir_nw_change_pct"] = np.where(
                prev_summary["rank_icir"].abs() > 1e-12,
                prev_summary["rank_ic_nw_icir"] / prev_summary["rank_icir"] - 1,
                np.nan,
            )

    return prev_summary


def build_selected_factor_direction_table(
    selected_factors_df: pd.DataFrame,
    prev_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each selected factor, find the horizon with max abs(rank_ic_nw_icir).
    Use the sign of rank_ic_nw_icir as the direction.
    """
    selected = selected_factors_df["factor"].astype(str).tolist()

    score_col = "rank_ic_nw_icir"

    if score_col not in prev_summary.columns:
        raise ValueError(f"Previous summary must contain {score_col}.")

    work = prev_summary[prev_summary["factor"].astype(str).isin(selected)].copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work["abs_rank_ic_nw_icir"] = work[score_col].abs()

    rows = []

    for factor in selected:
        tmp = work[work["factor"].astype(str) == factor].copy()

        if tmp.empty or tmp["abs_rank_ic_nw_icir"].dropna().empty:
            best_horizon = np.nan
            best_score = np.nan
            direction = 1.0
        else:
            best = tmp.sort_values("abs_rank_ic_nw_icir", ascending=False).iloc[0]
            best_horizon = best["horizon"]
            best_score = best[score_col]
            direction = np.sign(best_score)

            if not np.isfinite(direction) or direction == 0:
                direction = 1.0

        rows.append(
            {
                "factor": factor,
                "factor_label": composite_factor_label(factor),
                "indicator_group": composite_indicator_group(factor),
                "best_horizon_for_direction": best_horizon,
                "rank_ic_nw_icir_at_best_horizon": best_score,
                "direction": direction,
                "direction_note": (
                    "multiplied by sign(rank_ic_nw_icir)"
                    if ALIGN_SELECTED_FACTORS_BY_NW_RANK_ICIR
                    else "not direction-adjusted"
                ),
            }
        )

    direction_df = pd.DataFrame(rows)

    if not ALIGN_SELECTED_FACTORS_BY_NW_RANK_ICIR:
        direction_df["direction"] = 1.0

    direction_df["equal_weight"] = direction_df["direction"] / len(direction_df)

    return direction_df


# =============================================================================
# 3. Median / MAD z-score and composite factor
# =============================================================================

def _cross_sectional_mad_zscore(
    df: pd.DataFrame,
    col: str,
    valid_universe: pd.Series,
    mad_scale: float = MAD_SCALE,
    z_cap: float | None = MAD_Z_CAP,
) -> pd.Series:
    """
    Cross-sectional robust z-score by trade_day:

        z = (x - median(x)) / (1.4826 * MAD)

    where:
        MAD = median(|x - median(x)|)

    Only valid_universe rows are used in median and MAD calculation.
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


def add_selected_composite_factor(
    df: pd.DataFrame,
    selected_direction_df: pd.DataFrame,
    composite_col: str = COMPOSITE_FACTOR_COL,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """
    Add MAD z-scored selected factors and equal-weight composite factor.

    Composite:
        composite = mean(direction_j * z_j)

    direction_j is sign(rank_ic_nw_icir) if ALIGN_SELECTED_FACTORS_BY_NW_RANK_ICIR=True.
    """
    df = df.copy()

    selected_factors = selected_direction_df["factor"].astype(str).tolist()

    missing = [f for f in selected_factors if f not in df.columns]
    if missing:
        raise ValueError(
            f"Selected factors not found in factor panel: {missing}. "
            "Please check selected factor names and technical factor columns."
        )

    if not INCLUDE_SUSPENDED and "volume" in df.columns:
        valid_universe = df["volume"].fillna(0) > 0
    else:
        valid_universe = pd.Series(True, index=df.index)

    z_cols = []
    signed_z_cols = []

    direction_map = selected_direction_df.set_index("factor")["direction"].to_dict()

    for factor in selected_factors:
        z_col = f"z_mad__{factor}"
        signed_z_col = f"signed_z__{factor}"

        df[z_col] = _cross_sectional_mad_zscore(
            df=df,
            col=factor,
            valid_universe=valid_universe,
            mad_scale=MAD_SCALE,
            z_cap=MAD_Z_CAP,
        )

        direction = direction_map.get(factor, 1.0)

        df[signed_z_col] = df[z_col] * direction

        z_cols.append(z_col)
        signed_z_cols.append(signed_z_col)

    if MIN_VALID_SELECTED_FACTOR_COUNT is None:
        min_valid_count = len(signed_z_cols)
    else:
        min_valid_count = int(MIN_VALID_SELECTED_FACTOR_COUNT)

    valid_count = df[signed_z_cols].notna().sum(axis=1)

    df[composite_col] = df[signed_z_cols].mean(axis=1)
    df.loc[valid_count < min_valid_count, composite_col] = np.nan

    composite_stats = (
        pd.DataFrame(
            {
                "trade_day": df["trade_day"],
                "valid_selected_factor_count": valid_count,
                "composite_notna": df[composite_col].notna(),
            }
        )
        .groupby("trade_day", as_index=False)
        .agg(
            avg_valid_factor_count=("valid_selected_factor_count", "mean"),
            min_valid_factor_count=("valid_selected_factor_count", "min"),
            max_valid_factor_count=("valid_selected_factor_count", "max"),
            composite_obs=("composite_notna", "sum"),
        )
    )

    return df, z_cols + signed_z_cols, composite_stats


# =============================================================================
# 4. Build panel and run composite IC
# =============================================================================

def run_composite_factor_ic():
    selected_factors_df = load_selected_factors_for_composite()
    prev_summary = load_previous_summary_for_composite()

    selected_direction_df = build_selected_factor_direction_table(
        selected_factors_df=selected_factors_df,
        prev_summary=prev_summary,
    )

    selected_factors = selected_direction_df["factor"].astype(str).tolist()

    print("\nSelected factors used in composite:")
    print(selected_direction_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # Build factor panel
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
    # Composite factor
    # -------------------------------------------------------------------------
    df, composite_aux_cols, composite_stats = add_selected_composite_factor(
        df=df,
        selected_direction_df=selected_direction_df,
        composite_col=COMPOSITE_FACTOR_COL,
    )

    # -------------------------------------------------------------------------
    # Industry + size exposure for neutral IC
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


    keep_cols = ["code", "trade_day", "price", COMPOSITE_FACTOR_COL]

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

    composite_summary, composite_daily = compute_ic_summary_fast(
        df_ic,
        factor_cols=[COMPOSITE_FACTOR_COL],
        horizons=HORIZONS,
        min_obs=MIN_OBS,
        tradable_only=not INCLUDE_SUSPENDED,
        exposure_cols=exposure_cols if DO_INDUSTRY_SIZE_NEUTRAL_IC else None,
        neutralize_return_too=NEUTRALIZE_RETURN_TOO,
    )

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

    composite_summary["factor_label"] = "Composite selected factor, MAD-z equal weight"
    composite_summary["indicator_group"] = "Composite"

    selected_single_summary = prev_summary[
        prev_summary["factor"].astype(str).isin(selected_factors)
    ].copy()

    selected_single_summary["factor_label"] = selected_single_summary["factor"].map(
        composite_factor_label
    )
    selected_single_summary["indicator_group"] = selected_single_summary["factor"].map(
        composite_indicator_group
    )

    comparison_long = pd.concat(
        [
            selected_single_summary.assign(factor_type="selected_single_factor"),
            composite_summary.assign(factor_type="composite_factor"),
        ],
        ignore_index=True,
        sort=False,
    )

    comparison_by_horizon = build_composite_comparison_by_horizon(
        composite_summary=composite_summary,
        selected_single_summary=selected_single_summary,
    )

    composite_summary.to_csv(
        COMPOSITE_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    composite_daily.to_csv(
        COMPOSITE_DAILY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    with pd.ExcelWriter(COMPOSITE_COMPARE_EXCEL, engine="openpyxl") as writer:
        composite_summary.to_excel(writer, sheet_name="Composite_Summary", index=False)
        comparison_by_horizon.to_excel(writer, sheet_name="Comparison_By_Horizon", index=False)
        comparison_long.to_excel(writer, sheet_name="Comparison_Long", index=False)
        selected_direction_df.to_excel(writer, sheet_name="Selected_Factors_Weights", index=False)
        selected_single_summary.to_excel(writer, sheet_name="Selected_Single_Factors", index=False)
        composite_stats.to_excel(writer, sheet_name="Composite_Stats_By_Day", index=False)
        composite_daily.to_excel(writer, sheet_name="Composite_Daily_IC", index=False)

    print(f"\nSaved composite summary CSV: {COMPOSITE_SUMMARY_CSV}")
    print(f"Saved composite daily IC CSV: {COMPOSITE_DAILY_CSV}")
    print(f"Saved composite comparison Excel: {COMPOSITE_COMPARE_EXCEL}")

    print("\nComposite factor IC summary:")
    print(composite_summary.to_string(index=False))

    print("\nComparison by horizon:")
    print(comparison_by_horizon.to_string(index=False))

    return {
        "composite_summary": composite_summary,
        "composite_daily": composite_daily,
        "comparison_by_horizon": comparison_by_horizon,
        "comparison_long": comparison_long,
        "selected_direction_df": selected_direction_df,
        "selected_single_summary": selected_single_summary,
        "composite_stats": composite_stats,
    }

def build_composite_comparison_by_horizon(
    composite_summary: pd.DataFrame,
    selected_single_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare composite factor with:
    1. Best selected single factor by abs(rank_ic_nw_icir)
    2. Average selected single factor performance
    """
    score_col = "rank_ic_nw_icir"

    comp = composite_summary.copy()
    selected = selected_single_summary.copy()

    if score_col not in selected.columns:
        selected[score_col] = np.nan

    if score_col not in comp.columns:
        comp[score_col] = np.nan

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
        comp_h = comp[comp["horizon"] == h].iloc[0].to_dict()
        selected_h = selected[selected["horizon"] == h].copy()

        row = {
            "horizon": h,
            "composite_factor": COMPOSITE_FACTOR_COL,
        }

        for m in metrics:
            row[f"composite_{m}"] = comp_h.get(m, np.nan)

        if not selected_h.empty:
            selected_h["_abs_score"] = selected_h[score_col].abs()

            if selected_h["_abs_score"].dropna().empty:
                best = selected_h.iloc[0]
            else:
                best = selected_h.sort_values("_abs_score", ascending=False).iloc[0]

            row["best_single_factor"] = best.get("factor", np.nan)
            row["best_single_factor_label"] = best.get("factor_label", np.nan)

            for m in metrics:
                row[f"best_single_{m}"] = best.get(m, np.nan)

            for m in metrics:
                if m in selected_h.columns:
                    row[f"avg_selected_single_{m}"] = selected_h[m].mean()
                else:
                    row[f"avg_selected_single_{m}"] = np.nan

            # Main differences
            row["diff_composite_minus_best_rank_ic_nw_icir"] = (
                row.get("composite_rank_ic_nw_icir", np.nan)
                - row.get("best_single_rank_ic_nw_icir", np.nan)
            )

            row["diff_composite_minus_avg_rank_ic_nw_icir"] = (
                row.get("composite_rank_ic_nw_icir", np.nan)
                - row.get("avg_selected_single_rank_ic_nw_icir", np.nan)
            )

            row["diff_composite_minus_best_rank_ic_mean"] = (
                row.get("composite_rank_ic_mean", np.nan)
                - row.get("best_single_rank_ic_mean", np.nan)
            )

            row["diff_composite_minus_avg_rank_ic_mean"] = (
                row.get("composite_rank_ic_mean", np.nan)
                - row.get("avg_selected_single_rank_ic_mean", np.nan)
            )
        else:
            row["best_single_factor"] = np.nan
            row["best_single_factor_label"] = np.nan

        records.append(row)

    out = pd.DataFrame(records)
    return out

if __name__ == "__main__":
    composite_result = run_composite_factor_ic()