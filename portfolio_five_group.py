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
    from .composite_weighted import *
except ImportError:
    from core_base import *
    from core_neutralization import *
    from factor_selection_ward import *
    from composite_equal import *
    from composite_weighted import *

# Requirement:
# 1. Calculate current composite factor signal for each stock.
# 2. Sort stocks by signal every rebalance date.
# 3. Split into 5 groups: Q1 lowest signal, Q5 highest signal.
# 4. Rebalance every 1 / 5 / 10 / 20 trading days.
# 5. Use equal-weight portfolio return within each group.
# 6. Plot cumulative return curves for Q1-Q5 and Q5-Q1.


# =============================================================================
# CONFIG
# =============================================================================

GROUP_TEST_OUTDIR = OUTDIR / "five_group_test"
GROUP_TEST_OUTDIR.mkdir(parents=True, exist_ok=True)

SELECTION_OUTDIR = OUTDIR / "neutral_factor_redundancy"

# Prefer Ward-selected factors.
SELECTED_FACTORS_CSV_FOR_GROUP_TEST = (
    SELECTION_OUTDIR / "selected_factors_by_ward_corr_gt_0p9_rank_ic_nw_icir.csv"
)

# Fallback to compatibility file if Ward file does not exist.
SELECTED_FACTORS_CSV_FALLBACK = (
    SELECTION_OUTDIR / "selected_factors_by_corr_gt_0p9_rank_ic_nw_icir.csv"
)

PREVIOUS_SUMMARY_CSV_FOR_GROUP_TEST = (
    OUTDIR / "technical_factor_ic_summary_winsor_industry_size_neutral_2005_2026.csv"
)

PREVIOUS_DAILY_CSV_FOR_GROUP_TEST = (
    OUTDIR / "technical_factor_daily_ic_winsor_industry_size_neutral_2005_2026.csv"
)

GROUP_TEST_EXCEL = GROUP_TEST_OUTDIR / "five_group_portfolio_test.xlsx"

# Signal construction
SIGNAL_MODE = "weighted"
# "weighted" = use abs(NW Rank ICIR)-weighted composite
# "equal"    = use equal-weight composite

GROUP_SIGNAL_EQUAL_COL = "group_signal_equal_mad_z"
GROUP_SIGNAL_WEIGHTED_COL = "group_signal_weighted_mad_z"

# Rebalance periods
GROUP_REBALANCE_PERIODS = [1, 5, 10, 20]

# Group settings
N_GROUPS = 5
MIN_STOCKS_TO_GROUP = 50

# Median / MAD z-score settings
GROUP_MAD_SCALE = 1.4826
GROUP_MAD_Z_CAP = None

# If None, require all selected factors to be valid.
# If you want more stock coverage, set e.g. MIN_VALID_FACTOR_COUNT_FOR_SIGNAL = 3
MIN_VALID_FACTOR_COUNT_FOR_SIGNAL = None

# Weight score
GROUP_WEIGHT_SCORE_COL = "rank_ic_nw_icir"

# No transaction cost in this version.
# You can add cost later if needed.
TRANSACTION_COST_BPS = 0.0


# =============================================================================
# Helper functions
# =============================================================================

def group_test_factor_label(factor: str) -> str:
    if "selection_factor_label" in globals():
        return selection_factor_label(factor)

    if "factor_label" in globals():
        try:
            return factor_label(factor)
        except Exception:
            pass

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


def group_test_newey_west_std_icir(ic_series: pd.Series, horizon: int) -> dict:
    """
    Local fallback Newey-West calculation.
    Used only if previous summary does not already contain rank_ic_nw_icir.
    """
    x = pd.Series(ic_series).dropna().astype(float).to_numpy()
    n = len(x)

    if n < 2:
        return {
            "nw_std": np.nan,
            "nw_icir": np.nan,
            "nw_lag": horizon + 10,
            "nw_n_obs": n,
        }

    mu = np.mean(x)
    xc = x - mu

    L = int(horizon) + 10
    L = min(L, n - 1)

    nw_var = np.dot(xc, xc) / n

    for lag in range(1, L + 1):
        gamma_l = np.dot(xc[lag:], xc[:-lag]) / n
        weight = 1 - lag / (L + 1)
        nw_var += 2 * weight * gamma_l

    if not np.isfinite(nw_var) or nw_var <= 0:
        nw_std = np.nan
        nw_icir = np.nan
    else:
        nw_std = np.sqrt(nw_var)
        nw_icir = mu / nw_std if nw_std != 0 else np.nan

    return {
        "nw_std": nw_std,
        "nw_icir": nw_icir,
        "nw_lag": L,
        "nw_n_obs": n,
    }


def group_test_compute_newey_west_icir_table(daily_df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for (factor, horizon), grp in daily_df.groupby(["factor", "horizon"]):
        horizon = int(horizon)

        raw_stats = group_test_newey_west_std_icir(grp["ic"], horizon)
        rank_stats = group_test_newey_west_std_icir(grp["rank_ic"], horizon)

        records.append(
            {
                "factor": factor,
                "horizon": horizon,
                "ic_nw_std": raw_stats["nw_std"],
                "ic_nw_icir": raw_stats["nw_icir"],
                "rank_ic_nw_std": rank_stats["nw_std"],
                "rank_ic_nw_icir": rank_stats["nw_icir"],
                "nw_lag": raw_stats["nw_lag"],
                "nw_n_obs": raw_stats["nw_n_obs"],
            }
        )

    return pd.DataFrame(records)


def load_selected_factors_for_group_test() -> pd.DataFrame:
    """
    Load selected factors.

    Priority:
    1. selection_result in memory
    2. Ward-selected CSV
    3. Compatibility selected CSV
    """
    if (
        "selection_result" in globals()
        and isinstance(selection_result, dict)
        and "selected_factors" in selection_result
    ):
        selected_df = selection_result["selected_factors"].copy()
    else:
        if SELECTED_FACTORS_CSV_FOR_GROUP_TEST.exists():
            selected_df = pd.read_csv(SELECTED_FACTORS_CSV_FOR_GROUP_TEST)
        elif SELECTED_FACTORS_CSV_FALLBACK.exists():
            selected_df = pd.read_csv(SELECTED_FACTORS_CSV_FALLBACK)
        else:
            raise FileNotFoundError(
                "Cannot find selected factor file. Please run Ward selection first."
            )

    if "selected_factor" in selected_df.columns:
        factor_col = "selected_factor"
    elif "factor" in selected_df.columns:
        factor_col = "factor"
    else:
        raise ValueError(
            "Selected factor file must contain either 'selected_factor' or 'factor'."
        )

    factors = (
        selected_df[factor_col]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    out = pd.DataFrame({"factor": factors})
    out["factor_label"] = out["factor"].map(group_test_factor_label)

    return out


def load_previous_summary_for_group_test() -> pd.DataFrame:
    """
    Load previous factor IC summary and ensure rank_ic_nw_icir exists.
    """
    if "summary" in globals() and isinstance(summary, pd.DataFrame):
        prev_summary = summary.copy()
    else:
        if not PREVIOUS_SUMMARY_CSV_FOR_GROUP_TEST.exists():
            raise FileNotFoundError(
                f"Cannot find summary CSV: {PREVIOUS_SUMMARY_CSV_FOR_GROUP_TEST}"
            )
        prev_summary = pd.read_csv(PREVIOUS_SUMMARY_CSV_FOR_GROUP_TEST)

    if GROUP_WEIGHT_SCORE_COL not in prev_summary.columns:
        print(f"{GROUP_WEIGHT_SCORE_COL} not found in summary. Computing from daily IC ...")

        if "daily" in globals() and isinstance(daily, pd.DataFrame):
            prev_daily = daily.copy()
        else:
            if not PREVIOUS_DAILY_CSV_FOR_GROUP_TEST.exists():
                raise FileNotFoundError(
                    f"Cannot find daily IC CSV: {PREVIOUS_DAILY_CSV_FOR_GROUP_TEST}"
                )
            prev_daily = pd.read_csv(PREVIOUS_DAILY_CSV_FOR_GROUP_TEST)

        if "compute_newey_west_icir_table" in globals():
            nw_df = compute_newey_west_icir_table(prev_daily)
        else:
            nw_df = group_test_compute_newey_west_icir_table(prev_daily)

        prev_summary = prev_summary.merge(
            nw_df,
            on=["factor", "horizon"],
            how="left",
        )

    return prev_summary


def build_group_test_weight_table(
    selected_factors_df: pd.DataFrame,
    prev_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each selected factor:
    - choose horizon with highest abs(rank_ic_nw_icir)
    - direction = sign(rank_ic_nw_icir)
    - weighted composite weight = abs(rank_ic_nw_icir) / sum(abs)
    """
    selected = selected_factors_df["factor"].astype(str).tolist()

    work = prev_summary[prev_summary["factor"].astype(str).isin(selected)].copy()

    if GROUP_WEIGHT_SCORE_COL not in work.columns:
        raise ValueError(f"Previous summary must contain {GROUP_WEIGHT_SCORE_COL}.")

    work[GROUP_WEIGHT_SCORE_COL] = pd.to_numeric(
        work[GROUP_WEIGHT_SCORE_COL],
        errors="coerce",
    )
    work["abs_score"] = work[GROUP_WEIGHT_SCORE_COL].abs()

    records = []

    for factor in selected:
        tmp = work[work["factor"].astype(str) == factor].copy()

        if tmp.empty or tmp["abs_score"].dropna().empty:
            best_horizon = np.nan
            score = np.nan
            abs_score = np.nan
            direction = 1.0
        else:
            best = tmp.sort_values("abs_score", ascending=False).iloc[0]
            best_horizon = best["horizon"]
            score = best[GROUP_WEIGHT_SCORE_COL]
            abs_score = abs(score)
            direction = np.sign(score)

            if not np.isfinite(direction) or direction == 0:
                direction = 1.0

        records.append(
            {
                "factor": factor,
                "factor_label": group_test_factor_label(factor),
                "best_horizon_for_weight": best_horizon,
                "rank_ic_nw_icir_for_weight": score,
                "abs_rank_ic_nw_icir_for_weight": abs_score,
                "direction": direction,
            }
        )

    weight_df = pd.DataFrame(records)

    weight_df["raw_weight"] = pd.to_numeric(
        weight_df["abs_rank_ic_nw_icir_for_weight"],
        errors="coerce",
    ).fillna(0.0)

    weight_sum = weight_df["raw_weight"].sum()

    if weight_sum <= 0:
        print("Warning: all weights are zero or missing. Falling back to equal weights.")
        weight_df["normalized_weight"] = 1.0 / len(weight_df)
    else:
        weight_df["normalized_weight"] = weight_df["raw_weight"] / weight_sum

    weight_df["equal_signed_weight"] = weight_df["direction"] / len(weight_df)
    weight_df["weighted_signed_weight"] = (
        weight_df["direction"] * weight_df["normalized_weight"]
    )

    return weight_df


def group_test_mad_zscore_by_day(
    df: pd.DataFrame,
    factor_col: str,
    valid_universe: pd.Series,
) -> pd.Series:
    """
    Cross-sectional robust z-score:

        z = (x - median) / (1.4826 * MAD)

    by trade_day.
    """
    x = pd.to_numeric(df[factor_col], errors="coerce").where(valid_universe)

    med = x.groupby(df["trade_day"]).transform("median")
    mad = (x - med).abs().groupby(df["trade_day"]).transform("median")

    denom = GROUP_MAD_SCALE * mad
    z = (x - med) / denom

    z = z.where(np.isfinite(z) & np.isfinite(denom) & (denom > 0))

    if GROUP_MAD_Z_CAP is not None:
        z = z.clip(lower=-GROUP_MAD_Z_CAP, upper=GROUP_MAD_Z_CAP)

    return z


# =============================================================================
# 1. Build stock-level signal panel
# =============================================================================

def build_group_test_signal_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build stock-level daily signal panel.

    Output columns include:
    - code
    - trade_day
    - vwap_price
    - volume
    - group_signal_equal_mad_z
    - group_signal_weighted_mad_z
    """
    selected_factors_df = load_selected_factors_for_group_test()
    prev_summary = load_previous_summary_for_group_test()

    weights = build_group_test_weight_table(
        selected_factors_df=selected_factors_df,
        prev_summary=prev_summary,
    )

    selected_factors = weights["factor"].astype(str).tolist()

    print("\nSelected factors and weights for group test:")
    print(weights.to_string(index=False))

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

    missing = [f for f in selected_factors if f not in all_factor_cols]
    if missing:
        raise ValueError(f"Selected factors are not generated in factor panel: {missing}")

    if not INCLUDE_SUSPENDED and "volume" in df.columns:
        valid_universe = df["volume"].fillna(0) > 0
    else:
        valid_universe = pd.Series(True, index=df.index)

    equal_component_cols = []
    weighted_component_cols = []
    z_cols = []

    equal_weight_map = weights.set_index("factor")["equal_signed_weight"].to_dict()
    weighted_weight_map = weights.set_index("factor")["weighted_signed_weight"].to_dict()

    for factor in selected_factors:
        z_col = f"group_z_mad__{factor}"
        equal_component_col = f"group_equal_component__{factor}"
        weighted_component_col = f"group_weighted_component__{factor}"

        df[z_col] = group_test_mad_zscore_by_day(
            df=df,
            factor_col=factor,
            valid_universe=valid_universe,
        )

        df[equal_component_col] = df[z_col] * equal_weight_map.get(factor, 0.0)
        df[weighted_component_col] = df[z_col] * weighted_weight_map.get(factor, 0.0)

        z_cols.append(z_col)
        equal_component_cols.append(equal_component_col)
        weighted_component_cols.append(weighted_component_col)

    if MIN_VALID_FACTOR_COUNT_FOR_SIGNAL is None:
        min_valid_count = len(z_cols)
    else:
        min_valid_count = int(MIN_VALID_FACTOR_COUNT_FOR_SIGNAL)

    valid_factor_count = df[z_cols].notna().sum(axis=1)

    df[GROUP_SIGNAL_EQUAL_COL] = df[equal_component_cols].sum(axis=1)
    df[GROUP_SIGNAL_WEIGHTED_COL] = df[weighted_component_cols].sum(axis=1)

    df.loc[valid_factor_count < min_valid_count, GROUP_SIGNAL_EQUAL_COL] = np.nan
    df.loc[valid_factor_count < min_valid_count, GROUP_SIGNAL_WEIGHTED_COL] = np.nan

    keep_cols = [
        "code",
        "trade_day",
        "price",
        "vwap_price",
        "volume",
        GROUP_SIGNAL_EQUAL_COL,
        GROUP_SIGNAL_WEIGHTED_COL,
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]

    signal_panel = df[keep_cols].copy()
    signal_panel = signal_panel.sort_values(["code", "trade_day"]).reset_index(drop=True)

    return signal_panel, weights


# =============================================================================
# 2. Quintile grouping
# =============================================================================

def assign_quintile_groups_for_day(
    signal_day_df: pd.DataFrame,
    signal_col: str,
    n_groups: int = N_GROUPS,
    min_stocks: int = MIN_STOCKS_TO_GROUP,
) -> pd.DataFrame:
    """
    Assign Q1-Q5 based on signal ranking.

    Q1 = lowest signal
    Q5 = highest signal
    """
    x = signal_day_df.copy()

    if not INCLUDE_SUSPENDED and "volume" in x.columns:
        x = x[x["volume"].fillna(0) > 0].copy()

    x = x.dropna(subset=[signal_col]).copy()

    if len(x) < min_stocks:
        return pd.DataFrame(columns=["code", "group", signal_col])

    # rank(method="first") avoids qcut failure from duplicated signal values.
    ranks = x[signal_col].rank(method="first")

    labels = [f"Q{i}" for i in range(1, n_groups + 1)]

    x["group"] = pd.qcut(
        ranks,
        q=n_groups,
        labels=labels,
    ).astype(str)

    return x[["code", "group", signal_col]].copy()


# =============================================================================
# 3. Backtest logic
# =============================================================================

def build_five_group_returns(
    signal_panel: pd.DataFrame,
    signal_col: str,
    rebalance_periods: list[int] = GROUP_REBALANCE_PERIODS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Five-group equal-weight portfolio backtest.

    Signal timing:
        signal is known on day t.

    Trading timing:
        enter on VWAP[t+1],
        then hold until next rebalance.

    Daily stock return:
        vwap_price[t] / vwap_price[t-1] - 1

    For rebalance period h:
        use t signal,
        enter at t+1,
        hold daily returns from t+1 -> t+2, ..., t+h -> t+h+1.
    """
    panel = signal_panel.copy()
    panel["trade_day"] = pd.to_datetime(panel["trade_day"]).dt.normalize()

    panel = panel.sort_values(["code", "trade_day"]).reset_index(drop=True)

    panel["ret_vwap_1d"] = (
        panel.groupby("code", sort=False)["vwap_price"].pct_change()
    )

    panel["ret_vwap_1d"] = panel["ret_vwap_1d"].replace([np.inf, -np.inf], np.nan)

    trading_days = sorted(panel["trade_day"].dropna().unique())

    # For fast daily lookup
    ret_by_day = {
        day: sub.set_index("code")["ret_vwap_1d"]
        for day, sub in panel[["code", "trade_day", "ret_vwap_1d"]].groupby("trade_day")
    }

    signal_by_day = {
        day: sub[["code", "trade_day", "volume", signal_col]].copy()
        for day, sub in panel[["code", "trade_day", "volume", signal_col]].groupby("trade_day")
    }

    daily_return_records = []
    rebalance_records = []

    for h in rebalance_periods:
        print(f"\nRunning 5-group backtest, rebalance_period={h}d ...")

        # Need start_i + h + 1 to exist because exit day is t+h+1
        valid_start_end = len(trading_days) - h

        for start_i in range(0, valid_start_end, h):
            signal_day = trading_days[start_i]

            if start_i + h + 1 >= len(trading_days):
                break

            entry_day = trading_days[start_i + 1]
            exit_day = trading_days[start_i + h + 1]

            signal_day_df = signal_by_day.get(signal_day)

            if signal_day_df is None or signal_day_df.empty:
                continue

            group_df = assign_quintile_groups_for_day(
                signal_day_df=signal_day_df,
                signal_col=signal_col,
                n_groups=N_GROUPS,
                min_stocks=MIN_STOCKS_TO_GROUP,
            )

            if group_df.empty:
                continue

            group_members = {
                group: group_df.loc[group_df["group"] == group, "code"].astype(str).tolist()
                for group in sorted(group_df["group"].unique())
            }

            for group, codes in group_members.items():
                temp = group_df[group_df["group"] == group]
                rebalance_records.append(
                    {
                        "horizon": h,
                        "signal_day": signal_day,
                        "entry_day": entry_day,
                        "exit_day": exit_day,
                        "group": group,
                        "n_stocks": len(codes),
                        "avg_signal": temp[signal_col].mean(),
                        "median_signal": temp[signal_col].median(),
                        "min_signal": temp[signal_col].min(),
                        "max_signal": temp[signal_col].max(),
                    }
                )

            # Daily holding returns.
            # interval_i means holding from trading_days[interval_i] to trading_days[interval_i + 1].
            for interval_i in range(start_i + 1, start_i + h + 1):
                return_day = trading_days[interval_i + 1]
                ret_series = ret_by_day.get(return_day)

                if ret_series is None:
                    continue

                for group, codes in group_members.items():
                    stock_ret = ret_series.reindex(codes).dropna()

                    if stock_ret.empty:
                        group_ret = np.nan
                        n_ret_obs = 0
                    else:
                        group_ret = float(stock_ret.mean())
                        n_ret_obs = int(stock_ret.shape[0])

                    daily_return_records.append(
                        {
                            "horizon": h,
                            "trade_day": return_day,
                            "signal_day": signal_day,
                            "entry_day": entry_day,
                            "exit_day": exit_day,
                            "group": group,
                            "ret": group_ret,
                            "n_ret_obs": n_ret_obs,
                            "n_stocks_at_rebalance": len(codes),
                        }
                    )

    daily_returns = pd.DataFrame(daily_return_records)
    rebalance_info = pd.DataFrame(rebalance_records)

    if daily_returns.empty:
        raise ValueError("No daily group returns generated. Please check signal coverage.")

    return daily_returns, rebalance_info, panel


# =============================================================================
# 4. Cumulative return and summary
# =============================================================================

def max_drawdown(nav: pd.Series) -> float:
    nav = pd.Series(nav).dropna()

    if nav.empty:
        return np.nan

    running_max = nav.cummax()
    drawdown = nav / running_max - 1

    return float(drawdown.min())


def build_group_nav_and_summary(
    daily_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build cumulative NAV and performance summary.
    """
    group_order = [f"Q{i}" for i in range(1, N_GROUPS + 1)]

    nav_frames = []
    summary_records = []

    for h, sub in daily_returns.groupby("horizon"):
        wide = (
            sub.pivot_table(
                index="trade_day",
                columns="group",
                values="ret",
                aggfunc="mean",
            )
            .sort_index()
        )

        # Make sure Q1-Q5 columns exist
        for group in group_order:
            if group not in wide.columns:
                wide[group] = np.nan

        wide = wide[group_order].copy()

        # Long high signal, short low signal
        wide["Q5_minus_Q1"] = wide["Q5"] - wide["Q1"]

        ret_cols = group_order + ["Q5_minus_Q1"]

        nav = (1.0 + wide[ret_cols].fillna(0.0)).cumprod()
        nav.insert(0, "trade_day", nav.index)
        nav.insert(0, "horizon", h)

        nav_frames.append(nav.reset_index(drop=True))

        for col in ret_cols:
            r = wide[col].dropna()

            if r.empty:
                summary_records.append(
                    {
                        "horizon": h,
                        "portfolio": col,
                        "n_days": 0,
                        "total_return": np.nan,
                        "annual_return": np.nan,
                        "annual_vol": np.nan,
                        "sharpe_no_rf": np.nan,
                        "max_drawdown": np.nan,
                        "avg_daily_return": np.nan,
                        "daily_win_rate": np.nan,
                    }
                )
                continue

            nav_col = (1.0 + r).cumprod()
            n_days = len(r)

            total_return = float(nav_col.iloc[-1] - 1.0)
            annual_return = float(nav_col.iloc[-1] ** (252.0 / n_days) - 1.0)
            annual_vol = float(r.std(ddof=1) * np.sqrt(252.0))
            sharpe = annual_return / annual_vol if annual_vol and annual_vol > 0 else np.nan

            summary_records.append(
                {
                    "horizon": h,
                    "portfolio": col,
                    "n_days": n_days,
                    "total_return": total_return,
                    "annual_return": annual_return,
                    "annual_vol": annual_vol,
                    "sharpe_no_rf": sharpe,
                    "max_drawdown": max_drawdown(nav_col),
                    "avg_daily_return": float(r.mean()),
                    "daily_win_rate": float((r > 0).mean()),
                }
            )

    nav_df = pd.concat(nav_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_records)

    return nav_df, summary_df


# =============================================================================
# 5. Plot
# =============================================================================

def plot_five_group_nav_curves(nav_df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    group_cols = [f"Q{i}" for i in range(1, N_GROUPS + 1)]
    plot_cols = group_cols + ["Q5_minus_Q1"]

    for h, sub in nav_df.groupby("horizon"):
        sub = sub.sort_values("trade_day").copy()
        sub["trade_day"] = pd.to_datetime(sub["trade_day"])

        fig, ax = plt.subplots(figsize=(12, 7))

        for col in plot_cols:
            if col in sub.columns:
                ax.plot(sub["trade_day"], sub[col], label=col)

        ax.set_title(f"Five-group portfolio cumulative return, rebalance every {h} trading day(s)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative NAV")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        fig_path = output_dir / f"five_group_nav_h{h}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved plot: {fig_path}")


# =============================================================================
# 6. Save Excel
# =============================================================================

def save_five_group_outputs(
    summary_df: pd.DataFrame,
    nav_df: pd.DataFrame,
    daily_returns: pd.DataFrame,
    rebalance_info: pd.DataFrame,
    weights: pd.DataFrame,
    excel_path: Path = GROUP_TEST_EXCEL,
):
    config_df = pd.DataFrame(
        [
            {"item": "signal_mode", "value": SIGNAL_MODE},
            {"item": "signal_equal_col", "value": GROUP_SIGNAL_EQUAL_COL},
            {"item": "signal_weighted_col", "value": GROUP_SIGNAL_WEIGHTED_COL},
            {"item": "rebalance_periods", "value": str(GROUP_REBALANCE_PERIODS)},
            {"item": "n_groups", "value": N_GROUPS},
            {"item": "min_stocks_to_group", "value": MIN_STOCKS_TO_GROUP},
            {"item": "transaction_cost_bps", "value": TRANSACTION_COST_BPS},
            {"item": "execution_assumption", "value": "signal at t, enter at VWAP[t+1]"},
            {"item": "portfolio_weighting", "value": "equal-weight within each quintile"},
            {"item": "Q1", "value": "lowest signal"},
            {"item": "Q5", "value": "highest signal"},
        ]
    )

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Performance_Summary", index=False)
        nav_df.to_excel(writer, sheet_name="Cumulative_NAV", index=False)
        daily_returns.to_excel(writer, sheet_name="Daily_Group_Returns", index=False)
        rebalance_info.to_excel(writer, sheet_name="Rebalance_Info", index=False)
        weights.to_excel(writer, sheet_name="Signal_Weights", index=False)
        config_df.to_excel(writer, sheet_name="Config", index=False)

    print(f"Saved five-group test Excel: {excel_path}")


# =============================================================================
# 7. Run full five-group test
# =============================================================================

def run_five_group_portfolio_test():
    signal_panel, weights = build_group_test_signal_panel()

    if SIGNAL_MODE == "weighted":
        signal_col = GROUP_SIGNAL_WEIGHTED_COL
    elif SIGNAL_MODE == "equal":
        signal_col = GROUP_SIGNAL_EQUAL_COL
    else:
        raise ValueError("SIGNAL_MODE must be 'weighted' or 'equal'.")

    print(f"\nUsing signal column for grouping: {signal_col}")

    daily_returns, rebalance_info, panel_with_signal = build_five_group_returns(
        signal_panel=signal_panel,
        signal_col=signal_col,
        rebalance_periods=GROUP_REBALANCE_PERIODS,
    )

    nav_df, performance_summary = build_group_nav_and_summary(daily_returns)

    # Save CSV
    performance_summary.to_csv(
        GROUP_TEST_OUTDIR / "five_group_performance_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    nav_df.to_csv(
        GROUP_TEST_OUTDIR / "five_group_cumulative_nav.csv",
        index=False,
        encoding="utf-8-sig",
    )

    daily_returns.to_csv(
        GROUP_TEST_OUTDIR / "five_group_daily_returns.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rebalance_info.to_csv(
        GROUP_TEST_OUTDIR / "five_group_rebalance_info.csv",
        index=False,
        encoding="utf-8-sig",
    )

    weights.to_csv(
        GROUP_TEST_OUTDIR / "five_group_signal_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save Excel
    save_five_group_outputs(
        summary_df=performance_summary,
        nav_df=nav_df,
        daily_returns=daily_returns,
        rebalance_info=rebalance_info,
        weights=weights,
        excel_path=GROUP_TEST_EXCEL,
    )

    # Plots
    plot_five_group_nav_curves(
        nav_df=nav_df,
        output_dir=GROUP_TEST_OUTDIR,
    )

    print("\nFive-group performance summary:")
    print(performance_summary.to_string(index=False))

    return {
        "signal_panel": panel_with_signal,
        "weights": weights,
        "daily_returns": daily_returns,
        "rebalance_info": rebalance_info,
        "nav": nav_df,
        "performance_summary": performance_summary,
    }


if __name__ == "__main__":
    five_group_result = run_five_group_portfolio_test()