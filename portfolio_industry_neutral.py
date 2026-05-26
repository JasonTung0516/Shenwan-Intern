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
    from .portfolio_five_group import *
except ImportError:
    from core_base import *
    from core_neutralization import *
    from portfolio_five_group import *

# Teacher requirement:
# 1. Still split stocks into Q1-Q5 by composite signal.
# 2. But portfolio weight is no longer simple equal-weight across all stocks.
# 3. Within each industry: equal-weight stocks.
# 4. Across industries: use the same target industry weights for every quintile.
# 5. This makes Q1-Q5 industry-unbiased at the portfolio level.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
# =============================================================================

IND_NEUTRAL_GROUP_TEST_OUTDIR = OUTDIR / "five_group_test_industry_neutral"
IND_NEUTRAL_GROUP_TEST_OUTDIR.mkdir(parents=True, exist_ok=True)

IND_NEUTRAL_GROUP_TEST_EXCEL = (
    IND_NEUTRAL_GROUP_TEST_OUTDIR / "five_group_portfolio_test_industry_neutral.xlsx"
)

# Industry weight method:
# "universe_equal_count":
#     industry weight = number of valid stocks in industry / total valid stocks
#
# "universe_mkt_cap":
#     industry weight = total market cap of industry / total market cap
#
# I recommend starting with universe_equal_count because your current group test
# is based on equal-weight stock portfolios.
INDUSTRY_WEIGHT_METHOD = "universe_equal_count"

# If True, only use industries that appear in all Q1-Q5 on that rebalance date.
# This guarantees the five groups use exactly the same industry set.
USE_COMMON_INDUSTRIES_ONLY = True

# If an industry has no valid return on one holding day, renormalize remaining industry weights.
RENORMALIZE_MISSING_INDUSTRY_RETURNS = True

# Exclude stocks without valid industry classification.
DROP_UNKNOWN_INDUSTRY = True

MIN_INDUSTRIES_FOR_REBALANCE = 3


# =============================================================================
# 1. Add industry label and market cap to signal panel
# =============================================================================

def add_industry_to_group_signal_panel(signal_panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Add:
    - industry dummy columns
    - mkt_cap
    - log_mkt_cap
    - industry label

    This reuses add_industry_size_exposures() already defined in your code.
    """
    panel = signal_panel.copy()

    required_cols = ["code", "trade_day", "price"]
    missing = [c for c in required_cols if c not in panel.columns]
    if missing:
        raise ValueError(f"signal_panel missing required columns for industry exposure: {missing}")

    panel, exposure_cols, industry_cols = add_industry_size_exposures(
        panel,
        industry_xlsx=INDUSTRY_XLSX,
        market_cap_xlsx=MARKET_CAP_XLSX,
        base_date=MARKET_CAP_BASE_DATE,
        price_col="price",
    )

    # Derive single industry label from 0/1 dummy columns.
    arr = panel[industry_cols].to_numpy(dtype=float, copy=False)
    max_pos = np.nanargmax(arr, axis=1)
    max_val = np.nanmax(arr, axis=1)

    labels = np.array(industry_cols, dtype=object)
    panel["industry"] = np.where(max_val > 0, labels[max_pos], "Unknown")

    panel["industry_valid_for_group"] = (
        panel["industry_valid"].astype(bool)
        & panel["industry"].notna()
        & (panel["industry"].astype(str) != "Unknown")
    )

    return panel, industry_cols


# =============================================================================
# 2. Assign Q1-Q5 with industry info
# =============================================================================

def assign_quintile_groups_for_day_industry_neutral(
    signal_day_df: pd.DataFrame,
    signal_col: str,
    n_groups: int = N_GROUPS,
    min_stocks: int = MIN_STOCKS_TO_GROUP,
) -> pd.DataFrame:
    """
    Assign Q1-Q5 based on signal ranking.

    Keep industry and mkt_cap columns for industry-neutral weighting.
    """
    x = signal_day_df.copy()

    if not INCLUDE_SUSPENDED and "volume" in x.columns:
        x = x[x["volume"].fillna(0) > 0].copy()

    if DROP_UNKNOWN_INDUSTRY:
        x = x[x["industry_valid_for_group"].astype(bool)].copy()

    x = x.dropna(subset=[signal_col, "industry"]).copy()

    if len(x) < min_stocks:
        return pd.DataFrame(columns=["code", "group", "industry", signal_col, "mkt_cap"])

    ranks = x[signal_col].rank(method="first")
    labels = [f"Q{i}" for i in range(1, n_groups + 1)]

    x["group"] = pd.qcut(
        ranks,
        q=n_groups,
        labels=labels,
    ).astype(str)

    keep_cols = ["code", "group", "industry", signal_col]
    if "mkt_cap" in x.columns:
        keep_cols.append("mkt_cap")

    return x[keep_cols].copy()


# =============================================================================
# 3. Industry target weights
# =============================================================================

def get_target_industry_weights_for_day(
    signal_day_df: pd.DataFrame,
    signal_col: str,
    method: str = INDUSTRY_WEIGHT_METHOD,
) -> pd.Series:
    """
    Compute target industry weights from the whole valid universe on the signal day.

    These same industry weights will be applied to Q1-Q5.
    """
    universe = signal_day_df.copy()

    if not INCLUDE_SUSPENDED and "volume" in universe.columns:
        universe = universe[universe["volume"].fillna(0) > 0].copy()

    if DROP_UNKNOWN_INDUSTRY:
        universe = universe[universe["industry_valid_for_group"].astype(bool)].copy()

    universe = universe.dropna(subset=[signal_col, "industry"]).copy()

    if universe.empty:
        return pd.Series(dtype=float)

    if method == "universe_equal_count":
        w = universe["industry"].value_counts(normalize=True).astype(float)

    elif method == "universe_mkt_cap":
        if "mkt_cap" not in universe.columns:
            raise ValueError("mkt_cap not found. Cannot use INDUSTRY_WEIGHT_METHOD='universe_mkt_cap'.")

        tmp = universe.dropna(subset=["mkt_cap"]).copy()
        tmp = tmp[tmp["mkt_cap"] > 0].copy()

        if tmp.empty:
            return pd.Series(dtype=float)

        w = tmp.groupby("industry")["mkt_cap"].sum()
        w = w / w.sum()

    else:
        raise ValueError(
            "INDUSTRY_WEIGHT_METHOD must be 'universe_equal_count' or 'universe_mkt_cap'."
        )

    w = w.replace([np.inf, -np.inf], np.nan).dropna()
    w = w[w > 0]

    if w.empty:
        return pd.Series(dtype=float)

    return w / w.sum()


# =============================================================================
# 4. Industry-neutral five-group returns
# =============================================================================

def build_five_group_returns_industry_neutral(
    signal_panel: pd.DataFrame,
    signal_col: str,
    rebalance_periods: list[int] = GROUP_REBALANCE_PERIODS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Industry-neutral five-group portfolio backtest.

    Signal timing:
        signal known at day t.

    Execution timing:
        enter at VWAP[t+1].

    Portfolio weighting:
        within industry: equal-weight stocks
        across industries: same target industry weights for Q1-Q5
    """
    panel = signal_panel.copy()
    panel["trade_day"] = pd.to_datetime(panel["trade_day"]).dt.normalize()
    panel = panel.sort_values(["code", "trade_day"]).reset_index(drop=True)

    panel["ret_vwap_1d"] = panel.groupby("code", sort=False)["vwap_price"].pct_change()
    panel["ret_vwap_1d"] = panel["ret_vwap_1d"].replace([np.inf, -np.inf], np.nan)

    trading_days = sorted(panel["trade_day"].dropna().unique())

    ret_by_day = {
        day: sub.set_index("code")["ret_vwap_1d"]
        for day, sub in panel[["code", "trade_day", "ret_vwap_1d"]].groupby("trade_day")
    }

    signal_cols = [
        "code",
        "trade_day",
        "volume",
        "industry",
        "industry_valid_for_group",
        signal_col,
    ]

    if "mkt_cap" in panel.columns:
        signal_cols.append("mkt_cap")

    signal_by_day = {
        day: sub[signal_cols].copy()
        for day, sub in panel[signal_cols].groupby("trade_day")
    }

    daily_return_records = []
    rebalance_records = []
    industry_return_records = []

    for h in rebalance_periods:
        print(f"\nRunning industry-neutral 5-group backtest, rebalance_period={h}d ...")

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

            group_df = assign_quintile_groups_for_day_industry_neutral(
                signal_day_df=signal_day_df,
                signal_col=signal_col,
                n_groups=N_GROUPS,
                min_stocks=MIN_STOCKS_TO_GROUP,
            )

            if group_df.empty:
                continue

            required_groups = [f"Q{i}" for i in range(1, N_GROUPS + 1)]
            available_groups = set(group_df["group"].unique())

            if not set(required_groups).issubset(available_groups):
                continue

            target_industry_weight = get_target_industry_weights_for_day(
                signal_day_df=signal_day_df,
                signal_col=signal_col,
                method=INDUSTRY_WEIGHT_METHOD,
            )

            if target_industry_weight.empty:
                continue

            if USE_COMMON_INDUSTRIES_ONLY:
                industry_sets = []

                for group in required_groups:
                    inds = set(group_df.loc[group_df["group"] == group, "industry"].astype(str))
                    industry_sets.append(inds)

                common_industries = set(target_industry_weight.index.astype(str))

                for inds in industry_sets:
                    common_industries = common_industries.intersection(inds)

                common_industries = sorted(common_industries)

                if len(common_industries) < MIN_INDUSTRIES_FOR_REBALANCE:
                    continue

                target_industry_weight = target_industry_weight.loc[common_industries]
                target_industry_weight = target_industry_weight / target_industry_weight.sum()

                group_df = group_df[group_df["industry"].isin(common_industries)].copy()

            else:
                common_industries = sorted(target_industry_weight.index.astype(str).tolist())

            if group_df.empty:
                continue

            # Rebalance-level diagnostics
            for group in required_groups:
                gtmp = group_df[group_df["group"] == group].copy()

                rebalance_records.append(
                    {
                        "horizon": h,
                        "signal_day": signal_day,
                        "entry_day": entry_day,
                        "exit_day": exit_day,
                        "group": group,
                        "n_stocks": len(gtmp),
                        "n_industries": gtmp["industry"].nunique(),
                        "industry_weight_method": INDUSTRY_WEIGHT_METHOD,
                        "used_common_industries_only": USE_COMMON_INDUSTRIES_ONLY,
                        "avg_signal": gtmp[signal_col].mean(),
                        "median_signal": gtmp[signal_col].median(),
                        "min_signal": gtmp[signal_col].min(),
                        "max_signal": gtmp[signal_col].max(),
                    }
                )

            # Daily holding returns
            for interval_i in range(start_i + 1, start_i + h + 1):
                return_day = trading_days[interval_i + 1]
                ret_series = ret_by_day.get(return_day)

                if ret_series is None:
                    continue

                for group in required_groups:
                    gtmp = group_df[group_df["group"] == group].copy()

                    industry_rets = []
                    industry_weights = []

                    for industry, target_w in target_industry_weight.items():
                        codes = (
                            gtmp.loc[gtmp["industry"] == industry, "code"]
                            .astype(str)
                            .tolist()
                        )

                        if len(codes) == 0:
                            ind_ret = np.nan
                            n_ret_obs = 0
                            n_stocks_industry = 0
                        else:
                            stock_ret = ret_series.reindex(codes).dropna()
                            n_ret_obs = int(stock_ret.shape[0])
                            n_stocks_industry = len(codes)
                            ind_ret = float(stock_ret.mean()) if n_ret_obs > 0 else np.nan

                        industry_return_records.append(
                            {
                                "horizon": h,
                                "trade_day": return_day,
                                "signal_day": signal_day,
                                "entry_day": entry_day,
                                "exit_day": exit_day,
                                "group": group,
                                "industry": industry,
                                "target_industry_weight": float(target_w),
                                "industry_ret_equal_weight": ind_ret,
                                "n_stocks_industry_at_rebalance": n_stocks_industry,
                                "n_ret_obs": n_ret_obs,
                            }
                        )

                        if np.isfinite(ind_ret):
                            industry_rets.append(ind_ret)
                            industry_weights.append(float(target_w))

                    if len(industry_rets) == 0:
                        group_ret = np.nan
                        weight_sum_used = 0.0
                    else:
                        industry_rets = np.asarray(industry_rets, dtype=float)
                        industry_weights = np.asarray(industry_weights, dtype=float)

                        if RENORMALIZE_MISSING_INDUSTRY_RETURNS:
                            industry_weights = industry_weights / industry_weights.sum()

                        group_ret = float(np.sum(industry_weights * industry_rets))
                        weight_sum_used = float(industry_weights.sum())

                    daily_return_records.append(
                        {
                            "horizon": h,
                            "trade_day": return_day,
                            "signal_day": signal_day,
                            "entry_day": entry_day,
                            "exit_day": exit_day,
                            "group": group,
                            "ret": group_ret,
                            "n_ret_obs": int(
                                group_df.loc[group_df["group"] == group, "code"].nunique()
                            ),
                            "n_stocks_at_rebalance": int(
                                group_df.loc[group_df["group"] == group, "code"].nunique()
                            ),
                            "n_industries_used": len(industry_rets),
                            "industry_weight_sum_used": weight_sum_used,
                        }
                    )

    daily_returns = pd.DataFrame(daily_return_records)
    rebalance_info = pd.DataFrame(rebalance_records)
    industry_daily_returns = pd.DataFrame(industry_return_records)

    if daily_returns.empty:
        raise ValueError(
            "No industry-neutral daily group returns generated. "
            "Please check signal coverage, industry coverage, and MIN_INDUSTRIES_FOR_REBALANCE."
        )

    return daily_returns, rebalance_info, industry_daily_returns, panel


# =============================================================================
# 5. Save and plot
# =============================================================================

def save_industry_neutral_five_group_outputs(
    summary_df: pd.DataFrame,
    nav_df: pd.DataFrame,
    daily_returns: pd.DataFrame,
    rebalance_info: pd.DataFrame,
    industry_daily_returns: pd.DataFrame,
    weights: pd.DataFrame,
    excel_path: Path = IND_NEUTRAL_GROUP_TEST_EXCEL,
):
    config_df = pd.DataFrame(
        [
            {"item": "signal_mode", "value": SIGNAL_MODE},
            {"item": "industry_weight_method", "value": INDUSTRY_WEIGHT_METHOD},
            {"item": "use_common_industries_only", "value": USE_COMMON_INDUSTRIES_ONLY},
            {"item": "rebalance_periods", "value": str(GROUP_REBALANCE_PERIODS)},
            {"item": "n_groups", "value": N_GROUPS},
            {"item": "min_stocks_to_group", "value": MIN_STOCKS_TO_GROUP},
            {"item": "min_industries_for_rebalance", "value": MIN_INDUSTRIES_FOR_REBALANCE},
            {"item": "execution_assumption", "value": "signal at t, enter at VWAP[t+1]"},
            {"item": "portfolio_weighting", "value": "within industry equal-weight; across industries same target industry weights for Q1-Q5"},
            {"item": "Q1", "value": "lowest signal"},
            {"item": "Q5", "value": "highest signal"},
        ]
    )

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Performance_Summary", index=False)
        nav_df.to_excel(writer, sheet_name="Cumulative_NAV", index=False)
        daily_returns.to_excel(writer, sheet_name="Daily_Group_Returns", index=False)
        rebalance_info.to_excel(writer, sheet_name="Rebalance_Info", index=False)

        if len(industry_daily_returns) <= 1_048_000:
            industry_daily_returns.to_excel(writer, sheet_name="Daily_Industry_Returns", index=False)

        weights.to_excel(writer, sheet_name="Signal_Weights", index=False)
        config_df.to_excel(writer, sheet_name="Config", index=False)

    print(f"Saved industry-neutral five-group test Excel: {excel_path}")


def plot_industry_neutral_five_group_nav_curves(nav_df: pd.DataFrame, output_dir: Path):
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

        ax.set_title(
            f"Industry-neutral five-group cumulative return, "
            f"rebalance every {h} trading day(s)"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative NAV")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        fig_path = output_dir / f"industry_neutral_five_group_nav_h{h}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved plot: {fig_path}")


# =============================================================================
# 6. Run industry-neutral five-group test
# =============================================================================

def run_industry_neutral_five_group_portfolio_test():
    # Reuse your existing signal construction.
    signal_panel, weights = build_group_test_signal_panel()

    if SIGNAL_MODE == "weighted":
        signal_col = GROUP_SIGNAL_WEIGHTED_COL
    elif SIGNAL_MODE == "equal":
        signal_col = GROUP_SIGNAL_EQUAL_COL
    else:
        raise ValueError("SIGNAL_MODE must be 'weighted' or 'equal'.")

    print(f"\nUsing signal column for industry-neutral grouping: {signal_col}")

    signal_panel_industry, industry_cols = add_industry_to_group_signal_panel(signal_panel)

    daily_returns, rebalance_info, industry_daily_returns, panel_with_signal = (
        build_five_group_returns_industry_neutral(
            signal_panel=signal_panel_industry,
            signal_col=signal_col,
            rebalance_periods=GROUP_REBALANCE_PERIODS,
        )
    )

    nav_df, performance_summary = build_group_nav_and_summary(daily_returns)

    # Save CSV
    performance_summary.to_csv(
        IND_NEUTRAL_GROUP_TEST_OUTDIR / "industry_neutral_five_group_performance_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    nav_df.to_csv(
        IND_NEUTRAL_GROUP_TEST_OUTDIR / "industry_neutral_five_group_cumulative_nav.csv",
        index=False,
        encoding="utf-8-sig",
    )

    daily_returns.to_csv(
        IND_NEUTRAL_GROUP_TEST_OUTDIR / "industry_neutral_five_group_daily_returns.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rebalance_info.to_csv(
        IND_NEUTRAL_GROUP_TEST_OUTDIR / "industry_neutral_five_group_rebalance_info.csv",
        index=False,
        encoding="utf-8-sig",
    )

    industry_daily_returns.to_csv(
        IND_NEUTRAL_GROUP_TEST_OUTDIR / "industry_neutral_daily_industry_returns.csv",
        index=False,
        encoding="utf-8-sig",
    )

    weights.to_csv(
        IND_NEUTRAL_GROUP_TEST_OUTDIR / "industry_neutral_signal_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save Excel
    save_industry_neutral_five_group_outputs(
        summary_df=performance_summary,
        nav_df=nav_df,
        daily_returns=daily_returns,
        rebalance_info=rebalance_info,
        industry_daily_returns=industry_daily_returns,
        weights=weights,
        excel_path=IND_NEUTRAL_GROUP_TEST_EXCEL,
    )

    # Plots
    plot_industry_neutral_five_group_nav_curves(
        nav_df=nav_df,
        output_dir=IND_NEUTRAL_GROUP_TEST_OUTDIR,
    )

    print("\nIndustry-neutral five-group performance summary:")
    print(performance_summary.to_string(index=False))

    return {
        "signal_panel": panel_with_signal,
        "weights": weights,
        "daily_returns": daily_returns,
        "rebalance_info": rebalance_info,
        "industry_daily_returns": industry_daily_returns,
        "nav": nav_df,
        "performance_summary": performance_summary,
    }


if __name__ == "__main__":
    industry_neutral_five_group_result = run_industry_neutral_five_group_portfolio_test()