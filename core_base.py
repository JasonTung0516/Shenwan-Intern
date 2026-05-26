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

# 1. Data loading / cleaning
def find_input_files(
    data_dir: str | Path,
    years: Sequence[int] = YEARS,
    file_pattern: str = FILE_PATTERN,
    allow_missing: bool = ALLOW_MISSING_FILES,
) -> list[Path]:
    """Find stock_daily_YYYY.parquet files in data_dir."""
    data_dir = Path(data_dir)

    if data_dir.is_file():
        return [data_dir]

    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {data_dir}")

    files = [data_dir / file_pattern.format(year=y) for y in years]
    missing = [fp for fp in files if not fp.exists()]
    existing = [fp for fp in files if fp.exists()]

    if missing:
        print("Missing parquet files:")
        for fp in missing:
            print(f"  - {fp}")
        if not allow_missing:
            raise FileNotFoundError(
                "Some yearly parquet files are missing. "
                "Move all stock_daily_YYYY.parquet files into DATA_DIR, "
                "or set ALLOW_MISSING_FILES = True."
            )

    if not existing:
        raise FileNotFoundError(f"No parquet files found under: {data_dir}")

    return existing


def _get_parquet_columns(fp: Path) -> list[str] | None:
    """Read parquet schema columns without loading the full file."""
    try:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(fp).schema_arrow.names)
    except Exception:
        return None


def read_parquet_files(files: Sequence[str | Path]) -> pd.DataFrame:
    """Read parquet files. Only necessary columns are loaded when possible."""
    frames: list[pd.DataFrame] = []
    preferred_cols = ["code", "trade_day", "datetime", "close", "volume", "amount", "factor"]

    for fp in files:
        fp = Path(fp)
        print(f"Reading {fp} ...")

        available_cols = _get_parquet_columns(fp)
        if available_cols is not None:
            cols = [c for c in preferred_cols if c in available_cols]
            df_part = pd.read_parquet(fp, columns=cols)
        else:
            df_part = pd.read_parquet(fp)
            keep_cols = [c for c in preferred_cols if c in df_part.columns]
            df_part = df_part[keep_cols].copy()

        frames.append(df_part)

    df = pd.concat(frames, ignore_index=True)
    return df



def make_a_share_mask(code: pd.Series) -> pd.Series:
    """
    Basic A-share stock filter.

    Keeps common A-share stocks:
    - SZ: 000/001/002/003/300/301
    - SH: 600/601/603/605/688/689
    - BJ: 430/831-839/871-873/920
    """
    code = code.astype(str).str.upper()
    return (
        code.str.match(r"^(000|001|002|003|300|301)\d{3}\.SZ$")
        | code.str.match(r"^(600|601|603|605|688|689)\d{3}\.SH$")
        | code.str.match(r"^(430|831|832|833|834|835|836|837|838|839|871|872|873|920)\d{3}\.BJ$")
    )

def calc_raw_vwap_auto(
    amount: pd.Series,
    volume: pd.Series,
    close: pd.Series,
    unit_mode: str = "auto",
) -> pd.Series:
    """
    Calculate raw VWAP from amount and volume.

    VWAP = traded amount / traded volume

    Different data vendors use different units:
    - volume may be shares or lots
    - amount may be RMB or thousand RMB

    unit_mode="auto" tries common conventions and chooses the VWAP
    that is closest to raw close price.
    """
    amount = pd.to_numeric(amount, errors="coerce")
    volume = pd.to_numeric(volume, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")

    valid = (
        amount.notna()
        & volume.notna()
        & close.notna()
        & (amount > 0)
        & (volume > 0)
        & (close > 0)
    )

    candidates = {
        # amount in RMB, volume in shares
        "amount_div_volume": amount / volume,

        # amount in RMB, volume in lots / hands
        "amount_div_volume_100": amount / (volume * 100),

        # amount in thousand RMB, volume in shares
        "amount_1000_div_volume": amount * 1000 / volume,

        # amount in thousand RMB, volume in lots / hands
        "amount_1000_div_volume_100": amount * 1000 / (volume * 100),
    }

    if unit_mode != "auto":
        if unit_mode not in candidates:
            raise ValueError(
                f"Unknown VWAP_UNIT_MODE={unit_mode}. "
                f"Available modes: {list(candidates.keys()) + ['auto']}"
            )
        return candidates[unit_mode].replace([np.inf, -np.inf], np.nan)

    # Choose the candidate closest to close price.
    scores = {}

    for name, vwap in candidates.items():
        ratio = vwap[valid] / close[valid]
        ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()

        if len(ratio) == 0:
            scores[name] = np.inf
            continue

        # Use log distance, robust to scale.
        scores[name] = float(np.nanmedian(np.abs(np.log(ratio))))

    best_mode = min(scores, key=scores.get)
    print(f"VWAP unit mode selected: {best_mode}, score={scores[best_mode]:.6f}")

    return candidates[best_mode].replace([np.inf, -np.inf], np.nan)

def prepare_data(
    df: pd.DataFrame,
    code_col: str = "code",
    date_col: str = "trade_day",
    close_col: str = "close",
    factor_col: str = "factor",
    price_mode: str = PRICE_MODE,
    filter_a_share: bool = FILTER_A_SHARE,
) -> pd.DataFrame:
    """Clean input data and create price column for factor calculation."""
    df = df.copy()

    if date_col not in df.columns:
        if "datetime" in df.columns:
            date_col = "datetime"
        else:
            raise ValueError(f"Cannot find date column: {date_col} or datetime")

    required = [code_col, date_col, close_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df = df.rename(columns={code_col: "code", date_col: "trade_day", close_col: "close"})
    df["code"] = df["code"].astype(str).str.upper()

    if filter_a_share:
        before = len(df)
        df = df[make_a_share_mask(df["code"])].copy()
        print(f"A-share filter: {before:,} rows -> {len(df):,} rows")

    df = df.drop_duplicates(["code", "trade_day"], keep="last")
    df = df.sort_values(["code", "trade_day"]).reset_index(drop=True)

    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    if price_mode == "adjusted":
        if factor_col in df.columns:
            df[factor_col] = pd.to_numeric(df[factor_col], errors="coerce")
            df["price"] = df["close"] * df[factor_col].where(df[factor_col].notna(), 1.0)
        else:
            print("Warning: factor column not found. Falling back to raw close price.")
            df["price"] = df["close"]
    elif price_mode == "raw":
        df["price"] = df["close"]
    else:
        raise ValueError("PRICE_MODE must be 'adjusted' or 'raw'")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    if "amount" in df.columns and "volume" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

        df["vwap_raw"] = calc_raw_vwap_auto(
            amount=df["amount"],
            volume=df["volume"],
            close=df["close"],
            unit_mode=VWAP_UNIT_MODE,
        )

        if price_mode == "adjusted" and factor_col in df.columns:
            df["vwap_price"] = df["vwap_raw"] * df[factor_col].where(df[factor_col].notna(), 1.0)
        else:
            df["vwap_price"] = df["vwap_raw"]

        df["vwap_price"] = df["vwap_price"].replace([np.inf, -np.inf], np.nan)
    else:
        raise ValueError(
            "VWAP return requires both 'amount' and 'volume' columns in the parquet data."
        )

    df["price"] = df["price"].replace([np.inf, -np.inf], np.nan)
    df = df[df["price"].notna() & (df["price"] > 0)].copy()
    return df


# =============================================================================
# 2. Technical indicators: MA / EMA / RSI / MACD
# =============================================================================


def _grouped_rolling_mean(df: pd.DataFrame, value_col: str, window: int) -> pd.Series:
    """Fast grouped rolling mean aligned to the original index."""
    out = (
        df.groupby("code", sort=False)[value_col]
        .rolling(window, min_periods=window)
        .mean()
        .droplevel(0)
    )
    return out.reindex(df.index)


def _grouped_ewm_mean(
    df: pd.DataFrame,
    value_col: str,
    span: int | None = None,
    alpha: float | None = None,
    min_periods: int | None = None,
) -> pd.Series:
    """Fast grouped EWM mean aligned to the original index."""
    kwargs = {"adjust": False}
    if span is not None:
        kwargs["span"] = span
    if alpha is not None:
        kwargs["alpha"] = alpha
    if min_periods is not None:
        kwargs["min_periods"] = min_periods

    out = (
        df.groupby("code", sort=False)[value_col]
        .ewm(**kwargs)
        .mean()
        .droplevel(0)
    )
    return out.reindex(df.index)


def calc_rsi_wilder_from_panel(df: pd.DataFrame, price_col: str = "price", window: int = 14) -> pd.Series:
    """
    RSI using Wilder-style exponential smoothing, vectorized by code.

    RS = avg_gain / avg_loss
    RSI = 100 - 100 / (1 + RS)
    """
    delta = df.groupby("code", sort=False)[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    tmp = pd.DataFrame({"code": df["code"].values, "gain": gain.values, "loss": loss.values}, index=df.index)
    avg_gain = _grouped_ewm_mean(tmp, "gain", alpha=1 / window, min_periods=window)
    avg_loss = _grouped_ewm_mean(tmp, "loss", alpha=1 / window, min_periods=window)

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    # Edge cases
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)
    return rsi


def add_technical_factors(
    df: pd.DataFrame,
    price_col: str = "price",
    ma_windows: Iterable[int] = (5, 10, 20, 60),
    ema_windows: Iterable[int] = (5, 10, 20, 60),
) -> tuple[pd.DataFrame, list[str]]:
    """Add MA, EMA, RSI and MACD factors. Returns df and factor column list."""
    df = df.copy().sort_values(["code", "trade_day"]).reset_index(drop=True)
    factor_cols: list[str] = []

    print("Calculating MA factors ...")
    for n in ma_windows:
        ma_col = f"ma{n}"
        factor_col = f"close_ma{n}_gap"
        df[ma_col] = _grouped_rolling_mean(df, price_col, n)
        df[factor_col] = df[price_col] / df[ma_col] - 1
        factor_cols.append(factor_col)

    if "ma5" in df.columns and "ma20" in df.columns:
        df["ma5_ma20_gap"] = df["ma5"] / df["ma20"] - 1
        factor_cols.append("ma5_ma20_gap")
    if "ma10" in df.columns and "ma20" in df.columns:
        df["ma10_ma20_gap"] = df["ma10"] / df["ma20"] - 1
        factor_cols.append("ma10_ma20_gap")

    print("Calculating EMA factors ...")
    for n in ema_windows:
        ema_col = f"ema{n}"
        factor_col = f"close_ema{n}_gap"
        df[ema_col] = _grouped_ewm_mean(df, price_col, span=n, min_periods=n)
        df[factor_col] = df[price_col] / df[ema_col] - 1
        factor_cols.append(factor_col)

    if "ema5" in df.columns and "ema20" in df.columns:
        df["ema5_ema20_gap"] = df["ema5"] / df["ema20"] - 1
        factor_cols.append("ema5_ema20_gap")
    if "ema10" in df.columns and "ema20" in df.columns:
        df["ema10_ema20_gap"] = df["ema10"] / df["ema20"] - 1
        factor_cols.append("ema10_ema20_gap")

    print("Calculating RSI factors ...")
    df["rsi14"] = calc_rsi_wilder_from_panel(df, price_col=price_col, window=14)
    df["rsi14_centered"] = df["rsi14"] - 50
    factor_cols += ["rsi14", "rsi14_centered"]

    print("Calculating MACD factors ...")
    df["ema12"] = _grouped_ewm_mean(df, price_col, span=12, min_periods=12)
    df["ema26"] = _grouped_ewm_mean(df, price_col, span=26, min_periods=26)
    df["macd_dif"] = df["ema12"] - df["ema26"]
    df["macd_dea"] = _grouped_ewm_mean(df, "macd_dif", span=9, min_periods=9)
    df["macd_hist"] = df["macd_dif"] - df["macd_dea"]

    # Normalize MACD by price for cross-sectional comparability.
    df["macd_dif_pct"] = df["macd_dif"] / df[price_col]
    df["macd_dea_pct"] = df["macd_dea"] / df[price_col]
    df["macd_hist_pct"] = df["macd_hist"] / df[price_col]
    df["ema12_ema26_gap"] = df["ema12"] / df["ema26"] - 1
    factor_cols += ["macd_dif_pct", "macd_dea_pct", "macd_hist_pct", "ema12_ema26_gap"]

    df = df.replace([np.inf, -np.inf], np.nan)
    return df, factor_cols


# =============================================================================
# 3. Forward returns and IC / Rank IC
# =============================================================================


def add_forward_returns(
    df: pd.DataFrame,
    price_col: str = "price",
    return_price_col: str = "vwap_price",
    horizons: Iterable[int] = HORIZONS,
    use_next_day_vwap: bool = USE_NEXT_DAY_VWAP_RETURN,
) -> pd.DataFrame:
    """
    Add future h-day returns.

    Original version:
        fwd_ret_h = price[t+h] / price[t] - 1

    Improved realistic version:
        fwd_ret_h = VWAP[t+h+1] / VWAP[t+1] - 1

    Reason:
        If factors are calculated using day-t close, we cannot trade at day-t close.
        The earliest realistic trading price is next trading day's VWAP.
    """
    df = df.copy().sort_values(["code", "trade_day"]).reset_index(drop=True)

    print("Calculating forward returns ...")

    if use_next_day_vwap:
        if return_price_col not in df.columns:
            raise ValueError(
                f"{return_price_col} not found. "
                "Please calculate vwap_price before calling add_forward_returns()."
            )

        g = df.groupby("code", sort=False)[return_price_col]

        # Entry price: VWAP[t+1]
        entry = g.shift(-1)

        for h in horizons:
            # Exit price: VWAP[t+h+1]
            exit_ = g.shift(-(h + 1))
            df[f"fwd_ret_{h}d"] = exit_ / entry - 1

    else:
        # Old close-based return: price[t+h] / price[t] - 1
        g = df.groupby("code", sort=False)[price_col]

        for h in horizons:
            df[f"fwd_ret_{h}d"] = g.shift(-h) / df[price_col] - 1

    return df.replace([np.inf, -np.inf], np.nan)




def _rank_average(values: np.ndarray) -> np.ndarray:
    """Average ranks, equivalent to Spearman rank preprocessing."""
    try:
        from scipy.stats import rankdata  # type: ignore

        return rankdata(values, method="average").astype(float)
    except Exception:
        return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def _corr_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Fast Pearson correlation for one-dimensional finite arrays."""
    if x.size < 2 or y.size < 2:
        return np.nan
    x = x.astype(float, copy=False)
    y = y.astype(float, copy=False)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    if denom == 0 or not np.isfinite(denom):
        return np.nan
    return float(np.dot(x_centered, y_centered) / denom)

def newey_west_std_icir(
    ic_series: pd.Series,
    horizon: int,
) -> dict:
    """
    Newey-West corrected standard deviation and ICIR.

    For each factor and horizon, we have a daily IC series:
        IC_1, IC_2, ..., IC_T

    Raw ICIR:
        mean(IC) / std(IC)

    Newey-West ICIR:
        mean(IC) / sigma_NW

    sigma_NW uses Bartlett kernel with lag L = horizon + 10.
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

    # gamma_0
    nw_var = np.dot(xc, xc) / n

    # Bartlett kernel: weight_l = 1 - l / (L + 1)
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


def compute_newey_west_icir_table(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Newey-West ICIR for both:
    1. ordinary Pearson IC: daily['ic']
    2. Rank IC: daily['rank_ic']

    Output columns:
    - ic_nw_std
    - ic_nw_icir
    - rank_ic_nw_std
    - rank_ic_nw_icir
    - nw_lag
    - nw_n_obs
    """
    records = []

    for (factor, horizon), grp in daily.groupby(["factor", "horizon"]):
        horizon = int(horizon)

        raw_stats = newey_west_std_icir(grp["ic"], horizon)
        rank_stats = newey_west_std_icir(grp["rank_ic"], horizon)

        records.append(
            {
                "factor": factor,
                "horizon": horizon,

                # ordinary Pearson IC Newey-West correction
                "ic_nw_std": raw_stats["nw_std"],
                "ic_nw_icir": raw_stats["nw_icir"],

                # Rank IC Newey-West correction
                "rank_ic_nw_std": rank_stats["nw_std"],
                "rank_ic_nw_icir": rank_stats["nw_icir"],

                # same lag rule for both
                "nw_lag": raw_stats["nw_lag"],
                "nw_n_obs": raw_stats["nw_n_obs"],
            }
        )

    return pd.DataFrame(records)

def compute_newey_west_icir_table_all(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Newey-West adjusted ICIR for all available daily IC columns.

    Output examples:
    - ic_nw_icir
    - winsor_ic_nw_icir
    - rank_ic_nw_icir
    - neutral_ic_nw_icir
    - neutral_winsor_ic_nw_icir
    - neutral_rank_ic_nw_icir
    """
    ic_col_map = {
        "ic": "ic_nw",
        "winsor_ic": "winsor_ic_nw",
        "rank_ic": "rank_ic_nw",
        "neutral_ic": "neutral_ic_nw",
        "neutral_winsor_ic": "neutral_winsor_ic_nw",
        "neutral_rank_ic": "neutral_rank_ic_nw",
    }

    records = []

    for (factor, horizon), grp in daily.groupby(["factor", "horizon"], sort=True):
        horizon = int(horizon)

        rec = {
            "factor": factor,
            "horizon": horizon,
            "nw_lag": np.nan,
            "nw_n_obs": np.nan,
        }

        first_valid_stats = None
        rank_stats_for_main_cols = None

        for ic_col, prefix in ic_col_map.items():
            if ic_col not in grp.columns:
                continue

            if grp[ic_col].dropna().shape[0] < 2:
                rec[f"{prefix}_std"] = np.nan
                rec[f"{prefix}_icir"] = np.nan
                rec[f"{prefix}_lag"] = np.nan
                rec[f"{prefix}_n_obs"] = int(grp[ic_col].dropna().shape[0])
                continue

            stats = newey_west_std_icir(grp[ic_col], horizon)

            rec[f"{prefix}_std"] = stats["nw_std"]
            rec[f"{prefix}_icir"] = stats["nw_icir"]
            rec[f"{prefix}_lag"] = stats["nw_lag"]
            rec[f"{prefix}_n_obs"] = stats["nw_n_obs"]

            if first_valid_stats is None:
                first_valid_stats = stats

            if ic_col == "rank_ic":
                rank_stats_for_main_cols = stats

        # Keep old-style common columns for compatibility.
        main_stats = rank_stats_for_main_cols or first_valid_stats
        if main_stats is not None:
            rec["nw_lag"] = main_stats["nw_lag"]
            rec["nw_n_obs"] = main_stats["nw_n_obs"]

        records.append(rec)

    return pd.DataFrame(records)


def _winsorize_1d(
    x: np.ndarray,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> np.ndarray:
    """
    Winsorize a 1D array.

    Values below the lower quantile are replaced by the lower quantile.
    Values above the upper quantile are replaced by the upper quantile.

    Example:
    lower_q = 0.01, upper_q = 0.99
    means replacing the most extreme bottom 1% and top 1%.
    """
    x = np.asarray(x, dtype=float).copy()
    finite = np.isfinite(x)

    if finite.sum() < 2:
        return x

    lo, hi = np.nanquantile(x[finite], [lower_q, upper_q])

    if not np.isfinite(lo) or not np.isfinite(hi):
        return x

    x[finite] = np.clip(x[finite], lo, hi)
    return x



def compute_ic_summary_fast(
    df: pd.DataFrame,
    factor_cols: list[str],
    horizons: Iterable[int] = HORIZONS,
    min_obs: int = MIN_OBS,
    tradable_only: bool = True,
    winsor_q: float = WINSOR_Q,
    winsorize_factor: bool = WINSORIZE_FACTOR,
    winsorize_return: bool = WINSORIZE_RETURN,
    exposure_cols: list[str] | None = None,
    neutralize_return_too: bool = NEUTRALIZE_RETURN_TOO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute:
    1. Raw Pearson IC
    2. Winsorized Pearson IC
    3. Rank IC
    4. Industry + size neutralized IC

    Neutralized IC logic:
        factor_resid = residual from OLS:
            factor ~ industry_dummies + log_mkt_cap

        neutral_ic = corr(factor_resid, future_return)

    If neutralize_return_too=True:
        return_resid = residual from OLS:
            future_return ~ industry_dummies + log_mkt_cap

        neutral_ic = corr(factor_resid, return_resid)
    """
    work = df.copy()

    if tradable_only and "volume" in work.columns:
        before = len(work)
        work = work[work["volume"].fillna(0) > 0].copy()
        print(f"Tradable filter volume > 0: {before:,} rows -> {len(work):,} rows")

    daily_records: list[dict] = []
    horizons = list(horizons)

    do_neutral = exposure_cols is not None and len(exposure_cols) > 0

    if do_neutral:
        missing_exposures = [c for c in exposure_cols if c not in work.columns]
        if missing_exposures:
            raise ValueError(f"Missing exposure columns for neutral IC: {missing_exposures}")

        print(f"Neutral IC enabled. Exposure columns: {exposure_cols}")
        if neutralize_return_too:
            print("Neutralization mode: factor residual + return residual")
        else:
            print("Neutralization mode: factor residual only")

    for h in horizons:
        ret_col = f"fwd_ret_{h}d"

        if ret_col not in work.columns:
            print(f"Skip horizon={h}d because {ret_col} not found.")
            continue

        use_cols = ["trade_day", ret_col] + factor_cols
        if do_neutral:
            use_cols += exposure_cols
            if "industry_valid" in work.columns:
                use_cols += ["industry_valid"]

        sub = work[use_cols].dropna(subset=[ret_col]).copy()

        print(
            f"Computing raw / winsorized / rank / neutral IC "
            f"for horizon={h}d, rows={len(sub):,} ..."
        )

        for day, x in sub.groupby("trade_day", sort=True):
            y_all = x[ret_col].to_numpy(dtype=float, copy=False)

            X_all = None
            neutral_base_valid = None

            if do_neutral:
                X_all = x[exposure_cols].to_numpy(dtype=float, copy=False)
                neutral_base_valid = np.isfinite(X_all).all(axis=1)

                if "industry_valid" in x.columns:
                    neutral_base_valid = neutral_base_valid & x["industry_valid"].astype(bool).to_numpy()

            for factor in factor_cols:
                f_all = x[factor].to_numpy(dtype=float, copy=False)

                valid = np.isfinite(f_all) & np.isfinite(y_all)
                n = int(valid.sum())

                if n < min_obs:
                    continue

                f = f_all[valid]
                y = y_all[valid]

                if np.nanstd(f) == 0 or np.nanstd(y) == 0:
                    continue

                # 1. Raw Pearson IC
                ic = _corr_1d(f, y)

                # 2. Winsorized Pearson IC
                f_w = _winsorize_1d(f, winsor_q, 1 - winsor_q) if winsorize_factor else f
                y_w = _winsorize_1d(y, winsor_q, 1 - winsor_q) if winsorize_return else y

                if np.nanstd(f_w) == 0 or np.nanstd(y_w) == 0:
                    winsor_ic = np.nan
                else:
                    winsor_ic = _corr_1d(f_w, y_w)

                # 3. Rank IC
                rank_ic = _corr_1d(_rank_average(f), _rank_average(y))

                # 4. Industry + size neutralized IC                # ------------------------------------------------------------------
                neutral_ic = np.nan
                neutral_winsor_ic = np.nan
                neutral_rank_ic = np.nan
                neutral_n_obs = 0

                if do_neutral and X_all is not None and neutral_base_valid is not None:
                    neutral_valid_all = (
                        np.isfinite(f_all)
                        & np.isfinite(y_all)
                        & neutral_base_valid
                    )

                    neutral_n_obs = int(neutral_valid_all.sum())

                    if neutral_n_obs >= min_obs:
                        f_neu_raw = f_all[neutral_valid_all]
                        y_neu_raw = y_all[neutral_valid_all]
                        X_neu = X_all[neutral_valid_all, :]

                        if np.nanstd(f_neu_raw) > 0 and np.nanstd(y_neu_raw) > 0:
                            # Residualize factor against industry + log market cap
                            f_resid = _ols_residual_1d(f_neu_raw, X_neu)

                            # Optional: residualize return too
                            if neutralize_return_too:
                                y_target = _ols_residual_1d(y_neu_raw, X_neu)
                            else:
                                y_target = y_neu_raw

                            valid_resid = np.isfinite(f_resid) & np.isfinite(y_target)

                            if valid_resid.sum() >= min_obs:
                                f_resid = f_resid[valid_resid]
                                y_target = y_target[valid_resid]

                                if np.nanstd(f_resid) > 0 and np.nanstd(y_target) > 0:
                                    neutral_ic = _corr_1d(f_resid, y_target)

                                    # Winsorized neutral IC
                                    f_resid_w = (
                                        _winsorize_1d(f_resid, winsor_q, 1 - winsor_q)
                                        if winsorize_factor else f_resid
                                    )
                                    y_target_w = (
                                        _winsorize_1d(y_target, winsor_q, 1 - winsor_q)
                                        if winsorize_return else y_target
                                    )

                                    if np.nanstd(f_resid_w) > 0 and np.nanstd(y_target_w) > 0:
                                        neutral_winsor_ic = _corr_1d(f_resid_w, y_target_w)

                                    # Neutral Rank IC
                                    neutral_rank_ic = _corr_1d(
                                        _rank_average(f_resid),
                                        _rank_average(y_target),
                                    )

                if pd.isna(ic) or pd.isna(rank_ic):
                    continue

                daily_records.append(
                    {
                        "trade_day": day,
                        "factor": factor,
                        "horizon": h,

                        "ic": float(ic),
                        "winsor_ic": float(winsor_ic) if pd.notna(winsor_ic) else np.nan,
                        "rank_ic": float(rank_ic),
                        "n_obs": n,

                        "neutral_ic": float(neutral_ic) if pd.notna(neutral_ic) else np.nan,
                        "neutral_winsor_ic": float(neutral_winsor_ic) if pd.notna(neutral_winsor_ic) else np.nan,
                        "neutral_rank_ic": float(neutral_rank_ic) if pd.notna(neutral_rank_ic) else np.nan,
                        "neutral_n_obs": neutral_n_obs,
                    }
                )

    daily = pd.DataFrame(daily_records)

    if daily.empty:
        return pd.DataFrame(), daily

    summary = (
        daily.groupby(["factor", "horizon"], as_index=False)
        .agg(
            # Raw IC
            ic_mean=("ic", "mean"),
            ic_std=("ic", "std"),
            ic_positive_ratio=("ic", lambda s: float((s > 0).mean())),

            # Winsorized IC
            winsor_ic_mean=("winsor_ic", "mean"),
            winsor_ic_std=("winsor_ic", "std"),
            winsor_ic_positive_ratio=("winsor_ic", lambda s: float((s > 0).mean())),

            # Rank IC
            rank_ic_mean=("rank_ic", "mean"),
            rank_ic_std=("rank_ic", "std"),
            rank_ic_positive_ratio=("rank_ic", lambda s: float((s > 0).mean())),

            # Neutral IC
            neutral_ic_mean=("neutral_ic", "mean"),
            neutral_ic_std=("neutral_ic", "std"),
            neutral_ic_positive_ratio=("neutral_ic", lambda s: float((s > 0).mean())),

            neutral_winsor_ic_mean=("neutral_winsor_ic", "mean"),
            neutral_winsor_ic_std=("neutral_winsor_ic", "std"),
            neutral_winsor_ic_positive_ratio=("neutral_winsor_ic", lambda s: float((s > 0).mean())),

            neutral_rank_ic_mean=("neutral_rank_ic", "mean"),
            neutral_rank_ic_std=("neutral_rank_ic", "std"),
            neutral_rank_ic_positive_ratio=("neutral_rank_ic", lambda s: float((s > 0).mean())),

            # Diagnostics
            abs_rank_ic_mean=("rank_ic", lambda s: float(s.abs().mean())),
            neutral_abs_rank_ic_mean=("neutral_rank_ic", lambda s: float(s.abs().mean())),

            sample_days=("rank_ic", "count"),
            neutral_sample_days=("neutral_rank_ic", "count"),

            avg_obs_per_day=("n_obs", "mean"),
            avg_neutral_obs_per_day=("neutral_n_obs", "mean"),

            start_day=("trade_day", "min"),
            end_day=("trade_day", "max"),
        )
        .reset_index(drop=True)
    )

    # ICIR
    summary["icir"] = summary["ic_mean"] / summary["ic_std"]
    summary["winsor_icir"] = summary["winsor_ic_mean"] / summary["winsor_ic_std"]
    summary["rank_icir"] = summary["rank_ic_mean"] / summary["rank_ic_std"]

    summary["neutral_icir"] = summary["neutral_ic_mean"] / summary["neutral_ic_std"]
    summary["neutral_winsor_icir"] = (
        summary["neutral_winsor_ic_mean"] / summary["neutral_winsor_ic_std"]
    )
    summary["neutral_rank_icir"] = (
        summary["neutral_rank_ic_mean"] / summary["neutral_rank_ic_std"]
    )

    # Comparison: raw / winsorized / neutral vs Rank IC
    comparison = (
        daily.groupby(["factor", "horizon"])
        .apply(
            lambda g: pd.Series(
                {
                    "raw_rank_abs_gap_mean": float((g["ic"] - g["rank_ic"]).abs().mean()),
                    "winsor_rank_abs_gap_mean": float((g["winsor_ic"] - g["rank_ic"]).abs().mean()),

                    "raw_rank_sign_match_ratio": float(
                        (np.sign(g["ic"]) == np.sign(g["rank_ic"])).mean()
                    ),
                    "winsor_rank_sign_match_ratio": float(
                        (np.sign(g["winsor_ic"]) == np.sign(g["rank_ic"])).mean()
                    ),

                    "neutral_rank_abs_gap_mean": float(
                        (g["neutral_ic"] - g["neutral_rank_ic"]).abs().mean()
                    ),
                    "neutral_winsor_rank_abs_gap_mean": float(
                        (g["neutral_winsor_ic"] - g["neutral_rank_ic"]).abs().mean()
                    ),
                    "neutral_rank_sign_match_ratio": float(
                        (np.sign(g["neutral_ic"]) == np.sign(g["neutral_rank_ic"])).mean()
                    ),
                    "neutral_winsor_rank_sign_match_ratio": float(
                        (np.sign(g["neutral_winsor_ic"]) == np.sign(g["neutral_rank_ic"])).mean()
                    ),
                }
            )
        )
        .reset_index()
    )

    summary = summary.merge(comparison, on=["factor", "horizon"], how="left")

    summary["gap_reduction_mean"] = (
        summary["raw_rank_abs_gap_mean"] - summary["winsor_rank_abs_gap_mean"]
    )

    summary["neutral_gap_reduction_mean"] = (
        summary["neutral_rank_abs_gap_mean"] - summary["neutral_winsor_rank_abs_gap_mean"]
    )

    summary = summary.replace([np.inf, -np.inf], np.nan)

    summary = summary[
        [
            "factor",
            "horizon",

            # Raw IC
            "ic_mean",
            "ic_std",
            "icir",
            "ic_positive_ratio",

            # Winsorized IC
            "winsor_ic_mean",
            "winsor_ic_std",
            "winsor_icir",
            "winsor_ic_positive_ratio",

            # Rank IC
            "rank_ic_mean",
            "rank_ic_std",
            "rank_icir",
            "rank_ic_positive_ratio",

            # Neutral IC
            "neutral_ic_mean",
            "neutral_ic_std",
            "neutral_icir",
            "neutral_ic_positive_ratio",

            "neutral_winsor_ic_mean",
            "neutral_winsor_ic_std",
            "neutral_winsor_icir",
            "neutral_winsor_ic_positive_ratio",

            "neutral_rank_ic_mean",
            "neutral_rank_ic_std",
            "neutral_rank_icir",
            "neutral_rank_ic_positive_ratio",

            # Gap diagnostics
            "raw_rank_abs_gap_mean",
            "winsor_rank_abs_gap_mean",
            "gap_reduction_mean",
            "raw_rank_sign_match_ratio",
            "winsor_rank_sign_match_ratio",

            "neutral_rank_abs_gap_mean",
            "neutral_winsor_rank_abs_gap_mean",
            "neutral_gap_reduction_mean",
            "neutral_rank_sign_match_ratio",
            "neutral_winsor_rank_sign_match_ratio",

            # Other diagnostics
            "abs_rank_ic_mean",
            "neutral_abs_rank_ic_mean",
            "sample_days",
            "neutral_sample_days",
            "avg_obs_per_day",
            "avg_neutral_obs_per_day",
            "start_day",
            "end_day",
        ]
    ]

    summary = summary.sort_values(
        ["horizon", "neutral_rank_icir"],
        ascending=[True, False],
    ).reset_index(drop=True)

    daily = daily.sort_values(
        ["horizon", "trade_day", "factor"],
    ).reset_index(drop=True)

    return summary, daily
