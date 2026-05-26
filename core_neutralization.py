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
except ImportError:
    from core_base import *

# Industry + market cap exposure preparation
def _extract_6digit_ticker(x) -> str | None:
    if pd.isna(x):
        return None

    s = str(x).strip().upper()

    # First try to find exact 6-digit code
    import re
    m = re.search(r"(\d{6})", s)
    if m:
        return m.group(1)

    # If Excel reads 000001 as 1 or 1.0
    m = re.search(r"(\d+)", s)
    if m:
        return m.group(1).zfill(6)

    return None


def _infer_a_share_code_from_ticker(ticker: str) -> str | None:
    """
    Infer exchange suffix from 6-digit A-share ticker.
    """
    if ticker is None or pd.isna(ticker):
        return None

    ticker = str(ticker).zfill(6)

    # Shenzhen
    if ticker.startswith(("000", "001", "002", "003", "300", "301")):
        return f"{ticker}.SZ"

    # Shanghai
    if ticker.startswith(("600", "601", "603", "605", "688", "689")):
        return f"{ticker}.SH"

    # Beijing
    if ticker.startswith(("430", "831", "832", "833", "834", "835", "836",
                          "837", "838", "839", "871", "872", "873", "920")):
        return f"{ticker}.BJ"

    return None


def _standardize_code(x) -> str | None:
    """
    Standardize code to 000001.SZ / 600000.SH / etc.
    """
    if pd.isna(x):
        return None

    s = str(x).strip().upper()

    # Already has exchange suffix
    if s.endswith((".SZ", ".SH", ".BJ")):
        ticker = _extract_6digit_ticker(s)
        suffix = s[-3:]
        if ticker is not None:
            return f"{ticker}{suffix}"

    ticker = _extract_6digit_ticker(s)
    if ticker is None:
        return None

    return _infer_a_share_code_from_ticker(ticker)


def load_industry_info(industry_xlsx: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """
    Load industry 0/1 dummy matrix.

    Expected structure:
    TICKER_SYMBOL, Bank, RealEstate, Health, ...
    """
    industry_xlsx = Path(industry_xlsx)
    ind = pd.read_excel(industry_xlsx)

    if "TICKER_SYMBOL" not in ind.columns:
        raise ValueError(
            f"Cannot find TICKER_SYMBOL column in industry file. "
            f"Columns are: {list(ind.columns)}"
        )

    ind = ind.copy()
    ind["ticker"] = ind["TICKER_SYMBOL"].map(_extract_6digit_ticker)

    industry_cols = [c for c in ind.columns if c not in ["TICKER_SYMBOL", "ticker"]]

    for c in industry_cols:
        ind[c] = pd.to_numeric(ind[c], errors="coerce").fillna(0.0)

    ind = ind.dropna(subset=["ticker"])
    ind = ind.drop_duplicates("ticker", keep="last")

    # Whether this stock has at least one valid industry dummy
    ind["industry_valid"] = ind[industry_cols].sum(axis=1) > 0

    return ind[["ticker", "industry_valid"] + industry_cols], industry_cols


def load_base_market_cap(market_cap_xlsx: str | Path) -> pd.DataFrame:
    """
    Load market cap file.

    Expected structure:
    S_INFO_WINDCODE, S_VAL_MV2
    """
    market_cap_xlsx = Path(market_cap_xlsx)
    mv = pd.read_excel(market_cap_xlsx)

    if "S_INFO_WINDCODE" not in mv.columns:
        raise ValueError(
            f"Cannot find S_INFO_WINDCODE column in market cap file. "
            f"Columns are: {list(mv.columns)}"
        )

    if "S_VAL_MV2" not in mv.columns:
        raise ValueError(
            f"Cannot find S_VAL_MV2 column in market cap file. "
            f"Columns are: {list(mv.columns)}"
        )

    mv = mv.copy()
    mv["code"] = mv["S_INFO_WINDCODE"].map(_standardize_code)
    mv["base_mkt_cap"] = pd.to_numeric(mv["S_VAL_MV2"], errors="coerce")

    mv = mv.dropna(subset=["code", "base_mkt_cap"])
    mv = mv[mv["base_mkt_cap"] > 0]
    mv = mv.drop_duplicates("code", keep="last")

    return mv[["code", "base_mkt_cap"]]


def add_industry_size_exposures(
    df: pd.DataFrame,
    industry_xlsx: str | Path = INDUSTRY_XLSX,
    market_cap_xlsx: str | Path = MARKET_CAP_XLSX,
    base_date: str = MARKET_CAP_BASE_DATE,
    price_col: str = "price",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Add industry dummy variables and log market cap to the daily stock panel.

    Because the market cap file only has one date, we infer other dates by price ratio:

        market_cap[t] = market_cap[base_date] * price[t] / price[base_date]

    Then:

        log_mkt_cap[t] = log(market_cap[t])

    Exposure matrix X = industry dummy matrix + log_mkt_cap
    """
    df = df.copy()

    print("Loading industry dummy matrix ...")
    industry_df, industry_cols = load_industry_info(industry_xlsx)

    print("Loading base market cap ...")
    mv_df = load_base_market_cap(market_cap_xlsx)

    df["ticker"] = df["code"].map(_extract_6digit_ticker)

    # Merge industry dummies
    before = len(df)
    df = df.merge(industry_df, on="ticker", how="left")
    print(f"Merge industry info: {before:,} rows -> {len(df):,} rows")

    # Fill missing industry dummies with 0, but keep industry_valid flag.
    for c in industry_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["industry_valid"] = df["industry_valid"].fillna(False).astype(bool)

    # Find base-date price for each stock.
    base_date = pd.to_datetime(base_date).normalize()

    price_base = (
        df[df["trade_day"] <= base_date]
        .sort_values(["code", "trade_day"])
        .groupby("code", sort=False)
        .tail(1)[["code", price_col]]
        .rename(columns={price_col: "base_price"})
    )

    price_base["base_price"] = pd.to_numeric(price_base["base_price"], errors="coerce")
    price_base = price_base[price_base["base_price"] > 0]

    # Merge base market cap and base price
    df = df.merge(mv_df, on="code", how="left")
    df = df.merge(price_base, on="code", how="left")

    # Infer daily market cap from base market cap and price ratio
    df["mkt_cap"] = df["base_mkt_cap"] * df[price_col] / df["base_price"]
    df["mkt_cap"] = df["mkt_cap"].replace([np.inf, -np.inf], np.nan)

    df["log_mkt_cap"] = np.where(df["mkt_cap"] > 0, np.log(df["mkt_cap"]), np.nan)

    exposure_cols = industry_cols + ["log_mkt_cap"]

    valid_exposure = (
        df["industry_valid"]
        & df["log_mkt_cap"].notna()
        & np.isfinite(df["log_mkt_cap"])
    )

    print(f"Rows with valid industry + log market cap exposure: {valid_exposure.sum():,} / {len(df):,}")
    print(f"Industry dummy columns: {industry_cols}")
    print("Exposure matrix columns = industry dummies + log_mkt_cap")

    return df, exposure_cols, industry_cols


def _ols_residual_1d(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Residualize y against exposure matrix X using OLS.

    y_resid = y - X @ beta

    Here X should contain:
    - industry 0/1 dummy variables
    - log market cap

    We do not add an extra intercept because the industry dummy matrix already
    spans the intercept when each stock belongs to one industry.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)

    resid = np.full_like(y, np.nan, dtype=float)

    valid = np.isfinite(y) & np.isfinite(X).all(axis=1)

    if valid.sum() < 2:
        return resid

    yv = y[valid]
    Xv = X[valid]

    # Remove exposure columns that are all zero or completely invalid in this cross-section.
    col_keep = np.isfinite(Xv).all(axis=0) & (np.nanstd(Xv, axis=0) > 0)

    # Keep constant non-zero industry dummy if needed.
    # If all rows are in one industry, all dummies except one are zero,
    # the one non-zero dummy acts like an intercept.
    col_nonzero = np.nanmean(np.abs(Xv), axis=0) > 0
    col_keep = col_keep | col_nonzero

    Xv = Xv[:, col_keep]

    if Xv.shape[1] == 0:
        resid[valid] = yv - np.nanmean(yv)
        return resid

    try:
        beta = np.linalg.lstsq(Xv, yv, rcond=None)[0]
        resid[valid] = yv - Xv @ beta
    except np.linalg.LinAlgError:
        resid[valid] = yv - np.nanmean(yv)

    return resid
