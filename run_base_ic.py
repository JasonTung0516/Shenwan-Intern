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

# 4. Main runner
def run_analysis(
    data_dir: str | Path = DATA_DIR,
    outdir: str | Path = OUTDIR,
    years: Sequence[int] = YEARS,
    file_pattern: str = FILE_PATTERN,
    price_mode: str = PRICE_MODE,
    filter_a_share: bool = FILTER_A_SHARE,
    include_suspended: bool = INCLUDE_SUSPENDED,
    horizons: Sequence[int] = HORIZONS,
    min_obs: int = MIN_OBS,
    allow_missing_files: bool = ALLOW_MISSING_FILES,
    save_factor_data: bool = SAVE_FACTOR_DATA,
    reduce_memory_for_ic: bool = REDUCE_MEMORY_FOR_IC,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full factor calculation and IC test."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = find_input_files(data_dir, years=years, file_pattern=file_pattern, allow_missing=allow_missing_files)
    print(f"Found {len(files)} parquet files.")
    print(f"First file: {files[0]}")
    print(f"Last file : {files[-1]}")

    df_raw = read_parquet_files(files)
    print(f"Raw rows: {len(df_raw):,}")

    df = prepare_data(
        df_raw,
        price_mode=price_mode,
        filter_a_share=filter_a_share,
    )
    del df_raw

    print(f"Prepared rows: {len(df):,}")
    print(f"Codes: {df['code'].nunique():,}; days: {df['trade_day'].nunique():,}")
    print(f"Date range: {df['trade_day'].min().date()} to {df['trade_day'].max().date()}")

    df, factor_cols = add_technical_factors(df, price_col="price")
    print(f"Factor columns: {factor_cols}")

# Add industry dummy variables and log market cap exposure
    exposure_cols = []

    if DO_INDUSTRY_SIZE_NEUTRAL_IC:
        df, exposure_cols, industry_cols = add_industry_size_exposures(
            df,
            industry_xlsx=INDUSTRY_XLSX,
            market_cap_xlsx=MARKET_CAP_XLSX,
            base_date=MARKET_CAP_BASE_DATE,
            price_col="price",)
    else:
        industry_cols = []

    if reduce_memory_for_ic:
        keep_cols = ["code", "trade_day", "price"] + factor_cols

    # Keep vwap_price because add_forward_returns() uses it
    # when USE_NEXT_DAY_VWAP_RETURN = True.
        if USE_NEXT_DAY_VWAP_RETURN and "vwap_price" in df.columns:
            keep_cols.append("vwap_price")

        if "volume" in df.columns:
            keep_cols.append("volume")

        if DO_INDUSTRY_SIZE_NEUTRAL_IC:
            keep_cols += exposure_cols
            if "industry_valid" in df.columns:
                keep_cols.append("industry_valid")

    # Remove duplicate columns while preserving order
        keep_cols = list(dict.fromkeys(keep_cols))

        df = df[keep_cols].copy()
    

    df = add_forward_returns(
        df,
    price_col="price",
    return_price_col="vwap_price",
    horizons=horizons,
    use_next_day_vwap=USE_NEXT_DAY_VWAP_RETURN,)

    summary, daily = compute_ic_summary_fast(
    df,
    factor_cols=factor_cols,
    horizons=horizons,
    min_obs=min_obs,
    tradable_only=not include_suspended,
    exposure_cols=exposure_cols if DO_INDUSTRY_SIZE_NEUTRAL_IC else None,
    neutralize_return_too=NEUTRALIZE_RETURN_TOO,
)
    # =============================================================================
# Newey-West ICIR correction
# ordinary IC and Rank IC
# =============================================================================

    nw_icir_df = compute_newey_west_icir_table(daily)

    summary = summary.merge(
        nw_icir_df,
        on=["factor", "horizon"],
        how="left",
    )

# Compare raw ICIR vs Newey-West ICIR
    summary["icir_nw_diff"] = summary["ic_nw_icir"] - summary["icir"]
    summary["rank_icir_nw_diff"] = summary["rank_ic_nw_icir"] - summary["rank_icir"]

    summary["icir_nw_change_pct"] = np.where(
    summary["icir"].abs() > 1e-12,
    summary["ic_nw_icir"] / summary["icir"] - 1,
    np.nan,
)

    summary["rank_icir_nw_change_pct"] = np.where(
        summary["rank_icir"].abs() > 1e-12,
        summary["rank_ic_nw_icir"] / summary["rank_icir"] - 1,
        np.nan,
)   


    summary_path = outdir / "technical_factor_ic_summary_winsor_industry_size_neutral_2005_2026.csv"
    daily_path = outdir / "technical_factor_daily_ic_winsor_industry_size_neutral_2005_2026.csv"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    
    if save_factor_data:
        factor_path = outdir / "technical_factor_panel_2005_2026.parquet"
        df.to_parquet(factor_path, index=False)
        print(f"Saved full factor panel: {factor_path}")

    print(f"Saved summary: {summary_path}")
    print(f"Saved daily IC: {daily_path}")

    if not summary.empty:
        print("\nTop rows sorted by horizon and RankICIR:")
        print(summary.head(40).to_string(index=False))
    else:
        print("Summary is empty. Please check input data, code filter, and MIN_OBS.")

    return summary, daily


if __name__ == "__main__":
    summary, daily = run_analysis()
