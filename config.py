"""
Function：
1. Load 2005-2026 stock_daily_YYYY.parquet
2. Filter A share Stock
3. price = close * factor compute MA / EMA / RSI / MACD
4. solve 1/5/10/20 days return
5. 计算每日横截面 Pearson IC 和 Spearman Rank IC
6. 输出汇总表和每日 IC 明细
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
import pandas as pd


DATA_DIR = Path("Shenwan Securities Intern/PublicFactorIC")
OUTDIR = Path("/Users/zhaoshengdong/Desktop/ic_output_full_2005_2026")


YEARS = list(range(2022, 2027))
FILE_PATTERN = "stock_daily_{year}.parquet"

# price_mode:
# adjusted = close * factor
# raw      = close
PRICE_MODE = "adjusted"

# Stock Only, no index etc...
FILTER_A_SHARE = True

# 是否把成交量为 0 的行也放进 IC 计算，建议 False
INCLUDE_SUSPENDED = False

# 预测未来收益周期。
HORIZONS = [1, 5, 10, 20]

# 每天至少多少只股票才计算当天 IC。
MIN_OBS = 50

# 缺少年份文件时是否继续运行。
# False：缺文件直接报错，避免你以为自己跑的是全样本。
# True ：缺文件时打印提示并继续读取已有文件。
ALLOW_MISSING_FILES = False

# 是否保存完整因子面板。全样本会很大，通常先 False。
SAVE_FACTOR_DATA = False

# 为了省内存，计算完因子后只保留 IC 需要的列。
REDUCE_MEMORY_FOR_IC = True

# 1% / 99% winsorization
WINSOR_Q = 0.01

# Usually winsorize both factor values and future returns.
# If your teacher only wants to winsorize returns, set WINSORIZE_FACTOR = False.
WINSORIZE_FACTOR = True
WINSORIZE_RETURN = True

# Use VWAP[t+h+1] / VWAP[t+1] - 1
USE_NEXT_DAY_VWAP_RETURN = True

# How to calculate raw VWAP from amount and volume.
# "auto" tries several common unit conventions and chooses the one closest to close.
VWAP_UNIT_MODE = "auto"



# Teacher-provided files
INDUSTRY_XLSX = Path("行业信息.xlsx")
MARKET_CAP_XLSX = Path("20260515市值.xlsx")

# The market cap file is for this date
MARKET_CAP_BASE_DATE = "2026-05-15"

# Whether to compute industry + size neutralized IC
DO_INDUSTRY_SIZE_NEUTRAL_IC = True

# If you want full partial-correlation style IC, set this to True.
NEUTRALIZE_RETURN_TOO = False
