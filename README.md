# Technical IC Analysis Refactored

这个文件夹是把原来的超长 `technical_ic_analysis_all_years_python.py` 按任务拆出来的整理版。  
我尽量保留原来的计算逻辑，只做结构整理，不主动改变你的因子、收益率、IC、回测、交易成本、涨跌停失败率等计算方式。

## 推荐运行方式

在 `technical_ic_refactored` 的上一级目录运行：

```bash
python -m technical_ic_refactored.run_base_ic
python -m technical_ic_refactored.factor_redundancy
python -m technical_ic_refactored.factor_selection_ward
python -m technical_ic_refactored.composite_weighted
python -m technical_ic_refactored.portfolio_five_group
python -m technical_ic_refactored.portfolio_industry_neutral
python -m technical_ic_refactored.decile_limit_failure
python -m technical_ic_refactored.ic_decay
python -m technical_ic_refactored.turnover
```

不要一开始就全跑，因为很多步骤会重新读全量 parquet，耗时很长。

## 文件结构

| 文件 | 作用 |
|---|---|
| `config.py` | 全局路径、年份、参数、winsor、VWAP收益率设置、行业/市值文件路径 |
| `core_base.py` | 数据读取、清洗、VWAP、技术因子、未来收益、IC / Newey-West ICIR |
| `core_neutralization.py` | 行业信息、市值暴露、行业+规模中性化残差 |
| `run_base_ic.py` | 基础 MA / EMA / RSI / MACD 因子 IC 主程序 |
| `export_ic_summary_excel.py` | 把已有 IC CSV 整理成 Excel 汇总表 |
| `factor_redundancy.py` | 中性化因子相关性矩阵、冗余检验 |
| `factor_selection_ward.py` | Ward 聚类选非冗余因子 |
| `composite_equal.py` | 等权复合因子 IC |
| `composite_weighted.py` | ICIR 加权复合因子 IC |
| `portfolio_five_group.py` | Q1-Q5 五组分层回测 |
| `portfolio_industry_neutral.py` | 行业中性化权重下的 Q1-Q5 回测 |
| `compare_five_group_industry_neutral.py` | 比较原始五组 vs 行业中性五组结果 |
| `transaction_cost.py` | 五组回测交易成本模拟 |
| `limit_transaction_cost.py` | 交易成本 + 涨跌停交易约束模拟 |
| `decile_limit_failure.py` | Q1-Q10 十组分层 + 涨跌停交易失败率诊断 |
| `ic_decay.py` | 复合信号 1-60 日 IC decay |
| `turnover.py` | Q1-Q5 换手率统计 |
| `run_all.py` | 可选总入口；默认不全跑，避免耗时 |

## 你之后改代码时的建议

1. 只改 `config.py` 里的路径和全局参数。
2. 一个研究任务对应一个模块，例如：
   - 看 IC：改 `run_base_ic.py` / `ic_decay.py`
   - 看分层收益：改 `portfolio_five_group.py` 或 `portfolio_industry_neutral.py`
   - 看涨跌停交易失败：改 `decile_limit_failure.py`
   - 看换手率：改 `turnover.py`
3. 不要在基础函数模块里直接跑主程序。现在大多数 `run_xxx()` 都只会在你直接运行该模块时执行。
4. 老师临时加任务时，优先新建一个模块，不要继续往一个文件末尾追加。
