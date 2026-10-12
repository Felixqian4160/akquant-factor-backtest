import importlib
import sys
from pathlib import Path

sys.path.append(str(Path().cwd().parent))

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from FactorArithmetic.convergence_factor import calculate_convergence_factor
from src import standardlize
from src.analyzer import FactorAnalyzer
import src.data_provider as data_provider_module

data_provider_module = importlib.reload(data_provider_module)
LoadH5Data = data_provider_module.LoadH5Data

# plt显示中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
load_data: LoadH5Data = LoadH5Data()

# ===== CELL =====

def get_ashare_data(start_dt: str, end_dt: str, feilds: str) -> pd.DataFrame:
    """
    获取本地文件中获取数据。

    参数:
        start_dt (str): 开始日期，格式为YYYY-MM-DD。
        end_dt (str): 结束日期，格式为YYYY-MM-DD。
        feilds (str): 数据字段，用于指定要获取的数据。

    返回:
        pd.DataFrame: 获取到的A股数据。

    示例:
        >>> get_ashare_data('2022-01-01', '2022-01-31', 'close')
               000001.SZ  000002.SZ  600000.SH  ...
        date
        2022-01-01     10.50      20.30      15.40
        2022-01-02     10.60      20.40      15.50
        ...
    """
    return load_data.get_data(
        f"data.{feilds}", start_date=start_dt, end_date=end_dt
    ).filter(regex="^[036].*(SH$|SZ$)", axis=1)

# ===== CELL =====

START, END = "2014-01-01", "2023-12-31"  # 设置回测时间
PERIODS: List[int] = [5, 10, 20, 60, 120]  # 计算因子所需间隔参数
BEGIN: pd.Timestamp = load_data.get_shifted_trading_date(
    START, -np.max(PERIODS)
)  # 前序计算期

# ===== CELL =====

# 获取收盘价数据
close_df: pd.DataFrame = get_ashare_data(BEGIN, END, "adjclose")

# 因子构建
pcf_factor: pd.DataFrame = calculate_convergence_factor(close_df, PERIODS).loc[
    START:END
]

# ===== CELL =====

# 单因子分析
pcf_far: FactorAnalyzer = FactorAnalyzer(
    pcf_factor.stack().dropna(),
    close_df.loc[START:END],
    quantiles=10,
    periods=(1,),
)

# ===== CELL =====

pcf_far.plot_ic_ts(method="rank")

# ===== CELL =====

pcf_far.plot_cumulative_returns_by_quantile()

# ===== CELL =====

pcf_far.plot_quantile_returns_bar()

# ===== CELL =====

# 获取收盘价数据
volume_df: pd.DataFrame = get_ashare_data(BEGIN, END, "volume")

# 因子构建
vcf_factor: pd.DataFrame = calculate_convergence_factor(volume_df, PERIODS).loc[
    START:END
]

# ===== CELL =====

# 单因子分析
vcf_far: FactorAnalyzer = FactorAnalyzer(
    vcf_factor.stack().dropna(),
    close_df.loc[START:END],
    quantiles=10,
    periods=(1,),
)

# ===== CELL =====

vcf_far.plot_ic_ts(method="rank")

# ===== CELL =====

vcf_far.plot_cumulative_returns_by_quantile()

# ===== CELL =====

vcf_far.plot_quantile_returns_bar()

# ===== CELL =====

# PCF\VCF截面标准化后合成PVCF因子
pvcf_factor: pd.DataFrame = standardlize(pcf_factor) + standardlize(vcf_factor)

# ===== CELL =====

corr_ts: pd.Series = pcf_factor.corrwith(vcf_factor, method="spearman", axis=1)
ax = corr_ts.plot(figsize=(18, 4), title="时序相关性")
ax.axhline(corr_ts.mean(), color="r")
print(f"corr均值{corr_ts.mean():.4f}")

# ===== CELL =====

# 单因子分析
pvcf_far: FactorAnalyzer = FactorAnalyzer(
    pvcf_factor.stack().dropna(),
    close_df.loc[START:END],
    quantiles=10,
    periods=(1,),
)

# ===== CELL =====

pvcf_far.plot_ic_ts(method="rank")

# ===== CELL =====

pvcf_far.plot_cumulative_returns_by_quantile()

# ===== CELL =====

pvcf_far.plot_quantile_returns_bar()

# ===== CELL =====

# 获取成交额数据
amount_df: pd.DataFrame = get_ashare_data(BEGIN, END, "amount")

# ===== CELL =====

# 计算因子
acf_factor: pd.DataFrame = calculate_convergence_factor(amount_df, PERIODS).loc[
    START:END
]

# ===== CELL =====

# 单因子分析
acf_far: FactorAnalyzer = FactorAnalyzer(
    acf_factor.stack().dropna(),
    close_df.loc[START:END],
    quantiles=10,
    periods=(1,),
)

# ===== CELL =====

acf_far.plot_ic_ts(method="rank")

# ===== CELL =====

acf_far.plot_cumulative_returns_by_quantile()

# ===== CELL =====

acf_far.plot_quantile_returns_bar()

# ===== CELL =====

turn_df: pd.DataFrame = get_ashare_data(BEGIN, END, "turn")

# ===== CELL =====

trcf_factor: pd.DataFrame = calculate_convergence_factor(turn_df, PERIODS).loc[
    START:END
]

# ===== CELL =====

# 单因子分析
trcf_far: FactorAnalyzer = FactorAnalyzer(
    trcf_factor.stack().dropna(),
    close_df.loc[START:END],
    quantiles=10,
    periods=(1,),
)

# ===== CELL =====

trcf_far.plot_ic_ts(method="rank")

# ===== CELL =====

trcf_far.plot_cumulative_returns_by_quantile()

# ===== CELL =====

trcf_far.plot_quantile_returns_bar()

# ===== CELL =====

def plot_ic_ir_chart(far_dict: Dict):
    """
    绘制IC和IR图表。

    参数:
    far_dict (Dict): 包含多个FactorAnalysisResult对象的字典。

    返回:
    无返回值，直接显示图表。
    """
    ic: List = [far.ic["period_1"].mean() for far in far_dict.values()]
    ir: List = [
        far.ic["period_1"].mean() / far.ic["period_1"].std()
        for far in far_dict.values()
    ]

    df: pd.DataFrame = pd.DataFrame({"IC": ic, "IR": ir}, index=far_dict.keys())

    ax = df["IC"].plot(kind="bar", color="blue", label="IC")
    ax2 = ax.twinx()
    df["IR"].plot(kind="line", color="red", ax=ax2, label="IR")
    ax.figure.legend()
    plt.show()

# ===== CELL =====

far_dict: Dict = {
    "PCF": pcf_far,
    "VCF": vcf_far,
    "PVCF": pvcf_far,
    "ACF": acf_far,
    "TRCF": trcf_far,
}

plot_ic_ir_chart(far_dict)