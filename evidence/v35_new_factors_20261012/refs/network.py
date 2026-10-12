from pathlib import Path
import sys


sys.path.append(str(Path().cwd().parent))
sys.path.append("/data1/hugo/workspace")
sys.path.insert(0, "/data1/hugo/workspace/qlib_ddb")

# ===== CELL =====

from src.generator import NetworkCentralityFactor
from src.analyze import factor_group_analysis
from src.factor_algo import generate_factor

# ===== CELL =====

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# 显示中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===== CELL =====

def plot_factor_perf(pivot_df: pd.DataFrame, title: str):
    """
    绘制因子分组表现图。

    生成两幅子图：
    - 左侧：各分组累计收益曲线（按行累计求和）
    - 右侧：各分组平均日收益柱状图（按列平均）

    :param pivot_df: 因子分组收益透视表，行索引为日期，列为分组编号，值为日收益。
    :type pivot_df: pandas.DataFrame
    :param title: 图表标题。
    :type title: str
    :return: (fig, axes) 元组；`fig` 为 Figure 对象，`axes` 为长度为 2 的子图数组。
    :rtype: Tuple[matplotlib.figure.Figure, numpy.ndarray]

    注意
    ----
    - y 轴使用百分比格式（`PercentFormatter(xmax=1.0)`），适用于值已为“收益率”。
    - 若 `pivot_df` 的值不是比例（如以百分数或整数存储），请先转换为 0-1 区间比例。
    - 图右侧以 0 为基线，便于比较各分组的平均收益。
    """
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, 6),
        gridspec_kw={"width_ratios": [2, 1]},
        constrained_layout=True,
    )
    pivot_df.cumsum().plot(ax=axes[0])
    pivot_df.mean().plot.bar(ax=axes[1])
    axes[1].axhline(0, color="black")

    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=2))
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=2))

    fig.suptitle(title, fontsize=14)
    return fig, axes

# ===== CELL =====

factor_generator = NetworkCentralityFactor(
    codes="ashares", start_date="2017-01-01", end_date="2025-11-30"
)
returns_df = factor_generator.fetch_data()

# ===== CELL =====

# 计算SCC因子
scc_df = factor_generator.get_scc_factor(
    window=20,  # 20天滑动窗口
    ignore_errors=False,  # 忽略单个窗口错误
    show_progress=True,  # 显示进度条
    use_cache=True,  # 使用缓存
)

# ===== CELL =====

# 储存
scc_df.to_parquet("../data/scc_factor.parquet")

# ===== CELL =====

score_df: pd.DataFrame = scc_df.stack().to_frame("scc")

# ===== CELL =====

pred_label_df, pivot_df = factor_group_analysis(
    score_df,
    factor_col="scc",
    group_count=10,
    filter_limit=True,
    filter_suspended=True,
)

plot_factor_perf(pivot_df, "因子分析-SCC")

# ===== CELL =====

tcc_df: pd.DataFrame = factor_generator.get_tcc_factor(
    window=20,  # 20天滑动窗口
    ignore_errors=False,  # 忽略单个窗口错误
    show_progress=True,  # 显示进度条
    use_cache=True,  # 使用缓存
)

# ===== CELL =====

# 储存
tcc_df.to_parquet("data/tcc_factor.parquet")

# ===== CELL =====

tcc_stack: pd.DataFrame = tcc_df.stack().to_frame("tcc")

# ===== CELL =====

pred_label_df, pivot_df = factor_group_analysis(
    tcc_stack,
    factor_col="tcc",
    group_count=10,
    filter_limit=True,
    filter_suspended=True,
)

plot_factor_perf(pivot_df, "因子分析-TCC")

# ===== CELL =====

# 在没有计算scc和tcc的情况下，计算CC因子
# cc_df: pd.DataFrame = factor_generator.get_cc_factor(
#     window=20,  # 20天滑动窗口
#     ignore_errors=False,  # 忽略单个窗口错误
#     show_progress=True,  # 显示进度条
#     use_cache=True,  # 使用缓存
# )

# 这里直接使用已经计算好的scc_df和tcc_df
cc_df: pd.DataFrame = (scc_df + tcc_df) * 0.5

cc_stack: pd.DataFrame = cc_df.stack().to_frame("cc")

# ===== CELL =====

pred_label_df, pivot_df = factor_group_analysis(
    cc_stack,
    factor_col="cc",
    group_count=10,
    filter_limit=True,
    filter_suspended=True,
)

plot_factor_perf(pivot_df, "因子分析-CC")

# ===== CELL =====

from src.data_provider import QlibDataProvider

from KnightTask.StockLinkStrategy.src.qlib_backtest_engine import (
    BacktestConfig,
    StrategyConfig,
    QlibBacktestEngine,
)

# ===== CELL =====

QlibDataProvider.init_qlib_once()

# ===== CELL =====

scc_score: pd.DataFrame = pd.read_parquet("data/scc_factor.parquet")

# ===== CELL =====

scc_score: pd.DataFrame = scc_score.stack().to_frame("scc")
scc_score.index.names = ["datetime", "instrument"]

# ===== CELL =====

comustom_backtest_config = BacktestConfig(
    start_time="2017-01-01", end_time="2025-11-30"
)

engine = QlibBacktestEngine(
    scc_score, comustom_backtest_config, StrategyConfig(topk=50)
)
portfolio_metric_dict, indicator_dict = engine.run_backtest()

# ===== CELL =====

engine.generate_report_graph()

# ===== CELL =====

engine.calculate_risk_metrics()