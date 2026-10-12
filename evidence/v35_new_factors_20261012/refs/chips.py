import empyrical as ep
import pandas as pd

from pathlib import Path
import qlib
from qlib.data import D
from qlib.utils import init_instance_by_config
from typing import List, Tuple, Dict
from scr.turnover_coefficient_ops import ARC, VRC, SRC, KRC
from scr.cyq_ops import (
    CYQK_C_T,
    CYQK_C_U,
    CYQK_C_TN,
    ASR_T,
    ASR_U,
    ASR_TN,
    CKDW_T,
    CKDW_U,
    CKDW_TN,
    PRP_T,
    PRP_U,
    PRP_TN,
)
from scr.factor_analyze import clean_factor_data, get_factor_group_returns
from scr.qlib_workflow import run_model, get_dataset_config, get_tsdataset_config
from scr.plotting import plot_dist_chips, model_performance_graph, report_graph

import matplotlib.pyplot as plt

# plt中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
# plt显示负号
plt.rcParams["axes.unicode_minus"] = False

# ===== CELL =====

custom_ops: List = [ARC, VRC, SRC, KRC, CYQK_C_T, CYQK_C_U, CYQK_C_TN,
                    ASR_T, ASR_U, ASR_TN, CKDW_T, CKDW_U, CKDW_TN, PRP_T, PRP_U, PRP_TN]
qlib.init(provider_uri="qlib_data",
          region="cn", custom_ops=custom_ops)

# ===== CELL =====

###################################
# 参数配置
###################################
# 数据处理器参数配置：整体数据开始结束时间，训练集开始结束时间，股票池
TARIN_PERIODS: Tuple = ("2014-01-01", "2017-12-31")
VALID_PERIODS: Tuple = ("2018-01-01", "2020-12-31")
TEST_PERIODS: Tuple = ("2021-01-01", "2023-02-17")


dataset_config: Dict = get_dataset_config(
    "pool", TARIN_PERIODS, VALID_PERIODS, TEST_PERIODS, "TurnCoeffChips"
)


if Path("factor_data/turnovercoeff_dataset.pkl").exists():
    import pickle

    with open("factor_data/turnovercoeff_dataset.pkl", "rb") as f:
        turncoeff_dataset = pickle.load(f)
else:
    # 实例化数据集，从基础行情数据计算出的包含所有特征（因子）和标签值的数据集。
    turncoeff_dataset = init_instance_by_config(dataset_config)  # 类型DatasetH

    # 保存数据方便后续使用
    turncoeff_dataset.config(dump_all=True, recursive=True)
    turncoeff_dataset.to_pickle(
        path="factor_data/turnovercoeff_dataset.pkl", dump_all=True
    )

record_dict: Dict = run_model(
    turncoeff_dataset,
    "transformer",
    start_time=TEST_PERIODS[0],
    end_time=TEST_PERIODS[1],
    model_kw={"d_feat": 4, "num_layers": 16, "nhead": 8, "method": "normal"},
    experiment_name="turncoeffchips",
    trained_model="trained_model.pkl",
)

# ===== CELL =====

df_test: pd.DataFrame = turncoeff_dataset.prepare(
    segments="test", data_key="raw")
df_test.rename(columns={'LABEL0': "next_ret"}, inplace=True)

# ===== CELL =====

clean_factor: pd.DataFrame = clean_factor_data(df_test)
# 计算分组收益率
group_returns: pd.DataFrame = get_factor_group_returns(
    clean_factor, quantile=5, no_raise=False
)
# 计算累计收益率
group_cum: pd.DataFrame = ep.cum_returns(group_returns)

# 画图
for factor_name, df in group_cum.groupby(level=0, axis=1):
    df.plot(title=factor_name, figsize=(12, 6))
    plt.axhline(0, ls="--", color="black")

# ===== CELL =====

try:
    recorder = record_dict["recorder"]
except NameError:
    # 使用已有模型
    from qlib.workflow import R
    import pickle

    with open("factor_data/turnovercoeff_dataset.pkl", "rb") as f:
        turncoeff_dataset = pickle.load(f)
    # tfs pool 0accc05c53f74dc6827853d7b310307e
    # csi300 34342b9c77c941788eea7b85a27e243e
    # gbdt pool 277db25d948e4423b541d3382044c285
    with R.start():
        recorder = R.get_recorder(
            recorder_name="mlflow_recorder",
            recorder_id="b6fb35b51626493f835c86c7d8a2b1b3",
        )

# ===== CELL =====

label_df: pd.DataFrame = turncoeff_dataset.prepare("test", col_set="label")
label_df.columns = ["label"]
pred_df: pd.DataFrame = recorder.load_object("pred.pkl")

# 创建测试集"预测"和“标签”对照表
pred_label_df: pd.DataFrame = pd.concat([pred_df, label_df], axis=1, sort=True).reindex(
    label_df.index
)

# ===== CELL =====

model_performance_graph(pred_label_df, duplicates='drop')

# ===== CELL =====

report_normal_1day_df: pd.DataFrame = recorder.load_object(
    "portfolio_analysis/report_normal_1day.pkl")

# ===== CELL =====

report_graph(report_normal_1day_df)

# ===== CELL =====

# 数据获取
pool: List = D.list_instruments(D.instruments("csi300"), as_list=True)
print(pool[0])
frame: pd.DataFrame = D.features(
    [pool[0]],
    fields=["$close", "$high", "$low", "$vol", "$turnover_rate"],
)
frame.reset_index(level=0, inplace=True, drop=True)
frame.columns = frame.columns.str.replace("$", "", regex=True)

slice_frame: pd.DataFrame = frame.iloc[-80:].copy()

# ===== CELL =====

plot_dist_chips(slice_frame, "triang", "002202.SZ-三角分布")

# ===== CELL =====

plot_dist_chips(slice_frame, "uniform", "002202.SZ-平均分布")

# ===== CELL =====

plot_dist_chips(slice_frame, "turn_coeff", "002202.SZ-换手率衰减")

# ===== CELL =====

###################################
# 参数配置
###################################
# 数据处理器参数配置：整体数据开始结束时间，训练集开始结束时间，股票池
TARIN_PERIODS: Tuple = ("2014-01-01", "2017-12-31")
VALID_PERIODS: Tuple = ("2018-01-01", "2019-12-31")
TEST_PERIODS: Tuple = ("2020-01-01", "2023-02-17")


dataset_config: Dict = get_tsdataset_config(
    "csi300",
    TARIN_PERIODS,
    VALID_PERIODS,
    TEST_PERIODS,
    20,
    "Chips",
)


if Path("factor_data/chip_ts_dataset.pkl").exists():
    import pickle

    with open("factor_data/chip_ts_dataset.pkl", "rb") as f:
        cyq_ts_dataset = pickle.load(f)
else:
    cyq_ts_dataset = init_instance_by_config(dataset_config)  # 类型DatasetH

    # 保存数据方便后续使用
    cyq_ts_dataset.config(dump_all=True, recursive=True)
    cyq_ts_dataset.to_pickle(path="cyq_ts_dataset.pkl", dump_all=True)

cyq_record: Dict = run_model(
    cyq_ts_dataset,
    "transformer_ts",
    start_time=TEST_PERIODS[0],
    end_time=TEST_PERIODS[1],
    model_kw={"d_feat": 12,"num_layers":8, "seed": 42,'n_jobs':20},
    experiment_name="cyq",
    trained_model="trained_model.pkl",
)

# ===== CELL =====

# TSDatasetH生成数据并非pd.DataFrame
# 可以使用DatasetH或者QlibData来生成因子数据 因子生成差不多需要200~300min
try:
    cyq_dataset.prepare(segments="test", data_key="raw")
except NameError:
    import pickle

    if Path("factor_data/chip_dataset.pkl").exists():
        with open("factor_data/chip_dataset.pkl", "rb") as f:
            cyq_dataset = pickle.load(f)

    else:
        dataset_config: Dict = get_dataset_config(
            "csi300",
            TARIN_PERIODS,
            VALID_PERIODS,
            TEST_PERIODS,
            "Chips",
        )
        cyq_dataset = init_instance_by_config(dataset_config)  # 类型DatasetH

        # 保存数据方便后续使用
        cyq_dataset.config(dump_all=True, recursive=True)
        cyq_dataset.to_pickle(path="cyq_dataset.pkl", dump_all=True)

df_test: pd.DataFrame = cyq_dataset.prepare(segments="test", data_key="raw")
df_test.rename(columns={"LABEL0": "next_ret"}, inplace=True)

# ===== CELL =====

clean_factor: pd.DataFrame = clean_factor_data(df_test)
# 计算分组收益率
group_returns: pd.DataFrame = get_factor_group_returns(
    clean_factor, quantile=5, no_raise=True
)
# 计算累计收益率
group_cum: pd.DataFrame = ep.cum_returns(group_returns)

# 画图
for factor_name, df in group_cum.groupby(level=0, axis=1):
    df.plot(title=factor_name, figsize=(12, 6))
    plt.axhline(0, ls="--", color="black")

# ===== CELL =====

try:
    recorder = record_dict["recorder"]
except NameError:
    # 使用已有模型
    from qlib.workflow import R
    import pickle

    with open("factor_data/chip_dataset.pkl", "rb") as f:
        cyq_dataset = pickle.load(f)
    # tcn layer 8:5a8fbb3d07ca4bcf8883589dcebad44d
    # transform layer 2:e7539ce11aad4c8c92c3e2be518b9106
    # transform layer 8 :e8e62a5185394314b5a7d79a17160059
    with R.start():
        recorder = R.get_recorder(
            recorder_name="mlflow_recorder",
            recorder_id="5a8fbb3d07ca4bcf8883589dcebad44d",
        )

# ===== CELL =====

label_df = recorder.load_object("label.pkl")
label_df.columns = ["label"]
pred_df: pd.DataFrame = recorder.load_object("pred.pkl")

# 创建测试集"预测"和“标签”对照表
pred_label_df: pd.DataFrame = pd.concat([pred_df, label_df], axis=1, sort=True).reindex(
    label_df.index
)

# ===== CELL =====

model_performance_graph(pred_label_df, duplicates='drop')

# ===== CELL =====

report_normal_1day_df: pd.DataFrame = recorder.load_object(
    "portfolio_analysis/report_normal_1day.pkl")

# ===== CELL =====

report_graph(report_normal_1day_df)