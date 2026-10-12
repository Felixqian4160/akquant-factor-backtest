import empyrical as ep
import pandas as pd
import numpy as np
import qlib
from qlib.data import D
from qlib.workflow import R  # 实验记录管理器
# from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord, SignalRecord
from qlib.data.dataset.loader import StaticDataLoader
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset import DatasetH
from qlib.data.dataset.processor import DropnaLabel, ProcessInf, CSRankNorm, Fillna
# from qlib.utils import init_instance_by_config
from typing import List, Tuple, Dict

from scr.core import calc_sigma, calc_weight
from scr.factor_analyze import clean_factor_data, get_factor_group_returns
from scr.qlib_workflow import run_model
from scr.plotting import model_performance_graph, report_graph

import matplotlib.pyplot as plt
import seaborn as sns

# plt中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
# plt显示负号
plt.rcParams["axes.unicode_minus"] = False

# ===== CELL =====

qlib.init(provider_uri="qlib_data", region="cn")

# ===== CELL =====

# 使用D.feature与DataLoader,DataHandlerLP,DatasetH获取数据的数据MutiIndex索引不同
# 前者Instrument,datetime后者是datetime,Instrument
POOLS: List = D.list_instruments(D.instruments("pool"), as_list=True)
pct_chg: pd.DataFrame = D.features(POOLS, fields=["$close/Ref($close,1)-1"])
pct_chg: pd.DataFrame = pct_chg.unstack(level=0)["$close/Ref($close,1)-1"]

# 未来期收益
next_ret: pd.DataFrame = D.features(POOLS, fields=["Ref($open,-2)/Ref($open,-1)-1"])
next_ret.columns = ["next_ret"]
next_ret: pd.DataFrame = next_ret.swaplevel()
next_ret.sort_index(inplace=True)

# 基准
bench: pd.DataFrame = D.features(["000300.SH"], fields=["$close/Ref($close,1)-1"])
bench: pd.Series = bench.droplevel(level=0).iloc[:, 0]

# ===== CELL =====

# 计算w
w: pd.DataFrame = pct_chg.pipe(calc_sigma).pipe(calc_weight)
# 计算st因子
STR: pd.DataFrame = w.rolling(20).cov(pct_chg)

STR: pd.Series = STR.stack()
STR.name = "STR"

# ===== CELL =====

feature_df: pd.DataFrame = pd.concat((next_ret, STR), axis=1)
feature_df.columns = pd.MultiIndex.from_tuples(
    [("label", "next_ret"), ("feature", "STR")]
)

feature_df.head()

# ===== CELL =====

score_df:pd.DataFrame = feature_df.dropna().copy()
score_df.columns = ['label','score']

model_performance_graph(score_df)

# ===== CELL =====

# 计算获得惊恐度,准准收益使用的沪深300收益
sigma: pd.DataFrame = pct_chg.pipe(calc_sigma, bench=bench)
# 加权决策分
weighted: pd.DataFrame = sigma.mul(pct_chg)
# 加权决策分均值
avg_score: pd.DataFrame = weighted.rolling(20).mean()

avg_score_ser: pd.Series = avg_score.stack()
avg_score_ser.name = "avg_score"

# 加权决策分标准差
std_score: pd.DataFrame = weighted.rolling(20).std()

std_score_ser: pd.Series = std_score.stack()
std_score_ser.name = "std_score"

# 等权合成惊恐度得分 - 后续可以用qlib的模型合成寻找最优
terrified_score: pd.DataFrame = (avg_score + std_score) * 0.5

terrified_score_ser: pd.Series = terrified_score.stack()
terrified_score_ser.name = "terrified_score"

terrified_df: pd.DataFrame = pd.concat(
    (avg_score_ser, std_score_ser, terrified_score_ser, next_ret), axis=1
)
terrified_df.sort_index(inplace=True)

terrified_df.head()

# ===== CELL =====

group_returns: pd.DataFrame = (terrified_df.pipe(pd.DataFrame.dropna)
                                           .pipe(clean_factor_data)
                                           .pipe(get_factor_group_returns, quantile=5))

group_cum:pd.DataFrame = ep.cum_returns(group_returns)

# ===== CELL =====

# 画图
for factor_name, df in group_cum.groupby(level=0, axis=1):
    df.plot(title=factor_name, figsize=(12, 6))
    plt.axhline(0, ls="--", color="black")

# ===== CELL =====

test_df:pd.DataFrame = terrified_df[['avg_score','std_score','next_ret']].copy()
test_df.columns = pd.MultiIndex.from_tuples([("feature",'avg_score'),('feature','std_score'),('label',"next_ret")])

# ===== CELL =====

TARIN_PERIODS: Tuple = ("2014-01-01", "2017-12-31")
VALID_PERIODS: Tuple = ("2018-01-01", "2019-12-31")
TEST_PERIODS: Tuple = ("2020-01-01", "2023-02-17")

learn_processors = [DropnaLabel()]
infer_processors = [ProcessInf(), CSRankNorm(), Fillna()]

sdl = StaticDataLoader(config=test_df)
dh_pr = DataHandlerLP(
    instruments=POOLS,
    start_time=TARIN_PERIODS[0],
    end_time=TEST_PERIODS[1],
    process_type=DataHandlerLP.PTYPE_A,
    learn_processors=learn_processors,
    infer_processors=infer_processors,
    data_loader=sdl,
)

ds = DatasetH(
    dh_pr,
    segments={"train": TARIN_PERIODS, "valid": VALID_PERIODS, "test": TEST_PERIODS},
)

# ===== CELL =====

record_dict: Dict = run_model(
    ds,
    "gbdt",
    start_time=TEST_PERIODS[0],
    end_time=TEST_PERIODS[1],
    experiment_name="terrified",
    trained_model="trained_model.pkl",
)

# ===== CELL =====

try:
    recorder = record_dict['recorder']
except NameError:
    # 使用已有模型
    from qlib.workflow import R
    import pickle

    with open("../筹码分布算法/factor_data/turnovercoeff_dataset.pkl", "rb") as f:
        turncoeff_dataset = pickle.load(f)

    with R.start():
        recorder = R.get_recorder(
            recorder_name="mlflow_recorder",
            recorder_id="97284ccb8e274ffe83e34fa8f9d84b7e",
        )

label_df = recorder.load_object("label.pkl")
label_df.columns = ["label"]
pred_df: pd.DataFrame = recorder.load_object("pred.pkl")

# 创建测试集"预测"和“标签”对照表
pred_label_df: pd.DataFrame = pd.concat([pred_df, label_df], axis=1, sort=True).reindex(
    label_df.index
)

# ===== CELL =====

model_performance_graph(pred_label_df,duplicates="drop")

# ===== CELL =====

report_normal_1day_df: pd.DataFrame = recorder.load_object(
    "portfolio_analysis/report_normal_1day.pkl")

# ===== CELL =====

report_graph(report_normal_1day_df)

# ===== CELL =====

def get_stv_feature() -> str:
    abs_ret: str = "Abs($close/Ref($close,1)-1)"
    return f"If({abs_ret}>=0.1,{abs_ret}*100,$turnover_rate)"

# ===== CELL =====

sigma_frame:pd.DataFrame = D.features(POOLS,fields=[get_stv_feature()])

sigma_frame.columns = ['sigma']

sigma_frame:pd.DataFrame = sigma_frame.unstack(level=0)['sigma']

# ===== CELL =====

stv_w:pd.DataFrame = calc_weight(sigma_frame)
STV:pd.DataFrame = stv_w.rolling(20).cov(pct_chg)

STV:pd.Series = STV.stack()
STV.name = "STV"

# ===== CELL =====

feature_stv: pd.DataFrame = pd.concat(
    (next_ret, STV), axis=1
)
feature_stv.columns = pd.MultiIndex.from_tuples(
    [("label", "next_ret"), ("feature", "STV")]
)

feature_stv.head()

# ===== CELL =====

stv_score:pd.DataFrame = feature_stv.dropna().copy()
stv_score.columns = ['label','score']

model_performance_graph(stv_score)

# ===== CELL =====

all_data: pd.DataFrame = pd.concat(
    [STR, STV, avg_score_ser, std_score_ser, next_ret], axis=1, sort=True
).dropna()

all_data.columns = pd.MultiIndex.from_tuples(
    [
        ("feature", "STR"),
        ("feature", "STV"),
        ("feature", "avg_score"),
        ("feature", "std_score"),
        ("label", "next_ret"),
    ]
)

# ===== CELL =====

TARIN_PERIODS: Tuple = ("2014-01-01", "2017-12-31")
VALID_PERIODS: Tuple = ("2018-01-01", "2019-12-31")
TEST_PERIODS: Tuple = ("2020-01-01", "2023-02-17")

learn_processors = [DropnaLabel()]
infer_processors = [ProcessInf(), CSRankNorm(), Fillna()]

sdl = StaticDataLoader(config=all_data)
dh_pr = DataHandlerLP(
    instruments=POOLS,
    start_time=TARIN_PERIODS[0],
    end_time=TEST_PERIODS[1],
    process_type=DataHandlerLP.PTYPE_A,
    learn_processors=learn_processors,
    infer_processors=infer_processors,
    data_loader=sdl,
)

ds = DatasetH(
    dh_pr,
    segments={"train": TARIN_PERIODS, "valid": VALID_PERIODS, "test": TEST_PERIODS},
)

# ===== CELL =====

record_dict: Dict = run_model(
    ds,
    "gbdt",
    start_time=TEST_PERIODS[0],
    end_time=TEST_PERIODS[1],
    experiment_name="factor",
    trained_model="trained_model.pkl",
)

# ===== CELL =====

try:
    recorder = record_dict['recorder']
except NameError:
    # 使用已有模型
    from qlib.workflow import R
    import pickle

    with open("../筹码分布算法/factor_data/turnovercoeff_dataset.pkl", "rb") as f:
        turncoeff_dataset = pickle.load(f)

    with R.start():
        recorder = R.get_recorder(
            recorder_name="mlflow_recorder",
            recorder_id="7eefa18e7b4341f680927657e1de8fa0",
        )

label_df = recorder.load_object("label.pkl")
label_df.columns = ["label"]
pred_df: pd.DataFrame = recorder.load_object("pred.pkl")

# 创建测试集"预测"和“标签”对照表
pred_label_df: pd.DataFrame = pd.concat([pred_df, label_df], axis=1, sort=True).reindex(
    label_df.index
)

# ===== CELL =====

model_performance_graph(pred_label_df,duplicates="drop")

# ===== CELL =====

report_normal_1day_df: pd.DataFrame = recorder.load_object(
    "portfolio_analysis/report_normal_1day.pkl")

# ===== CELL =====

report_graph(report_normal_1day_df)

# ===== CELL =====



# ===== CELL =====



# ===== CELL =====

