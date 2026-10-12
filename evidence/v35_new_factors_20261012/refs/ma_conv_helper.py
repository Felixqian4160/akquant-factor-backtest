'''
Author: Hugo
Date: 2024-06-05 15:05:14
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2024-06-05 17:02:26
Description: 

FROM:20240414-开源证券-开源量化评论（91）：形态识别，均线的收敛与发散

价量收敛因子的计算方法
'''
from typing import List, Tuple, Union
import pandas as pd
import numpy as np

def calculate_convergence_factor(
    data: pd.DataFrame, windows: Union[List, Tuple]
) -> pd.DataFrame:
    """
    计算收敛因子。

    参数：
        - data (pd.DataFrame): 输入的数据框，包含需要计算收敛因子的数据。
        - windows (Union[List, Tuple]): 一个列表或元组，包含需要计算移动平均线的窗口大小。

    返回：
        - pd.DataFrame: 包含收敛因子的数据框，索引与输入数据相同，列名与输入数据相同。

    Raises:
        - ValueError: 如果输入的data不是DataFrame类型。
        - ValueError: 如果windows不是列表或元组类型。

    注意：
        - 这里包含数组自己本身及MA1。
    """
    if not isinstance(data, pd.DataFrame):

        raise ValueError("data must be a DataFrame")

    if isinstance(windows, (list, tuple)):
        windows = windows
    else:
        raise ValueError("windows must be a list or tuple")

    # NOTE:注意这里包含数组自己本身及MA1
    ma_arrs: List[np.ndarray] = np.stack(
        [data.values] + [data.rolling(window).mean().values for window in windows],
        axis=2,
    )
    std_arr: np.ndarray = np.std(ma_arrs, axis=2, ddof=1)
    factor: np.ndarray = np.log(1 + std_arr) * -1

    return pd.DataFrame(factor, index=data.index, columns=data.columns)

