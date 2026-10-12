import sys
sys.path.append('../..')

from BuildPeriodicDate import *

import numpy as np
import pandas as pd
import alphalens as al

from dateutil.parser import parse
from tqdm import tqdm_notebook
from scipy import stats # 统计检验
import statsmodels.api as sm # 回归


from jqdata import *
from jqfactor import (analyze_factor,Factor,calc_factors)

from pylab import mpl
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'

# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False

plt.style.use('seaborn')

# ===== CELL =====

# 这个应该是计算RP的核心部分
def calc_turnover_weight(arr: np.array) -> np.array:
    '''换手率衰减加权'''

    arr_ = arr.copy()
    arr_[0] = 0

    return np.multiply(np.cumprod(np.subtract(1, np.roll(arr_, -1))[::-1])[::-1], arr)

def calc_rp(turnover:pd.DataFrame,ma_avg:pd.DataFrame)->pd.Series:
    '''
    计算参考价格
    '''
     # 换手率衰减加权 这个位置决定了无法滚动 这里传入的数据长度N计算处理的就是N日的CGO
    turnover_weighs = turnover.apply(calc_turnover_weight)
    # 归一化
    scale_weights = turnover_weighs / turnover_weighs.sum()
   
    return (scale_weights * ma_avg).sum()  # 参考价格


def calc_roll_rp(df:pd.DataFrame,N:int) ->pd.DataFrame:
    '''
    计算roll_N日的参考价格
    ---------------
        传入的df数据结构必须为
        df:index-date columns level0-close,ma_avg,turnover_ratio,level1-code
        ------------------------------------------------------------
        |          |      close    |      avg      | turnover_ratio|
        |    day   |code1|code2|...|code1|code2|...|code1|code2|...|
        ------------------------------------------------------------
        |2020-11-01| 0.1 | 0.3 |...| 0.3 | 0.5 |...|0.03 | 0.9 |...|
        ------------------------------------------------------------
    '''
    iidx = np.arange(len(df))

    shape = (iidx.size - N + 1, N)

    strides = (iidx.strides[0], iidx.strides[0])

    step = np.lib.stride_tricks.as_strided(
        iidx, shape=shape, strides=strides, writeable=True)
    
    turnover = df['turnover_ratio']
    ma_avg = df['avg']
 
    rp_df = pd.concat((calc_rp(turnover.iloc[i],ma_avg.iloc[i]) for i in step),axis=1).T
    
    rp_df.index = df.index[N-1:]
    
    return rp_df
    

    
def calc_cgo(df:pd.DataFrame,N:int) -> pd.DataFrame:
    '''
    计算CGO因子
    传入的如果是N日的则计算N日的CGO
    -----------
       传入的df数据结构必须为
        df:index-date columns level0-close,ma_avg,turnover_ratio,level1-code
        ------------------------------------------------------------
        |          |      close    |      avg      | turnover_ratio|
        |    day   |code1|code2|...|code1|code2|...|code1|code2|...|
        ------------------------------------------------------------
        |2020-11-01| 0.1 | 0.3 |...| 0.3 | 0.5 |...|0.03 | 0.9 |...|
        ------------------------------------------------------------
    -----------
        return:
            pd.Series index-code values
    '''

    rp_df = calc_roll_rp(df,N)
    close = df['close'].reindex(rp_df.index)
    
    return close / rp_df - 1,rp_df


# ===== CELL =====

# 获取市值数据
# 绕过1W的数据限制
# 经测试发现比直接用offset快


def query_valuation(symbol: list,
                    start_date: str,
                    end_date: str,
                    fields: list,
                    limit=10000) -> pd.DataFrame:

    n_symbols = len(symbol)
    dates = get_trade_days(start_date, end_date)
    n_days = len(dates)

    if n_symbols * n_days > limit:

        n = limit // n_symbols

        df_list = []

        i = 0

        pos1, pos2 = n * i, n * (i + 1) - 1

        while pos2 < n_days:

            df = get_valuation(
                symbol,
                start_date=dates[pos1],
                end_date=dates[pos2],
                fields=fields)

            df_list.append(df)

            i += 1

            pos1, pos2 = n * i, n * (i + 1) - 1

        if pos1 < n_days:

            df = get_valuation(
                symbol,
                start_date=dates[pos1],
                end_date=dates[-1],
                fields=fields)

            df_list.append(df)

        df = pd.concat(df_list, axis=0)

    else:

        df = get_valuation(
            symbol, start_date=start_date, end_date=end_date, fields=fields)

    return df

# 获取CGO因子


def prepare_data(symbol: str, start: str, end: str, N: int) -> pd.DataFrame:
    '''CGO因子获取'''
    periods = GetPeriodicDate(start, end).get_periods  # 获取调仓片段

    cgo_list = []  # 储存切片
    rp_list = []

    for s, e in tqdm_notebook(periods, desc='CGO因子计算中'):

        begin = get_trade_days(end_date=s, count=N)[0].strftime('%Y-%m-%d')
        stocks = get_index_stocks(symbol, date=e)  # 成分股获取
        df = query_data(stocks, begin, e, N)  # 数据获取
        f, r = calc_cgo(df, N)
        cgo_list.append(f)
        rp_list.append(r)

    factor_df = pd.concat(cgo_list, sort=True)  # 拼接
    rp_df = pd.concat(rp_list, sort=True)
    return factor_df, rp_df

# 获取CGO因子-step1


def query_data(stocks: list, start: str, end: str, N: int) -> pd.DataFrame:
    '''
    获取计算CGO因子的基础数据
    ---------
        stocks:list
        N:计算均线的周期
    --------
        返回的数据结构如下
        ------------------------------------------------------------
        |          |      close    |      avg      | turnover_ratio|
        |    day   |code1|code2|...|code1|code2|...|code1|code2|...|
        ------------------------------------------------------------
        |2020-11-01| 0.1 | 0.3 |...| 0.3 | 0.5 |...|0.03 | 0.9 |...|
        ------------------------------------------------------------
    '''

    price = get_price(stocks, start, end, fields=['close', 'avg'], panel=False)

    # 获取换手率数据
    turn_df = query_valuation(
        stocks, start, end, fields=['turnover_ratio'])
    turn_df['turnover_ratio'] = turn_df['turnover_ratio'] / 100
    # 调整数据类型
    turn_df['day'] = pd.to_datetime(turn_df['day'])
    # 修改名称方便后续拼接
    turn_df.rename(columns={'day': 'time'}, inplace=True)
    # 数据拼接
    data = pd.merge(turn_df, price, on=['time', 'code'], how='right')
    data.sort_values('time', inplace=True)
    data.reset_index(drop=True, inplace=True)

    # .iloc[N-1:] # 去除计算ma_avg的前序期
    return data.pivot(index='time', columns='code')



def plot_nav(nav: pd.DataFrame, title: str):
    '''
    画图
    ---------
        nav数据结构
        -------------------------------------
        |  date    | 1  | 2  | 3  | 4  | 5  |
        -------------------------------------
        |2020-01-01|0.01|0.02|0.14|-0.2|0.08|
        -------------------------------------
        |2020-01-02|0.11|-0.2|-0.1|0.33|0.10|
        -------------------------------------
    '''

    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(18, 6))
    # 设置标题
    plt.title(title)
    # 1,5组设置不同的颜色和线型方便查看单调性
    ax.plot(nav[1.], color='Navy')
    ax.plot(nav[2.], color='LightGrey', ls='-.')
    ax.plot(nav[3.], color='DimGray', ls='-.')
    ax.plot(nav[4.], color='DarkKhaki', ls='-.')
    ax.plot(nav[5.], color='LightSteelBlue')

    ax.axhline(1, color='black', lw=0.5)
    # 多空单独反应
    ax1 = ax.twinx()
    ax1.plot(nav['G1-G5'], color='r', ls='--')
    ax1.grid(None)
    # 图例合并
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax1.get_legend_handles_labels()
    l2 = [l2[0] + '(right)']
    ax.legend(h1+h2, l1+l2)

# ===== CELL =====

class CGO_100(Factor):
    
    '''计算窗口为100的CGO因子'''
    
    import warnings
    warnings.filterwarnings("ignore")
    
    name = 'CGO_100'
    max_window = 100 
    
    dependencies = ['close','turnover_ratio','money','volume']
    
    def calc(self,data):
        
        avg = data['money'] / data['volume'] # 数据索引不是日期!!!
        
        turnover = data['turnover_ratio'] / 100
        turnover_weighs = turnover.apply(calc_turnover_weight)

        # 归一化
        scale_weights = turnover_weighs.div(turnover_weighs.sum())
        avg.index = scale_weights.index # 设置索引
        rp_price = (scale_weights * avg).sum()  # 参考价格
        
        
        return data['close'].iloc[-1] / rp_price - 1
    
class RP_100(Factor):
    
    '''计算窗口为100的参考价格(Reference Price)'''
    
    name = 'CGO_100'
    max_window = 100 
    
    dependencies = ['close','turnover_ratio','money','volume']
    
    def calc(self,data):
        
        avg = data['money'] / data['volume'] # 数据索引不是日期!!!
        
        turnover = data['turnover_ratio'] / 100
        turnover_weighs = turnover.apply(calc_turnover_weight)

        # 归一化
        scale_weights = turnover_weighs.div(turnover_weighs.sum())
        avg.index = scale_weights.index # 设置索引
        return (scale_weights * avg).sum()  # 参考价格
    

# 除非你真的很闲 不然真不建议 这样直接运行 因为你可能会等很久....
# far = analyze_factor(factor=CGO_100, 
#                      start_date='2014-01-01', 
#                      end_date='2020-11-30', 
#                      industry='sw_l1', 
#                      universe='000906.XSHG', 
#                      quantiles=5, 
#                      periods=(5,22), 
#                      weight_method='avg', 
#                      use_real_price=True, 
#                      skip_paused=True)

# ===== CELL =====

# 用时17min15s
cgo_df,rp_df = prepare_data('000906.XSHG','2010-01-01','2020-11-30',100)

# ===== CELL =====

# 因子储存
cgo_df.to_csv('../../Data/cgo_df.csv')
rp_df.to_csv('../../Data/rp_df.csv')

# ===== CELL =====

# 因子读取
cgo_df = pd.read_csv('../../Data/cgo_df.csv',index_col=[0],parse_dates=[0])
rp_df = pd.read_csv('../../Data/rp_df.csv',index_col=[0],parse_dates=[0])

# ===== CELL =====

# 去除na
drop_cgo_df = cgo_df.stack().replace([-np.inf, np.inf], np.nan).dropna()

plt.rcParams['font.family'] = 'serif'
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.title('全空间样本的CGO分布')
sns.distplot(drop_cgo_df)

plt.subplot(122)
plt.title('CGO中位数截面分布')
sns.distplot(drop_cgo_df.groupby(level='time').median())

# ===== CELL =====

# 中证800成分股中CGO大于零的股票的比例
count_num_1 = drop_cgo_df.groupby(level='time').apply(
    lambda x: np.sum(~np.signbit(x)) / len(x))

start = count_num_1.index.min().strftime('%Y-%m-%d')
end = count_num_1.index.max().strftime('%Y-%m-%d')


# 中证800成分股的CGO中位数的序列
count_num_2 = drop_cgo_df.groupby(level='time').median()

# 基准
benchmark = get_price('000906.XSHG', start, end, fields='close')


plt.figure(figsize=(12, 8))
# 画图1
plt.subplot(211)
line = benchmark['close'].plot(
    color='DarkGray', label='中证800', title='中证800成分股中CGO大于零的股票的比例')

out = count_num_1.plot(secondary_y=True, ax=line,
                       color='LightCoral', label='CGO大于0的比率')

h1, l1 = line.get_legend_handles_labels()
h2, l2 = out.get_legend_handles_labels()
l2 = [l2[0] + '(left)']
plt.legend(h1+h2, l1+l2, loc=1)

# 画图2
plt.subplot(212)
line = benchmark['close'].plot(
    color='DarkGray', label='中证800', title='中证800成分股的CGO中位数的序列')

out = count_num_2.plot(secondary_y=True, ax=line,
                       color='LightCoral', label='CGO中位数')

h1, l1 = line.get_legend_handles_labels()
h2, l2 = out.get_legend_handles_labels()
l2 = [l2[0] + '(left)']
plt.legend(h1+h2, l1+l2, loc=1)

# ===== CELL =====

price = get_price('000001.XSHE','2017-10-01','2018-01-01',fields='close')
test_cgo = drop_cgo_df.xs('000001.XSHE',level=1)
price_line = price['close'].plot(figsize=(6,4),color='r')
out = test_cgo.reindex(price.index).plot(ax=price_line,secondary_y=True,label='CGO')

h1,l1 = price_line.get_legend_handles_labels()
l1 = [l1[0]+'(left)']
h2,l2 = out.get_legend_handles_labels()

plt.legend(h1 + h2,l1 + l2)

# ===== CELL =====

# 起始日向前推5日 结束日向后推15日 预留后期分析空间
start = cgo_df.index.min().strftime('%Y-%m-%d')
start = OffsetDate(start,-5)
end = cgo_df.index.max().strftime('%Y-%m-%d')
end = OffsetDate(end,15)

stocks = cgo_df.columns.tolist()
pricing = get_price(stocks,start,end,fields='close',panel=False)
pricing = pd.pivot_table(pricing,index='time',columns='code',values='close')

# ===== CELL =====

# 周度所以periods=5 pricing数据滞后一期 避免未来数据
# 因子排序为升序
factor_far = al.utils.get_clean_factor_and_forward_returns(factor=drop_cgo_df, 
                                                          prices=pricing.shift(-1), 
                                                          quantiles=5, 
                                                          periods=(5,))

# ===== CELL =====

factor_far.head()

# ===== CELL =====

cgo_5_ic = al.performance.factor_information_coefficient(factor_far)
cgo_5_ic.columns = ['CGO']

al.plotting.plot_information_table(cgo_5_ic)
al.plotting.plot_ic_ts(cgo_5_ic)

# ===== CELL =====

# 计算分组收益
rets = pd.pivot_table(factor_far.reset_index(level=0),
                      index='date', columns='factor_quantile', values=5)

rets['G1-G5'] = rets[1] - rets[5] 
nav = np.exp(np.log1p(rets).cumsum()) # 计算净值

# ===== CELL =====

plot_nav(nav,'中证800内CGO分位数组合')

# ===== CELL =====

# demeand参数在截面上对所有的股票收益率做demean处理
# 也称为中心化（Zero-centered 或者 Mean-subtraction）
mean_quant_ret, std_quantile = al.performance.mean_return_by_quantile(
        factor_far,
        by_group=False, 
        demeaned=True,  
        group_adjust=False, # 是否行业中性化
    )

mean_quant_ret.plot.bar(title='因子收益：分组平均超额收益（收益中心化处理）')

# ===== CELL =====

import talib
from jqfactor import get_factor_values


def query_other_factor(symbol: str, start: str, end: str) -> pd.DataFrame:

    periods = GetPeriodicDate(start, end).get_periods  # 获取调仓片段
    df_list = []
    
    for s, e in tqdm_notebook(periods, desc='因子获取中'):

        s, e = s.strftime('%Y-%m-%d'), e.strftime('%Y-%m-%d')
        begin = get_trade_days(end_date=s, count=28)[0].strftime('%Y-%m-%d')
        stocks = get_index_stocks(
            symbol, date=e)  # 成分股获取

        # 技术因子
        technical_factor = get_techniacl_factor(stocks, begin, e)
        technical_factor = technical_factor.set_index(['time', 'code'])
        idx = technical_factor.index.levels[0][28:]
        technical_factor = technical_factor.loc[idx]

        # size因子

        other_factor = get_factor_values(
            stocks, factors=['size'], start_date=s, end_date=e)

        other_factor = dict2frame(other_factor)
        
        # pe,换手率因子
        
        valuation_df = query_valuation(
            stocks, begin, e, ['pe_ratio', 'turnover_ratio'])
        
        # 计算TUNROVER_20
        
        valuation_df['turnover_ratio'] = valuation_df.groupby(
            'code')['turnover_ratio'].transform(lambda x: x.rolling(20).mean())
        
        valuation_df = valuation_df.set_index(['day', 'code'])
        idx = technical_factor.index.levels[0][28:]
        valuation_df = valuation_df.loc[idx]
        
        # 储存
        df_list.append(pd.concat(
            [technical_factor, other_factor, valuation_df], axis=1))

    factor_df = pd.concat(df_list, sort=True)  # 拼接
    factor_df.index.names = ['time','code']
    return factor_df


def get_techniacl_factor(stocks: str, start: str, end: str) -> pd.DataFrame:
    '''
    获取28日威廉指标,14日ATR,20日ROC
    '''
    price = get_price(stocks, start_date=start, end_date=end,
                      fields=['close', 'high', 'low'], panel=False)

    technical_df = price[['time', 'code']].copy()

    # 计算威廉指标
    technical_df['WR_28'] = price.groupby('code', group_keys=False).apply(
        lambda x: talib.WILLR(x['high'], x['low'], x['close'], 28))

    # ATR
    technical_df['ATR_14'] = price.groupby('code', group_keys=False).apply(
        lambda x: talib.ATR(x['high'], x['low'], x['close'], 14))

    # ROC
    technical_df['ROC_20'] = price.groupby('code', group_keys=False).apply(
        lambda x: talib.ROC(x['close'], 20))

    # 振幅
    technical_df['ZF_15'] = price.groupby(
        'code', group_keys=False).apply(calc_amplitude, 15)

    return technical_df

def calc_amplitude(df: pd.DataFrame, N: int) -> pd.Series:
    '''计算振幅'''
    return (df['high'].expanding(N).max() - df['high'].expanding(N).min()) / df['close']

def dict2frame(dic: dict) -> pd.DataFrame:
    
    '''将data的dict格式转为df'''

    tmp_v = [v.stack() for v in dic.values()]
    name = [k.upper() for k in dic.keys()]
    df = pd.concat(tmp_v, axis=1)
    df.columns = name
    df.index.names = ['time', 'code']
    return df

# ===== CELL =====

# 大约3min23s
other_factor = query_other_factor('000906.XSHG','2010-01-01','2020-11-30')

# ===== CELL =====

# CGO因子与传统因子拼接
cgo_ser = cgo_df.stack()
cgo_ser.name = 'CGO'
data = pd.concat([cgo_ser,other_factor],axis=1)
data = data.loc[datetime.datetime(2010,1,1):]
data.index.names = ['time','code']

# ===== CELL =====

(data.corr().iloc[1:, 0].to_frame('相关系数').style.format('{:.2%}')
 .bar(align='zero', color=['CadetBlue', 'Crimson']))

# ===== CELL =====

# 获取年末季末时点

def GetTradePeriod(start_date: str, end_date: str, freq: str = 'ME') -> list:
    '''
    start_date/end_date:str YYYY-MM-DD
    freq:M月，Q季,Y年 默认ME E代表期末 S代表期初
    ================
    return  list[datetime.date]
    '''
    days = pd.Index(pd.to_datetime(get_trade_days(start_date, end_date)))
    idx_df = days.to_frame()

    if freq[-1] == 'E':
        day_range = idx_df.resample(freq[0]).last()
    else:
        day_range = idx_df.resample(freq[0]).first()

    day_range = day_range[0].dt.date

    return day_range.dropna().values.tolist()

# ===== CELL =====

# 获取周度节点
weekend = GetTradePeriod('2010-01-01','2020-11-30','W')

# 每周度收益 手动加入最后一期
next_return = pricing.loc[weekend + [datetime.date(2020, 12, 7)]].pct_change().shift(-1)
# 获取周度因子值
weekend_data = data.loc[weekend].copy()
weekend_data['NEXT_RET'] = next_return.stack() # 加入

# 查看数据结构
weekend_data.head()

# ===== CELL =====

# CGO根据0值区分大小分组
CGO_LOW = weekend_data.query('CGO <= 0')
CGO_HIGH = weekend_data.query('CGO > 0')

# ===== CELL =====

# 计算rank_IC
def calc_ic(df:pd.DataFrame)->pd.Series:
    '''rank ic'''
    SELECT_COL = [col for col in df.columns if col not in [
        'CGO', 'GROUP', 'NEXT_RET']]

    ic_df = pd.concat((df.groupby(level='time').apply(
        lambda x: stats.spearmanr(x['NEXT_RET'], x[i])[0]) for i in SELECT_COL), axis=1)

    ic_df.columns = SELECT_COL
    
    return ic_df

# ===== CELL =====

ic_df = pd.concat((calc_ic(df).mean()
                   for df in (CGO_LOW, CGO_HIGH, weekend_data)), axis=1)

ic_df.columns = ['CGO_低', 'CGO_高', 'CGO_全']

(ic_df.T.style.format('{:.2%}')
              .set_caption('中证800成分股CGO分层后各个因子的周度IC平均')
              .highlight_max(color='Crimson', axis=1)
              .highlight_min(color='CadetBlue', axis=1))

# ===== CELL =====

((ic_df['CGO_高'] - ic_df['CGO_低']).to_frame('IC_spread').style.format('{:.2%}')
 .set_caption('通过IC_spread选取风险因子')
 .highlight_max(color='Crimson')
 .highlight_min(color='CadetBlue'))

# ===== CELL =====

# 获取回测信息
def get_spread_ret(group: pd.DataFrame, h_factor: str, l_factor: str) -> pd.Series:
    '''
    group数据结构为:
        ----------------------------------------------------------------
             time  |    code   |A_FACTOR|B_FACTOR|...|E_FACTOR| NEXT_RET
        -----------|-----------|----------------------------------------
         2014-01-02|000001.XSHE| -0.038 | 0.192  |...| 1.056  | -0.045  
                   |-----------|----------------------------------------
                   |000002.XSHE| -0.122 | 0.153  |...| 0.610  | -0.064  
        ----------------------------------------------------------------
    h/l_factor:高/低分组选择的因子
    -----------------
        return 对于的收益序列
    '''

    # 获取高低分组
    CGO_H = group.query('CGO > 0')
    CGO_L = group.query('CGO <= 0')

    # 组合内分五档
    threshold_h = CGO_H[h_factor].quantile(0.25)
    threshold_l = CGO_L[l_factor].quantile(0.25)

    # 获取持仓
    H_GROUP = CGO_H.query(f'{h_factor} < @threshold_h')['NEXT_RET']
    L_GROUP = CGO_L.query(f'{l_factor} < @threshold_l')['NEXT_RET']

    # 等权持仓
    ret = np.mean(
        np.hstack((H_GROUP.values, L_GROUP.values)))

    h_ret = H_GROUP.mean()
    l_ret = L_GROUP.mean()
    
    # 数量统计
    h_count = len(H_GROUP)
    l_count = len(L_GROUP)

    return pd.Series([ret, h_ret, l_ret, h_count, l_count], index=['RET', 'H_RET', 'L_RET', 'H_NUM', 'L_NUM'])


# ===== CELL =====

# 获取回测信息
spread_df = weekend_data.groupby(level='time').apply(
    get_spread_ret, h_factor='ZF_15', l_factor='pe_ratio')

# 获取策略净值
cum = np.exp(np.log1p(spread_df[['RET', 'H_RET', 'L_RET']]).cumsum())
# 获取基准
benchmark = get_price('000300.XSHG', '2010-01-01',
                      '2020-11-30', fields='close')

benchmark = (benchmark['close'] / benchmark['close'][0]).reindex(cum.index)

# 画图
plt.rcParams['font.family'] = 'serif'
cum.fillna(method='ffill').plot.line(y=['RET', 'H_RET', 'L_RET'], figsize=(
    18, 6), title='CGO因子回测', color=['Crimson', 'Salmon', 'Tomato'])
benchmark.plot(color='DarkGray', label='benchmark')
plt.legend()

spread_df.plot.line(y=['H_NUM', 'L_NUM'], title='高低CGO分组数量情况', figsize=(18, 4))
plt.axhline(0, ls='--', color='black', lw=0.7)

# ===== CELL =====

'''因子获取'''

def get_test_data(symbol: str, start: str, end: str) -> pd.DataFrame:

    periods = GetTradePeriod(start, end, 'ME')  # 获取调仓片段

    df_list = []

    for watch_date in tqdm_notebook(periods, desc='因子获取中'):

        stocks = get_stockpool(
            symbol, watch_date.strftime('%Y-%m-%d'))  # 成分股获取

        factor_list = [CGO_20(), SUE0(), R20(), R375(), TURNOVER(), LOGCAP()]
        
        df_list.append(dict2frame(calc_factors(
            stocks, factors=factor_list, start_date=watch_date, end_date=watch_date)))

    factor_df = pd.concat(df_list, axis=1)
    factor_df = factor_df.stack()
    factor_df.index.name = ['time', 'code']

    return factor_df.unstack()

'''因子构建'''

# SUE因子构建
class SUE0(Factor):

    '''含漂移项'''

    name = 'SUE0'
    max_window = 1

    global fields
    # 获取T至T-7 共计8期
    fields = [f'net_profit_{i}' if i != 0 else 'net_profit' for i in range(8)]

    dependencies = fields

    def calc(self, data):

        # 数据结构为 columns为 net_profit至net_profit_7
        df = pd.concat([v.T for v in data.values()], axis=1)
        df.columns = fields
        df.fillna(0, inplace=True)

        # 漂移项可以根据过去两年盈利同比变化Q{i,t} - Q{i,t-4}的均值估计
        # 数据结构为array
        tmp = df.iloc[:, 1:-4].values - df.iloc[:, 5:].values

        C = np.mean(tmp, axis=1)  # 漂移项 array

        epsilon = np.std(tmp, axis=1)  # 残差项epsilon array

        Q = df.iloc[:, 4] + C # 带漂移项的季节性随机游走模型

        return (df.iloc[:, 0] - Q) / epsilon
    
class CGO_20(Factor):
    
    '''计算窗口为60的CGO因子'''
    
    import warnings
    warnings.filterwarnings("ignore")
    
    name = 'CGO_20'
    max_window = 20
    
    dependencies = ['close','turnover_ratio','money','volume']
    
    def calc(self,data):
        
        avg = data['money'] / data['volume'] # 数据索引不是日期!!!
        
        turnover = data['turnover_ratio'] / 100
        turnover_weighs = turnover.apply(calc_turnover_weight)

        # 归一化
        scale_weights = turnover_weighs.div(turnover_weighs.sum())
        avg.index = scale_weights.index # 设置索引
        rp_price = (scale_weights * avg).sum()  # 参考价格
        
        
        return data['close'].iloc[-1] / rp_price - 1
    
class R20(Factor):
    
    name = 'R20'
    max_window = 20
    dependencies = ['close']
    
    def calc(self,data):
        
        return data['close'].iloc[-1] / data['close'].iloc[0] - 1

class R375(Factor):
    
    name = 'R375'
    max_window = 375
    dependencies = ['close']
    
    def calc(self,data):
        
        return data['close'].iloc[-1] / data['close'].iloc[0] - 1
    
class TURNOVER(Factor):
    
    name = 'TURNOVER'
    max_window = 20
    dependencies = ['turnover_ratio']
    
    def calc(self,data):
        
        return data['turnover_ratio'].mean()
    
class LOGCAP(Factor):
    
    '''
    其实可以直接用get_factors获取size
    但是这样想统一用calc_factor获取
    '''
    name = 'LOGCAP'
    max_window = 1
    dependencies = ['market_cap']
    
    def calc(self,data):
       
        return np.log(data['market_cap']).iloc[-1]
    

'''获取过滤后的成分股'''

def get_stockpool(symbol:str,watch_date:str)->list:
    
    '''获取股票池'''
    
    if symbol == "A":
        stockList = get_index_stocks('000002.XSHG', date=watch_date) + get_index_stocks(
                    '399107.XSHE', date=watch_date)
    else:
        stockList = get_index_stocks('000002.XSHG', date=watch_date)
        
    stockList = del_st_stock(stockList,watch_date)
    stockList = del_iponday(stockList,watch_date,3 * 244)
    stockList = del_paused(stockList,watch_date)
    
    return stockList

def del_st_stock(securities: list, watch_date: str) -> list:
    '''过滤ST股票'''

    info_ser = get_extras('is_st', securities,
                          end_date=watch_date, df=True, count=1).iloc[0]

    return info_ser[info_ser == False].dropna().index.tolist()


def del_iponday(securities: list, watch_date: str, N: int=60) -> list:
    '''返回上市大于N日的股票'''
    return list(filter(lambda x: (parse(watch_date).date() - get_security_info(x, date=watch_date).start_date).days > N, securities))


def del_paused(securities: list, watch_date: str, N: int=21) -> list:
    '''返回N日内未停牌的股票'''
    pausd_df = get_price(securities, end_date=watch_date,
                         count=N, fields='paused', panel=False)
    cond = pausd_df.groupby('code')['paused'].sum()

    return cond[cond == 0].dropna().index.tolist()


# ===== CELL =====

# Fama_MacBeth
class fama_macbeth_regr(object):

    '''
    结果为经Newey West调整的后的结果
    代码参考https://bashtage.github.io/linearmodels/panel/models.html#linearmodels.panel.model.FamaMacBeth
    经测试与limearmodel结果一致
    ================
    df数据结构 MultiIndex level0-time,level1-code 必须要有NEXT_RET,
        ----------------------------------------------------------------
             time  |    code   |A_FACTOR|B_FACTOR|...|E_FACTOR| NEXT_RET
        -----------|-----------|----------------------------------------
         2014-01-02|000001.XSHE| -0.038 | 0.192  |...| 1.056  | -0.045  
                   |-----------|----------------------------------------
                   |000002.XSHE| -0.122 | 0.153  |...| 0.610  | -0.064  
        ----------------------------------------------------------------
    ===============
    仅做简单的缺失值处理后就进行回归
    '''

    def __init__(self, df: pd.DataFrame) -> None:

        # 简单的去除缺失值
        col = [i for i in df.columns if i != "NEXT_RET"]
        X = df[col]
        X = X.replace([-np.inf, np.inf], np.nan)
        X = X.fillna(0)
        X['CONST'] = 1  # 加入const

        self.X = X[['CONST'] + col]
        self.y = df['NEXT_RET'].fillna(0)

    def fit(self) -> None:

        X = self.X
        y = self.y

        xy = pd.concat([X, y], axis=1)
        self.all_df = xy
        # 回归项
        self.all_params = xy.groupby(level='time').apply(self.regr_ols)
        self._params = self.all_params.mean(0).values[:, None]  # beta项

    @property
    def get_paramser(self) -> pd.Series:

        return self.all_params.mean()
    
    @property
    def pvalues(self) -> pd.Series:

        from scipy import stats
        abs_tstats = np.abs(self.get_tstats)

        return pd.Series(2 * (1 - stats.t.cdf(abs_tstats, self.get_resids.shape[0])), 
                         index=abs_tstats.index, 
                         name='pvalues')
    
    @property
    def get_tstats(self) -> pd.Series:
        '''获取T值'''
        return self.get_paramser / self.get_stderr

    @property
    def get_stderr(self) -> pd.Series:

        # std.error
        e = self.all_params - self.all_params.mean(axis=0).T
        e = e[np.all(np.isfinite(e), 1)]
        nobs = e.shape[0]

        T = len(self.y)
        # Bartlett,NeweyWest取法
        bw = int(np.ceil(4 * (T / 100) ** (2 / 9)))  # L项

        w = self.kernel_weight_bartlett(bw)
        # Newey West
        cov = self.cov_kernel(e.values, w)

        std_error = np.sqrt(np.diag(cov / (nobs - 1)))  # 标准误

        return pd.Series(std_error, index=self.all_params.columns)

    @property
    def get_resids(self) -> pd.Series:
        '''残差'''
        ser = self.y - (self.X @ self._params).iloc[:, 0]
        ser.name = 'resids'
        return ser

    @property
    def get_rsquare(self) -> float:

        y = self.y

        weps = self.get_resids  # y - (X @ params).iloc[:, 0]

        residual_ss = float(weps.T @ weps)
        w = np.ones(y.shape[0])
        # 在有const时e的计算 否在e=y
        e = y - (w * y).sum() / w.sum()
        total_ss = float(w.T @ (e ** 2))
        # R方
        r2 = 1 - residual_ss / total_ss

        return r2

    @staticmethod
    def regr_ols(df: pd.DataFrame) -> pd.Series:
        '''OLS回归'''
        select_col = [col for col in df.columns if col != "NEXT_RET"]
        exog = df[select_col]
        dep = df['NEXT_RET']

        if exog.shape[0] < exog.shape[1] or np.linalg.matrix_rank(exog) != exog.shape[1]:
            return pd.Series([np.nan] * len(exog.columns), index=exog.columns)

        params = np.linalg.lstsq(exog, dep, rcond=None)[0]

        return pd.Series(params, index=select_col)

    @staticmethod
    def cov_kernel(z: np.array, w: np.array) -> np.array:

        k = len(w)
        n = z.shape[0]
        if k > n:
            raise ValueError(
                "Length of w ({0}) is larger than the number "
                "of elements in z ({1})".format(k, n)
            )
        s = z.T @ z
        for i in range(1, len(w)):
            op = z[i:].T @ z[:-i]
            s += w[i] * (op + op.T)

        s /= n
        return s

    @staticmethod
    def kernel_weight_bartlett(bw: float) -> np.array:

        return 1 - np.arange(int(bw) + 1) / (int(bw) + 1)

# ===== CELL =====

# 因子获取
monthly_data = get_test_data('000906.XSHG','2010-01-01','2020-11-30')
# 查询结构
monthly_data.head()

# ===== CELL =====

# 因子储存
monthly_data.to_csv('../../Data/monthly_data.csv')

# ===== CELL =====

# 因子读取
monthly_data = pd.read_csv('../../Data/monthly_data.csv',index_col=[0,1],parse_dates=[0])

# ===== CELL =====

# 获取周度节点
monthly = GetTradePeriod('2010-01-01','2020-11-30','ME')

# 每月度收益 手动加入最后一期
next_return = pricing.loc[monthly + [datetime.date(2020, 12, 7)]].pct_change().shift(-1)
next_return.index = pd.to_datetime(next_return.index)
monthly_data['NEXT_RET'] = next_return.stack() # 加入

# ===== CELL =====

# Fama-MacBeth回归分析
test_col = (
            ['CGO_20', 'R20', 'R375'],
            ['CGO_20', 'R20', 'R375', 'TURNOVER'],
            ['CGO_20', 'R20', 'R375', 'TURNOVER','LOGCAP'])

df_list = [] # 储存beta值
r2_list = [] # 储存R方
p_list = [] # 储存P-value

for i in test_col:
    model = fama_macbeth_regr(monthly_data[i + ['NEXT_RET']])
    model.fit()
    p_list.append(model.pvalues.reindex(['CGO_20', 'R20', 'R375', 'TURNOVER','LOGCAP']))
    df_list.append(model.get_paramser.reindex(['CGO_20', 'R20', 'R375', 'TURNOVER','LOGCAP']))
    r2_list.append(model.get_rsquare)
    
    
table = pd.concat(df_list,axis=1).iloc[1:]
table.columns = range(1,4)
table.loc['R_square'] = r2_list
(table.style.format('{:.4f}')
            .set_caption('CGO的Fama–MacBeth回归分析结果'))

# ===== CELL =====

# P值
p_df = pd.concat(p_list,axis=1).iloc[1:]
p_df.columns = range(1,df.shape[1]+1)

(p_df.style.format('{:.3f}')
         .set_caption('CGO的Fama–MacBeth回归的P值')
         .applymap(lambda x:'font-weight:bold' if x >0.05 else ''))

# ===== CELL =====

# SUE因子
sue_factor = test_data[['SUE0','NEXT_RET']].copy()
sue_factor.columns = ['factor',1]
sue_factor['factor_quantile'] = pd.qcut(sue_factor['factor'],5,labels=range(1,6))
sue_factor.index.names = ['date','asset']

mean_quant_ret, std_quantile = al.performance.mean_return_by_quantile(
        sue_factor,
        by_group=False, 
        demeaned=True,  
        group_adjust=False, # 是否行业中性化
    )

plt.rcParams['font.family'] = 'serif'
mean_quant_ret.plot.bar(title='SUE因子收益：分组平均超额收益（收益中心化处理）')

# ===== CELL =====

# 计算组合多空净值
def get_speard_portfolio(data: pd.DataFrame,l_col:str,s_col:str, long: str, short: str) -> pd.DataFrame:
    '''
    data:MultiIndex level0-time,level1-code
    '''
    df = data.copy()

    CGO_G = df.groupby(level='time')[l_col].transform(
        lambda x: pd.qcut(x, 5, labels=['G%s' % i for i in range(1, 6)]))
        
    SUE_G = df.groupby(level='time')[s_col].transform(
        lambda x: pd.qcut(x, 5, labels=['S%s' % i for i in range(1, 6)]))

    df['GROUP'] = SUE_G.astype(str) + CGO_G.astype(str)
    group_df = df.query('GROUP == @long or GROUP == @short')

    returns = group_df.groupby([pd.Grouper(level='time'), pd.Grouper(key='GROUP')])[
        'NEXT_RET'].mean()

    returns = returns.unstack()
    returns = returns.fillna(0)
    returns['excess_returns'] = returns[long] - returns[short]
    returns.rename(columns={long:'long',short:'short'},inplace=True)
    return returns

# ===== CELL =====

# 计算分组收益
A_speard_df = get_speard_portfolio(monthly_data,'CGO_20','SUE0','S5G5','S1G1')
B_speard_df = get_speard_portfolio(monthly_data,'CGO_20','SUE0','S1G5','S5G1')

# ===== CELL =====

# 画图
plt.rcParams['font.family'] = 'serif'
plt.figure(figsize=(14, 4))
plt.subplot(121)
plt.title('A组净值')
plt.plot(np.exp(np.log1p(A_speard_df).cumsum()))
plt.legend(['long', 'short', 'excess_returns'])
plt.subplot(122)
plt.title('B组净值')
plt.plot(np.exp(np.log1p(B_speard_df).cumsum()))
plt.legend(['long', 'short', 'excess_returns'])

# ===== CELL =====

# A组
a_return_avg = A_speard_df.mean()
a_return_avg.name = 'mean'
a_table = pd.concat([A_speard_df.apply(lambda x:pd.Series(stats.ttest_1samp(x, 0), index=['tstats', 'pvalues'])).T,
                     a_return_avg], axis=1)
                     
a_table.index = ['A_' + i for i in a_table.index.tolist()]
# B组
b_return_avg = B_speard_df.mean()
b_return_avg.name = 'mean'
b_table = pd.concat([B_speard_df.apply(lambda x:pd.Series(stats.ttest_1samp(x, 0), index=['tstats', 'pvalues'])).T,
                     b_return_avg], axis=1)

b_table.index = ['B_' + i for i in a_table.index.tolist()]

table = pd.concat([a_table, b_table]).T

# 可视化显示
tmp_ser = table.apply(lambda x: '{0[0]:.3f}_{0[1]:.3f}_{0[2]:.4f}'.format(x))
(tmp_ser.to_frame(1).T.style.format(lambda x: '{0[2]} ({0[0]})'.format(x.split('_')))
 .applymap(lambda x: 'font-weight:bold' if float(x.split('_')[1]) > 0.05 else '')
 .set_caption('CGO指标的测试结果'))

# ===== CELL =====

# 获取多空净值
speard_df = pd.concat([A_speard_df['excess_returns'],
                      B_speard_df['excess_returns']],axis=1)

speard_df.columns = ['A_L/S','B_L/S']

# 计算累计净值
cum = np.exp(np.log1p(speard_df).cumsum())

# ===== CELL =====

# 获取基准
benchmark = get_price('000300.XSHG', '2010-01-01',
                      '2020-11-30', fields='close')

benchmark = benchmark.reindex(cum.index)
benchmark = (benchmark['close'] / benchmark['close'][0])

# 画图
cum.plot(figsize=(18,5),title='多空收益净值曲线',color=['Crimson', 'Salmon'])
benchmark.plot(label='benchmark',color='DarkGray')
plt.axhline(1,color='black',lw=0.65,ls='--')
plt.legend();

# ===== CELL =====

# 计算RCGO
def calc_rcgo(df:pd.DataFrame,target:str,edog:list)->pd.Series:
    '''计算RCGO'''
    def calc_resid(df:pd.DataFrame)->pd.Series:
        beta = np.linalg.lstsq(df[edog],df[target])[0]
        return df[target] - df[edog] @ beta
    
    ser = df.groupby(level='time',group_keys=False).apply(calc_resid)
    ser.name = 'RCGO'
    return ser

# ===== CELL =====

# 计算RCGO
monthly_data['RCGO'] = calc_rcgo(monthly_data,'CGO_20',['R375','LOGCAP'])

# ===== CELL =====

# 计算组合收益
A_speard_df = get_speard_portfolio(monthly_data,'RCGO','SUE0','S5G5','S1G1')
B_speard_df = get_speard_portfolio(monthly_data,'RCGO','SUE0','S1G5','S5G1')

# ===== CELL =====

# 画图
plt.rcParams['font.family'] = 'serif'
plt.figure(figsize=(14, 4))
plt.subplot(121)
plt.title('A组净值')
plt.plot(np.exp(np.log1p(A_speard_df).cumsum()))
plt.legend(['long', 'short', 'excess_returns'])
plt.subplot(122)
plt.title('B组净值')
plt.plot(np.exp(np.log1p(B_speard_df).cumsum()))
plt.legend(['long', 'short', 'excess_returns'])

# ===== CELL =====

# A组
a_return_avg = A_speard_df.mean()
a_return_avg.name = 'mean'
a_table = pd.concat([A_speard_df.apply(lambda x:pd.Series(stats.ttest_1samp(x, 0), index=['tstats', 'pvalues'])).T,
                     a_return_avg], axis=1)

a_table.index = ['A_' + i for i in a_table.index.tolist()]
# B组
b_return_avg = B_speard_df.mean()
b_return_avg.name = 'mean'
b_table = pd.concat([B_speard_df.apply(lambda x:pd.Series(stats.ttest_1samp(x, 0), index=['tstats', 'pvalues'])).T,
                     b_return_avg], axis=1)

b_table.index = ['B_' + i for i in a_table.index.tolist()]

table = pd.concat([a_table, b_table]).T

# 可视化显示
tmp_ser = table.apply(lambda x: '{0[0]:.3f}_{0[1]:.3f}_{0[2]:.4f}'.format(x))
(tmp_ser.to_frame(1).T.style.format(lambda x: '{0[2]} ({0[0]})'.format(x.split('_')))
 .applymap(lambda x: 'font-weight:bold' if float(x.split('_')[1]) > 0.05 else '')
 .set_caption('RCGO指标的测试结果'))


# ===== CELL =====

# 获取多空净值
speard_df = pd.concat([A_speard_df['excess_returns'],
                      B_speard_df['excess_returns']],axis=1)

speard_df.columns = ['A_L/S','B_L/S']

# 计算累计净值
cum = np.exp(np.log1p(speard_df).cumsum())

# ===== CELL =====

# 获取基准
benchmark = get_price('000300.XSHG', '2010-01-01',
                      '2020-11-30', fields='close')

benchmark = benchmark.reindex(cum.index)
benchmark = (benchmark['close'] / benchmark['close'][0])

# 画图
cum.plot(figsize=(18,5),title='多空收益净值曲线',color=['Crimson', 'Salmon'])
benchmark.plot(label='benchmark',color='DarkGray')
plt.axhline(1,color='black',lw=0.65,ls='--')
plt.legend();