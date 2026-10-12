import pandas as pd
import numpy as np
import alphalens # 因子分析

import pickle

import scipy.stats as st
import statsmodels.api as sm

from tqdm import *
from jqdata import *

# 多进程
from CalFunc import *  # <===自定义的需要多进程计算的函数
from multiprocessing import Pool 

pd.set_option('display.max_r',10)

m_cpu= 2 # cpu几核就设置几

# 日期
import calendar # 日历
from dateutil.parser import parse

# 画图
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import matplotlib.ticker as ticker 
import matplotlib.gridspec as mg # 不规则子图
pd.plotting.register_matplotlib_converters()

# 设置字体 用来正常显示中文标签
mpl.rcParams['font.sans-serif'] = ['SimHei']
#plt.rcParams['font.family']='serif' # pd.plot中文
# 用来正常显示负号
mpl.rcParams['axes.unicode_minus'] = False
# 图表主题
plt.style.use('seaborn')

# ===== CELL =====

# 设置时间范围
start_date, end_date = '2014-01-01', '2020-04-01'
begin_date = get_trade_days(end_date=start_date,count=30)[0]
# 获取截面成分股
security_list = get_index_stocks('000002.XSHG') + get_index_stocks(
    '399107.XSHE')

# ===== CELL =====

# 获取日内数据 计算用于apb_1d
def GetData(security_list: list, start_date: str, end_date: str):

    days = get_trade_days(start_date, end_date)
    temp = []

    for trade in tqdm(days, desc='下载中'):

        price_df = get_price(
            security_list,
            start_date=str(trade) + ' 9:30:00',
            end_date=str(trade) + ' 15:00:00',
            frequency='30m',
            fields=['money', 'volume'],
            panel=False)

        temp.append(get_dailyapb(price_df))

    df = pd.concat(temp)

    return df


# 计算apb
def get_dailyapb(df: pd.DataFrame,) -> pd.Series:

    def cal_apb(df):

        if not df.empty and len(df) >= 5:
            # 未取对数 对数计算较为简单可以后续计算
            return np.mean(df['vwap']) / np.average(
                df['vwap'], weights=df['volume'])
        else:
            return np.nan

    df = df.query('volume!=0') # 过滤成交量为0的
    df['vwap'] = df['money'] / df['volume']
    df = df.dropna() # 过滤na
    apb = df.groupby(['code', pd.Grouper(key='time', freq='D')]).apply(cal_apb)
    apb.name = 'APB'
    return apb


# 计算apb_1d
def Cal_APBN(ser: pd.Series):
    if len(ser) >= 15:
        return np.mean(ser)
    else:
        return np.nan

# ===== CELL =====

# 11h24m24s
# 获取分钟数据
daily_df = GetData(security_list,begin_date,end_date)
daily_df = daily_df.to_frame()
daily_df.columns = ['apb'] 

# 数据储存
daily_df.to_csv('../../Data/apb_1d.csv')

# ===== CELL =====

def OffsetDate(end_date: str, count: int) -> datetime.date:
    '''
    end_date:为基准日期
    count:为正则后推，负为前推
    -----------
    return datetime.date
    '''

    trade_date = get_trade_days(end_date=end_date, count=1)[0]
    if count > 0:
        # 将end_date转为交易日

        trade_cal = get_all_trade_days().tolist()

        trade_idx = trade_cal.index(trade_date)

        return trade_cal[trade_idx + count]

    elif count < 0:

        return get_trade_days(end_date=trade_date, count=abs(count))[0]

    else:

        raise ValueError('别闹！')


# 多进程
def apply_parl(groupeds, func):

    with Pool(m_cpu) as p:

        res_list = p.map(func, [g for _, g in groupeds])

    return pd.concat(res_list)


def get_group_ids(df, key):

    codes = df.index.get_level_values(key).drop_duplicates()

    mapping = pd.Series(np.arange(0, len(codes)), index=codes)

    res = df.index.get_level_values(key).map(mapping) % (m_cpu * 10)
    
    return res.values

# ===== CELL =====

# 滞后一期方便计算next_ret
offset_date = OffsetDate(end_date,1)

# 数据获取
data = get_price(
    security_list,
    begin_date,
    offset_date,
    fields=['close', 'volume','paused'],
    panel=False)
    
data.set_index(['code', 'time'], inplace=True)
data = data[data['volume']!=0]

# ===== CELL =====

# 多进程计算

start = time.time()
print('vwap开始计算...')
ids = get_group_ids(data, key='code')
vwap = apply_parl(data.groupby(ids), rolling_5vwap)

end = time.time()
print("多进程(多个code处理)计算vwap :" + str(end - start))

data = data.reindex(vwap.index)
data['vwap'] = vwap

print('开始计算apb...')

start = time.time()

ids = get_group_ids(data, key='code')
apb = apply_parl(data.groupby(ids), rolling_5apb)

end = time.time()

print("多进程(多个code处理)计算apb :" + str(end - start))

data = data.reindex(apb.index)
data['apb'] = apb

print('计算完毕...')

# ===== CELL =====

# 存储
data.to_csv('../../Data/apb_5d.csv')

# ===== CELL =====

start = time.time()
print('vwap开始计算...')
ids = get_group_ids(data, key='code')
vwap = apply_parl(data.groupby(ids), rolling_30vwap)

end = time.time()
print("多进程(多个code处理)计算vwap :" + str(end - start))

data = data.reindex(vwap.index)
data['vwap'] = vwap

print('开始计算apb...')

start = time.time()

ids = get_group_ids(data, key='code')
apb = apply_parl(data.groupby(ids), rolling_30apb)

end = time.time()

print("多进程(多个code处理)计算apb :" + str(end - start))

data = data.reindex(apb.index)
data['apb'] = apb

print('计算完毕...')

# ===== CELL =====

# 存储
data.to_csv('../../Data/apb_30d.csv')

# ===== CELL =====

def load_factor(path):

    df = pd.read_csv(path, index_col=[0, 1], parse_dates=True)
    log_apb = np.log(df['apb'])
    apb_30d = log_apb.groupby(level='code').transform(
        lambda x: x.rolling(30).mean())

    factor_ser = apb_30d.swaplevel().sort_index()
    factor_ser.index.names = ['date', 'assent']

    return factor_ser


def GetFactorRet(xdf: pd.DataFrame):
    
    factor_returns = pd.pivot_table(
        xdf.reset_index(), index='date', columns='factor_quantile', values=1)
    
    factor_returns['long/short'] = factor_returns[5] - factor_returns[1]
    
    return factor_returns

def factor_summary(factor_list: list) -> pd.DataFrame:
    
    t_list = []
    f_list = []
    for f_df in factor_list:
        
        t_list.append(pd.Series(GetTReport(f_df)))
        f_list.append(pd.Series(GetICReport(f_df)))
        
    return pd.concat(t_list,axis=1),pd.concat(f_list,axis=1)
        
def GetTReport(factor_df: pd.DataFrame):

    rept = {}
    tvalue_ser = factor_df.groupby(level='date').apply(
        lambda x: Get_Tvalue(x['factor'], x[1]))
    abs_tvalue_ser = abs(tvalue_ser)

    rept['t均值'] = tvalue_ser.mean()
    rept['|t|均值'] = abs_tvalue_ser.mean()
    rept['|t|>2占比'] = np.sum(
        np.where(abs_tvalue_ser > 2, 1, 0)) / len(abs_tvalue_ser)
    rept['因子收益均值'] = np.mean(factor_df[1])

    return rept


def GetICReport(factor_df: pd.DataFrame):

    rept = {}
    rank_IC = factor_df.groupby(level='date').apply(
        lambda x: st.pearsonr(x[1].rank(), x['factor'].rank())[0])

    rept['ic均值'] = rank_IC.mean()
    rept['ic标准差'] = rank_IC.std()
    rept['IC_IR'] = rank_IC.mean()/rank_IC.std()
    rept['IC大于0的比率'] = len(rank_IC[rank_IC > 0])/len(rank_IC)
    
    return rept

# t_value


def Get_Tvalue(factor, returns):
    x = sm.add_constant(factor)
    mod = sm.OLS(returns, x)
    res = mod.fit()

    return res.tvalues[0]



def GetRisk(factor_df:pd.DataFrame):
    
    f_returns = GetFactorRet(factor_df)
    f_cum = (1+f_returns).cumprod()
    
    # 计算年化收益率
    annual_ret = (f_cum.iloc[-1] - 1)**(250. / float(len(f_cum))) - 1
    
    # 计算累计收益率
    cum_ret_rate = f_cum.iloc[-1] - 1
    
    mdd = GetMaxdrawdown(f_cum)
                                        
    # 
    volatility = np.std(f_returns)*np.sqrt(250)
    shape = (annual_ret-0.04) / volatility
    
    df = pd.concat([annual_ret,cum_ret_rate,mdd,shape,volatility],axis=1)
    df.columns = ['年化收益','累计收益','最大回撤','夏普','波动率']
    return df

def GetMaxdrawdown(df:pd.DataFrame):
    
    md = {}
    for label,col in df.items():
        
        # 最大回撤
        max_nv = np.maximum.accumulate(col)
        mdd = -np.min(col/max_nv-1)
        
        md[label] = mdd
        
    return pd.Series(md)

# ===== CELL =====

# 读取因子数据
factor_1d = load_factor('../../Data/apb_1d.csv')
factor_5d = load_factor('../../Data/apb_5d.csv')
factor_30d = load_factor('../../Data/apb_30d.csv')

# 因子回测期间
factor_1d = factor_1d.loc['2014-01-02':'2019-12-31']
factor_5d = factor_5d.loc['2014-01-02':'2019-12-31']
factor_30d = factor_30d.loc['2014-02-18':'2019-12-31']

print('apb_1d')
alphalens.utils.print_table(factor_1d.tail())

print('apb_5d')
alphalens.utils.print_table(factor_5d.tail())

print('apb_30d')
alphalens.utils.print_table(factor_30d.tail())

# ===== CELL =====

# 获取收盘数据
# 数据前推30日后推15日
pricing = get_price(security_list,begin_date,OffsetDate(end_date,15),fields='close',panel=False)
pricing = pd.pivot_table(pricing, index = 'time', columns = 'code', values = 'close').shift(-1)

# 查看数据结构
pricing.head()

# ===== CELL =====

# Ingest and format data
apb_30d = alphalens.utils.get_clean_factor_and_forward_returns(
    factor_30d, pricing, quantiles=5, periods=(1,))

apb_5d = alphalens.utils.get_clean_factor_and_forward_returns(
    factor_5d, pricing, quantiles=5, periods=(1,))

apb_1d = alphalens.utils.get_clean_factor_and_forward_returns(
    factor_1d, pricing, quantiles=5, periods=(1,))


# ===== CELL =====

# 结果
t_rept, f_rept = factor_summary([apb_1d, apb_5d, apb_30d])
t_rept.columns = ['apb_1d', 'apb_5d', 'apb_30d']
f_rept.columns = ['apb_1d', 'apb_5d', 'apb_30d']

print('回归法结果分析 ')
alphalens.utils.print_table(t_rept.T)

print('IC结果分析 ')
alphalens.utils.print_table(f_rept.T)

print('因子收益分析')
print('-'*40)
print('apb_1d')
alphalens.utils.print_table(GetRisk(apb_1d).T)

print('apb_5d')
alphalens.utils.print_table(GetRisk(apb_5d).T)

print('apb_30d')
alphalens.utils.print_table(GetRisk(apb_30d).T)

# 图表
(1+apb_1d.pipe(GetFactorRet)).cumprod().plot(figsize=(18, 8), title='apb_1d')
(1+apb_5d.pipe(GetFactorRet)).cumprod().plot(figsize=(18, 8), title='apb_5d')
(1+apb_30d.pipe(GetFactorRet)).cumprod().plot(figsize=(18, 8), title='apb_30d')