# 引入库
import sys
sys.path.append('../../..')

from BuildPeriodDate import (GetTradePeriod,tdaysoffset)

import numpy as np
import pandas as pd
import empyrical as ep
import alphalens as al
import scipy.stats as st
import alphalens.performance as perf
from alphalens.utils import print_table

from jqdata import *
from jqfactor import (Factor,calc_factors)

import itertools
from typing import (Tuple,List,Dict,Union,Callable)
from functools import reduce
from tqdm import tqdm_notebook
from dateutil.parser import parse


import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams['font.family']='serif' # pd.plot中文
# 用来正常显示负号
mpl.rcParams['axes.unicode_minus'] = False
# 图表主题
plt.style.use('seaborn')

# ===== CELL =====

# 获取股票池
def get_stockpool(symbol: str, watch_date: str) -> list:
    '''获取股票池'''

    if symbol == 'A':
        stockList = get_index_stocks('000001.XSHG', date=watch_date) + get_index_stocks(
            '399107.XSHE', date=watch_date)
    else:
        stockList = get_index_stocks(symbol, date=watch_date)
        
    stockList = del_st_stock(stockList, watch_date)  # 过滤ST
    stockList = del_iponday(stockList, watch_date)   # 过滤上市不足60日
    stockList = del_pause(stockList,watch_date)      # 过滤当日停牌的股票
    return stockList

def del_st_stock(securities: list, watch_date: str) -> list:
    '''过滤ST股票'''

    info_ser = get_extras('is_st', securities,
                          end_date=watch_date, df=True, count=1).iloc[0]

    return info_ser[info_ser == False].dropna().index.tolist()

def del_iponday(securities: list, watch_date: str, N: int = 60) -> list:
    '''返回上市大于N日的股票'''
    dic = {x: get_security_info(x, date=watch_date).start_date for x in securities if get_security_info(
        x, date=watch_date) != None}
    return [code for code, start_date in dic.items() if (parse(watch_date).date() - start_date).days > N]

def del_pause(securities:list,watch_date:str,N:int=10)->list:
    
    '''
    过滤停牌股,默认为过滤当日停牌的
    '''
    
    paused = get_price(securities,end_date=watch_date,count=N,fields='paused',panel=False)
    
    if N == 1:
        return paused.query('paused != 1')['code'].values.tolist()
    else:
        
        paused_num = paused.groupby('code')['paused'].sum()
        return paused_num[paused_num < N].index.values.tolist()

# ===== CELL =====

# 因子计算
def prepare_data(symbol: str, start: str, end: str, last_date: str, factor: Callable, grid_func_params: Dict) -> pd.DataFrame:
    '''
    symbol:股票池所属成份，A为全市场
    start,end:回测区间 频率为月度
    last_date:计算next_returns的最后一日的日期
    factor:构建的因子
    '''
    periods = GetTradePeriod(start, end, 'ME')  # 月度_周期获取
    periods = periods + [last_date]
    factor_dic = {}

    for tradeDate, nextDate in tqdm_notebook(zip(periods[:-1], periods[1:]), total=len(periods)-1, desc='因子计算'):

        # 股票池
        stock_pool = get_stockpool(symbol, tradeDate.strftime('%Y-%m-%d'))

        # 因子获取

        f = factor(stock_pool, tradeDate, 20)
        
        # 获取数据 这里仅需获取一次就能使用该数据进行N此不同参数运算
        f.get_data() 
        # 可以使用不同参数进行计算
        factor_frame = grid_func_params['func'](
            f, **grid_func_params['params'])

        # 计算下期收益率
        close = get_price(stock_pool, end_date=tradeDate, count=1,
                          fields='close', panel=False).set_index('code')

        next_close = get_price(stock_pool, end_date=nextDate,
                               count=1, fields='close', panel=False).set_index('code')

        next_returns = next_close['close'] / close['close'] - 1
        next_returns.name = 'next_ret'

        # 合并
        tmp = pd.concat((factor_frame, next_returns), axis=1)
        factor_dic[tradeDate] = tmp  # 放入容器

    factor_df = pd.concat(factor_dic)

    factor_df.index.names = ['date', 'code']

    return factor_df


def dict2frame(dic: dict) -> pd.DataFrame:
    '''将data的dict格式转为df'''

    df = pd.concat((df.iloc[-1] for df in dic.values()), axis=1)
    df.columns = list(dic.keys())

    return df

# ===== CELL =====

# 振幅因子
class AF_factor(object):

    '''
    lamb为切分变量
    group为高低分组high,low两个参数
    '''

    def __init__(self, securities: Union[str, list], watch_date: str, N: int) -> None:

        self.securities = securities

        self.watch_date = watch_date

        self._N = N

        self.max_window = N + 2

    def get_data(self):
        '''数据获取'''

        data = get_price(self.securities,
                         end_date=self.watch_date,
                         count=self.max_window,
                         fields=['close', 'high', 'low', 'paused'], panel=False)

        self.data = data.pivot(index='time', columns='code')

    def calc(self, lamb: float, group: str) -> pd.Series:
        '''因子计算'''
        self.lamb = round(lamb,2)
        self.group = group

        af_df = self._calc_af()
        af_df = af_df.iloc[-self._N:]

        cond1 = self._q_split()
        cond1 = cond1.iloc[-self._N:]

        cond2 = self._get_paused()
        cond2 = cond2.iloc[-self._N:]

        cond = cond1 * cond2

        ser = (af_df * cond).mean()
        ser.name = self.name
        return ser

    def _q_split(self) -> pd.DataFrame:
        '''分割'''
        close = self.data['close']
        
        cond = close.rank(pct=True) >= self.lamb
        
        if self.group == 'high':

            return cond

        elif self.group == 'low':

            return ~cond

        else:
            raise ValueError('group参数仅能为high,low.')

    def _calc_af(self) -> pd.DataFrame:
        '''计算振幅'''
        return self.data['high'] / self.data['low'] - 1

    def _get_paused(self) -> pd.DataFrame:
        '''一字跌停后一日标记为False'''
        close_df = self.data['close']
        high_df = self.data['high']
        low_df = self.data['low']
        paused = self.data['paused'] # 停牌
      
        # 跌停
        cond1 = (close_df / close_df.shift(1) - 1) < -0.09
        # 一字
        cond2 = (high_df == low_df)

        res = (cond1 & cond2)
        res = (paused + res).astype(bool)
        
        return (~res).shift(1)

    @property
    def name(self) -> str:

        return f'AF_{round(self.lamb,2)}_{self.group}'

# 因子参数构造
def grid_factor(factor: Callable, lamb: Union[float, list], group: Union[str, list]) -> pd.DataFrame:

    if isinstance(lamb, (float, int)):
        lamb = [lamb]

    if isinstance(group, str):

        group = [group]

    df_list = []
    for l, g in itertools.product(lamb, group):

        df_list.append(factor.calc(l, g))

    return pd.concat(df_list, axis=1)

# ===== CELL =====

# 风险指标
def risk_indicator_tear(returns:pd.DataFrame,period:str='monthly')->pd.DataFrame:
    
    tear = pd.DataFrame()
    tear['annual_return'] = ep.annual_return(returns,period)
    tear['annual_volatility'] = returns.apply(lambda x:ep.annual_volatility(x,period))
    tear['max_drawdown'] = returns.apply(lambda x:ep.max_drawdown(x))
    
    return tear

# ===== CELL =====

# 因子获取
factor_df = prepare_data('A','2010-01-01','2020-12-31','2021-01-31',
             AF_factor,{'func':grid_factor,'params':{'lamb':np.arange(0.2,1.1,0.1),'group':['high','low']}})

# ===== CELL =====

# 计算ic
def src_ic(df:pd.DataFrame)->pd.DataFrame:
    
    '''计算IC'''
    
    f = [col for col in df.columns if col != 'next_ret']
    
    _ic = df[f].apply(lambda x:st.spearmanr(x,df['next_ret'])[0])
    
    return _ic

# ===== CELL =====

# IC计算
ic_df = factor_df.fillna(0).groupby(level='date').apply(src_ic)

high_ic = ic_df.filter(regex='high')
low_ic = ic_df.filter(regex='low')

high_ic.columns = [f'%.2f%%' % (float(col[1]) * 100)
                       for col in high_ic.columns.str.split('_')]
low_ic.columns = [f'%.2f%%' % (float(col[1]) * 100)
                      for col in low_ic.columns.str.split('_')]

# 获取high组的mean ic
h_mean = high_ic.mean()
h_std = high_ic.std()
h_icir = h_mean / h_std

# 获取low组的mean ic
l_mean = low_ic.mean()
l_std = low_ic.std()
l_icir = l_mean / l_std

# ===== CELL =====

# 画图
mpl.rcParams['font.family'] = 'serif'
fig, icir_axes = plt.subplots(1, 2, figsize=(18, 6))

icir_axes[0].set_title('不同λ值下高价振幅因子V_high绩效指标')
h_icir.plot.bar(ax=icir_axes[0], label='ICIR(右轴)', color='#3A73AD')

icir_axes_tx1 = icir_axes[0].twinx()
h_mean.plot(ax=icir_axes_tx1, label='IC均值',marker='s', color='#E83A51')
plt.grid(False)

h1, l1 = icir_axes[0].get_legend_handles_labels()
h2, l2 = icir_axes_tx1.get_legend_handles_labels()
icir_axes[0].legend(h1+h2, l1+l2)


icir_axes[1].set_title('不同λ值下高价振幅因子V_low绩效指标')
l_icir.plot.bar(ax=icir_axes[1], label='ICIR(右轴)', color='#3A73AD')

icir_axes_tx2 = icir_axes[1].twinx()
l_mean.plot(ax=icir_axes_tx2, label='IC均值',marker='s', color='#E83A51')
plt.grid(False)

h1, l1 = icir_axes[1].get_legend_handles_labels()
h2, l2 = icir_axes_tx2.get_legend_handles_labels()
icir_axes[1].legend(h1+h2, l1+l2)
plt.grid(False)

# ===== CELL =====

# 获取分组
def add_group(ser: pd.Series, N: int = 5) -> pd.Series:

    name = ['G%s' % x for x in range(1, N+1)]
    return pd.qcut(ser, N, labels=name,duplicates='drop')

# 获取分组收益率
def get_algorithm_ret(factor_df: pd.DataFrame, col: str) -> pd.DataFrame:

    group_ser = factor_df.groupby(level='date')[col].apply(add_group)
    group_df = pd.concat((factor_df['next_ret'], group_ser), axis=1)

    group_ret = pd.pivot_table(group_df.reset_index(
    ), index='date', columns=col, values='next_ret')
    group_ret.columns = list(map(str, group_ret.columns))
    group_ret['excess_ret'] = group_ret['G1'] - group_ret['G5']

    return group_ret

# 获取多空收益
def get_excess_ret(factor_df: pd.DataFrame, col: list) -> pd.DataFrame:

    if isinstance(col, (str, float, int)):
        col = [col]

    df = pd.concat((get_algorithm_ret(factor_df, x)[
                   'excess_ret'] for x in col), axis=1)
    df.columns = col
    return df

# ===== CELL =====

# 获取高低分组
high_factor = factor_df.filter(regex='high').copy()
low_factor = factor_df.filter(regex='low').copy()

high_factor.columns = [f'%.2f%%' % (float(col[1]) * 100)
                       for col in high_factor.columns.str.split('_')]

low_factor.columns = [f'%.2f%%' % (float(col[1]) * 100)
                       for col in low_factor.columns.str.split('_')]

high_factor['next_ret'] = factor_df['next_ret']

# 计算多空收益
excess_ret = get_excess_ret(high_factor, list(
    map(lambda x: '%.2f%%' % x, range(20, 100, 10))))

excess_cum = ep.cum_returns(excess_ret).add(1)

# ===== CELL =====

# 画图
mpl.rcParams['font.family'] = 'serif'
fig, excess_plot = plt.subplots(1, 2)

def set_color_condition(objs, idx, color):

    objs[idx].set_color(color)
    
# 画净值
excess_cum[['20.00%', '40.00%', '60.00%', '80.00%']].plot(
    figsize=(18, 5), title='不同λ值高价振幅因子V_high多空对冲净值表现', ax=excess_plot[0])
excess_plot[0].axhline(1, color='black')

# 画分组平均收益率
lamb_2 = get_algorithm_ret(high_factor, '20.00%')
lamb_2.mean().plot.bar(title='λ=20%时高价振幅因子5分组年化收益率非单调',
                       ax=excess_plot[1], color='#3A73AD')

yaxis_format = mpl.ticker.FuncFormatter(lambda x, pos: '%.2f%%' % (x*100))
excess_plot[1].yaxis.set_major_formatter(
    mpl.ticker.FuncFormatter(yaxis_format))
bar_ct = excess_plot[1].containers[0]
idx = lamb_2.mean().index.get_loc('excess_ret')
set_color_condition(bar_ct, idx, '#C00000')


# ===== CELL =====

# 各组合多空对冲收益情况
risk_indicator_tear(excess_ret).T.style.format('{:.2%}')

# ===== CELL =====

# 画图
mpl.rcParams['font.family'] = 'serif'
fig, ic_mean_ax = plt.subplots(figsize=(12, 6))

ic_mean_ax.set_title('不同λ值下高价振幅因子和低价振幅因子 IC均值差距单调变化')
ic_mean_ax.plot(h_mean, label='V_high IC均值', marker='s', color='#4F81BD')
ic_mean_ax.plot(l_mean, label='V_low IC均值', marker='s', color='#BA2830')
ic_mean_ax.legend();

# ===== CELL =====

low_factor['next_ret'] = 0
V_factor = high_factor - low_factor

# ===== CELL =====

v_ic = V_factor.fillna(0).groupby(level='date').apply(src_ic)
v_mean = v_ic.mean()
v_std = v_ic.std()
v_icir = v_mean / h_std

# ===== CELL =====

# 画图
mpl.rcParams['font.family'] = 'serif'
fig, v_icir_ax = plt.subplots(figsize=(18, 6))

v_icir_ax.set_title('不同λ值下高价振幅因子V_high绩效指标')
h_icir.plot.bar(ax=v_icir_ax, label='ICIR(右轴)', color='#3A73AD')

icir_ax_tx1 = v_icir_ax.twinx()
v_mean.plot(ax=icir_ax_tx1, label='IC均值',marker='s', color='#E83A51')
plt.grid(False)

h1, l1 = v_icir_ax.get_legend_handles_labels()
h2, l2 = icir_ax_tx1.get_legend_handles_labels()
v_icir_ax.legend(h1+h2, l1+l2);

# ===== CELL =====

# 计算多空收益
v_excess_ret = get_excess_ret(V_factor, list(
    map(lambda x: '%.2f%%' % x, range(20, 100, 10))))

excess_cum = ep.cum_returns(v_excess_ret).add(1)

# ===== CELL =====

# 画图
mpl.rcParams['font.family'] = 'serif'
fig, excess_plot = plt.subplots(1, 2)

# 画净值
excess_cum[['20.00%', '30.00%', '40.00%', '50.00%']].plot(
    figsize=(18, 5), title='不同λ值下理想振幅因子多空对冲净值表现', ax=excess_plot[0])
excess_plot[0].axhline(1, color='black')

# 画分组平均收益率
lamb_2 = get_algorithm_ret(V_factor, '20.00%')
lamb_2.mean().plot.bar(title='λ=20%时理想振幅因子5分组年化收益率单调',
                       ax=excess_plot[1], color='#3A73AD')

yaxis_format = mpl.ticker.FuncFormatter(lambda x, pos: '%.2f%%' % (x*100))
excess_plot[1].yaxis.set_major_formatter(
    mpl.ticker.FuncFormatter(yaxis_format))
bar_ct = excess_plot[1].containers[0]
idx = lamb_2.mean().index.get_loc('excess_ret')
set_color_condition(bar_ct, idx, '#C00000')

# ===== CELL =====

# 各组合多空对冲收益情况
risk_indicator_tear(v_excess_ret).T.style.format('{:.2%}')

# ===== CELL =====

# 因子获取
factor_hs300 = prepare_data('000300.XSHG','2010-01-01','2020-12-31','2021-01-31',
             AF_factor,{'func':grid_factor,'params':{'lamb':np.arange(0.2,1.1,0.1),'group':['high','low']}})

factor_zz500 = prepare_data('000905.XSHG','2010-01-01','2020-12-31','2021-01-31',
             AF_factor,{'func':grid_factor,'params':{'lamb':np.arange(0.2,1.1,0.1),'group':['high','low']}})

# ===== CELL =====

# 获取V_factor
def get_v_factor(factor_df:pd.DataFrame)->pd.DataFrame:
    
    # 获取高低分组
    high_factor = factor_df.filter(regex='high')
    low_factor = factor_df.filter(regex='low')
    
    # 重命名
    high_factor.columns = [f'%.2f%%' % (float(col[1]) * 100)
                       for col in high_factor.columns.str.split('_')]

    low_factor.columns = [f'%.2f%%' % (float(col[1]) * 100)
                           for col in low_factor.columns.str.split('_')]
    
    v_factor = high_factor - low_factor
    v_factor['next_ret'] = factor_df['next_ret']
    
    return v_factor

# ===== CELL =====

# 获取v_factor
hs300_v = get_v_factor(factor_hs300)
zz500_v = get_v_factor(factor_zz500)

# 计算多空收益
hs300_ret = get_excess_ret(hs300_v, list(
    map(lambda x: '%.2f%%' % x, range(20, 100, 10))))

zz500_ret = get_excess_ret(zz500_v, list(
    map(lambda x: '%.2f%%' % x, range(20, 100, 10))))

hs300_cum = ep.cum_returns(hs300_ret).add(1)
zz500_cum = ep.cum_returns(zz500_ret).add(1)

# ===== CELL =====

# 画图
fig, v_axes = plt.subplots(2, 2, figsize=(18, 11))

v_axes[0, 0].set_title('沪深300分组平均收益')
hs300_ret.mean().plot.bar(ax=v_axes[0, 0], color='#3A73AD')
v_axes[0, 1].set_title('中证500分组平均收益')
zz500_ret.mean().plot.bar(ax=v_axes[0, 1], color='#3A73AD')
v_axes[1, 0].set_title('沪深300分组净值')
hs300_cum.plot(ax=v_axes[1, 0])
v_axes[1, 1].set_title('中证500分组净值')
zz500_cum.plot(ax=v_axes[1, 1])
plt.subplots_adjust(hspace=0.3);

# ===== CELL =====

hs300_ic = hs300_v.groupby(level='date').apply(src_ic)
zz500_ic = zz500_v.groupby(level='date').apply(src_ic)

# ===== CELL =====

# 获取IC相关信息
def ic_table(ic_df:pd.DataFrame)->pd.DataFrame:
    
    df = pd.DataFrame()
    df['IC Mean'] = ic_df.mean()
    df['IC Std'] = ic_df.std()
    df['ICIR'] = df['IC Mean'] / df['IC Std']
    
    return df

# ===== CELL =====

print_table(ic_table(hs300_ic),name='沪深300 IC信息')
print_table(ic_table(zz500_ic),name='中证500 IC信息')

# ===== CELL =====

print_table(risk_indicator_tear(hs300_ret), name='沪深300风险指标', fmt='{:.2%}')
print_table(risk_indicator_tear(zz500_ret), name='中证500风险指标', fmt='{:.2%}')

# ===== CELL =====

# 画图
mpl.rcParams['font.family'] = 'serif'

beginDt = hs300_cum.index.min()
endDt = hs300_cum.index.max()
benchmark = get_price('000300.XSHG', beginDt, endDt, fields='close')
benchmark = benchmark.reindex(hs300_cum.index)

hs300_cum['60.00%'].plot(figsize=(18, 5), title='不同样本空间多空对冲净值', label='沪深300')
zz500_cum['50.00%'].plot(label='中证500', color='#7232A1')
(benchmark['close'] / benchmark['close'][0]
 ).plot(label='沪深300', color='darkgray', ls='--')
plt.legend();