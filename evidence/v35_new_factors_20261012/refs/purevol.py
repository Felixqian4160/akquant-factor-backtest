import numpy as np
import pandas as pd
import alphalens as al
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from sklearn import linear_model

from dateutil.parser import parse
from tqdm import tqdm_notebook # 进度条

from jqdata import *
from jqfactor import (Factor,calc_factors,neutralize,get_factor_values)

# 画图
import seaborn as sns
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
plt.style.use('seaborn')

# ===== CELL =====

def get_stockpool(watch_date:str)->list:
    
    '''获取股票池'''
    
    stockList = get_index_stocks('000002.XSHG', date=watch_date) + get_index_stocks(
                '399107.XSHE', date=watch_date)
    
    stockList = del_st_stock(stockList,watch_date)
    stockList = del_iponday(stockList,watch_date)
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


def plot_nav(nav: pd.DataFrame, title: str):
    
    '''画图'''
    
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

class ID_Vol(Factor):

    name = 'ID_Vol'
    max_window = 21
    dependencies = ['market_cap', 'pb_ratio', 'close']

    def calc(self, data):
        
        # 获取bm
        bm_df = 1. / data['pb_ratio']
        # 收益率
        returns = data['close'].pct_change()
        # 分组标记
        market_cap_label = self.add_label(data['market_cap'], 10)
        bm_label = self.add_label(bm_df, 10)
        # 构造因子
        smb = self.calc_longshort(data, market_cap_label, returns, 1, 10)
        hml = self.calc_longshort(data, bm_label, returns, 10, 1)
        mkt = (data['market_cap'].div(data['market_cap'].sum(
            axis=1), axis=0) * returns).sum(axis=1)

        # 添加阶矩项项
        exog = sm.add_constant(pd.concat([mkt, smb, hml], axis=1))
        EXOG_COL = ['constant', 'mkt', 'smb', 'hml']
        exog.columns = EXOG_COL

        returns = returns.iloc[-20:]
        exog = exog.reindex(returns.index)  # 日期索引对齐
        # OLS回归获取残差
        mod = sm.OLS(returns, exog)
        res = mod.fit()
        resid_df = res.resid

        return resid_df.iloc[-20:].std()

    def calc_longshort(self, data, label_df: pd.DataFrame, returns: pd.DataFrame, high: int, low: int) -> pd.Series:
        '''计算多空收益 high - low'''

        # 计算组合权重
        l_w = self.get_weight(label_df, data['market_cap'], high)
        s_w = self.get_weight(label_df, data['market_cap'], low)

        h = (l_w * returns).sum(axis=1)
        l = (s_w * returns).sum(axis=1)

        return h - l

    @staticmethod
    def add_label(df: pd.DataFrame, N: int) -> pd.DataFrame:
        '''获取分组 分位数分组(每组数量相同)'''
        return df.apply(lambda x: pd.qcut(x, N, labels=[i for i in range(1, N+1)]), axis=1)

    @staticmethod
    def get_weight(label_df: pd.DataFrame, market_cap: pd.DataFrame, N: int) -> pd.DataFrame:
        '''获取目标组的权重'''

        cond = (label_df == N)
        cap = cond * market_cap  # 该组市值

        return cap.div(cap.sum(axis=1), axis=0)
    
def get_factor(start_date:str,end_date:str,factor):
    
    # 屏蔽Factor类提取行情数据的警告
    import warnings
    warnings.filterwarnings("ignore")
        
    periods = GetTradePeriod(start_date,end_date)
    
    factor_list = []
    for i in tqdm_notebook(periods,desc='数据获取中'):
        
        stocks = get_stockpool(i.strftime('%Y-%m-%d'))
        
        factor_list.append(calc_factors(stocks,
                                    factors=[factor],
                                    start_date=i,
                                    end_date=i)[factor.name])
        
    factor_df = pd.concat(factor_list,sort=False)
    
    return factor_df

def get_factor_price(factor:pd.DataFrame,last_date:str)->pd.DataFrame:
    
    periods = [i.strftime('%Y-%m-%d') for i in factor.index.tolist()] + [last_date]
    stocks = factor.columns.tolist()
    df_list = []
    for i in tqdm_notebook(periods,desc='收益率下载中'):
        
        df_list.append(get_price(stocks,end_date=i,count=1,fields='close',panel=False))
        
    df = pd.concat(df_list)
    close = pd.pivot_table(df,index='time',columns='code',values='close')
    return close

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

# 获取特质因子
id_vol = get_factor('2013-01-01','2020-09-30',ID_Vol())

# ===== CELL =====

# 获取价格数据
pricing = get_factor_price(id_vol,'2020-10-31')

# ===== CELL =====

# 数据储存
id_vol.to_csv('../Data/id_vol.csv')
pricing.to_csv('../Data/priceing.csv')

# ===== CELL =====

# 数据读取
id_vol = pd.read_csv('../Data/id_vol.csv',index_col=[0],parse_dates=[0])
pricing = pd.read_csv('../Data/priceing.csv',index_col=[0],parse_dates=[0])

# 以2014年为起点
id_vol_1 = id_vol.loc['2014-01-01':]
pricing_1 = pricing.loc['2014-01-01':]

# ===== CELL =====

# 因子分组排序默认未从小到大
id_vol_factor = al.utils.get_clean_factor_and_forward_returns(id_vol_1.stack(),
                                                            pricing_1,
                                                            periods=(1,),
                                                            quantiles=5)
# 计算分组收益
rets = pd.pivot_table(id_vol_factor.reset_index(level=0),
                      index='date', columns='factor_quantile', values=1)

rets['G1-G5'] = rets[1] - rets[5] 
nav = (1 + rets).cumprod() # 计算净值

# ===== CELL =====

plot_nav(nav,'传统特质波动率因子ID_Vol的5分组及多空对冲净值走势')

# ===== CELL =====

id_vol_ic = al.performance.factor_information_coefficient(id_vol_factor)
id_vol_ic.columns=['ID_Vol']
al.plotting.plot_information_table(id_vol_ic)

# ===== CELL =====

def get_idvol_deturn20(factor: pd.DataFrame) -> pd.DataFrame:
    '''
    计算idvol_deturn20因子
    -------------
        本质是因子与换手率因子正交，如果传入的是ID_Vol因子则生成idvol_deturn20
    '''
    stocks = factor.columns.tolist()
    periods = [i.strftime('%Y-%m-%d') for i in factor.index]

    vol_list = []
    for i in tqdm_notebook(periods, desc='数据获取中'):

        tmp_dic = get_factor_values(
            stocks, factors=['VOL20'], end_date=i, count=1)
        vol_list.append(neutralize(tmp_dic['VOL20'], how='market_cap', date=i))

    turn20 = pd.concat(vol_list)
    turn20 = turn20.fillna(0)
    
    # 正交化

    def get_resid(x): return sm.OLS(factor.loc[:, x].fillna(
        0), sm.add_constant(turn20.loc[:, x])).fit().resid
    
    idvol_deturn20 = pd.concat((get_resid(code) for code in stocks), axis=1)
    idvol_deturn20.columns = stocks
    
    return idvol_deturn20, turn20  # 返回idvol_deturn20和turn20因子

# ===== CELL =====

# 获取ID_Vol_deTurn20和换手率因子
idvol_deturn20,turn20 = get_idvol_deturn20(id_vol_1)

# ===== CELL =====

# 转换格式
idvol_deturn20_ = idvol_deturn20.stack().dropna()
turn20_ = turn20.stack().dropna()

# 因子分组排序默认未从小到大
idvol_deturn20_factor = al.utils.get_clean_factor_and_forward_returns(idvol_deturn20_,
                                                            pricing_1,
                                                            periods=(1,),
                                                            quantiles=5)

turn20_factor = al.utils.get_clean_factor_and_forward_returns(turn20_,
                                                            pricing_1,
                                                            periods=(1,),
                                                            quantiles=5)

# 计算分组收益
rets1 = pd.pivot_table(idvol_deturn20_factor.reset_index(level=0),
                      index='date', columns='factor_quantile', values=1)

rets1['G1-G5'] = rets1[1] - rets1[5] 
nav1 = (1 + rets1).cumprod() # 计算净值

# ===== CELL =====

plot_nav(nav1,'D_Vol_deTurn20的5分组及多空对冲净值走势')

# ===== CELL =====

idvol_deturn20_ic = al.performance.factor_information_coefficient(idvol_deturn20_factor)
turn_ic = al.performance.factor_information_coefficient(turn20_factor)

id_vol_ic['ID_Vol_deTurn20'] = idvol_deturn20_ic
id_vol_ic['turn20'] = turn_ic
al.plotting.plot_information_table(id_vol_ic)

# ===== CELL =====

index_price = get_price('000001.XSHG', start_date='2005-05-01',
                        end_date='2020-09-30', fields='close')

index_price.pct_change().plot(figsize=(18, 6), title='指数每日收益率：波动聚集性')

# ===== CELL =====

acf_df = id_vol.apply(lambda x:acf(x,nlags=12))
corr_df = pd.concat((id_vol.corrwith(id_vol.shift(-i)) for i in range(0,13)),axis=1)

df = pd.concat((corr_df.T.mean(axis=1),acf_df.mean(axis=1)),axis=1)
df.columns = ['平均截面相关系数','平均时序相关系数']

# ===== CELL =====

plt.rcParams['font.family'] = 'serif'
df.plot.bar(title='特质波动率因子 ID_Vol的平均时序相关系数',figsize=(10,6))

# ===== CELL =====

def get_id_vol_deCorr(factor:pd.DataFrame,N:int)->pd.DataFrame:
    '''获取id_vol_deCorr因子'''
    N = N + 1
    
    def get_resid(factor:pd.DataFrame)->pd.Series:
        
        x = factor.iloc[:6].T.fillna(0)
        y = factor.iloc[-1].dropna()
        
        x = x.reindex(y.index)
        #regr = linear_model.Ridge()
        #regr.fit(x, y)
        #resid = y - regr.predict(x)
        resid = sm.OLS(y,sm.add_constant(x)).fit().resid
        resid.name = factor.index[-1]
        
        return resid
    
    iidx = np.arange(len(factor))
    
    shape = (iidx.size - N + 1, N)
    
    strides = (iidx.strides[0], iidx.strides[0])
    
    res = np.lib.stride_tricks.as_strided(
        iidx, shape=shape, strides=strides, writeable=True)
    
    return pd.concat((get_resid(factor.iloc[i]) for i in res),axis=1,sort=True).T

# ===== CELL =====

# 计算因子
id_vol_deCorr = get_id_vol_deCorr(id_vol,6)

# ===== CELL =====

# 因子分组排序默认未从小到大
id_vol_deCorr_ = id_vol_deCorr.loc['2014-01-01':].stack().dropna()
id_vol_deCorr_factor = al.utils.get_clean_factor_and_forward_returns(id_vol_deCorr_,
                                                            pricing_1,
                                                            periods=(1,),
                                                            quantiles=5)
# 计算分组收益
rets2 = pd.pivot_table(id_vol_deCorr_factor.reset_index(level=0),
                      index='date', columns='factor_quantile', values=1)

rets2['G1-G5'] = rets2[1] - rets2[5] 
nav2 = (1 + rets2).cumprod() # 计算净值

# ===== CELL =====

plot_nav(nav2,'纯波动率因子的5分组及多空对冲净值走势')

# ===== CELL =====

id_vol_deCorr_ic= al.performance.factor_information_coefficient(id_vol_deCorr_factor)
id_vol_ic['id_vol_deCorr'] = id_vol_deCorr_ic
al.plotting.plot_information_table(id_vol_ic)

# ===== CELL =====

acf_df = id_vol_deCorr.apply(lambda x:acf(x,nlags=12))
corr_df = pd.concat((id_vol_deCorr.corrwith(id_vol.shift(-i)) for i in range(0,13)),axis=1)

df = pd.concat((corr_df.T.mean(axis=1),acf_df.mean(axis=1)),axis=1)
df.columns = ['平均截面相关系数','平均时序相关系数']
df.plot.bar(title='特质波动率因子ID_Vol_DeCorr的平均时序相关系数',figsize=(10,6))

# ===== CELL =====

idvol_deCturn20,turn20 = get_idvol_deturn20(id_vol_deCorr)

# ===== CELL =====

# 转换格式
idvol_deCturn20_ = idvol_deCturn20.stack().dropna()

# 因子分组排序默认未从小到大
idvol_deturn20_factor = al.utils.get_clean_factor_and_forward_returns(idvol_deCturn20_,
                                                            pricing_1,
                                                            periods=(1,),
                                                            quantiles=5)

# 计算分组收益
rets3 = pd.pivot_table(idvol_deturn20_factor.reset_index(level=0),
                      index='date', columns='factor_quantile', values=1)

rets3['G1-G5'] = rets3[1] - rets3[5] 
nav3 = (1 + rets3).cumprod() # 计算净值

# ===== CELL =====

compare_df = pd.concat([nav3['G1-G5'],nav1['G1-G5']],axis=1)
compare_df.columns = ['deCorr正交','deTurn正交']

compare_df.plot(figsize=(18,6),title='新旧因子正交换手后的多空对冲净值走势',color=['Crimson','DarkGray'])
plt.axhline(1,color='black',lw=0.5)

# ===== CELL =====

class CH3_ID_Vol(Factor):

    name = 'CH3_ID_Vol'
    max_window = 21
    dependencies = ['market_cap', 'pe_ratio', 'close']

    def calc(self, data):
        
        # 过滤市值底部30%是的股票
        market_cap = data['market_cap']
        market_cap_ser = market_cap.iloc[-1]
        # 30%
        threshold = market_cap_ser.quantile(0.3)
        stocks = market_cap_ser[market_cap_ser > threshold].index.tolist()
        
        # 获取符合条件的股票数据
        pe_df = data['pe_ratio'][stocks]
        market_cap = market_cap[stocks]
        close_df = data['close'][stocks]
        
        # 获取ep
        ep_df = 1. / pe_df
        # 收益率
        returns = close_df.pct_change()

        # 分组标记
        market_cap_label = self.add_label(market_cap, 10)
        ep_label = self.add_label(ep_df, 10)
        # 构造因子
        smb = self.calc_longshort(market_cap, market_cap_label, returns, 1, 10)
        vmg = self.calc_longshort(market_cap, ep_label, returns, 10, 1)
        mkt = (market_cap.div(market_cap.sum(
            axis=1), axis=0) * returns).sum(axis=1)

        # 添加阶矩项项
        exog = sm.add_constant(pd.concat([mkt, smb, vmg], axis=1))
        EXOG_COL = ['constant', 'mkt', 'smb', 'vmg']
        exog.columns = EXOG_COL

        returns = returns.iloc[-20:]
        exog = exog.reindex(returns.index)  # 日期索引对齐
        # OLS回归获取残差
        mod = sm.OLS(returns, exog)
        res = mod.fit()
        resid_df = res.resid

        return resid_df.iloc[-20:].std()

    def calc_longshort(self, market_cap:pd.DataFrame, label_df: pd.DataFrame, returns: pd.DataFrame, high: int, low: int) -> pd.Series:
        '''计算多空收益 high - low'''

        # 计算组合权重
        l_w = self.get_weight(label_df, market_cap, high)
        s_w = self.get_weight(label_df, market_cap, low)

        h = (l_w * returns).sum(axis=1)
        l = (s_w * returns).sum(axis=1)

        return h - l

    @staticmethod
    def add_label(df: pd.DataFrame, N: int) -> pd.DataFrame:
        '''获取分组 分位数分组(每组数量相同)'''
        return df.apply(lambda x: pd.qcut(x, N, labels=[i for i in range(1, N+1)]), axis=1)

    @staticmethod
    def get_weight(label_df: pd.DataFrame, market_cap: pd.DataFrame, N: int) -> pd.DataFrame:
        '''获取目标组的权重'''

        cond = (label_df == N)
        cap = cond * market_cap  # 该组市值

        return cap.div(cap.sum(axis=1), axis=0)

# ===== CELL =====

ch3_id_vol = get_factor('2013-01-01','2020-09-30',CH3_ID_Vol())

# 获取价格数据
ch3_pricing = get_factor_price(ch3_id_vol,'2020-10-31')

# ===== CELL =====

# 数据储存
ch3_id_vol.to_csv('../Data/ch3_id_vol.csv')
ch3_pricing.to_csv('../Data/ch3_pricing.csv')

# ===== CELL =====

# 数据读取
ch3_id_vol = pd.read_csv('../Data/ch3_id_vol.csv',index_col=[0],parse_dates=[0])
ch3_pricing = pd.read_csv('../Data/ch3_pricing.csv',index_col=[0],parse_dates=[0])

# 以2014年为起点
ch3_id_vol_1 = ch3_id_vol.loc['2014-01-01':]
ch3_pricing_1 = ch3_pricing.loc['2014-01-01':]

# ===== CELL =====

ch3_id_vol_ = ch3_id_vol_1.stack().dropna()
# 因子分组排序默认未从小到大
ch3_id_vol_factor = al.utils.get_clean_factor_and_forward_returns(ch3_id_vol_,
                                                            ch3_pricing_1,
                                                            periods=(1,),
                                                            quantiles=5)
# 计算分组收益
rets4 = pd.pivot_table(ch3_id_vol_factor.reset_index(level=0),
                      index='date', columns='factor_quantile', values=1)

rets4['G1-G5'] = rets4[1] - rets4[5] 
nav4 = (1 + rets4).cumprod() # 计算净值

# ===== CELL =====

plot_nav(nav4,'CH3_ID_Vol')

# ===== CELL =====

ch3_id_vol_ic = al.performance.factor_information_coefficient(ch3_id_vol_factor)
ch3_id_vol_ic.columns=['CH3_ID_Vol']
al.plotting.plot_information_table(ch3_id_vol_ic)

# ===== CELL =====

