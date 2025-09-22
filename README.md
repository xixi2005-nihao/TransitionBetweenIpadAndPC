
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from datetime import datetime
import time

import os
import itertools  

#展示全部dataframe，忽略警告
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None) 
import warnings
warnings.filterwarnings("ignore") 

# initial_cash=10000
assets =[  'ADA', 'AVAX', 'BCH', 'BNB','BTC','DOGE', 'DOT', 'ETH', 'ETC', 'FIL', 'LINK', 'LTC', 'POL', 'NEAR', 'SOL', 'TRX', 'UNI', 'XRP']#,'XLM', 'AAVE', 'VET', 'ATOM','ALGO','MKR']
# assets = ['ADA']
months=[ '01', '02', '03', '04', '05', '06', '07', '08', '09','10', '11', '12']

#交易规则https://www.binance.com/zh-CN/futures/trading-rules/perpetual
#stepSize={'ETC':0.01, 'MATIC':1, 'XRP':0.1, 'ADA':1, 'TRX':1, 'BTC':0.001, 'ETH':0.001, 'SOL':1, 'BNB':0.01, 'LTC':0.001, 'DOGE':1, 'DOT':0.1, 'AVAX':0.01,'ATOM':0.01,'UNI':1}
stepSize={
    'ADA': 1,
    'AVAX': 1,
    'BCH': 0.001,
    'BNB': 0.01,
    'BTC': 0.001,
    'DOGE': 1,
    'DOT': 0.1,
    'ETH': 0.001,
    'ETC': 0.01,
    'FIL': 0.1,
    'LINK': 0.01,
    'LTC': 0.001,
    'POL': 1,
    'NEAR': 1,
    'SOL': 1,
    'TRX': 1,
    'UNI': 1,
    'XRP': 0.1,
    'XLM': 1,
    'AAVE':0.001, 
    'VET':0.1, 
    'ATOM':0.01,
    'ALGO':1,
    'MKR':0.0001
}
#所有配对组合
b = list(itertools.permutations(assets, 2))
print(b)

import pandas as pd
import numpy as np
import calendar

 

# 生成时间戳 DataFrame
# time_range = pd.date_range(start=start_timestamp, end=end_timestamp, freq='30min')
# timestamp_df = pd.DataFrame({'timestamp': time_range})

# 存储每个资产的数据
assets_df = {}

# ms 转换后近似到最近一分钟
def round_to_next_minute(timestamp):
    return timestamp.ceil(freq='min').replace(second=0, microsecond=0)

for asset in assets:
    monthly_file = f"F:/Research_pku/趋势跟随wma/2021_01_01/data/{asset}USDT_30m.csv"
               
    try:
        # 加载月度数据
        df = pd.read_csv(monthly_file)
        # df = df.iloc[:, 1:] 
        # 检查第一行是否是列名timestamp	open	high	low	close	volume	close_time	quote_asset_volume	number_of_trades	taker_buy_base_volume	taker_buy_quote_volume	ignore
        # if not all(col in df.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore']):
        #     first_row_data = df.columns
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
            # original_data_types = df.dtypes
            # df.loc[-1] = first_row_data
            # df = df.astype(original_data_types)
            # df.index = df.index + 1
            # df.sort_index(inplace=True)

  
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df['close'] = df['close'].ffill().bfill()  # 先向前，再向后填充
        df['close_avg'] = np.where((df['taker_buy_volume'] != 0 ) & ( df['taker_buy_quote_volume']!=0) ,
            df['taker_buy_quote_volume'] / df['taker_buy_volume'], df['close'] )

        # df['close']=df['taker_buy_quote_volume']/df['taker_buy_volume']
 
        assets_df[asset]=df.copy() 
        
    except FileNotFoundError:
        print(f"Monthly file not found: {monthly_file}")
        continue
Asset_Total_Cash = {
    'Asset': ['DOT', 'ADA', 'FIL', 'BTC', 'DOGE', 'NEAR', 'UNI', 'BNB', 'ETH', 'LINK', 'LTC', 'TRX', 'SOL', 'XRP', 'AVAX', 'POL'],
    'Count': [35, 30, 30, 20, 20, 20, 20, 15, 10, 10, 10, 10, 10, 10, 5, 5],
    'Total_Cash': [175000.0, 150000.0, 150000.0, 100000.0, 100000.0, 100000.0, 100000.0, 75000.0, 50000.0, 50000.0, 50000.0, 50000.0, 50000.0, 50000.0, 25000.0, 25000.0]
}
# Asset_Total_Cash = {
#     'Asset': ['ADA'],
#     'Count': [30],
#     'Total_Cash': [150000.0]
# }
Asset_Total_Cash = pd.DataFrame(Asset_Total_Cash)

def get_num_decimals(step_size):
    """
    计算 step_size 对应的小数位数，如果 step_size 是整数，返回 0
    """
    if isinstance(step_size, int) or step_size.is_integer():
        return 0
    return len(str(step_size).split('.')[-1])



def truncate(value, step_size):
    """
    该函数用于将 value 按照 step_size 进行截断
    """
    # 计算需要保留的小数位数
    num_decimals = get_num_decimals(step_size)
    # 乘以 10 的 num_decimals 次方，取整，再除以 10 的 num_decimals 次方
    factor = 10 ** num_decimals
    result = int(value * factor) / factor
    return result
def calculate_sharpe_ratio(portfolio_value_long, portfolio_value_short, risk_free_rate=0.00):
    # 将输入参数转换为 numpy 数组
    initial=portfolio_value_long[0]
    # print('initial total cash long and short:',portfolio_value_long[0],portfolio_value_short[0])
    portfolio_value_long = np.array(portfolio_value_long)
    portfolio_value_short = np.array(portfolio_value_short)

    portfolio_value_long = portfolio_value_long - portfolio_value_long[0]
    portfolio_value_short = portfolio_value_short - portfolio_value_short[0]
    # 计算持仓价值，将多头和空头的组合profit相加
    portfolio_value_sum= (portfolio_value_long + portfolio_value_short+initial) 
    if np.any(portfolio_value_sum < 0):
        return 0
    portfolio_value_series = pd.Series(portfolio_value_sum)
    # 计算收益率
    returns = portfolio_value_series.pct_change().fillna(0)
    # 计算平均收益率
    average_return = np.mean(returns) * 365 * 48  # 年化
    # 计算收益率的标准差
    std_return = np.std(returns) * np.sqrt(365 * 48)
    # 计算夏普比率
    if std_return == 0:
        return 0  # 避免除零错误
    sharpe_ratio = (average_return - risk_free_rate) / std_return
    return sharpe_ratio

def wma_linear_reverse(S, n, k):
    # 生成权重，越近期的数据权重越大
    weights = np.array([i for i in range(k + 1)])
    m = np.sum(weights)
    window = S[n - k:n + 1]
    return np.sum(weights * window) / m


def calculate_wma(data, short_window, long_window):
    short_wma = []
    long_wma = []
    close_prices = data['close_avg'].values

    for i in range(len(close_prices)):
        if i < short_window - 1:
            short_wma.append(np.nan)
        else:
            short_wma.append(wma_linear_reverse(close_prices, i, short_window - 1))

        if i < long_window - 1:
            long_wma.append(np.nan)
        else:
            long_wma.append(wma_linear_reverse(close_prices, i, long_window - 1))

    data['Short_WMA'] = short_wma
    data['Long_WMA'] = long_wma
    return data


# # 示例数据
# data = pd.DataFrame({
#     'close': [10, 12, 15, 14, 16, 18, 20]
# })
# short_window = 3#第一个赋予0
# long_window = 5

# result = calculate_wma(data, short_window, long_window)
#最开始直接开仓 ，且结尾开仓处理成unrealized profit

# 策略实现
def trend_following_strategy_long(data, short_window, long_window):
      
    wma=calculate_wma(data, short_window, long_window)
    # 计算短期和长期均线
    data['Short_MA'] = wma['Short_WMA']#data['close'].rolling(window=short_window).mean()
    data['Long_MA'] =wma['Long_WMA']# data['close'].rolling(window=long_window).mean()

    # 定义开仓和平仓条件
    data['Open_Condition'] = (data['Short_MA'] > data['Long_MA'])
    data['Close_Condition'] = (data['Short_MA'] < data['Long_MA'])

    # 生成 Position 列
    data['Position'] = np.where(
        (data['Open_Condition']),  # 开仓
        1,
        np.where(
            (data['Close_Condition']),  # 平仓
            0,
            np.nan  # 无操作保持不变
        )
    )
    data['Position'] = data['Position'].ffill().fillna(0).astype(int)  # 持仓状态
    data['Signal'] = data['Position'].diff().fillna(0).astype(int)  # 信号列
    mask = data['Signal'] == -1
    data.loc[mask, 'Position'] = 1#算关仓那一步的Strategy_Return
    # 计算收益
    data['Return'] = data['close'].pct_change()
    data['Strategy_Return'] = data['Position'] * data['Return']
    data = data.reset_index(drop=True)
    return data 
def get_num_decimals(step_size):
    """
    计算 step_size 对应的小数位数，如果 step_size 是整数，返回 0
    """
    if isinstance(step_size, int) or step_size.is_integer():
        return 0
    return len(str(step_size).split('.')[-1])


def truncate(value, step_size):
    """
    该函数用于将 value 按照 step_size 进行截断
    """
    # 计算需要保留的小数位数
    num_decimals = get_num_decimals(step_size)
    # 乘以 10 的 num_decimals 次方，取整，再除以 10 的 num_decimals 次方
    factor = 10 ** num_decimals
    result = int(value * factor) / factor
    return result
 

def backtest_long(data, initial_cash_per_trade,asset):
    # 初始化 Portfolio_Value 列为 0
    data['Portfolio_Value'] = 0
    data = data.astype({'Portfolio_Value': 'float64'})
    data['currency']=asset  
    
    # 查找开仓和闭仓的时间点
    data['Open_Condition'] = (data['Signal'] == 1)
    data['Close_Condition'] = (data['Signal'] == -1)
    
    # 找到开仓和闭仓的索引
    open_indices = data[data['Open_Condition']].index
    close_indices = data[data['Close_Condition']].index

    # 初始化开盘和平仓的收益
    trade_profit = []
    
 
    # 计算每一笔交易的收益
    for start, end in zip(open_indices, close_indices):
 
        # 设置对应的 Portfolio_Value
        data.loc[start, 'Portfolio_Value'] = initial_cash_per_trade
        qty=truncate( initial_cash_per_trade/data.loc[start, 'close'], stepSize[data.loc[start, 'currency']])
        #开仓第二行开始减去手续费
        data.loc[start+1:end-1, 'Portfolio_Value'] =data.loc[start , 'Portfolio_Value']+qty *(data.loc[start+1:end-1, 'close']-data.loc[start, 'close'])-qty*(data.loc[start, 'close'])*2/10000
        #最后关仓时减去手续费
        data.loc[end, 'Portfolio_Value'] =data.loc[start , 'Portfolio_Value']+qty *(data.loc[end, 'close'] - data.loc[start, 'close'])-qty*(data.loc[start, 'close']+data.loc[end, 'close'])*2/10000
              
        # 计算该交易的利润
        trade_profit.append(data.loc[end, 'Portfolio_Value']-initial_cash_per_trade)
    if len(open_indices) > len(close_indices):
        data.loc[open_indices[-1], 'Portfolio_Value'] = initial_cash_per_trade         
        qty=truncate( initial_cash_per_trade/data.loc[open_indices[-1], 'close'], stepSize[data.loc[open_indices[-1], 'currency']])
        #开仓第二行开始减去手续费
        data.loc[open_indices[-1]+1:, 'Portfolio_Value'] =data.loc[open_indices[-1] , 'Portfolio_Value']+qty *(data.loc[open_indices[-1]+1:, 'close']-data.loc[open_indices[-1], 'close'])-qty*(data.loc[open_indices[-1], 'close'])*2/10000
    data['Portfolio_Value'] = data['Portfolio_Value'].astype(float)
    data['Portfolio_Value'] = data['Portfolio_Value'].replace(0, pd.NA)
    data['Portfolio_Value'] = data['Portfolio_Value'].ffill()
    data=data.fillna(0) 
    # data['Portfolio_Value_everytrade'] = data['Portfolio_Value'] 
    # data['Portfolio_Value_everytrade'] = data['Portfolio_Value_everytrade'].mask(data['Portfolio_Value_everytrade'] == 0).bfill()#一开始就放钱，记录每次交易放钱的当笔钱数变化，空仓期间记录的是上一笔关仓时的钱数
    # data['return_everytrade']=data['Portfolio_Value_everytrade'].pct_change().fillna(0)
    # data.loc[data['Signal'] == -1, 'return_everytrade'] = 0

    # 按照交易利润更新 Portfolio_Value
    for i, start in enumerate(open_indices[1:]):  # 从第二个开盘开始更新
        data.loc[start:, 'Portfolio_Value'] +=  trade_profit[i]
        # print('扣除手续费的收益率:',(trade_profit[i]-4.5/10000*initial_cash_per_trade)/initial_cash_per_trade)

    # 按固定金额持有计算 Buy and Hold 策略
    data['Buy_and_Hold_Value'] = (1 + data['Return']).cumprod() * initial_cash_per_trade
    
    # print(f"long Total Trades: {len(trade_profit)}")
    # #print(f"avg return after rate: {sum(trade_profit)/len(trade_profit/initial_cash_per_trade)}") if len(trade_profit)!= 0 else print(0)
    # print("avg return after trading fee:", sum(trade_profit)/len(trade_profit)/initial_cash_per_trade if len(trade_profit)!= 0 else print(0))
    # total_profit_afterrate= sum(trade_profit)-initial_cash_per_trade*len(trade_profit)*9/10000

    return data, trade_profit, open_indices, close_indices 
 

# 策略实现
def trend_following_strategy_short(data, short_window, long_window):
    wma=calculate_wma(data, short_window, long_window)
    # 计算短期和长期均线
    data['Short_MA'] = wma['Short_WMA']#data['close'].rolling(window=short_window).mean()
    data['Long_MA'] =wma['Long_WMA']# data['close'].rolling(window=long_window).mean()

    data['Open_Condition'] = (data['Short_MA'] < data['Long_MA'])
    data['Close_Condition'] = (data['Short_MA'] > data['Long_MA'])

    # 使用向量化逻辑生成 Position 和 Signal
    data['Position'] = 0
    data['Position'] = np.where(
        (data['Open_Condition'] ),  # 开仓条件
        -1,
        np.where(
            (data['Close_Condition']  ),  # 平仓条件
            0,
            np.nan  # 无操作，保持之前的状态
        )
    )
    data['Position'] = data['Position'].ffill().fillna(0).astype(int)
    data['Signal'] = data['Position'].diff().fillna(0).astype(int)
    #设置关仓的那一行持仓不为0，把这一行Strategy_Return算出
    mask = data['Signal'] == 1
    data.loc[mask, 'Position'] = -1
    # 计算每日收益
    data['Return'] = data['close'].pct_change()
    data['Strategy_Return'] = data['Position'] * data['Return']

    return data
 

def backtest_short(data, initial_cash_per_trade,asset):
    # 初始化 Portfolio_Value 列为 0
    data['Portfolio_Value'] = 0
    data = data.astype({'Portfolio_Value': 'float64'})
    data['currency']=asset
    
    # 查找开仓和闭仓的时间点
    data['Open_Condition'] = (data['Signal'] == -1)
    data['Close_Condition'] = (data['Signal'] == 1)
    
    # 找到开仓和闭仓的索引
    open_indices = data[data['Open_Condition']].index
    close_indices = data[data['Close_Condition']].index

    # 初始化开盘和平仓的收益
    trade_profit = []
    # 如果最后的开盘没有对应的平仓，给它设定一个默认的平仓点（最后一个时间点）
    # if len(open_indices) > len(close_indices):
    #     close_indices = close_indices.append(pd.Index([data.index[-1]]))    
    # print(len(open_indices),len(close_indices))
    # 计算每一笔交易的收益
    for start, end in zip(open_indices, close_indices):
        data.loc[start, 'Portfolio_Value'] = initial_cash_per_trade
        #data.loc[start + 1:end, 'Portfolio_Value'] = cumulative_returns * initial_cash_per_trade-4.5/10000*initial_cash_per_trade
        qty=truncate( initial_cash_per_trade/data.loc[start, 'close'], stepSize[data.loc[start, 'currency']])
    
        data.loc[start+1:end-1, 'Portfolio_Value'] =data.loc[start , 'Portfolio_Value']+qty *(data.loc[start, 'close']-data.loc[start+1:end-1, 'close'])-qty*(data.loc[start, 'close'])*2/10000
       
        data.loc[end, 'Portfolio_Value'] =data.loc[start , 'Portfolio_Value']+qty *(data.loc[start, 'close'] - data.loc[end, 'close'])-qty*(data.loc[start, 'close']+data.loc[end, 'close'])*2/10000
                     

        # 计算该交易的利润
        trade_profit.append(data.loc[end, 'Portfolio_Value'] -initial_cash_per_trade )#记录的去掉本金和手续费的profit
    if len(open_indices) > len(close_indices):
        print(len(open_indices),len(close_indices))

        data.loc[open_indices[-1], 'Portfolio_Value'] = initial_cash_per_trade         
        qty=truncate( initial_cash_per_trade/data.loc[open_indices[-1], 'close'], stepSize[data.loc[open_indices[-1], 'currency']])
        data.loc[open_indices[-1]+1:, 'Portfolio_Value'] =data.loc[open_indices[-1] , 'Portfolio_Value']+qty *(data.loc[open_indices[-1], 'close']-data.loc[open_indices[-1]+1:, 'close'])-qty*(data.loc[open_indices[-1], 'close'])*2/10000
    
    data['Portfolio_Value'] = data['Portfolio_Value'].astype(float)
    data['Portfolio_Value'] = data['Portfolio_Value'].replace(0, pd.NA)
    data['Portfolio_Value'] = data['Portfolio_Value'].ffill()#补齐every trade之间为0的部分
    data=data.fillna(0) 
    # data['Portfolio_Value_everytrade'] = data['Portfolio_Value'] 
    # data['Portfolio_Value_everytrade'] = data['Portfolio_Value_everytrade'].mask(data['Portfolio_Value_everytrade'] == 0).bfill()
    # data['return_everytrade']=data['Portfolio_Value_everytrade'].pct_change().fillna(0)#可能出现负数没办法算收益
    # data.loc[data['Signal'] == 1, 'return_everytrade'] = 0
    
    # 按照交易利润更新 Portfolio_Value
    for i, start in enumerate(open_indices[1:]):  # 从第二个开盘开始更新
        data.loc[start:, 'Portfolio_Value'] +=  trade_profit[i]#去掉手续费的收益额
        #print('扣除手续费的收益率:',(trade_profit[i]-4.5/10000*initial_cash_per_trade)/initial_cash_per_trade)

    # 按固定金额持有计算 Buy and Hold 策略
    data['Buy_and_Hold_Value'] = (1 + data['Return']).cumprod() * initial_cash_per_trade
    
    # print(f"short Total Trades: {len(trade_profit)}")
    # print("avg return after trading fee:", sum(trade_profit)/len(trade_profit)/initial_cash_per_trade  if len(trade_profit)!= 0 else print(0))
    # total_profit_afterrate= sum(trade_profit)-initial_cash_per_trade*len(trade_profit)*9/10000
     
    return data, trade_profit, open_indices, close_indices #这里面只看end时累计实现收益，不要看Portfolio_Value的变化代码里其实没算
 
  
 
# assets_df = copy.deepcopy(assets_df_copy)
minutes = 30
  
long_short_pairs = []
short_windows = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 500, 1000]
long_windows = [30, 40, 50, 60, 70, 80, 90, 100, 300, 500, 700, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
for short_window in short_windows:
    for long_window in long_windows:
        long_short_pairs.append((short_window, long_window))


pct=1/3 #实盘是1/3/50，看趋势图和mr配对用1/3

# fig, axes = plt.subplots(len(long_short_pairs), len(long_short_pairs), figsize=(20, 15))
# fig.suptitle(f'Heatmap of Equity Curve', fontsize=16)
# returns_matrix = np.zeros(len(short_windows),len(long_windows))
# sharp_matrix = np.zeros(len(short_windows),len(long_windows))
start_Date='2021-01-01 00:00:00'#'2025-02-01 07:30:00'
merged_data = assets_df[assets[0]][assets_df[assets[0]]['timestamp']>=start_Date][['timestamp']].copy()
 
merged_data_long_sum = assets_df[assets[0]][assets_df[assets[0]]['timestamp']>=start_Date][['timestamp']].copy()
merged_data_long_sum['Portfolio_Value'] = 0.0
merged_data_long_sum=merged_data_long_sum.reset_index(drop=True)
 
merged_data_short_sum = assets_df[assets[0]][assets_df[assets[0]]['timestamp']>=start_Date][['timestamp']].copy()
merged_data_short_sum['Portfolio_Value'] = 0.0
merged_data_short_sum=merged_data_short_sum.reset_index(drop=True)
total_return =0.0
sharp=0.0
deapct=1/3
from datetime import datetime, timedelta

for pair in long_short_pairs:
    for asset1 in assets:
        short_window = pair[0]
        long_window = pair[1] 
        # print(pair, asset1)

        reference_time = datetime.strptime(start_Date, '%Y-%m-%d %H:%M:%S')

        half_hour = timedelta(minutes=30)
        time_delta = long_window * half_hour
        previous_time =reference_time- time_delta

        asset1_df=assets_df[asset1][assets_df[asset1]['timestamp']>=previous_time].copy()
        total_cash=pct/len(long_short_pairs)*(float(Asset_Total_Cash[Asset_Total_Cash['Asset'] == asset1]['Total_Cash'].iloc[0]) if len(Asset_Total_Cash[Asset_Total_Cash['Asset'] == asset1]['Total_Cash']) > 0 else 0.0)#Asset_Total_Cash[Asset_Total_Cash['Asset']==asset1]['Total_Cash']
        
        # 生成 long 策略和 short 策略的数据 
        data1_long = trend_following_strategy_long(asset1_df, short_window=short_window, long_window=long_window )
        data1_long = backtest_long(data1_long, total_cash / 2,asset1)
        data1_short = trend_following_strategy_short(asset1_df, short_window=short_window, long_window=long_window )
        data1_short = backtest_short(data1_short, total_cash / 2,asset1)
        # 填充空值
        data1_long[0]['Portfolio_Value'] = data1_long[0]['Portfolio_Value'].fillna(0)
        data1_short[0]['Portfolio_Value'] = data1_short[0]['Portfolio_Value'].fillna(0)
        # data1_long[0].to_csv(f'/Users/sigrid/Desktop/向量化/macd趋势跟随改进/曲线图/wmalong{asset1}{short_window}_{long_window}.csv')
        # data1_short[0].to_csv(f'/Users/sigrid/Desktop/向量化/macd趋势跟随改进/曲线图/wmashort{asset1}{short_window}_{long_window}.csv')              
        # 合并 long 和 short 数据
        merged_data_long = pd.merge(merged_data.copy(), data1_long[0][['timestamp', 'Portfolio_Value']], on='timestamp', how='left') 
        merged_data_long['Portfolio_Value'] = merged_data_long['Portfolio_Value'].replace(0, np.nan).bfill() 
        merged_data_long['Portfolio_Value'] =merged_data_long['Portfolio_Value'].fillna(0)  
        # print(merged_data_long.head(), merged_data_long_sum.head())   
        merged_data_long_sum['Portfolio_Value'] = merged_data_long_sum['Portfolio_Value'] + merged_data_long['Portfolio_Value']
        # print( data1_long[0].head() ,merged_data_long.head(),merged_data_long_sum.head())

        merged_data_short = pd.merge(merged_data.copy(), data1_short[0][['timestamp', 'Portfolio_Value']], on='timestamp', how='left') 
        merged_data_short['Portfolio_Value'] = merged_data_short['Portfolio_Value'].replace(0, np.nan).bfill() 
        merged_data_short['Portfolio_Value'] =merged_data_short['Portfolio_Value'].fillna(0)     
        merged_data_short_sum['Portfolio_Value'] = merged_data_short_sum['Portfolio_Value'] + merged_data_short['Portfolio_Value']
        # 计算总收益并转化为百分比
        #if ( sum(data1_short[1])+sum(data1_long[1]))==(merged_data_long_sum['Portfolio_Value'].iloc[-1] -merged_data_long_sum['Portfolio_Value'].iloc[0]+ merged_data_short_sum['Portfolio_Value'].iloc[-1] - merged_data_short_sum['Portfolio_Value'].iloc[0]):#验证
        total_return = total_return+( sum(data1_short[1])+sum(data1_long[1]))  

    return_ratio = total_return/(Asset_Total_Cash.Total_Cash.sum())
    sharp=calculate_sharpe_ratio(merged_data_long_sum['Portfolio_Value'], merged_data_short_sum['Portfolio_Value'])
    print(f"pair:{pair},   return ratio:{return_ratio:.5f},   sharp ratio:{sharp:.5f}")
    # returns_matrix[i, j] = total_return
    # sharp_matrix[i, j] = sharp

# plt.figure(figsize=(10, 8))
# plt.plot(merged_data_long_sum.timestamp,merged_data_long_sum['Portfolio_Value'] - merged_data_long_sum['Portfolio_Value'].iloc[0], label="Long", color="green", linewidth=0.2)
# plt.plot(merged_data_long_sum.timestamp,merged_data_short_sum['Portfolio_Value'] - merged_data_short_sum['Portfolio_Value'].iloc[0]   , label="Short", color="blue", linewidth=0.2)
# plt.plot(merged_data_long_sum.timestamp,merged_data_long_sum['Portfolio_Value'] + merged_data_short_sum['Portfolio_Value']- merged_data_long_sum['Portfolio_Value'].iloc[0] - merged_data_short_sum['Portfolio_Value'].iloc[0]  , label="L&S", color="red", linewidth=0.2)
 

# plt.title(f"ALL assets with dea1/3  short windows macd", fontsize=8)#\nReturn: {total_return:.2f}%
# plt.tick_params(axis="both", which="major", labelsize=6)
# plt.legend(fontsize=6)
 
# plt.show()


