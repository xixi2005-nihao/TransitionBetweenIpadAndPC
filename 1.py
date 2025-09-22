import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
import time
import os
import itertools  
from numba import jit, float64, int64, void, types
import numba

# 展示全部dataframe，忽略警告
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None) 
import warnings
warnings.filterwarnings("ignore") 

# initial_cash=10000
assets =[  'ADA', 'AVAX', 'BCH', 'BNB','BTC','DOGE', 'DOT', 'ETH', 'ETC', 'FIL', 'LINK', 'LTC', 'POL', 'NEAR', 'SOL', 'TRX', 'UNI', 'XRP']#,'XLM', 'AAVE', 'VET', 'ATOM','ALGO','MKR']
# assets = ['ADA']
months=[ '01', '02', '03', '04', '05', '06', '07', '08', '09','10', '11', '12']

#交易规则https://www.binance.com/zh-CN/futures/trading-rules/perpetual
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
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df['close'] = df['close'].ffill().bfill()  # 先向前，再向后填充
        df['close_avg'] = np.where((df['taker_buy_volume'] != 0 ) & ( df['taker_buy_quote_volume']!=0) ,
            df['taker_buy_quote_volume'] / df['taker_buy_volume'], df['close'] )

        assets_df[asset]=df.copy() 
        
    except FileNotFoundError:
        print(f"Monthly file not found: {monthly_file}")
        continue

Asset_Total_Cash = {
    'Asset': ['DOT', 'ADA', 'FIL', 'BTC', 'DOGE', 'NEAR', 'UNI', 'BNB', 'ETH', 'LINK', 'LTC', 'TRX', 'SOL', 'XRP', 'AVAX', 'POL'],
    'Count': [35, 30, 30, 20, 20, 20, 20, 15, 10, 10, 10, 10, 10, 10, 5, 5],
    'Total_Cash': [175000.0, 150000.0, 150000.0, 100000.0, 100000.0, 100000.0, 100000.0, 75000.0, 50000.0, 50000.0, 50000.0, 50000.0, 50000.0, 50000.0, 25000.0, 25000.0]
}
Asset_Total_Cash = pd.DataFrame(Asset_Total_Cash)

# ============================ NUMBA 加速函数 ============================

@jit(nopython=True)
def get_num_decimals_numba(step_size):
    """使用Numba加速的小数位数计算"""
    if step_size >= 1.0:
        return 0
    step_str = f"{step_size:.10f}"
    # 移除末尾的0
    step_str = step_str.rstrip('0')
    if '.' in step_str:
        return len(step_str.split('.')[1])
    return 0

@jit(nopython=True)
def truncate_numba(value, step_size):
    """使用Numba加速的截断函数"""
    if step_size >= 1.0:
        return int(value)
    else:
        decimals = get_num_decimals_numba(step_size)
        factor = 10.0 ** decimals
        return int(value * factor) / factor

@jit(nopython=True)
def wma_linear_reverse_numba(S, n, k):
    """使用Numba加速的WMA计算"""
    weights = np.arange(k + 1, dtype=np.float64)
    m = np.sum(weights)
    window = S[n - k:n + 1]
    return np.sum(weights * window) / m

@jit(nopython=True)
def calculate_wma_numba(close_prices, short_window, long_window):
    """使用Numba加速的WMA计算"""
    n = len(close_prices)
    short_wma = np.full(n, np.nan, dtype=np.float64)
    long_wma = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(n):
        if i >= short_window - 1:
            short_wma[i] = wma_linear_reverse_numba(close_prices, i, short_window - 1)
        if i >= long_window - 1:
            long_wma[i] = wma_linear_reverse_numba(close_prices, i, long_window - 1)
    
    return short_wma, long_wma

@jit(nopython=True)
def generate_signals_long_numba(short_ma, long_ma):
    """使用Numba加速的长策略信号生成"""
    n = len(short_ma)
    position = np.zeros(n, dtype=np.int64)
    signal = np.zeros(n, dtype=np.int64)
    
    for i in range(1, n):
        if not np.isnan(short_ma[i]) and not np.isnan(long_ma[i]):
            # 开仓条件：短期均线上穿长期均线
            if short_ma[i] > long_ma[i] and position[i-1] == 0:
                position[i] = 1
                signal[i] = 1
            # 平仓条件：短期均线下穿长期均线
            elif short_ma[i] < long_ma[i] and position[i-1] == 1:
                position[i] = 0
                signal[i] = -1
            else:
                position[i] = position[i-1]
        else:
            position[i] = position[i-1]
    
    return position, signal

@jit(nopython=True)
def generate_signals_short_numba(short_ma, long_ma):
    """使用Numba加速的短策略信号生成"""
    n = len(short_ma)
    position = np.zeros(n, dtype=np.int64)
    signal = np.zeros(n, dtype=np.int64)
    
    for i in range(1, n):
        if not np.isnan(short_ma[i]) and not np.isnan(long_ma[i]):
            # 开仓条件：短期均线下穿长期均线
            if short_ma[i] < long_ma[i] and position[i-1] == 0:
                position[i] = -1
                signal[i] = -1
            # 平仓条件：短期均线上穿长期均线
            elif short_ma[i] > long_ma[i] and position[i-1] == -1:
                position[i] = 0
                signal[i] = 1
            else:
                position[i] = position[i-1]
        else:
            position[i] = position[i-1]
    
    return position, signal

@jit(nopython=True)
def backtest_long_numba(close_prices, signals, initial_cash, step_size, fee_rate=0.0002):
    """使用Numba加速的长策略回测"""
    n = len(close_prices)
    portfolio_value = np.zeros(n, dtype=np.float64)
    position = 0  # 0: 无持仓, 1: 多头
    entry_price = 0.0
    cash = initial_cash
    qty = 0.0
    
    for i in range(n):
        if signals[i] == 1 and position == 0:  # 开多头
            qty = truncate_numba(cash / close_prices[i], step_size)
            fee = qty * close_prices[i] * fee_rate
            cash -= fee
            position = 1
            entry_price = close_prices[i]
            portfolio_value[i] = cash
            
        elif signals[i] == -1 and position == 1:  # 平多头
            profit = qty * (close_prices[i] - entry_price)
            fee = qty * close_prices[i] * fee_rate
            cash += profit - fee
            position = 0
            portfolio_value[i] = cash
            
        elif position == 1:  # 持仓中
            unrealized_profit = qty * (close_prices[i] - entry_price)
            portfolio_value[i] = cash + unrealized_profit
            
        else:  # 无持仓
            portfolio_value[i] = cash
    
    return portfolio_value

@jit(nopython=True)
def backtest_short_numba(close_prices, signals, initial_cash, step_size, fee_rate=0.0002):
    """使用Numba加速的短策略回测"""
    n = len(close_prices)
    portfolio_value = np.zeros(n, dtype=np.float64)
    position = 0  # 0: 无持仓, -1: 空头
    entry_price = 0.0
    cash = initial_cash
    qty = 0.0
    
    for i in range(n):
        if signals[i] == -1 and position == 0:  # 开空头
            qty = truncate_numba(cash / close_prices[i], step_size)
            fee = qty * close_prices[i] * fee_rate
            cash -= fee
            position = -1
            entry_price = close_prices[i]
            portfolio_value[i] = cash
            
        elif signals[i] == 1 and position == -1:  # 平空头
            profit = qty * (entry_price - close_prices[i])
            fee = qty * close_prices[i] * fee_rate
            cash += profit - fee
            position = 0
            portfolio_value[i] = cash
            
        elif position == -1:  # 持仓中
            unrealized_profit = qty * (entry_price - close_prices[i])
            portfolio_value[i] = cash + unrealized_profit
            
        else:  # 无持仓
            portfolio_value[i] = cash
    
    return portfolio_value

@jit(nopython=True)
def calculate_sharpe_ratio_numba(portfolio_value_long, portfolio_value_short, risk_free_rate=0.00):
    """使用Numba加速的夏普比率计算"""
    initial = portfolio_value_long[0]
    n = len(portfolio_value_long)
    
    # 计算组合总价值
    portfolio_value_sum = np.zeros(n, dtype=np.float64)
    for i in range(n):
        portfolio_value_sum[i] = portfolio_value_long[i] + portfolio_value_short[i] - initial
    
    # 检查是否有负值
    if np.any(portfolio_value_sum < 0):
        return 0.0
    
    # 计算收益率
    returns = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if portfolio_value_sum[i-1] != 0:
            returns[i] = (portfolio_value_sum[i] - portfolio_value_sum[i-1]) / portfolio_value_sum[i-1]
        else:
            returns[i] = 0.0
    
    # 计算年化收益率和标准差
    average_return = np.mean(returns) * 365 * 48
    std_return = np.std(returns) * np.sqrt(365 * 48)
    
    # 计算夏普比率
    if std_return == 0:
        return 0.0
    sharpe_ratio = (average_return - risk_free_rate) / std_return
    return sharpe_ratio

# ============================ 包装函数 ============================

def precompile_numba_functions():
    """预编译Numba函数以避免第一次运行时的编译开销"""
    # 创建测试数据
    test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    test_signals = np.array([0, 1, 0, -1, 0], dtype=np.float64)
    
    # 预编译所有Numba函数
    wma_linear_reverse_numba(test_data, 2, 1)
    calculate_wma_numba(test_data, 3, 5)
    generate_signals_long_numba(test_data[:3], test_data[:3])
    generate_signals_short_numba(test_data[:3], test_data[:3])
    backtest_long_numba(test_data, test_signals, 1000.0, 0.01)
    backtest_short_numba(test_data, test_signals, 1000.0, 0.01)
    calculate_sharpe_ratio_numba(test_data, test_data)
    truncate_numba(1.2345, 0.01)

# 在程序开始时预编译
print("预编译Numba函数...")
precompile_numba_functions()
print("Numba函数预编译完成")

def calculate_wma(data, short_window, long_window):
    """包装函数，使用Numba加速版本"""
    close_prices = data['close_avg'].fillna(method='ffill').fillna(method='bfill').values.astype(np.float64)
    short_wma, long_wma = calculate_wma_numba(close_prices, short_window, long_window)
    
    data = data.copy()
    data['Short_WMA'] = short_wma
    data['Long_WMA'] = long_wma
    return data

def trend_following_strategy_long(data, short_window, long_window):
    """使用Numba加速的长策略"""
    wma = calculate_wma(data, short_window, long_window)
    
    # 使用Numba加速信号生成
    short_ma = wma['Short_WMA'].values.astype(np.float64)
    long_ma = wma['Long_WMA'].values.astype(np.float64)
    position, signal = generate_signals_long_numba(short_ma, long_ma)
    
    data = data.copy()
    data['Short_MA'] = short_ma
    data['Long_MA'] = long_ma
    data['Position'] = position
    data['Signal'] = signal
    
    # 计算收益
    data['Return'] = data['close'].pct_change().fillna(0)
    data['Strategy_Return'] = data['Position'] * data['Return']
    
    return data

def trend_following_strategy_short(data, short_window, long_window):
    """使用Numba加速的短策略"""
    wma = calculate_wma(data, short_window, long_window)
    
    # 使用Numba加速信号生成
    short_ma = wma['Short_WMA'].values.astype(np.float64)
    long_ma = wma['Long_WMA'].values.astype(np.float64)
    position, signal = generate_signals_short_numba(short_ma, long_ma)
    
    data = data.copy()
    data['Short_MA'] = short_ma
    data['Long_MA'] = long_ma
    data['Position'] = position
    data['Signal'] = signal
    
    # 计算收益
    data['Return'] = data['close'].pct_change().fillna(0)
    data['Strategy_Return'] = data['Position'] * data['Return']
    
    return data

def backtest_long(data, initial_cash_per_trade, asset):
    """使用Numba加速的长回测"""
    # 使用Numba加速的核心计算
    close_prices = data['close'].fillna(method='ffill').fillna(method='bfill').values.astype(np.float64)
    signals = data['Signal'].values.astype(np.float64)
    step_size_val = stepSize[asset]
    
    portfolio_values = backtest_long_numba(close_prices, signals, initial_cash_per_trade, step_size_val)
    
    data = data.copy()
    data['Portfolio_Value'] = portfolio_values
    data['currency'] = asset
    
    # 计算Buy and Hold策略
    data['Buy_and_Hold_Value'] = (1 + data['Return']).cumprod() * initial_cash_per_trade
    
    # 计算交易利润
    open_indices = np.where(signals == 1)[0]
    close_indices = np.where(signals == -1)[0]
    trade_profit = []
    
    for start, end in zip(open_indices, close_indices):
        if end < len(portfolio_values):
            trade_profit.append(portfolio_values[end] - initial_cash_per_trade)
    
    return data, trade_profit, open_indices, close_indices

def backtest_short(data, initial_cash_per_trade, asset):
    """使用Numba加速的短回测"""
    close_prices = data['close'].fillna(method='ffill').fillna(method='bfill').values.astype(np.float64)
    signals = data['Signal'].values.astype(np.float64)
    step_size_val = stepSize[asset]
    
    portfolio_values = backtest_short_numba(close_prices, signals, initial_cash_per_trade, step_size_val)
    
    data = data.copy()
    data['Portfolio_Value'] = portfolio_values
    data['currency'] = asset
    data['Buy_and_Hold_Value'] = (1 + data['Return']).cumprod() * initial_cash_per_trade
    
    # 计算交易利润
    open_indices = np.where(signals == -1)[0]
    close_indices = np.where(signals == 1)[0]
    trade_profit = []
    
    for start, end in zip(open_indices, close_indices):
        if end < len(portfolio_values):
            trade_profit.append(portfolio_values[end] - initial_cash_per_trade)
    
    return data, trade_profit, open_indices, close_indices

def get_num_decimals(step_size):
    """原始的小数位数计算函数（用于非Numba环境）"""
    if isinstance(step_size, int) or step_size.is_integer():
        return 0
    return len(str(step_size).split('.')[-1])

def truncate(value, step_size):
    """原始的截断函数（用于非Numba环境）"""
    num_decimals = get_num_decimals(step_size)
    factor = 10 ** num_decimals
    result = int(value * factor) / factor
    return result

# ============================ 主程序 ============================

minutes = 30
long_short_pairs = []
short_windows = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 500, 1000]
long_windows = [30, 40, 50, 60, 70, 80, 90, 100, 300, 500, 700, 1000, 1500, 2000, 2500, 3000, 3500, 4000]

for short_window in short_windows:
    for long_window in long_windows:
        if short_window < long_window:  # 确保短期窗口小于长期窗口
            long_short_pairs.append((short_window, long_window))

print(f"总共有 {len(long_short_pairs)} 个窗口组合需要测试")

pct = 1/3  # 实盘是1/3/50，看趋势图和mr配对用1/3

start_Date = '2021-01-01 00:00:00'

# 准备合并数据
merged_data = assets_df[assets[0]][assets_df[assets[0]]['timestamp'] >= start_Date][['timestamp']].copy()
 
merged_data_long_sum = assets_df[assets[0]][assets_df[assets[0]]['timestamp'] >= start_Date][['timestamp']].copy()
merged_data_long_sum['Portfolio_Value'] = 0.0
merged_data_long_sum = merged_data_long_sum.reset_index(drop=True)
 
merged_data_short_sum = assets_df[assets[0]][assets_df[assets[0]]['timestamp'] >= start_Date][['timestamp']].copy()
merged_data_short_sum['Portfolio_Value'] = 0.0
merged_data_short_sum = merged_data_short_sum.reset_index(drop=True)

# 用于存储结果的列表
results = []

print("开始回测...")
start_time = time.time()

for pair_idx, pair in enumerate(long_short_pairs):
    short_window, long_window = pair
    
    # 重置合并数据
    merged_data_long_sum_current = merged_data_long_sum.copy()
    merged_data_short_sum_current = merged_data_short_sum.copy()
    
    total_return = 0.0
    
    for asset1 in assets:
        reference_time = datetime.strptime(start_Date, '%Y-%m-%d %H:%M:%S')
        half_hour = timedelta(minutes=30)
        time_delta = long_window * half_hour
        previous_time = reference_time - time_delta

        asset1_df = assets_df[asset1][assets_df[asset1]['timestamp'] >= previous_time].copy()
        
        # 获取该资产的总资金
        asset_cash_info = Asset_Total_Cash[Asset_Total_Cash['Asset'] == asset1]
        if len(asset_cash_info) > 0:
            total_cash = pct / len(long_short_pairs) * float(asset_cash_info['Total_Cash'].iloc[0])
        else:
            total_cash = 0.0
        
        if total_cash > 0:
            # 生成 long 策略和 short 策略的数据 
            data1_long = trend_following_strategy_long(asset1_df, short_window=short_window, long_window=long_window)
            data1_long_result = backtest_long(data1_long[0] if isinstance(data1_long, tuple) else data1_long, 
                                            total_cash / 2, asset1)
            
            data1_short = trend_following_strategy_short(asset1_df, short_window=short_window, long_window=long_window)
            data1_short_result = backtest_short(data1_short[0] if isinstance(data1_short, tuple) else data1_short, 
                                              total_cash / 2, asset1)
            
            # 合并 long 和 short 数据
            merged_data_long = pd.merge(merged_data.copy(), 
                                      data1_long_result[0][['timestamp', 'Portfolio_Value']], 
                                      on='timestamp', how='left') 
            merged_data_long['Portfolio_Value'] = merged_data_long['Portfolio_Value'].fillna(0)
            
            merged_data_short = pd.merge(merged_data.copy(), 
                                       data1_short_result[0][['timestamp', 'Portfolio_Value']], 
                                       on='timestamp', how='left') 
            merged_data_short['Portfolio_Value'] = merged_data_short['Portfolio_Value'].fillna(0)
            
            # 累加到当前组合
            merged_data_long_sum_current['Portfolio_Value'] += merged_data_long['Portfolio_Value']
            merged_data_short_sum_current['Portfolio_Value'] += merged_data_short['Portfolio_Value']
            
            # 累加收益
            if len(data1_long_result[1]) > 0:
                total_return += sum(data1_long_result[1])
            if len(data1_short_result[1]) > 0:
                total_return += sum(data1_short_result[1])
    
    # 计算夏普比率
    sharp = calculate_sharpe_ratio_numba(
        merged_data_long_sum_current['Portfolio_Value'].values.astype(np.float64),
        merged_data_short_sum_current['Portfolio_Value'].values.astype(np.float64)
    )
    
    return_ratio = total_return / Asset_Total_Cash['Total_Cash'].sum()
    
    results.append({
        'short_window': short_window,
        'long_window': long_window,
        'return_ratio': return_ratio,
        'sharp_ratio': sharp,
        'total_return': total_return
    })
    
    # 每10个组合打印一次进度
    if (pair_idx + 1) % 10 == 0 or (pair_idx + 1) == len(long_short_pairs):
        elapsed_time = time.time() - start_time
        remaining_pairs = len(long_short_pairs) - (pair_idx + 1)
        estimated_remaining = elapsed_time / (pair_idx + 1) * remaining_pairs if pair_idx > 0 else 0
        
        print(f"进度: {pair_idx + 1}/{len(long_short_pairs)} "
              f"({(pair_idx + 1)/len(long_short_pairs)*100:.1f}%) - "
              f"已用时间: {elapsed_time/60:.1f}分钟 - "
              f"预计剩余: {estimated_remaining/60:.1f}分钟")
        
        # 打印当前组合结果
        current_result = results[-1]
        print(f"组合 {pair_idx + 1}: short={current_result['short_window']}, "
              f"long={current_result['long_window']}, "
              f"return={current_result['return_ratio']:.5f}, "
              f"sharp={current_result['sharp_ratio']:.5f}")

# 将结果转换为DataFrame
results_df = pd.DataFrame(results)

# 找到最佳组合
best_return = results_df.loc[results_df['return_ratio'].idxmax()]
best_sharp = results_df.loc[results_df['sharp_ratio'].idxmax()]

print("\n" + "="*50)
print("回测完成!")
print(f"总耗时: {(time.time() - start_time)/60:.2f} 分钟")
print(f"测试组合数量: {len(long_short_pairs)}")
print(f"资产数量: {len(assets)}")
print(f"总数据点: {len(merged_data)}")

print("\n最佳收益组合:")
print(f"短期窗口: {best_return['short_window']}, 长期窗口: {best_return['long_window']}")
print(f"收益率: {best_return['return_ratio']:.5f}")
print(f"夏普比率: {best_return['sharp_ratio']:.5f}")

print("\n最佳夏普比率组合:")
print(f"短期窗口: {best_sharp['short_window']}, 长期窗口: {best_sharp['long_window']}")
print(f"收益率: {best_sharp['return_ratio']:.5f}")
print(f"夏普比率: {best_sharp['sharp_ratio']:.5f}")

# 保存结果到CSV
results_df.to_csv('numba_optimized_backtest_results.csv', index=False)
print("\n结果已保存到: numba_optimized_backtest_results.csv")

# 可选：绘制最佳组合的权益曲线
if len(results) > 0:
    # 重新运行最佳组合以获取详细数据
    best_short, best_long = best_sharp['short_window'], best_sharp['long_window']
    
    # 这里可以添加绘制权益曲线的代码
    # 由于时间关系，这部分可以根据需要自行添加

print("程序执行完毕!")
