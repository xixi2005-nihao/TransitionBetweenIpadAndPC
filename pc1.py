import os
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from numba import njit

# ====== 全局显示与警告设置 ======
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
warnings.filterwarnings("ignore")

# ====== 资产、参数 ======
assets = ['ADA', 'AVAX', 'BCH', 'BNB', 'BTC', 'DOGE', 'DOT', 'ETH', 'ETC', 'FIL', 'LINK', 'LTC', 'POL', 'NEAR', 'SOL', 'TRX', 'UNI', 'XRP']
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
b = list(itertools.permutations(assets, 2))
print(b)

stepSize = {
    'ADA': 1, 'AVAX': 1, 'BCH': 0.001, 'BNB': 0.01, 'BTC': 0.001, 'DOGE': 1, 'DOT': 0.1, 'ETH': 0.001, 'ETC': 0.01,
    'FIL': 0.1, 'LINK': 0.01, 'LTC': 0.001, 'POL': 1, 'NEAR': 1, 'SOL': 1, 'TRX': 1, 'UNI': 1, 'XRP': 0.1,
    'XLM': 1, 'AAVE': 0.001, 'VET': 0.1, 'ATOM': 0.01, 'ALGO': 1, 'MKR': 0.0001
}

# ====== 预缓存 stepSize 小数位 ======
def _num_decimals_from_step(ss):
    if isinstance(ss, int) or (isinstance(ss, float) and ss.is_integer()):
        return 0
    s = str(ss)
    if '.' in s:
        return len(s.split('.')[-1].rstrip('0'))
    return 0

STEP_DECIMALS = {k: _num_decimals_from_step(v) for k, v in stepSize.items()}

def truncate(value, asset):
    nd = STEP_DECIMALS[asset]
    factor = 10 ** nd
    return int(value * factor) / factor

# ====== 数据载入 ======
assets_df = {}
def round_to_next_minute(timestamp):
    return timestamp.ceil(freq='min').replace(second=0, microsecond=0)

for asset in assets:
    monthly_file = f"F:/Research_pku/趋势跟随wma/2021_01_01/data/{asset}USDT_30m.csv"
    try:
        df = pd.read_csv(monthly_file)
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 
                     'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df['close'] = df['close'].ffill().bfill()
        df['close_avg'] = np.where(
            (df['taker_buy_volume'] != 0) & (df['taker_buy_quote_volume'] != 0),
            df['taker_buy_quote_volume'] / df['taker_buy_volume'],
            df['close']
        )
        assets_df[asset] = df.copy()
    except FileNotFoundError:
        print(f"Monthly file not found: {monthly_file}")
        continue

# ====== 资金表 ======
Asset_Total_Cash = {
    'Asset': ['DOT', 'ADA', 'FIL', 'BTC', 'DOGE', 'NEAR', 'UNI', 'BNB', 'ETH', 'LINK', 'LTC', 'TRX', 'SOL', 'XRP', 'AVAX', 'POL'],
    'Count': [35, 30, 30, 20, 20, 20, 20, 15, 10, 10, 10, 10, 10, 10, 5, 5],
    'Total_Cash': [175000.0, 150000.0, 150000.0, 100000.0, 100000.0, 100000.0, 100000.0, 75000.0, 50000.0, 50000.0, 50000.0, 50000.0, 50000.0, 50000.0, 25000.0, 25000.0]
}
Asset_Total_Cash = pd.DataFrame(Asset_Total_Cash)

# ====== WMA计算 ======
@njit
def wma_linear_reverse(S, n, k):
    weights = np.arange(k + 1)
    m = np.sum(weights)
    window = S[n - k : n + 1]
    return np.sum(weights * window) / m

@njit
def calculate_wma_numba(close_prices, short_window, long_window):
    n = len(close_prices)
    short_wma = np.full(n, np.nan)
    long_wma = np.full(n, np.nan)
    
    for i in range(short_window - 1, n):
        short_wma[i] = wma_linear_reverse(close_prices, i, short_window - 1)
    
    for i in range(long_window - 1, n):
        long_wma[i] = wma_linear_reverse(close_prices, i, long_window - 1)
    
    return short_wma, long_wma

def calculate_wma(data, short_window, long_window):
    close_prices = data['close_avg'].values
    short_wma, long_wma = calculate_wma_numba(close_prices, short_window, long_window)
    data['Short_WMA'] = short_wma
    data['Long_WMA'] = long_wma
    return data

# ====== 夏普比率计算 ======
def calculate_sharpe_ratio(portfolio_value_long, portfolio_value_short, risk_free_rate=0.00):
    initial = portfolio_value_long[0]
    portfolio_value_long = np.array(portfolio_value_long)
    portfolio_value_short = np.array(portfolio_value_short)

    portfolio_value_long = portfolio_value_long - portfolio_value_long[0]
    portfolio_value_short = portfolio_value_short - portfolio_value_short[0]
    
    portfolio_value_sum = (portfolio_value_long + portfolio_value_short + initial)
    if np.any(portfolio_value_sum < 0):
        return 0
        
    portfolio_value_series = pd.Series(portfolio_value_sum)
    returns = portfolio_value_series.pct_change().fillna(0)
    average_return = np.mean(returns) * 365 * 48
    std_return = np.std(returns) * np.sqrt(365 * 48)
    
    if std_return == 0:
        return 0
        
    sharpe_ratio = (average_return - risk_free_rate) / std_return
    return sharpe_ratio

# ====== 策略实现 ======
def trend_following_strategy_long(data, short_window, long_window):
    data = calculate_wma(data, short_window, long_window)
    data['Short_MA'] = data['Short_WMA']
    data['Long_MA'] = data['Long_WMA']

    data['Open_Condition'] = (data['Short_MA'] > data['Long_MA'])
    data['Close_Condition'] = (data['Short_MA'] < data['Long_MA'])

    data['Position'] = np.where(
        data['Open_Condition'], 1,
        np.where(data['Close_Condition'], 0, np.nan)
    )
    data['Position'] = data['Position'].ffill().fillna(0).astype(int)
    data['Signal'] = data['Position'].diff().fillna(0).astype(int)
    
    mask = data['Signal'] == -1
    data.loc[mask, 'Position'] = 1
    
    data['Return'] = data['close'].pct_change()
    data['Strategy_Return'] = data['Position'] * data['Return']

    return data

def trend_following_strategy_short(data, short_window, long_window):
    data = calculate_wma(data, short_window, long_window)
    data['Short_MA'] = data['Short_WMA']
    data['Long_MA'] = data['Long_WMA']

    data['Open_Condition'] = (data['Short_MA'] < data['Long_MA'])
    data['Close_Condition'] = (data['Short_MA'] > data['Long_MA'])

    data['Position'] = np.where(
        data['Open_Condition'], -1,
        np.where(data['Close_Condition'], 0, np.nan)
    )
    data['Position'] = data['Position'].ffill().fillna(0).astype(int)
    data['Signal'] = data['Position'].diff().fillna(0).astype(int)
    
    mask = data['Signal'] == 1
    data.loc[mask, 'Position'] = -1
    
    data['Return'] = data['close'].pct_change()
    data['Strategy_Return'] = data['Position'] * data['Return']

    return data

# ====== 回测加速 ======
@njit
def _scan_take_profit_long(prices, start_idx, end_idx, start_price, qty, initial_cash, profit_threshold, withdraw_threshold):
    open_fee = qty * start_price * 4.5 / 10000.0
    max_profit = 0.0
    reached = False
    final_end = end_idx
    
    # 确保 end_idx 不超过数组边界
    if end_idx >= len(prices):
        end_idx = len(prices) - 1
        final_end = end_idx
    
    for i in range(start_idx + 1, end_idx + 1):  # 包含 end_idx
        if i >= len(prices):  # 确保不越界
            break
            
        current_price = prices[i]
        current_profit = qty * (current_price - start_price) - open_fee
        
        if current_profit > max_profit:
            max_profit = current_profit
            
        if (not reached) and (current_profit >= initial_cash * profit_threshold):
            reached = True
            
        if reached:
            drawdown = max_profit - current_profit
            if max_profit > 0 and (drawdown > max_profit * withdraw_threshold):
                final_end = i
                return final_end, 1  # 触发止盈
                
    return final_end, 0  # 未触发止盈

@njit
def _scan_take_profit_short(prices, start_idx, end_idx, start_price, qty, initial_cash, profit_threshold, withdraw_threshold):
    open_fee = qty * start_price * 4.5 / 10000.0
    max_profit = 0.0
    reached = False
    final_end = end_idx
    
    # 确保 end_idx 不超过数组边界
    if end_idx >= len(prices):
        end_idx = len(prices) - 1
        final_end = end_idx
    
    for i in range(start_idx + 1, end_idx + 1):  # 包含 end_idx
        if i >= len(prices):  # 确保不越界
            break
            
        current_price = prices[i]
        current_profit = qty * (start_price - current_price) - open_fee
        
        if current_profit > max_profit:
            max_profit = current_profit
            
        if (not reached) and (current_profit >= initial_cash * profit_threshold):
            reached = True
            
        if reached:
            drawdown = max_profit - current_profit
            if max_profit > 0 and (drawdown > max_profit * withdraw_threshold):
                final_end = i
                return final_end, 1  # 触发止盈
                
    return final_end, 0  # 未触发止盈

def backtest_long(data, initial_cash_per_trade, asset, profit_threshold, withdraw_threshold):
    data = data.copy().reset_index(drop=True)  # 重置索引确保从0开始
    n = len(data)
    pv = np.zeros(n, dtype=np.float64)
    data['currency'] = asset

    open_idx = data.index[data['Signal'] == 1].to_numpy()
    close_idx = data.index[data['Signal'] == -1].to_numpy()
    open_ts = pd.Index(data.loc[open_idx, 'timestamp'])

    # 确保索引对齐
    if len(close_idx) > len(open_idx) and open_idx[0] > close_idx[0]:
        close_idx = close_idx[1:]
    if len(close_idx) < len(open_idx):
        open_idx = open_idx[:-1]

    prices = data['close'].to_numpy(np.float64)
    trade_profit = []
    actual_close_indices = []
    triggered_list = []  # 记录每次交易是否触发止盈

    for i, start in enumerate(open_idx):
        original_end = close_idx[i] if i < len(close_idx) else n - 1
        # 确保 original_end 不超过数组边界
        original_end = min(original_end, n - 1)
        
        start_price = prices[start]
        qty = truncate(initial_cash_per_trade / start_price, asset)
        pv[start] = initial_cash_per_trade
        open_fee = qty * start_price * 4.5 / 10000.0

        final_end, triggered = _scan_take_profit_long(
            prices, start, original_end, start_price, qty, 
            initial_cash_per_trade, profit_threshold, withdraw_threshold
        )
        
        # 确保 final_end 不超过数组边界
        final_end = min(final_end, n - 1)
        
        end_price = prices[final_end]
        close_fee = qty * end_price * 4.5 / 10000.0
        total_profit = qty * (end_price - start_price) - (open_fee + close_fee)
        final_value = initial_cash_per_trade + total_profit

        if final_end - start > 1:
            # 确保索引不越界
            end_range = min(final_end, n)
            pv[start+1:end_range] = (
                initial_cash_per_trade + 
                qty * (prices[start+1:end_range] - start_price) - 
                open_fee
            )
        pv[final_end] = final_value

        trade_profit.append(total_profit)
        actual_close_indices.append(final_end)
        triggered_list.append(triggered)  # 记录是否触发止盈

    if len(open_idx) > len(actual_close_indices):
        start = open_idx[-1]
        # 确保 start 不超过数组边界
        if start < n:
            start_price = prices[start]
            qty = truncate(initial_cash_per_trade / start_price, asset)
            open_fee = qty * start_price * 4.5 / 10000.0
            pv[start] = initial_cash_per_trade
            if start + 1 < n:
                pv[start+1:] = (
                    initial_cash_per_trade + 
                    qty * (prices[start+1:] - start_price) - 
                    open_fee
                )

    pv = pd.Series(pv).replace(0, np.nan).ffill().bfill().fillna(0).to_numpy()
    cum_pv = pv.copy()
    for i, start in enumerate(open_idx[1:]):
        if i < len(trade_profit):
            # 确保索引不越界
            start = min(start, n - 1)
            cum_pv[start:] += trade_profit[i]

    data['Portfolio_Value'] = cum_pv
    data['Buy_and_Hold_Value'] = (1 + data['Return']).cumprod() * initial_cash_per_trade

    close_ts = pd.Index(data.loc[actual_close_indices, 'timestamp']) if actual_close_indices else pd.Index([])
    return data, trade_profit, open_ts, close_ts, triggered_list  # 返回是否触发止盈的列表

def backtest_short(data, initial_cash_per_trade, asset, profit_threshold, withdraw_threshold):
    data = data.copy().reset_index(drop=True)  # 重置索引确保从0开始
    n = len(data)
    pv = np.zeros(n, dtype=np.float64)
    data['currency'] = asset

    open_idx = data.index[data['Signal'] == -1].to_numpy()
    close_idx = data.index[data['Signal'] == 1].to_numpy()
    open_ts = pd.Index(data.loc[open_idx, 'timestamp'])

    # 确保索引对齐
    if len(close_idx) > len(open_idx) and open_idx[0] > close_idx[0]:
        close_idx = close_idx[1:]
    if len(close_idx) < len(open_idx):
        open_idx = open_idx[:-1]

    prices = data['close'].to_numpy(np.float64)
    trade_profit = []
    actual_close_indices = []
    triggered_list = []  # 记录每次交易是否触发止盈

    for i, start in enumerate(open_idx):
        original_end = close_idx[i] if i < len(close_idx) else n - 1
        # 确保 original_end 不超过数组边界
        original_end = min(original_end, n - 1)
        
        start_price = prices[start]
        qty = truncate(initial_cash_per_trade / start_price, asset)
        pv[start] = initial_cash_per_trade
        open_fee = qty * start_price * 4.5 / 10000.0

        final_end, triggered = _scan_take_profit_short(
            prices, start, original_end, start_price, qty, 
            initial_cash_per_trade, profit_threshold, withdraw_threshold
        )
        
        # 确保 final_end 不超过数组边界
        final_end = min(final_end, n - 1)
        
        end_price = prices[final_end]
        close_fee = qty * end_price * 4.5 / 10000.0
        total_profit = qty * (start_price - end_price) - (open_fee + close_fee)
        final_value = initial_cash_per_trade + total_profit

        if final_end - start > 1:
            # 确保索引不越界
            end_range = min(final_end, n)
            pv[start+1:end_range] = (
                initial_cash_per_trade + 
                qty * (start_price - prices[start+1:end_range]) - 
                open_fee
            )
        pv[final_end] = final_value

        trade_profit.append(total_profit)
        actual_close_indices.append(final_end)
        triggered_list.append(triggered)  # 记录是否触发止盈

    if len(open_idx) > len(actual_close_indices):
        start = open_idx[-1]
        # 确保 start 不超过数组边界
        if start < n:
            start_price = prices[start]
            qty = truncate(initial_cash_per_trade / start_price, asset)
            open_fee = qty * start_price * 4.5 / 10000.0
            pv[start] = initial_cash_per_trade
            if start + 1 < n:
                pv[start+1:] = (
                    initial_cash_per_trade + 
                    qty * (start_price - prices[start+1:]) - 
                    open_fee
                )

    pv = pd.Series(pv).replace(0, np.nan).ffill().bfill().fillna(0).to_numpy()
    cum_pv = pv.copy()
    for i, start in enumerate(open_idx[1:]):
        if i < len(trade_profit):
            # 确保索引不越界
            start = min(start, n - 1)
            cum_pv[start:] += trade_profit[i]

    data['Portfolio_Value'] = cum_pv
    data['Buy_and_Hold_Value'] = (1 + data['Return']).cumprod() * initial_cash_per_trade

    close_ts = pd.Index(data.loc[actual_close_indices, 'timestamp']) if actual_close_indices else pd.Index([])
    return data, trade_profit, open_ts, close_ts, triggered_list  # 返回是否触发止盈的列表

# ====== 主流程 ======
minutes = 30
long_short_pairs = [(40, 50), (30, 60), (50, 500), (10, 1500), (150, 700), (30, 500), (40, 1000), (80, 1000), (20, 1500), (100, 700)]
pct = 1/3
start_Date = '2022-01-01 00:00:00'

# 统一时间轴
base_timestamps = assets_df[assets[0]][assets_df[assets[0]]['timestamp'] >= start_Date]['timestamp'].copy().reset_index(drop=True)
merged_data = pd.DataFrame({'timestamp': base_timestamps})

# 使用 numpy 数组做加总容器
merged_long_sum = np.zeros(len(merged_data), dtype=np.float64)
merged_short_sum = np.zeros(len(merged_data), dtype=np.float64)

# profit_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
# withdraw_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
profit_thresholds = [0.6, 0.7, 0.8, 0.9, 1.0]
withdraw_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


for profit_threshold in profit_thresholds:
    for withdraw_threshold in withdraw_thresholds:
        total_return = 0.0
        merged_long_sum.fill(0.0)
        merged_short_sum.fill(0.0)
        total_trades = 0
        triggered_trades = 0

        for asset1 in Asset_Total_Cash.Asset.tolist():
            for pair in long_short_pairs:
                short_window, long_window = pair
                # print(f"Processing {asset1}, {short_window}, {long_window}, Profit: {profit_threshold}, Withdraw: {withdraw_threshold}")

                reference_time = datetime.strptime(start_Date, '%Y-%m-%d %H:%M:%S')
                half_hour = timedelta(minutes=30)
                time_delta = long_window * half_hour
                previous_time = reference_time - time_delta

                asset1_df = assets_df[asset1][assets_df[asset1]['timestamp'] >= previous_time].copy()
                total_cash = pct / len(long_short_pairs) * float(Asset_Total_Cash[Asset_Total_Cash['Asset'] == asset1]['Total_Cash'].iloc[0])

                # 生成策略
                data1_long = trend_following_strategy_long(asset1_df, short_window=short_window, long_window=long_window)
                data1_long_result = backtest_long(data1_long, total_cash / 2, asset1, profit_threshold, withdraw_threshold)

                data1_short = trend_following_strategy_short(asset1_df, short_window=short_window, long_window=long_window)
                data1_short_result = backtest_short(data1_short, total_cash / 2, asset1, profit_threshold, withdraw_threshold)

                # 对齐到基准时间轴
                l_df = data1_long_result[0].set_index('timestamp').reindex(base_timestamps)
                l_pv = l_df['Portfolio_Value'].replace(0, np.nan).bfill().fillna(0).to_numpy(np.float64)
                
                s_df = data1_short_result[0].set_index('timestamp').reindex(base_timestamps)
                s_pv = s_df['Portfolio_Value'].replace(0, np.nan).bfill().fillna(0).to_numpy(np.float64)
                
                merged_long_sum += l_pv
                merged_short_sum += s_pv
                
                total_return += (sum(data1_short_result[1]) + sum(data1_long_result[1]))
                
                # 统计止盈触发情况
                total_trades += (len(data1_long_result[4]) + len(data1_short_result[4]))
                triggered_trades += (sum(data1_long_result[4]) + sum(data1_short_result[4]))

        # 计算止盈触发概率
        if total_trades > 0:
            trigger_probability = triggered_trades / total_trades
        else:
            trigger_probability = 0
            
        return_ratio = total_return / Asset_Total_Cash.Total_Cash.sum()
        sharpe_ratio = calculate_sharpe_ratio(merged_long_sum, merged_short_sum)
        
        print(f'Profit_threshold:{profit_threshold}   Withdraw_threshold:{withdraw_threshold}   '
              f'Return_ratio:{return_ratio:.5f}   Sharpe ratio:{sharpe_ratio:.5f}   '
              f'Trigger Probability:{trigger_probability:.4f}   Total Trades:{total_trades}   Triggered Trades:{triggered_trades}')

# # 画图
# plt.figure(figsize=(18, 8))
# plt.plot(base_timestamps, merged_long_sum - merged_long_sum[0], label="Long", color="green", linewidth=0.2)
# plt.plot(base_timestamps, merged_short_sum - merged_short_sum[0], label="Short", color="blue", linewidth=0.2)
# plt.plot(base_timestamps, merged_long_sum + merged_short_sum - merged_long_sum[0] - merged_short_sum[0], 
#          label="L&S", color="red", linewidth=0.2)

# plt.title(f"ALL assets wma 利润超过15%后回撤10%平仓short", fontsize=8)
# plt.tick_params(axis="both", which="major", labelsize=6)
# plt.legend(fontsize=6)
# plt.show()

# sharp = calculate_sharpe_ratio(merged_long_sum, merged_short_sum)
# print('sharpe ratio:', sharp)
# print('total return:', total_return)
