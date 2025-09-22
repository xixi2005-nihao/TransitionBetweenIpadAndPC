# ==========================================================
#  0. 环境 & 参数
# ==========================================================
import os
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from numba import njit

# 全局显示与警告
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
warnings.filterwarnings("ignore")

# 资产列表
assets = ['ADA', 'AVAX', 'BCH', 'BNB', 'BTC', 'DOGE',
          'DOT', 'ETH', 'ETC', 'FIL', 'LINK', 'LTC', 'POL',
          'NEAR', 'SOL', 'TRX', 'UNI', 'XRP']

# 交易粒度
stepSize = {
    'ADA': 1, 'AVAX': 1, 'BCH': 0.001, 'BNB': 0.01, 'BTC': 0.001,
    'DOGE': 1, 'DOT': 0.1, 'ETH': 0.001, 'ETC': 0.01,
    'FIL': 0.1, 'LINK': 0.01, 'LTC': 0.001, 'POL': 1, 'NEAR': 1,
    'SOL': 1, 'TRX': 1, 'UNI': 1, 'XRP': 0.1
}

# ==========================================================
#  1. 工具函数
# ==========================================================
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

def round_to_next_minute(timestamp):
    return timestamp.ceil(freq='min').replace(second=0, microsecond=0)

# ==========================================================
#  2. 读数据
# ==========================================================
assets_df = {}
for asset in assets:
    f = f"F:/Research_pku/趋势跟随wma/2021_01_01/data/{asset}USDT_30m.csv"
    try:
        df = pd.read_csv(f)
        df.columns = ['timestamp', 'open', 'high', 'low', 'close',
                      'volume', 'close_time', 'quote_volume', 'count',
                      'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close'] = df['close'].ffill().bfill()
        df['close_avg'] = np.where(
            (df['taker_buy_volume'] != 0) & (df['taker_buy_quote_volume'] != 0),
            df['taker_buy_quote_volume'] / df['taker_buy_volume'],
            df['close']
        )
        assets_df[asset] = df.copy()
    except FileNotFoundError:
        print(f"Monthly file not found: {f}")

# ==========================================================
#  3. 资金配置
# ==========================================================
Asset_Total_Cash = pd.DataFrame({
    'Asset': ['DOT', 'ADA', 'FIL', 'BTC', 'DOGE', 'NEAR',
              'UNI', 'BNB', 'ETH', 'LINK', 'LTC', 'TRX', 'SOL', 'XRP',
              'AVAX', 'POL'],
    'Count': [35, 30, 30, 20, 20, 20, 20, 15, 10, 10, 10, 10, 10, 10, 5, 5],
    'Total_Cash': [175000.0, 150000.0, 150000.0, 100000.0, 100000.0,
                   100000.0, 100000.0, 75000.0, 50000.0, 50000.0,
                   50000.0, 50000.0, 50000.0, 50000.0, 25000.0, 25000.0]
})

# ==========================================================
#  4. WMA 计算（Numba 加速）
# ==========================================================
@njit
def wma_linear_reverse(S, n, k):
    weights = np.arange(k + 1)
    m = np.sum(weights)
    window = S[n - k: n + 1]
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

# ==========================================================
#  5. 策略信号
# ==========================================================
def trend_following_strategy_long(data, short_window, long_window):
    data = calculate_wma(data, short_window, long_window)
    data['Short_MA'] = data['Short_WMA']
    data['Long_MA'] = data['Long_WMA']
    data['Open_Condition'] = data['Short_MA'] > data['Long_MA']
    data['Close_Condition'] = data['Short_MA'] < data['Long_MA']
    data['Position'] = np.where(
        data['Open_Condition'], 1,
        np.where(data['Close_Condition'], 0, np.nan)
    )
    data['Position'] = data['Position'].ffill().fillna(0).astype(int)
    data['Signal'] = data['Position'].diff().fillna(0).astype(int)
    # 保证平仓那一根 K 线仍算收益
    mask = data['Signal'] == -1
    data.loc[mask, 'Position'] = 1
    data['Return'] = data['close'].pct_change()
    data['Strategy_Return'] = data['Position'] * data['Return']
    return data

def trend_following_strategy_short(data, short_window, long_window):
    data = calculate_wma(data, short_window, long_window)
    data['Short_MA'] = data['Short_WMA']
    data['Long_MA'] = data['Long_WMA']
    data['Open_Condition'] = data['Short_MA'] < data['Long_MA']
    data['Close_Condition'] = data['Short_MA'] > data['Long_MA']
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

# ==========================================================
#  6. 回测核心（Numba 加速）
# ==========================================================
@njit
def _backtest_long_numba(prices, signals, initial_cash_per_trade, asset_step_inv, fee_rate):
    n = len(prices)
    pv = np.zeros(n)
    trade_profit = []
    open_idx = np.where(signals == 1)[0]
    close_idx = np.where(signals == -1)[0]

    # 对齐
    if len(close_idx) > len(open_idx) and open_idx[0] > close_idx[0]:
        close_idx = close_idx[1:]
    if len(close_idx) < len(open_idx):
        open_idx = open_idx[:-1]

    for i in range(len(open_idx)):
        start = open_idx[i]
        end = close_idx[i] if i < len(close_idx) else n - 1
        start_price = prices[start]
        qty = int((initial_cash_per_trade / start_price) * asset_step_inv) / asset_step_inv
        open_fee = qty * start_price * fee_rate
        pv[start] = initial_cash_per_trade

        for j in range(start + 1, end + 1):
            pv[j] = initial_cash_per_trade + qty * (prices[j] - start_price) - open_fee
            if j == end:
                close_fee = qty * prices[j] * fee_rate
                pv[j] -= close_fee
                trade_profit.append(pv[j] - initial_cash_per_trade)

    # 最后一笔未平仓
    if len(open_idx) > len(close_idx):
        start = open_idx[-1]
        start_price = prices[start]
        qty = int((initial_cash_per_trade / start_price) * asset_step_inv) / asset_step_inv
        open_fee = qty * start_price * fee_rate
        pv[start] = initial_cash_per_trade
        for j in range(start + 1, n):
            pv[j] = initial_cash_per_trade + qty * (prices[j] - start_price) - open_fee

    # 利润累加
    cum_trade = 0.0
    for i in range(1, len(open_idx)):
        cum_trade += trade_profit[i - 1]
        start = open_idx[i]
        for j in range(start, n):
            pv[j] += cum_trade

    return pv, trade_profit

@njit
def _backtest_short_numba(prices, signals, initial_cash_per_trade, asset_step_inv, fee_rate):
    n = len(prices)
    pv = np.zeros(n)
    trade_profit = []
    open_idx = np.where(signals == -1)[0]
    close_idx = np.where(signals == 1)[0]

    if len(close_idx) > len(open_idx) and open_idx[0] > close_idx[0]:
        close_idx = close_idx[1:]
    if len(close_idx) < len(open_idx):
        open_idx = open_idx[:-1]

    for i in range(len(open_idx)):
        start = open_idx[i]
        end = close_idx[i] if i < len(close_idx) else n - 1
        start_price = prices[start]
        qty = int((initial_cash_per_trade / start_price) * asset_step_inv) / asset_step_inv
        open_fee = qty * start_price * fee_rate
        pv[start] = initial_cash_per_trade

        for j in range(start + 1, end + 1):
            pv[j] = initial_cash_per_trade + qty * (start_price - prices[j]) - open_fee
            if j == end:
                close_fee = qty * prices[j] * fee_rate
                pv[j] -= close_fee
                trade_profit.append(pv[j] - initial_cash_per_trade)

    if len(open_idx) > len(close_idx):
        start = open_idx[-1]
        start_price = prices[start]
        qty = int((initial_cash_per_trade / start_price) * asset_step_inv) / asset_step_inv
        open_fee = qty * start_price * fee_rate
        pv[start] = initial_cash_per_trade
        for j in range(start + 1, n):
            pv[j] = initial_cash_per_trade + qty * (start_price - prices[j]) - open_fee

    cum_trade = 0.0
    for i in range(1, len(open_idx)):
        cum_trade += trade_profit[i - 1]
        start = open_idx[i]
        for j in range(start, n):
            pv[j] += cum_trade

    return pv, trade_profit

def backtest_long(data, initial_cash_per_trade, asset):
    prices = data['close'].values
    signals = data['Signal'].values
    asset_step_inv = 10 ** STEP_DECIMALS[asset]
    pv, trade_profit = _backtest_long_numba(prices, signals, initial_cash_per_trade, asset_step_inv, 4.5 / 10000)
    data['Portfolio_Value'] = pv
    data['Buy_and_Hold_Value'] = (1 + data['Return']).cumprod() * initial_cash_per_trade
    open_ts = data.loc[data['Signal'] == 1, 'timestamp']
    close_ts = data.loc[data['Signal'] == -1, 'timestamp']
    return data, trade_profit, open_ts, close_ts

def backtest_short(data, initial_cash_per_trade, asset):
    prices = data['close'].values
    signals = data['Signal'].values
    asset_step_inv = 10 ** STEP_DECIMALS[asset]
    pv, trade_profit = _backtest_short_numba(prices, signals, initial_cash_per_trade, asset_step_inv, 4.5 / 10000)
    data['Portfolio_Value'] = pv
    data['Buy_and_Hold_Value'] = (1 + data['Return']).cumprod() * initial_cash_per_trade
    open_ts = data.loc[data['Signal'] == -1, 'timestamp']
    close_ts = data.loc[data['Signal'] == 1, 'timestamp']
    return data, trade_profit, open_ts, close_ts

# ==========================================================
#  7. 夏普比率
# ==========================================================
def calculate_sharpe_ratio(pv_long, pv_short, risk_free=0.0):
    initial = pv_long[0]
    pv_long = np.array(pv_long)
    pv_short = np.array(pv_short)
    pv_sum = (pv_long - pv_long[0]) + (pv_short - pv_short[0]) + initial
    if np.any(pv_sum < 0):
        return 0.0
    ret = pd.Series(pv_sum).pct_change().fillna(0)
    mean = ret.mean() * 365 * 48
    std = ret.std() * np.sqrt(365 * 48)
    return 0.0 if std == 0 else (mean - risk_free) / std

# ==========================================================
#  8. 主循环
# ==========================================================
minutes = 30
long_short_pairs = [(40, 50), (30, 60), (50, 500), (10, 1500),
                    (150, 700), (30, 500), (40, 1000), (80, 1000), (20, 1500), (100, 700)]
pct = 1 / 3
start_Date = '2022-01-01 00:00:00'

# 统一时间轴
base_timestamps = assets_df[assets[0]].loc[
    assets_df[assets[0]]['timestamp'] >= start_Date, 'timestamp'
].reset_index(drop=True)

merged_long_sum = np.zeros(len(base_timestamps), dtype=np.float64)
merged_short_sum = np.zeros(len(base_timestamps), dtype=np.float64)

profit_thresholds = [0.6, 0.7, 0.8, 0.9, 1.0]
withdraw_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for pft in profit_thresholds:
    for wdt in withdraw_thresholds:
        total_return = 0.0
        merged_long_sum.fill(0.0)
        merged_short_sum.fill(0.0)
        total_trades = 0
        triggered_trades = 0

        for asset1 in Asset_Total_Cash['Asset']:
            for (sw, lw) in long_short_pairs:
                ref = datetime.strptime(start_Date, '%Y-%m-%d %H:%M:%S')
                prev = ref - timedelta(minutes=30 * lw)
                df1 = assets_df[asset1].query('timestamp >= @prev').copy()
                cash = pct / len(long_short_pairs) * float(
                    Asset_Total_Cash.loc[Asset_Total_Cash['Asset'] == asset1, 'Total_Cash'].iloc[0]
                )

                long_df = trend_following_strategy_long(df1, sw, lw)
                long_df, long_profit, _, _ = backtest_long(long_df, cash / 2, asset1)

                short_df = trend_following_strategy_short(df1, sw, lw)
                short_df, short_profit, _, _ = backtest_short(short_df, cash / 2, asset1)

                # 对齐到统一时间轴
                l_pv = long_df.set_index('timestamp')['Portfolio_Value'].reindex(base_timestamps, method='ffill').fillna(0).values
                s_pv = short_df.set_index('timestamp')['Portfolio_Value'].reindex(base_timestamps, method='ffill').fillna(0).values

                merged_long_sum += l_pv
                merged_short_sum += s_pv
                total_return += (sum(long_profit) + sum(short_profit))
                total_trades += (len(long_profit) + len(short_profit))

        sharpe = calculate_sharpe_ratio(merged_long_sum, merged_short_sum)
        ret_ratio = total_return / Asset_Total_Cash['Total_Cash'].sum()
        print(f'Profit:{pft} Withdraw:{wdt} Return:{ret_ratio:.5f} Sharpe:{sharpe:.5f}')
