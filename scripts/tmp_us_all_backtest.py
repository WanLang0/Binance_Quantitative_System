# -*- coding: utf-8 -*-
"""全部美股标的回测对比：背离+量能 vs 经典策略
对 strategy_records.db 中历史测试过的全部 114 只美股 ticker，
用同一份 1h 行情(2025-01-01~今)、同一模拟器，分别跑：
  - 经典指标：KDJ / RSI / MACD / EMA / 双均线 / 布林带
  - MACD变体：MACD基准 / MACD+背离 / 背离+均线 / 背离+量能 / 背离+均线+量能
模式=仅做多(与历史记录主口径一致)，TP/SL=不设。
输出每标的每策略收益/回撤/笔数/夏普，并汇总跨标的稳健性排名。
"""
import os, sys, io, time, warnings, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
PROXY = "http://127.0.0.1:7892"
os.environ.setdefault("HTTP_PROXY", PROXY)
os.environ.setdefault("HTTPS_PROXY", PROXY)

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ta
from datetime import datetime
from indicators import TechnicalIndicators

CHANNEL = "Yahoo Finance 美股1h行情"

# ===== 全部美股标的(来自 strategy_records.db, 已剔除韩股/合约存根) =====
TICKERS = ['AAPL','ABBV','ABT','AEM','ALAB','AMAT','AMD','AMGN','AMT','AMZN','APH','ASML','ASTS','AU','AVGO','AXON','BA','BABA','CDE','CGNX','CIEN','CNQ','COHR','COP','COST','CRDO','CRM','CVX','DIS','DLR','DUK','DVN','ECHO','EQIX','FCX','FNV','GD','GDS','GE','GFI','GILD','GILT','GLW','GOOGL','GSAT','HD','HPE','HWM','IBM','INTC','IRDM','JNJ','JPM','KGC','KMI','KO','LIN','LITE','LLY','LMT','LRCX','MCD','MCHP','META','MPC','MRK','MRVL','MSFT','MU','NEE','NEM','NOC','NOK','NOW','NVDA','NVO','OKE','ORCL','OXY','PEP','PFE','PG','PLD','PSX','QQQ','RGLD','RKLB','RMBS','RTX','SATL','SHEL','SIMO','SMCI','SONY','STM','STX','TDG','TJX','TMO','TRGP','TSAT','TSLA','TSM','TTE','TXN','UPS','V','VLO','VSAT','WDC','WMB','WMT','WPM','XOM']

COMM = 0.001
PIVOT_ORDER = 5
VOL_MULT = 1.5

_sess = None
def _get_session():
    global _sess
    if _sess is not None:
        return _sess
    s = requests.Session()
    s.headers['User-Agent'] = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                               '(KHTML, like Gecko) Chrome/120.0 Safari/537.36')
    s.proxies = {'http': PROXY, 'https': PROXY}
    retry = Retry(total=3, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    s.mount('https://', HTTPAdapter(max_retries=retry))
    try:
        s.get('https://query1.finance.yahoo.com/v8/finance/chart/QQQ?range=1d&interval=1d', timeout=10)
    except Exception:
        pass
    _sess = s
    return s


def fetch_1h(ticker, start='2025-01-01', tries=5):
    for i in range(tries):
        try:
            df = yf.download(ticker, interval='1h', start=start, end=datetime.now(),
                             progress=False, auto_adjust=True, session=_get_session())
            if df is None or df.empty:
                time.sleep(4); continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if 'Open' in df.columns:
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            else:
                df = df[['open', 'high', 'low', 'close', 'volume']].copy()
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.index.name = 'timestamp'
            df = df.dropna(subset=['close'])
            df = df[df['close'] > 0]
            if len(df) >= 200:
                return df
        except Exception as e:
            print('  fetch err:', repr(e)[:70], file=sys.stderr)
        time.sleep(3)
    return None


# ===== 经典指标信号(用项目引擎一致的 TechnicalIndicators) =====
def classic_signals(df, name):
    if name == 'KDJ':
        ip = {"kdj": True, "kdj_k_period": 9, "kdj_d_period": 3, "kdj_j_period": 3,
              "kdj_buy_threshold": 20, "kdj_sell_threshold": 80}
    elif name == 'RSI':
        ip = {"rsi": True, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70}
    elif name == 'MACD':
        ip = {"macd": True, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9}
    elif name == 'EMA':
        ip = {"ema": True, "ema_short": 12, "ema_long": 26, "ema_periods": [12, 26]}
    elif name == '双均线':
        ip = {"ma_cross": True, "ma_cross_short": 10, "ma_cross_long": 30, "ma_cross_periods": [10, 30]}
    elif name == '布林带':
        ip = {"boll": True, "bb_period": 20, "bb_std": 2.0}
    else:
        ip = {"macd": True, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9}
    dft = TechnicalIndicators.calculate_all_indicators(df, ip)
    # 用引擎计算信号(or 模式)
    sig = TechnicalIndicators._calc_signal(dft, ip) if hasattr(TechnicalIndicators, '_calc_signal') else None
    return sig


def classic_signals_via_engine(df, name, sig_engine):
    if name == 'KDJ':
        ip = {"kdj": True, "kdj_k_period": 9, "kdj_d_period": 3, "kdj_j_period": 3,
              "kdj_buy_threshold": 20, "kdj_sell_threshold": 80}
    elif name == 'RSI':
        ip = {"rsi": True, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70}
    elif name == 'MACD':
        ip = {"macd": True, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9}
    elif name == 'EMA':
        ip = {"ema": True, "ema_short": 12, "ema_long": 26, "ema_periods": [12, 26]}
    elif name == '双均线':
        ip = {"ma_cross": True, "ma_cross_short": 10, "ma_cross_long": 30, "ma_cross_periods": [10, 30]}
    elif name == '布林带':
        ip = {"boll": True, "bb_period": 20, "bb_std": 2.0}
    else:
        ip = {"macd": True, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9}
    dft = TechnicalIndicators.calculate_all_indicators(df, ip)
    return sig_engine.calculate_signals(dft, ip)


# ===== MACD 变体信号(背离/均线/量能) =====
def _find_pivots(s, order=PIVOT_ORDER, kind='high'):
    arr = s.to_numpy()
    piv = np.zeros(len(arr), dtype=bool)
    for i in range(order, len(arr) - order):
        win = arr[i - order:i + order + 1]
        if kind == 'high':
            piv[i] = np.argmax(win) == order and arr[i] == win.max()
            piv[i] = piv[i] and arr[i] > arr[i - 1]
        else:
            piv[i] = np.argmin(win) == order and arr[i] == win.min()
            piv[i] = piv[i] and arr[i] < arr[i - 1]
    return pd.Series(piv, index=s.index)


def _divergence(df, macd_col='MACD', order=PIVOT_ORDER):
    px_high = _find_pivots(df['high'], order, 'high')
    px_low = _find_pivots(df['low'], order, 'low')
    macd = df[macd_col].to_numpy()
    hi = df['high'].to_numpy(); lo = df['low'].to_numpy()
    top = np.zeros(len(df), dtype=bool); bot = np.zeros(len(df), dtype=bool)
    ph = None; pm = None
    for i in np.where(px_high.to_numpy())[0]:
        p = hi[i]; m = macd[i]
        if ph is not None and p > ph and m < pm:
            top[i] = True
        ph = p; pm = m
    pl = None; pm2 = None
    for i in np.where(px_low.to_numpy())[0]:
        p = lo[i]; m = macd[i]
        if pl is not None and p < pl and m > pm2:
            bot[i] = True
        pl = p; pm2 = m
    return pd.Series(top, index=df.index), pd.Series(bot, index=df.index)


def _macd_series(df):
    dft = TechnicalIndicators.calculate_macd(df, 12, 26, 9)
    df = df.copy()
    df['MACD'] = dft['MACD']; df['MACD_signal'] = dft['MACD_signal']
    buy = (df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))
    sell = (df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1))
    return buy, sell, df


def _sma(col, period=20):
    return ta.trend.SMAIndicator(col, window=period).sma_indicator()


def build_variant_signals(df, use_div, use_ma, use_vol):
    macd_buy, macd_sell, dfd = _macd_series(df)
    dfd['sma20'] = _sma(dfd['close'], 20)
    dfd['vol_ma20'] = dfd['volume'].rolling(20).mean()
    dfd['vol_up'] = dfd['volume'] > dfd['vol_ma20'] * VOL_MULT
    if use_div:
        top_div, bot_div = _divergence(dfd)
        dfd['top_div'] = top_div; dfd['bot_div'] = bot_div
        buy = macd_buy | bot_div
        sell = macd_sell | top_div
    else:
        buy = macd_buy.copy(); sell = macd_sell.copy()
    close = dfd['close']; sma = dfd['sma20']; vol_up = dfd['vol_up']
    ma_ok_buy = close > sma
    ma_ok_sell = close < sma
    if use_ma:
        buy = buy & ma_ok_buy
        sell = sell & ma_ok_sell
    if use_vol:
        buy = buy & vol_up
        sell = sell & vol_up
    sig = pd.Series(0, index=dfd.index)
    sig[buy] = 1
    sig[sell] = -1
    return dfd, sig


# ===== 模拟器(与参考脚本一致, 仅做多) =====
def simulate(df, signals, mode='long_only', initial=10000.0, comm=COMM):
    cash = initial; units = 0.0; entry = 0.0; side = 0; n = 0
    eq_pts = []
    for i, (ts, row) in enumerate(df.iterrows()):
        price = row['close']
        if not np.isfinite(price) or price <= 0:
            continue
        sig = int(signals.iloc[i]) if i > 0 else 0
        eq = cash + (units * price if side > 0 else units * entry + (entry - price) * units if side < 0 else 0)
        if eq <= 0:
            return None
        eq_pts.append((ts, eq))
        if sig == 1 and side <= 0:
            if side < 0:
                cash = (cash + units * entry + (entry - price) * units) - units * price * comm
                n += 1; side = 0; units = 0
            u = (cash * 0.95) / (price * (1 + comm))
            if u > 0:
                cash -= u * price * (1 + comm); units = u; entry = price; side = 1
        elif sig == -1 and side >= 0:
            if side > 0:
                cash = cash + units * price * (1 - comm)
                n += 1; side = 0; units = 0
            if mode == 'long_short':
                u = (cash * 0.95) / price
                if u > 0:
                    cash -= u * price * (1 + comm); units = u; entry = price; side = -1
    if side != 0 and len(df) > 0:
        price = df['close'].iloc[-1]
        cash = cash + units * price * (1 - comm)
        n += 1
        eq_pts.append((df.index[-1], cash))
    if n == 0:
        return None
    eq = pd.Series(dict(eq_pts)).sort_index()
    peak = eq.cummax()
    mdd = ((eq - peak) / peak * 100).min()
    rr = eq.pct_change().dropna()
    sh = float(rr.mean() / rr.std() * np.sqrt(1764)) if len(rr) >= 10 and rr.std() > 0 else None
    return {'ret': (eq.iloc[-1] / initial - 1) * 100, 'mdd': mdd, 'n': n, 'sh': sh}


# ===== 主循环 =====
from backtest_engine import BacktestEngine
sig_engine = BacktestEngine(signal_mode='or')

CLASSICS = ['KDJ', 'RSI', 'MACD', 'EMA', '双均线', '布林带']
VARIANTS = {
    'MACD基准':       (False, False, False),
    'MACD+背离':      (True,  False, False),
    '背离+均线':      (True,  True,  False),
    '背离+量能':      (True,  False, True),
    '背离+均线+量能': (True,  True,  True),
}

all_results = {}     # ticker -> {strat -> {ret,mdd,n,sh}}
skip_fail = []

for TICKER in TICKERS:
    df = fetch_1h(TICKER)
    if df is None:
        print(f"{TICKER}: 数据获取失败，跳过")
        skip_fail.append(TICKER)
        continue
    days = (df.index[-1] - df.index[0]).days
    per = f"{df.index[0]:%Y-%m-%d}~{df.index[-1]:%Y-%m-%d}"
    buy_hold = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    print(f"\n>>> {TICKER}: {len(df)}根/{days}天 ({per})  买入持有 {buy_hold:+.1f}%", flush=True)

    res = {}
    # 经典指标
    for name in CLASSICS:
        try:
            sig = classic_signals_via_engine(df, name, sig_engine)
            r = simulate(df, sig, 'long_only')
            if r:
                res[name] = r
        except Exception as e:
            print(f"   {name} err: {repr(e)[:60]}", file=sys.stderr)
    # MACD 变体
    for vname, (u_d, u_m, u_v) in VARIANTS.items():
        try:
            _, sig = build_variant_signals(df, u_d, u_m, u_v)
            r = simulate(df, sig, 'long_only')
            if r:
                res[vname] = r
        except Exception as e:
            print(f"   {vname} err: {repr(e)[:60]}", file=sys.stderr)

    all_results[TICKER] = {'buy_hold': buy_hold, 'days': days, 'period': per, 'strat': res}
    print(f"   {res}")

os.makedirs('scripts/results', exist_ok=True)
with open('scripts/results/us_all_backtest_1h_2025.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=1, default=float)
print(f"\n完成: 成功 {len(all_results)} 只, 数据获取失败 {len(skip_fail)} 只: {skip_fail}")
print('saved → scripts/results/us_all_backtest_1h_2025.json')
