# -*- coding: utf-8 -*-
"""教科书组合策略四年验证（ETH/BTC/NEAR/TON，4h，2023-2026）：
A. MACD+20日均线：水上金叉做多 / 水下死叉做空（趋势过滤）
B. MACD+KDJ 区分使用：MACD>0时KDJ金叉才买 / MACD<0时KDJ死叉才卖（动能过滤）
C. 布林带+RSI反转(+MACD动能)：触上轨且RSI>70做空 / 触下轨且RSI<30做多（超买超卖反转）
对比基准：ETH 4h KDJ 8/5 仅多（四年唯一正收益）"""
import os, sys, io, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
os.environ.setdefault("HTTP_PROXY", "http://192.168.11.188:7892")
os.environ.setdefault("HTTPS_PROXY", "http://192.168.11.188:7892")

import numpy as np
import pandas as pd
import ta
from data_fetcher import BinanceDataFetcher

START, END = "2023-01-01", "2026-08-26"
INITIAL = 10000.0; COMM = 0.001
SYMBOLS = ["ETH/USDT:USDT", "BTC/USDT:USDT", "NEAR/USDT:USDT", "TON/USDT:USDT"]
YEARS = [("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
         ("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-08-26")]

def build_signals(df):
    """计算三套组合信号，返回 dict[name] = (buy, sell) 布尔Series"""
    close = df['close']
    sma120 = ta.trend.SMAIndicator(close, window=120).sma_indicator()      # 20日均线(4h×120)
    macd = ta.trend.MACD(close)
    macd_line, macd_sig, macd_hist = macd.macd(), macd.macd_signal(), macd.macd_diff()
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], close, window=9, smooth_window=3)
    k, d = stoch.stoch(), stoch.stoch_signal()
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)

    macd_gold = (macd_line > macd_sig) & (macd_line.shift(1) <= macd_sig.shift(1))
    macd_dead = (macd_line < macd_sig) & (macd_line.shift(1) >= macd_sig.shift(1))
    kdj_gold = (k > d) & (k.shift(1) <= d.shift(1))
    kdj_dead = (k < d) & (k.shift(1) >= d.shift(1))

    sig = {}
    # A. MACD+20日均线：水上金叉做多 / 水下死叉做空
    sig['A MACD+MA20趋势'] = (
        macd_gold & (macd_line > 0) & (close > sma120),
        macd_dead & (macd_line < 0) & (close < sma120),
    )
    # B. MACD+KDJ 区分使用：MACD>0 时 KDJ 金叉才买；MACD<0 时 KDJ 死叉才卖
    sig['B KDJ+MACD过滤'] = (
        kdj_gold & (macd_line > 0),
        kdj_dead & (macd_line < 0),
    )
    # C1. 布林带+RSI 反转：下轨+RSI<30 做多；上轨+RSI>70 做空
    sig['C 布林+RSI反转'] = (
        (close <= bb.bollinger_lband()) & (rsi < 30),
        (close >= bb.bollinger_hband()) & (rsi > 70),
    )
    # C2. C1 + MACD 动能减弱确认（hist 同向收敛）
    sig['C2 布林+RSI+MACD'] = (
        (close <= bb.bollinger_lband()) & (rsi < 30) & (macd_hist > macd_hist.shift(1)),
        (close >= bb.bollinger_hband()) & (rsi > 70) & (macd_hist < macd_hist.shift(1)),
    )
    return sig

def _close(cash, units, entry, price, side, comm):
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm

def simulate(df, buy, sell, tp, sl, initial):
    """双向模拟器（buy触发开多/平空，sell触发平多/开空）"""
    cash = initial; units = 0.0; entry = 0.0; side = 0
    n = 0; wins = 0
    eq_pts = []
    for i, (ts, row) in enumerate(df.iterrows()):
        price = row['close']
        if not np.isfinite(price) or price <= 0:
            continue
        b = bool(buy.iloc[i]) if i > 0 else False
        s = bool(sell.iloc[i]) if i > 0 else False
        if side != 0 and entry > 0:
            r = (price - entry) / entry if side > 0 else (entry - price) / entry
            if (tp and r >= tp) or (sl and r <= -sl):
                before = cash
                cash = _close(cash, units, entry, price, side, COMM)
                n += 1
                if cash > before * (1 - COMM):
                    wins += 1
                side = 0; units = 0; entry = 0
                eq_pts.append((ts, cash))
                continue
        if side > 0:
            eq = cash + units * price
        elif side < 0:
            eq = cash + units * entry + (entry - price) * units
        else:
            eq = cash
        if eq <= 0:
            eq_pts.append((ts, 0.0))
            break
        eq_pts.append((ts, eq))
        if b and side <= 0:
            if side < 0:
                before = cash
                cash = _close(cash, units, entry, price, side, COMM)
                n += 1
                if cash > before * (1 - COMM):
                    wins += 1
                side = 0; units = 0
            invest = cash * 0.95
            u = invest / (price * (1 + COMM))
            if u > 0:
                cash -= u * price * (1 + COMM)
                units = u; entry = price; side = 1
        elif s and side >= 0:
            if side > 0:
                before = cash
                cash = _close(cash, units, entry, price, side, COMM)
                n += 1
                if cash > before * (1 - COMM):
                    wins += 1
                side = 0; units = 0
            notional = cash * 0.95
            u = notional / price
            if u > 0:
                cash -= u * price * (1 + COMM)
                units = u; entry = price; side = -1
    if side != 0 and eq_pts:
        before = cash
        cash = _close(cash, units, entry, df['close'].iloc[-1], side, COMM)
        n += 1
        if cash > before * (1 - COMM):
            wins += 1
        eq_pts.append((df.index[-1], cash))
    return cash, n, wins

fetcher = BinanceDataFetcher(); fetcher.set_market_type('future')

def fetch_retry(sym, tries=6):
    for i in range(tries):
        df = fetcher.fetch_historical_data(sym, START, END, "4h")
        if df is not None and len(df) >= 1000:
            return df
        time.sleep(2 * (i + 1))
    return None

print(f"{'策略':<20}{'品种':<7}{'2023':<9}{'2024':<9}{'2025':<9}{'2026':<9}{'总笔数':<7}{'胜率%':<6}", flush=True)
for sym in SYMBOLS:
    df = fetch_retry(sym)
    if df is None:
        print(f"{sym}: 无数据", flush=True)
        continue
    sigs = build_signals(df)
    tag = sym.split('/')[0]
    for name, (buy, sell) in sigs.items():
        year_rets = {}
        total_n = 0; total_wins = 0
        for year, start, end in YEARS:
            mask = (df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end) + pd.Timedelta(days=1))
            seg = df[mask]
            if len(seg) < 500:
                year_rets[year] = None
                continue
            final, n, wins = simulate(seg, buy[mask], sell[mask], 0.05, 0.05, INITIAL)
            year_rets[year] = (final / INITIAL - 1) * 100
            total_n += n; total_wins += wins
        cells = " ".join(f"{year_rets.get(y):+.1f}" if year_rets.get(y) is not None else "  —  "
                         for y in ["2023", "2024", "2025", "2026"])
        wr = total_wins / total_n * 100 if total_n else 0
        print(f"{name:<20}{tag:<7}{cells}  {total_n:<7}{wr:<6.0f}", flush=True)
