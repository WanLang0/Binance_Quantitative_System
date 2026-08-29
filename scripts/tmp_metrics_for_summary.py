# -*- coding: utf-8 -*-
"""为最优策略汇总页补算：每个策略的交易次数与胜率（复刻 daily-return 模拟器并记录每笔盈亏）"""
import os, sys, io, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
os.environ.setdefault("HTTP_PROXY", "http://192.168.11.188:7892")
os.environ.setdefault("HTTPS_PROXY", "http://192.168.11.188:7892")

import numpy as np
import pandas as pd
from data_fetcher import BinanceDataFetcher
from indicators import TechnicalIndicators
from backtest_engine import BacktestEngine

INITIAL = 10000.0; COMM = 0.001
BASE = {
    "RSI": {"rsi": True, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70},
    "KDJ": {"kdj": True, "kdj_k_period": 9, "kdj_d_period": 3, "kdj_j_period": 3,
            "kdj_buy_threshold": 20, "kdj_sell_threshold": 80},
    "布林带": {"boll": True, "bb_period": 20, "bb_std": 2.0},
    "EMA": {"ema": True, "ema_short": 12, "ema_long": 26, "ema_periods": [12, 26]},
    "MACD": {"macd": True, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
}

def merged(names):
    p = {}
    for n in names:
        p.update(BASE[n])
    return p

# (品种, 周期, 策略名列表, 模式, 止盈, 止损, 窗口)
CONFIGS = [
    ("NEAR/USDT:USDT", "4h", ["EMA"], "long_only", 0.02, 0.02, ("2026-01-01", "2026-08-26")),
    ("NEAR/USDT:USDT", "4h", ["EMA", "MACD"], "long_only", 0.03, 0.03, ("2026-01-01", "2026-08-26")),
    ("TON/USDT:USDT", "4h", ["MACD"], "long_short", 0.05, 0.05, ("2026-01-01", "2026-08-26")),
    ("TQQQ/USDT:USDT", "4h", ["RSI", "MACD"], "long_short", None, None, ("2026-01-01", "2026-08-26")),
    ("TQQQ/USDT:USDT", "4h", ["MACD"], "long_short", 0.05, 0.05, ("2026-01-01", "2026-08-26")),
    ("MUB/USDT:USDT", "4h", ["KDJ"], "long_short", None, None, ("2026-01-01", "2026-08-26")),
    ("TQQQ/USDT:USDT", "1h", ["KDJ"], "long_short", None, None, ("2026-01-01", "2026-08-26")),
    ("TON/USDT:USDT", "4h", ["MACD"], "long_short", None, None, ("2026-01-01", "2026-08-26")),
    ("MUUB/USDT:USDT", "4h", ["KDJ"], "long_short", None, None, ("2026-01-01", "2026-08-26")),
    ("BCH/USDT:USDT", "4h", ["RSI", "EMA"], "long_short", None, None, ("2025-01-01", "2025-12-31")),
    ("LTC/USDT:USDT", "1h", ["RSI"], "long_short", None, None, ("2025-01-01", "2025-12-31")),
    ("XRP/USDT:USDT", "4h", ["RSI"], "long_short", None, None, ("2025-01-01", "2025-12-31")),
    ("AVAX/USDT:USDT", "4h", ["EMA", "MACD"], "long_short", 0.05, 0.05, ("2025-01-01", "2025-12-31")),
    ("NEAR/USDT:USDT", "4h", ["RSI"], "long_short", None, None, ("2025-01-01", "2025-12-31")),
    ("BTC/USDT:USDT", "4h", ["布林带"], "long_short", None, None, ("2025-01-01", "2025-12-31")),
]

def _close(cash, units, entry, price, side, comm):
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm

def simulate_trades(df, signals, tp, sl, mode):
    """复刻双向模拟器，额外记录每笔往返盈亏(USDT)"""
    cash = INITIAL; units = 0.0; entry = 0.0; side = 0
    cash_at_open = None; pnls = []
    for i, (ts, row) in enumerate(df.iterrows()):
        price = row['close']
        if not np.isfinite(price) or price <= 0:
            continue
        sig = int(signals.iloc[i]) if i > 0 else 0
        if side != 0 and entry > 0:
            r = (price - entry) / entry if side > 0 else (entry - price) / entry
            if (tp and r >= tp) or (sl and r <= -sl):
                cash = _close(cash, units, entry, price, side, COMM)
                pnls.append(cash - cash_at_open)
                side = 0; units = 0; entry = 0
                continue
        if side > 0:
            eq = cash + units * price
        elif side < 0:
            eq = cash + units * entry + (entry - price) * units
        else:
            eq = cash
        if eq <= 0:
            return None
        if sig == 1 and side <= 0:
            if side < 0:
                cash = _close(cash, units, entry, price, side, COMM)
                pnls.append(cash - cash_at_open)
                side = 0; units = 0
            invest = cash * 0.95
            u = invest / (price * (1 + COMM))
            if u > 0:
                cash_at_open = cash
                cash -= u * price * (1 + COMM)
                units = u; entry = price; side = 1
        elif sig == -1 and side >= 0:
            if side > 0:
                cash = _close(cash, units, entry, price, side, COMM)
                pnls.append(cash - cash_at_open)
                side = 0; units = 0
            if mode == "long_short":
                notional = cash * 0.95
                u = notional / price
                if u > 0:
                    cash_at_open = cash
                    cash -= u * price * (1 + COMM)
                    units = u; entry = price; side = -1
    if side != 0 and len(df) > 0:
        cash = _close(cash, units, entry, df['close'].iloc[-1], side, COMM)
        pnls.append(cash - cash_at_open)
    return pnls

fetcher = BinanceDataFetcher(); fetcher.set_market_type('future')
engine = BacktestEngine(signal_mode='or')

def fetch_retry(sym, tf, start, end, tries=6):
    df = None
    for i in range(tries):
        df = fetcher.fetch_historical_data(sym, start, end, tf)
        if df is not None and len(df) >= 100:
            return df
        time.sleep(2 * (i + 1))
    return df

print(f"{'品种':<16}{'周期':<5}{'策略':<12}{'模式':<12}{'交易次数':<7}{'胜率%':<7}{'总收益%':<8}", flush=True)
cache = {}
for sym, tf, names, mode, tp, sl, (start, end) in CONFIGS:
    key = (sym, tf, start, end)
    if key not in cache:
        cache[key] = fetch_retry(sym, tf, start, end)
    df = cache[key]
    if df is None or df.empty:
        print(f"{sym} {tf}: 无数据", flush=True)
        continue
    ip = merged(names)
    dft = TechnicalIndicators.calculate_all_indicators(df, ip)
    signals = engine.calculate_signals(dft, ip)
    pnls = simulate_trades(df, signals, tp, sl, mode)
    if not pnls:
        print(f"{sym} {tf} {'+'.join(names)}: 无交易", flush=True)
        continue
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / len(pnls) * 100
    # 总收益验证
    last_cash = INITIAL + sum(pnls)
    total_ret = (last_cash / INITIAL - 1) * 100
    print(f"{sym:<16}{tf:<5}{'+'.join(names):<12}{mode:<12}{len(pnls):<7}{win_rate:<7.1f}{total_ret:<8.1f}", flush=True)
