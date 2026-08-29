# -*- coding: utf-8 -*-
"""补充 2023/2024 年回测：对汇总页加密币策略逐笔重算收益/交易次数/胜率，输出跨年矩阵"""
import os, sys, io, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7892")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7892")

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

# (标签, 市场, 品种, 周期, 策略, 模式, 止盈, 止损) —— ETH KDJ 走回测引擎(仅多/现货)，其余走双向模拟器
PERP_CONFIGS = [
    ("NEAR 4h EMA 2/2", "NEAR/USDT:USDT", "4h", ["EMA"], "long_only", 0.02, 0.02),
    ("NEAR 4h EMA+MACD 3/3", "NEAR/USDT:USDT", "4h", ["EMA", "MACD"], "long_only", 0.03, 0.03),
    ("TON 4h MACD 5/5", "TON/USDT:USDT", "4h", ["MACD"], "long_short", 0.05, 0.05),
    ("TON 4h MACD", "TON/USDT:USDT", "4h", ["MACD"], "long_short", None, None),
    ("BCH 4h RSI+EMA", "BCH/USDT:USDT", "4h", ["RSI", "EMA"], "long_short", None, None),
    ("LTC 1h RSI", "LTC/USDT:USDT", "1h", ["RSI"], "long_short", None, None),
    ("XRP 4h RSI", "XRP/USDT:USDT", "4h", ["RSI"], "long_short", None, None),
    ("AVAX 4h EMA+MACD 5/5", "AVAX/USDT:USDT", "4h", ["EMA", "MACD"], "long_short", 0.05, 0.05),
    ("NEAR 4h RSI", "NEAR/USDT:USDT", "4h", ["RSI"], "long_short", None, None),
    ("BTC 4h 布林带", "BTC/USDT:USDT", "4h", ["布林带"], "long_short", None, None),
]
ETH_CONFIGS = [
    ("ETH 4h KDJ 5/5", 0.05, 0.05),
    ("ETH 4h KDJ 8/5", 0.08, 0.05),
]
YEARS = [("2025", "2025-01-01", "2025-12-31")]

def _close(cash, units, entry, price, side, comm):
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm

def simulate_trades(df, signals, tp, sl, mode):
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
    for i in range(tries):
        df = fetcher.fetch_historical_data(sym, start, end, tf)
        if df is not None and len(df) >= 100:
            return df
        time.sleep(2 * (i + 1))
    return df

cache = {}
def get(sym, tf, start, end):
    key = (sym, tf, start, end)
    if key not in cache:
        cache[key] = fetch_retry(sym, tf, start, end)
    return cache[key]

print(f"{'配置':<24}{'年':<6}{'收益%':<9}{'交易':<6}{'胜率%':<7}{'数据量'}", flush=True)
for year, start, end in YEARS:
    for label, sym, tf, names, mode, tp, sl in PERP_CONFIGS:
        df = get(sym, tf, start, end)
        if df is None or df.empty:
            print(f"{label:<24}{year:<6}无数据", flush=True)
            continue
        ip = merged(names)
        dft = TechnicalIndicators.calculate_all_indicators(df, ip)
        signals = engine.calculate_signals(dft, ip)
        pnls = simulate_trades(df, signals, tp, sl, mode)
        if not pnls:
            print(f"{label:<24}{year:<6}无交易", flush=True)
            continue
        wins = sum(1 for p in pnls if p > 0)
        ret = (INITIAL + sum(pnls)) / INITIAL * 100 - 100
        print(f"{label:<24}{year:<6}{ret:<9.1f}{len(pnls):<6}{wins/len(pnls)*100:<7.1f}{len(df)}根", flush=True)

    # ETH KDJ 走回测引擎（现货、仅多、OR）
    fetcher.set_market_type('spot')
    for label, tp, sl in ETH_CONFIGS:
        df = get("ETH/USDT", "4h", start, end)
        if df is None or df.empty:
            print(f"{label:<24}{year:<6}无数据", flush=True)
            continue
        ip = merged(["KDJ"])
        dft = TechnicalIndicators.calculate_all_indicators(df, ip)
        eng = BacktestEngine(INITIAL, COMM, take_profit=tp, stop_loss=sl, timeframe="4h", signal_mode='or')
        res = eng.run_backtest(dft, ip)
        print(f"{label:<24}{year:<6}{res['total_return']:<9.1f}{res['total_trades']:<6}{res['win_rate']:<7.1f}{len(df)}根", flush=True)
    fetcher.set_market_type('future')
