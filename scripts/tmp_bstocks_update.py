# -*- coding: utf-8 -*-
"""bStocks 美股代币样本外复验：用当前全部历史数据重测当时的优秀配置（口径：双向模拟1倍保证金/0.1%手续费）"""
import os, sys, io, time, warnings
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

RSI = {"rsi": True, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70}
MACD = {"macd": True, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9}
KDJ = {"kdj": True, "kdj_k_period": 9, "kdj_d_period": 3, "kdj_j_period": 3,
       "kdj_buy_threshold": 20, "kdj_sell_threshold": 80}
def merge(*ds):
    m = {}
    for d in ds:
        m.update(d)
    return m

# (币, 市场, 周期, 策略, tp, sl, 当时收益)
CONFIGS = [
    ("TQQQ/USDT:USDT", 'future', "4h", "RSI+MACD", merge(RSI, MACD), None, None, "+42.3%"),
    ("TQQQ/USDT:USDT", 'future', "4h", "MACD", MACD, 0.05, 0.05, "+32.3%"),
    ("TQQQ/USDT:USDT", 'future', "1h", "KDJ", KDJ, None, None, "+54.5%"),
    ("MUB/USDT", 'spot', "4h", "KDJ", KDJ, None, None, "+201.8%"),
    ("MUUB/USDT", 'spot', "4h", "KDJ", KDJ, None, None, "+136.8%"),
]

fetcher = BinanceDataFetcher()

def fetch(sym, market, tf):
    fetcher.set_market_type(market)
    for i in range(5):
        try:
            df = fetcher.fetch_historical_data(sym, "2026-01-01", "2026-08-30", tf)
            if df is not None and not df.empty and len(df) >= 60:
                return df
        except Exception:
            pass
        time.sleep(3)
    return None

def _close(cash, units, entry, price, side, comm=0.001):
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm

def simulate(df, signals, tp, sl, initial=10000.0):
    cash = initial; units = 0.0; entry = 0.0; side = 0; n = 0
    eq_pts = []
    for i, (ts, row) in enumerate(df.iterrows()):
        price = row['close']
        if not np.isfinite(price) or price <= 0:
            continue
        sig = int(signals.iloc[i]) if i > 0 else 0
        if side != 0 and entry > 0:
            r = (price - entry) / entry if side > 0 else (entry - price) / entry
            if (tp and r >= tp) or (sl and r <= -sl):
                cash = _close(cash, units, entry, price, side)
                side = 0; units = 0; n += 1
                eq_pts.append((ts, cash)); continue
        eq = cash + (units * price if side > 0 else units * entry + (entry - price) * units if side < 0 else 0)
        if eq <= 0:
            return None
        eq_pts.append((ts, eq))
        if sig == 1 and side <= 0:
            if side < 0:
                cash = _close(cash, units, entry, price, side); n += 1; side = 0; units = 0
            u = (cash * 0.95) / (price * 1.001)
            if u > 0:
                cash -= u * price * 1.001; units = u; entry = price; side = 1
        elif sig == -1 and side >= 0:
            if side > 0:
                cash = _close(cash, units, entry, price, side); n += 1; side = 0; units = 0
            u = (cash * 0.95) / price
            if u > 0:
                cash -= u * price * 1.001; units = u; entry = price; side = -1
    if side != 0 and len(df) > 0:
        cash = _close(cash, units, entry, df['close'].iloc[-1], side)
        n += 1
        eq_pts.append((df.index[-1], cash))
    if n == 0:
        return None
    eq = pd.Series(dict(eq_pts)).sort_index()
    days = max((eq.index[-1] - eq.index[0]).days, 1)
    peak = eq.cummax(); mdd = ((eq - peak) / peak * 100).min()
    wins = None
    return {'ret': (eq.iloc[-1] / initial - 1) * 100, 'daily': (eq.iloc[-1] / initial - 1) * 100 / days,
            'mdd': mdd, 'trades': n, 'days': days}

out = []
sig_engine = BacktestEngine(signal_mode='or')
print(f"{'配置':<44}{'当时':>8}{'现在':>9}{'日均':>7}{'回撤':>8}{'笔数':>5}{'样本天数':>7}", flush=True)
for sym, market, tf, sname, ip, tp, sl, was in CONFIGS:
    df = fetch(sym, market, tf)
    if df is None:
        print(f"{sym} {tf} {sname:<10} 数据不可得", flush=True)
        continue
    dft = TechnicalIndicators.calculate_all_indicators(df, ip)
    signals = sig_engine.calculate_signals(dft, ip)
    r = simulate(df, signals, tp, sl)
    if r is None:
        print(f"{sym.split('/')[0]} {tf} {sname:<10} 无信号", flush=True)
        continue
    line = f"{sym.split('/')[0]:<6}{tf} {sname:<10}{'不设' if not tp else '5/5':<5}{was:>8}{r['ret']:>+8.1f}%{r['daily']:>+6.2f}%{r['mdd']:>7.1f}%{r['trades']:>5}{r['days']:>7}"
    print(line, flush=True)
    out.append(line)

with open(os.path.join('scripts', 'results', 'bstocks_update.txt'), 'w', encoding='utf-8') as f:
    f.write("\n".join(out))
