# -*- coding: utf-8 -*-
"""山寨10币逐币数据提取：EMA(12/26)仅多不设 + KDJ 8/5仅多，四年累计（用于拆分汇总行入库）"""
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

YEARS = [("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
         ("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-08-26")]
COINS = ["DOGE", "TRX", "LINK", "DOT", "BCH", "LTC", "UNI", "APT", "ICP", "XLM"]
EMA_IP = {"ema": True, "ema_short": 12, "ema_long": 26, "ema_periods": [12, 26]}
KDJ_IP = {"kdj": True, "kdj_k_period": 9, "kdj_d_period": 3, "kdj_j_period": 3,
          "kdj_buy_threshold": 20, "kdj_sell_threshold": 80}

fetcher = BinanceDataFetcher(); fetcher.set_market_type('future')

def fetch(sym, start, end, tries=5):
    for i in range(tries):
        try:
            df = fetcher.fetch_historical_data(sym, start, end, "4h")
            if df is not None and not df.empty and len(df) >= 100:
                return df
        except Exception:
            pass
        time.sleep(2 * (i + 1))
    return None

def sim_longonly(df, signals, tp=None, sl=None, initial=10000.0, comm=0.001):
    cash = initial; units = 0.0; n = 0
    eq_pts = []
    for i, (ts, row) in enumerate(df.iterrows()):
        price = row['close']
        if not np.isfinite(price) or price <= 0:
            continue
        sig = int(signals.iloc[i]) if i > 0 else 0
        if units > 0:
            r = (price - entry) / entry if (entry := entry or 1) else 0
        eq = cash + units * price
        eq_pts.append((ts, eq))
        if sig == 1 and units == 0:
            u = (cash * 0.95) / (price * (1 + comm))
            if u > 0:
                cash -= u * price * (1 + comm); units = u; entry = price
        elif sig == -1 and units > 0:
            cash += units * price * (1 - comm); n += 1; units = 0; entry = 0
    if units > 0 and len(df) > 0:
        cash += units * df['close'].iloc[-1] * (1 - comm); n += 1
    if n == 0:
        return None
    eq = pd.Series(dict(eq_pts)).sort_index()
    peak = eq.cummax()
    return (eq.iloc[-1] / initial - 1) * 100, ((eq - peak) / peak * 100).min(), n

sig_engine = BacktestEngine(signal_mode='or')
print(f"{'币':<6}{'EMA仅多不设四年':>14}{'回撤':>8}{'笔':>4} | {'KDJ8/5仅多四年':>13}{'回撤':>8}{'笔':>4}", flush=True)
results = {}
for coin in COINS:
    sym = f"{coin}/USDT:USDT"
    out = {}
    for tag, ip, tp, sl in [('EMA', EMA_IP, None, None), ('KDJ', KDJ_IP, 0.08, 0.05)]:
        cum, mdd, trades = 1.0, None, 0
        ok = True
        for year, start, end in YEARS:
            df = fetch(sym, start, end)
            if df is None:
                continue
            dft = TechnicalIndicators.calculate_all_indicators(df, ip)
            signals = sig_engine.calculate_signals(dft, ip)
            r = sim_longonly(df, signals, tp, sl)
            if r is None:
                continue
            cum *= (1 + r[0] / 100)
            trades += r[2]
            mdd = r[1] if mdd is None else min(mdd, r[1])
        out[tag] = ((cum - 1) * 100, mdd, trades)
    results[coin] = out
    e, k = out['EMA'], out['KDJ']
    print(f"{coin:<6}{e[0]:>+13.1f}%{e[1] or 0:>7.1f}%{e[2]:>4} | {k[0]:>+12.1f}%{(k[1] or 0):>7.1f}%{k[2]:>4}", flush=True)

import json
with open(os.path.join('scripts', 'results', 'midcap_persymbol.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
print('saved')
