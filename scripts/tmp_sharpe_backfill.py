# -*- coding: utf-8 -*-
"""夏普比率回填 v2：修正三点
1) 四年权益曲线按复利链接（每年乘以上年末净值系数），不再直接拼接各自1万基的曲线
2) tp/sl 是否执行按各记录原始口径：最优2/M7=执行；山寨20/实盘2=不执行（与入库数值一致）
3) 最优XLM键名改为 XLM_OPT，避免与山寨XLM_EMA撞键"""
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
MIDCAP = ["DOGE", "TRX", "LINK", "DOT", "BCH", "LTC", "UNI", "APT", "ICP", "XLM"]
EMA_IP = {"ema": True, "ema_short": 12, "ema_long": 26, "ema_periods": [12, 26]}
KDJ_IP = {"kdj": True, "kdj_k_period": 9, "kdj_d_period": 3, "kdj_j_period": 3,
          "kdj_buy_threshold": 20, "kdj_sell_threshold": 80}
RSI_IP = {"rsi": True, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70}
MACD_IP = {"macd": True, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9}
DMA_IP = {"ma_cross": True, "ma_cross_short": 10, "ma_cross_long": 30, "ma_cross_periods": [10, 30]}
def merge(*ds):
    m = {}
    for d in ds:
        m.update(d)
    return m

# (key, 币, 市场, 指标, 模式, tp, sl, 窗口, 执行tp/sl)
CONFIGS = [
    ('ETH_KDJ', 'ETH', 'future', KDJ_IP, 'LO', 0.08, 0.05, 'y4', True),
    ('XLM_OPT', 'XLM', 'future', EMA_IP, 'LS', 0.05, 0.05, 'y4', True),
]
for c in MIDCAP:
    CONFIGS.append((f'{c}_EMA', c, 'future', EMA_IP, 'LO', None, None, 'y4', False))
    CONFIGS.append((f'{c}_KDJ', c, 'future', KDJ_IP, 'LO', 0.08, 0.05, 'y4', False))
M7_CFG = {'AAPL': (DMA_IP, 'LO', None, None), 'MSFT': (EMA_IP, 'LS', None, None),
          'NVDA': (merge(RSI_IP, MACD_IP), 'LO', None, None), 'GOOGL': (DMA_IP, 'LO', None, None),
          'AMZN': (EMA_IP, 'LO', None, None), 'META': (RSI_IP, 'LO', 0.05, 0.05),
          'TSLA': (EMA_IP, 'LS', 0.05, 0.05)}
for c, (ip, md, tp, sl) in M7_CFG.items():
    CONFIGS.append((f'{c}_M7', c, 'future', ip, md, tp, sl, 'full', True))
for c in ('MUB', 'MUUB'):
    CONFIGS.append((f'{c}_SB', c, 'spot', KDJ_IP, 'LO', None, None, 'full', False))

fut = BinanceDataFetcher(); fut.set_market_type('future')
spot = BinanceDataFetcher(); spot.set_market_type('spot')
fetchers = {'future': fut, 'spot': spot}
sig_engine = BacktestEngine(signal_mode='or')

def fetch(fm, sym, start, end, tries=5):
    for i in range(tries):
        try:
            df = fm.fetch_historical_data(sym, start, end, "4h")
            if df is not None and not df.empty and len(df) >= 100:
                return df
        except Exception:
            pass
        time.sleep(2 * (i + 1))
    return None

def _close(cash, units, entry, price, side, comm=0.001):
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm

def simulate(df, signals, tp, sl, mode, use_tpsl, initial=10000.0, comm=0.001):
    cash = initial; units = 0.0; entry = 0.0; side = 0; n = 0
    eq_pts = []
    for i, (ts, row) in enumerate(df.iterrows()):
        price = row['close']
        if not np.isfinite(price) or price <= 0:
            continue
        sig = int(signals.iloc[i]) if i > 0 else 0
        if use_tpsl and side != 0 and entry > 0:
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
            u = (cash * 0.95) / (price * (1 + comm))
            if u > 0:
                cash -= u * price * (1 + comm); units = u; entry = price; side = 1
        elif sig == -1 and side >= 0:
            if side > 0:
                cash = _close(cash, units, entry, price, side); n += 1; side = 0; units = 0
            if mode == 'LS':
                u = (cash * 0.95) / price
                if u > 0:
                    cash -= u * price * (1 + comm); units = u; entry = price; side = -1
    if side != 0 and len(df) > 0:
        cash = _close(cash, units, entry, df['close'].iloc[-1], side)
        n += 1
        eq_pts.append((df.index[-1], cash))
    if n == 0:
        return None
    return pd.Series(dict(eq_pts)).sort_index()

def sharpe_of(eq):
    if eq is None or len(eq) < 30:
        return None
    r = eq.pct_change().dropna()
    if len(r) < 20 or r.std() <= 0:
        return None
    return float(r.mean() / r.std() * np.sqrt(2190))

out = {}
for key, coin, mkt, ip, md, tp, sl, win, use_tpsl in CONFIGS:
    fm = fetchers[mkt]
    sym = f"{coin}/USDT:USDT" if mkt == 'future' else f"{coin}/USDT"
    eq_parts, mult = [], 1.0
    try:
        if win == 'y4':
            for _, s, e in YEARS:
                df = fetch(fm, sym, s, e)
                if df is None:
                    continue
                dft = TechnicalIndicators.calculate_all_indicators(df, ip)
                signals = sig_engine.calculate_signals(dft, ip)
                eq = simulate(df, signals, tp, sl, md, use_tpsl)
                if eq is None:
                    continue
                eq_parts.append(eq * mult)          # 复利链接：乘以上年末净值系数
                mult *= eq.iloc[-1] / 10000.0
        else:
            df = fetch(fm, sym, "2026-01-01", "2026-08-30")
            if df is not None:
                dft = TechnicalIndicators.calculate_all_indicators(df, ip)
                signals = sig_engine.calculate_signals(dft, ip)
                eq = simulate(df, signals, tp, sl, md, use_tpsl)
                if eq is not None:
                    eq_parts.append(eq)
        sh = None
        if eq_parts:
            full = pd.concat(eq_parts)
            full = full[~full.index.duplicated(keep='last')].sort_index()
            sh = sharpe_of(full)
            out[key] = None if sh is None else round(sh, 2)
            print(f"{key:<12} 夏普={out[key]}  四年累计{(mult - 1) * 100:+.1f}%", flush=True)
            continue
    except Exception as ex:
        print(f"{key}: 异常 {type(ex).__name__} {str(ex)[:60]}", flush=True)
    out[key] = None
    print(f"{key:<12} 夏普=None", flush=True)

import json
with open(os.path.join('scripts', 'results', 'sharpe_backfill.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
done = sum(1 for v in out.values() if v is not None)
print(f"\n完成 {done}/{len(CONFIGS)}，saved → scripts/results/sharpe_backfill.json")
