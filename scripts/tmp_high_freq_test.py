# -*- coding: utf-8 -*-
"""高频周期验证：多币种全扫描组合(15币)在 1h/15m 的四年表现 + BTC-ETH配对15m版。
回答：是不是4h周期太久了，更高频是否更好。"""
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
SYMBOLS = [f"{b}/USDT:USDT" for b in
           ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "TRX",
            "AVAX", "LINK", "DOT", "LTC", "BCH", "NEAR", "TON"]]
YEARS = [("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
         ("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-08-26")]
TIMEFRAMES = ["1h", "15m"]
BASE = {
    "KDJ": {"kdj": True, "kdj_k_period": 9, "kdj_d_period": 3, "kdj_j_period": 3,
            "kdj_buy_threshold": 20, "kdj_sell_threshold": 80},
    "MACD": {"macd": True, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
}
STRATS = [
    ("KDJ8/5仅多", BASE["KDJ"], "long_only", 0.08, 0.05),
    ("KDJ双向5/5", BASE["KDJ"], "long_short", 0.05, 0.05),
    ("MACD双向5/5", BASE["MACD"], "long_short", 0.05, 0.05),
]

def _close(cash, units, entry, price, side, comm):
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm

def simulate(df, signals, tp, sl, mode, initial):
    cash = initial; units = 0.0; entry = 0.0; side = 0
    wins = 0; n = 0
    eq_pts = []
    for i, (ts, row) in enumerate(df.iterrows()):
        price = row['close']
        if not np.isfinite(price) or price <= 0:
            continue
        sig = int(signals.iloc[i]) if i > 0 else 0
        if side != 0 and entry > 0:
            r = (price - entry) / entry if side > 0 else (entry - price) / entry
            if (tp and r >= tp) or (sl and r <= -sl):
                before = cash
                cash = _close(cash, units, entry, price, side, COMM)
                n += 1
                if cash > before:
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
        if sig == 1 and side <= 0:
            if side < 0:
                before = cash
                cash = _close(cash, units, entry, price, side, COMM)
                n += 1
                if cash > before:
                    wins += 1
                side = 0; units = 0
            invest = cash * 0.95
            u = invest / (price * (1 + COMM))
            if u > 0:
                cash -= u * price * (1 + COMM)
                units = u; entry = price; side = 1
        elif sig == -1 and side >= 0:
            if side > 0:
                before = cash
                cash = _close(cash, units, entry, price, side, COMM)
                n += 1
                if cash > before:
                    wins += 1
                side = 0; units = 0
            if mode == "long_short":
                notional = cash * 0.95
                u = notional / price
                if u > 0:
                    cash -= u * price * (1 + COMM)
                    units = u; entry = price; side = -1
    if side != 0 and eq_pts:
        before = cash
        cash = _close(cash, units, entry, df['close'].iloc[-1], side, COMM)
        n += 1
        if cash > before:
            wins += 1
        eq_pts.append((df.index[-1], cash))
    eq = pd.Series(dict(eq_pts)).sort_index()
    return eq, n, wins

fetcher = BinanceDataFetcher(); fetcher.set_market_type('future')
engine = BacktestEngine(signal_mode='or')

def fetch_retry(sym, tf, start, end, tries=6):
    for i in range(tries):
        df = fetcher.fetch_historical_data(sym, start, end, tf)
        if df is not None and len(df) >= 500:
            return df
        time.sleep(2 * (i + 1))
    return df

print(f"{'周期':<5}{'策略':<12}{'年':<6}{'币数':<5}{'收益%':<9}{'交易':<7}{'胜率%':<7}", flush=True)
for tf in TIMEFRAMES:
    for year, start, end in YEARS:
        data = {}
        for s in SYMBOLS:
            df = fetch_retry(s, tf, start, end)
            if df is not None and not df.empty:
                data[s] = df
        for name, ip, mode, tp, sl in STRATS:
            port = None; total_n = 0; total_w = 0
            sub = INITIAL / max(len(data), 1)
            for s, df in data.items():
                dft = TechnicalIndicators.calculate_all_indicators(df, ip)
                signals = engine.calculate_signals(dft, ip)
                eq, n, w = simulate(df, signals, tp, sl, mode, sub)
                if eq is None or eq.empty:
                    continue
                total_n += n; total_w += w
                daily = eq.groupby(eq.index.date).last()
                port = daily if port is None else port.add(daily, fill_value=0)
            if port is None or port.empty:
                print(f"{tf:<5}{name:<12}{year:<6}无数据", flush=True)
                continue
            ret = (port.iloc[-1] / INITIAL - 1) * 100
            wr = total_w / total_n * 100 if total_n else 0
            print(f"{tf:<5}{name:<12}{year:<6}{len(data):<5}{ret:<9.1f}{total_n:<7}{wr:<7.1f}", flush=True)

# ---- BTC-ETH 配对 15m 版（E2.0/X0.5/S3.5，窗口=30天≈2880根） ----
print("\n=== BTC-ETH 配对 15m (E2.0/X0.5/S3.5, win2880) ===", flush=True)
for year, start, end in YEARS:
    btc = fetch_retry("BTC/USDT:USDT", "15m", start, end)
    eth = fetch_retry("ETH/USDT:USDT", "15m", start, end)
    if btc is None or eth is None:
        print(f"{year}: 数据不足", flush=True)
        continue
    df = btc[['close']].join(eth[['close']], lsuffix='_btc', rsuffix='_eth', how='inner').dropna()
    lr = np.log(df['close_eth'] / df['close_btc'])
    mu = lr.rolling(2880).mean().shift(1)
    sd = lr.rolling(2880).std().shift(1)
    df['z'] = (lr - mu) / sd
    df = df.dropna()
    cash = INITIAL; pos = 0; entry_btc = entry_eth = 0.0; hold = 0
    wins = 0; ntrade = 0
    for ts, row in df.iterrows():
        zz = row['z']; btc_p = row['close_btc']; eth_p = row['close_eth']
        if pos != 0:
            hold += 1
            if pos == 1:
                pnl = (eth_p / entry_eth - 1) + (1 - btc_p / entry_btc)
            else:
                pnl = (1 - eth_p / entry_eth) + (btc_p / entry_btc - 1)
            eq = cash * (1 + pnl / 2)
            closed = False
            before = cash
            if abs(zz) <= 0.5:
                cash = eq * (1 - COMM); closed = True
            elif (pos == 1 and zz <= -3.5) or (pos == -1 and zz >= 3.5):
                cash = eq * (1 - COMM); closed = True
            elif hold >= 2880:
                cash = eq * (1 - COMM); closed = True
            if closed:
                ntrade += 1
                if cash > before:
                    wins += 1
                pos = 0; hold = 0
            continue
        if zz >= 2.0:
            pos = -1
        elif zz <= -2.0:
            pos = 1
        if pos != 0:
            entry_btc, entry_eth = btc_p, eth_p
            cash *= (1 - COMM)
            hold = 0
    if pos != 0:
        last = df.iloc[-1]
        if pos == 1:
            pnl = (last['close_eth'] / entry_eth - 1) + (1 - last['close_btc'] / entry_btc)
        else:
            pnl = (1 - last['close_eth'] / entry_eth) + (last['close_btc'] / entry_btc - 1)
        cash *= (1 + pnl / 2) * (1 - COMM)
        ntrade += 1
    ret = (cash / INITIAL - 1) * 100
    wr = wins / ntrade * 100 if ntrade else 0
    print(f"{year}: 收益{ret:+.1f}% 笔数{ntrade} 胜率{wr:.0f}%", flush=True)
