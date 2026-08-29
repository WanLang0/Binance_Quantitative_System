# -*- coding: utf-8 -*-
"""多币种扫描组合回测：同一策略同时监控15个主流币（各1/15资金独立运行），
统计组合层面的年度收益/回撤，验证"时刻监控大多数币、随时多空"是否优于单币。"""
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
SYMBOLS = [f"{b}/USDT:USDT" for b in
           ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "TRX",
            "AVAX", "LINK", "DOT", "LTC", "BCH", "NEAR", "TON"]]
YEARS = [("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
         ("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-08-26")]
BASE = {
    "KDJ": {"kdj": True, "kdj_k_period": 9, "kdj_d_period": 3, "kdj_j_period": 3,
            "kdj_buy_threshold": 20, "kdj_sell_threshold": 80},
    "MACD": {"macd": True, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
    "RSI": {"rsi": True, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70},
    "EMA": {"ema": True, "ema_short": 12, "ema_long": 26, "ema_periods": [12, 26]},
}
# (名称, 策略参数, 模式, 止盈, 止损)
STRATS = [
    ("KDJ 8/5 仅多", BASE["KDJ"], "long_only", 0.08, 0.05),
    ("KDJ 双向 5/5", BASE["KDJ"], "long_short", 0.05, 0.05),
    ("MACD 双向 5/5", BASE["MACD"], "long_short", 0.05, 0.05),
    ("KDJ+MACD 双向 5/5", {**BASE["KDJ"], **BASE["MACD"]}, "long_short", 0.05, 0.05),
]

def _close(cash, units, entry, price, side, comm):
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm

def simulate(df, signals, tp, sl, mode, initial):
    """返回(日权益Series, 交易数, 胜数)"""
    cash = initial; units = 0.0; entry = 0.0; side = 0
    cash_at_open = None; wins = 0; n = 0
    eq_pts = []
    for i, (ts, row) in enumerate(df.iterrows()):
        price = row['close']
        if not np.isfinite(price) or price <= 0:
            continue
        sig = int(signals.iloc[i]) if i > 0 else 0
        if side != 0 and entry > 0:
            r = (price - entry) / entry if side > 0 else (entry - price) / entry
            if (tp and r >= tp) or (sl and r <= -sl):
                cash = _close(cash, units, entry, price, side, COMM)
                n += 1
                if cash > cash_at_open:
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
                cash = _close(cash, units, entry, price, side, COMM)
                n += 1
                if cash > cash_at_open:
                    wins += 1
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
                n += 1
                if cash > cash_at_open:
                    wins += 1
                side = 0; units = 0
            if mode == "long_short":
                notional = cash * 0.95
                u = notional / price
                if u > 0:
                    cash_at_open = cash
                    cash -= u * price * (1 + COMM)
                    units = u; entry = price; side = -1
    if side != 0 and eq_pts:
        cash = _close(cash, units, entry, df['close'].iloc[-1], side, COMM)
        n += 1
        if cash > cash_at_open:
            wins += 1
        eq_pts.append((df.index[-1], cash))
    eq = pd.Series(dict(eq_pts)).sort_index()
    daily = eq.groupby(eq.index.date).last()
    return daily, n, wins

fetcher = BinanceDataFetcher(); fetcher.set_market_type('future')
engine = BacktestEngine(signal_mode='or')

def fetch_retry(sym, start, end, tries=6):
    for i in range(tries):
        df = fetcher.fetch_historical_data(sym, start, end, "4h")
        if df is not None and len(df) >= 100:
            return df
        time.sleep(2 * (i + 1))
    return df

print("策略                  年     币数  组合收益%  组合回撤%  总交易  胜率%  盈利币数", flush=True)
for year, start, end in YEARS:
    # 预拉数据（按币缓存）
    data = {}
    for s in SYMBOLS:
        df = fetch_retry(s, start, end)
        if df is not None and not df.empty:
            data[s] = df
    for name, ip, mode, tp, sl in STRATS:
        port = None; total_trades = 0; total_wins = 0; profit_coins = 0; n_coins = 0
        sub = INITIAL / max(len(data), 1)
        for s, df in data.items():
            dft = TechnicalIndicators.calculate_all_indicators(df, ip)
            signals = engine.calculate_signals(dft, ip)
            daily, n, wins = simulate(df, signals, tp, sl, mode, sub)
            if daily is None or daily.empty:
                continue
            n_coins += 1
            total_trades += n; total_wins += wins
            if daily.iloc[-1] > sub:
                profit_coins += 1
            port = daily if port is None else port.add(daily, fill_value=0)
        if port is None or port.empty:
            print(f"{name:<22}{year:<6}无数据", flush=True)
            continue
        ret = (port.iloc[-1] / INITIAL - 1) * 100
        peak = port.expanding().max()
        mdd = ((port - peak) / peak * 100).min()
        wr = total_wins / total_trades * 100 if total_trades else 0
        print(f"{name:<22}{year:<6}{n_coins:<6}{ret:<10.1f}{mdd:<10.1f}{total_trades:<7}{wr:<6.1f}{profit_coins}/{n_coins}", flush=True)
