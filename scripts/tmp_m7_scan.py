# -*- coding: utf-8 -*-
"""美股七姐妹代币策略扫描：7币×7策略×2模式×2止盈止损，全样本+前半后半稳定性对照（真合约可双向做空）"""
import os, sys, io, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7892")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7892")

from collections import defaultdict
import numpy as np
import pandas as pd
from data_fetcher import BinanceDataFetcher
from indicators import TechnicalIndicators
from backtest_engine import BacktestEngine

COINS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
MODES = ["long_short", "long_only"]
TPSL = [(None, None), (0.05, 0.05)]
TPSL_NAME = {(None, None): "无", (0.05, 0.05): "5/5"}

RSI = {"rsi": True, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70}
KDJ = {"kdj": True, "kdj_k_period": 9, "kdj_d_period": 3, "kdj_j_period": 3,
       "kdj_buy_threshold": 20, "kdj_sell_threshold": 80}
MACD = {"macd": True, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9}
EMA = {"ema": True, "ema_short": 12, "ema_long": 26, "ema_periods": [12, 26]}
DMA = {"ma_cross": True, "ma_cross_short": 10, "ma_cross_long": 30, "ma_cross_periods": [10, 30]}
BOLL = {"boll": True, "bb_period": 20, "bb_std": 2.0}
def merge(*ds):
    m = {}
    for d in ds:
        m.update(d)
    return m
COMBOS = {"KDJ": KDJ, "RSI": RSI, "MACD": MACD, "EMA": EMA, "双均线": DMA, "布林带": BOLL,
          "RSI+MACD": merge(RSI, MACD)}

fetcher = BinanceDataFetcher(); fetcher.set_market_type('future')

def fetch(sym):
    for i in range(5):
        try:
            df = fetcher.fetch_historical_data(sym, "2026-01-01", "2026-08-30", "4h")
            if df is not None and len(df) >= 200:
                return df
        except Exception:
            pass
        time.sleep(3)
    return None

def _close(cash, units, entry, price, side, comm=0.001):
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm

def simulate(df, signals, tp, sl, mode, initial=10000.0):
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
            if mode == "long_short":
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
    peak = eq.cummax()
    return {'ret': (eq.iloc[-1] / initial - 1) * 100,
            'mdd': ((eq - peak) / peak * 100).min(), 'trades': n}

out_lines = []
def w(s=""):
    out_lines.append(s); print(s, flush=True)

sig_engine = BacktestEngine(signal_mode='or')
records = []
for coin in COINS:
    sym = f"{coin}/USDT:USDT"
    df = fetch(sym)
    if df is None:
        w(f"{sym}: 数据不足，跳过"); continue
    days = (df.index[-1] - df.index[0]).days
    half = df.index[len(df) // 2]
    df_a, df_b = df[df.index < half], df[df.index >= half]
    w(f">>> {coin}: {len(df)}根/{days}天（前半{len(df_a)} 后半{len(df_b)}）")
    # 标的自身表现
    w(f"    标的自身: 全程{(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:+.1f}% | "
      f"前半{(df_a['close'].iloc[-1]/df_a['close'].iloc[0]-1)*100:+.1f}% | "
      f"后半{(df_b['close'].iloc[-1]/df_b['close'].iloc[0]-1)*100:+.1f}%")
    for sname, ip in COMBOS.items():
        dft = TechnicalIndicators.calculate_all_indicators(df, ip)
        signals = sig_engine.calculate_signals(dft, ip)
        sa = signals.reindex(df_a.index).fillna(0)
        sb = signals.reindex(df_b.index).fillna(0)
        for mode in MODES:
            for tp, sl in TPSL:
                rec = {'coin': coin, 'strat': sname, 'mode': '双向' if mode == 'long_short' else '仅多',
                       'tpsl': TPSL_NAME[(tp, sl)]}
                r_full = simulate(df, signals, tp, sl, mode)
                r_a = simulate(df_a, sa, tp, sl, mode)
                r_b = simulate(df_b, sb, tp, sl, mode)
                rec['full'] = r_full['ret'] if r_full else None
                rec['mdd'] = r_full['mdd'] if r_full else None
                rec['trades'] = r_full['trades'] if r_full else 0
                rec['half_a'] = r_a['ret'] if r_a else None
                rec['half_b'] = r_b['ret'] if r_b else None
                records.append(rec)
        print(".", end="", flush=True)
    print(" ok", flush=True)

df_all = pd.DataFrame(records)
w("\n" + "=" * 100)
w("【A】策略普适性（7币全样本为正的币数 / 平均收益%）")
agg = df_all.groupby(['strat', 'mode', 'tpsl']).agg(
    pos=('full', lambda s: sum(1 for v in s if v is not None and v > 0)),
    avg=('full', lambda s: np.mean([v for v in s if v is not None]) if any(v is not None for v in s) else 0),
    blow=('full', lambda s: sum(1 for v in s if v is None))).reset_index()
agg = agg.sort_values(['pos', 'avg'], ascending=False)
w(f"{'策略':<10}{'模式':<4}{'TP/SL':<5}{'正收益币数':>8}{'平均收益%':>9}{'爆仓':>4}")
for _, r in agg.iterrows():
    w(f"{r['strat']:<10}{r['mode']:<4}{r['tpsl']:<5}{int(r['pos'])}/7{'':<5}{r['avg']:>+9.1f}{int(r['blow']):>4}")

w("\n【B】每币 Top3 配置（全样本收益，附前半/后半对照）")
for coin in COINS:
    sub = df_all[(df_all['coin'] == coin) & df_all['full'].notna()].nlargest(3, 'full')
    for _, r in sub.iterrows():
        ha = f"{r['half_a']:+.0f}%" if r['half_a'] is not None else '--'
        hb = f"{r['half_b']:+.0f}%" if r['half_b'] is not None else '--'
        w(f"  {coin:<6}{r['strat']:<10}{r['mode']:<3}{r['tpsl']:<4} 全程{r['full']:+8.1f}% 回撤{r['mdd']:>6.1f}% "
          f"{int(r['trades']):>3}笔 | 前半{ha:>7} 后半{hb:>7}")

w("\n【C】全样本Top10（含稳定性：前半/后半同号才算稳）")
top = df_all[df_all['full'].notna()].nlargest(10, 'full')
stable_cnt = 0
for _, r in top.iterrows():
    ha, hb = r['half_a'], r['half_b']
    stable = (ha is not None and hb is not None and ha > 0 and hb > 0)
    if stable:
        stable_cnt += 1
    w(f"  {r['coin']:<6}{r['strat']:<10}{r['mode']:<3}{r['tpsl']:<4} {r['full']:+8.1f}% "
      f"{'★前后半都赚' if stable else '（单段行情依赖）'}")

out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "m7_scan.txt")
os.makedirs(os.path.dirname(out_file), exist_ok=True)
with open(out_file, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))
print(f"\nSaved -> {out_file}", flush=True)
