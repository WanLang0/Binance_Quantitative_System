# -*- coding: utf-8 -*-
"""缠论可量化子集验证：
  一买近似 = 底分型(3根K线中间低点最低) + 前期跌幅>X%（超跌抄底，同 KDJ 家族）
  三买近似 = 突破N根高点后回踩至突破位±2%不破、企稳转涨入场（趋势回踩，同 EMA 家族）
  三卖近似 = 三买镜像（跌破N根低点反弹不破、转跌入场做空）
  注：简化版，未做缠论K线包含关系合并；分型用 low/high 判定
  基准对比：ETH 4h KDJ 8/5 仅多（四年+40.5%）| XLM 4h EMA 双向 5/5（四年+239.1%）
"""
import os, sys, io, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7892")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7892")

from collections import defaultdict
import numpy as np
import pandas as pd
from data_fetcher import BinanceDataFetcher

YEARS = [("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
         ("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-08-26")]
COINS = ["ETH", "XLM", "BTC", "BCH", "LTC", "ADA"]
INITIAL = 10000.0; COMM = 0.001

# 一买参数：前期回撤阈值X × 回看窗口M（4h根）
B1_PARAMS = [(0.10, 120), (0.15, 120), (0.20, 180), (0.30, 180)]
# 一买出场：(tp, sl, max_hold)
B1_EXITS = [(0.08, 0.05, None), (0.05, 0.05, None), (None, None, 42)]
# 三买参数：突破窗口N
B3_PARAMS = [20, 30, 42]
# 三买出场
B3_EXITS = [(0.05, 0.05, None), (0.08, 0.05, None), (None, None, 90)]

fetcher = BinanceDataFetcher(); fetcher.set_market_type('future')

def fetch(sym, start, end, tries=6):
    for i in range(tries):
        try:
            df = fetcher.fetch_historical_data(sym, start, end, "4h")
            if df is not None and not df.empty and len(df) >= 100:
                return df
        except Exception:
            pass
        time.sleep(2 * (i + 1))
    return None

# ---------------- 信号生成 ----------------
def first_buy_signal(df, drop, lookback):
    """一买近似：底分型 + 前期高点回撤超 drop。返回 +1 信号 Series"""
    low = df['low'].values
    close = df['close'].values
    n = len(df)
    sig = np.zeros(n)
    # 前期高点（lookback 根内的最高 high，截至 i-1）
    hh = pd.Series(df['high']).rolling(lookback).max().shift(1).values
    for i in range(1, n - 1):
        # 底分型：中间K线 low 严格低于两侧
        if not (low[i] < low[i - 1] and low[i] < low[i + 1]):
            continue
        if np.isnan(hh[i]):
            continue
        if low[i] / hh[i] - 1 <= -drop:      # 从前期高点回撤超过阈值
            sig[i + 1] = 1                    # 分型确认后下一根入场
    return pd.Series(sig, index=df.index)

def third_buy_signal(df, win):
    """三买/三卖近似：突破win根高点→2~14根内回踩突破位上方≤2%→收盘再创回踩前新高入场(+1)
       镜像：跌破win根低点→反弹→收盘再创回踩前新低入场(-1)。返回信号 Series"""
    high = df['high'].values; low = df['low'].values; close = df['close'].values
    n = len(df)
    sig = np.zeros(n)
    hh = pd.Series(high).rolling(win).max().shift(1).values
    ll = pd.Series(low).rolling(win).min().shift(1).values
    i = 0
    while i < n:
        if np.isnan(hh[i]) or np.isnan(ll[i]):
            i += 1; continue
        if close[i] > hh[i]:
            level, direction = hh[i], 1
        elif close[i] < ll[i]:
            level, direction = ll[i], -1
        else:
            i += 1; continue
        j = i + 1
        pb_done = False; pivot = 0.0; entered_j = None
        while j < n and j - i <= 14:
            if direction == 1:
                if close[j] < level:            # 收盘跌回突破位下：突破失败
                    break
                if not pb_done:
                    if low[j] <= level * 1.02:  # 回踩到突破位附近
                        pb_done = True
                        pivot = high[i:j].max() # 回踩前的局部高点（不含回踩根）
                elif close[j] > pivot:          # 收盘再创回踩前新高 → 三买入场
                    sig[j] = 1; entered_j = j; break
            else:
                if close[j] > level:
                    break
                if not pb_done:
                    if high[j] >= level * 0.98: # 反弹到跌破位附近
                        pb_done = True
                        pivot = low[i:j].min()
                elif close[j] < pivot:          # 收盘再创回踩前新低 → 三卖入场
                    sig[j] = -1; entered_j = j; break
            j += 1
        i = (entered_j + 1) if entered_j is not None else (j + 1)
    return pd.Series(sig, index=df.index)

# ---------------- 双向模拟器（带最大持有期） ----------------
def _close(cash, units, entry, price, side, comm=COMM):
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm

def simulate(df, signals, tp, sl, mode, max_hold=None, initial=INITIAL):
    cash = initial; units = 0.0; entry = 0.0; side = 0; hold = 0
    eq_pts = []; n_trades = 0
    close = df['close'].values
    sig = signals.values
    for i in range(len(df)):
        price = close[i]
        if not np.isfinite(price) or price <= 0:
            continue
        s = int(sig[i]) if i > 0 else 0
        if side != 0 and entry > 0:
            r = (price - entry) / entry if side > 0 else (entry - price) / entry
            if (tp and r >= tp) or (sl and r <= -sl) or (max_hold and hold >= max_hold):
                cash = _close(cash, units, entry, price, side)
                side = 0; units = 0; entry = 0; hold = 0; n_trades += 1
                eq_pts.append((df.index[i], cash)); continue
        if side > 0:
            eq = cash + units * price
        elif side < 0:
            eq = cash + units * entry + (entry - price) * units
        else:
            eq = cash
        if eq <= 0:
            return None
        eq_pts.append((df.index[i], eq))
        if s == 1 and side <= 0:
            if side < 0:
                cash = _close(cash, units, entry, price, side)
                n_trades += 1; side = 0; units = 0; hold = 0
            u = (cash * 0.95) / (price * (1 + COMM))
            if u > 0:
                cash -= u * price * (1 + COMM); units = u; entry = price; side = 1; hold = 0
        elif s == -1 and side >= 0:
            if side > 0:
                cash = _close(cash, units, entry, price, side)
                n_trades += 1; side = 0; units = 0; hold = 0
            if mode == 'long_short':
                u = (cash * 0.95) / price
                if u > 0:
                    cash -= u * price * (1 + COMM); units = u; entry = price; side = -1; hold = 0
        if side != 0:
            hold += 1
    if side != 0 and len(df) > 0:
        cash = _close(cash, units, entry, close[-1], side)
        n_trades += 1
        eq_pts.append((df.index[-1], cash))
    if n_trades == 0:
        return None
    eq = pd.Series(dict(eq_pts)).sort_index()
    return ((eq.iloc[-1] / initial - 1) * 100, n_trades)

# ---------------- 主流程 ----------------
out_lines = []
def w(s=""):
    out_lines.append(s); print(s, flush=True)

data = {}
for coin in COINS:
    sym = f"{coin}/USDT:USDT"
    dfs = []
    for year, start, end in YEARS:
        df = fetch(sym, start, end)
        if df is not None:
            dfs.append(df)
    if dfs:
        full = pd.concat(dfs).sort_index()
        data[coin] = full[~full.index.duplicated(keep='first')]
        print(f"{coin}: {len(data[coin])}根", flush=True)

# 预生成信号
b1_sigs = {}   # (coin, drop, lb) -> Series
for coin, df in data.items():
    for drop, lb in B1_PARAMS:
        b1_sigs[(coin, drop, lb)] = first_buy_signal(df, drop, lb)
b3_sigs = {}
for coin, df in data.items():
    for win in B3_PARAMS:
        b3_sigs[(coin, win)] = third_buy_signal(df, win)

w("\n" + "=" * 100)
w("【A】一买近似（底分型+前期超跌，仅做多）：6币 × 4参数 × 3出场 = 72配置 × 4年")
rows = []
for drop, lb in B1_PARAMS:
    for tp, sl, mh in B1_EXITS:
        label = f"回撤{int(drop*100)}%/{lb}根 tp{int(tp*100) if tp else '无'}/sl{int(sl*100) if sl else '无'}" + \
                (f" 持有{mh}根" if mh else "")
        rets = {}; trades = 0; valid = 0
        for coin in COINS:
            if coin not in data:
                continue
            yearly = {}
            for year, start, end in YEARS:
                mask = (data[coin].index >= pd.Timestamp(start, tz=data[coin].index.tz)) if data[coin].index.tz else \
                       (data[coin].index >= start)
                dyy = data[coin][mask]
                # 按年切片回测（信号取交集）
                sub = dyy[dyy.index <= (end + ' 23:59')]
                sig = b1_sigs[(coin, drop, lb)].reindex(sub.index).fillna(0)
                r = simulate(sub, sig, tp, sl, 'long_only', mh)
                yearly[year] = r[0] if r else None
                trades += r[1] if r else 0
            vals = [v for v in yearly.values() if v is not None]
            if vals:
                cum = 1.0
                for v in vals:
                    cum *= (1 + v / 100)
                rets[coin] = (cum - 1) * 100; valid += 1
        pos = sum(1 for v in rets.values() if v > 0)
        avg = sum(rets.values()) / len(rets) if rets else 0
        detail = " ".join(f"{c}{rets[c]:+.0f}%" for c in COINS if c in rets)
        rows.append(('一买', label, pos, valid, avg, trades, detail))
        w(f"  {label:<38} 正收益{pos}/{valid} 平均{avg:+7.1f}% 共{trades}笔 | {detail}")

w("\n" + "=" * 100)
w("【B】三买/三卖近似（突破回踩，双向 vs 仅多）：6币 × 3参数 × 3出场 × 2模式")
for win in B3_PARAMS:
    for tp, sl, mh in B3_EXITS:
        for mode, mlabel in [('long_only', '仅多'), ('long_short', '双向')]:
            label = f"N={win} tp{int(tp*100) if tp else '无'}/sl{int(sl*100) if sl else '无'}" + \
                    (f" 持有{mh}根" if mh else "") + f" {mlabel}"
            rets = {}; trades = 0
            for coin in COINS:
                if coin not in data:
                    continue
                yearly = {}
                for year, start, end in YEARS:
                    sub = data[coin][(data[coin].index >= pd.Timestamp(start)) & (data[coin].index <= pd.Timestamp(end))]
                    if len(sub) < 100:
                        continue
                    sig = b3_sigs[(coin, win)].reindex(sub.index).fillna(0)
                    r = simulate(sub, sig, tp, sl, mode, mh)
                    yearly[year] = r[0] if r else None
                    trades += r[1] if r else 0
                vals = [v for v in yearly.values() if v is not None]
                if vals:
                    cum = 1.0
                    for v in vals:
                        cum *= (1 + v / 100)
                    rets[coin] = (cum - 1) * 100
            pos = sum(1 for v in rets.values() if v > 0)
            avg = sum(rets.values()) / len(rets) if rets else 0
            detail = " ".join(f"{c}{rets[c]:+.0f}%" for c in COINS if c in rets)
            rows.append(('三买', label, pos, len(rets), avg, trades, detail))
            w(f"  {label:<44} 正收益{pos}/{len(rets)} 平均{avg:+7.1f}% 共{trades}笔 | {detail}")

w("\n" + "=" * 100)
w("【C】与基准对比")
w("  基准1: ETH 4h KDJ 8/5 仅多      四年 +40.5%（ETH）")
w("  基准2: XLM 4h EMA 双向 5/5      四年 +239.1%（XLM）")
best_b1 = max([r for r in rows if r[0] == '一买'], key=lambda r: r[4])
best_b3 = max([r for r in rows if r[0] == '三买'], key=lambda r: r[4])
w(f"  一买最佳: {best_b1[1]} | 正收益{best_b1[2]}/{best_b1[3]} 平均{best_b1[4]:+.1f}%")
w(f"  三买最佳: {best_b3[1]} | 正收益{best_b3[2]}/{best_b3[3]} 平均{best_b3[4]:+.1f}%")

out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "chan_test_summary.txt")
os.makedirs(os.path.dirname(out_file), exist_ok=True)
with open(out_file, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))
print(f"\nSaved -> {out_file}", flush=True)
