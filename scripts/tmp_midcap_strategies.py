# -*- coding: utf-8 -*-
"""市值10-20名高波动山寨币：多策略×双向/仅多×止盈止损 四年矩阵，找跨币跨年稳健配置"""
import os, sys, io, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
os.environ.setdefault("HTTP_PROXY", "http://192.168.11.188:7892")
os.environ.setdefault("HTTPS_PROXY", "http://192.168.11.188:7892")

import pandas as pd
import numpy as np
from data_fetcher import BinanceDataFetcher
from indicators import TechnicalIndicators
from backtest_engine import BacktestEngine

YEARS = [("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
         ("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-08-26")]
# 市值10-20名高波动币（上轮KDJ已测的 ADA/AVAX/TON/NEAR/XRP 不重复，另加 DOGE/TRX/LINK/DOT/BCH/LTC/UNI/APT/ICP/XLM）
COINS = ["DOGE", "TRX", "LINK", "DOT", "BCH", "LTC", "UNI", "APT", "ICP", "XLM"]
INITIAL = 10000.0; COMM = 0.001
MODES = ["long_short", "long_only"]   # 双向 / 仅做多
TPSL = [(0.08, 0.05), (0.05, 0.05), (None, None)]  # 8/5（基准同款）、5/5、不设
TPSL_NAME = {(0.08, 0.05): "8/5", (0.05, 0.05): "5/5", (None, None): "无"}

BASE = {
    "KDJ": {"kdj": True, "kdj_k_period": 9, "kdj_d_period": 3, "kdj_j_period": 3,
            "kdj_buy_threshold": 20, "kdj_sell_threshold": 80},
    "RSI": {"rsi": True, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70},
    "MACD": {"macd": True, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
    "EMA": {"ema": True, "ema_short": 12, "ema_long": 26, "ema_periods": [12, 26]},
    "双均线": {"ma_cross": True, "ma_cross_short": 10, "ma_cross_long": 30,
               "ma_cross_periods": [10, 30]},
    "布林带": {"boll": True, "bb_period": 20, "bb_std": 2.0},
}
COMBOS = dict(BASE)
merged = dict(BASE["RSI"]); merged.update(BASE["布林带"])
COMBOS["RSI+布林"] = merged

def _close(cash, units, entry, price, side, comm):
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm

def simulate(df, signals, tp, sl, mode, initial=INITIAL, comm=COMM):
    """双向模拟(1倍保证金做空)，返回 (年收益%, 最大回撤%, 交易数) 或 None(爆仓/无交易)"""
    cash = initial; units = 0.0; entry = 0.0; side = 0
    eq_pts = []; n_trades = 0
    for i, (ts, row) in enumerate(df.iterrows()):
        price = row['close']
        if not np.isfinite(price) or price <= 0:
            continue
        sig = int(signals.iloc[i]) if i > 0 else 0
        if side != 0 and entry > 0:
            r = (price - entry) / entry if side > 0 else (entry - price) / entry
            if (tp and r >= tp) or (sl and r <= -sl):
                cash = _close(cash, units, entry, price, side, comm)
                side = 0; units = 0; entry = 0; n_trades += 1
                eq_pts.append((ts, cash)); continue
        if side > 0:
            eq = cash + units * price
        elif side < 0:
            eq = cash + units * entry + (entry - price) * units
        else:
            eq = cash
        if eq <= 0:
            return None  # 爆仓
        eq_pts.append((ts, eq))
        if sig == 1 and side <= 0:
            if side < 0:
                cash = _close(cash, units, entry, price, side, comm)
                n_trades += 1; side = 0; units = 0
            invest = cash * 0.95
            u = invest / (price * (1 + comm))
            if u > 0:
                cash -= u * price * (1 + comm)
                units = u; entry = price; side = 1
        elif sig == -1 and side >= 0:
            if side > 0:
                cash = _close(cash, units, entry, price, side, comm)
                n_trades += 1; side = 0; units = 0
            if mode == "long_short":
                notional = cash * 0.95
                u = notional / price
                if u > 0:
                    cash -= u * price * (1 + comm)
                    units = u; entry = price; side = -1
    if side != 0 and len(df) > 0:
        cash = _close(cash, units, entry, df['close'].iloc[-1], side, comm)
        n_trades += 1
        eq_pts.append((df.index[-1], cash))
    if n_trades == 0:
        return None
    eq = pd.Series(dict(eq_pts)).sort_index()
    peak = eq.cummax(); mdd = ((eq - peak) / peak * 100).min()
    return ((eq.iloc[-1] / initial - 1) * 100, mdd, n_trades)

fetcher = BinanceDataFetcher(); fetcher.set_market_type('future')
sig_engine = BacktestEngine(signal_mode='or')

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

records = []   # {币, 策略, 模式, tpsl, 年: (ret, mdd, trades)}
for coin in COINS:
    sym = f"{coin}/USDT:USDT"
    print(f"\n>>> {sym}", flush=True)
    year_data = {}
    for year, start, end in YEARS:
        df = fetch(sym, start, end)
        if df is None:
            print(f"  {year}: 拉取失败", flush=True); continue
        year_data[year] = df
    if not year_data:
        continue
    for sname, ip in COMBOS.items():
        for mode in MODES:
            for tp, sl in TPSL:
                rec = {'币': coin, '策略': sname, '模式': '双向' if mode == 'long_short' else '仅多',
                       '止盈止损': TPSL_NAME[(tp, sl)], 'ret': {}, 'mdd': {}, 'trades': 0, '爆仓': 0}
                valid = True
                for year, df in year_data.items():
                    dft = TechnicalIndicators.calculate_all_indicators(df, ip)
                    signals = sig_engine.calculate_signals(dft, ip)
                    if signals.abs().sum() == 0:
                        rec['ret'][year] = 0.0; rec['mdd'][year] = 0.0; continue
                    r = simulate(df, signals, tp, sl, mode)
                    if r is None:
                        # 区分爆仓(有信号)与无交易
                        rec['ret'][year] = -100.0; rec['mdd'][year] = -100.0; rec['爆仓'] += 1
                    else:
                        rec['ret'][year] = r[0]; rec['mdd'][year] = r[1]; rec['trades'] += r[2]
                recs_ret = [rec['ret'].get(y) for y, _, _ in YEARS if y in rec['ret']]
                if not recs_ret:
                    valid = False
                if valid:
                    records.append(rec)
                print(".", end="", flush=True)
    print(" ok", flush=True)

# ============ 汇总输出 ============
out_lines = []
def w(s=""):
    out_lines.append(s); print(s, flush=True)

w("\n" + "=" * 100)
w("【A】每币最优配置（四年累计最高；括号=各年收益%，*为爆仓年）")
for coin in COINS:
    best, best_key = None, None
    for rec in records:
        if rec['币'] != coin:
            continue
        rets = [rec['ret'].get(y, 0.0) for y, _, _ in YEARS]
        cum = 1.0
        for r in rets:
            cum *= (1 + r / 100)
        cum = (cum - 1) * 100
        score = cum + min(rets) * 2  # 累计收益 + 最差年加权（惩罚灾难年）
        if best is None or score > best[0]:
            best = (score, cum, rec); 
    if best:
        rec = best[2]
        ystr = " ".join(f"{rec['ret'].get(y, 0):+.0f}{'*' if rec['ret'].get(y, 0) <= -99 else ''}"
                        for y, _, _ in YEARS if y in rec['ret'])
        w(f"  {coin:<5} {rec['策略']:<8} {rec['模式']:<3} {rec['止盈止损']:<4} 四年{best[1]:+8.1f}% | 各年: {ystr} | 共{rec['trades']}笔")

w("\n" + "=" * 100)
w("【B】策略普适性（按策略×模式×止盈止损：10币中四年累计为正的币数 / 平均四年累计%）")
from collections import defaultdict
agg = defaultdict(list)
for rec in records:
    key = (rec['策略'], rec['模式'], rec['止盈止损'])
    rets = [rec['ret'].get(y, 0.0) for y, _, _ in YEARS if y in rec['ret']]
    cum = 1.0
    for r in rets:
        cum *= (1 + r / 100)
    agg[key].append((rec['币'], (cum - 1) * 100, rec['爆仓']))
rows = []
for key, lst in agg.items():
    pos = sum(1 for _, c, _ in lst if c > 0)
    avg = sum(c for _, c, _ in lst) / len(lst)
    blow = sum(b for _, _, b in lst)
    rows.append((key[0], key[1], key[2], pos, avg, blow, len(lst)))
rows.sort(key=lambda x: (-x[3], -x[4]))
w(f"{'策略':<8}{'模式':<4}{'止盈止损':<5}{'正收益币数':>8}{'平均四年累计%':>10}{'爆仓年数':>7}")
for r in rows:
    w(f"{r[0]:<8}{r[1]:<4}{r[2]:<5}{r[3]}/{r[6]:<7}{r[4]:>+10.1f}{r[5]:>7}")

w("\n" + "=" * 100)
w("【C】Top 15 稳健配置（四年累计高 + 最差年惩罚后得分最高）")
scored = []
for rec in records:
    rets = [rec['ret'].get(y, 0.0) for y, _, _ in YEARS if y in rec['ret']]
    if len(rets) < 4 or rec['爆仓'] > 0:
        continue
    cum = 1.0
    for r in rets:
        cum *= (1 + r / 100)
    cum = (cum - 1) * 100
    scored.append((cum + min(rets) * 2, cum, min(rets), max(rets), rec))
scored.sort(key=lambda x: -x[0])
w(f"{'币':<6}{'策略':<8}{'模式':<4}{'TP/SL':<5}{'四年累计%':>9}{'最差年%':>8}{'最好年%':>8}")
for s, cum, worst, best_y, rec in scored[:15]:
    w(f"{rec['币']:<6}{rec['策略']:<8}{rec['模式']:<4}{rec['止盈止损']:<5}{cum:>+9.1f}{worst:>+8.1f}{best_y:>+8.1f}")

w("\n【D】跨币复现配置（同一策略配置在>=4个币上四年为正的，附币名单）")
for key, lst in agg.items():
    pos = [(c0, c) for c0, c, _ in lst if c > 0]
    if len(pos) >= 4:
        w(f"  {key[0]} {key[1]} {key[2]}: {len(pos)}币为正 -> "
          + ", ".join(f"{c0}{c:+.0f}%" for c0, c in sorted(pos, key=lambda x: -x[1])))

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "midcap_strategies_summary.txt")
with open(out_file, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))
print(f"\nSaved -> {out_file}", flush=True)
