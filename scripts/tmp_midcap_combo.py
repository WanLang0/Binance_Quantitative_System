# -*- coding: utf-8 -*-
"""山寨币组合策略系统测试：9组合(AND过滤) vs EMA单策略基准，10币×双向/仅多×不设/5/5×四年"""
import os, sys, io, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7892")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7892")

from collections import defaultdict
import pandas as pd
import numpy as np
from data_fetcher import BinanceDataFetcher
from indicators import TechnicalIndicators
from backtest_engine import BacktestEngine

YEARS = [("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
         ("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-08-26")]
COINS = ["DOGE", "TRX", "LINK", "DOT", "BCH", "LTC", "UNI", "APT", "ICP", "XLM"]
INITIAL = 10000.0; COMM = 0.001
MODES = ["long_short", "long_only"]
TPSL = [(None, None), (0.05, 0.05)]
TPSL_NAME = {(None, None): "无", (0.05, 0.05): "5/5"}

EMA = {"ema": True, "ema_short": 12, "ema_long": 26, "ema_periods": [12, 26]}
MACD = {"macd": True, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9}
DMA = {"ma_cross": True, "ma_cross_short": 10, "ma_cross_long": 30, "ma_cross_periods": [10, 30]}
RSI = {"rsi": True, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70}
KDJ = {"kdj": True, "kdj_k_period": 9, "kdj_d_period": 3, "kdj_j_period": 3,
       "kdj_buy_threshold": 20, "kdj_sell_threshold": 80}
BOLL = {"boll": True, "bb_period": 20, "bb_std": 2.0}

def merge(*ds):
    m = {}
    for d in ds:
        m.update(d)
    return m

COMBOS = {
    "EMA(基准)": EMA,                                   # 单策略基准（AND/OR等价）
    "EMA+MACD": merge(EMA, MACD),
    "EMA+双均线": merge(EMA, DMA),
    "双均线+MACD": merge(DMA, MACD),
    "EMA+MACD+双均线": merge(EMA, MACD, DMA),
    "EMA+RSI": merge(EMA, RSI),
    "EMA+KDJ": merge(EMA, KDJ),
    "EMA+布林": merge(EMA, BOLL),
    "MACD+RSI": merge(MACD, RSI),
    "RSI+布林": merge(RSI, BOLL),
}

def _close(cash, units, entry, price, side, comm):
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm

def simulate(df, signals, tp, sl, mode, initial=INITIAL, comm=COMM):
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
            u = (cash * 0.95) / (price * (1 + comm))
            if u > 0:
                cash -= u * price * (1 + comm); units = u; entry = price; side = 1
        elif sig == -1 and side >= 0:
            if side > 0:
                cash = _close(cash, units, entry, price, side, comm)
                n_trades += 1; side = 0; units = 0
            if mode == "long_short":
                u = (cash * 0.95) / price
                if u > 0:
                    cash -= u * price * (1 + comm); units = u; entry = price; side = -1
    if side != 0 and len(df) > 0:
        cash = _close(cash, units, entry, df['close'].iloc[-1], side, comm)
        n_trades += 1
        eq_pts.append((df.index[-1], cash))
    if n_trades == 0:
        return None  # 无交易
    eq = pd.Series(dict(eq_pts)).sort_index()
    peak = eq.cummax(); mdd = ((eq - peak) / peak * 100).min()
    return ((eq.iloc[-1] / initial - 1) * 100, mdd, n_trades)

fetcher = BinanceDataFetcher(); fetcher.set_market_type('future')
sig_and = BacktestEngine(signal_mode='and')

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

out_lines = []
def w(s=""):
    out_lines.append(s); print(s, flush=True)

records = []
no_signal_cnt = defaultdict(int)
for coin in COINS:
    sym = f"{coin}/USDT:USDT"
    print(f"\n>>> {sym}", flush=True)
    year_data = {}
    for year, start, end in YEARS:
        df = fetch(sym, start, end)
        if df is not None:
            year_data[year] = df
    if not year_data:
        continue
    for sname, ip in COMBOS.items():
        for mode in MODES:
            for tp, sl in TPSL:
                rec = {'币': coin, '策略': sname, '模式': '双向' if mode == 'long_short' else '仅多',
                       '止盈止损': TPSL_NAME[(tp, sl)], 'ret': {}, 'mdd': {}, 'trades': 0, '爆仓': 0, '无信号年': 0}
                for year, df in year_data.items():
                    dft = TechnicalIndicators.calculate_all_indicators(df, ip)
                    signals = sig_and.calculate_signals(dft, ip)
                    if signals.abs().sum() == 0:
                        rec['ret'][year] = 0.0; rec['mdd'][year] = 0.0; rec['无信号年'] += 1
                        no_signal_cnt[sname] += 1
                        continue
                    r = simulate(df, signals, tp, sl, mode)
                    if r is None:
                        rec['ret'][year] = -100.0; rec['mdd'][year] = -100.0; rec['爆仓'] += 1
                    else:
                        rec['ret'][year] = r[0]; rec['mdd'][year] = r[1]; rec['trades'] += r[2]
                records.append(rec)
                print(".", end="", flush=True)
    print(" ok", flush=True)

# ============ 汇总 ============
w("\n" + "=" * 100)
w("【A】组合普适性总表（AND过滤；10币中四年累计为正的币数 / 平均四年累计% / 爆仓币·年 / 无信号币·年）")
agg = defaultdict(list)
for rec in records:
    key = (rec['策略'], rec['模式'], rec['止盈止损'])
    rets = [rec['ret'].get(y, 0.0) for y, _, _ in YEARS]
    cum = 1.0
    for r in rets:
        cum *= (1 + r / 100)
    agg[key].append((rec['币'], (cum - 1) * 100, rec['爆仓'], rec['无信号年'], rec['trades']))
rows = []
for key, lst in agg.items():
    pos = sum(1 for _, c, _, _, _ in lst if c > 0)
    avg = sum(c for _, c, _, _, _ in lst) / len(lst)
    blow = sum(b for _, _, b, _, _ in lst)
    nsig = sum(n for _, _, _, n, _ in lst)
    trades = sum(t for _, _, _, _, t in lst)
    rows.append((key[0], key[1], key[2], pos, avg, blow, nsig, trades, len(lst)))
rows.sort(key=lambda x: (-x[3], -x[4]))
w(f"{'组合':<14}{'模式':<4}{'TP/SL':<5}{'正收益币数':>9}{'平均四年累计%':>10}{'爆仓':>5}{'无信号':>6}{'四年总交易':>8}")
for r in rows:
    w(f"{r[0]:<14}{r[1]:<4}{r[2]:<5}{r[3]}/{r[8]:<8}{r[4]:>+10.1f}{r[5]:>5}{r[6]:>6}{r[7]:>8}")

w("\n" + "=" * 100)
w("【B】vs 基准（EMA单策略 同模式同止盈止损的差值；正=组合更优）")
base_map = {r[1] + r[2]: r for r in rows if r[0] == 'EMA(基准)'}
for r in rows:
    if r[0] == 'EMA(基准)':
        continue
    b = base_map.get(r[1] + r[2])
    if b:
        w(f"  {r[0]:<14}{r[1]:<4}{r[2]:<5} 正收益 {r[3]}/{r[8]} vs 基准{b[3]}/{b[8]} | 平均累计 {r[4]:+.1f}% vs {b[4]:+.1f}% "
          f"(差{r[4]-b[4]:+.1f}pp)")

w("\n" + "=" * 100)
w("【C】Top 15 组合配置（四年累计+最差年惩罚得分）")
scored = []
for rec in records:
    rets = [rec['ret'].get(y, 0.0) for y, _, _ in YEARS if y in rec['ret']]
    if len(rets) < 4 or rec['爆仓'] > 0 or rec['trades'] < 8:
        continue
    cum = 1.0
    for r in rets:
        cum *= (1 + r / 100)
    cum = (cum - 1) * 100
    scored.append((cum + min(rets) * 2, cum, min(rets), rec))
scored.sort(key=lambda x: -x[0])
w(f"{'币':<6}{'组合':<14}{'模式':<4}{'TP/SL':<5}{'四年累计%':>9}{'最差年%':>8}{'交易':>5}")
for s, cum, worst, rec in scored[:15]:
    w(f"{rec['币']:<6}{rec['策略']:<14}{rec['模式']:<4}{rec['止盈止损']:<5}{cum:>+9.1f}{worst:>+8.1f}{rec['trades']:>5}")

w("\n【D】每币最优（组合是否打败该币的EMA单策略基准）")
for coin in COINS:
    best, base = None, None
    for rec in records:
        if rec['币'] != coin:
            continue
        rets = [rec['ret'].get(y, 0.0) for y, _, _ in YEARS]
        cum = 1.0
        for r in rets:
            cum *= (1 + r / 100)
        cum = (cum - 1) * 100
        score = cum + min(rets) * 2
        if rec['策略'] == 'EMA(基准)' and rec['模式'] == '双向' and rec['止盈止损'] == '5/5':
            base = cum
        if best is None or score > best[0]:
            best = (score, cum, rec)
    if best:
        rec = best[2]
        tag = f"（EMA双向5/5基准{base:+.0f}%）" if base is not None else ""
        w(f"  {coin:<5} {rec['策略']:<14} {rec['模式']:<3} {rec['止盈止损']:<4} 四年{best[1]:+8.1f}% {tag}")

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "midcap_combo_summary.txt")
with open(out_file, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))
print(f"\nSaved -> {out_file}", flush=True)
