# -*- coding: utf-8 -*-
"""一致性抽检：美股代币×5 + 加密货币×5 随机抽样，按表格列条件（策略/模式/止盈止损/周期/日期）
独立重跑回测，与库中记录对比收益/回撤/夏普。固定随机种子可复现。"""
import os, sys, io, time, json, random, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7892")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7892")

import numpy as np
import pandas as pd
import sqlite3
from data_fetcher import BinanceDataFetcher
from indicators import TechnicalIndicators
from backtest_engine import BacktestEngine

# ---- 指标参数（与库中记录口径一致）----
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

def parse_strategy(s):
    """表格'策略'列文本 → 指标参数（按关键字，顺序防误配：组合优先）"""
    s = s or ''
    has = lambda *ws: all(w in s for w in ws)
    if has('RSI') and has('MACD'):
        return merge(RSI, MACD)
    if has('RSI') and has('EMA'):
        return merge(RSI, EMA)
    if has('EMA') and has('MACD'):
        return merge(EMA, MACD)
    if 'KDJ' in s:
        return KDJ
    if '双均线' in s:
        return DMA
    if '布林' in s:
        return BOLL
    if 'RSI' in s:
        return RSI
    if 'MACD' in s:
        return MACD
    if 'EMA' in s:
        return EMA
    return None

def parse_period(p):
    """表格'日期'列 → 回测窗口列表（四年则逐年链式）"""
    p = p or ''
    if '四年' in p:
        return [("2023-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31"),
                ("2025-01-01", "2025-12-31"), ("2026-01-01", "2026-08-26")]
    if '2025' in p:
        return [("2025-01-01", "2025-12-31")]
    if '各股自上市' in p:
        return [("2026-01-01", "2026-08-30")]
    return [("2026-01-01", "2026-08-26")]  # 2026年X~8月

def parse_tpsl(s):
    s = (s or '').strip()
    if s in ('不设', '', '—', '任意', None):
        return None, None
    import re
    m = re.match(r'(\d+)%/(\d+)%', s)
    return (int(m.group(1)) / 100, int(m.group(2)) / 100) if m else (None, None)

# ---- 抽样 ----
c = sqlite3.connect('data/users.db')
c.row_factory = sqlite3.Row
rows = c.execute("SELECT symbol, timeframe, strategy, mode, tpsl, period, ret, mdd, sharpe "
                 "FROM strategy_records WHERE trades_detail IS NOT NULL AND trades_detail!=''").fetchall()
pool = [dict(r) for r in rows]
US = {'AAPL','MSFT','NVDA','GOOGL','AMZN','META','TSLA','MU','QQQ','TQQQ','MUB','MUUB'}
us_pool = [r for r in pool if r['symbol'].split('/')[0].split('~')[0] in US]
cr_pool = [r for r in pool if r['symbol'].split('/')[0].split('~')[0] not in US]
random.seed(20260830)
us_pick = random.sample(us_pool, 5)
cr_pick = random.sample(cr_pool, 5)
print('=== 抽中样本 ===')
for tag, picks in [('美股代币', us_pick), ('加密货币', cr_pick)]:
    for r in picks:
        print(f"[{tag}] {r['symbol']} {r['timeframe']} {r['strategy'][:22]} {r['mode']} {r['tpsl']} "
              f"{r['period']} → 记录 ret={r['ret']} mdd={r['mdd']} sharpe={r['sharpe']}")

# ---- 回测 ----
fut = BinanceDataFetcher(); fut.set_market_type('future')
spot = BinanceDataFetcher(); spot.set_market_type('spot')
sig_engine = BacktestEngine(signal_mode='or')
BARS = {'4h': 2190, '1h': 8760}

def fetch(fm, sym, s, e, tf, tries=5):
    for i in range(tries):
        try:
            df = fm.fetch_historical_data(sym, s, e, tf)
            if df is not None and not df.empty and len(df) >= 50:
                return df
        except Exception:
            pass
        time.sleep(2 * (i + 1))
    return None

def _close(cash, units, entry, price, side, comm=0.001):
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm

def run(df, signals, tp, sl, mode, tf, initial=10000.0, comm=0.001):
    """严格按表格条件：指定了止盈止损就执行；返回权益曲线 Series"""
    cash = initial; units = 0.0; entry = 0.0; side = 0; n = 0
    eq_pts = []
    for i, (ts, row) in enumerate(df.iterrows()):
        price = row['close']
        if not np.isfinite(price) or price <= 0:
            continue
        sig = int(signals.iloc[i]) if i > 0 else 0
        if side != 0 and entry > 0 and (tp or sl):
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

def retest(r):
    ip = parse_strategy(r['strategy'])
    if ip is None:
        return None, '策略文本无法解析'
    tp, sl = parse_tpsl(r['tpsl'])
    mode = 'LS' if '双向' in (r['mode'] or '') else 'LO'
    coin = r['symbol'].split('/')[0]
    mkt = 'spot' if coin in ('MUB', 'MUUB') else 'future'
    fm = spot if mkt == 'spot' else fut
    sym = f"{coin}/USDT" if mkt == 'spot' else (r['symbol'] if r['symbol'].endswith(':USDT') else f"{coin}/USDT:USDT")
    parts, mult = [], 1.0
    for s, e in parse_period(r['period']):
        df = fetch(fm, sym, s, e, r['timeframe'])
        if df is None:
            continue
        dft = TechnicalIndicators.calculate_all_indicators(df, ip)
        signals = sig_engine.calculate_signals(dft, ip)
        eq = run(df, signals, tp, sl, mode, r['timeframe'], initial=10000.0 * mult)
        if eq is None:
            continue
        parts.append(eq * mult)          # 复利链接：乘以上年末净值系数
        mult *= eq.iloc[-1] / 10000.0
    return parts, (sym, mkt, mode, tp, sl)

def chain(parts, tf):
    """拼接多年权益曲线，统一计算收益/回撤/夏普"""
    if not parts:
        return None
    full = pd.concat(parts)
    full = full[~full.index.duplicated(keep='last')].sort_index()
    peak = full.cummax()
    mdd = ((full - peak) / peak * 100).min()
    rr = full.pct_change().dropna()
    sh = float(rr.mean() / rr.std() * np.sqrt(BARS.get(tf, 2190))) if len(rr) >= 20 and rr.std() > 0 else None
    return (full.iloc[-1] / 10000 - 1) * 100, mdd, sh

print('\n=== 重跑对比 ===')
results = []
for tag, picks in [('美股代币', us_pick), ('加密货币', cr_pick)]:
    for r in picks:
        eqs, meta = retest(r)
        if not eqs:
            print(f"[{tag}] {r['symbol']} {r['strategy'][:18]} 跳过: {meta}")
            continue
        chained = chain(eqs, r['timeframe'])
        if chained is None:
            print(f"[{tag}] {r['symbol']} 无有效数据")
            continue
        ret_c, mdd_c, sh_c = chained
        rec_ret = float((r['ret'] or '0').replace('%', '').replace('+', ''))
        dev = abs(ret_c - rec_ret)
        ok = dev <= max(3.0, abs(rec_ret) * 0.15)
        results.append((tag, r, ret_c, mdd_c, sh_c, dev, ok))
        print(f"[{'OK ' if ok else '偏差'}] [{tag}] {r['symbol']:14} {r['strategy'][:20]:22} "
              f"重跑{ret_c:+8.1f}% vs 记录{rec_ret:+8.1f}% (差{dev:5.1f}pp)  "
              f"mdd {mdd_c:.1f}/{r['mdd']}  sh {sh_c if sh_c is None else round(sh_c,2)}/{r['sharpe']}", flush=True)

n_ok = sum(1 for *_, ok in results if ok)
print(f"\n一致 {n_ok}/{len(results)}")
with open('scripts/results/consistency_sample.json', 'w', encoding='utf-8') as f:
    json.dump([{'tag': t, 'symbol': r['symbol'], 'strategy': r['strategy'], 'ret_rec': r['ret'],
                'ret_retest': round(rc, 1), 'dev_pp': round(d, 1), 'ok': ok}
               for t, r, rc, mc, sc, d, ok in results], f, ensure_ascii=False, indent=1)
