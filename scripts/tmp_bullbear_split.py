# -*- coding: utf-8 -*-
"""牛熊分段对比：BTC日线SMA200为牛熊分界线（收盘>年线=牛，<年线=熊），
统计 macd+背离+量能 三方案在前40币上的牛段/熊段表现（每段独立1万初始，同口径模拟）。
数据直接用本地缓存，不联网。BTC 1h 拼接 2023 年缓存使 SMA200 更早起效。
"""
import io, sys, os, json, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from divergence_signals import DIVERGENCE_VARIANTS, build_variant_signals

COMM = 0.001
INITIAL = 10000.0
VARIANT = 'macd+背离+量能'
UD, UM, UV = DIVERGENCE_VARIANTS[VARIANT]
CONFIGS = [
    ('15m', 'long_short', 'tpsl5', '方案一 15m双向 tpsl5'),
    ('15m', 'long_only', 'none', '方案二 15m仅多 不设'),
    ('1h', 'long_short', 'sl5', '方案三 1h双向 sl5'),
]
TPSL_CFG = {'none': (None, None), 'sl5': (None, 0.05), 'tpsl5': (0.05, 0.05)}
PERIODS = {'1h': 8766, '15m': 35040}
MIN_BARS = {'15m': 1300, '1h': 330}   # 段内最少bar数(≈14天)
MIN_DAYS = 14                          # 区段最短天数

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
detail = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'results', 'top40all_macd_vol_detail_2024_2026.json'),
                        encoding='utf-8'))['results']
TICKERS = list(detail.keys())  # 市值降序前40


def load(base, tf):
    p = os.path.join(CACHE, f'{base}_{tf}.pkl')
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_pickle(p)
        return df if df is not None and len(df) else None
    except Exception:
        return None


# ---- BTC 日线 + SMA200 → 牛熊状态 ----
btc = load('BTC', '1h')
p23 = os.path.join(CACHE, 'BTCUSDT_1h_2023.pkl')
if os.path.exists(p23):
    d23 = pd.read_pickle(p23)
    btc = pd.concat([d23, btc]) if btc is not None else d23
    btc = btc[~btc.index.duplicated(keep='last')].sort_index()
daily = btc['close'].resample('1D').last().dropna()
sma200 = daily.rolling(200).mean()
bull_daily = (daily > sma200)

# ---- 连续牛/熊区间 ----
segs = []
st = None
for i in range(len(bull_daily)):
    if st is None:
        st = [bull_daily.index[i], bull_daily.index[i], bool(bull_daily.iloc[i])]
    elif bool(bull_daily.iloc[i]) == st[2]:
        st[1] = bull_daily.index[i]
    else:
        segs.append(tuple(st)); st = [bull_daily.index[i], bull_daily.index[i], bool(bull_daily.iloc[i])]
if st: segs.append(tuple(st))
segs = [(s, e, b) for s, e, b in segs if (e - s).days >= MIN_DAYS]
print('== 牛熊分段（BTC日线收盘 vs SMA200，段最短%d天，自SMA200生效起） ==' % MIN_DAYS)
for s, e, b in segs:
    c0, c1 = daily.loc[s: e].iloc[0], daily.loc[s: e].iloc[-1]
    print(f"  {'牛' if b else '熊'} {s.date()} ~ {e.date()}  {(e - s).days:4d}天  BTC {c0:,.0f}→{c1:,.0f} ({(c1 / c0 - 1) * 100:+.1f}%)")


def _close(cash, units, entry, price, side, comm=COMM):
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm


def simulate(df, signals, tp=None, sl=None, mode='long_only', ppy=8766, initial=INITIAL):
    idx = df.index.to_numpy(); close = df['close'].to_numpy(); sig = signals.to_numpy()
    cash = initial; units = 0.0; entry = 0.0; side = 0; n = 0
    eq_t = []; eq_v = []
    for i in range(len(df)):
        price = close[i]
        if not np.isfinite(price) or price <= 0:
            continue
        s = int(sig[i]) if i > 0 else 0
        if side != 0 and entry > 0:
            r = (price - entry) / entry if side > 0 else (entry - price) / entry
            if (tp and r >= tp) or (sl and r <= -sl):
                cash = _close(cash, units, entry, price, side); side = 0; units = 0; n += 1
                eq_t.append(idx[i]); eq_v.append(cash); continue
        eq = cash + (units * price if side > 0 else units * entry + (entry - price) * units if side < 0 else 0)
        if eq <= 0:
            return None
        eq_t.append(idx[i]); eq_v.append(eq)
        if s == 1 and side <= 0:
            if side < 0:
                cash = _close(cash, units, entry, price, side); n += 1; side = 0; units = 0
            u = (cash * 0.95) / (price * (1 + comm))
            if u > 0:
                cash -= u * price * (1 + comm); units = u; entry = price; side = 1
        elif s == -1 and side >= 0:
            if side > 0:
                cash = _close(cash, units, entry, price, side); n += 1; side = 0; units = 0
            if mode == 'long_short':
                u = (cash * 0.95) / price
                if u > 0:
                    cash -= u * price * (1 + comm); units = u; entry = price; side = -1
    if side != 0 and len(df) > 0:
        cash = _close(cash, units, entry, close[-1], side); n += 1
        eq_t.append(idx[-1]); eq_v.append(cash)
    if n == 0:
        return None
    eq = pd.Series(eq_v, index=pd.DatetimeIndex(eq_t)).sort_index()
    peak = eq.cummax(); mdd = ((eq - peak) / peak * 100).min()
    return {'ret': (eq.iloc[-1] / initial - 1) * 100, 'mdd': mdd, 'n': n}


# ---- 逐币逐tf计算信号，逐段模拟 ----
stats = {c[3]: {'牛': [], '熊': []} for c in CONFIGS}
sig_cache = {}
for k, base in enumerate(TICKERS, 1):
    for tf in ('15m', '1h'):
        df = load(base, tf)
        if df is None:
            continue
        try:
            _, sig = build_variant_signals(df, UD, UM, UV)
        except Exception:
            continue
        for ctf, cmode, ctpsl, cname in CONFIGS:
            if ctf != tf:
                continue
            tp, sl = TPSL_CFG[ctpsl]
            for s, e, is_bull in segs:
                m = (df.index >= s) & (df.index < e + pd.Timedelta(days=1))
                dw = df[m]
                if len(dw) < MIN_BARS[tf]:
                    continue
                try:
                    r = simulate(dw, sig.loc[dw.index], tp=tp, sl=sl, mode=cmode, ppy=PERIODS[tf])
                except Exception:
                    r = None
                if r:
                    stats[cname]['牛' if is_bull else '熊'].append(r)
        print(f'[{k}/{len(TICKERS)}] {base} {tf} done', flush=True)

print('\n== 三方案 × 牛/熊段 表现（段-币为样本，每段独立1万） ==')
for ctf, cmode, ctpsl, cname in CONFIGS:
    print(f"\n【{cname}】")
    for ph in ('牛', '熊'):
        rs = stats[cname][ph]
        if not rs:
            print(f'  {ph}市段: 无样本'); continue
        rets = [r['ret'] for r in rs]; mdds = [r['mdd'] for r in rs]; ns = [r['n'] for r in rs]
        # 每段内先按币平均，避免币多的段权重偏大——这里样本即(币,段)，直接均值
        print(f"  {ph}市段 样本{len(rs):3d} | 均收益{np.mean(rets):+8.1f}%  中位{np.median(rets):+8.1f}%"
              f"  正收益{sum(x > 0 for x in rets)}/{len(rs)}"
              f"  | 平均MDD{np.mean(mdds):6.1f}%  最差{min(mdds):6.1f}%"
              f"  | 交易次数均{np.mean(ns):6.1f}")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results',
                       'top40all_bullbear_split.json'), 'w', encoding='utf-8') as f:
    json.dump({'segs': [(str(s.date()), str(e.date()), b) for s, e, b in segs],
               'stats': {k: {ph: v[ph] for ph in ('牛', '熊')} for k, v in stats.items()}},
              f, ensure_ascii=False, indent=1, default=float)
print('\n→ scripts/results/top40all_bullbear_split.json')
