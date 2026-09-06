# -*- coding: utf-8 -*-
"""macd+背离+量能 × 前30 × 三方案 × 出场优化（胜率实验）

约束：入场信号完全不变（不减少交易次数），只改出场管理：
  BE2/BE3: 保本止损——浮盈达 2%/3% 后，止损上移到 +0.4%（覆盖双边手续费0.2%+缓冲）
  TR3:     移动锁盈——浮盈达 3% 后，锁定最高浮盈回撤 1.5%（回落即出）
  TP3/TP4: 收紧止盈（方案一 tp5→3；方案三 增设 tp4，原无止盈）
统计：胜率(净手续费)、平均盈/亏、每笔期望、交易次数、收益、MDD。
窗口：2024-01-01 → 现在（整段，单币），与基线同口径对比。
"""
import os, sys, io, warnings, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

PROXY = "http://127.0.0.1:7892"
os.environ.setdefault("HTTP_PROXY", PROXY); os.environ.setdefault("HTTPS_PROXY", PROXY)

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from collections import defaultdict

from divergence_signals import DIVERGENCE_VARIANTS, build_variant_signals
from tmp_top40_all_macd_vol import COMM, INITIAL, YEAR_BARS_MIN, PERIODS, fetch_klines

VARIANT = 'macd+背离+量能'
UD, UM, UV = DIVERGENCE_VARIANTS[VARIANT]
START = pd.Timestamp('2024-01-01', tz='UTC')
FEE_RT = 2 * COMM          # 双边手续费 0.2%（价格口径）

summary = json.load(open('scripts/results/top40all_macd_vol_summary_2024_2026.json', encoding='utf-8'))
top30 = summary['meta']['tickers'][:30]

# 方案 × 变体：tp/sl 为价格口径；be=(触发,保本位)；tr=(触发,回撤锁)
PLANS = {
    '方案一 15m双向tpsl5': dict(tf='15m', mode='long_short',
        variants={
            'base': dict(tp=0.05, sl=0.05),
            'BE2':  dict(tp=0.05, sl=0.05, be=(0.02, 0.004)),
            'TR3':  dict(tp=0.05, sl=0.05, tr=(0.03, 0.015)),
            'TP3':  dict(tp=0.03, sl=0.05),
        }),
    '方案二 15m仅多不设': dict(tf='15m', mode='long_only',
        variants={
            'base': dict(tp=None, sl=None),
            'BE2':  dict(tp=None, sl=None, be=(0.02, 0.004)),
            'TR3':  dict(tp=None, sl=None, tr=(0.03, 0.015)),
        }),
    '方案三 1h双向sl5': dict(tf='1h', mode='long_short',
        variants={
            'base': dict(tp=None, sl=0.05),
            'BE2':  dict(tp=None, sl=0.05, be=(0.02, 0.004)),
            'TR3':  dict(tp=None, sl=0.05, tr=(0.03, 0.015)),
            'TP4':  dict(tp=0.04, sl=0.05),
        }),
}


def simulate_v(df, signals, tp, sl, mode, be=None, tr=None, initial=INITIAL, comm=COMM):
    """整段模拟，逐笔记录净收益率（价格口径-双边手续费）。"""
    close = df['close'].to_numpy(); sig = signals.to_numpy()
    idx = df.index
    cash = initial; units = 0.0; entry = 0.0; side = 0
    r_max = 0.0                       # 持仓期间最大有利波动（close口径）
    trades = []                       # 每笔净收益率
    eq_t = []; eq_v = []

    def _close_at(price):
        nonlocal cash, units, entry, side, r_max
        fee = units * price * comm
        if side > 0:
            cash += units * price - fee
        else:
            cash += units * entry + (entry - price) * units - fee
        r_net = ((price - entry) / entry if side > 0 else (entry - price) / entry) - 2 * comm
        trades.append(r_net)
        units = 0.0; entry = 0.0; side = 0; r_max = 0.0

    for i in range(len(df)):
        price = close[i]
        if not np.isfinite(price) or price <= 0:
            continue
        s = int(sig[i]) if i > 0 else 0
        if side != 0 and entry > 0:
            r = (price - entry) / entry if side > 0 else (entry - price) / entry
            r_max = max(r_max, r)
            # 有效出场水位 level：跌破即出（负=亏损止损位，正=保本/锁盈位）
            level = -sl if sl else None
            if be and r_max >= be[0]:
                level = max(level, be[1]) if level is not None else be[1]
            if tr and r_max >= tr[0]:
                lock = r_max - tr[1]
                level = max(level, lock) if level is not None else lock
            if (level is not None and r <= level) or (tp and r >= tp):
                _close_at(price)
                eq_t.append(idx[i]); eq_v.append(cash); continue
        eq = cash + (units * price if side > 0 else
                     units * entry + (entry - price) * units if side < 0 else 0)
        if eq <= 0:
            return None
        eq_t.append(idx[i]); eq_v.append(eq)
        if s == 1 and side <= 0:
            if side < 0:
                _close_at(price)
            u = (cash * 0.95) / (price * (1 + comm))
            if u > 0:
                cash -= u * price * (1 + comm); units = u; entry = price; side = 1; r_max = 0.0
        elif s == -1 and side >= 0:
            if side > 0:
                _close_at(price)
            if mode == 'long_short':
                u = (cash * 0.95) / price
                if u > 0:
                    cash -= u * price * (1 + comm); units = u; entry = price; side = -1; r_max = 0.0
    if side != 0 and len(df) > 0:
        _close_at(close[-1])
        eq_t.append(idx[-1]); eq_v.append(cash)
    if not trades:
        return None
    eq = pd.Series(eq_v, index=pd.DatetimeIndex(eq_t)).sort_index()
    mdd = ((eq - eq.cummax()) / eq.cummax() * 100).min()
    wins = [t for t in trades if t > 0]; losses = [t for t in trades if t <= 0]
    return {
        'ret': (eq.iloc[-1] / initial - 1) * 100, 'mdd': mdd, 'n': len(trades),
        'winrate': len(wins) / len(trades) * 100,
        'avg_win': float(np.mean(wins)) * 100 if wins else 0.0,
        'avg_loss': float(np.mean(losses)) * 100 if losses else 0.0,
        'exp': float(np.mean(trades)) * 100,          # 每笔期望（净%）
    }


# res[plan][vname] = list of per-coin dicts
res = {p: defaultdict(list) for p in PLANS}
for k, base in enumerate(top30, 1):
    cache = {}
    for pname, pcfg in PLANS.items():
        tf = pcfg['tf']
        if tf not in cache:
            df = fetch_klines(base, tf)
            if df is None or len(df) < YEAR_BARS_MIN:
                cache[tf] = None; continue
            try:
                _, sig = build_variant_signals(df, UD, UM, UV)
            except Exception:
                cache[tf] = None; continue
            m = df.index >= START
            cache[tf] = (df[m], sig.loc[df[m].index])
        c = cache[tf]
        if not c:
            continue
        dw, sw = c
        for vname, v in pcfg['variants'].items():
            r = simulate_v(dw, sw, v.get('tp'), v.get('sl'), pcfg['mode'],
                           be=v.get('be'), tr=v.get('tr'))
            if r:
                res[pname][vname].append(r)
    print(f'[{k}/{len(top30)}] {base} done', flush=True)

print('\n== 出场优化 × 胜率（前30，2024-01→今，入场信号不变） ==')
print(f'胜率=净手续费后每笔收益>0 占比；期望=每笔平均净收益%；n=总交易')

out = {}
for pname, pcfg in PLANS.items():
    print(f'\n【{pname}】')
    print(f'  {"变体":5} {"胜率均值":>8} {"胜率中位":>8} {"均盈%":>7} {"均亏%":>7} {"每笔期望%":>9} '
          f'{"总笔数":>7} {"均收益%":>9} {"均MDD%":>8} {"最差MDD%":>9}')
    out[pname] = {}
    for vname in pcfg['variants']:
        lst = res[pname][vname]
        if not lst:
            continue
        wr = [x['winrate'] for x in lst]
        d = {'wr_mean': round(float(np.mean(wr)), 1), 'wr_med': round(float(np.median(wr)), 1),
             'avg_win': round(float(np.mean([x['avg_win'] for x in lst])), 2),
             'avg_loss': round(float(np.mean([x['avg_loss'] for x in lst])), 2),
             'exp': round(float(np.mean([x['exp'] for x in lst])), 3),
             'n': int(sum(x['n'] for x in lst)),
             'ret': round(float(np.mean([x['ret'] for x in lst])), 1),
             'mdd': round(float(np.mean([x['mdd'] for x in lst])), 1),
             'mdd_max': round(float(min(x['mdd'] for x in lst)), 1)}
        out[pname][vname] = d
        print(f'  {vname:5} {d["wr_mean"]:>7.1f}% {d["wr_med"]:>7.1f}% {d["avg_win"]:>+7.2f} '
              f'{d["avg_loss"]:>+7.2f} {d["exp"]:>+9.3f} {d["n"]:>7} {d["ret"]:>+9.1f} '
              f'{d["mdd"]:>8.1f} {d["mdd_max"]:>9.1f}')

with open('scripts/results/top30_winrate_variants.json', 'w', encoding='utf-8') as f:
    json.dump({'variant': VARIANT, 'window': '2024-01-01→now', 'plans': out}, f,
              ensure_ascii=False, indent=1, default=float)
print('\n→ scripts/results/top30_winrate_variants.json')
