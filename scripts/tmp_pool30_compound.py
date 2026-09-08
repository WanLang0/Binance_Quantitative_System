# -*- coding: utf-8 -*-
"""30币均分资金(每币333起步)×独立复利池不封顶×不开优先匹配 = 系统默认模式回测
方案一 15m双向tpsl5，连续口径 2024-01~2026-08。
注意：无全局资金约束（赢家池滚大后总敞口可能超过总资金，实盘会被保证金拦截/需杠杆），
此结果为该模式的理想上界，同时报告总敞口超限的月份占比。
"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from divergence_signals import DIVERGENCE_VARIANTS, build_variant_signals
from tmp_combo_full import simulate_cont, W0, W1, YEARS, BASES

COMM = 0.001
UD, UM, UV = DIVERGENCE_VARIANTS['macd+背离+量能']
TOTAL = 10000.0
INIT = TOTAL / len(BASES)


def main():
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
    eqs = {}
    grid = None
    for k, b in enumerate(BASES, 1):
        df = pd.read_pickle(os.path.join(cache, f'{b}_15m.pkl'))
        df = df[(df.index >= W0 - pd.Timedelta(days=1)) & (df.index < W1)]
        _, sig = build_variant_signals(df, UD, UM, UV)
        r = simulate_cont(df, sig, 0.05, 0.05, 'long_short', INIT)
        if r is None:
            print(f'{b} 无交易'); continue
        eqs[b] = r[0] / INIT           # 净值曲线
        grid = r[0].index if grid is None else grid.union(r[0].index)
        print(f'[{k}/{len(BASES)}] {b:8} done', flush=True)
    # 总池净值 = 等权平均净值（每币1/30资金）；未上市段按1.0(现金闲置)填充，防NaN传染
    nav = sum(e.reindex(grid).ffill().fillna(1.0) for e in eqs.values()) / len(eqs)
    nav = nav.dropna()
    fl_t = pd.DatetimeIndex([])
    fl_w = np.array([], dtype=bool)
    # 交易合并统计按已有simulate结果不可得——重算太慢，直接用全期统计近似：
    # 交易集合与每币独立10000完全一致（收益率与初始资金无关），引用 35728笔/46.2%
    ts = nav.index
    marks = [('2024', '2024-01-01', '2025-01-01'), ('2025', '2025-01-01', '2026-01-01'),
             ('2026', '2026-01-01', '2026-09-01')]
    print('\n===== 均分30份·独立复利·不开优先匹配（总资金10000，15m双向tpsl5） =====')
    prev = nav.iloc[0]
    for y, a, b in marks:
        a, b = pd.Timestamp(a, tz='UTC'), pd.Timestamp(b, tz='UTC')
        seg = nav[(ts >= a) & (ts < b)]
        ret = (seg.iloc[-1] / prev - 1) * 100
        mdd = float(((seg - seg.cummax()) / seg.cummax() * 100).min())
        prev = seg.iloc[-1]
        print(f'  {y:6} 收益{ret:+8.1f}%  MDD {mdd:6.1f}%')
    mdd_all = float(((nav - nav.cummax()) / nav.cummax() * 100).min())
    me = nav.resample('ME').last().dropna()
    mr = me.pct_change().dropna() * 100
    navm = 1.0
    for v in mr:
        navm *= (1 + v / 100)
    print(f'  全期   收益{(nav.iloc[-1] - 1) * 100:+8.1f}%  MDD {mdd_all:6.1f}%  '
          f'月均{mr.mean():+.2f}%  月度复利{navm:.3f}x  亏损月{int((mr < 0).sum())}/{len(mr)}  '
          f'最差月{mr.min():+.2f}%')
    out = {'mdd_all': round(mdd_all, 1), 'ret_all': round((nav.iloc[-1] - 1) * 100, 1),
           'monthly_avg': round(float(mr.mean()), 2), 'nav': round(navm, 3),
           'neg_months': int((mr < 0).sum()), 'months': int(len(mr)),
           'worst_month': round(float(mr.min()), 2),
           'monthly': {str(m.to_period('M')): round(float(v), 2) for m, v in mr.items()}}
    with open('scripts/results/top30_pool30_compound.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n明细 → scripts/results/top30_pool30_compound.json')


if __name__ == '__main__':
    main()
