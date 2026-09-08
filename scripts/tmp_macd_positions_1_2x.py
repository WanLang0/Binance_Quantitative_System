# -*- coding: utf-8 -*-
"""1.2x量能阈值下的持仓份数统计（对照 1.5x 基准）
复用 tmp_macd_3param_positions 的模拟器，仅把量能阈值 1.5 -> 1.2。
MDD 对照从 scripts/results/macd_vol_threshold_1h.json 读取。
"""
import os, sys, io, json, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from tmp_macd_3param_positions import (load_data, simulate_hold, VARIANTS,
                                       K_SL, K_TP, MAINT, CLIP_HI)
from tmp_combo_full import W0
from indicators import TechnicalIndicators

RES15 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results',
                     'macd_3param_positions.json')
RES_MDD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results',
                       'macd_vol_threshold_1h.json')


def build_signal12(df, f, s, g):
    dft = TechnicalIndicators.calculate_macd(df, f, s, g)
    macd_buy = (dft['MACD'] > dft['MACD_signal']) & (dft['MACD'].shift(1) <= dft['MACD_signal'].shift(1))
    macd_sell = (dft['MACD'] < dft['MACD_signal']) & (dft['MACD'].shift(1) >= dft['MACD_signal'].shift(1))
    vol_up = df['volume'] > df['volume'].rolling(20).mean() * 1.2
    sig = pd.Series(0, index=df.index)
    sig[(macd_buy & vol_up).fillna(False)] = 1
    sig[(macd_sell & vol_up).fillna(False)] = -1
    return sig


def positions_stats(data, f, s, g):
    grid = None
    for base, (df, atr) in data.items():
        grid = df.index if grid is None else grid.union(df.index)
    grid = grid[grid >= W0]
    holds = {}
    for base, (df, atr) in data.items():
        sig = build_signal12(df, f, s, g)
        holds[base] = simulate_hold(df, sig, atr, K_SL, K_TP, W0)
    occ = {b: h.reindex(grid).ffill().fillna(False).astype(int) for b, h in holds.items()}
    cnt = sum(occ.values())
    cnt = cnt[cnt.index >= W0].dropna()
    total = len(cnt)
    vc = cnt.value_counts().sort_index()
    dist = {int(k): round(int(v) / total * 100, 2) for k, v in vc.items()}
    stats = {'mean': round(float(cnt.mean()), 2), 'median': float(cnt.median()),
             'p25': float(cnt.quantile(0.25)), 'p75': float(cnt.quantile(0.75)),
             'p90': float(cnt.quantile(0.90)), 'max': int(cnt.max()),
             'zero_pct': round(float((cnt == 0).mean()) * 100, 2), 'distribution': dist}
    yrs = {}
    for y, a, b in [('2024', '2024-01-01', '2025-01-01'),
                    ('2025', '2025-01-01', '2026-01-01'),
                    ('2026', '2026-01-01', '2026-09-01')]:
        a = pd.Timestamp(a, tz='UTC'); b = pd.Timestamp(b, tz='UTC')
        seg = cnt[(cnt.index >= a) & (cnt.index < b)]
        if len(seg):
            yrs[y] = {'mean': round(float(seg.mean()), 2), 'max': int(seg.max()),
                      'zero_pct': round(float((seg == 0).mean()) * 100, 2)}
    return stats, yrs


def main():
    data = load_data('1h')
    print(f'1h 周期 {len(data)} 币 · 量能阈值 1.2x', flush=True)
    with open(RES15, encoding='utf-8') as fh:
        pos15 = json.load(fh)
    with open(RES_MDD, encoding='utf-8') as fh:
        mdd = json.load(fh)

    results = {}
    for name, (f, s, g) in VARIANTS.items():
        stats, yrs = positions_stats(data, f, s, g)
        results[name] = {'stats': stats, 'yearly': yrs}
        old = pos15[name]['stats']
        print(f'\n===== {name} (1h · 量能1.2x vs 1.5x) =====', flush=True)
        print(f'  1.2x: 平均 {stats["mean"]} 份 · 中位 {stats["median"]} · '
              f'P25/P75/P90 {stats["p25"]:.0f}/{stats["p75"]:.0f}/{stats["p90"]:.0f} · '
              f'峰值 {stats["max"]} 份 · 空仓占比 {stats["zero_pct"]}%', flush=True)
        print(f'  1.5x: 平均 {old["mean"]} 份 · 中位 {old["median"]} · '
              f'P25/P75/P90 {old["p25"]:.0f}/{old["p75"]:.0f}/{old["p90"]:.0f} · '
              f'峰值 {old["max"]} 份 · 空仓占比 {old["zero_pct"]}%', flush=True)
        print(f'  1.2x 分布(%): {stats["distribution"]}', flush=True)
        for y, d in yrs.items():
            o = pos15[name]['yearly'].get(y, {})
            print(f'  {y}: 1.2x 均 {d["mean"]} / 峰 {d["max"]} / 空仓 {d["zero_pct"]}%  '
                  f'vs 1.5x 均 {o.get("mean")} / 峰 {o.get("max")} / 空仓 {o.get("zero_pct")}%', flush=True)
        # MDD 对照
        for lev in (1, 2, 4):
            b = mdd['vol1_5x'][name][f'lev{lev}x']; n = mdd['vol1_2x'][name][f'lev{lev}x']
            print(f'  [{lev}x] MDD {b["mdd"]}% -> {n["mdd"]}% · 单币MDD均 {b["coin_mdd_avg"]}% -> {n["coin_mdd_avg"]}% · '
                  f'总收益 {b["ret"]:+.0f}% -> {n["ret"]:+.0f}%', flush=True)

    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results',
                     'macd_positions_1_2x.json')
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)
    print(f'\n明细 → {p}', flush=True)


if __name__ == '__main__':
    main()
