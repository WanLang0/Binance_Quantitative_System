# -*- coding: utf-8 -*-
"""C3 调优方案(止损1.0/止盈1.2 ATR) × 杠杆(1/2/4) 杠杆放大测试
口径: 1h · 30币均分独立复利 · 1.2x量能(vol20) · ATR14 截断[1%,8%] · C判定(止盈盘中/止损收盘确认) · 双向 · 0.05%/边
重点: 高杠杆下月稳定性是否保留、爆仓币数、回撤放大程度。
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from tmp_tp_close_backtest import load_data, simulate_lev, COMM, INIT, MAINT, VARIANTS, W0
from tmp_macd_tune_scan import build_signal2

CFG = dict(vol_mult=1.2, vol_win=20, k_sl=1.0, k_tp=1.2, trend='none', tp_mode='intraday')


def run_lev(data, f, s, g, lev, cfg):
    eqs = {}; coin_mdd = []; n_all = 0; w_all = 0; grid = None; liq_coins = 0
    for base, (df, atr) in data.items():
        sig = build_signal2(df, f, s, g, cfg['vol_mult'], cfg['vol_win'], cfg['trend'], cfg['tp_mode'])
        r = simulate_lev(df, sig, INIT, atr, cfg['k_sl'], cfg['k_tp'], lev, W0, cfg['tp_mode'])
        if r is None:
            continue
        eq, fl_t, fl_w, n_liq, n_trade = r
        e = eq / INIT
        eqs[base] = e
        coin_mdd.append(float(((e - e.cummax()) / e.cummax() * 100).min()))
        if n_liq > 0:
            liq_coins += 1
        m = fl_t >= W0
        n_all += int(m.sum()); w_all += int(fl_w[m].sum())
        grid = eq.index if grid is None else grid.union(eq.index)
    if not eqs:
        return None
    nav = sum(e.reindex(grid).ffill().fillna(1.0) for e in eqs.values()) / len(eqs)
    nav = nav[nav.index >= W0].dropna()
    me = nav.resample('ME').last().dropna()
    mr = me.pct_change().dropna() * 100
    months = [float(v) for v in mr.values]
    mdd = float(((nav - nav.cummax()) / nav.cummax() * 100).min())
    tot = float((nav.iloc[-1] - 1) * 100)
    yr = {}
    for y, a, b in [('2024', '2024-01-01', '2025-01-01'), ('2025', '2025-01-01', '2026-01-01'),
                    ('2026', '2026-01-01', '2026-09-01')]:
        a, b = pd.Timestamp(a, tz='UTC'), pd.Timestamp(b, tz='UTC')
        seg = nav[(nav.index >= a) & (nav.index < b)]
        if len(seg):
            base_v = nav[nav.index < a].iloc[-1] if len(nav[nav.index < a]) else 1.0
            yr[y] = round(float((seg.iloc[-1] / base_v - 1) * 100), 1)
    monthly = {str(t.to_period('M')): round(float(v), 2) for t, v in mr.items()}
    return dict(ret=round(tot, 1), mdd=round(mdd, 1),
                coin_mdd_avg=round(float(np.mean(coin_mdd)), 1), coin_mdd_max=round(float(np.min(coin_mdd)), 1),
                trades=n_all, winrate=round(w_all / n_all * 100, 1) if n_all else None,
                liq_coins=liq_coins,
                avg=round(float(np.mean(months)), 2) if months else 0.0,
                std=round(float(np.std(months)), 2) if months else 0.0,
                mn=round(float(np.min(months)), 2) if months else 0.0,
                posr=round(sum(1 for m in months if m > 0) / len(months) * 100, 1) if months else 0.0,
                yearly=yr, monthly=monthly)


def main():
    data = load_data('1h')
    print(f'C3方案(止损1.0/止盈1.2 ATR) · 1h · {len(data)} 币\n', flush=True)
    out = {}
    for name, (f, s, g) in VARIANTS.items():
        out[name] = {}
        print(f'########## {name} ##########', flush=True)
        for lev in (1, 2, 4):
            r = run_lev(data, f, s, g, lev, CFG)
            out[name][f'lev{lev}x'] = r
            print(f"  [{lev}x] 总{r['ret']:>+9.1f}% MDD{r['mdd']:>7.1f}% 单币MDD均{r['coin_mdd_avg']:>6.1f}% "
                  f"笔{r['trades']:>5} 胜{r['winrate']:>5}% 爆{r['liq_coins']} | "
                  f"月均{r['avg']:>+6.2f}% 月std{r['std']:>5.2f} 最差{r['mn']:>+7.2f}% 正{r['posr']:>4.0f}%", flush=True)
            print(f"        年度 {r['yearly']}", flush=True)
        print('', flush=True)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'macd_c3_lev_1h.json')
    with open(p, 'w', encoding='utf-8') as fp:
        json.dump(out, fp, ensure_ascii=False, indent=1, default=float)
    print(f'明细 → {p}')


if __name__ == '__main__':
    main()
