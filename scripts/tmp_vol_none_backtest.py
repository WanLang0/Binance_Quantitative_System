# -*- coding: utf-8 -*-
"""量能过滤完全移除（无过滤）对照回测：1.5x / 1.2x / 无量能 三档
三种MACD参数 × 杠杆(1/2/4)，1h 周期，口径与 tmp_vol12_backtest.py 完全一致。
1.5x/1.2x 基准从 scripts/results/macd_vol_threshold_1h.json 读取，只重跑"无量能"档。
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from indicators import TechnicalIndicators
from tmp_combo_full import W0
from tmp_vol12_backtest import VARIANTS, load_data, simulate_lev, INIT, K_SL, K_TP

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'macd_vol_threshold_1h.json')


def build_signal_nofilter(df, f, s, g):
    dft = TechnicalIndicators.calculate_macd(df, f, s, g)
    macd_buy = (dft['MACD'] > dft['MACD_signal']) & (dft['MACD'].shift(1) <= dft['MACD_signal'].shift(1))
    macd_sell = (dft['MACD'] < dft['MACD_signal']) & (dft['MACD'].shift(1) >= dft['MACD_signal'].shift(1))
    sig = pd.Series(0, index=df.index)
    sig[macd_buy.fillna(False)] = 1
    sig[macd_sell.fillna(False)] = -1
    return sig


def run_nofilter(data, f, s, g, lev):
    eqs = {}; coin_mdd = []; n_all = 0; w_all = 0; grid = None; liq_coins = 0
    for base, (df, atr) in data.items():
        sig = build_signal_nofilter(df, f, s, g)
        r = simulate_lev(df, sig, INIT, atr, K_SL, K_TP, lev, W0)
        if r is None:
            continue
        eq, fl_t, fl_w, n_liq, n_trade = r
        e = eq / INIT
        eqs[base] = e
        cd = float(((e - e.cummax()) / e.cummax() * 100).min())
        coin_mdd.append(cd)
        if n_liq > 0:
            liq_coins += 1
        m = fl_t >= W0
        n_all += int(m.sum()); w_all += int(fl_w[m].sum())
        grid = eq.index if grid is None else grid.union(eq.index)
    if not eqs:
        return None
    nav = sum(e.reindex(grid).ffill().fillna(1.0) for e in eqs.values()) / len(eqs)
    nav = nav[nav.index >= W0].dropna()
    mdd = float(((nav - nav.cummax()) / nav.cummax() * 100).min())
    ret = float((nav.iloc[-1] - 1) * 100)
    me = nav.resample('ME').last().dropna()
    mret = (me.pct_change().dropna() * 100)
    monthly = {ts.strftime('%Y-%m'): round(float(v), 2) for ts, v in mret.items()}
    yr = {}
    for y, a, b in [('2024', '2024-01-01', '2025-01-01'), ('2025', '2025-01-01', '2026-01-01'),
                    ('2026', '2026-01-01', '2026-09-01')]:
        a, b = pd.Timestamp(a, tz='UTC'), pd.Timestamp(b, tz='UTC')
        seg = nav[(nav.index >= a) & (nav.index < b)]
        if len(seg):
            prev = nav[nav.index < a]
            yr[y] = round(float((seg.iloc[-1] / (prev.iloc[-1] if len(prev) else 1.0) - 1) * 100), 1)
    return dict(ret=round(ret, 1), mdd=round(mdd, 1),
                coin_mdd_avg=round(float(np.mean(coin_mdd)), 1), coin_mdd_max=round(float(np.min(coin_mdd)), 1),
                trades=n_all, winrate=round(w_all / n_all * 100, 1) if n_all else None,
                liq_coins=liq_coins, yearly=yr, monthly=monthly)


def main():
    print('加载数据...', flush=True)
    data = load_data('1h')
    print(f'1h 周期 {len(data)} 币', flush=True)
    with open(RES, encoding='utf-8') as f:
        base_res = json.load(f)
    results = {}
    for name, (f, s, g) in VARIANTS.items():
        out = {}
        for lev in (1, 2, 4):
            out[f'lev{lev}x'] = run_nofilter(data, f, s, g, lev)
        results[name] = out
        print(f'== 无量能 {name} 完成 ==', flush=True)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'macd_vol_none_1h.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=float)
    for name in VARIANTS:
        print(f'\n===== {name} (1h) =====', flush=True)
        for lev in (1, 2, 4):
            b15 = base_res['vol1_5x'][name][f'lev{lev}x']
            b12 = base_res['vol1_2x'][name][f'lev{lev}x']
            n = results[name][f'lev{lev}x']
            print(f'  [{lev}x] 1.5x基准 : 总{b15["ret"]:>+9.1f}% MDD{b15["mdd"]:>7.1f}% 单币均{b15["coin_mdd_avg"]:>6.1f}% 笔{b15["trades"]:>5} 胜{b15["winrate"]}%', flush=True)
            print(f'        1.2x      : 总{b12["ret"]:>+9.1f}% MDD{b12["mdd"]:>7.1f}% 单币均{b12["coin_mdd_avg"]:>6.1f}% 笔{b12["trades"]:>5} 胜{b12["winrate"]}%', flush=True)
            print(f'        无量能    : 总{n["ret"]:>+9.1f}% MDD{n["mdd"]:>7.1f}% 单币均{n["coin_mdd_avg"]:>6.1f}% 笔{n["trades"]:>5} 胜{n["winrate"]}%', flush=True)
            print(f'        年度(1.5x/1.2x/无): {b15["yearly"]} / {b12["yearly"]} / {n["yearly"]}', flush=True)
    print(f'\n逐月明细 → {p}')


if __name__ == '__main__':
    main()
