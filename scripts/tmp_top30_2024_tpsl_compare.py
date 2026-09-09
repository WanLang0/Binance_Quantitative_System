# -*- coding: utf-8 -*-
"""2024+ 时代池(30币) × 三种止盈止损口径对照 × 2024-01~2026-09 回测
口径A: C3盘中止盈   止1.0×ATR / 盈1.2×ATR, 盘中止盈+止损收盘确认
口径B: 宽止损盘中   止1.5×ATR / 盈2.0×ATR, 盘中止盈+止损收盘确认
口径C: 宽止损收盘   止1.5×ATR / 盈2.0×ATR, 止盈收盘确认+止损收盘确认
公共: 1h · 30币均分独立复利 · 1.2x量能(vol20) · ATR14截断[1%,8%] · 双向 · 0.05%/边
三参数(12/26/9, 12/16/5, 12/16/7) × 杠杆(1/2/4)
输出: 总收益 / 最大回撤 / 月度均回撤(每月内MDD均值) / 最差月 / 月std / 交易次数 / 胜率 / 爆仓
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange

from tmp_tp_close_backtest import simulate_lev
from tmp_combo_full import BASES, W0, W1

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
INIT = 10000.0 / len(BASES)
PER = 14
VARIANTS = {'12/26/9': (12, 26, 9), '12/16/5': (12, 16, 5), '12/16/7': (12, 16, 7)}
SCHEMES = {  # 名称: (k_sl, k_tp, tp_mode)
    'C3盘中_1.0/1.2': (1.0, 1.2, 'intraday'),
    '盘中_1.5/2.0':   (1.5, 2.0, 'intraday'),
    '收盘_1.5/2.0':   (1.5, 2.0, 'close'),
}


def load_data():
    data = {}
    for base in BASES:
        p = os.path.join(CACHE, f'{base}_1h.pkl')
        if not os.path.exists(p):
            continue
        df = pd.read_pickle(p)
        df = df[df.index < W1]
        if len(df) < 300:
            continue
        atr = AverageTrueRange(df['high'], df['low'], df['close'], PER).average_true_range()
        data[base] = (df, atr)
    return data


def build_sig(df, f, s, g):
    from tmp_macd_tune_scan import build_signal2
    return build_signal2(df, f, s, g, 1.2, 20, 'none', 'intraday')


def run(data, sigs, k_sl, k_tp, tp_mode, lev):
    eqs = {}; coin_mdd = []; n_all = 0; w_all = 0; grid = None; liq_coins = 0
    for base, (df, atr) in data.items():
        r = simulate_lev(df, sigs[base], INIT, atr, k_sl, k_tp, lev, W0, tp_mode)
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
    # 月度均回撤: 每月内(高点->低点)的最大回撤, 再对全部月份取平均
    g = nav.groupby(pd.Grouper(freq='ME'))
    month_dds = [float(((seg - seg.cummax()) / seg.cummax() * 100).min()) for _, seg in g if len(seg) > 1]
    avg_mdd_m = round(float(np.mean(month_dds)), 2) if month_dds else None
    max_mdd_m = round(float(np.min(month_dds)), 2) if month_dds else None
    return dict(ret=round(tot, 1), mdd=round(mdd, 1), avg_mdd_m=avg_mdd_m, max_mdd_m=max_mdd_m,
                coin_mdd_avg=round(float(np.mean(coin_mdd)), 1),
                trades=n_all, winrate=round(w_all / n_all * 100, 1) if n_all else None,
                liq_coins=liq_coins,
                mn=round(float(np.min(months)), 2) if months else 0.0,
                std=round(float(np.std(months)), 2) if months else 0.0)


def main():
    data = load_data()
    print(f'2024+ 时代池({len(data)}币) · 三口径对照 · 1h · 窗口{W0.date()}~{W1.date()}\n', flush=True)
    out = {}
    for name, (f, s, g) in VARIANTS.items():
        out[name] = {}
        sigs = {b: build_sig(df, f, s, g) for b, (df, _) in data.items()}
        print(f'########## {name} ##########', flush=True)
        for sname, (k_sl, k_tp, tp_mode) in SCHEMES.items():
            out[name][sname] = {}
            for lev in (1, 2, 4):
                r = run(data, sigs, k_sl, k_tp, tp_mode, lev)
                out[name][sname][f'lev{lev}x'] = r
                print(f"  {sname} [{lev}x] 总{r['ret']:>+9.1f}% MDD{r['mdd']:>7.1f}% "
                      f"月均回撤{r['avg_mdd_m']:>6.2f}% 月最大回撤{r['max_mdd_m']:>7.2f}% "
                      f"笔{r['trades']:>5} 胜{r['winrate']:>5}% 爆{r['liq_coins']} | "
                      f"最差月{r['mn']:>+7.2f}% 月std{r['std']:>5.2f}", flush=True)
            print('', flush=True)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'top30_2024_tpsl_compare_1h.json')
    with open(p, 'w', encoding='utf-8') as fp:
        json.dump(out, fp, ensure_ascii=False, indent=1, default=float)
    print(f'明细 → {p}')


if __name__ == '__main__':
    main()
