# -*- coding: utf-8 -*-
"""2021 年初市值前30 可测池(26币·完整数据) × C3盘中止盈 × 2021-01~2024-01 回测
口径: 1h · 30币均分独立复利 · 1.2x量能(vol20) · ATR14止1.0/盈1.2截断[1%,8%] · C判定(止盈盘中/止损收盘确认) · 双向 · 0.05%/边
窗口: 2020-10-01 预热 → 2021-01-01 ~ 2024-01-01 回测
三种 MACD 参数(12/26/9, 12/16/5, 12/16/7) × 杠杆(1/2/4)
输出: 总收益/全局MDD/单币MDD/笔数/胜率/爆仓/月度std/最差月/月正比例/年度/逐月
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange

from tmp_tp_close_backtest import simulate_lev, COMM, MAINT, VARIANTS
from tmp_macd_tune_scan import build_signal2

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')

# 2021 年初市值前30 可测池（CMC 2021-01-03 快照，剔除稳定币/封装/无合约/无完整数据，按市值顺延补足；实际26币）
POOL = ['BTC','ETH','LTC','XRP','DOT','BCH','ADA','BNB','LINK','XLM','XMR',
        'THETA','TRX','VET','XTZ','UNI','DOGE','ATOM','NEO','FIL','DASH','ETC',
        'ZEC','ALGO','NEAR','AVAX']
INIT = 10000.0 / len(POOL)
PER, CLIP_HI = 14, 0.08
WARM = pd.Timestamp('2020-10-01', tz='UTC')
W0 = pd.Timestamp('2021-01-01', tz='UTC')
W1 = pd.Timestamp('2024-01-01', tz='UTC')
CFG = dict(vol_mult=1.2, vol_win=20, k_sl=1.0, k_tp=1.2, trend='none', tp_mode='intraday')


def load_data():
    data = {}
    for base in POOL:
        p = os.path.join(CACHE, f'{base}USDT_1h_2021_2023.pkl')
        if not os.path.exists(p):
            continue
        df = pd.read_pickle(p)
        df = df[df.index < W1]
        if len(df) < 300:
            continue
        atr = AverageTrueRange(df['high'], df['low'], df['close'], PER).average_true_range()
        data[base] = (df, atr)
    return data


def run(data, f, s, g, lev):
    eqs = {}; coin_mdd = []; n_all = 0; w_all = 0; grid = None; liq_coins = 0
    for base, (df, atr) in data.items():
        sig = build_signal2(df, f, s, g, CFG['vol_mult'], CFG['vol_win'], CFG['trend'], CFG['tp_mode'])
        r = simulate_lev(df, sig, INIT, atr, CFG['k_sl'], CFG['k_tp'], lev, W0, CFG['tp_mode'])
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
    for y, a, b in [('2021', '2021-01-01', '2022-01-01'), ('2022', '2022-01-01', '2023-01-01'),
                    ('2023', '2023-01-01', '2024-01-01')]:
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
    data = load_data()
    print(f'2021年初市值前30可测池({len(data)}币) · C3盘中止盈 · 1h · 窗口2021-01~2024-01\n', flush=True)
    print(f'币池: {list(data.keys())}\n', flush=True)
    out = {}
    for name, (f, s, g) in VARIANTS.items():
        out[name] = {}
        print(f'########## {name} ##########', flush=True)
        for lev in (1, 2, 4):
            r = run(data, f, s, g, lev)
            out[name][f'lev{lev}x'] = r
            print(f"  [{lev}x] 总{r['ret']:>+9.1f}% MDD{r['mdd']:>7.1f}% 单币MDD均{r['coin_mdd_avg']:>6.1f}% "
                  f"笔{r['trades']:>5} 胜{r['winrate']:>5}% 爆{r['liq_coins']} | "
                  f"月均{r['avg']:>+6.2f}% 月std{r['std']:>5.2f} 最差{r['mn']:>+7.2f}% 正{r['posr']:>4.0f}%", flush=True)
            print(f"        年度 {r['yearly']}", flush=True)
        print('', flush=True)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'top30_2021_c3_intraday_1h.json')
    with open(p, 'w', encoding='utf-8') as fp:
        json.dump(out, fp, ensure_ascii=False, indent=1, default=float)
    print(f'明细 → {p}')


if __name__ == '__main__':
    main()
