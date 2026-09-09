# -*- coding: utf-8 -*-
"""MACD 量能1.2x · 止盈口径=盘中触发 · 1h · 30币均分独立复利
输出每策略(3种参数)×杠杆(1/2/4) 的：逐月组合收益% + 逐月平均持币份数(同时持仓币数/30)。
口径与 tmp_tp_close_backtest.py 完全一致（仅 tp_mode='intraday'）。
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange

from tmp_combo_full import BASES, W0, W1
from tmp_tp_close_backtest import load_data, build_signal, VARIANTS, VOL_MULT, COMM, INIT, MAINT, PER, K_SL, K_TP, CLIP_HI


def simulate(df, signals, initial, atr, lev, t0, k_sl=K_SL, k_tp=K_TP, tp_mode='intraday'):
    """同 tmp_tp_close_backtest.simulate_lev，额外记录逐bar是否持仓 ip"""
    idx = df.index.to_numpy(); close = df['close'].to_numpy()
    hi = df['high'].to_numpy(); lo = df['low'].to_numpy()
    sig = signals.to_numpy()
    atr_a = atr.to_numpy() if atr is not None else None
    cash = initial; units = 0.0; entry = 0.0; side = 0; margin = 0.0
    eq_t = []; eq_v = []; in_pos = np.zeros(len(df), dtype=bool)
    cur_tp = cur_sl = 0.0
    for i in range(len(df)):
        price = close[i]
        if not np.isfinite(price) or price <= 0:
            continue
        s = int(sig[i]) if i > 0 else 0
        if side != 0 and entry > 0:
            if side > 0:
                tp_lvl, sl_lvl = entry * (1 + cur_tp), entry * (1 - cur_sl)
                hit_tp = hi[i] >= tp_lvl if tp_mode == 'intraday' else price >= tp_lvl
                hit_sl_c = price <= sl_lvl
            else:
                tp_lvl, sl_lvl = entry * (1 - cur_tp), entry * (1 + cur_sl)
                hit_tp = lo[i] <= tp_lvl if tp_mode == 'intraday' else price <= tp_lvl
                hit_sl_c = price >= sl_lvl
            notional = units * entry
            maint = notional * MAINT
            equity = cash + margin + (units * (price - entry) if side > 0 else units * (entry - price))
            exit_price = None; liquidated = False
            if equity <= maint:
                liquidated = True
            elif hit_sl_c:
                exit_price = sl_lvl
            elif hit_tp:
                exit_price = tp_lvl if tp_mode == 'intraday' else price
            if liquidated or exit_price is not None:
                if liquidated:
                    # 逐仓爆仓: 亏掉保证金, 只保留未占用钱包余额 (与 simulate_lev 修复一致)
                    cash = max(cash, 0.0)
                else:
                    pnl = units * (exit_price - entry) if side > 0 else units * (entry - exit_price)
                    cash = cash + margin + pnl - abs(units * exit_price * COMM)
                cash = max(cash, 0.0); units = 0.0; side = 0; margin = 0.0
                eq_t.append(idx[i]); eq_v.append(cash)
                in_pos[i] = False
                continue
        equity = cash + margin + (units * (price - entry) if side > 0 else units * (entry - price) if side < 0 else 0)
        if equity <= 0:
            return None
        eq_t.append(idx[i]); eq_v.append(equity)
        if s == 1 and side <= 0 and idx[i] >= t0:
            if side < 0:
                pnl = units * (entry - price)
                cash = cash + margin + pnl - abs(units * price * COMM)
                cash = max(cash, 0.0); units = 0.0; side = 0; margin = 0.0
            margin = cash * 0.95
            units = (margin * lev) / price
            entry = price; side = 1
            # 开仓手续费按名义值 notional=margin*lev 收取 (与 simulate_lev 一致)
            cash -= margin + abs(units * price * COMM)
            if atr_a is not None and np.isfinite(atr_a[i]) and atr_a[i] > 0:
                apct = atr_a[i] / price
                cur_sl = float(np.clip(k_sl * apct, 0.01, CLIP_HI))
                cur_tp = float(np.clip(k_tp * apct, 0.01, CLIP_HI))
            else:
                cur_sl, cur_tp = 0.05, 0.05
        elif s == -1 and side >= 0 and idx[i] >= t0:
            if side > 0:
                pnl = units * (price - entry)
                cash = cash + margin + pnl - abs(units * price * COMM)
                cash = max(cash, 0.0); units = 0.0; side = 0; margin = 0.0
            margin = cash * 0.95
            units = (margin * lev) / price
            entry = price; side = -1
            cash -= margin + abs(units * price * COMM)
            if atr_a is not None and np.isfinite(atr_a[i]) and atr_a[i] > 0:
                apct = atr_a[i] / price
                cur_sl = float(np.clip(k_sl * apct, 0.01, CLIP_HI))
                cur_tp = float(np.clip(k_tp * apct, 0.01, CLIP_HI))
            else:
                cur_sl, cur_tp = 0.05, 0.05
        in_pos[i] = (side != 0)
    if side != 0 and len(df) > 0:
        pnl = units * (close[-1] - entry) if side > 0 else units * (entry - close[-1])
        cash = max(cash + margin + pnl - abs(units * close[-1] * COMM), 0.0)
        eq_t.append(idx[-1]); eq_v.append(cash)
        in_pos[-1] = False
    if not eq_t:
        return None
    eq = pd.Series(eq_v, index=pd.DatetimeIndex(eq_t)).sort_index()
    eq = eq[~eq.index.duplicated(keep='last')]
    return eq, pd.Series(in_pos, index=df.index)


def run_monthly(data, f, s, g, lev):
    eqs = {}; inps = {}; grid = None
    for base, (df, atr) in data.items():
        sig = build_signal(df, f, s, g, VOL_MULT)
        r = simulate(df, sig, INIT, atr, lev, W0)
        if r is None:
            continue
        eq, ip = r
        eqs[base] = eq / INIT
        inps[base] = ip
        grid = eq.index if grid is None else grid.union(eq.index)
    if not eqs:
        return None
    # 组合净值 = 等权平均
    nav = sum(e.reindex(grid).ffill().fillna(1.0) for e in eqs.values()) / len(eqs)
    nav = nav[nav.index >= W0].dropna()
    # 持币份数矩阵
    ipgrid = None
    for ip in inps.values():
        ipgrid = ip.index if ipgrid is None else ipgrid.union(ip.index)
    mat = pd.DataFrame({b: ip.reindex(ipgrid).fillna(False).astype(bool) for b, ip in inps.items()})
    mat = mat[mat.index >= W0]
    n_pos = mat.sum(axis=1)
    # 逐月收益
    me = nav.resample('ME').last().dropna()
    mr = me.pct_change().dropna() * 100
    # 逐月平均持币份数（第一个月用当月bar均值近似）
    mo_pos = n_pos.resample('ME').mean()
    monthly = {}
    for t, v in mr.items():
        key = str(t.to_period('M'))
        monthly[key] = {
            'ret': round(float(v), 2),
            'pos': round(float(mo_pos.get(t, np.nan)), 1),
            'pos_pct': round(float(mo_pos.get(t, np.nan) / len(inps) * 100), 1),
        }
    return monthly


def main():
    print('加载数据...', flush=True)
    data = load_data('1h')
    print(f'1h 周期 {len(data)} 币 · 量能{VOL_MULT}x', flush=True)
    out = {}
    for name, (f, s, g) in VARIANTS.items():
        out[name] = {}
        for lev in (1, 2, 4):
            out[name][f'lev{lev}x'] = run_monthly(data, f, s, g, lev)
            print(f'== {name} {lev}x 完成 ==', flush=True)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'macd_vol_monthly_pos_1h.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    # 控制台摘要
    for name in VARIANTS:
        print(f'\n===== {name} · 盘中触发 · 逐月 =====', flush=True)
        for lev in (1, 2, 4):
            data2 = out[name][f'lev{lev}x']
            if not data2:
                continue
            rs = [v['ret'] for v in data2.values()]
            ps = [v['pos'] for v in data2.values()]
            print(f'  [{lev}x] 月均{np.mean(rs):+7.2f}%  亏损月{sum(r<0 for r in rs)}/{len(rs)}  '
                  f'最差月{min(rs):+7.2f}%  月均持币{np.mean(ps):4.1f}份', flush=True)
    print(f'\n明细 → {p}', flush=True)


if __name__ == '__main__':
    main()
