# -*- coding: utf-8 -*-
"""止盈判定对照回测：盘中触发（现行C判定） vs 收盘价确认
口径: 1.2x量能, 三种MACD参数(12/26/9, 12/16/5, 12/16/7) × 杠杆(1/2/4), 1h,
30币均分30份独立复利, 0.05%/边, ATR14止1.5/盈2截断[1%,8%], 止损始终收盘确认。
- tp_mode='intraday': 止盈盘中触发(高点/低点触及即成交于止盈价) —— 现行口径
- tp_mode='close'   : 止盈收盘确认(收盘价穿越止盈位, 按收盘价市价成交)
输出: 总收益/MDD/单币MDD/笔数/胜率/爆仓币数 对照。
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange

from indicators import TechnicalIndicators
from tmp_combo_full import BASES, W0, W1

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
COMM = 0.0005
INIT = 10000.0 / len(BASES)
MAINT = 0.005
CLIP_HI = 0.08
PER, K_SL, K_TP = 14, 1.5, 2.0
VOL_MULT = 1.2
VARIANTS = {'12/26/9': (12, 26, 9), '12/16/5': (12, 16, 5), '12/16/7': (12, 16, 7)}


def load_data(tf):
    data = {}
    for base in BASES:
        p = os.path.join(CACHE, f'{base}_{tf}.pkl')
        if not os.path.exists(p):
            continue
        df = pd.read_pickle(p)
        df = df[df.index < W1]
        if len(df) < 300:
            continue
        atr = AverageTrueRange(df['high'], df['low'], df['close'], PER).average_true_range()
        data[base] = (df, atr)
    return data


def build_signal(df, f, s, g, vol_mult):
    dft = TechnicalIndicators.calculate_macd(df, f, s, g)
    macd_buy = (dft['MACD'] > dft['MACD_signal']) & (dft['MACD'].shift(1) <= dft['MACD_signal'].shift(1))
    macd_sell = (dft['MACD'] < dft['MACD_signal']) & (dft['MACD'].shift(1) >= dft['MACD_signal'].shift(1))
    vol_up = df['volume'] > df['volume'].rolling(20).mean() * vol_mult
    sig = pd.Series(0, index=df.index)
    sig[(macd_buy & vol_up).fillna(False)] = 1
    sig[(macd_sell & vol_up).fillna(False)] = -1
    return sig


def simulate_lev(df, signals, initial, atr, k_sl, k_tp, lev, t0, tp_mode='intraday'):
    idx = df.index.to_numpy(); close = df['close'].to_numpy()
    hi = df['high'].to_numpy(); lo = df['low'].to_numpy()
    sig = signals.to_numpy()
    atr_a = atr.to_numpy() if atr is not None else None
    cash = initial; units = 0.0; entry = 0.0; side = 0; margin = 0.0
    eq_t = []; eq_v = []; fl_t = []; fl_w = []
    entry_cash = initial; n_liq = 0; n_trade = 0
    cur_tp = cur_sl = 0.0
    for i in range(len(df)):
        price = close[i]
        if not np.isfinite(price) or price <= 0:
            continue
        s = int(sig[i]) if i > 0 else 0
        if side != 0 and entry > 0:
            if side > 0:
                tp_lvl, sl_lvl = entry * (1 + cur_tp), entry * (1 - cur_sl)
                if tp_mode == 'intraday':
                    hit_tp = hi[i] >= tp_lvl
                else:
                    hit_tp = price >= tp_lvl
                hit_sl_c = price <= sl_lvl
            else:
                tp_lvl, sl_lvl = entry * (1 - cur_tp), entry * (1 + cur_sl)
                if tp_mode == 'intraday':
                    hit_tp = lo[i] <= tp_lvl
                else:
                    hit_tp = price <= tp_lvl
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
                # 盘中触发按止盈价成交; 收盘确认按收盘价市价成交(可能优于止盈位)
                exit_price = tp_lvl if tp_mode == 'intraday' else price
            if liquidated or exit_price is not None:
                n_trade += 1
                if liquidated:
                    n_liq += 1
                    cash = cash + margin - maint
                else:
                    pnl = units * (exit_price - entry) if side > 0 else units * (entry - exit_price)
                    cash = cash + margin + pnl - abs(units * exit_price * COMM)
                fl_t.append(idx[i]); fl_w.append(not liquidated and cash > entry_cash)
                cash = max(cash, 0.0); units = 0.0; side = 0; margin = 0.0
                eq_t.append(idx[i]); eq_v.append(cash)
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
            cash -= margin * (1 + COMM)
            if atr_a is not None and np.isfinite(atr_a[i]) and atr_a[i] > 0:
                apct = atr_a[i] / price
                cur_sl = float(np.clip(k_sl * apct, 0.01, CLIP_HI))
                cur_tp = float(np.clip(k_tp * apct, 0.01, CLIP_HI))
            else:
                cur_sl, cur_tp = 0.05, 0.05
            entry_cash = cash + margin
        elif s == -1 and side >= 0 and idx[i] >= t0:
            if side > 0:
                pnl = units * (price - entry)
                cash = cash + margin + pnl - abs(units * price * COMM)
                cash = max(cash, 0.0); units = 0.0; side = 0; margin = 0.0
            margin = cash * 0.95
            units = (margin * lev) / price
            entry = price; side = -1
            cash -= margin * (1 + COMM)
            if atr_a is not None and np.isfinite(atr_a[i]) and atr_a[i] > 0:
                apct = atr_a[i] / price
                cur_sl = float(np.clip(k_sl * apct, 0.01, CLIP_HI))
                cur_tp = float(np.clip(k_tp * apct, 0.01, CLIP_HI))
            else:
                cur_sl, cur_tp = 0.05, 0.05
            entry_cash = cash + margin
    if side != 0 and len(df) > 0:
        pnl = units * (close[-1] - entry) if side > 0 else units * (entry - close[-1])
        cash = max(cash + margin + pnl - abs(units * close[-1] * COMM), 0.0)
        fl_t.append(idx[-1]); fl_w.append(cash > entry_cash)
        eq_t.append(idx[-1]); eq_v.append(cash)
    if not fl_t:
        return None
    eq = pd.Series(eq_v, index=pd.DatetimeIndex(eq_t)).sort_index()
    eq = eq[~eq.index.duplicated(keep='last')]
    return eq, pd.DatetimeIndex(fl_t), np.array(fl_w), n_liq, n_trade


def run(data, f, s, g, lev, tp_mode):
    eqs = {}; coin_mdd = []; n_all = 0; w_all = 0; grid = None; liq_coins = 0
    for base, (df, atr) in data.items():
        sig = build_signal(df, f, s, g, VOL_MULT)
        r = simulate_lev(df, sig, INIT, atr, K_SL, K_TP, lev, W0, tp_mode)
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
    yr = {}
    for y, a, b in [('2024', '2024-01-01', '2025-01-01'), ('2025', '2025-01-01', '2026-01-01'),
                    ('2026', '2026-01-01', '2026-09-01')]:
        a, b = pd.Timestamp(a, tz='UTC'), pd.Timestamp(b, tz='UTC')
        seg = nav[(nav.index >= a) & (nav.index < b)]
        if len(seg):
            yr[y] = round(float((seg.iloc[-1] / (nav[nav.index < a].iloc[-1] if len(nav[nav.index < a]) else 1.0) - 1) * 100), 1)
    return dict(ret=round(ret, 1), mdd=round(mdd, 1),
                coin_mdd_avg=round(float(np.mean(coin_mdd)), 1), coin_mdd_max=round(float(np.min(coin_mdd)), 1),
                trades=n_all, winrate=round(w_all / n_all * 100, 1) if n_all else None,
                liq_coins=liq_coins, yearly=yr)


def main():
    print('加载数据...', flush=True)
    data = load_data('1h')
    print(f'1h 周期 {len(data)} 币 · 量能{VOL_MULT}x', flush=True)
    results = {}
    for name, (f, s, g) in VARIANTS.items():
        results[name] = {}
        for lev in (1, 2, 4):
            results[name][f'lev{lev}x'] = {
                'tp_intraday': run(data, f, s, g, lev, 'intraday'),
                'tp_close': run(data, f, s, g, lev, 'close'),
            }
        print(f'== {name} 完成 ==', flush=True)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'tp_close_vs_intraday_1h.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=float)
    for name in VARIANTS:
        print(f'\n===== {name} (1h · 1.2x量能) =====', flush=True)
        for lev in (1, 2, 4):
            a = results[name][f'lev{lev}x']['tp_intraday']
            b = results[name][f'lev{lev}x']['tp_close']
            print(f'  [{lev}x] 盘中触发: 总{a["ret"]:>+9.1f}% MDD{a["mdd"]:>7.1f}% 单币MDD均{a["coin_mdd_avg"]:>6.1f}% '
                  f'笔{a["trades"]:>5} 胜{a["winrate"]:>5}% 爆{a["liq_coins"]}', flush=True)
            print(f'        收盘确认: 总{b["ret"]:>+9.1f}% MDD{b["mdd"]:>7.1f}% 单币MDD均{b["coin_mdd_avg"]:>6.1f}% '
                  f'笔{b["trades"]:>5} 胜{b["winrate"]:>5}% 爆{b["liq_coins"]}', flush=True)
            print(f'        年度 盘中{a["yearly"]} -> 收盘{b["yearly"]}', flush=True)
    print(f'\n明细 → {p}')


if __name__ == '__main__':
    main()
