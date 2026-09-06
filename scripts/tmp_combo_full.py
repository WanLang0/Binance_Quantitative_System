# -*- coding: utf-8 -*-
"""30市值币 × 方案一+方案三五五开组合 · 连续全期回测（2024-01 ~ 2026-08）

与之前月度切片（月末强平）不同，本测试为连续口径：
- 每币资金对半：5000 跑方案一(15m 双向 tpsl5) + 5000 跑方案三(1h 双向 sl5)
- 两个子策略各自连续运行（月末不强平），仅窗口末强平一次
- 组合权益 = 15m权益曲线 + 1h权益曲线(对齐到15m时间戳, 前值填充)
- 统计：年度收益/下单数、全局及分年最大回撤、月收益均值、胜率、逐月收益
- 同口径重跑纯方案一(10000)与纯方案三(10000)作对比

其余口径不变：2023-10预热、95%仓位、单边手续费0.1%、止盈止损按收盘价、做空现金背书。
"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from divergence_signals import DIVERGENCE_VARIANTS, build_variant_signals

COMM = 0.001
VARIANT = 'macd+背离+量能'
UD, UM, UV = DIVERGENCE_VARIANTS[VARIANT]
BASES = ['BTC', 'ETH', 'BNB', 'XRP', 'SOL', 'TRX', 'HYPE', 'ZEC', 'DOGE', 'XMR',
         'LINK', 'ADA', 'XLM', 'BCH', 'CC', 'LTC', 'UNI', 'GRAM', 'HBAR', 'AVAX',
         'SUI', 'NEAR', 'M', 'TAO', 'ASTER', 'AAVE', 'ONDO', 'MORPHO', 'DOT', 'ICP']
W0 = pd.Timestamp('2024-01-01', tz='UTC')
W1 = pd.Timestamp('2026-09-01', tz='UTC')
YEARS = ('2024', '2025', '2026')


def _close(cash, units, entry, price, side, comm=COMM):
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm


def simulate_cont(df, signals, tp, sl, mode, initial):
    """连续回测：返回 (逐bar权益Series, 平仓时间索引数组, 胜/负数组) 或 None"""
    idx = df.index.to_numpy(); close = df['close'].to_numpy(); sig = signals.to_numpy()
    cash = initial; units = 0.0; entry = 0.0; side = 0
    eq_t = []; eq_v = []
    fl_t = []; fl_w = []
    entry_cash = initial
    for i in range(len(df)):
        price = close[i]
        if not np.isfinite(price) or price <= 0:
            continue
        s = int(sig[i]) if i > 0 else 0
        if side != 0 and entry > 0:
            r = (price - entry) / entry if side > 0 else (entry - price) / entry
            if (tp and r >= tp) or (sl and r <= -sl):
                c2 = _close(cash, units, entry, price, side)
                fl_t.append(idx[i]); fl_w.append(c2 > entry_cash)
                cash = c2; units = 0.0; side = 0
                eq_t.append(idx[i]); eq_v.append(cash)
                continue
        eq = cash + (units * price if side > 0 else units * entry + (entry - price) * units if side < 0 else 0)
        if eq <= 0:
            return None
        eq_t.append(idx[i]); eq_v.append(eq)
        if s == 1 and side <= 0:
            if side < 0:
                c2 = _close(cash, units, entry, price, side)
                fl_t.append(idx[i]); fl_w.append(c2 > entry_cash)
                cash = c2; units = 0.0; side = 0
            u = (cash * 0.95) / (price * (1 + COMM))
            if u > 0:
                entry_cash = cash
                cash -= u * price * (1 + COMM); units = u; entry = price; side = 1
        elif s == -1 and side >= 0:
            if side > 0:
                c2 = _close(cash, units, entry, price, side)
                fl_t.append(idx[i]); fl_w.append(c2 > entry_cash)
                cash = c2; units = 0.0; side = 0
            if mode == 'long_short':
                u = (cash * 0.95) / price
                if u > 0:
                    entry_cash = cash
                    cash -= u * price * (1 + COMM); units = u; entry = price; side = -1
    if side != 0 and len(df) > 0:
        c2 = _close(cash, units, entry, close[-1], side)
        fl_t.append(idx[-1]); fl_w.append(c2 > entry_cash)
        cash = c2
        eq_t.append(idx[-1]); eq_v.append(cash)
    eq = pd.Series(eq_v, index=pd.DatetimeIndex(eq_t)).sort_index()
    eq = eq[~eq.index.duplicated(keep='last')]   # 同bar开平仓都记录过权益，保留最后一次
    if not fl_t:
        return None
    return eq, pd.DatetimeIndex(fl_t), np.array(fl_w)


def yearly_stats(eq, fl_t, fl_w):
    """按年统计：收益/下单数/胜率 + 全期MDD与分年MDD"""
    out = {}
    ts = eq.index
    marks = [('2024', pd.Timestamp('2024-01-01', tz='UTC'), pd.Timestamp('2025-01-01', tz='UTC')),
             ('2025', pd.Timestamp('2025-01-01', tz='UTC'), pd.Timestamp('2026-01-01', tz='UTC')),
             ('2026', pd.Timestamp('2026-01-01', tz='UTC'), W1)]
    prev = eq.iloc[0]
    for y, a, b in marks:
        seg = eq[(ts >= a) & (ts < b)]
        n = int(((fl_t >= a) & (fl_t < b)).sum())
        w = int(fl_w[(fl_t >= a) & (fl_t < b)].sum()) if n else 0
        if len(seg):
            ret = (seg.iloc[-1] / prev - 1) * 100
            peak = seg.cummax()
            mdd = float(((seg - peak) / peak * 100).min())
            prev = seg.iloc[-1]
        else:
            ret, mdd = None, None
        out[y] = {'ret': ret, 'n': n, 'wins': w,
                  'wr': round(w / n * 100, 1) if n else None, 'mdd': mdd}
    peak = eq.cummax()
    out['mdd_all'] = float(((eq - peak) / peak * 100).min())
    out['ret_all'] = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    return out


def month_series(eq):
    """月末权益 → 月收益率Series"""
    me = eq.resample('ME').last().dropna()
    return me.pct_change().dropna() * 100


def main():
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
    rows = {'一': {}, '三': {}, '组合': {}}
    months_grid = pd.period_range('2024-02', W1, freq='M')
    month_rets = {'一': {}, '三': {}, '组合': {}}
    skipped = []
    for k, base in enumerate(BASES, 1):
        p15 = os.path.join(cache, f'{base}_15m.pkl')
        p1h = os.path.join(cache, f'{base}_1h.pkl')
        if not (os.path.exists(p15) and os.path.exists(p1h)):
            skipped.append(base); continue
        ok = True
        res = {}
        for tf, init, key in (('15m', 10000.0, '一'), ('1h', 10000.0, '三')):
            df = pd.read_pickle(p15 if tf == '15m' else p1h)
            df = df[(df.index >= W0 - pd.Timedelta(days=1)) & (df.index < W1)]
            if df is None or len(df) < 500:
                ok = False; break
            _, sig = build_variant_signals(df, UD, UM, UV)
            tp, sl = (0.05, 0.05) if tf == '15m' else (None, 0.05)
            r = simulate_cont(df, sig, tp, sl, 'long_short', init)
            if r is None:
                ok = False; break
            res[key] = r
        if not ok:
            skipped.append(base); continue
        # 纯方案
        for key in ('一', '三'):
            eq, ft, fw = res[key]
            rows[key][base] = yearly_stats(eq, ft, fw)
            ms = month_series(eq)
            month_rets[key][base] = ms
        # 组合：各5000（把纯方案权益缩半相加 = 等价于各投一半资金）
        eqA, ftA, fwA = res['一']
        eqB, ftB, fwB = res['三']
        grid = eqA.index.union(eqB.index)
        eqC = ((eqA / 2).reindex(grid).ffill() + (eqB / 2).reindex(grid).ffill()).dropna()
        ftC = ftA.append(ftB)
        order = np.argsort(ftC.values)
        wC = np.concatenate([fwA, fwB])[order]
        rows['组合'][base] = yearly_stats(eqC, ftC, wC)
        month_rets['组合'][base] = month_series(eqC)
        print(f'[{k}/{len(BASES)}] {base:8} done', flush=True)
    print(f'剔除(数据不足/无交易): {skipped}')

    # ---- 汇总输出 ----
    def agg(key):
        rr = rows[key]
        out = {'per_year': {}}
        for y in YEARS:
            rets = [v[y]['ret'] for v in rr.values()
                    if v[y]['ret'] is not None and np.isfinite(v[y]['ret'])]
            mdds = [v[y]['mdd'] for v in rr.values()
                    if v[y]['mdd'] is not None and np.isfinite(v[y]['mdd'])]
            ns = [v[y]['n'] for v in rr.values()]
            ws = [v[y]['wins'] for v in rr.values()]
            out['per_year'][y] = {'ret': round(float(np.mean(rets)), 1),
                                  'mdd': round(float(np.mean(mdds)), 1),
                                  'mdd_max': round(float(min(mdds)), 1),
                                  'n': int(sum(ns)),
                                  'wr': round(sum(ws) / sum(ns) * 100, 1) if sum(ns) else None}
        out['mdd_all_avg'] = round(float(np.mean([v['mdd_all'] for v in rr.values()
                                                  if np.isfinite(v['mdd_all'])])), 1)
        out['mdd_all_max'] = round(float(min(v['mdd_all'] for v in rr.values()
                                             if np.isfinite(v['mdd_all']))), 1)
        out['ret_all_avg'] = round(float(np.mean([v['ret_all'] for v in rr.values()
                                                  if np.isfinite(v['ret_all'])])), 1)
        return out

    # 月度（跨币平均）
    def month_agg(key):
        ms = {}
        for b, m in month_rets[key].items():
            for mo, v in m.items():
                ms.setdefault(str(mo.to_period('M')), []).append(float(v))
        grid = [str(m) for m in months_grid]
        vals = [np.mean(ms[g]) if g in ms else None for g in grid]
        vv = [v for v in vals if v is not None]
        nav = 1.0
        for v in vv:
            nav *= (1 + v / 100)
        return grid, vals, {'avg': float(np.mean(vv)), 'nav': nav,
                            'neg': int(sum(v < 0 for v in vv)),
                            'worst': float(min(vv)), 'best': float(max(vv)),
                            'months': len(vv)}

    result = {}
    for key, label in (('一', '纯方案一 15m双向tpsl5'), ('三', '纯方案三 1h双向sl5'), ('组合', '组合 一+三 对半')):
        a = agg(key)
        grid, vals, ms = month_agg(key)
        result[label] = {'yearly': a['per_year'], 'mdd_all_avg': a['mdd_all_avg'],
                         'mdd_all_max': a['mdd_all_max'], 'ret_all_avg': a['ret_all_avg'],
                         'monthly_avg': round(ms['avg'], 2), 'monthly_nav': round(ms['nav'], 3),
                         'neg_months': ms['neg'], 'months': ms['months'],
                         'worst_month': round(ms['worst'], 2), 'best_month': round(ms['best'], 2),
                         'monthly_series': {g: (round(v, 2) if v is not None else None) for g, v in zip(grid, vals)}}
        print(f'\n===== {label}（连续口径，30币，剔除{len(skipped)}只后{len(rows[key])}只） =====')
        for y in YEARS:
            d = a['per_year'][y]
            print(f"  {y:6} 年收益{d['ret']:+7.1f}%  下单{d['n']:6d}笔  胜率{d['wr']}%  "
                  f"平均MDD{d['mdd']:6.1f}%  最差MDD{d['mdd_max']:6.1f}%")
        print(f"  全期   收益{a['ret_all_avg']:+7.1f}%  平均MDD{a['mdd_all_avg']}%  最差MDD{a['mdd_all_max']}%  "
              f"月均{ms['avg']:+.2f}%  月度复利{ms['nav']:.3f}x  亏损月{ms['neg']}/{ms['months']}  "
              f"最差月{ms['worst']:+.2f}%")

    with open('scripts/results/top30_combo_full_2024_2026.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1, default=float)
    print('\n明细 → scripts/results/top30_combo_full_2024_2026.json')

    # 组合逐月
    grid = list(result['组合 一+三 对半']['monthly_series'].keys())
    print('\n== 组合(一+三对半) 逐月收益（跨币平均） ==')
    for g in grid:
        v = result['组合 一+三 对半']['monthly_series'][g]
        print(f"  {g}  {v:+7.2f}%" if v is not None else f'  {g}      —')


if __name__ == '__main__':
    main()
