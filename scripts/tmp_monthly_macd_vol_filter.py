# -*- coding: utf-8 -*-
"""综合量化30币 × macd+背离+量能 × 三方案 × 双杀过滤器对比（月度口径）

变体：
- base    : 原版（30币基线，应与 40币回测切30币口径一致）
- confirm : 反向信号二次确认——信号反转时只平仓，下一根K线收盘无反向信号才开新仓
- adx20   : ADX(14) >= 20 才允许开仓（持仓的止盈止损/反转平仓不受限）
- adx25   : ADX(14) >= 25 才允许开仓

口径不变：2023-10预热、初始1万、95%仓位、单边手续费0.1%、月末强平、
止盈止损按收盘价、本地缓存不联网、胜率=单笔回款>开仓本金。
"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import ta

from divergence_signals import DIVERGENCE_VARIANTS, build_variant_signals

COMM = 0.001
INITIAL = 10000.0
VARIANT = 'macd+背离+量能'
UD, UM, UV = DIVERGENCE_VARIANTS[VARIANT]
CONFIGS = [
    ('15m', 'long_short', 'tpsl5', '方案一 15m双向 tpsl5'),
    ('15m', 'long_only', 'none', '方案二 15m仅多 不设'),
    ('1h', 'long_short', 'sl5', '方案三 1h双向 sl5'),
]
TPSL_CFG = {'none': (None, None), 'sl5': (None, 0.05), 'tpsl5': (0.05, 0.05)}
MIN_BARS = {'15m': 1400, '1h': 350}
FILTERS = [('base', False, None), ('confirm', True, None),
           ('adx20', False, 20.0), ('adx25', False, 25.0)]
MONTHS = pd.period_range('2024-01', pd.Timestamp.utcnow(), freq='M')
YEARS = ('2024', '2025', '2026')

BASES = ['BTC', 'ETH', 'BNB', 'XRP', 'SOL', 'TRX', 'HYPE', 'ZEC', 'DOGE', 'XMR',
         'LINK', 'ADA', 'XLM', 'BCH', 'CC', 'LTC', 'UNI', 'GRAM', 'HBAR', 'AVAX',
         'SUI', 'NEAR', 'M', 'TAO', 'ASTER', 'AAVE', 'ONDO', 'MORPHO', 'DOT', 'ICP']


def _close(cash, units, entry, price, side, comm=COMM):
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm


def simulate(df, signals, adx, tp=None, sl=None, mode='long_only',
             confirm=False, adx_min=None, initial=INITIAL, comm=COMM):
    """月度窗口回测。adx: numpy数组或None；confirm: 信号延迟一根确认；adx_min: 开仓ADX阈值"""
    close = df['close'].to_numpy(); sig = signals.to_numpy()
    cash = initial; units = 0.0; entry = 0.0; side = 0
    n = wins = losses = 0
    entry_cash = initial
    pending = 0   # 二次确认：待开仓方向

    def _flat(price):
        nonlocal cash, units, side, n, wins, losses
        c2 = _close(cash, units, entry, price, side)
        if c2 > entry_cash:
            wins += 1
        else:
            losses += 1
        cash = c2; units = 0.0; side = 0; n += 1

    def _open(direction, price):
        nonlocal cash, units, entry, side, entry_cash
        u = (cash * 0.95) / (price * (1 + comm)) if direction > 0 else (cash * 0.95) / price
        if u <= 0:
            return
        entry_cash = cash
        cash -= u * price * (1 + comm); units = u; entry = price; side = direction

    def _adx_ok(i):
        if adx_min is None:
            return True
        v = adx[i]
        return np.isfinite(v) and v >= adx_min

    for i in range(len(df)):
        price = close[i]
        if not np.isfinite(price) or price <= 0:
            continue
        s = int(sig[i]) if i > 0 else 0
        # 止盈/止损（按收盘价）
        if side != 0 and entry > 0:
            r = (price - entry) / entry if side > 0 else (entry - price) / entry
            if (tp and r >= tp) or (sl and r <= -sl):
                _flat(price)
                pending = 0
                continue
        eq = cash + (units * price if side > 0 else units * entry + (entry - price) * units if side < 0 else 0)
        if eq <= 0:
            return None
        if not confirm:
            # 原版逻辑 + ADX开仓过滤
            if s == 1 and side <= 0:
                if side < 0:
                    _flat(price)
                if _adx_ok(i):
                    _open(1, price)
            elif s == -1 and side >= 0:
                if side > 0:
                    _flat(price)
                if mode == 'long_short' and _adx_ok(i):
                    _open(-1, price)
        else:
            # 二次确认：信号bar只平仓+记pending；下一根无反向信号才开仓
            if s != 0:
                if side != 0 and ((side > 0 and s < 0) or (side < 0 and s > 0)):
                    _flat(price)
                if mode == 'long_short' or s == 1:
                    pending = s
                else:  # 仅多模式遇卖出信号：平多即可，不留空pending
                    pending = 0
            elif pending != 0 and side == 0:
                if _adx_ok(i):
                    _open(pending, price)
                pending = 0
    if side != 0 and len(df) > 0:
        _flat(close[-1])
    if n == 0:
        return None
    return {'ret': (cash / initial - 1) * 100, 'n': n, 'wins': wins, 'losses': losses}


def main():
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
    # res[月][配置名][过滤器] = {base: {...}}
    res = {str(m): {c[3]: {f[0]: {} for f in FILTERS} for c in CONFIGS} for m in MONTHS}
    for k, base in enumerate(BASES, 1):
        for tf in ('15m', '1h'):
            p = os.path.join(cache_dir, f'{base}_{tf}.pkl')
            if not os.path.exists(p):
                continue
            df = pd.read_pickle(p)
            if df is None or len(df) < 200:
                continue
            try:
                _, sig = build_variant_signals(df, UD, UM, UV)
            except Exception as e:
                print(f'{base:8} {tf} 信号出错: {e}')
                continue
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx().to_numpy()
            for m in MONTHS:
                w0 = m.to_timestamp(how='start').tz_localize('UTC')
                w1 = (m + 1).to_timestamp(how='start').tz_localize('UTC')
                dw = df[(df.index >= w0) & (df.index < w1)]
                if len(dw) < MIN_BARS[tf]:
                    continue
                sig_w = sig.loc[dw.index]
                a0 = df.index.get_loc(dw.index[0])
                adx_w = adx[a0:a0 + len(dw)]
                for ctf, cmode, ctpsl, cname in CONFIGS:
                    if ctf != tf:
                        continue
                    tp, sl = TPSL_CFG[ctpsl]
                    for fname, cfm, amin in FILTERS:
                        try:
                            r = simulate(dw, sig_w, adx_w, tp=tp, sl=sl, mode=cmode,
                                         confirm=cfm, adx_min=amin)
                        except Exception:
                            r = None
                        if r:
                            res[str(m)][cname][fname][base] = r
        print(f'[{k}/{len(BASES)}] {base:8} done', flush=True)

    # ---- 汇总 ----
    out = {'monthly': {}, 'summary': {}}
    for ctf, cmode, ctpsl, cname in CONFIGS:
        for fname, _, _ in FILTERS:
            rows = []
            for m in MONTHS:
                d = res[str(m)][cname][fname]
                if not d:
                    rows.append({'month': str(m)})
                    continue
                rets = [v['ret'] for v in d.values()]
                ws = sum(v['wins'] for v in d.values())
                ls = sum(v['losses'] for v in d.values())
                rows.append({'month': str(m),
                             'mean': round(float(np.mean(rets)), 2),
                             'pos': int(sum(x > 0 for x in rets)), 'cnt': len(rets),
                             'n': int(sum(v['n'] for v in d.values())),
                             'wins': int(ws), 'losses': int(ls)})
            out['monthly'][f'{cname}|{fname}'] = rows
            vals = [r['mean'] for r in rows if r.get('mean') is not None]
            tot_n = sum(r.get('n', 0) for r in rows)
            tot_w = sum(r.get('wins', 0) for r in rows)
            tot_l = sum(r.get('losses', 0) for r in rows)
            nav = 1.0
            for v in vals:
                nav *= (1 + v / 100)
            yr = {}
            for y in YEARS:
                yv = [r['mean'] for r in rows if r.get('mean') is not None and r['month'].startswith(y)]
                yn = sum(r.get('n', 0) for r in rows if r['month'].startswith(y))
                yw = sum(r.get('wins', 0) for r in rows if r['month'].startswith(y))
                yl = sum(r.get('losses', 0) for r in rows if r['month'].startswith(y))
                yr[y] = {'n': int(yn), 'win_rate': round(yw / (yw + yl) * 100, 1) if (yw + yl) else None,
                         'ret_sum': round(float(np.sum(yv)), 1) if yv else None}
            out['summary'][f'{cname}|{fname}'] = {
                'months': len(vals), 'avg': round(float(np.mean(vals)), 2),
                'neg_months': int(sum(v < 0 for v in vals)),
                'worst': round(float(min(vals)), 2), 'best': round(float(max(vals)), 2),
                'nav': round(nav, 3), 'n_all': int(tot_n),
                'win_rate': round(tot_w / (tot_w + tot_l) * 100, 1) if (tot_w + tot_l) else None,
                'per_year': yr,
            }
    with open('scripts/results/top30_macd_vol_monthly_filter.json', 'w', encoding='utf-8') as f:
        json.dump({'summary': out['summary'], 'monthly': out['monthly'],
                   'per_coin': res}, f, ensure_ascii=False, default=float)
    print('明细 → scripts/results/top30_macd_vol_monthly_filter.json')

    # ---- 控制台 ----
    for ctf, cmode, ctpsl, cname in CONFIGS:
        print(f"\n===== {cname} =====")
        print(f"{'月份':9}" + ''.join(f"{fn:>16}" for fn, _, _ in FILTERS))
        rows0 = out['monthly'][f'{cname}|base']
        for i, rm in enumerate(rows0):
            cells = []
            for fn, _, _ in FILTERS:
                r = out['monthly'][f'{cname}|{fn}'][i]
                cells.append(f"{r['mean']:+7.2f}%" if r.get('mean') is not None else '—')
            print(f"{rm['month']:9}" + ''.join(f"{x:>16}" for x in cells))
        print('— 汇总 —')
        for fn, _, _ in FILTERS:
            s = out['summary'][f'{cname}|{fn}']
            print(f"  {fn:8} 月均{s['avg']:+7.2f}%  净值{s['nav']:7.3f}x  亏损月{s['neg_months']}/{s['months']}"
                  f"  最差{s['worst']:+6.2f}%  下单{s['n_all']:6d}  胜率{s['win_rate']}%")
        for fn, _, _ in FILTERS:
            yr = out['summary'][f'{cname}|{fn}']['per_year']
            ytxt = '  '.join(f"{y}: {v['n']}笔/{v['win_rate']}%" for y, v in yr.items())
            print(f"  {fn:8} 年度: {ytxt}")


if __name__ == '__main__':
    main()
