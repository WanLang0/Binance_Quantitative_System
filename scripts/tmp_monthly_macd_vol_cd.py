# -*- coding: utf-8 -*-
"""市值前40 × macd+背离+量能 × 三方案 × 止损后冷却期对比（月度口径）

在 tmp_monthly_macd_vol.py 基础上增加：
- 止损平仓后 N 根K线冷却：冷却期内信号反转可平仓、但不开新仓
- 新增统计：每月下单数、胜率（每笔平仓盈亏>0 记胜）
- 对比档位：cd0(原版) / cd8(15m=2h,1h=8h) / cd16(15m=4h,1h=16h)

口径不变：2023-10预热、初始1万、95%仓位、单边手续费0.1%、
月末强平、止盈止损按收盘价、做空现金背书、本地缓存不联网。
方案二(15m仅多不设)无止损，冷却不影响，仅跑cd0作sanity check。
"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

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
COOLDOWNS = [0, 8, 16]   # 止损后冷却K线根数
MONTHS = pd.period_range('2024-01', pd.Timestamp.utcnow(), freq='M')
YEARS = ('2024', '2025', '2026')

BASES = json.load(open('scripts/results/top40all_macd_vol_summary_2024_2026.json',
                       encoding='utf-8'))['meta']['tickers']


def _close(cash, units, entry, price, side, comm=COMM):
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm


def simulate(df, signals, tp=None, sl=None, mode='long_only', cooldown=0,
             initial=INITIAL, comm=COMM):
    """月度窗口回测，带止损冷却与胜率统计。返回 dict 或 None(无交易/爆仓)"""
    close = df['close'].to_numpy(); sig = signals.to_numpy()
    cash = initial; units = 0.0; entry = 0.0; side = 0
    n = wins = losses = 0
    cool_until = -1
    entry_cash = initial   # 开仓前现金快照（胜负=平仓回款是否超过它）

    def _flat(price):
        """平仓并记胜负（平仓后现金 > 开仓前现金 记胜）"""
        nonlocal cash, units, side, n, wins, losses
        c2 = _close(cash, units, entry, price, side)
        if c2 > entry_cash:
            wins += 1
        else:
            losses += 1
        cash = c2; units = 0.0; side = 0; n += 1

    for i in range(len(df)):
        price = close[i]
        if not np.isfinite(price) or price <= 0:
            continue
        s = int(sig[i]) if i > 0 else 0
        # 止盈/止损（按收盘价）
        if side != 0 and entry > 0:
            r = (price - entry) / entry if side > 0 else (entry - price) / entry
            hit_tp = tp and r >= tp
            hit_sl = sl and r <= -sl
            if hit_tp or hit_sl:
                _flat(price)
                if hit_sl and not hit_tp:      # 仅止损触发才进入冷却
                    cool_until = i + cooldown
                continue
        eq = cash + (units * price if side > 0 else units * entry + (entry - price) * units if side < 0 else 0)
        if eq <= 0:
            return None
        # 开仓（冷却期内禁止；持仓遇反向信号仍可平仓）
        if s == 1 and side <= 0:
            if side < 0:
                _flat(price)                    # 信号反转平空
            if i >= cool_until:                 # 冷却期内不起新仓
                u = (cash * 0.95) / (price * (1 + comm))
                if u > 0:
                    entry_cash = cash
                    cash -= u * price * (1 + comm); units = u; entry = price; side = 1
        elif s == -1 and side >= 0:
            if side > 0:
                _flat(price)                    # 信号反转平多
            if mode == 'long_short' and i >= cool_until:
                u = (cash * 0.95) / price
                if u > 0:
                    entry_cash = cash
                    cash -= u * price * (1 + comm); units = u; entry = price; side = -1
    if side != 0 and len(df) > 0:
        _flat(close[-1])                        # 月末强平
    if n == 0:
        return None
    return {'ret': (cash / initial - 1) * 100, 'n': n,
            'wins': wins, 'losses': losses}


def main():
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
    # res[月][配置名][cd档] = {base: {'ret','n','wins','losses'}}
    res = {str(m): {c[3]: {cd: {} for cd in COOLDOWNS} for c in CONFIGS} for m in MONTHS}
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
            for m in MONTHS:
                w0 = m.to_timestamp(how='start').tz_localize('UTC')
                w1 = (m + 1).to_timestamp(how='start').tz_localize('UTC')
                dw = df[(df.index >= w0) & (df.index < w1)]
                if len(dw) < MIN_BARS[tf]:
                    continue
                for ctf, cmode, ctpsl, cname in CONFIGS:
                    if ctf != tf:
                        continue
                    tp, sl = TPSL_CFG[ctpsl]
                    cds = (0,) if sl is None else COOLDOWNS   # 无止损方案不受冷却影响
                    for cd in cds:
                        try:
                            r = simulate(dw, sig.loc[dw.index], tp=tp, sl=sl,
                                         mode=cmode, cooldown=cd)
                        except Exception:
                            r = None
                        if r:
                            res[str(m)][cname][cd][base] = r
        print(f'[{k}/{len(BASES)}] {base:8} done', flush=True)

    # ---- 汇总 ----
    out = {'variant': VARIANT, 'cooldowns': COOLDOWNS, 'monthly': {}, 'summary': {}}
    for ctf, cmode, ctpsl, cname in CONFIGS:
        _, sl_cfg = TPSL_CFG[ctpsl]
        cds = (0,) if sl_cfg is None else COOLDOWNS   # 无止损方案只跑cd0
        for cd in cds:
            rows = []
            for m in MONTHS:
                d = res[str(m)][cname][cd]
                if not d:
                    rows.append({'month': str(m)})
                    continue
                rets = [v['ret'] for v in d.values()]
                ns = sum(v['n'] for v in d.values())
                ws = sum(v['wins'] for v in d.values())
                ls = sum(v['losses'] for v in d.values())
                rows.append({'month': str(m),
                             'mean': round(float(np.mean(rets)), 2),
                             'med': round(float(np.median(rets)), 2),
                             'pos': int(sum(x > 0 for x in rets)), 'cnt': len(rets),
                             'n': int(ns), 'wins': int(ws), 'losses': int(ls)})
            out['monthly'][f'{cname}|cd{cd}'] = rows
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
                yr[y] = {'months': len(yv), 'ret_sum': round(float(np.sum(yv)), 1) if yv else None,
                         'n': int(yn), 'win_rate': round(yw / (yw + yl) * 100, 1) if (yw + yl) else None}
            out['summary'][f'{cname}|cd{cd}'] = {
                'months': len(vals), 'avg': round(float(np.mean(vals)), 2),
                'med': round(float(np.median(vals)), 2),
                'neg_months': int(sum(v < 0 for v in vals)),
                'worst': round(float(min(vals)), 2), 'best': round(float(max(vals)), 2),
                'nav': round(nav, 3), 'n_all': int(tot_n),
                'win_rate': round(tot_w / (tot_w + tot_l) * 100, 1) if (tot_w + tot_l) else None,
                'per_year': yr,
            }
    with open('scripts/results/top40all_macd_vol_monthly_cooldown.json', 'w', encoding='utf-8') as f:
        json.dump({'summary': out['summary'], 'monthly': out['monthly'],
                   'per_coin': {k2: {c: v for c, v in r.items()} for k2, r in res.items()}},
                  f, ensure_ascii=False, default=float)
    print('明细 → scripts/results/top40all_macd_vol_monthly_cooldown.json')

    # ---- 控制台输出 ----
    for ctf, cmode, ctpsl, cname in CONFIGS:
        _, sl_cfg = TPSL_CFG[ctpsl]
        cds = (0,) if sl_cfg is None else COOLDOWNS
        print(f"\n===== {cname} =====")
        print(f"{'月份':9}" + ''.join(f"{'cd'+str(cd)+' 月收益/下单/胜率':>26}" for cd in cds))
        rows0 = out['monthly'][f'{cname}|cd{cds[0]}']
        for i, rm in enumerate(rows0):
            cells = []
            for cd in cds:
                r = out['monthly'][f'{cname}|cd{cd}'][i]
                if r.get('mean') is None:
                    cells.append('—')
                else:
                    wr = r['wins'] / (r['wins'] + r['losses']) * 100 if (r['wins'] + r['losses']) else 0
                    cells.append(f"{r['mean']:+7.2f}% {r['n']:4d}笔 {wr:4.1f}%")
            print(f"{rm['month']:9}" + ''.join(f"{x:>26}" for x in cells))
        print(f"{'— 汇总 —':9}")
        for cd in cds:
            s = out['summary'][f'{cname}|cd{cd}']
            print(f"  cd{cd:<3} 算术月均{s['avg']:+7.2f}%  复利净值{s['nav']:7.3f}x  "
                  f"亏损月{s['neg_months']:2}/{s['months']}  最差{s['worst']:+6.2f}%  "
                  f"总下单{s['n_all']:6d}  胜率{s['win_rate']}%")


if __name__ == '__main__':
    main()
