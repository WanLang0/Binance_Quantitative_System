# -*- coding: utf-8 -*-
"""市值前40 × macd+背离+量能 × 三方案 × 逐月盈亏

方案一 15m 双向多空 止盈止损各5% (tpsl5)
方案二 15m 仅做多   不设止盈止损
方案三 1h  双向多空 只止损5% (sl5)

口径与 tmp_top40_all_macd_vol.py 完全一致（2023-10起预热、初始1万、95%仓位、
单边手续费0.1%、窗口末强平、止损按收盘价、做空现金背书），
仅把年度窗口换成逐月窗口（月末强平）。数据全部来自本地缓存，不联网。
输出：每月三方案跨币平均收益 + 逐月复利净值。
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
PERIODS = {'1h': 8766, '15m': 35040}
# 月窗口最小bar数（约半个月），过滤上市不满半月的币
MIN_BARS = {'15m': 1400, '1h': 350}
MONTHS = pd.period_range('2024-01', pd.Timestamp.utcnow(), freq='M')

BASES = json.load(open('scripts/results/top40all_macd_vol_summary_2024_2026.json',
                       encoding='utf-8'))['meta']['tickers']


def _close(cash, units, entry, price, side, comm=COMM):
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm


def simulate(df, signals, tp=None, sl=None, mode='long_only', ppy=8766,
             initial=INITIAL, comm=COMM):
    """与 tmp_top40_all_macd_vol.simulate 逐行一致（去掉sharpe计算）"""
    idx = df.index.to_numpy(); close = df['close'].to_numpy(); sig = signals.to_numpy()
    cash = initial; units = 0.0; entry = 0.0; side = 0; n = 0
    for i in range(len(df)):
        price = close[i]
        if not np.isfinite(price) or price <= 0:
            continue
        s = int(sig[i]) if i > 0 else 0
        if side != 0 and entry > 0:
            r = (price - entry) / entry if side > 0 else (entry - price) / entry
            if (tp and r >= tp) or (sl and r <= -sl):
                cash = _close(cash, units, entry, price, side); side = 0; units = 0; n += 1
                continue
        eq = cash + (units * price if side > 0 else units * entry + (entry - price) * units if side < 0 else 0)
        if eq <= 0:
            return None
        if s == 1 and side <= 0:
            if side < 0:
                cash = _close(cash, units, entry, price, side); n += 1; side = 0; units = 0
            u = (cash * 0.95) / (price * (1 + comm))
            if u > 0:
                cash -= u * price * (1 + comm); units = u; entry = price; side = 1
        elif s == -1 and side >= 0:
            if side > 0:
                cash = _close(cash, units, entry, price, side); n += 1; side = 0; units = 0
            if mode == 'long_short':
                u = (cash * 0.95) / price
                if u > 0:
                    cash -= u * price * (1 + comm); units = u; entry = price; side = -1
    if side != 0 and len(df) > 0:
        cash = _close(cash, units, entry, close[-1], side)
        n += 1
    if n == 0:
        return None
    return {'ret': (cash / initial - 1) * 100, 'n': n}


def main():
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
    # res[月标签][配置名] = {base: ret}
    res = {str(m): {c[3]: {} for c in CONFIGS} for m in MONTHS}
    for k, base in enumerate(BASES, 1):
        for tf in ('15m', '1h'):
            p = os.path.join(cache_dir, f'{base}_{tf}.pkl')
            if not os.path.exists(p):
                print(f'[{k}/{len(BASES)}] {base:8} {tf} 无缓存，跳过')
                continue
            df = pd.read_pickle(p)
            if df is None or len(df) < 200:
                continue
            try:
                _, sig = build_variant_signals(df, UD, UM, UV)
            except Exception as e:
                print(f'[{k}/{len(BASES)}] {base:8} {tf} 信号出错: {e}')
                continue
            for m in MONTHS:
                w0 = m.to_timestamp(how='start')
                w1 = (m + 1).to_timestamp(how='start')
                mask = (df.index >= w0.tz_localize('UTC')) & (df.index < w1.tz_localize('UTC'))
                dw = df[mask]
                if len(dw) < MIN_BARS[tf]:
                    continue
                for ctf, cmode, ctpsl, cname in CONFIGS:
                    if ctf != tf:
                        continue
                    tp, sl = TPSL_CFG[ctpsl]
                    try:
                        r = simulate(dw, sig.loc[dw.index], tp=tp, sl=sl, mode=cmode)
                    except Exception:
                        r = None
                    if r:
                        res[str(m)][cname][base] = r['ret']
        print(f'[{k}/{len(BASES)}] {base:8} done', flush=True)

    # ---- 汇总：每月跨币平均/中位/正收益数，逐月复利净值 ----
    table = []
    for m in MONTHS:
        key = str(m)
        row = {'month': key}
        for _, _, _, cname in CONFIGS:
            d = res[key][cname]
            if not d:
                row[cname] = None
                continue
            rets = list(d.values())
            row[cname] = {'mean': round(float(np.mean(rets)), 2),
                          'med': round(float(np.median(rets)), 2),
                          'pos': int(sum(x > 0 for x in rets)),
                          'cnt': len(rets)}
        table.append(row)

    out = 'scripts/results/top40all_macd_vol_monthly_2024_2026.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'variant': VARIANT, 'monthly': table,
                   'per_coin': {k: {c: v for c, v in r.items()} for k, r in res.items()}},
                  f, ensure_ascii=False, indent=1)
    print(f'\n月度明细 → {out}')

    print('\n== 每月盈亏（跨币平均，%） ==')
    hdr = f"{'月份':8}" + ''.join(f"{c[3]:>22}" for c in CONFIGS)
    print(hdr)
    nav = {c[3]: 1.0 for c in CONFIGS}
    for row in table:
        cells = []
        for c in CONFIGS:
            d = row[c[3]]
            if d:
                cells.append(f"{d['mean']:+8.2f} ({d['pos']:2}/{d['cnt']:2})")
                nav[c[3]] *= (1 + d['mean'] / 100)
            else:
                cells.append(f"{'—':>15}")
        print(f"{row['month']:8}" + ''.join(f"{x:>22}" for x in cells))
    print('\n== 逐月复利净值（1 + 月均收益累乘） ==')
    for c in CONFIGS:
        print(f"{c[3]:24} {nav[c[3]]:.3f}x")


if __name__ == '__main__':
    main()
