# -*- coding: utf-8 -*-
"""前40新币三档配置单笔胜率统计 v2（修复：所有平仓路径都记录，含手续费净收益率）"""
import os, sys, io, json, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from divergence_signals import DIVERGENCE_VARIANTS, build_variant_signals

COMM = 0.001
INITIAL = 10000.0
YEAR_BARS_MIN = 200
WINDOWS = [
    ('2024', '2024-01-01', '2025-01-01'),
    ('2025', '2025-01-01', '2026-01-01'),
    ('2026YTD', '2026-01-01', datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')),
]
VARIANT = 'macd+背离+量能'
UD, UM, UV = DIVERGENCE_VARIANTS[VARIANT]
CONFIGS = [
    ('15m', 'long_only', 'none', '15m 仅做多 不设'),
    ('15m', 'long_short', 'tpsl5', '15m 双向多空 止盈止损5%'),
    ('1h', 'long_short', 'sl5', '1h 双向多空 只止损5%'),
]
TPSL_CFG = {'none': (None, None), 'sl5': (None, 0.05), 'tpsl5': (0.05, 0.05)}

HERE = os.path.dirname(os.path.abspath(__file__))
DETAIL = os.path.join(HERE, 'results', 'top40next_macd_vol_detail_2024_2026.json')
CACHE_DIR = os.path.join(HERE, 'cache')


def simulate_trades(df, signals, tp=None, sl=None, mode='long_only',
                    initial=INITIAL, comm=COMM):
    """同口径模拟；所有平仓路径（止盈止损/反手/年末强平）都记录净收益率(含手续费)"""
    idx = df.index.to_numpy(); close = df['close'].to_numpy(); sig = signals.to_numpy()
    cash = initial; units = 0.0; entry = 0.0; side = 0
    cost = 0.0                       # 开仓成本（含手续费）
    trades = []

    def close_pos(price):
        """平仓：返回 (净收益率%, 方向标签)；非平仓收益按现金背书口径"""
        nonlocal cash, units, side
        if side > 0:
            proceeds = units * price * (1 - comm)
        else:
            proceeds = units * entry + (entry - price) * units - units * price * comm
        ret = (proceeds - cost) / cost * 100
        d = '多' if side > 0 else '空'
        cash += proceeds
        units = 0.0; side = 0
        return ret, d

    for i in range(len(df)):
        price = close[i]
        if not np.isfinite(price) or price <= 0:
            continue
        s = int(sig[i]) if i > 0 else 0
        # 1) 止盈止损
        if side != 0 and entry > 0:
            r = (price - entry) / entry if side > 0 else (entry - price) / entry
            if (tp and r >= tp) or (sl and r <= -sl):
                ret, d = close_pos(price)
                trades.append((ret, d))
                continue
        # 2) 权益检查
        eq = cash + (units * price if side > 0 else units * entry + (entry - price) * units if side < 0 else 0)
        if eq <= 0:
            break
        # 3) 信号
        if s == 1 and side <= 0:
            if side < 0:
                ret, d = close_pos(price)
                trades.append((ret, d))
            u = (cash * 0.95) / (price * (1 + comm))
            if u > 0:
                cash -= u * price * (1 + comm)
                cost = u * price * (1 + comm)
                units = u; entry = price; side = 1
        elif s == -1 and side >= 0:
            if side > 0:
                ret, d = close_pos(price)
                trades.append((ret, d))
            if mode == 'long_short':
                u = (cash * 0.95) / price
                if u > 0:
                    cash -= u * price * (1 + comm)
                    cost = u * price * (1 + comm)
                    units = u; entry = price; side = -1
    # 年末强平
    if side != 0 and len(df) > 0:
        ret, d = close_pos(close[-1])
        trades.append((ret, d))
    return trades


def main():
    d = json.load(open(DETAIL, encoding='utf-8'))
    bases = sorted(d['results'].keys())
    stats = {c[3]: {'rets': [], 'year': {}, 'dir': Counter(), 'dir_win': Counter()} for c in CONFIGS}

    for base in bases:
        for tf in sorted({c[0] for c in CONFIGS}):
            p = os.path.join(CACHE_DIR, f'{base}_{tf}.pkl')
            if not os.path.exists(p):
                continue
            try:
                df = pd.read_pickle(p)
            except Exception:
                continue
            if df is None or len(df) < YEAR_BARS_MIN:
                continue
            try:
                _, sig = build_variant_signals(df, UD, UM, UV)
            except Exception:
                continue
            for wname, w0, w1 in WINDOWS:
                m = (df.index >= pd.Timestamp(w0, tz='UTC')) & (df.index < pd.Timestamp(w1, tz='UTC'))
                dw = df[m]
                if len(dw) < YEAR_BARS_MIN:
                    continue
                for ctf, cmode, ctpsl, cname in CONFIGS:
                    if ctf != tf:
                        continue
                    tp, sl = TPSL_CFG[ctpsl]
                    tr = simulate_trades(dw, sig.loc[dw.index], tp=tp, sl=sl, mode=cmode)
                    st = stats[cname]
                    st['rets'].extend(tr)
                    py = st['year'].setdefault(wname, [0, 0])
                    py[0] += len(tr)
                    py[1] += sum(1 for r, _ in tr if r > 0)
                    for r, dd in tr:
                        st['dir'][dd] += 1
                        if r > 0:
                            st['dir_win'][dd] += 1
        print(f'  {base} 完成', flush=True)

    print('\n========== 前40（市值21-40新币） ==========')
    for cname, st in stats.items():
        rets = [r for r, _ in st['rets']]
        n = len(rets)
        if n == 0:
            continue
        wins = [r for r in rets if r > 0]
        losses = [r for r in rets if r <= 0]
        aw = np.mean(wins) if wins else 0
        al = np.mean(losses) if losses else 0
        print('\n【%s】 总%d笔' % (cname, n))
        print('  总胜率: %d/%d = %.1f%%' % (len(wins), n, len(wins) / n * 100))
        print('  均盈利单 %+.2f%% / 均亏损单 %+.2f%% / 盈亏比 %.2f' % (aw, al, (aw / abs(al)) if al else float('inf')))
        print('  最大单笔盈利 %+.1f%% / 最大单笔亏损 %.1f%%' % (max(wins) if wins else 0, min(losses) if losses else 0))
        print('  分年胜率:', '  '.join('%s:%.1f%%(%d/%d)' % (y, w[1] / w[0] * 100, w[1], w[0])
                                       for y, w in sorted(st['year'].items())))
        print('  分方向胜率:', '  '.join('%s:%.1f%%(%d/%d)' % (dd, st['dir_win'][dd] / st['dir'][dd] * 100,
                                                        st['dir_win'][dd], st['dir'][dd]) for dd in st['dir']))


if __name__ == '__main__':
    main()
