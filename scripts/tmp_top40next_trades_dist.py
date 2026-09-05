# -*- coding: utf-8 -*-
"""前40新币三档配置：逐笔开仓时间分布分析（复用磁盘K线缓存，不重新拉数）
输出与前20时间分布同口径（北京时间小时分布、top3小时、美股时段占比），便于对比。
"""
import os, sys, io, time, warnings, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from divergence_signals import DIVERGENCE_VARIANTS, build_variant_signals

PROXY = "http://127.0.0.1:7892"
os.environ.setdefault("HTTP_PROXY", PROXY); os.environ.setdefault("HTTPS_PROXY", PROXY)

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


def load_cached(base, tf):
    p = os.path.join(CACHE_DIR, f'{base}_{tf}.pkl')
    if os.path.exists(p):
        try:
            return pd.read_pickle(p)
        except Exception:
            return None
    return None


def simulate_trades(df, signals, tp=None, sl=None, mode='long_only',
                    initial=INITIAL, comm=COMM):
    """同口径模拟，额外记录每笔开仓 (时间, 方向)"""
    idx = df.index.to_numpy(); close = df['close'].to_numpy(); sig = signals.to_numpy()
    cash = initial; units = 0.0; entry = 0.0; side = 0; n = 0
    trades = []
    for i in range(len(df)):
        price = close[i]
        if not np.isfinite(price) or price <= 0:
            continue
        s = int(sig[i]) if i > 0 else 0
        if side != 0 and entry > 0:
            r = (price - entry) / entry if side > 0 else (entry - price) / entry
            if (tp and r >= tp) or (sl and r <= -sl):
                cash = cash + units * price * (1 - comm) if side > 0 else \
                    cash + units * entry + (entry - price) * units - units * price * comm
                side = 0; units = 0; n += 1
                continue
        eq = cash + (units * price if side > 0 else units * entry + (entry - price) * units if side < 0 else 0)
        if eq <= 0:
            break
        if s == 1 and side <= 0:
            if side < 0:
                cash = cash + units * entry + (entry - price) * units - units * price * comm
                n += 1; side = 0; units = 0
            u = (cash * 0.95) / (price * (1 + comm))
            if u > 0:
                cash -= u * price * (1 + comm); units = u; entry = price; side = 1
                trades.append((pd.Timestamp(idx[i]).to_pydatetime(), '多'))
        elif s == -1 and side >= 0:
            if side > 0:
                cash += units * price * (1 - comm)
                n += 1; side = 0; units = 0
            if mode == 'long_short':
                u = (cash * 0.95) / price
                if u > 0:
                    cash -= u * price * (1 + comm); units = u; entry = price; side = -1
                    trades.append((pd.Timestamp(idx[i]).to_pydatetime(), '空'))
    return trades


def main():
    d = json.load(open(DETAIL, encoding='utf-8'))
    bases = sorted(d['results'].keys())
    print(f'新币 {len(bases)} 只: {bases}')
    stats = {c[3]: {'hour': Counter(), 'wd': Counter(), 'dir': Counter(), 'total': 0} for c in CONFIGS}

    for base in bases:
        for tf in sorted({c[0] for c in CONFIGS}):
            df = load_cached(base, tf)
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
                    try:
                        tr = simulate_trades(dw, sig.loc[dw.index], tp=tp, sl=sl, mode=cmode)
                    except Exception:
                        tr = []
                    st = stats[cname]
                    for t, side in tr:
                        # K线时间为UTC → 北京+8
                        bj = t.hour + 8
                        st['hour'][bj % 24] += 1
                        st['wd'][t.weekday()] += 1
                        st['dir'][side] += 1
                        st['total'] += 1
        print(f'  {base} 完成', flush=True)

    WD = ['一', '二', '三', '四', '五', '六', '日']
    for cname, st in stats.items():
        n = max(st['total'], 1)
        print('\n' + '=' * 96)
        print('【%s】 总开仓 %d 笔  多/空 = %d/%d' % (cname, st['total'], st['dir'].get('多', 0), st['dir'].get('空', 0)))
        print('  开仓小时分布（北京时间）:')
        cells = []
        for bj in range(24):
            p = st['hour'].get(bj, 0) / n * 100
            cells.append('%02d时 %5.1f%% %s' % (bj, p, '#' * int(p / 1.5)))
        for i in range(0, 24, 2):
            print('    ' + ' | '.join(cells[i:i + 2]))
        top3 = st['hour'].most_common(3)
        print('  最活跃3小时(北京): %s 合计 %.1f%%（均匀基线12.5%%）' % (
            ', '.join('%02d时 %.1f%%' % (h, c / n * 100) for h, c in top3),
            sum(c for _, c in top3) / n * 100))
        # 美股时段 北京21-04（即 UTC13-20）
        us_pct = sum(st['hour'].get(bj, 0) for bj in [21, 22, 23, 0, 1, 2, 3, 4]) / n * 100
        print('  美股交易时段(北京21-04)占比: %.1f%%' % us_pct)
        print('  星期分布: %s' % ' '.join('%s%.1f%%' % (WD[i], st['wd'].get(i, 0) / n * 100) for i in range(7)))


if __name__ == '__main__':
    main()
