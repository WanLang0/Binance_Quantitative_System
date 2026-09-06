# -*- coding: utf-8 -*-
"""市值前30 × macd+背离+量能 × 三方案 × 牛熊分段（BTC 200日均线牛熊分界）

牛熊定义：BTC 日线收盘 > SMA200 为牛市段，< 为熊市段（2024-01-01 起）。
短于21天的段并入前一段。每段独立模拟（段末强平），口径与年度回测一致。
输出：各方案在牛市段/熊市段的 收益/MDD/交易次数。
"""
import os, sys, io, time, warnings, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

PROXY = "http://127.0.0.1:7892"
os.environ.setdefault("HTTP_PROXY", PROXY); os.environ.setdefault("HTTPS_PROXY", PROXY)

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone

from divergence_signals import DIVERGENCE_VARIANTS, build_variant_signals
from tmp_top40_all_macd_vol import (COMM, INITIAL, YEAR_BARS_MIN, TPSL_CFG, PERIODS,
                                    fetch_klines, simulate)

VARIANT = 'macd+背离+量能'
UD, UM, UV = DIVERGENCE_VARIANTS[VARIANT]
CONFIGS = [
    ('15m', 'long_short', 'tpsl5', '方案一 15m双向 tpsl5'),
    ('15m', 'long_only', 'none', '方案二 15m仅多 不设'),
    ('1h', 'long_short', 'sl5', '方案三 1h双向 sl5'),
]
ANALYSIS_START = pd.Timestamp('2024-01-01', tz='UTC')
MIN_SEG_DAYS = 21

summary = json.load(open('scripts/results/top40all_macd_vol_summary_2024_2026.json', encoding='utf-8'))
detail_all = json.load(open('scripts/results/top40all_macd_vol_detail_2024_2026.json', encoding='utf-8'))['results']
top30 = [t for t in summary['meta']['tickers'] if t in detail_all][:30]
print(f'前30名单: {top30}')


def get_bull_bear_segments():
    btc = fetch_klines('BTC', '1d', start='2022-07-01')
    sma200 = btc['close'].rolling(200).mean()
    state = (btc['close'] > sma200).dropna()
    state = state[state.index >= ANALYSIS_START - pd.Timedelta(days=10)]
    # 连续区间
    raw = []
    cur_s, cur_v = None, None
    for d, v in state.items():
        if cur_v is None or v != cur_v:
            if cur_v is not None:
                raw.append((cur_s, d, cur_v))
            cur_s, cur_v = d, v
    raw.append((cur_s, state.index[-1] + pd.Timedelta(days=1), cur_v))
    # 合并 <MIN_SEG_DAYS 的段到前一段
    segs = []
    for s, e, v in raw:
        if segs and (e - s) < pd.Timedelta(days=MIN_SEG_DAYS):
            segs[-1] = (segs[-1][0], e, segs[-1][2])
        else:
            segs.append((s, e, v))
    segs[0] = (max(segs[0][0], ANALYSIS_START), segs[0][1], segs[0][2])
    return segs, btc, sma200


segs, btc, sma200 = get_bull_bear_segments()
print(f'\n== 牛熊分段（BTC 收盘 vs SMA200，{ANALYSIS_START.date()} 起，短段<{MIN_SEG_DAYS}天并入前段） ==')
for s, e, v in segs:
    n = (e - s).days
    print(f"  {'🐂牛' if v else '🐻熊'}  {s.date()} → {e.date()}  {n:4d}天")

# ---- 模拟：每币每tf算一次全量信号，再按段切片 ----
# res[cname][regime] = {base: {'tot':%, 'mdd':%, 'n':int, 'segs':[(ret,mdd,n)]}}
res = {c[3]: {'bull': {}, 'bear': {}} for c in CONFIGS}
for k, base in enumerate(top30, 1):
    for tf in sorted({c[0] for c in CONFIGS}):
        df = fetch_klines(base, tf)
        if df is None or len(df) < YEAR_BARS_MIN:
            continue
        try:
            _, sig = build_variant_signals(df, UD, UM, UV)
        except Exception:
            continue
        for ctf, cmode, ctpsl, cname in CONFIGS:
            if ctf != tf:
                continue
            tp, sl = TPSL_CFG[ctpsl]
            for s, e, v in segs:
                dw = df[(df.index >= s) & (df.index < e)]
                if len(dw) < YEAR_BARS_MIN:
                    continue
                try:
                    r = simulate(dw, sig.loc[dw.index], tp=tp, sl=sl, mode=cmode, ppy=PERIODS[tf])
                except Exception:
                    r = None
                if r:
                    res[cname]['bull' if v else 'bear'].setdefault(base, []).append(r)
    print(f'[{k}/{len(top30)}] {base} done', flush=True)

print('\n== 牛熊分段结果（前30 × macd+背离+量能） ==')
out_cfgs = []
for cname in res:
    print(f'\n【{cname}】')
    cfg_out = {'name': cname}
    for reg, rname in (('bull', '牛市段'), ('bear', '熊市段')):
        per = res[cname][reg]
        if not per:
            print(f'  {rname}: 无'); continue
        tots, mdds, ns, cnt = [], [], [], 0
        for b, lst in per.items():
            t = 1.0
            for r in lst:
                t *= 1 + r['ret'] / 100
            tots.append((t - 1) * 100)
            mdds.append(min(r['mdd'] for r in lst))
            ns.append(sum(r['n'] for r in lst))
        d = {'ret': round(float(np.mean(tots)), 1), 'med': round(float(np.median(tots)), 1),
             'pos': int(sum(x > 0 for x in tots)), 'cnt': len(tots),
             'mdd': round(float(np.mean(mdds)), 1), 'mdd_max': round(float(min(mdds)), 1),
             'n': int(sum(ns)), 'n_avg': round(float(np.mean(ns)), 1)}
        print(f"  {rname}  均收益{d['ret']:+8.1f}%  中位{d['med']:+8.1f}%  正收益{d['pos']:2}/{d['cnt']:2}"
              f"  | 平均MDD {d['mdd']:6.1f}%  最差 {d['mdd_max']:6.1f}%"
              f"  | 交易 总{d['n']:6d} 均{d['n_avg']:6.1f}")
        cfg_out[reg] = d
    out_cfgs.append(cfg_out)

with open('scripts/results/top30_bullbear_summary.json', 'w', encoding='utf-8') as f:
    json.dump({'meta': {'updated': datetime.now().strftime('%Y-%m-%d'),
                        'regime': 'BTC日线收盘 vs SMA200 牛熊分界，2024-01起，短段并入前段',
                        'segments': [[str(s.date()), str(e.date()), bool(v)] for s, e, v in segs],
                        'tickers': top30},
               'strategies': [{'name': VARIANT, 'configs': out_cfgs}]}, f,
              ensure_ascii=False, indent=1, default=float)
print('\nsummary → scripts/results/top30_bullbear_summary.json')
