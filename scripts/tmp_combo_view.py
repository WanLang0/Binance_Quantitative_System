# -*- coding: utf-8 -*-
"""组合平滑测试：资金对半分两个方案，月收益=两方案跨币平均收益的均值
数据源：top30_macd_vol_monthly_filter.json 的 base 档（30币、原版三方案）
组合A = 方案一 + 方案二（用户指定）
组合B = 方案一 + 方案三（此前验证过亏损月完全错开）
"""
import json
import numpy as np

d = json.load(open('scripts/results/top30_macd_vol_monthly_filter.json', encoding='utf-8'))
M = d['monthly']
C1 = '方案一 15m双向 tpsl5|base'
C2 = '方案二 15m仅多 不设|base'
C3 = '方案三 1h双向 sl5|base'
S = d['summary']

def combo(keys):
    """按月取各方案mean的平均；返回逐月列表与汇总"""
    rows = []
    months = [r['month'] for r in M[keys[0]]]
    for m in months:
        vals = []
        for k in keys:
            r = next(x for x in M[k] if x['month'] == m)
            vals.append(r.get('mean'))
        if any(v is None for v in vals):
            rows.append((m, None))
        else:
            rows.append((m, sum(vals) / len(vals)))
    vals = [v for _, v in rows if v is not None]
    nav = 1.0
    for v in vals:
        nav *= (1 + v / 100)
    n_all = sum(S[k]['n_all'] for k in keys)
    w = sum(S[k]['win_rate'] / 100 * S[k]['n_all'] for k in keys)
    return rows, {'avg': np.mean(vals), 'nav': nav,
                  'neg': sum(1 for v in vals if v < 0), 'worst': min(vals),
                  'best': max(vals), 'n': n_all, 'wr': w / n_all * 100, 'months': len(vals)}

print(f"{'月份':9}{'一+二 对半':>12}{'一+三 对半':>12}   (方案一    方案二    方案三)")
rowsA, sA = combo([C1, C2])
rowsB, sB = combo([C1, C3])
for (mA, vA), (mB, vB) in zip(rowsA, rowsB):
    r1 = next((x.get('mean') for x in M[C1] if x['month'] == mA), None)
    r2 = next((x.get('mean') for x in M[C2] if x['month'] == mA), None)
    r3 = next((x.get('mean') for x in M[C3] if x['month'] == mA), None)
    fa = f"{vA:+7.2f}%" if vA is not None else '     —'
    fb = f"{vB:+7.2f}%" if vB is not None else '     —'
    f1 = f"{r1:+7.2f}%" if r1 is not None else '     —'
    f2 = f"{r2:+7.2f}%" if r2 is not None else '     —'
    f3 = f"{r3:+7.2f}%" if r3 is not None else '     —'
    print(f"{mA:9}{fa:>12}{fb:>12}   ({f1} {f2} {f3})")

print('\n===== 汇总（32个完整月，30币） =====')
for name, s in [('方案一 单跑', S[C1]), ('方案二 单跑', S[C2]), ('方案三 单跑', S[C3]),
                ('组合 一+二 对半', sA), ('组合 一+三 对半', sB)]:
    if 'neg' in s:
        print(f"{name:14} 月均{s['avg']:+7.2f}%  净值{s['nav']:7.3f}x  亏损月{s['neg']:2}/{s['months']}"
              f"  最差{s['worst']:+6.2f}%  最好{s['best']:+6.2f}%  下单{s['n']:6d}  胜率{s['wr']:.1f}%")
    else:
        print(f"{name:14} 月均{s['avg']:+7.2f}%  净值{s['nav']:7.3f}x  亏损月{s['neg_months']:2}/{s['months']}"
              f"  最差{s['worst']:+6.2f}%  最好{s['best']:+6.2f}%  下单{s['n_all']:6d}  胜率{s['win_rate']}%")

# 年度
print('\n===== 年度收益（各月均值加总，%） =====')
for y in ('2024', '2025', '2026'):
    vA = [v for m, v in rowsA if v is not None and m.startswith(y)]
    vB = [v for m, v in rowsB if v is not None and m.startswith(y)]
    print(f"{y}: 一+二 {sum(vA):+7.1f}%   一+三 {sum(vB):+7.1f}%")
