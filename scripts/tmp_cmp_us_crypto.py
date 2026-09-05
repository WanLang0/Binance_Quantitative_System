# -*- coding: utf-8 -*-
"""对比美股 vs 虚拟币 MACD三变体回测矩阵（同口径，虚拟币取1h组）"""
import json
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
us = json.load(open(os.path.join(R, 'us_macd_div_summary_2024_2026.json'), encoding='utf-8'))
cr = json.load(open(os.path.join(R, 'top20_macd_div_summary_2024_2026.json'), encoding='utf-8'))

YRS = ['2024', '2025', '2026YTD']


def cell(d):
    if not d or d.get('ret') is None:
        return '—'
    return '%+.1f' % d['ret']


def cfg_map(m, tf=None):
    out = {}
    for s in m['strategies']:
        for c in s['configs']:
            if tf and c['tf'] != tf:
                continue
            out[(s['name'], c['mode'], c['tpsl'])] = c
    return out


CU = cfg_map(us)
CC = cfg_map(cr, tf='1h')

pairs = []
for mode in ['仅做多', '双向多空']:
    for tpsl in ['不设', '只止损5%', '止盈止损各5%']:
        pairs.append((mode, tpsl))

print('=' * 110)
print('BH 基准（跨标的均值）')
print('  美股   :', {y: us['meta']['bh']['1h'].get(y) for y in YRS})
print('  虚拟币 :', {y: cr['meta']['bh']['1h'].get(y) for y in YRS})
print()

for v in ['macd+背离', 'MACD+背离']:
    pass

variants_us = [s['name'] for s in us['strategies']]
variants_cr = [s['name'] for s in cr['strategies']]
print('变体映射：美股 %s ↔ 虚拟币 %s' % (variants_us, variants_cr))
vmap = dict(zip(variants_us, variants_cr))

for mode, tpsl in pairs:
    print('=' * 110)
    print('【%s | %s】（均为 1h）' % (mode, tpsl))
    print('%-22s %-6s %8s %8s %8s %9s %7s %8s %8s %8s %8s' % (
        '变体', '市场', '2024', '2025', '2026YTD', '三年合计', '总夏普', '均回撤', '最深回撤', '年下单数', '跑赢BH/年'))
    for vu in variants_us:
        vc = vmap[vu]
        for label, mm, vv in (('美股', CU, vu), ('虚拟币', CC, vc)):
            c = mm.get((vv, mode, tpsl))
            if not c:
                continue
            n_yr = 0
            beat_sum = 0
            cnt_sum = 0
            for y in YRS:
                d = c.get(y)
                if d and d.get('beat') is not None:
                    beat_sum += d['beat']
                    cnt_sum += d['cnt']
                    n_yr += 1
            n_avg = (c.get('n_all', 0) / 3.0) if c.get('n_all') else 0
            print('%-22s %-6s %7s%% %7s%% %7s%% %8s%% %7s %7s%% %7s%% %8.0f %6d/%d' % (
                vv, label,
                cell(c.get('2024')), cell(c.get('2025')), cell(c.get('2026YTD')),
                '%+.1f' % c['tot'] if c.get('tot') is not None else '—',
                '%+.2f' % c['sh_all'] if c.get('sh_all') is not None else '—',
                '%.1f' % c['mdd_all_avg'] if c.get('mdd_all_avg') is not None else '—',
                '%.1f' % c['mdd_all_max'] if c.get('mdd_all_max') is not None else '—',
                n_avg, beat_sum, cnt_sum))
    print()

# 汇总：两市场各自「最优配置」
print('=' * 110)
print('【各市场最优配置（按三年合计最高）】')


def best(mm):
    rows = []
    for k, c in mm.items():
        if c.get('tot') is not None:
            rows.append((c['tot'], k, c))
    rows.sort(reverse=True)
    return rows[:3]


for label, mm in (('美股', CU), ('虚拟币(1h)', CC)):
    print(' ', label)
    for tot, (v, mode, tpsl), c in best(mm):
        print('    %-22s %-6s %-8s 三年合计%+.1f%% 夏普%s 均回撤%s%%' % (
            v, mode, tpsl, tot,
            '%+.2f' % c['sh_all'] if c.get('sh_all') is not None else '—',
            '%.1f' % c['mdd_all_avg'] if c.get('mdd_all_avg') is not None else '—'))
