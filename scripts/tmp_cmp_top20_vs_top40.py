# -*- coding: utf-8 -*-
"""对比：市值前40全部(本次) vs 之前只测前20 的 macd+背离+量能 三方案结果
并把本次明细分解为 前20 / 第21-40 两组，口径与 summary 相同。
"""
import io, sys, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np

TESTED_TOP20 = {'BTC','ETH','BNB','XRP','SOL','TRX','HYPE','ZEC','DOGE','XMR',
                'LINK','ADA','XLM','BCH','CC','LTC','UNI','GRAM','HBAR','AVAX'}
YEARS = ('2024', '2025', '2026YTD')

old = json.load(open('scripts/results/top20_macd_div_summary_2024_2026.json', encoding='utf-8'))
detail = json.load(open('scripts/results/top40all_macd_vol_detail_2024_2026.json',
                        encoding='utf-8'))['results']

# 旧前20：macd+背离+量能 的三档，按 (tf, mode中文, tpsl中文) 索引
old_cfgs = {}
for s in old.get('strategies', []):
    if s.get('name') != 'macd+背离+量能':
        continue
    for c in s['configs']:
        old_cfgs[(c['tf'], c['mode'], c['tpsl'])] = c

# 本次三方案
CONFIGS = [
    ('15m', 'long_short', 'tpsl5', '方案一 15m双向 tpsl5', '双向多空', '止盈止损各5%'),
    ('15m', 'long_only', 'none', '方案二 15m仅多 不设', '仅做多', '不设'),
    ('1h', 'long_short', 'sl5', '方案三 1h双向 sl5', '双向多空', '只止损5%'),
]

groups = {
    '前20(本次)': [b for b in detail if b in TESTED_TOP20],
    '第21-40(本次)': [b for b in detail if b not in TESTED_TOP20],
    '前40全部(本次)': list(detail.keys()),
}


def agg(bases, ctf, cname, yr):
    cells = [detail[b][ctf][yr][cname] for b in bases
             if ctf in detail.get(b, {}) and yr in detail[b][ctf]
             and detail[b][ctf][yr].get(cname)]
    if not cells:
        return None
    rets = [c['ret'] for c in cells]
    mdds = [c['mdd'] for c in cells if c.get('mdd') is not None]
    ns = [c['n'] for c in cells]
    return {'ret': float(np.mean(rets)), 'pos': sum(x > 0 for x in rets), 'cnt': len(rets),
            'mdd': float(np.mean(mdds)), 'mdd_max': float(min(mdds)), 'n': int(sum(ns)),
            'n_avg': float(np.mean(ns))}


for ctf, cmode, ctpsl, cname, omode, otpsl in CONFIGS:
    oc = old_cfgs.get((ctf, omode, otpsl))
    print(f"\n===== {cname} =====")
    hdr = f"{'':14}"
    for yr in YEARS:
        hdr += f"| {yr+' 收益':>12} {'MDD均/最差':>15} {'交易次数':>9} "
    print(hdr)
    rows = [('前20(旧测)', None)]
    # 旧测直接取 summary 字段
    for yr in YEARS:
        pass
    line = f"{'前20(旧测)':14}"
    for yr in YEARS:
        d = (oc or {}).get(yr)
        if d:
            line += f"| {d['ret']:+11.1f}% {d['mdd']:6.1f}/{d['mdd_max']:5.1f}% {d['n']:9d} "
        else:
            line += f"| {'—':>12} {'—':>15} {'—':>9} "
    print(line)
    for gname, bases in groups.items():
        line = f"{gname:14}"
        for yr in YEARS:
            a = agg(bases, ctf, cname, yr)
            if a:
                line += (f"| {a['ret']:+11.1f}% {a['mdd']:6.1f}/{a['mdd_max']:5.1f}% "
                         f"{a['n']:9d} ")
            else:
                line += f"| {'—':>12} {'—':>15} {'—':>9} "
        print(line)
    # 三年合计
    line_t = f"{'三年合计':14}"
    d = oc
    line_t += f"| 旧前20 {d['tot']:+.1f}%  MDD均{d.get('mdd_all_avg')}% 最差{d.get('mdd_all_max')}%  总交易{d.get('n_all')}" if d else ''
    print(line_t)
    for gname, bases in groups.items():
        tot, all_n, mdd_a, mdd_m = 1.0, 0, [], []
        for yr in YEARS:
            a = agg(bases, ctf, cname, yr)
            if a:
                tot *= 1 + a['ret'] / 100
                all_n += a['n']
                mdd_a.append(a['mdd']); mdd_m.append(a['mdd_max'])
        print(f"{'':14}| {gname} {(tot-1)*100:+.1f}%  MDD均{np.mean(mdd_a):.1f}% "
              f"最差{min(mdd_m):.1f}%  总交易{all_n}")

print('\n注：旧测前20数据截至约9/5，本次截至9/6；口径完全一致。')
print('旧测"前20"币数随年份为20/20/20；本次前20随年份 20→20→20，第21-40因上市时间 9→17→20。')
