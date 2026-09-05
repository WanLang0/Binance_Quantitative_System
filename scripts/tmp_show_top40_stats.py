# -*- coding: utf-8 -*-
"""展示前40扩展验证 summary 的回撤/下单数明细"""
import json
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
d = json.load(open(os.path.join(R, 'top40next_macd_vol_summary_2024_2026.json'), encoding='utf-8'))
s = d['strategies'][0]

print('【%s】（市值21-40新币）' % d['meta']['title'])
print()
print('%-10s %-8s %-12s | %s' % ('周期', '方向', '止盈止损', '年度: 收益 / 均回撤 / 最大回撤 / 下单数 / 下单数每币 / 夏普'))
print('-' * 110)
for c in s['configs']:
    tag = '%s %s %s' % (c['tf'], c['mode'], c['tpsl'])
    for y in ('2024', '2025', '2026YTD'):
        dd = c.get(y)
        if not dd:
            print('%-32s %s: 无数据' % (tag, y))
            continue
        print('%-32s %s: %+7.1f%%  均回撤%6.1f%%  最大回撤%7.1f%%  下单%5d(每币%5.1f)  夏普%+.2f  正收益%d/%d 跑赢BH%d' % (
            tag, y, dd['ret'], dd['mdd'], dd['mdd_max'], dd['n'], dd['n_avg'], dd['sh'],
            dd['pos'], dd['cnt'], dd['beat']))
    print('%-32s 三年合计%+.1f%%  三年下单%d  三年均回撤%.1f%%  三年最深%.1f%%  总夏普%+.2f' % (
        '', c['tot'], c['n_all'], c['mdd_all_avg'], c['mdd_all_max'], c['sh_all']))
    print()
