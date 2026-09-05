# -*- coding: utf-8 -*-
"""生成「实盘推荐配置」JSON（虚拟币×美股 各三档），供最优策略页表格展示
数据来源：两个全配置矩阵 summary（不重跑回测，仅抽取指定配置）
"""
import json
import os
from datetime import datetime

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
OUT = os.path.join(R, 'recommended_configs.json')
YRS = ['2024', '2025', '2026YTD']

cr = json.load(open(os.path.join(R, 'top20_macd_div_summary_2024_2026.json'), encoding='utf-8'))
us = json.load(open(os.path.join(R, 'us_macd_div_summary_2024_2026.json'), encoding='utf-8'))


def pick(matrix, variant, tf=None, mode=None, tpsl=None):
    """从矩阵中按 (变体, 周期, 方向, 止盈止损) 找配置（tpsl 兼容别名）"""
    alias = {'止盈止损各5%': '止盈止损5%', '止盈止损5%': '止盈止损各5%'}
    for s in matrix['strategies']:
        if s['name'] != variant:
            continue
        for c in s['configs']:
            if tf is not None and c['tf'] != tf:
                continue
            if mode is not None and c['mode'] != mode:
                continue
            if tpsl is not None and c['tpsl'] not in (tpsl, alias.get(tpsl)):
                continue
            return c
    return None


def cell(c, y):
    d = (c or {}).get(y)
    if not d or d.get('ret') is None:
        return None
    return d


def mk(risk, label, matrix, variant, tf, mode, tpsl, note, bh):
    c = pick(matrix, variant, tf, mode, tpsl)
    assert c is not None, f'找不到配置: {variant}/{tf}/{mode}/{tpsl}'
    return {
        'risk': risk, 'label': label, 'variant': variant,
        'tf': tf or c['tf'], 'mode': mode or c['mode'], 'tpsl': tpsl or c['tpsl'],
        'years': {y: cell(c, y) for y in YRS},
        'tot': c.get('tot'), 'n_all': c.get('n_all'),
        'mdd_all_avg': c.get('mdd_all_avg'), 'mdd_all_max': c.get('mdd_all_max'),
        'sh_all': c.get('sh_all'), 'note': note, 'bh': bh,
    }


data = {
    'meta': {
        'updated': datetime.now().strftime('%Y-%m-%d'),
        'title': '⭐ 实盘推荐配置（风险低→高）',
        'desc': ('从「市值前20合约币 × 三种MACD组合策略」「美股代币 × 三种MACD组合策略」全配置回测矩阵中，'
                 '按风险档各选三组实盘推荐配置；口径与矩阵一致（初始1万USDT、95%仓位、单边手续费0.1%、'
                 '分年强平、止损按收盘价、做空为现金背书模拟）。'),
    },
    'crypto': [
        mk('低', '稳健·回撤最浅', cr, 'macd+背离+量能', '15m', '仅做多', '不设',
           '无杠杆/1x 首选；不做空、不设止盈止损，三年最深回撤-32.6%，任何年度均正收益。',
           cr['meta']['bh']['15m']),
        mk('中', '高频·多空双向', cr, 'macd+背离+量能', '15m', '双向多空', '止盈止损5%',
           '年均8000+笔高频双向，止盈止损各5%控回撤；1-2x 杠杆下总夏普最高(1.41)，适合优先匹配多份额。',
           cr['meta']['bh']['15m']),
        mk('高', '进取·收益最高', cr, 'macd+背离+量能', '1h', '双向多空', '只止损5%',
           '三年合计+1035%、总夏普1.37，只止损5%无爆仓；2026年币圈横盘仍+69.5%，主推进取配置。',
           cr['meta']['bh']['1h']),
    ],
    'us': [
        mk('低', '风控优先', us, 'MACD+背离', '1h', '仅做多', '止盈止损各5%',
           '止盈止损各5%回撤最浅(-19.1%)；注意美股所有配置均跑输买入持有，此档定位「降波动的持有替代」。',
           us['meta']['bh']['1h']),
        mk('中', '均衡·夏普最高', us, 'MACD+背离', '1h', '仅做多', '不设',
           '总夏普0.74为美股最高；不设止盈止损保留趋势收益，三年合计+109%。',
           us['meta']['bh']['1h']),
        mk('高', '收益最高', us, 'MACD+背离+量能+均线', '1h', '仅做多', '不设',
           '量能+均线过滤后三年+119.7%为美股最高，年下单数仅~160笔、更挑标的；美股双向多空全面劣于仅做多，不推荐做空。',
           us['meta']['bh']['1h']),
    ],
}

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=1, default=float)
print('已生成 →', OUT)
for mkt in ('crypto', 'us'):
    print('\n【%s】' % mkt)
    for r in data[mkt]:
        print('  %s %s: %s %s %s %s | 三年%+.1f%% 夏普%s 均回撤%s%%' % (
            r['risk'], r['label'], r['variant'], r['tf'], r['mode'], r['tpsl'],
            r['tot'] or 0, r['sh_all'], r['mdd_all_avg']))
