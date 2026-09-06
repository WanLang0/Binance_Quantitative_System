# -*- coding: utf-8 -*-
"""市值前30 × macd+背离+量能 × 三方案 汇总（基于前40明细JSON，取市值排名前30）"""
import io, sys, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np

YEARS = ('2024', '2025', '2026YTD')
CONFIGS = [
    ('15m', '方案一 15m双向 tpsl5'),
    ('15m', '方案二 15m仅多 不设'),
    ('1h', '方案三 1h双向 sl5'),
]

summary = json.load(open('scripts/results/top40all_macd_vol_summary_2024_2026.json', encoding='utf-8'))
detail = json.load(open('scripts/results/top40all_macd_vol_detail_2024_2026.json', encoding='utf-8'))['results']
tickers = summary['meta']['tickers']          # 市值降序
top30 = [t for t in tickers if t in detail][:30]
print(f'市值前30名单({len(top30)}只): {top30}\n')


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


out = {'meta': {'updated': summary['meta']['updated'], 'title': '市值前30 × macd+背离+量能 × 三方案',
                'tickers': top30, 'desc': summary['meta']['desc']},
       'strategies': [{'name': 'macd+背离+量能', 'configs': []}]}

for ctf, cname in CONFIGS:
    cfg = {'name': cname, 'tf': ctf}
    tot, all_n, mdd_a, mdd_m = 1.0, 0, [], []
    print(f"【{cname}】")
    for yr in YEARS:
        a = agg(top30, ctf, cname, yr)
        if not a:
            print(f'  {yr}: 无'); continue
        tot *= 1 + a['ret'] / 100
        all_n += a['n']; mdd_a.append(a['mdd']); mdd_m.append(a['mdd_max'])
        print(f"  {yr:8} 均收益{a['ret']:+7.1f}%  正收益{a['pos']:2}/{a['cnt']:2}"
              f"  | 平均MDD {a['mdd']:6.1f}%  最差MDD {a['mdd_max']:6.1f}%"
              f"  | 交易次数 总{a['n']:6d} 均值{a['n_avg']:6.1f}")
        cfg[yr] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in a.items()}
    cfg['tot'] = round((tot - 1) * 100, 1)
    cfg['n_all'] = all_n
    cfg['mdd_all_avg'] = round(float(np.mean(mdd_a)), 1) if mdd_a else None
    cfg['mdd_all_max'] = round(float(min(mdd_m)), 1) if mdd_m else None
    out['strategies'][0]['configs'].append(cfg)
    print(f"  三年合计 {cfg['tot']:+.1f}%  平均MDD {cfg['mdd_all_avg']}%  最差MDD {cfg['mdd_all_max']}%"
          f"  总交易 {all_n}\n")

with open('scripts/results/top30all_macd_vol_summary_2024_2026.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1, default=float)
print('summary → scripts/results/top30all_macd_vol_summary_2024_2026.json')
