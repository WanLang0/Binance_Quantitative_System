# -*- coding: utf-8 -*-
"""从40币冷却期回测明细中，切出综合量化30币清单口径的月度统计对比"""
import json
import numpy as np

BASES30 = ['BTC', 'ETH', 'BNB', 'XRP', 'SOL', 'TRX', 'HYPE', 'ZEC', 'DOGE', 'XMR',
           'LINK', 'ADA', 'XLM', 'BCH', 'CC', 'LTC', 'UNI', 'GRAM', 'HBAR', 'AVAX',
           'SUI', 'NEAR', 'M', 'TAO', 'ASTER', 'AAVE', 'ONDO', 'MORPHO', 'DOT', 'ICP']

d = json.load(open('scripts/results/top40all_macd_vol_monthly_cooldown.json', encoding='utf-8'))
pc = d['per_coin']   # {月: {配置名: {cd(str): {base: {...}}}}}
CFGS = [('方案一 15m双向 tpsl5', '0'), ('方案二 15m仅多 不设', '0'),
        ('方案三 1h双向 sl5', '0'), ('方案三 1h双向 sl5', '16')]
months = sorted(pc.keys())


def stats(cfg, cd, bases=None):
    vals40, vals, nav, tot_n, tot_w, tot_l, rows = [], [], 1.0, 0, 0, 0, []
    for m in months:
        dd = pc[m].get(cfg, {}).get(cd, {})
        if not dd:
            continue
        rets40 = [v['ret'] for v in dd.values()]
        rets = [v['ret'] for b, v in dd.items() if bases is None or b in bases]
        vals40.append(float(np.mean(rets40)))
        if rets:
            v = float(np.mean(rets))
            vals.append(v); nav *= (1 + v / 100)
            rows.append((m, v, sum(1 for x in rets if x > 0), len(rets)))
            sel = dd.items() if bases is None else [(b, x) for b, x in dd.items() if b in bases]
            tot_n += sum(x['n'] for _, x in sel)
            tot_w += sum(x['wins'] for _, x in sel)
            tot_l += sum(x['losses'] for _, x in sel)
    return vals40, vals, nav, tot_n, tot_w, tot_l, rows


for cfg, cd in CFGS:
    v40, _, _, _, _, _, _ = stats(cfg, cd)
    _, v30, nav30, n30, w30, l30, _ = stats(cfg, cd, bases=BASES30)
    print(f"{cfg} cd{cd}")
    print(f"  40币: 月均{np.mean(v40):+.2f}%  亏损月{sum(1 for x in v40 if x < 0)}/{len(v40)}")
    print(f"  30币: 月均{np.mean(v30):+.2f}%  复利{nav30:.3f}x  "
          f"亏损月{sum(1 for x in v30 if x < 0)}/{len(v30)}  最差{min(v30):+.2f}%  "
          f"总下单{n30}  胜率{w30 / (w30 + l30) * 100:.1f}%\n")

for cfg, cd in [('方案一 15m双向 tpsl5', '0'), ('方案三 1h双向 sl5', '0')]:
    *_, rows = stats(cfg, cd, bases=BASES30)
    print(f"--- {cfg} 30币逐月 ---")
    for m, v, pos, cnt in rows:
        print(f"  {m}  {v:+7.2f}%  ({pos}/{cnt})")
