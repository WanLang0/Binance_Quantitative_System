# -*- coding: utf-8 -*-
import json

d = json.load(open('scripts/results/top40all_macd_vol_monthly_cooldown.json', encoding='utf-8'))
for key, s in d['summary'].items():
    yr = s.get('per_year', {})
    ytxt = '  '.join(f"{y}: 下单{v['n']}笔 胜率{v['win_rate']}% 月收益和{v['ret_sum']}%" for y, v in yr.items())
    print(f"{key}\n  {ytxt}")
