# -*- coding: utf-8 -*-
import json
import numpy as np

d = json.load(open('scripts/results/top40all_macd_vol_monthly_2024_2026.json', encoding='utf-8'))['monthly']
cfgs = [k for k in d[0] if k != 'month']
for k in cfgs:
    vals = [r[k]['mean'] for r in d if r.get(k)]
    print(f"{k}: 算术月均 {np.mean(vals):+.2f}%  中位月 {np.median(vals):+.2f}%  样本月数 {len(vals)}")
