# -*- coding: utf-8 -*-
import pandas as pd, glob, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
f = sorted(glob.glob('scripts/results/daily_return_2025/daily_*.xlsx'))[-1]
df = pd.read_excel(f)
print('总记录', len(df))
d = df[(df['交易次数'] > 0) & (df['总收益率%'] > 0)]
t = d.sort_values('日均收益率%', ascending=False).head(12)
print('=== 正收益且日均 Top12 ===')
print(t[['品种','周期','模式','策略','止盈%','止损%','总收益率%','日均收益率%','日波动%','最差日%','盈利日','最大回撤%','交易次数']].to_string(index=False))
s = d[d['日波动%'] < 2].sort_values('日均收益率%', ascending=False).head(10)
print('=== 稳定组(日波动<2%) Top10 ===')
print(s[['品种','周期','模式','策略','止盈%','止损%','总收益率%','日均收益率%','日波动%','盈利日','最大回撤%']].to_string(index=False))
print('=== 2026年冠军在2025的交叉验证 ===')
for sym in ['TON', 'NEAR', 'ETH', 'BTC']:
    x = df[df['品种'].str.contains(sym)]
    if not x.empty:
        b = x.sort_values('日均收益率%', ascending=False).iloc[0]
        print(f"{sym} 2025最优: {b['周期']} {b['策略']} {b['模式']} 收益{b['总收益率%']}% 日均{b['日均收益率%']}% 回撤{b['最大回撤%']}%")
