# -*- coding: utf-8 -*-
"""从2026寻优Excel补齐2025组策略的2026单元格"""
import io, sys, glob
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
f = sorted(glob.glob('scripts/results/daily_return_futures_top/daily_*.xlsx'))[-1]
df = pd.read_excel(f)
want = [
    ("BCH/USDT:USDT", "4h", "RSI+EMA", 0, 0),
    ("LTC/USDT:USDT", "1h", "RSI", 0, 0),
    ("XRP/USDT:USDT", "4h", "RSI", 0, 0),
    ("AVAX/USDT:USDT", "4h", "EMA+MACD", 5.0, 5.0),
    ("NEAR/USDT:USDT", "4h", "RSI", 0, 0),
    ("BTC/USDT:USDT", "4h", "布林带", 0, 0),
]
for sym, tf, strat, tp, sl in want:
    r = df[(df['品种'] == sym) & (df['周期'] == tf) & (df['策略'] == strat)
           & (df['模式'] == 'long_short') & (df['止盈%'] == tp) & (df['止损%'] == sl)]
    if r.empty:
        print(f"{sym} {tf} {strat}: 未找到", flush=True)
    else:
        row = r.iloc[0]
        print(f"{sym} {tf} {strat} 2026: 收益{row['总收益率%']}% 交易{row['交易次数']} 回撤{row['最大回撤%']}%", flush=True)
