# -*- coding: utf-8 -*-
import sqlite3, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
c = sqlite3.connect('data/strategy_records.db')
tested = {r[0].split('/')[0].split('~')[0].upper() for r in c.execute("SELECT DISTINCT symbol FROM strategy_records")}

# 用户 15 组清单的全部候选代码
candidates = {
 '存储': ['000660.KS','MU','WDC','STX','RMBS','SIMO'],
 'CPO': ['AVGO','APH','MRVL','GLW','LITE','NOK','COHR','CIEN','ALAB','CRDO'],
 'AI芯片': ['NVDA','TSM','MU','AVGO','AMD','ASML','INTC','LRCX','AMAT','TXN'],
 '云计算': ['MSFT','AMZN','GOOGL','META','ORCL','BABA','IBM','CRM','NOW','HPE'],
 '半导体': ['NVDA','TSM','MU','AVGO','AMD','ASML','INTC','LRCX','AMAT','TXN'],
 '数据中心': ['NVDA','AMD','INTC','MRVL','EQIX','DLR','MCHP','SMCI','GDS'],
 '航天': ['GE','RTX','BA','LMT','HWM','NOC','RKLB','ASTS','GSAT'],
 '卫星通信': ['RKLB','ASTS','SATS','GSAT','VSAT','IRDM','GILT','SATL','TSAT'],
 '机器人': ['MSFT','TSLA','MU','AMD','INTC','SONY','NOC','STM','CGNX','SERVE'],
 '军工': ['GE','RTX','BA','LMT','HWM','GD','NOC','TDG','AXON','RKLB'],
 '石油': ['XOM','CVX','SHEL','TTE','COP','MPC','CNQ','VLO','PSX','WMB'],
 '天然气': ['XOM','CVX','COP','PSX','WMB','KMI','TRGP','OKE','OXY','DVN'],
 '黄金': ['NEM','FCX','AEM','WPM','AU','FNV','GFI','KGC','RGLD','CDE'],
 '生物医药': ['LLY','JNJ','ABBV','MRK','AMGN','TMO','ABT','GILD','PFE','NVO'],
 '消费': ['AMZN','WMT','COST','KO','PG','HD','PEP','DIS','MCD','TJX'],
}

untested = {}
for sector, syms in candidates.items():
    untested[sector] = [s for s in syms if s.upper() not in tested]
total = sum(len(v) for v in untested.values())
print("未测试标的总数:", total, "\n")
for sector, syms in untested.items():
    print(f"[{sector}] {len(syms)}: {', '.join(syms)}")
