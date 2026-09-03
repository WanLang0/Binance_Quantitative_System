# -*- coding: utf-8 -*-
"""对比：道琼斯30 + 纳指100 成分 vs 币安主网美股永续(188) vs 当前综合清单(US_STOCK_BASES)"""
import os, sys, io, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(root, 'app.py'), encoding='utf-8') as f:
    app_src = f.read()
m = re.search(r'US_STOCK_BASES = \[(.*?)\]', app_src, re.S)
composite = set(re.findall(r"'([A-Z0-9]+)'", m.group(1)))
print(f"当前综合清单 US_STOCK_BASES: {len(composite)} 只")

base_dir = os.path.dirname(os.path.abspath(__file__))
binance = set(json.load(open(os.path.join(base_dir, 'tmp_binance_us_bases.json'), encoding='utf-8')))
print(f"币安主网美股永续: {len(binance)} 只")

DOW = {'AAPL','AMGN','AMZN','AXP','BA','CAT','CRM','CSCO','CVX','DIS','DOW','GS','HD','HON',
       'IBM','INTC','JNJ','JPM','KO','MCD','MMM','MRK','MSFT','NKE','NVDA','PG','TRV','UNH',
       'V','WMT','GOOGL'}
NDX = {'NVDA','AAPL','GOOGL','GOOG','MSFT','AMZN','AVGO','META','TSLA','MU','WMT','AMD','INTC',
       'ASML','CSCO','PLTR','COST','AMAT','LRCX','PANW','NFLX','KLAC','TXN','AMGN','LIN','CRWD',
       'TMUS','PEP','MRVL','SHOP','STX','GILD','ADI','QCOM','BKNG','WDC','HON','VRTX','ISRG',
       'FTNT','ADBE','ABNB','ADP','APP','CEG','MELI','DASH','CSX','MAR','CMCSA','INTU','DDOG',
       'MNST','CDNS','REGN','CTAS','MDLZ','SNPS','ROST','ORLY','MPWR','WBD','PCAR','AEP','BKR',
       'TRI','FAST','NXPI','FANG','ADSK','EA','PYPL','AXON','XEL','CCEP','EXC','TTWO','ODFL',
       'IDXX','WDAY','MCHP','PAYX','KDP','TEAM','FER','ROP','MSTR','DXCM','GEHC','ALNY','KHC',
       'CPRT','ALAB','CRWV','NBIS','RKLB','TER'}

idx = set(DOW) | set(NDX)
print(f"道琼斯30+纳指100 去重后成分总数: {len(idx)}")

have = idx & binance & composite
missing = (idx & binance) - composite
no_binance = idx - binance

print(f"\n=== 已在综合清单（{len(have)}）===")
print(json.dumps(sorted(have), ensure_ascii=False))
print(f"\n=== 币安主网可交易 但 综合清单缺失 —— 可新增（{len(missing)}）===")
print(json.dumps(sorted(missing), ensure_ascii=False))
print(f"\n=== 指数成分 但 币安主网无上市 —— 无法新增（{len(no_binance)}）===")
print(json.dumps(sorted(no_binance), ensure_ascii=False))
print("\n--- 可新增明细 ---")
for b in sorted(missing):
    tags = []
    if b in DOW: tags.append('道琼斯')
    if b in NDX: tags.append('纳指100')
    print(f"  {b:8} {'/'.join(tags)}")
