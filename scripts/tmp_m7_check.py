# -*- coding: utf-8 -*-
"""七姐妹代币可用性核查：主网永续合约存在性 + 数据起点"""
import os, sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7892")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7892")

import ccxt

ex = ccxt.binanceusdm()
ex.proxies = {'http': 'http://127.0.0.1:7892', 'https': 'http://127.0.0.1:7892'}
markets = ex.load_markets()

M7 = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA']
found = {}
for b in M7:
    sym = f"{b}/USDT:USDT"
    if sym in markets and markets[sym].get('swap'):
        found[b] = sym

print("七姐妹主网永续合约:")
for b, s in found.items():
    print(f"  {s:22} active={markets[s].get('active')}")
missing = [b for b in M7 if b not in found and b != 'GOOG']
print("无合约:", missing)

# 数据起点（最早4h K线）
import time
print("\n数据起点（4h，合约）:")
for b, s in found.items():
    try:
        candles = ex.fetch_ohlcv(s, '4h', limit=1)  # 最新1根
        latest = ex.fetch_ohlcv(s, '4h', limit=500)
        # 用 since 二分太慢，直接拉1000根看最早
        batch = ex.fetch_ohlcv(s, '4h', limit=1000)
        earliest = batch[0][0] if batch else 0
        # 继续往前翻一页
        prev = ex.fetch_ohlcv(s, '4h', since=None, limit=1000, params={'endTime': earliest})
        days = (candles[0][0] - (prev[0][0] if prev else earliest)) / 86400000
        import datetime
        t0 = datetime.datetime.fromtimestamp((prev[0][0] if prev else earliest) / 1000)
        print(f"  {b:6} 数据约 {days:.0f} 天，起点 ~{t0.date()}，当前根数(首翻){len(batch)}")
        time.sleep(0.3)
    except Exception as e:
        print(f"  {b:6} 查询失败: {type(e).__name__}: {e}")
