# -*- coding: utf-8 -*-
"""检查美股永续在 exchangeInfo 中的标识字段"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import requests

proxies = {'http': 'http://127.0.0.1:7892', 'https': 'http://127.0.0.1:7892'}
s = requests.Session()
s.proxies = proxies
s.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36'

r = s.get('https://fapi.binance.com/fapi/v1/exchangeInfo', timeout=20)
r.raise_for_status()
data = r.json()
syms = data.get('symbols', [])

# 已知美股永续：NVDA, TREE, TSLA, AAPL
known = {'NVDA', 'TREE', 'TSLA', 'AAPL'}
for sym in syms:
    ba = sym.get('baseAsset', '')
    if ba in known:
        print(f"=== {sym['symbol']} (base={ba}) ===")
        print(json.dumps(sym, ensure_ascii=False, indent=2, default=str))
        print()
