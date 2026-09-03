# -*- coding: utf-8 -*-
"""精确核查综合清单里 3 只存疑代币在币安主网的真实身份"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import requests
proxies = {'http': 'http://127.0.0.1:7892', 'https': 'http://127.0.0.1:7892'}
s = requests.Session(); s.proxies = proxies
s.headers['User-Agent'] = 'Mozilla/5.0 Chrome/120.0'
data = s.get('https://fapi.binance.com/fapi/v1/exchangeInfo', timeout=25).json()
targets = {'CVX','STX','STXX','TREE','CXMT','MUU','NVDL','SKHY','SKHYNIX'}
for sym in data['symbols']:
    ba = sym.get('baseAsset','')
    if ba in targets:
        print(f"{sym['symbol']:20} base={ba:10} ctype={sym.get('contractType'):16} uType={sym.get('underlyingType'):8} sub={sym.get('underlyingSubType')} status={sym.get('status')}")
