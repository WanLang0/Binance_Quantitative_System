# -*- coding: utf-8 -*-
"""测试币安主网 API 的代理连通性"""
import os, io, sys, ssl
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import requests

proxies = {'http': 'http://127.0.0.1:7892', 'https': 'http://127.0.0.1:7892'}

# 增加 SSL 安全设置
s = requests.Session()
s.proxies = proxies
s.headers['User-Agent'] = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                           '(KHTML, like Gecko) Chrome/120.0 Safari/537.36')

targets = [
    'https://fapi.binance.com/fapi/v1/ping',
    'https://fapi.binance.com/fapi/v1/time',
    'https://fapi.binance.com/fapi/v1/exchangeInfo',
]
for t in targets:
    try:
        r = s.get(t, timeout=15)
        print(f"OK  {t} -> {r.status_code} {len(r.text)} bytes")
        if 'exchangeInfo' in t:
            print("  head:", r.text[:200])
    except Exception as e:
        print(f"ERR {t} -> {repr(e)[:160]}")
