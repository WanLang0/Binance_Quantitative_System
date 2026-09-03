# -*- coding: utf-8 -*-
"""审计：币安正式主网 API (fapi.binance.com /fapi/v1/exchangeInfo) 全量列出所有美股永续
准确标识: contractType == 'TRADIFI_PERPETUAL' 且 underlyingType == 'EQUITY'"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import requests

proxies = {'http': 'http://127.0.0.1:7892', 'https': 'http://127.0.0.1:7892'}
s = requests.Session()
s.proxies = proxies
s.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36'

r = s.get('https://fapi.binance.com/fapi/v1/exchangeInfo', timeout=25)
r.raise_for_status()
data = r.json()
syms = data.get('symbols', [])

found = []
for sym in syms:
    if sym.get('contractType') == 'TRADIFI_PERPETUAL':
        found.append({
            'symbol': sym['symbol'],
            'base': sym.get('baseAsset'),
            'status': sym.get('status'),
            'underlyingType': sym.get('underlyingType'),
            'sub': sym.get('underlyingSubType'),
            'onboard': sym.get('onboardDate'),
            'qty_prec': sym.get('quantityPrecision'),
        })

bases = sorted({f['base'] for f in found})
print(f"币安主网 TRADIFI 美股永续总数: {len(found)}  (去重 base: {len(bases)})")
print("\n全部美股永续 (symbol / status / base):")
for f in sorted(found, key=lambda x: x['base']):
    print(f"{f['symbol']:22} status={f['status']:9} base={f['base']:10} sub={f['sub']}")

print(f"\n=== 去重 base 清单 ({len(bases)}) ===")
print(json.dumps(bases, ensure_ascii=False))

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp_binance_us_bases.json'), 'w', encoding='utf-8') as fp:
    json.dump(bases, fp, ensure_ascii=False, indent=2)
print("\n已写入 tmp_binance_us_bases.json")
