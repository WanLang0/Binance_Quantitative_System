# -*- coding: utf-8 -*-
"""核对：当前综合清单 55 只 vs 币安主网最新 188 只美股永续——找出已下架/不存在的"""
import os, sys, io, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(root, 'app.py'), encoding='utf-8') as f:
    app_src = f.read()
m = re.search(r'US_STOCK_BASES = \[(.*?)\]', app_src, re.S)
composite = set(re.findall(r"'([A-Z0-9]+)'", m.group(1)))
base_dir = os.path.dirname(os.path.abspath(__file__))
binance = set(json.load(open(os.path.join(base_dir, 'tmp_binance_us_bases.json'), encoding='utf-8')))

# 当前综合清单里不在币安主网的（需移除/存疑）
not_on_binance = composite - binance
# 币安主网里综合清单没有的（可新增）
missing = binance - composite

print(f"当前综合清单 {len(composite)} 只")
print(f"币安主网最新 {len(binance)} 只")
print(f"\n=== 当前综合里有，但币安主网最新清单没有 —— 需移除/存疑（{len(not_on_binance)}）===")
print(json.dumps(sorted(not_on_binance), ensure_ascii=False))

print(f"\n=== 币安主网最新有，但当前综合没有 —— 可新增（{len(missing)}）===")
print(json.dumps(sorted(missing), ensure_ascii=False))

# 特别标注：综合清单里疑似不在币安的
print("\n--- 存疑项是否在指数内 ---")
for b in sorted(not_on_binance):
    print(f"  {b:10}")
