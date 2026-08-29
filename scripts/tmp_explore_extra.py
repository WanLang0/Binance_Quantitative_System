# -*- coding: utf-8 -*-
"""补齐：2026-07~08 的 1h OR RSI+布林带，及近期 BTC/ETH 8月(止盈3止损3)"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

files = [
    r"results\eth\1h_OR\ETH_1h_2026-07-01至08-25_OR_止损5_止盈5_20260825_231712.xlsx",
    r"results\BTC_8月\BTC_1h_8月_RSI_BB_止损3止盈3_20260825_235430.xlsx",
    r"results\ETH_8月\ETH_1h_8月_RSI_BB_止损3止盈3_20260825_235618.xlsx",
    r"results\RSI_BB_4币_8月\RSI_BB_1h_8月_4币_20260825_233820.xlsx",
]
for f in files:
    p = os.path.join(ROOT, f)
    if not os.path.exists(p):
        print(f"[缺失] {os.path.basename(f)}"); continue
    df = pd.read_excel(p, sheet_name="策略对比")
    print(f"\n== {os.path.basename(f)}")
    print(df.to_string(index=False))
