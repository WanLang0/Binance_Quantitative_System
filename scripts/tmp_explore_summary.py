# -*- coding: utf-8 -*-
"""探查汇总 Excel 策略对比表的结构，确认列名"""
import os, sys, io, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

files = [
    r"results\eth\1h_OR\ETH_1h_全时段汇总_OR_止损5_止盈5_20260825_230150.xlsx",
    r"results\eth\15m_AND\ETH_15m_全时段汇总_止损5_止盈10_20260825_224221.xlsx",
    r"results\rs\RSI_BB_1h_8月_4币_20260825_233820.xlsx",
]
for f in files:
    p = os.path.join(ROOT, f)
    if not os.path.exists(p):
        print(f"[缺失] {f}"); continue
    try:
        xl = pd.ExcelFile(p)
        print(f"\n== {os.path.basename(f)} | sheets={xl.sheet_names}")
        df = xl.parse(xl.sheet_names[0])
        print(" 列名:", list(df.columns))
        print(df.head(6).to_string())
    except Exception as e:
        print(f"[错误] {f}: {e}")
