# -*- coding: utf-8 -*-
"""检查候选品种在 2026H1 的 15m/5m 数据可用性"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7892")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7892")
from data_fetcher import BinanceDataFetcher

candidates = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "BNB/USDT", "MUUB/USDT",
              "SNDKB/USDT", "SKHYB/USDT", "MUB/USDT", "NVDAB/USDT", "LITEB/USDT", "MU/USDT"]
for tf in ["15m", "5m"]:
    print(f"--- {tf} 2026-01-01 ~ 2026-06-30 ---")
    f = BinanceDataFetcher()
    f.set_market_type('spot')
    for s in candidates:
        try:
            df = f.fetch_historical_data(s, "2026-01-01", "2026-06-30", tf)
            if df is not None and not df.empty:
                print(f"  {s}: {len(df)}根 {df.index[0].strftime('%y%m%d')}~{df.index[-1].strftime('%y%m%d')}")
            else:
                print(f"  {s}: 无数据")
        except Exception as e:
            print(f"  {s}: ERR {e}")
