# -*- coding: utf-8 -*-
import os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7892"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7892"
from data_fetcher import BinanceDataFetcher

syms = ["XRP/USDT", "BNB/USDT", "SPCXB/USDT", "SNDKB/USDT", "NVDA/USDT", "NVDAB/USDT"]
periods = [("2025H1","2025-01-01","2025-06-30"), ("2025H2","2025-07-01","2025-12-31"), ("2026H1","2026-01-01","2026-08-24")]
for sym in syms:
    f = BinanceDataFetcher(); f.set_market_type('spot')
    avail = []
    for p, s, e in periods:
        try:
            df = f.fetch_historical_data(sym, s, e, "15m")
            if df is not None and not df.empty:
                avail.append(f"{p}({len(df)}根,{df.index[0].strftime('%y%m%d')}~{df.index[-1].strftime('%y%m%d')})")
            else:
                avail.append(f"{p}(无)")
        except Exception as ex:
            avail.append(f"{p}(ERR)")
    print(f"{sym}: " + " | ".join(avail))
