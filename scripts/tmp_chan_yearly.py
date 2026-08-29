# -*- coding: utf-8 -*-
"""缠论两个最优配置的逐年矩阵（验证跨年稳健性，非单次挑选）"""
import os, sys, io, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7892")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7892")

import numpy as np
import pandas as pd
from data_fetcher import BinanceDataFetcher

YEARS = [("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
         ("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-08-26")]
COINS = ["ETH", "XLM", "BTC", "BCH", "LTC", "ADA"]
INITIAL = 10000.0; COMM = 0.001

# exec(tmp_chan_test.py 里的函数定义，避免重复代码)
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_chan_test.py"), encoding='utf-8').read()
ns = {'np': np, 'pd': pd, '__file__': __file__}
exec(compile(src.split("# ---------------- 主流程")[0], 'chan_funcs', 'exec'), ns)
first_buy_signal = ns['first_buy_signal']
third_buy_signal = ns['third_buy_signal']
simulate = ns['simulate']
BinanceDataFetcher = ns['BinanceDataFetcher']

fetcher = BinanceDataFetcher(); fetcher.set_market_type('future')

def fetch(sym, start, end, tries=6):
    for i in range(tries):
        try:
            df = fetcher.fetch_historical_data(sym, start, end, "4h")
            if df is not None and not df.empty and len(df) >= 100:
                return df
        except Exception:
            pass
        time.sleep(2 * (i + 1))
    return None

out_lines = []
def w(s=""):
    out_lines.append(s); print(s, flush=True)

CONFIGS = [
    ("一买深跌30% 8/5 仅多", 'b1', 0.30, 180, 0.08, 0.05, None, 'long_only'),
    ("一买深跌30% 5/5 仅多", 'b1', 0.30, 180, 0.05, 0.05, None, 'long_only'),
    ("三买 N=20 不设/持90 仅多", 'b3', 20, None, None, None, 90, 'long_only'),
]

for title, kind, p1, p2, tp, sl, mh, mode in CONFIGS:
    w("\n" + "=" * 92)
    w(f"◆ {title}")
    w(f"{'币':<6}{'2023':>9}{'2024':>9}{'2025':>9}{'2026':>9}{'四年累计':>9}{'笔数':>5}")
    for coin in COINS:
        sym = f"{coin}/USDT:USDT"
        # 全期数据生成信号后按年切片（保证跨年笔归属一致）
        dfs = []
        for year, start, end in YEARS:
            df = fetch(sym, start, end)
            if df is not None:
                dfs.append(df)
        if not dfs:
            continue
        full = pd.concat(dfs).sort_index()
        full = full[~full.index.duplicated(keep='first')]
        if kind == 'b1':
            sig_all = first_buy_signal(full, p1, p2)
        else:
            sig_all = third_buy_signal(full, p1)
        cells = []; cum = 1.0; trades = 0
        for year, start, end in YEARS:
            sub = full[(full.index >= pd.Timestamp(start)) & (full.index <= pd.Timestamp(end))]
            sig = sig_all.reindex(sub.index).fillna(0)
            r = simulate(sub, sig, tp, sl, mode, mh)
            if r is None:
                cells.append(f"{'--':>9}")
            else:
                cells.append(f"{r[0]:>+8.1f}%")
                cum *= (1 + r[0] / 100)
                trades += r[1]
        w(f"{coin:<6}{''.join(cells)}{(cum-1)*100:>+9.1f}%{trades:>5}")

out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "chan_best_yearly.txt")
with open(out_file, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))
print(f"\nSaved -> {out_file}", flush=True)
