# -*- coding: utf-8 -*-
"""补算 MUB/MUUB（现货代币）交易次数与胜率"""
import os, sys, io, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7892")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7892")

from data_fetcher import BinanceDataFetcher
from indicators import TechnicalIndicators
from backtest_engine import BacktestEngine
from tmp_metrics_for_summary import simulate_trades, BASE

fetcher = BinanceDataFetcher(); fetcher.set_market_type('spot')
engine = BacktestEngine(signal_mode='or')

for sym in ["MUB/USDT", "MUUB/USDT"]:
    df = None
    for i in range(6):
        df = fetcher.fetch_historical_data(sym, "2026-01-01", "2026-08-26", "4h")
        if df is not None and len(df) >= 100:
            break
        time.sleep(2 * (i + 1))
    if df is None or df.empty:
        print(f"{sym}: 无数据", flush=True)
        continue
    ip = dict(BASE["KDJ"])
    dft = TechnicalIndicators.calculate_all_indicators(df, ip)
    signals = engine.calculate_signals(dft, ip)
    pnls = simulate_trades(df, signals, None, None, "long_short")
    if not pnls:
        print(f"{sym}: 无交易", flush=True)
        continue
    wins = sum(1 for p in pnls if p > 0)
    total = (10000 + sum(pnls)) / 100 - 100
    print(f"{sym}: 交易{len(pnls)}笔 胜率{wins/len(pnls)*100:.1f}% 总收益{total:.1f}%", flush=True)
