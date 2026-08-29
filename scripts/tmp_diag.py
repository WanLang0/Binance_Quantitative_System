# -*- coding: utf-8 -*-
"""诊断：单品种15m数据拉取 + 单次回测耗时"""
import os, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7892")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7892")
from data_fetcher import BinanceDataFetcher
from indicators import TechnicalIndicators
from backtest_engine import BacktestEngine

t = time.time()
f = BinanceDataFetcher(); f.set_market_type('spot')
df = f.fetch_historical_data("BTC/USDT", "2026-01-01", "2026-06-30", "15m")
print(f"拉取: {len(df)}根, 耗时{time.time()-t:.1f}s", flush=True)
ip = {"kdj": True, "macd": True, "kdj_k_period":9,"kdj_d_period":3,"kdj_j_period":3,
      "kdj_buy_threshold":20,"kdj_sell_threshold":80,
      "macd_fast":12,"macd_slow":26,"macd_signal":9}
dft = TechnicalIndicators.calculate_all_indicators(df, ip)
print(f"指标计算: 耗时{time.time()-t:.1f}s", flush=True)
eng = BacktestEngine(10000, 0.001, take_profit=0.10, stop_loss=0.05, timeframe="15m", weekly_close=False)
t2 = time.time()
res = eng.run_backtest(dft, ip)
print(f"回测: 耗时{time.time()-t2:.1f}s, total_return={res['total_return']:.2f}%", flush=True)
eq = res['equity_curve']
print("equity_curve cols:", list(eq.columns), flush=True)
print("timestamp dtype:", eq['timestamp'].dtype, flush=True)
print("index:", eq.index[:2].tolist(), flush=True)
