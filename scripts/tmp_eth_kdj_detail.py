# -*- coding: utf-8 -*-
"""ETH 4h KDJ 8/5 仅多：年度最大回撤 + 月度收益明细（口径与四年矩阵一致：现货、OR、tp8%/sl5%）"""
import os, sys, io, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7892")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7892")

import pandas as pd
from data_fetcher import BinanceDataFetcher
from indicators import TechnicalIndicators
from backtest_engine import BacktestEngine

YEARS = [("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
         ("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-08-26")]
INITIAL = 10000
ip = {"kdj": True, "kdj_k_period": 9, "kdj_d_period": 3, "kdj_j_period": 3,
      "kdj_buy_threshold": 20, "kdj_sell_threshold": 80}

fetcher = BinanceDataFetcher(); fetcher.set_market_type('spot')

for year, start, end in YEARS:
    df = None
    for i in range(6):
        df = fetcher.fetch_historical_data("ETH/USDT", start, end, "4h")
        if df is not None and len(df) > 500:
            break
        time.sleep(2 * (i + 1))
    dft = TechnicalIndicators.calculate_all_indicators(df, ip)
    eng = BacktestEngine(INITIAL, 0.001, take_profit=0.08, stop_loss=0.05,
                         timeframe="4h", signal_mode='or')
    res = eng.run_backtest(dft, ip)
    eq = res['equity_curve'][['timestamp', 'equity']].copy()
    eq['ts'] = pd.to_datetime(eq['timestamp']).dt.tz_localize(None)
    eq = eq.set_index('ts')['equity'].sort_index()
    # 年度最大回撤
    peak = eq.cummax()
    mdd = ((eq - peak) / peak * 100).min()
    # 月度收益（月末权益）
    monthly = eq.groupby(eq.index.to_period('M')).last()
    prev = INITIAL
    rows = []
    for p, v in monthly.items():
        rows.append(f"{p.month}月:{(v/prev-1)*100:+.1f}%")
        prev = v
    year_ret = (eq.iloc[-1] / INITIAL - 1) * 100
    pos_m = sum(1 for i in range(1, len(monthly) + 1)
                if (monthly.iloc[i-1] / (INITIAL if i == 1 else monthly.iloc[i-2]) - 1) > 0)
    print(f"===== {year}年 =====", flush=True)
    print(f"年度收益 {year_ret:+.1f}% | 年度最大回撤 {mdd:.1f}% | 盈利月 {pos_m}/{len(monthly)}", flush=True)
    print("  ".join(rows), flush=True)
    print(f"交易 {res['total_trades']} 笔 | 胜率 {res['win_rate']:.1f}% | 止盈 {res['take_profit_count']} 止损 {res['stop_loss_count']}", flush=True)
    print(flush=True)
