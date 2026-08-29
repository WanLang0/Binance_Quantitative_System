# -*- coding: utf-8 -*-
"""ETH 4h KDJ 8/5 同款策略移植到其他币种：四年矩阵（口径与基准一致：现货、OR、tp8%/sl5%、仅做多）"""
import os, sys, io, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
os.environ.setdefault("HTTP_PROXY", "http://192.168.11.188:7892")
os.environ.setdefault("HTTPS_PROXY", "http://192.168.11.188:7892")

import pandas as pd
from data_fetcher import BinanceDataFetcher
from indicators import TechnicalIndicators
from backtest_engine import BacktestEngine

YEARS = [("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
         ("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-08-26")]
COINS = ["ETH", "BTC", "BNB", "SOL", "XRP", "ADA", "NEAR", "AVAX", "TON"]
INITIAL = 10000
ip = {"kdj": True, "kdj_k_period": 9, "kdj_d_period": 3, "kdj_j_period": 3,
      "kdj_buy_threshold": 20, "kdj_sell_threshold": 80}

fetcher = BinanceDataFetcher(); fetcher.set_market_type('spot')

def fetch(sym, start, end):
    for i in range(6):
        try:
            df = fetcher.fetch_historical_data(sym, start, end, "4h")
            if df is not None and len(df) > 500:
                return df
        except Exception as e:
            print(f"  [重试{i+1}] {sym} {start[:7]} 拉取异常: {type(e).__name__}", flush=True)
        time.sleep(2 * (i + 1))
    return None

results = {}
for coin in COINS:
    sym = f"{coin}/USDT"
    print(f"\n>>> {sym}", flush=True)
    rows = []
    for year, start, end in YEARS:
        df = fetch(sym, start, end)
        if df is None:
            print(f"  {year}: 数据拉取失败，跳过", flush=True)
            rows.append(None)
            continue
        dft = TechnicalIndicators.calculate_all_indicators(df, ip)
        eng = BacktestEngine(INITIAL, 0.001, take_profit=0.08, stop_loss=0.05,
                             timeframe="4h", signal_mode='or')
        res = eng.run_backtest(dft, ip)
        eq = res['equity_curve'][['timestamp', 'equity']].copy()
        eq['ts'] = pd.to_datetime(eq['timestamp']).dt.tz_localize(None)
        eq = eq.set_index('ts')['equity'].sort_index()
        peak = eq.cummax()
        mdd = ((eq - peak) / peak * 100).min()
        year_ret = (eq.iloc[-1] / INITIAL - 1) * 100
        print(f"  {year}: {year_ret:+7.1f}% | 回撤{mdd:6.1f}% | 交易{res['total_trades']:3d} | "
              f"胜率{res['win_rate']:5.1f}% | 止盈{res['take_profit_count']} 止损{res['stop_loss_count']}", flush=True)
        rows.append((year_ret, mdd, res['total_trades'], res['win_rate']))
    results[coin] = rows

print("\n" + "=" * 78)
print("四年矩阵汇总（年收益% | 累计复利%）")
print(f"{'币种':<6}{'2023':>10}{'2024':>10}{'2025':>10}{'2026':>10}{'四年累计':>10}", flush=True)
for coin, rows in results.items():
    cells, cum, ok = [], 1.0, True
    for r in rows:
        if r is None:
            cells.append(f"{'--':>10}"); ok = False
        else:
            cells.append(f"{r[0]:>+9.1f}%"); cum *= (1 + r[0] / 100)
    total = f"{(cum-1)*100:>+9.1f}%" if ok else "  不完整"
    print(f"{coin:<6}{''.join(cells)}{total:>10}", flush=True)
