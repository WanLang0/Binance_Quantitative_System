# -*- coding: utf-8 -*-
"""
组合优化(实时进度+分品种保存)：在「周回撤≤5%」约束下，最大化周均盈利
品种: BTC/ETH/XRP/BNB；周期 15m；时段 2026H1
条件: 初始资金10000, 手续费0.1%, 周末不清仓
"""
import os, sys, io, itertools, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7892")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7892")

from datetime import datetime
import pandas as pd
from data_fetcher import BinanceDataFetcher
from indicators import TechnicalIndicators
from backtest_engine import BacktestEngine

START = "2026-01-01"; END = "2026-06-30"
INITIAL_CAPITAL = 10000; COMMISSION = 0.001; TIMEFRAME = "15m"; WEEKLY_CLOSE = False
SYMBOLS = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "BNB/USDT"]
ALL_INDICATORS = ["RSI", "KDJ", "布林带", "EMA", "MACD"]
STOP_LOSS_OPTIONS = [0.03, 0.05, 0.07]
TAKE_PROFIT_OPTIONS = [0.05, 0.08, 0.10, 0.15]

def build_params(combo):
    p = {}
    if "RSI" in combo:
        p.update({"rsi": True, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70})
    if "KDJ" in combo:
        p.update({"kdj": True, "kdj_k_period": 9, "kdj_d_period": 3, "kdj_j_period": 3,
                  "kdj_buy_threshold": 20, "kdj_sell_threshold": 80})
    if "布林带" in combo:
        p.update({"boll": True, "bb_period": 20, "bb_std": 2.0})
    if "EMA" in combo:
        p.update({"ema": True, "ema_short": 12, "ema_long": 26, "ema_periods": [12, 26]})
    if "MACD" in combo:
        p.update({"macd": True, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9})
    return p

COMBOS = []
for r in range(2, len(ALL_INDICATORS) + 1):
    for combo in itertools.combinations(ALL_INDICATORS, r):
        COMBOS.append(list(combo))
print("多策略组合数:", len(COMBOS), flush=True)
PER_SYMBOL = len(COMBOS) * len(STOP_LOSS_OPTIONS) * len(TAKE_PROFIT_OPTIONS)
print("每品种回测数:", PER_SYMBOL, flush=True)

def week_metrics(equity_curve):
    if equity_curve is None or equity_curve.empty:
        return None, None
    ec = equity_curve.copy()
    dt = pd.to_datetime(ec['timestamp']).dt.tz_localize(None)
    ec['week'] = dt.dt.to_period('W-MON').dt.start_time
    weekly = ec.groupby('week').agg(first_equity=('equity', 'first'), last_equity=('equity', 'last')).reset_index()
    weekly = weekly.sort_values('week').reset_index(drop=True)
    eq = weekly['last_equity'].values
    if len(eq) == 0:
        return None, None
    peak = -1e18; max_dd = 0.0
    for v in eq:
        peak = max(peak, v); dd = (v - peak) / peak * 100; max_dd = min(max_dd, dd)
    rets = []
    prev = None
    for _, r in weekly.iterrows():
        if prev is not None and prev > 0:
            rets.append((r['last_equity'] / prev - 1) * 100)
        prev = r['last_equity']
    return abs(max_dd), (sum(rets) / len(rets) if rets else 0.0)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "combo_optimize_2026H1")
os.makedirs(out_dir, exist_ok=True)
fetcher = BinanceDataFetcher(); fetcher.set_market_type('spot')

all_rows = []
for s in SYMBOLS:
    print(f"\n=== {s} 拉取中... ===", flush=True)
    df = fetcher.fetch_historical_data(s, START, END, TIMEFRAME)
    if df is None or df.empty:
        print(f"  {s}: 无数据", flush=True); continue
    print(f"  {s}: {len(df)}根", flush=True)
    idx = 0
    for combo in COMBOS:
        combo_name = "+".join(combo)
        ip = build_params(combo)
        dft = TechnicalIndicators.calculate_all_indicators(df, ip)
        for sl in STOP_LOSS_OPTIONS:
            for tp in TAKE_PROFIT_OPTIONS:
                engine = BacktestEngine(INITIAL_CAPITAL, COMMISSION, take_profit=tp,
                                        stop_loss=sl, timeframe=TIMEFRAME, weekly_close=WEEKLY_CLOSE)
                try:
                    res = engine.run_backtest(dft, ip)
                except Exception:
                    continue
                wdd, wavg = week_metrics(res.get('equity_curve'))
                if wdd is None:
                    continue
                idx += 1
                all_rows.append({
                    '品种': s, '策略组合': combo_name, '策略数': len(combo),
                    '止损%': round(sl * 100, 0), '止盈%': round(tp * 100, 0),
                    '收益率%': round(res['total_return'], 2),
                    '整段回撤%': round(res['max_drawdown'], 2),
                    '周均盈利%': round(wavg, 2), '周回撤%': round(wdd, 2),
                    '买入次数': res.get('take_profit_count', 0) + res.get('stop_loss_count', 0) + res.get('normal_sell_count', 0),
                    '夏普': round(res['sharpe_ratio'], 2), '胜率%': round(res['win_rate'], 2),
                })
        if idx % 30 == 0:
            print(f"  {s} 已完成 {idx}/{PER_SYMBOL}", flush=True)
    # 每品种落地一次
    tmp = pd.DataFrame(all_rows)
    tmp.to_excel(os.path.join(out_dir, f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{s.replace('/','_')}.xlsx"), index=False)
    print(f"  {s} 完成，累计记录 {len(all_rows)}", flush=True)

df = pd.DataFrame(all_rows)
out_path = os.path.join(out_dir, f"突破回撤5_2026H1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
df.to_excel(out_path, index=False)
f = df[(df['周回撤%'] <= 5) & (df['买入次数'] > 0)].copy()
print("\n总回测:", len(df), flush=True)
print("周回撤<=5% 且有交易:", len(f), flush=True)
if not f.empty:
    f = f.sort_values('周均盈利%', ascending=False).reset_index(drop=True)
    print("\n=== 满足约束 Top 20 ===", flush=True)
    print(f.head(20).to_string(index=False, max_colwidth=18), flush=True)
    f.to_excel(os.path.join(out_dir, "满足约束_周回撤5.xlsx"), index=False)
print(f"✅ 已输出: {out_path}", flush=True)
