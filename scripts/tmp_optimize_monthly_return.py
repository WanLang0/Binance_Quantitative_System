# -*- coding: utf-8 -*-
"""
ETH/BTC 策略寻优：放宽月回撤，追求「月均收益率」最大化
品种: ETH/USDT, BTC/USDT；周期 15m/1h/4h；时段 2026-01-01 ~ 2026-08-26
条件: 初始资金10000, 手续费0.1%, 周末不清仓, 信号合成 OR
目标: 月均收益率（每月收益率均值）最大化，同时报告盈利月占比
"""
import os, sys, io, itertools, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7892")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7892")

from datetime import datetime
import pandas as pd
import numpy as np
from data_fetcher import BinanceDataFetcher
from indicators import TechnicalIndicators
from backtest_engine import BacktestEngine

START = "2026-01-01"; END = "2026-08-26"
INITIAL_CAPITAL = 10000; COMMISSION = 0.001; WEEKLY_CLOSE = False
SYMBOLS = ["ETH/USDT", "BTC/USDT"]
TIMEFRAMES = ["15m", "1h", "4h"]
ALL_INDICATORS = ["RSI", "KDJ", "布林带", "EMA", "MACD", "双均线交叉"]
# 放宽月回撤，追求高收益：止损止盈空间拉大
STOP_LOSS_OPTIONS = [0.03, 0.05, 0.08]
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
    if "双均线交叉" in combo:
        p.update({"ma_cross": True, "ma_cross_short": 10, "ma_cross_long": 30,
                  "ma_cross_periods": [10, 30]})
    return p

# 组合：单策略 + 双策略 + 三策略（OR 合成）
COMBOS = []
for r in range(1, 4):
    for combo in itertools.combinations(ALL_INDICATORS, r):
        COMBOS.append(list(combo))
print("策略组合数:", len(COMBOS), flush=True)

def month_metrics(equity_curve):
    """返回 (月均收益率%, 盈利月数, 总月数, 每月收益率列表, 月内最大回撤%)"""
    if equity_curve is None or equity_curve.empty:
        return None
    ec = equity_curve.copy()
    dt = pd.to_datetime(ec['timestamp']).dt.tz_localize(None)
    ec['ym'] = dt.dt.to_period('M')
    monthly = ec.groupby('ym').agg(last_equity=('equity', 'last')).reset_index().sort_values('ym')
    rets = []
    prev = INITIAL_CAPITAL
    for _, r in monthly.iterrows():
        rets.append((r['last_equity'] / prev - 1) * 100)
        prev = r['last_equity']
    avg_month = sum(rets) / len(rets) if rets else 0.0
    pos_months = sum(1 for x in rets if x > 0)
    # 月内最大回撤（参考值，不再作硬约束）
    worst_month_dd = 0.0
    for ym, grp in ec.groupby('ym'):
        eq = grp['equity'].values
        peak = -np.inf; month_dd = 0.0
        for v in eq:
            peak = max(peak, v)
            month_dd = min(month_dd, (v - peak) / peak * 100)
        worst_month_dd = min(worst_month_dd, month_dd)
    return avg_month, pos_months, len(rets), rets, abs(worst_month_dd)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "monthly_return_optimize")
os.makedirs(out_dir, exist_ok=True)
fetcher = BinanceDataFetcher(); fetcher.set_market_type('spot')

all_rows = []
for s in SYMBOLS:
    for tf in TIMEFRAMES:
        print(f"\n=== {s} {tf} 拉取中... ===", flush=True)
        df = fetcher.fetch_historical_data(s, START, END, tf)
        if df is None or df.empty:
            print(f"  无数据", flush=True); continue
        print(f"  {len(df)}根K线", flush=True)
        t0 = time.time()
        done = 0
        for combo in COMBOS:
            ip = build_params(combo)
            dft = TechnicalIndicators.calculate_all_indicators(df, ip)
            for sl in STOP_LOSS_OPTIONS:
                for tp in TAKE_PROFIT_OPTIONS:
                    engine = BacktestEngine(INITIAL_CAPITAL, COMMISSION, take_profit=tp,
                                            stop_loss=sl, timeframe=tf, weekly_close=WEEKLY_CLOSE,
                                            signal_mode='or')
                    try:
                        res = engine.run_backtest(dft, ip)
                    except Exception:
                        continue
                    m = month_metrics(res.get('equity_curve'))
                    if m is None:
                        continue
                    avg_month, pos_months, n_months, rets, mdd = m
                    done += 1
                    all_rows.append({
                        '品种': s, '周期': tf,
                        '策略组合': "+".join(combo), '策略数': len(combo),
                        '止损%': round(sl * 100, 1), '止盈%': round(tp * 100, 1),
                        '总收益率%': round(res.get('total_return', 0), 2),
                        '月均收益率%': round(avg_month, 2),
                        '盈利月': f"{pos_months}/{n_months}",
                        '月内最大回撤%': round(mdd, 2),
                        '整段回撤%': round(res.get('max_drawdown', 0), 2),
                        '交易次数': res.get('total_trades', 0),
                        '夏普': round(res.get('sharpe_ratio', 0), 2),
                        '胜率%': round(res.get('win_rate', 0), 2),
                        '每月明细': ",".join(f"{x:.1f}" for x in rets),
                    })
        print(f"  完成 {done} 次回测，耗时 {time.time()-t0:.1f}s", flush=True)

df_out = pd.DataFrame(all_rows)
out_path = os.path.join(out_dir, f"monthly_return_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
df_out.to_excel(out_path, index=False)
print(f"\n✅ 全部完成，总回测 {len(df_out)} 条", flush=True)

# 只保留有交易的，按月均收益率降序
f = df_out[df_out['交易次数'] > 0].sort_values('月均收益率%', ascending=False).reset_index(drop=True)
print("\n=== 月均收益率 Top 40 ===", flush=True)
print(f.head(40)[['品种','周期','策略组合','止损%','止盈%','总收益率%','月均收益率%','盈利月','月内最大回撤%','交易次数','胜率%']].to_string(index=False, max_colwidth=14), flush=True)
f.to_excel(os.path.join(out_dir, f"按_月均收益率_排序_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"), index=False)
print(f"结果文件: {out_path}", flush=True)
