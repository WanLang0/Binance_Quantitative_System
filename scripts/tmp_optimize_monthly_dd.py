# -*- coding: utf-8 -*-
"""
ETH 策略寻优：在「月内最大回撤≤3%」约束下，最大化总收益率
品种: ETH/USDT；周期 15m/1h/4h；时段 2026-01-01 ~ 2026-08-26
条件: 初始资金10000, 手续费0.1%, 周末不清仓, 信号合成 OR
"""
import os, sys, io, itertools, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
os.environ.setdefault("HTTP_PROXY", "http://192.168.11.188:7892")
os.environ.setdefault("HTTPS_PROXY", "http://192.168.11.188:7892")

from datetime import datetime
import pandas as pd
import numpy as np
from data_fetcher import BinanceDataFetcher
from indicators import TechnicalIndicators
from backtest_engine import BacktestEngine

START = "2026-01-01"; END = "2026-08-26"
INITIAL_CAPITAL = 10000; COMMISSION = 0.001; WEEKLY_CLOSE = False
SYMBOL = "BTC/USDT"
TIMEFRAMES = ["15m", "1h", "4h"]
ALL_INDICATORS = ["RSI", "KDJ", "布林带", "EMA", "MACD", "双均线交叉"]
# 月回撤≤3% 约束严格，止损不能太大，否则一笔止损就会让当月回撤触顶
STOP_LOSS_OPTIONS = [0.01, 0.02, 0.03]
TAKE_PROFIT_OPTIONS = [0.03, 0.05, 0.08]
MAX_MONTHLY_DD = 3.0  # %

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

# 组合：单策略 + 双策略组合（OR 合成），先覆盖最可能出解的空间
COMBOS = []
for r in range(1, 3):
    for combo in itertools.combinations(ALL_INDICATORS, r):
        COMBOS.append(list(combo))
print("策略组合数:", len(COMBOS), flush=True)

def month_metrics(equity_curve):
    """返回 (月内最大回撤%, 总收益率%)。月内最大回撤=每个自然月内净值从月内高点回落的最大幅度，取最差月。"""
    if equity_curve is None or equity_curve.empty:
        return None, None
    ec = equity_curve.copy()
    dt = pd.to_datetime(ec['timestamp']).dt.tz_localize(None)
    ec['ym'] = dt.dt.to_period('M')
    worst_month_dd = 0.0  # 最差月的月内回撤（负数绝对值）
    for ym, grp in ec.groupby('ym'):
        eq = grp['equity'].values
        if len(eq) == 0:
            continue
        peak = -np.inf
        month_dd = 0.0
        for v in eq:
            if v > peak:
                peak = v
            dd = (v - peak) / peak * 100
            if dd < month_dd:
                month_dd = dd
        if month_dd < worst_month_dd:
            worst_month_dd = month_dd
    final = ec['equity'].iloc[-1]
    total_ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    return abs(worst_month_dd), total_ret

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "monthly_dd_optimize_btc")
os.makedirs(out_dir, exist_ok=True)
fetcher = BinanceDataFetcher(); fetcher.set_market_type('spot')

all_rows = []
for tf in TIMEFRAMES:
    print(f"\n=== ETH {tf} 拉取中... ===", flush=True)
    df = fetcher.fetch_historical_data(SYMBOL, START, END, tf)
    if df is None or df.empty:
        print(f"  {tf}: 无数据", flush=True); continue
    print(f"  {tf}: {len(df)}根K线", flush=True)
    t0 = time.time()
    done = 0
    for combo in COMBOS:
        combo_name = "+".join(combo)
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
                mdd, ret = month_metrics(res.get('equity_curve'))
                if mdd is None:
                    continue
                done += 1
                all_rows.append({
                    '周期': tf, '策略组合': combo_name, '策略数': len(combo),
                    '止损%': round(sl * 100, 1), '止盈%': round(tp * 100, 1),
                    '收益率%': round(ret, 2),
                    '月内最大回撤%': round(mdd, 2),
                    '整段回撤%': round(res.get('max_drawdown', 0), 2),
                    '交易次数': res.get('total_trades', 0),
                    '夏普': round(res.get('sharpe_ratio', 0), 2),
                    '胜率%': round(res.get('win_rate', 0), 2),
                })
    print(f"  {tf} 完成 {done} 次回测，耗时 {time.time()-t0:.1f}s", flush=True)

df_out = pd.DataFrame(all_rows)
out_path = os.path.join(out_dir, f"eth_monthly_dd3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
df_out.to_excel(out_path, index=False)
print(f"\n✅ 全部完成，总回测 {len(df_out)} 条", flush=True)

# 筛选满足约束且有交易的
f = df_out[(df_out['月内最大回撤%'] <= MAX_MONTHLY_DD) & (df_out['交易次数'] > 0)].copy()
print(f"月内回撤≤{MAX_MONTHLY_DD}% 且有交易: {len(f)} 条", flush=True)
if not f.empty:
    f = f.sort_values('收益率%', ascending=False).reset_index(drop=True)
    print("\n=== 满足约束 Top 30（按收益率降序） ===", flush=True)
    print(f.head(30).to_string(index=False, max_colwidth=16), flush=True)
    f.to_excel(os.path.join(out_dir, f"满足约束_月回撤3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"), index=False)
else:
    # 若无解，放宽提示最优的接近解
    print("无解：放宽到5%看最优接近解", flush=True)
    g = df_out[df_out['交易次数'] > 0].sort_values('月内最大回撤%').head(20)
    print(g.to_string(index=False, max_colwidth=16), flush=True)
print(f"结果文件: {out_path}", flush=True)
