# -*- coding: utf-8 -*-
"""
寻找「日均收益最高且稳定」的策略：加密货币 + 美股代币(bStocks)，做多/做空双向模拟
目标参考: 日均 1%（现实提示: 1%/日复利≈年化3678%，仅作寻优方向，不承诺达到）
"""
import os, sys, io, warnings, time
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

START = "2025-01-01"; END = "2025-12-31"
INITIAL = 10000.0; COMM = 0.001
# 市值靠前的加密货币 USDT 永续（非 bStocks，长期存续）
SYMBOLS = [f"{b}/USDT:USDT" for b in
           ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "TRX",
            "AVAX", "LINK", "DOT", "LTC", "BCH", "NEAR", "TON"]]
TIMEFRAMES = ["1h", "4h"]
TPSL = [(0.02, 0.02), (0.03, 0.03), (0.05, 0.05), (None, None)]
MODES = ["long_short", "long_only"]

BASE = {
    "RSI": {"rsi": True, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70},
    "KDJ": {"kdj": True, "kdj_k_period": 9, "kdj_d_period": 3, "kdj_j_period": 3,
            "kdj_buy_threshold": 20, "kdj_sell_threshold": 80},
    "布林带": {"boll": True, "bb_period": 20, "bb_std": 2.0},
    "EMA": {"ema": True, "ema_short": 12, "ema_long": 26, "ema_periods": [12, 26]},
    "MACD": {"macd": True, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
    "双均线交叉": {"ma_cross": True, "ma_cross_short": 10, "ma_cross_long": 30,
                  "ma_cross_periods": [10, 30]},
}
COMBOS = dict(BASE)
for pair in [("RSI", "KDJ"), ("KDJ", "MACD"), ("RSI", "MACD"), ("EMA", "MACD"), ("RSI", "EMA")]:
    merged = {}
    for k in pair:
        merged.update(BASE[k])
    COMBOS["+".join(pair)] = merged

def _close(cash, units, entry, price, side, comm):
    """平仓返回新现金"""
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm

def simulate(df, signals, tp, sl, mode, initial=INITIAL, comm=COMM):
    """双向模拟(1倍保证金做空): +1信号开多/平空, -1信号平多/开空(long_short)或仅平多(long_only)"""
    cash = initial; units = 0.0; entry = 0.0; side = 0
    eq_pts = []; n_trades = 0
    for i, (ts, row) in enumerate(df.iterrows()):
        price = row['close']
        if not np.isfinite(price) or price <= 0:
            continue
        sig = int(signals.iloc[i]) if i > 0 else 0
        # 止盈止损(按收盘价判断)
        if side != 0 and entry > 0:
            r = (price - entry) / entry if side > 0 else (entry - price) / entry
            if (tp and r >= tp) or (sl and r <= -sl):
                cash = _close(cash, units, entry, price, side, comm)
                side = 0; units = 0; entry = 0; n_trades += 1
                eq_pts.append((ts, cash))
                continue
        # 权益: 多头=cash+市值; 空头=cash+锁定保证金+浮动盈亏
        if side > 0:
            eq = cash + units * price
        elif side < 0:
            eq = cash + units * entry + (entry - price) * units
        else:
            eq = cash
        if eq <= 0:  # 爆仓: 归零终止
            eq_pts.append((ts, 0.0))
            return None
        eq_pts.append((ts, eq))
        if sig == 1 and side <= 0:
            if side < 0:
                cash = _close(cash, units, entry, price, side, comm)
                n_trades += 1; side = 0; units = 0
            invest = cash * 0.95
            u = invest / (price * (1 + comm))
            if u > 0:
                cash -= u * price * (1 + comm)
                units = u; entry = price; side = 1
        elif sig == -1 and side >= 0:
            if side > 0:
                cash = _close(cash, units, entry, price, side, comm)
                n_trades += 1; side = 0; units = 0
            if mode == "long_short":
                notional = cash * 0.95
                u = notional / price
                if u > 0:
                    cash -= u * price * (1 + comm)  # 锁定全额保证金+手续费
                    units = u; entry = price; side = -1
    # 收盘强制平仓
    if side != 0 and len(df) > 0:
        cash = _close(cash, units, entry, df['close'].iloc[-1], side, comm)
        n_trades += 1
        eq_pts.append((df.index[-1], cash))
    eq = pd.Series(dict(eq_pts)).sort_index()
    if eq.empty:
        return None
    daily = eq.groupby(eq.index.date).last()
    rets = daily.pct_change().dropna() * 100
    peak = eq.expanding().max(); mdd = ((eq - peak) / peak * 100).min()
    return {
        'total_ret': (eq.iloc[-1] / initial - 1) * 100,
        'avg_daily': rets.mean() if len(rets) else 0.0,
        'std_daily': rets.std() if len(rets) else 0.0,
        'worst_day': rets.min() if len(rets) else 0.0,
        'pos_days': f"{int((rets > 0).sum())}/{len(rets)}",
        'mdd': mdd, 'trades': n_trades,
    }

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "daily_return_2025")
os.makedirs(out_dir, exist_ok=True)
fetcher = BinanceDataFetcher(); fetcher.set_market_type('future')
sig_engine = BacktestEngine(signal_mode='or')

rows = []
def fetch_with_retry(fetcher, s, tf, tries=6):
    """带重试的数据拉取：代理不稳定时指数退避"""
    for i in range(tries):
        df = fetcher.fetch_historical_data(s, START, END, tf)
        if df is not None and not df.empty and len(df) >= 100:
            return df
        time.sleep(2 * (i + 1))
    return df

for s in SYMBOLS:
    for tf in TIMEFRAMES:
        df = fetch_with_retry(fetcher, s, tf)
        if df is None or df.empty or len(df) < 100:
            print(f"{s} {tf}: 数据不足，跳过", flush=True)
            continue
        print(f"{s} {tf}: {len(df)}根", flush=True)
        for name, ip in COMBOS.items():
            dft = TechnicalIndicators.calculate_all_indicators(df, ip)
            signals = sig_engine.calculate_signals(dft, ip)
            if signals.abs().sum() == 0:
                continue
            for tp, sl in TPSL:
                for mode in MODES:
                    m = simulate(df, signals, tp, sl, mode)
                    if m is None or m['trades'] == 0:
                        continue
                    if m['total_ret'] > 1000 or m['mdd'] <= -100:  # 异常保护
                        continue
                    rows.append({'品种': s, '周期': tf, '模式': mode, '策略': name,
                                 '止盈%': tp * 100 if tp else 0, '止损%': sl * 100 if sl else 0,
                                 '总收益率%': round(m['total_ret'], 2),
                                 '日均收益率%': round(m['avg_daily'], 3),
                                 '日波动%': round(m['std_daily'], 2),
                                 '最差日%': round(m['worst_day'], 2),
                                 '盈利日': m['pos_days'], '最大回撤%': round(m['mdd'], 2),
                                 '交易次数': m['trades']})

out = pd.DataFrame(rows)
out.to_excel(os.path.join(out_dir, f"daily_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"), index=False)
print(f"\n✅ 完成，共 {len(out)} 条", flush=True)
top = out.sort_values('日均收益率%', ascending=False).head(25)
print("\n=== 日均收益率 Top 25 ===", flush=True)
print(top.to_string(index=False, max_colwidth=12), flush=True)
stable = out[out['日波动%'] < 2].sort_values('日均收益率%', ascending=False).head(15)
print("\n=== 稳定组(日波动<2%) Top 15 ===", flush=True)
print(stable.to_string(index=False, max_colwidth=12), flush=True)
