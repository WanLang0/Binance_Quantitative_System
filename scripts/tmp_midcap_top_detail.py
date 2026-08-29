# -*- coding: utf-8 -*-
"""Midcap Top配置细节验证：年度回撤 + 月度盈亏 + 逐月明细（重点看四年全正的 ICP/LTC 与 XLM 巨额收益的来源）"""
import os, sys, io, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
os.environ.setdefault("HTTP_PROXY", "http://192.168.11.188:7892")
os.environ.setdefault("HTTPS_PROXY", "http://192.168.11.188:7892")

import pandas as pd
import numpy as np
from data_fetcher import BinanceDataFetcher
from indicators import TechnicalIndicators
from backtest_engine import BacktestEngine

YEARS = [("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
         ("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-08-26")]
INITIAL = 10000.0; COMM = 0.001

# 主扫描的Top配置: (币, 策略参数, 模式, tp, sl, 备注)
CONFIGS = [
    ("XLM", {"ema": True, "ema_short": 12, "ema_long": 26, "ema_periods": [12, 26]}, "long_short", None, None, "EMA双向 无止盈止损"),
    ("ICP", {"ma_cross": True, "ma_cross_short": 10, "ma_cross_long": 30,
             "ma_cross_periods": [10, 30]}, "long_short", None, None, "双均线双向 无止盈止损"),
    ("LTC", {"rsi": True, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70}, "long_short", None, None, "RSI双向 无止盈止损"),
    ("BCH", {"ema": True, "ema_short": 12, "ema_long": 26, "ema_periods": [12, 26]}, "long_only", None, None, "EMA仅多 无止盈止损"),
    ("XLM", {"ema": True, "ema_short": 12, "ema_long": 26, "ema_periods": [12, 26]}, "long_short", 0.05, 0.05, "EMA双向 5/5"),
]

def _close(cash, units, entry, price, side, comm):
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm

def simulate_detail(df, signals, tp, sl, mode, initial=INITIAL, comm=COMM):
    """返回逐K线权益序列与逐笔交易记录（单笔收益=平仓后权益/开仓前权益-1）"""
    cash = initial; units = 0.0; entry = 0.0; side = 0
    eq_pts = []; trades = []; n = 0; open_eq = initial  # open_eq=开仓前全部现金
    for i, (ts, row) in enumerate(df.iterrows()):
        price = row['close']
        if not np.isfinite(price) or price <= 0:
            continue
        sig = int(signals.iloc[i]) if i > 0 else 0
        if side != 0 and entry > 0:
            r = (price - entry) / entry if side > 0 else (entry - price) / entry
            if (tp and r >= tp) or (sl and r <= -sl):
                cash = _close(cash, units, entry, price, side, comm)
                trades.append((ts, '多' if side > 0 else '空', (cash / open_eq - 1) * 100))
                side = 0; units = 0; entry = 0; n += 1
                eq_pts.append((ts, cash)); continue
        if side > 0:
            eq = cash + units * price
        elif side < 0:
            eq = cash + units * entry + (entry - price) * units
        else:
            eq = cash
        if eq <= 0:
            eq_pts.append((ts, 0.0)); break
        eq_pts.append((ts, eq))
        if sig == 1 and side <= 0:
            if side < 0:
                cash = _close(cash, units, entry, price, side, comm)
                trades.append((ts, '空', (cash / open_eq - 1) * 100))
                n += 1; side = 0; units = 0
            invest = cash * 0.95
            u = invest / (price * (1 + comm))
            if u > 0:
                open_eq = cash
                cash -= u * price * (1 + comm); units = u; entry = price; side = 1
        elif sig == -1 and side >= 0:
            if side > 0:
                cash = _close(cash, units, entry, price, side, comm)
                trades.append((ts, '多', (cash / open_eq - 1) * 100))
                n += 1; side = 0; units = 0
            if mode == "long_short":
                u = (cash * 0.95) / price
                if u > 0:
                    open_eq = cash
                    cash -= u * price * (1 + comm); units = u; entry = price; side = -1
    if side != 0 and len(df) > 0:
        cash = _close(cash, units, entry, df['close'].iloc[-1], side, comm)
        trades.append((df.index[-1], '多' if side > 0 else '空', (cash / open_eq - 1) * 100))
        n += 1; eq_pts.append((df.index[-1], cash))
    return pd.Series(dict(eq_pts)).sort_index(), trades

fetcher = BinanceDataFetcher(); fetcher.set_market_type('future')
sig_engine = BacktestEngine(signal_mode='or')

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

for coin, ip, mode, tp, sl, title in CONFIGS:
    sym = f"{coin}/USDT:USDT"
    w("\n" + "=" * 90)
    w(f"◆ {coin} {title}")
    cum = 1.0; all_trades = []
    for year, start, end in YEARS:
        df = fetch(sym, start, end)
        if df is None:
            w(f"  {year}: 数据失败"); continue
        dft = TechnicalIndicators.calculate_all_indicators(df, ip)
        signals = sig_engine.calculate_signals(dft, ip)
        eq, trades = simulate_detail(df, signals, tp, sl, mode)
        all_trades += trades
        peak = eq.cummax(); mdd = ((eq - peak) / peak * 100).min()
        yr = (eq.iloc[-1] / INITIAL - 1) * 100
        cum *= (1 + yr / 100)
        monthly = eq.groupby(eq.index.to_period('M')).last()
        prev = INITIAL; pos_m = 0; mrows = []
        for p, v in monthly.items():
            r = (v / prev - 1) * 100
            if r > 0: pos_m += 1
            mrows.append(f"{p.month}月{r:+.0f}%"); prev = v
        tr = pd.DataFrame(trades, columns=['ts', 'side', 'ret'])
        wr = (tr['ret'] > 0).mean() * 100 if len(tr) else 0
        long_pnl = tr[tr['side'] == '多']['ret'].sum() if len(tr) else 0
        short_pnl = tr[tr['side'] == '空']['ret'].sum() if len(tr) else 0
        w(f"  {year}: {yr:+7.1f}% | 回撤{mdd:6.1f}% | {len(tr)}笔 胜率{wr:.0f}% | "
          f"多头贡献{long_pnl:+.1f}pp 空头贡献{short_pnl:+.1f}pp | 盈利月{pos_m}/{len(monthly)}")
        w("    月度: " + " ".join(mrows))
    w(f"  ★ 四年累计: {(cum-1)*100:+.1f}%")

out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "midcap_top_detail.txt")
with open(out_file, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))
print(f"\nSaved -> {out_file}", flush=True)
