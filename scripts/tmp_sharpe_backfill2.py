# -*- coding: utf-8 -*-
"""夏普回填第二批：22条历史记录复原回测（2025全年6 + 均衡激进6 + 山寨Top5 + 最终推荐BTC1 + 稳健4）
每条带累计收益核对：|计算值-记录值| 相对偏差>20% 的标记 SKIP，不写入"""
import os, sys, io, time, warnings, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7892")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7892")

import numpy as np
import pandas as pd
from data_fetcher import BinanceDataFetcher
from indicators import TechnicalIndicators
from backtest_engine import BacktestEngine

YEARS = [("2023-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31"),
         ("2025-01-01", "2025-12-31"), ("2026-01-01", "2026-08-26")]
RSI = {"rsi": True, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70}
KDJ = {"kdj": True, "kdj_k_period": 9, "kdj_d_period": 3, "kdj_j_period": 3,
       "kdj_buy_threshold": 20, "kdj_sell_threshold": 80}
MACD = {"macd": True, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9}
EMA = {"ema": True, "ema_short": 12, "ema_long": 26, "ema_periods": [12, 26]}
DMA = {"ma_cross": True, "ma_cross_short": 10, "ma_cross_long": 30, "ma_cross_periods": [10, 30]}
BOLL = {"boll": True, "bb_period": 20, "bb_std": 2.0}
def merge(*ds):
    m = {}
    for d in ds:
        m.update(d)
    return m

W25 = [("2025-01-01", "2025-12-31")]
W26 = [("2026-01-01", "2026-08-26")]
W26F = [("2026-01-01", "2026-08-30")]   # 美股代币按数据自然起点
# (key, 币, 市场, 周期, 指标, 模式LO/LS, tp, sl, 窗口, 记录收益%)
CONFIGS = [
    # A: 2025全年回测（交叉验证）
    ('AVAX_25', 'AVAX', 'future', '4h', merge(EMA, MACD), 'LS', 0.05, 0.05, W25, 214.4),
    ('BCH_25', 'BCH', 'future', '4h', merge(RSI, EMA), 'LS', None, None, W25, 337.2),
    ('BTC_25', 'BTC', 'future', '4h', BOLL, 'LS', None, None, W25, 65.4),
    ('LTC_25', 'LTC', 'future', '1h', RSI, 'LS', None, None, W25, 293.1),
    ('NEAR_25', 'NEAR', 'future', '4h', RSI, 'LS', None, None, W25, 134.6),
    ('XRP_25', 'XRP', 'future', '4h', RSI, 'LS', None, None, W25, 230.7),
    # B: 均衡/激进型（2026）
    ('MUB_LS', 'MUB', 'spot', '4h', KDJ, 'LS', None, None, W26F, 201.8),
    ('TQQQ_RM', 'TQQQ', 'future', '4h', merge(RSI, MACD), 'LS', None, None, W26F, 42.3),
    ('TQQQ_M', 'TQQQ', 'future', '4h', MACD, 'LS', 0.05, 0.05, W26F, 32.3),
    ('MUUB_LS', 'MUUB', 'spot', '4h', KDJ, 'LS', None, None, W26F, 136.8),
    ('TON_AGG', 'TON', 'future', '4h', MACD, 'LS', None, None, W26, 189.6),
    ('TQQQ_1H', 'TQQQ', 'future', '1h', KDJ, 'LS', None, None, W26F, 54.5),
    # C: 山寨扫描Top（四年）
    ('ICP_DMA', 'ICP', 'future', '4h', DMA, 'LS', None, None, YEARS, 330.9),
    ('LINK_BB', 'LINK', 'future', '4h', BOLL, 'LO', None, None, YEARS, 175.9),
    ('LTC_RSI', 'LTC', 'future', '4h', RSI, 'LS', None, None, YEARS, 311.2),
    ('XLM_LS', 'XLM', 'future', '4h', EMA, 'LS', None, None, YEARS, 815.4),
    # D: 最终推荐 BTC（四年，8/5 执行）
    ('BTC_KDJ', 'BTC', 'future', '4h', KDJ, 'LO', 0.08, 0.05, YEARS, 19.2),
    # E: 稳健型（2026）
    ('ETH_26', 'ETH', 'future', '4h', KDJ, 'LO', 0.05, 0.05, W26, 32.2),
    ('NEAR_E26', 'NEAR', 'future', '4h', EMA, 'LO', 0.02, 0.02, W26, 32.9),
    ('NEAR_EM26', 'NEAR', 'future', '4h', merge(EMA, MACD), 'LO', 0.03, 0.03, W26, 49.6),
    ('TON_26', 'TON', 'future', '4h', MACD, 'LS', 0.05, 0.05, W26, 50.1),
]

fut = BinanceDataFetcher(); fut.set_market_type('future')
spot = BinanceDataFetcher(); spot.set_market_type('spot')
fetchers = {'future': fut, 'spot': spot}
sig_engine = BacktestEngine(signal_mode='or')
BARS_YEAR = {'4h': 2190, '1h': 8760}

def fetch(fm, sym, start, end, tf, tries=5):
    for i in range(tries):
        try:
            df = fm.fetch_historical_data(sym, start, end, tf)
            if df is not None and not df.empty and len(df) >= 100:
                return df
        except Exception:
            pass
        time.sleep(2 * (i + 1))
    return None

def _close(cash, units, entry, price, side, comm=0.001):
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm

def simulate(df, signals, tp, sl, mode, initial=10000.0, comm=0.001):
    cash = initial; units = 0.0; entry = 0.0; side = 0; n = 0
    eq_pts = []
    for i, (ts, row) in enumerate(df.iterrows()):
        price = row['close']
        if not np.isfinite(price) or price <= 0:
            continue
        sig = int(signals.iloc[i]) if i > 0 else 0
        if side != 0 and entry > 0:
            r = (price - entry) / entry if side > 0 else (entry - price) / entry
            if (tp and r >= tp) or (sl and r <= -sl):
                cash = _close(cash, units, entry, price, side)
                side = 0; units = 0; n += 1
                eq_pts.append((ts, cash)); continue
        eq = cash + (units * price if side > 0 else units * entry + (entry - price) * units if side < 0 else 0)
        if eq <= 0:
            return None
        eq_pts.append((ts, eq))
        if sig == 1 and side <= 0:
            if side < 0:
                cash = _close(cash, units, entry, price, side); n += 1; side = 0; units = 0
            u = (cash * 0.95) / (price * (1 + comm))
            if u > 0:
                cash -= u * price * (1 + comm); units = u; entry = price; side = 1
        elif sig == -1 and side >= 0:
            if side > 0:
                cash = _close(cash, units, entry, price, side); n += 1; side = 0; units = 0
            if mode == 'LS':
                u = (cash * 0.95) / price
                if u > 0:
                    cash -= u * price * (1 + comm); units = u; entry = price; side = -1
    if side != 0 and len(df) > 0:
        cash = _close(cash, units, entry, df['close'].iloc[-1], side)
        n += 1
        eq_pts.append((df.index[-1], cash))
    if n == 0:
        return None
    return pd.Series(dict(eq_pts)).sort_index()

def sharpe_of(eq, tf):
    if eq is None or len(eq) < 30:
        return None
    r = eq.pct_change().dropna()
    if len(r) < 20 or r.std() <= 0:
        return None
    return float(r.mean() / r.std() * np.sqrt(BARS_YEAR[tf]))

out, skipped = {}, []
for key, coin, mkt, tf, ip, md, tp, sl, windows, ret_rec in CONFIGS:
    fm = fetchers[mkt]
    sym = f"{coin}/USDT:USDT" if mkt == 'future' else f"{coin}/USDT"
    eq_parts, mult = [], 1.0
    try:
        for s, e in windows:
            df = fetch(fm, sym, s, e, tf)
            if df is None:
                continue
            dft = TechnicalIndicators.calculate_all_indicators(df, ip)
            signals = sig_engine.calculate_signals(dft, ip)
            eq = simulate(df, signals, tp, sl, md)
            if eq is None:
                continue
            eq_parts.append(eq * mult)
            mult *= eq.iloc[-1] / 10000.0
        if not eq_parts:
            raise RuntimeError('无有效数据')
        full = pd.concat(eq_parts)
        full = full[~full.index.duplicated(keep='last')].sort_index()
        sh = sharpe_of(full, tf)
        ret_calc = (mult - 1) * 100
        dev = abs(ret_calc - ret_rec) / max(abs(ret_rec), 1)
        ok = dev <= 0.20
        if ok and sh is not None:
            out[key] = round(sh, 2)
            print(f"{key:<10} 夏普={out[key]:>6}  计算收益{ret_calc:+8.1f}% vs 记录{ret_rec:+8.1f}%  偏差{dev*100:.0f}%  OK", flush=True)
        else:
            skipped.append((key, ret_calc, ret_rec, sh))
            print(f"{key:<10} 夏普={'None' if sh is None else round(sh,2):>6}  计算收益{ret_calc:+8.1f}% vs 记录{ret_rec:+8.1f}%  偏差{dev*100:.0f}%  SKIP", flush=True)
    except Exception as ex:
        skipped.append((key, None, ret_rec, None))
        print(f"{key:<10} 异常 {type(ex).__name__} {str(ex)[:60]}", flush=True)

with open(os.path.join('scripts', 'results', 'sharpe_backfill2.json'), 'w', encoding='utf-8') as f:
    json.dump({'ok': out, 'skipped': [list(map(str, s)) for s in skipped]}, f, ensure_ascii=False, indent=1)
print(f"\n完成 {len(out)}/{len(CONFIGS)}，SKIP {len(skipped)} → scripts/results/sharpe_backfill2.json")
