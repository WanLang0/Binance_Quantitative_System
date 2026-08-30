# -*- coding: utf-8 -*-
"""交易明细回填：重跑50条已验证夏普的配置，抓取每笔交易（开/平仓时间、方向、数量、价格、单笔收益）
输出 scripts/results/trades_detail.json，键为 symbol|strategy|mode|tpsl|timeframe（迁移按此匹配入库）
口径与夏普回填完全一致（mc山寨/实盘不执行tp/sl，其余执行）"""
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
W26F = [("2026-01-01", "2026-08-30")]

def cfg(sym, strat, mode, tpsl, coin, mkt, tf, ip, md, tp, sl, windows, use_tpsl):
    return (sym, strat, mode, tpsl, coin, mkt, tf, ip, md, tp, sl, windows, use_tpsl)

MIDCAP = ["DOGE", "TRX", "LINK", "DOT", "BCH", "LTC", "UNI", "APT", "ICP", "XLM"]
CONFIGS = [
    # ---- 第一批（夏普已验证31条）----
    cfg('ETH/USDT', 'KDJ(9,3,3) K上穿D且K<20买 / K下穿D且K>80卖', '仅做多', '8%/5%',
        'ETH', 'future', '4h', KDJ, 'LO', 0.08, 0.05, YEARS, True),
    cfg('XLM/USDT', 'EMA(12/26)', '双向', '5%/5%', 'XLM', 'future', '4h', EMA, 'LS', 0.05, 0.05, YEARS, True),
    cfg('AAPL/USDT:USDT', '双均线', '仅做多', '不设', 'AAPL', 'future', '4h', DMA, 'LO', None, None, W26F, True),
    cfg('MSFT/USDT:USDT', 'EMA', '双向', '不设', 'MSFT', 'future', '4h', EMA, 'LS', None, None, W26F, True),
    cfg('NVDA/USDT:USDT', 'RSI+MACD', '仅做多', '不设', 'NVDA', 'future', '4h', merge(RSI, MACD), 'LO', None, None, W26F, True),
    cfg('GOOGL/USDT:USDT', '双均线', '仅做多', '不设', 'GOOGL', 'future', '4h', DMA, 'LO', None, None, W26F, True),
    cfg('AMZN/USDT:USDT', 'EMA', '仅做多', '不设', 'AMZN', 'future', '4h', EMA, 'LO', None, None, W26F, True),
    cfg('META/USDT:USDT', 'RSI', '仅做多', '5%/5%', 'META', 'future', '4h', RSI, 'LO', 0.05, 0.05, W26F, True),
    cfg('TSLA/USDT:USDT', 'EMA', '双向', '5%/5%', 'TSLA', 'future', '4h', EMA, 'LS', 0.05, 0.05, W26F, True),
    cfg('MUB/USDT', 'KDJ', '仅做多', '不设', 'MUB', 'spot', '4h', KDJ, 'LO', None, None, W26F, False),
    cfg('MUUB/USDT', 'KDJ', '仅做多', '不设', 'MUUB', 'spot', '4h', KDJ, 'LO', None, None, W26F, False),
]
for c in MIDCAP:
    CONFIGS.append(cfg(f'{c}/USDT', 'EMA(12/26)', '仅做多', '不设', c, 'future', '4h', EMA, 'LO', None, None, YEARS, False))
    CONFIGS.append(cfg(f'{c}/USDT', 'KDJ(9,3,3)', '仅做多', '8%/5%', c, 'future', '4h', KDJ, 'LO', 0.08, 0.05, YEARS, False))
# ---- 第二批（夏普已验证19条）----
CONFIGS += [
    cfg('AVAX/USDT', 'EMA+MACD', '双向', '5%/5%', 'AVAX', 'future', '4h', merge(EMA, MACD), 'LS', 0.05, 0.05, W25, True),
    cfg('BCH/USDT', 'RSI+EMA', '双向', '不设', 'BCH', 'future', '4h', merge(RSI, EMA), 'LS', None, None, W25, True),
    cfg('BTC/USDT', '布林带', '双向', '不设', 'BTC', 'future', '4h', BOLL, 'LS', None, None, W25, True),
    cfg('LTC/USDT', 'RSI', '双向', '不设', 'LTC', 'future', '1h', RSI, 'LS', None, None, W25, True),
    cfg('NEAR/USDT', 'RSI', '双向', '不设', 'NEAR', 'future', '4h', RSI, 'LS', None, None, W25, True),
    cfg('XRP/USDT', 'RSI', '双向', '不设', 'XRP', 'future', '4h', RSI, 'LS', None, None, W25, True),
    cfg('MUB/USDT', 'KDJ', '双向(模拟)', '不设', 'MUB', 'spot', '4h', KDJ, 'LS', None, None, W26F, False),
    cfg('TQQQ/USDT', 'RSI+MACD', '双向', '不设', 'TQQQ', 'future', '4h', merge(RSI, MACD), 'LS', None, None, W26F, True),
    cfg('TQQQ/USDT', 'MACD', '双向', '5%/5%', 'TQQQ', 'future', '4h', MACD, 'LS', 0.05, 0.05, W26F, True),
    cfg('TQQQ/USDT', 'KDJ', '双向', '不设', 'TQQQ', 'future', '1h', KDJ, 'LS', None, None, W26F, True),
    cfg('TON/USDT', 'MACD', '双向', '不设', 'TON', 'future', '4h', MACD, 'LS', None, None, W26, True),
    cfg('TON/USDT', 'MACD', '双向', '5%/5%', 'TON', 'future', '4h', MACD, 'LS', 0.05, 0.05, W26, True),
    cfg('ICP/USDT', '双均线(10/30) 双向 不设', '双向', '不设', 'ICP', 'future', '4h', DMA, 'LS', None, None, YEARS, True),
    cfg('LINK/USDT', '布林带 仅多 不设', '仅做多', '不设', 'LINK', 'future', '4h', BOLL, 'LO', None, None, YEARS, True),
    cfg('LTC/USDT', 'RSI(14) 双向 不设', '双向', '不设', 'LTC', 'future', '4h', RSI, 'LS', None, None, YEARS, True),
    cfg('XLM/USDT', 'EMA(12/26) 双向 不设', '双向', '不设', 'XLM', 'future', '4h', EMA, 'LS', None, None, YEARS, True),
    cfg('BCH/USDT', 'EMA 仅多 不设', '仅做多', '不设', 'BCH', 'future', '4h', EMA, 'LO', None, None, YEARS, False),
    cfg('NEAR/USDT', 'EMA', '仅做多', '2%/2%', 'NEAR', 'future', '4h', EMA, 'LO', 0.02, 0.02, W26, True),
    cfg('NEAR/USDT', 'EMA+MACD', '仅做多', '3%/3%', 'NEAR', 'future', '4h', merge(EMA, MACD), 'LO', 0.03, 0.03, W26, True),
]

fut = BinanceDataFetcher(); fut.set_market_type('future')
spot = BinanceDataFetcher(); spot.set_market_type('spot')
fetchers = {'future': fut, 'spot': spot}
sig_engine = BacktestEngine(signal_mode='or')

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

def simulate_trades(df, signals, tp, sl, mode, use_tpsl, initial=10000.0, comm=0.001):
    """同夏普口径模拟，同时记录每笔交易"""
    cash = initial; units = 0.0; entry = 0.0; side = 0
    trades, cur = [], None  # cur = [t_in, p_in, side, qty]
    for i, (ts, row) in enumerate(df.iterrows()):
        price = row['close']
        if not np.isfinite(price) or price <= 0:
            continue
        sig = int(signals.iloc[i]) if i > 0 else 0
        t = ts.strftime('%Y-%m-%d %H:%M')
        if use_tpsl and side != 0 and entry > 0:
            r = (price - entry) / entry if side > 0 else (entry - price) / entry
            if (tp and r >= tp) or (sl and r <= -sl):
                before = cash
                cash = _close(cash, units, entry, price, side)
                trades.append({'in': cur[0], 'pin': cur[1], 'side': cur[2], 'qty': round(cur[3], 6),
                               'out': t, 'pout': price, 'ret': round(((price * 0.999) / (cur[1] * 1.001) - 1) * 100, 2) if side > 0 else round((cur[1] - price * 1.001) / (cur[1] * 1.001) * 100, 2)})
                side = 0; units = 0; cur = None
                continue
        eq = cash + (units * price if side > 0 else units * entry + (entry - price) * units if side < 0 else 0)
        if eq <= 0:
            return None
        if sig == 1 and side <= 0:
            if side < 0:
                before = cash
                cash = _close(cash, units, entry, price, side)
                trades.append({'in': cur[0], 'pin': cur[1], 'side': cur[2], 'qty': round(cur[3], 6),
                               'out': t, 'pout': price, 'ret': round(((price * 0.999) / (cur[1] * 1.001) - 1) * 100, 2) if side > 0 else round((cur[1] - price * 1.001) / (cur[1] * 1.001) * 100, 2)})
                side = 0; units = 0; cur = None
            u = (cash * 0.95) / (price * (1 + comm))
            if u > 0:
                cash -= u * price * (1 + comm); units = u; entry = price; side = 1
                cur = [t, price, '多', u]
        elif sig == -1 and side >= 0:
            if side > 0:
                before = cash
                cash = _close(cash, units, entry, price, side)
                trades.append({'in': cur[0], 'pin': cur[1], 'side': cur[2], 'qty': round(cur[3], 6),
                               'out': t, 'pout': price, 'ret': round(((price * 0.999) / (cur[1] * 1.001) - 1) * 100, 2) if side > 0 else round((cur[1] - price * 1.001) / (cur[1] * 1.001) * 100, 2)})
                side = 0; units = 0; cur = None
            if mode == 'LS':
                u = (cash * 0.95) / price
                if u > 0:
                    cash -= u * price * (1 + comm); units = u; entry = price; side = -1
                    cur = [t, price, '空', u]
    if side != 0 and len(df) > 0:
        before = cash
        cash = _close(cash, units, entry, df['close'].iloc[-1], side)
        trades.append({'in': cur[0], 'pin': cur[1], 'side': cur[2], 'qty': round(cur[3], 6),
                       'out': df.index[-1].strftime('%Y-%m-%d %H:%M'), 'pout': df['close'].iloc[-1],
                       'ret': round(((price * 0.999) / (cur[1] * 1.001) - 1) * 100, 2) if side > 0 else round((cur[1] - price * 1.001) / (cur[1] * 1.001) * 100, 2)})
    return trades, cash

out = {}
for sym, strat, mode, tpsl, coin, mkt, tf, ip, md, tp, sl, windows, use_tpsl in CONFIGS:
    fm = fetchers[mkt]
    usym = f"{coin}/USDT:USDT" if mkt == 'future' else f"{coin}/USDT"
    all_trades, mult = [], 1.0
    try:
        for s, e in windows:
            df = fetch(fm, usym, s, e, tf)
            if df is None:
                continue
            dft = TechnicalIndicators.calculate_all_indicators(df, ip)
            signals = sig_engine.calculate_signals(dft, ip)
            r = simulate_trades(df, signals, tp, sl, md, use_tpsl, initial=10000.0 * mult)
            if r is None:
                continue
            tr, cash = r
            all_trades.extend(tr)
            mult = cash / 10000.0
        key = f"{sym}|{strat}|{mode}|{tpsl}|{tf}"
        out[key] = {'trades': all_trades, 'n': len(all_trades), 'ret_total': round((mult - 1) * 100, 1)}
        print(f"{key[:56]:58} {len(all_trades):>4}笔  累计{(mult - 1) * 100:+9.1f}%", flush=True)
    except Exception as ex:
        print(f"{sym} {strat[:20]} 异常 {type(ex).__name__} {str(ex)[:60]}", flush=True)

path = os.path.join('scripts', 'results', 'trades_detail.json')
with open(path, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False)
total = sum(v['n'] for v in out.values())
print(f"\n完成 {len(out)}/{len(CONFIGS)} 配置，共 {total} 笔交易 → {path}")
