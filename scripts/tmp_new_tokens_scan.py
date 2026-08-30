# -*- coding: utf-8 -*-
"""UNITREE/CXMT 扫描入库：7策略×2模式×2止盈止损=28组/币（合约口径，同七姐妹），
Top配置入库（ret/夏普/胜率/交易流水/日期精确到日），全量结果存 results"""
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
import strategies_store as store

COINS = ['UNITREE', 'CXMT']
MODES = ['long_short', 'long_only']
TPSL = [(None, None), (0.05, 0.05)]
TPSL_NAME = {(None, None): '不设', (0.05, 0.05): '5%/5%'}
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
COMBOS = {'KDJ': KDJ, 'RSI': RSI, 'MACD': MACD, 'EMA': EMA, '双均线': DMA, '布林带': BOLL,
          'RSI+MACD': merge(RSI, MACD)}

fut = BinanceDataFetcher(); fut.set_market_type('future')
sig_engine = BacktestEngine(signal_mode='or')

def fetch(sym, tries=5):
    for i in range(tries):
        try:
            df = fut.fetch_historical_data(sym, '2026-01-01', '2026-08-30', '4h')
            if df is not None and len(df) >= 40:
                return df
        except Exception:
            pass
        time.sleep(3)
    return None

def _close(cash, units, entry, price, side, comm=0.001):
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm

def simulate(df, signals, tp, sl, mode, initial=10000.0, comm=0.001):
    cash = initial; units = 0.0; entry = 0.0; side = 0; n = 0
    eq_pts, trades, cur = [], [], None
    for i, (ts, row) in enumerate(df.iterrows()):
        price = row['close']
        if not np.isfinite(price) or price <= 0:
            continue
        sig = int(signals.iloc[i]) if i > 0 else 0
        t = ts.strftime('%Y-%m-%d %H:%M')
        if side != 0 and entry > 0:
            r = (price - entry) / entry if side > 0 else (entry - price) / entry
            if (tp and r >= tp) or (sl and r <= -sl):
                before = cash
                cash = _close(cash, units, entry, price, side)
                trades.append({'in': cur[0], 'pin': cur[1], 'side': cur[2], 'qty': round(cur[3], 6),
                               'out': t, 'pout': price,
                               'ret': round(((price * 0.999) / (cur[1] * 1.001) - 1) * 100, 2) if side > 0 else round((cur[1] - price * 1.001) / (cur[1] * 1.001) * 100, 2)})
                side = 0; units = 0; n += 1; cur = None
                eq_pts.append((ts, cash)); continue
        eq = cash + (units * price if side > 0 else units * entry + (entry - price) * units if side < 0 else 0)
        if eq <= 0:
            return None
        eq_pts.append((ts, eq))
        if sig == 1 and side <= 0:
            if side < 0:
                before = cash
                cash = _close(cash, units, entry, price, side)
                trades.append({'in': cur[0], 'pin': cur[1], 'side': cur[2], 'qty': round(cur[3], 6),
                               'out': t, 'pout': price,
                               'ret': round(((price * 0.999) / (cur[1] * 1.001) - 1) * 100, 2) if side > 0 else round((cur[1] - price * 1.001) / (cur[1] * 1.001) * 100, 2)})
                n += 1; side = 0; units = 0; cur = None
            u = (cash * 0.95) / (price * (1 + comm))
            if u > 0:
                cash -= u * price * (1 + comm); units = u; entry = price; side = 1
                cur = [t, price, '多', u]
        elif sig == -1 and side >= 0:
            if side > 0:
                before = cash
                cash = _close(cash, units, entry, price, side)
                trades.append({'in': cur[0], 'pin': cur[1], 'side': cur[2], 'qty': round(cur[3], 6),
                               'out': t, 'pout': price,
                               'ret': round(((price * 0.999) / (cur[1] * 1.001) - 1) * 100, 2) if side > 0 else round((cur[1] - price * 1.001) / (cur[1] * 1.001) * 100, 2)})
                n += 1; side = 0; units = 0; cur = None
            if mode == 'long_short':
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
        n += 1
        eq_pts.append((df.index[-1], cash))
    if n == 0:
        return None
    eq = pd.Series(dict(eq_pts)).sort_index()
    peak = eq.cummax()
    mdd = ((eq - peak) / peak * 100).min()
    rr = eq.pct_change().dropna()
    sh = float(rr.mean() / rr.std() * np.sqrt(2190)) if len(rr) >= 10 and rr.std() > 0 else None
    return {'ret': (eq.iloc[-1] / initial - 1) * 100, 'mdd': mdd, 'n': n, 'sh': sh, 'trades': trades}

results = {}
for coin in COINS:
    sym = f'{coin}/USDT:USDT'
    df = fetch(sym)
    if df is None:
        print(f"{coin}: 数据不足，跳过"); continue
    days = (df.index[-1] - df.index[0]).days
    per = f"{df.index[0]:%Y-%m-%d}~{df.index[-1]:%Y-%m-%d}"
    print(f"\n>>> {coin}: {len(df)}根/{days}天 ({per})  标的自身{(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:+.1f}%")
    out = []
    for strat, ip in COMBOS.items():
        dft = TechnicalIndicators.calculate_all_indicators(df, ip)
        signals = sig_engine.calculate_signals(dft, ip)
        for mode in MODES:
            for tp, sl in TPSL:
                r = simulate(df, signals, tp, sl, mode)
                if r is None:
                    continue
                out.append({'coin': coin, 'sym': sym, 'strat': strat,
                            'mode': '双向' if mode == 'long_short' else '仅做多',
                            'tpsl': TPSL_NAME[(tp, sl)], 'period': per, 'days': days, **r})
    out.sort(key=lambda x: -x['ret'])
    results[coin] = out
    print(f"{'策略':<10}{'模式':<6}{'TP/SL':<6}{'收益':>9}{'回撤':>8}{'笔':>4}{'夏普':>7}")
    for o in out[:8]:
        print(f"{o['strat']:<10}{o['mode']:<6}{o['tpsl']:<6}{o['ret']:>+8.1f}%{o['mdd']:>7.1f}%"
              f"{o['n']:>4}{'' if o['sh'] is None else format(round(o['sh'], 2), '>7')}")

with open('scripts/results/new_tokens_scan.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=1, default=float)

# ---- Top1/币 入库（全链路：ret/夏普/胜率/流水/日期）----
SRC = '美股新股代币扫描（2026-08）'
for coin, out in results.items():
    if not out:
        continue
    top = out[0]
    wins = sum(1 for t in top['trades'] if t['ret'] > 0)
    winrate = f"{wins / len(top['trades']) * 100:.1f}%" if top['trades'] else '—'
    rec = {'symbol': top['sym'], 'timeframe': '4h', 'strategy': top['strat'],
           'mode': top['mode'], 'tpsl': top['tpsl'],
           'ret': f"{top['ret']:+.1f}%", 'daily': f"{top['ret'] / max(top['days'], 1):+.2f}%",
           'trades': str(top['n']), 'winrate': winrate, 'mdd': f"{top['mdd']:.1f}%",
           'stability': f"上市{top['days']}天{top['n']}笔",
           'market': '主网合约', 'source': SRC,
           'note': f"{coin}新股代币（bStocks中概），28组扫描冠军；样本仅{top['days']}天无统计意义，观察用",
           'sharpe': None if top['sh'] is None else f"{top['sh']:.2f}",
           'period': top['period']}
    ok, msg = store.add_history(rec)
    if ok:
        with store._conn() as c:
            row = c.execute("SELECT id FROM strategy_records WHERE symbol=? AND strategy=? AND mode=? "
                            "AND tpsl=? AND source=?", (top['sym'], top['strat'], top['mode'],
                                                        top['tpsl'], SRC)).fetchone()
            c.execute("UPDATE strategy_records SET trades_detail=? WHERE id=?",
                      (json.dumps({'trades': top['trades'], 'n': len(top['trades']),
                                   'ret_total': round(top['ret'], 1)}, ensure_ascii=False), row[0]))
    print(f"入库 {coin}: {top['strat']} {top['mode']} {top['tpsl']} {top['ret']:+.1f}% → {msg}")
print('\nsaved → scripts/results/new_tokens_scan.json')
