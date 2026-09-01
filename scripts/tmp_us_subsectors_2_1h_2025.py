# -*- coding: utf-8 -*-
"""美股细分行业板块龙头股 Yahoo Finance 1h 回测（第二批）：2025全年 + 2026至今

板块：机器人 / 军工 / 石油 / 天然气 / 黄金 / 生物医药 / 消费（并入必需消费）。
渠道与美股七姐妹一致：yfinance 拉美股 1h K线，覆盖 2025-01-01 ~ 今。
每标的独立扫描策略，仅做多口径收益最高者入库，标注渠道与所属板块。
"""
import os, sys, io, time, warnings, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7892")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7892")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from indicators import TechnicalIndicators
from backtest_engine import BacktestEngine
import strategies_store as store

CHANNEL = "Yahoo Finance 美股1h行情"
MARKET_LABEL = "美股真实股票(非币安币) 主网现货"

# 15 只细分行业板块龙头股（第二批）
TICKERS = ['SONY', 'AXON', 'RKLB', 'TTE', 'PSX', 'SHEL', 'COP',
           'NEM', 'FNV', 'GFI', 'MRK', 'ABBV', 'PFE', 'PEP', 'KO']
NAMES = {'SONY': '索尼', 'AXON': 'Axon Enterprise', 'RKLB': 'Rocket Lab',
         'TTE': 'TotalEnergies', 'PSX': 'Phillips 66', 'SHEL': '壳牌', 'COP': '康菲石油',
         'NEM': '纽蒙特', 'FNV': 'Franco-Nevada', 'GFI': '金田',
         'MRK': '默沙东', 'ABBV': '艾伯维', 'PFE': '辉瑞',
         'PEP': '百事', 'KO': '可口可乐'}
SECTORS = {'SONY': '机器人',
           'AXON': '军工', 'RKLB': '军工',
           'TTE': '石油', 'PSX': '石油', 'SHEL': '石油', 'COP': '天然气',
           'NEM': '黄金', 'FNV': '黄金', 'GFI': '黄金',
           'MRK': '生物医药', 'ABBV': '生物医药', 'PFE': '生物医药',
           'PEP': '必需消费', 'KO': '必需消费'}

MODES = ['long_only', 'long_short']
MODE_NAME = {'long_only': '仅做多', 'long_short': '双向(模拟)'}
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

sig_engine = BacktestEngine(signal_mode='or')


def fetch_1h(ticker, start='2025-01-01', tries=5):
    for i in range(tries):
        try:
            df = yf.download(ticker, interval='1h', start=start, end=datetime.now(),
                             progress=False, auto_adjust=True)
            if df is None or df.empty:
                time.sleep(3); continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.index.name = 'timestamp'
            df = df.dropna(subset=['close'])
            df = df[df['close'] > 0]
            if len(df) >= 200:
                return df
        except Exception as e:
            print('fetch err:', e)
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
                cash = _close(cash, units, entry, price, side)
                trades.append({'in': cur[0], 'pin': cur[1], 'side': cur[2], 'qty': round(cur[3], 6),
                               'out': t, 'pout': price,
                               'ret': round(((price * 0.999) / (cur[1] * 1.001) - 1) * 100, 2) if side > 0
                               else round((cur[1] - price * 1.001) / (cur[1] * 1.001) * 100, 2)})
                side = 0; units = 0; n += 1; cur = None
                eq_pts.append((ts, cash)); continue
        eq = cash + (units * price if side > 0 else units * entry + (entry - price) * units if side < 0 else 0)
        if eq <= 0:
            return None
        eq_pts.append((ts, eq))
        if sig == 1 and side <= 0:
            if side < 0:
                cash = _close(cash, units, entry, price, side)
                trades.append({'in': cur[0], 'pin': cur[1], 'side': cur[2], 'qty': round(cur[3], 6),
                               'out': t, 'pout': price,
                               'ret': round(((price * 0.999) / (cur[1] * 1.001) - 1) * 100, 2) if side > 0
                               else round((cur[1] - price * 1.001) / (cur[1] * 1.001) * 100, 2)})
                n += 1; side = 0; units = 0; cur = None
            u = (cash * 0.95) / (price * (1 + comm))
            if u > 0:
                cash -= u * price * (1 + comm); units = u; entry = price; side = 1
                cur = [t, price, '多', u]
        elif sig == -1 and side >= 0:
            if side > 0:
                cash = _close(cash, units, entry, price, side)
                trades.append({'in': cur[0], 'pin': cur[1], 'side': cur[2], 'qty': round(cur[3], 6),
                               'out': t, 'pout': price,
                               'ret': round(((price * 0.999) / (cur[1] * 1.001) - 1) * 100, 2) if side > 0
                               else round((cur[1] - price * 1.001) / (cur[1] * 1.001) * 100, 2)})
                n += 1; side = 0; units = 0; cur = None
            if mode == 'long_short':
                u = (cash * 0.95) / price
                if u > 0:
                    cash -= u * price * (1 + comm); units = u; entry = price; side = -1
                    cur = [t, price, '空', u]
    if side != 0 and len(df) > 0:
        price = df['close'].iloc[-1]
        cash = _close(cash, units, entry, price, side)
        trades.append({'in': cur[0], 'pin': cur[1], 'side': cur[2], 'qty': round(cur[3], 6),
                       'out': df.index[-1].strftime('%Y-%m-%d %H:%M'), 'pout': price,
                       'ret': round(((price * 0.999) / (cur[1] * 1.001) - 1) * 100, 2) if side > 0
                       else round((cur[1] - price * 1.001) / (cur[1] * 1.001) * 100, 2)})
        n += 1
        eq_pts.append((df.index[-1], cash))
    if n == 0:
        return None
    eq = pd.Series(dict(eq_pts)).sort_index()
    peak = eq.cummax()
    mdd = ((eq - peak) / peak * 100).min()
    rr = eq.pct_change().dropna()
    sh = float(rr.mean() / rr.std() * np.sqrt(1764)) if len(rr) >= 10 and rr.std() > 0 else None
    return {'ret': (eq.iloc[-1] / initial - 1) * 100, 'mdd': mdd, 'n': n, 'sh': sh, 'trades': trades}


store.init_tables()
all_results = {}
for TICKER in TICKERS:
    NAME = NAMES[TICKER]
    SECTOR = SECTORS[TICKER]
    df = fetch_1h(TICKER)
    if df is None:
        print(f"{TICKER}: 数据获取失败，跳过"); continue
    days = (df.index[-1] - df.index[0]).days
    per = f"{df.index[0]:%Y-%m-%d}~{df.index[-1]:%Y-%m-%d}"
    buy_hold = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    print(f"\n>>> {TICKER}（{NAME}·{SECTOR}）: {len(df)}根/{days}天 ({per})  买入持有 {buy_hold:+.1f}%")

    out = []
    for strat, ip in COMBOS.items():
        dft = TechnicalIndicators.calculate_all_indicators(df, ip)
        signals = sig_engine.calculate_signals(dft, ip)
        for mode in MODES:
            for tp, sl in TPSL:
                r = simulate(df, signals, tp, sl, mode)
                if r is None:
                    continue
                out.append({'ticker': TICKER, 'name': NAME, 'sector': SECTOR, 'strat': strat,
                            'mode': MODE_NAME[mode], 'tpsl': TPSL_NAME[(tp, sl)],
                            'period': per, 'days': days, **r})
    out.sort(key=lambda x: -x['ret'])
    all_results[TICKER] = out

    print(f"  {'策略':<10}{'模式':<10}{'TP/SL':<7}{'收益':>9}{'回撤':>8}{'笔':>4}{'夏普':>7}")
    for o in out[:10]:
        print(f"  {o['strat']:<10}{o['mode']:<10}{o['tpsl']:<7}{o['ret']:>+8.1f}%{o['mdd']:>7.1f}%"
              f"{o['n']:>4}{'' if o['sh'] is None else format(round(o['sh'], 2), '>7')}")

    only = [o for o in out if o['mode'] == '仅做多']
    only.sort(key=lambda x: -x['ret'])
    top = only[0] if only else out[0]
    top_global = out[0]
    wins = sum(1 for t in top['trades'] if t['ret'] > 0)
    winrate = f"{wins / len(top['trades']) * 100:.1f}%" if top['trades'] else '—'
    note = (f"{CHANNEL}，{NAME}（{SECTOR}板块）美股真实股票{TICKER}，{days}天{top['n']}笔；"
            f"标的自身买入持有{buy_hold:+.1f}%；"
            + (f"双向模拟版全场最高{top_global['strat']}/{top_global['tpsl']}:{top_global['ret']:+.1f}%（仅参考，个股现货做空不适用）"
               if top_global['ret'] > top['ret'] else "仅做多即全场最高"))
    rec = {'symbol': TICKER, 'timeframe': '1h', 'strategy': top['strat'],
           'mode': top['mode'], 'tpsl': top['tpsl'],
           'ret': f"{top['ret']:+.1f}%", 'daily': f"{top['ret'] / max(top['days'], 1):+.2f}%",
           'trades': str(top['n']), 'winrate': winrate, 'mdd': f"{top['mdd']:.1f}%",
           'stability': f"{top['days']}天{top['n']}笔",
           'market': MARKET_LABEL,
           'source': f"{CHANNEL}（{per}）",
           'note': note,
           'sharpe': None if top['sh'] is None else f"{top['sh']:.2f}",
           'period': per}
    ok, msg = store.add_history(rec)
    print(f"  入库({TICKER}·{SECTOR}): {top['strat']} {top['mode']} {top['tpsl']} {top['ret']:+.1f}% → {msg}")
    if ok:
        with store._conn() as c:
            row = c.execute("SELECT id FROM strategy_records WHERE symbol=? AND strategy=? AND mode=? "
                            "AND tpsl=? AND source=?",
                            (TICKER, top['strat'], top['mode'], top['tpsl'], rec['source'])).fetchone()
            if row:
                c.execute("UPDATE strategy_records SET trades_detail=? WHERE id=?",
                          (json.dumps({'trades': top['trades'], 'n': len(top['trades']),
                                       'ret_total': round(top['ret'], 1)}, ensure_ascii=False), row[0]))

os.makedirs('scripts/results', exist_ok=True)
with open('scripts/results/us_subsectors2_yahoo_1h_2025.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=1, default=float)
print('\nsaved → scripts/results/us_subsectors2_yahoo_1h_2025.json')
