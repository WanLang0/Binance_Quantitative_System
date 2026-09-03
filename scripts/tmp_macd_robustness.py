# -*- coding: utf-8 -*-
"""横跨多标的的『稳健性』分析：找在多数标的上都表现好的策略，而非单标的最优（防过拟合）。

对每个固定策略，跨全部 13 只标的统计：
  - 胜率(跑赢买入持有的标的占比)
  - 正收益占比
  - 中位收益 / 平均收益
  - 平均夏普及夏普为正占比
  - 平均交易数（反映“尽快买卖”的活跃度）
  - 稳健性综合评分（收益胜率+夏普+回撤 归一化）

仅研究，不写库。数据加载与模拟器与前序脚本完全一致，保证可复现/可比。
"""
import os, sys, io, time, warnings, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
PROXY = "http://127.0.0.1:7892"
os.environ.setdefault("HTTP_PROXY", PROXY)
os.environ.setdefault("HTTPS_PROXY", PROXY)

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ta
from datetime import datetime

_sess = None
def _get_session():
    global _sess
    if _sess is not None:
        return _sess
    s = requests.Session()
    s.headers['User-Agent'] = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                               '(KHTML, like Gecko) Chrome/120.0 Safari/537.36')
    s.proxies = {'http': PROXY, 'https': PROXY}
    retry = Retry(total=3, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    s.mount('https://', HTTPAdapter(max_retries=retry))
    try:
        s.get('https://query1.finance.yahoo.com/v8/finance/chart/QQQ?range=1d&interval=1d', timeout=10)
    except Exception:
        pass
    _sess = s
    return s

TICKERS = ['QQQ', 'NVDA', 'AAPL', 'GOOGL',
           'NEM', 'FNV', 'GFI', 'AEM', 'WPM', 'AU', 'KGC', 'RGLD', 'CDE']
NAME = {'QQQ': '纳指100 ETF', 'NVDA': '英伟达', 'AAPL': '苹果', 'GOOGL': '谷歌',
        'NEM': '纽蒙特黄金', 'FNV': '弗兰科-内华达', 'GFI': '金田', 'AEM': '阿格尼科鹰矿',
        'WPM': '惠顿贵金属', 'AU': '盎格鲁黄金', 'KGC': '金罗斯黄金', 'RGLD': '皇家黄金', 'CDE': '科罗拉多矿'}
COMM = 0.001


def fetch_1h(ticker, start='2025-01-01', tries=5):
    for i in range(tries):
        try:
            df = yf.download(ticker, interval='1h', start=start, end=datetime.now(),
                             progress=False, auto_adjust=True, session=_get_session())
            if df is None or df.empty:
                time.sleep(5); continue
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


def _sma(col, period):
    return ta.trend.SMAIndicator(col, window=period).sma_indicator()


def _macd_series(df, fast, slow, signal):
    macd = ta.trend.MACD(df['close'], window_fast=fast, window_slow=slow, window_sign=signal)
    df = df.copy()
    df['MACD'] = macd.macd(); df['MACD_signal'] = macd.macd_signal(); df['hist'] = macd.macd_diff()
    return df


PIVOT_ORDER = 5
VOL_MULT = 1.5


def _find_pivots(s, order=PIVOT_ORDER, kind='high'):
    arr = s.to_numpy(); piv = np.zeros(len(arr), dtype=bool)
    for i in range(order, len(arr) - order):
        win = arr[i - order:i + order + 1]
        if kind == 'high':
            piv[i] = np.argmax(win) == order and arr[i] == win.max()
            piv[i] = piv[i] and arr[i] > arr[i - 1]
        else:
            piv[i] = np.argmin(win) == order and arr[i] == win.min()
            piv[i] = piv[i] and arr[i] < arr[i - 1]
    return pd.Series(piv, index=s.index)


def _divergence(df, macd_col='MACD', order=PIVOT_ORDER):
    px_high = _find_pivots(df['high'], order, 'high')
    px_low = _find_pivots(df['low'], order, 'low')
    macd = df[macd_col].to_numpy()
    hi = df['high'].to_numpy(); lo = df['low'].to_numpy()
    top = np.zeros(len(df), dtype=bool); bot = np.zeros(len(df), dtype=bool)
    ph = None; pm = None
    for i in np.where(px_high.to_numpy())[0]:
        p = hi[i]; m = macd[i]
        if ph is not None and p > ph and m < pm:
            top[i] = True
        ph = p; pm = m
    pl = None; pm2 = None
    for i in np.where(px_low.to_numpy())[0]:
        p = lo[i]; m = macd[i]
        if pl is not None and p < pl and m > pm2:
            bot[i] = True
        pl = p; pm2 = m
    return pd.Series(top, index=df.index), pd.Series(bot, index=df.index)


def build_original_signals(df, use_div, use_ma, use_vol):
    dfd = _macd_series(df, 12, 26, 9)
    dfd['sma20'] = _sma(dfd['close'], 20)
    dfd['vol_ma20'] = dfd['volume'].rolling(20).mean()
    dfd['vol_up'] = dfd['volume'] > dfd['vol_ma20'] * VOL_MULT
    macd_buy = (dfd['MACD'] > dfd['MACD_signal']) & (dfd['MACD'].shift(1) <= dfd['MACD_signal'].shift(1))
    macd_sell = (dfd['MACD'] < dfd['MACD_signal']) & (dfd['MACD'].shift(1) >= dfd['MACD_signal'].shift(1))
    if use_div:
        top_div, bot_div = _divergence(dfd)
        buy = macd_buy | bot_div; sell = macd_sell | top_div
    else:
        buy = macd_buy.copy(); sell = macd_sell.copy()
    close = dfd['close']; sma = dfd['sma20']; vol_up = dfd['vol_up']
    if use_ma:
        buy = buy & (close > sma); sell = sell & (close < sma)
    if use_vol:
        buy = buy & vol_up; sell = sell & vol_up
    sig = pd.Series(0, index=dfd.index); sig[buy] = 1; sig[sell] = -1
    return sig


ORIGINAL_VARIANTS = {
    'MACD基准':       (False, False, False),
    'MACD+背离':      (True,  False, False),
    '背离+均线':      (True,  True,  False),
    '背离+量能':      (True,  False, True),
    '背离+均线+量能': (True,  True,  True),
}


def sig_macd_fast(df, fast, slow, signal):
    d = _macd_series(df, fast, slow, signal); dd = d['MACD']; s = d['MACD_signal']
    buy = (dd > s) & (dd.shift(1) <= s.shift(1)); sell = (dd < s) & (dd.shift(1) >= s.shift(1))
    sig = pd.Series(0, index=df.index); sig[buy] = 1; sig[sell] = -1
    return sig


def sig_hist_reversal(df, fast, slow, signal):
    d = _macd_series(df, fast, slow, signal); h = d['hist']
    buy = (h > 0) & (h.shift(1) <= 0); sell = (h < 0) & (h.shift(1) >= 0)
    sig = pd.Series(0, index=df.index); sig[buy] = 1; sig[sell] = -1
    return sig


def sig_hist_slope(df, fast, slow, signal):
    d = _macd_series(df, fast, slow, signal); h = d['hist']; dh = h.diff()
    buy = (dh > 0) & (h > 0); sell = (dh < 0) & (h < 0)
    sig = pd.Series(0, index=df.index); sig[buy] = 1; sig[sell] = -1
    return sig


def sig_trend_pullback(df, fast, slow, signal, ma_period=20):
    d = _macd_series(df, fast, slow, signal)
    sma = _sma(df['close'], ma_period); close = df['close']
    trend_up = (close > sma) & (d['MACD'] > d['MACD_signal'])
    trend_dn = (close < sma) & (d['MACD'] < d['MACD_signal'])
    near = (close - sma).abs() / sma < 0.008
    buy = trend_up & near; sell = trend_dn & near
    sig = pd.Series(0, index=df.index); sig[buy] = 1; sig[sell] = -1
    return sig


IMPROVED_VARIANTS = {
    '快 MACD 8/17/9':    lambda df: sig_macd_fast(df, 8, 17, 9),
    '柱状图拐头12/26/9': lambda df: sig_hist_reversal(df, 12, 26, 9),
    '柱状图斜率12/26/9': lambda df: sig_hist_slope(df, 12, 26, 9),
    '顺势回踩MA20+MACD12': lambda df: sig_trend_pullback(df, 12, 26, 9),
    '顺势回踩MA20+MACD8':  lambda df: sig_trend_pullback(df, 8, 17, 9),
}

# 统一：name -> (group, fn)
ALL_VARIANTS = {}
for k, v in ORIGINAL_VARIANTS.items():
    ALL_VARIANTS[k] = ('原背离共振', (lambda df, du=v[0], ma=v[1], vo=v[2]: build_original_signals(df, du, ma, vo)))
for k, fn in IMPROVED_VARIANTS.items():
    ALL_VARIANTS[k] = ('改进版', fn)


def _close(cash, units, entry, price, side, comm=COMM):
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm


def simulate(df, signals, tp=None, sl=None, mode='long_only', initial=10000.0, comm=COMM):
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
                cash = _close(cash, units, entry, price, side); side = 0; units = 0; n += 1
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
            if mode == 'long_short':
                u = (cash * 0.95) / price
                if u > 0:
                    cash -= u * price * (1 + comm); units = u; entry = price; side = -1
    if side != 0 and len(df) > 0:
        price = df['close'].iloc[-1]
        cash = _close(cash, units, entry, price, side); n += 1
        eq_pts.append((df.index[-1], cash))
    if n == 0:
        return None
    eq = pd.Series(dict(eq_pts)).sort_index()
    peak = eq.cummax()
    mdd = ((eq - peak) / peak * 100).min()
    rr = eq.pct_change().dropna()
    sh = float(rr.mean() / rr.std() * np.sqrt(1764)) if len(rr) >= 10 and rr.std() > 0 else None
    return {'ret': (eq.iloc[-1] / initial - 1) * 100, 'mdd': mdd, 'n': n, 'sh': sh, 'eq_pts': eq_pts}


def main():
    # 每个策略 -> 每标的表现 + 买入持有
    strat_per_ticker = {v: {} for v in ALL_VARIANTS}
    bh = {}
    for TICKER in TICKERS:
        df = fetch_1h(TICKER)
        if df is None:
            print(f"{TICKER}: 获取失败，跳过"); continue
        bh[TICKER] = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        for vname, (grp, fn) in ALL_VARIANTS.items():
            try:
                sig = fn(df)
                r = simulate(df, sig)   # 仅做多 + 不设止盈止损（最干净可比口径）
                if r is not None:
                    strat_per_ticker[vname][TICKER] = r
            except Exception as e:
                print(f"{TICKER} {vname} err: {e}")
        print(f"下载完成 {TICKER} 买入持有 {bh[TICKER]:+.1f}%")
        time.sleep(1)

    # 稳健性聚合
    print("\n\n===== 跨标的稳健性（仅做多+不设止盈止损，13标的）=====")
    print(f"  {'策略':<20}{'标的':>4}{'胜率>BH':>8}{'正收益':>7}{'中位收益':>10}{'平均收益':>10}"
          f"{'平均回撤':>9}{'均夏普':>7}{'夏普正率':>8}{'均交易数':>9}{'稳健分':>7}")
    scores = []
    for vname, per in strat_per_ticker.items():
        n = len(per)
        if n == 0:
            continue
        rets = [r['ret'] for r in per.values()]
        mdds = [r['mdd'] for r in per.values()]
        shs = [r['sh'] for r in per.values() if r['sh'] is not None]
        ns = [r['n'] for r in per.values()]
        win = sum(1 for t, r in per.items() if r['ret'] > bh.get(t, 0))
        pos = sum(1 for r in rets if r > 0)
        win_rate = win / n * 100
        pos_rate = pos / n * 100
        med_ret = float(np.median(rets))
        avg_ret = float(np.mean(rets))
        avg_mdd = float(np.mean(mdds))
        avg_sh = float(np.mean(shs)) if shs else None
        sh_pos_rate = (sum(1 for s in shs if s > 0) / len(shs) * 100) if shs else 0
        avg_n = float(np.mean(ns))
        # 稳健性综合评分：胜率高、中位收益高、均夏普高、回撤小 → 归一化到 0-100
        comp = (win_rate / 100) * 40 + (max(0.0, med_ret) / 20) * 20 + (avg_sh / 1.5 if avg_sh else 0) * 25 + (1.0 - min(abs(avg_mdd), 40) / 40) * 15
        comp = round(min(comp, 100), 1)
        scores.append((vname, n, win_rate, pos_rate, med_ret, avg_ret, avg_mdd, avg_sh, sh_pos_rate, avg_n, comp))
        print(f"  {vname:<20}{n:>4}{win_rate:>7.0f}%{pos_rate:>6.0f}%{med_ret:>+9.1f}%{avg_ret:>+9.1f}%"
              f"{avg_mdd:>8.1f}%{'' if avg_sh is None else format(round(avg_sh,2),'>7')}"
              f"{sh_pos_rate:>7.0f}%{avg_n:>9.1f}{comp:>7.1f}")
    scores.sort(key=lambda x: -x[-1])
    print("\n—— 稳健性综合评分 Top 5 ——")
    for s in scores[:5]:
        print(f"  {s[0]:<20} 稳健分 {s[-1]}")
    print("\n—— 稳健性综合评分 Bottom 3 ——")
    for s in scores[-3:]:
        print(f"  {s[0]:<20} 稳健分 {s[-1]}")

    # 每标的“在多数选择中稳健”的策略：统计每个策略出现为每标的前2名的频率
    print("\n—— 每标的前2名策略（看是否有策略在多标的频繁进前2）——")
    from collections import Counter
    cnt = Counter()
    for TICKER in TICKERS:
        per = {v: per.get(TICKER) for v, per in strat_per_ticker.items()}
        valid = [(v, r) for v, r in per.items() if r is not None]
        valid.sort(key=lambda x: -x[1]['ret'])
        top2 = [v for v, r in valid[:2]]
        for v in top2:
            cnt[v] += 1
    print("  策略进入单标的最优2名次数：", dict(cnt.most_common()))

    os.makedirs('scripts/results', exist_ok=True)
    with open('scripts/results/robustness_1h_2025.json', 'w', encoding='utf-8') as f:
        json.dump({'bh': bh, 'strat_per_ticker': {k: {t: {kk: vv for kk, vv in v.items() if kk != 'eq_pts'} for t, v in per.items()} for k, per in strat_per_ticker.items()},
                   'scores': scores, 'top2_counter': dict(cnt.most_common())},
                  f, ensure_ascii=False, indent=1, default=float)
    print(f"\nsaved → scripts/results/robustness_1h_2025.json")


if __name__ == '__main__':
    main()
