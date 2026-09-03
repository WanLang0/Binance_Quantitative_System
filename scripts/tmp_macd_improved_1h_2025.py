# -*- coding: utf-8 -*-
"""改进版 MACD 策略研究（降低滞后，逻辑自洽），与「MACD背离+共振」在相同标的/周期/区间对比。

设计思路（针对单一 MACD 滞后 + 原版「领先背离+滞后均线」矛盾的痛点）：
1) 更快的 MACD 参数：8/17/9 与 5/13/9，比引擎默认 12/26 对转折更灵敏，减少滞后。
2) 提前信号：柱状图(hist)斜率为正/负收敛提前于金叉死叉 —— 用 hist 拐头作为预备信号。
3) 自洽的趋势过滤：只做顺势（均线多空同向），不用「抢先反转的背离」，避免逻辑拧巴。
4) 回踩均线进场：趋势确立后，等回调到 MA 附近再进，替代追涨金叉，自带止损位。
5) 出场更可控：均线反向/对侧信号离场 + 可选 5%止盈止损。

本脚本仅做研究对比，不写入数据库（避免污染现有 strategy_records 历史测试表）。
"""
import os, sys, io, time, warnings, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
# 本地代理（与 .env 一致；本机 127.0.0.1:7892 已测试可拉 Yahoo 1h）
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

# 与原脚本完全一致的标的 / 名称 / 周期 / 区间
TICKERS = ['QQQ', 'NVDA', 'AAPL', 'GOOGL',
           'NEM', 'FNV', 'GFI', 'AEM', 'WPM', 'AU', 'KGC', 'RGLD', 'CDE']
NAME = {'QQQ': '纳指100 ETF', 'NVDA': '英伟达', 'AAPL': '苹果', 'GOOGL': '谷歌',
        'NEM': '纽蒙特黄金', 'FNV': '弗兰科-内华达', 'GFI': '金田', 'AEM': '阿格尼科鹰矿',
        'WPM': '惠顿贵金属', 'AU': '盎格鲁黄金', 'KGC': '金罗斯黄金', 'RGLD': '皇家黄金', 'CDE': '科罗拉多矿'}
MODES = ['long_only', 'long_short']
MODE_NAME = {'long_only': '仅做多', 'long_short': '双向(模拟)'}
TPSL = [(None, None), (0.05, 0.05)]
TPSL_NAME = {(None, None): '不设', (0.05, 0.05): '5%/5%'}
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


def _ema(col, period):
    return ta.trend.EMAIndicator(col, window=period).ema_indicator()


def macd_params(fast, slow, signal):
    return (fast, slow, signal)


def _macd_series(df, fast, slow, signal):
    """MACD 与柱状图。返回 (df, MACD, signal, hist)。"""
    macd = ta.trend.MACD(df['close'], window_fast=fast, window_slow=slow, window_sign=signal)
    macd_v = macd.macd(); sig_v = macd.macd_signal(); hist_v = macd.macd_diff()
    df = df.copy()
    df['MACD'] = macd_v; df['MACD_signal'] = sig_v; df['hist'] = hist_v
    return df


def _cross(df, fast, slow, signal):
    """金叉/死叉信号（12/26/9 基准 & 更快参数共用一个函数）。"""
    if (fast, slow, signal) != (12, 26, 9):
        df = _macd_series(df, fast, slow, signal)
        d = df['MACD']; s = df['MACD_signal']
        buy = (d > s) & (d.shift(1) <= s.shift(1))
        sell = (d < s) & (d.shift(1) >= s.shift(1))
    else:
        d = df['MACD']; s = df['MACD_signal']
        buy = (d > s) & (d.shift(1) <= s.shift(1))
        sell = (d < s) & (d.shift(1) >= s.shift(1))
    return buy, sell


# ---- 各改进变体信号生成 ----
def sig_macd_128(df, fast=12, slow=26, signal=9):
    """基准：标准 MACD 金叉死叉。"""
    if (fast, slow, signal) != (12, 26, 9):
        df = _macd_series(df, fast, slow, signal)
    buy, sell = _cross(df, fast, slow, signal)
    return _to_sig(df, buy, sell)


def sig_macd_fast(df, fast, slow, signal):
    """更快参数：更灵敏的 MACD。"""
    d = _macd_series(df, fast, slow, signal)
    dd = d['MACD']; s = d['MACD_signal']
    buy = (dd > s) & (dd.shift(1) <= s.shift(1))
    sell = (dd < s) & (dd.shift(1) >= s.shift(1))
    return _to_sig(df, buy, sell)


def sig_hist_reversal(df, fast, slow, signal):
    """柱状图拐头：hist 从负数收窄转正(做多)，从正数收窄转负(做空)。比金叉提前。"""
    d = _macd_series(df, fast, slow, signal)
    h = d['hist']
    # 做多：前一根 hist<0 且当前 hist 转正（柱体由负转正）
    buy = (h > 0) & (h.shift(1) <= 0)
    sell = (h < 0) & (h.shift(1) >= 0)
    return _to_sig(df, buy, sell)


def sig_hist_slope(df, fast, slow, signal):
    """柱状图斜率：hist 连续放大(做多) / 连续收窄(做空)，更早捕捉动能方向。"""
    d = _macd_series(df, fast, slow, signal)
    h = d['hist']
    dh = h.diff()
    # 做多：hist 上坡且为正（动能增强）
    buy = (dh > 0) & (h > 0)
    # 做空：hist 下坡且为负
    sell = (dh < 0) & (h < 0)
    return _to_sig(df, buy, sell)


def sig_trend_pullback(df, fast, slow, signal, ma_period=20):
    """顺势回踩均线：确定趋势(价>均线 with MACD 多头)后，等回调到均线附近再进。"""
    d = _macd_series(df, fast, slow, signal)
    sma = _sma(df['close'], ma_period)
    close = df['close']
    macd_above = d['MACD'] > d['MACD_signal']
    trend_up = (close > sma) & macd_above
    trend_dn = (close < sma) & (d['MACD'] < d['MACD_signal'])
    # 回踩 = 跌到均线附近但趋势仍在（用 close 对均线的贴近判断）
    near = (close - sma).abs() / sma < 0.008
    buy = trend_up & near
    sell = trend_dn & near
    return _to_sig(df, buy, sell)


def sig_macd_slope_pullback(df, fast, slow, signal, ma_period=20):
    """快 MACD + 均线顺势回踩（组合：更快信号 + 自洽趋势过滤）。"""
    d = _macd_series(df, fast, slow, signal)
    sma = _sma(df['close'], ma_period)
    close = df['close']
    macd_above = d['MACD'] > d['MACD_signal']
    trend_up = (close > sma) & macd_above
    trend_dn = (close < sma) & (d['MACD'] < d['MACD_signal'])
    near = (close - sma).abs() / sma < 0.008
    buy = trend_up & near
    sell = trend_dn & near
    return _to_sig(df, buy, sell)


def _to_sig(df, buy, sell):
    sig = pd.Series(0, index=df.index)
    sig[buy] = 1
    sig[sell] = -1
    return sig


def _close(cash, units, entry, price, side, comm=COMM):
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm


def simulate(df, signals, tp, sl, mode, initial=10000.0, comm=COMM):
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
    return {'ret': (eq.iloc[-1] / initial - 1) * 100, 'mdd': mdd, 'n': n, 'sh': sh}


# 改进变体集合：名称 -> (生成函数, MACD参数, 说明)
IMPROVED_VARIANTS = {
    # 基准（外部对比用）
    '基准 MACD12/26/9': (lambda df: sig_macd_128(df), (12, 26, 9)),
    # 更快参数（降滞后）
    '快 MACD 8/17/9':  (lambda df, f=8, s=17, g=9: sig_macd_fast(df, f, s, g), (8, 17, 9)),
    '快 MACD 5/13/9':  (lambda df, f=5, s=13, g=9: sig_macd_fast(df, f, s, g), (5, 13, 9)),
    # 柱状图提前信号
    '柱状图拐头 12/26/9': (lambda df: sig_hist_reversal(df, 12, 26, 9), (12, 26, 9)),
    '柱状图斜率 12/26/9': (lambda df: sig_hist_slope(df, 12, 26, 9), (12, 26, 9)),
    # 顺势回踩均线（自洽），分别配快/慢 MACD
    '顺势回踩MA20+MACD12': (lambda df, f=12, s=26, g=9: sig_trend_pullback(df, f, s, g), (12, 26, 9)),
    '顺势回踩MA20+MACD8':  (lambda df, f=8, s=17, g=9: sig_trend_pullback(df, f, s, g), (8, 17, 9)),
}


def main():
    results = {}
    rows = []
    buy_holds = {}
    for TICKER in TICKERS:
        df = fetch_1h(TICKER)
        if df is None:
            print(f"{TICKER}: 数据获取失败，跳过"); continue
        # 基准 MACD 列（供 _cross 的 12/26/9 分支使用）
        df = _macd_series(df, 12, 26, 9)
        days = (df.index[-1] - df.index[0]).days
        per = f"{df.index[0]:%Y-%m-%d}~{df.index[-1]:%Y-%m-%d}"
        buy_hold = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        buy_holds[TICKER] = buy_hold
        print(f"\n>>> {TICKER}（{NAME[TICKER]}）: {len(df)}根/{days}天 ({per})  买入持有 {buy_hold:+.1f}%")

        out = []
        for vname, (fn, _) in IMPROVED_VARIANTS.items():
            sig = fn(df)
            n_sig = int(sig[sig != 0].sum())
            for mode in MODES:
                for tp, sl in TPSL:
                    r = simulate(df, sig, tp, sl, mode)
                    if r is None:
                        continue
                    out.append({'ticker': TICKER, 'name': NAME[TICKER], 'strat': vname,
                                'mode': MODE_NAME[mode], 'tpsl': TPSL_NAME[(tp, sl)],
                                'period': per, 'days': days, 'n_sig': n_sig, **r})
        out.sort(key=lambda x: -x['ret'])
        results[TICKER] = out
        rows.extend(out)

        print(f"  {'策略':<22}{'模式':<10}{'TP/SL':<7}{'收益':>9}{'回撤':>8}{'笔':>4}{'信号':>5}{'夏普':>7}")
        for o in out[:14]:
            print(f"  {o['strat']:<22}{o['mode']:<10}{o['tpsl']:<7}{o['ret']:>+8.1f}%{o['mdd']:>7.1f}%"
                  f"{o['n']:>4}{o['n_sig']:>5}{'' if o['sh'] is None else format(round(o['sh'], 2), '>7')}")

    # 汇总：各策略全标的平均收益 / 平均回撤 / 平均夏普（仅做多 & 不设止盈止损，最可比）
    agg = {}
    for o in rows:
        key = o['strat']
        agg.setdefault(key, []).append(o)
    print("\n\n===== 跨标的汇总（仅做多 + 不设止盈止损，最可比口径）=====")
    print(f"  {'策略':<22}{'标的数':>5}{'平均收益':>10}{'平均回撤':>10}{'平均夏普':>9}{'跑赢买入持有':>12}")
    summary = []
    for key, lst in agg.items():
        only = [o for o in lst if o['mode'] == '仅做多' and o['tpsl'] == '不设']
        if not only:
            continue
        avg_ret = np.mean([o['ret'] for o in only])
        avg_mdd = np.mean([o['mdd'] for o in only])
        avg_sh = np.mean([o['sh'] for o in only if o['sh'] is not None]) if any(o['sh'] is not None for o in only) else None
        # 该策略每标的是否跑赢同标的买入持有
        beat = sum(1 for o in only if o['ret'] > buy_holds.get(o['ticker'], 0))
        summary.append((key, len(only), avg_ret, avg_mdd, avg_sh, beat))
        print(f"  {key:<22}{len(only):>5}{avg_ret:>+9.1f}%{avg_mdd:>9.1f}%"
              f"{'' if avg_sh is None else format(round(avg_sh, 2), '>9')}{beat:>10}/{len(only)}")
    summary.sort(key=lambda x: -x[2])

    os.makedirs('scripts/results', exist_ok=True)
    with open('scripts/results/macd_improved_1h_2025.json', 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'summary': summary}, f, ensure_ascii=False, indent=1, default=float)
    print(f"\nsaved → scripts/results/macd_improved_1h_2025.json")


if __name__ == '__main__':
    main()
