# -*- coding: utf-8 -*-
"""Walk-forward 样本外验证：原「MACD背离+共振」5 变体 vs 改进版变体。

由于 Yahoo 1h 仅保留最近 730 天，无独立历史样本（2023-2024），采用 Walk-forward：
- 全窗口：2024-09 起 ~ 现在（可用 1h 数据）
- 样本内 IS：2024-09 ~ 2025-12（用行情选优）
- 样本外 OOS：2026-01 ~ 现在（验证 IS 选出的策略是否延续）

每个标的在 IS 按「仅做多+不设止盈止损」选出最优变体，再应用到 OOS，
检验 IS 领先的策略在 OOS 是否稳健、是否跑赢买入持有。仅研究，不写库。
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
MODE_L = '仅做多'
TPSL_N = '不设'
COMM = 0.001

IS_END = '2025-12-31'   # IS 窗口到 2025 年底；OOS 为 2026-01 起


def fetch_1h(ticker, start='2024-09-20', tries=6):
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
            if len(df) >= 300:
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

# 汇总所有策略为 name -> (group, fn)
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
    return {'ret': (eq.iloc[-1] / initial - 1) * 100, 'mdd': mdd, 'n': n, 'sh': sh}


def _run_on_window(df, sig, start, end):
    """在 [start,end] 窗口内切片运行。返回 dict 或 None。"""
    tz = getattr(df.index, 'tz', None)
    s_ts = pd.Timestamp(start); e_ts = pd.Timestamp(end)
    if tz is not None:
        if getattr(s_ts, 'tzinfo', None) is None:
            s_ts = s_ts.tz_localize(tz)
        else:
            s_ts = s_ts.tz_convert(tz)
        if getattr(e_ts, 'tzinfo', None) is None:
            e_ts = e_ts.tz_localize(tz)
        else:
            e_ts = e_ts.tz_convert(tz)
    mask = (df.index >= s_ts) & (df.index <= e_ts)
    if mask.sum() < 30:
        return None
    sub_df = df.loc[mask]
    sub_sig = sig.loc[mask]
    return simulate(sub_df, sub_sig)


def main():
    # 基准：始终选 MACD+背离 / MACD基准 / 背离+量能 / 顺势回踩MA20+MACD12
    FIXED_BASELINES = ['MACD+背离', 'MACD基准', '背离+量能', '顺势回踩MA20+MACD12', '背离+均线+量能']
    rows = []
    per_ticker = {}
    _dfs = {}
    _sigs = {}
    for TICKER in TICKERS:
        df = fetch_1h(TICKER)
        if df is None:
            print(f"{TICKER}: 获取失败，跳过"); continue
        _dfs[TICKER] = df
        # 统一用 12/26/9 作为基准列，其余变体在各自函数内重新计算
        sigs = {}
        for vname, (grp, fn) in ALL_VARIANTS.items():
            try:
                sigs[vname] = fn(df)
            except Exception as e:
                print(f"{TICKER} {vname} 信号生成失败: {e}")
        _sigs[TICKER] = sigs

        is_res = {}
        for vname, sig in sigs.items():
            r = _run_on_window(df, sig, df.index[0], IS_END)
            is_res[vname] = r
        # IS 选优：仅做多+不设止盈止损 + 夏普 > 0 有交易
        is_rank = [v for v, r in is_res.items()
                   if r is not None and r['n'] >= 3 and r['sh'] is not None and r['sh'] > 0]
        is_rank.sort(key=lambda v: -is_res[v]['sh'])
        best_is = is_rank[0] if is_rank else 'MACD+背离'

        # OOS 评估
        oos_start = IS_END
        oos_res = {}
        buy_hold_oos = None
        tz = getattr(df.index, 'tz', None)
        oos_ts = pd.Timestamp(oos_start)
        if tz is not None:
            oos_ts = oos_ts.tz_localize(tz)
        mask_oos = (df.index > oos_ts)
        if mask_oos.sum() >= 30:
            sub = df.loc[mask_oos]
            buy_hold_oos = (sub['close'].iloc[-1] / sub['close'].iloc[0] - 1) * 100
            for vname, sig in sigs.items():
                oos_res[vname] = _run_on_window(df, sig, oos_start, df.index[-1])

        # 记录 IS 最优在 OOS 的表现
        best_oos = oos_res.get(best_is)
        per_ticker[TICKER] = {
            'best_is': best_is, 'best_is_sh': is_res[best_is]['sh'] if is_res.get(best_is) else None,
            'best_is_ret': is_res[best_is]['ret'] if is_res.get(best_is) else None,
            'oos_best_strat_ret': best_oos['ret'] if best_oos else None,
            'oos_best_strat_mdd': best_oos['mdd'] if best_oos else None,
            'oos_best_strat_n': best_oos['n'] if best_oos else None,
            'oos_best_strat_sh': best_oos['sh'] if best_oos else None,
            'buy_hold_oos': buy_hold_oos,
            'is_rank_top5': is_rank[:5],
        }
        info = per_ticker[TICKER]
        print(f"\n>>> {TICKER}（{NAME[TICKER]}） IS最优={info['best_is']} "
              f"(IS夏普 {info['best_is_sh']:.2f})")
        print(f"    OOS[{oos_start}→末]  策略收益 "
              f"{'' if info['oos_best_strat_ret'] is None else format(info['oos_best_strat_ret'], '+.1f')}%"
              f"  买入持有 {info['buy_hold_oos']:+.1f}%")

        # OOS 全部策略排名（供看稳定性）
        oos_rank = sorted([(v, r) for v, r in oos_res.items() if r is not None], key=lambda x: -x[1]['ret'])
        for v, r in oos_rank[:6]:
            print(f"      OOS {v:<20} {r['ret']:+.1f}%  mdd {r['mdd']:.1f}%  n {r['n']}  sh "
                  f"{'' if r['sh'] is None else format(round(r['sh'],2))}")
        rows.append({'ticker': TICKER, **info})

    # 汇总
    print("\n\n===== Walk-forward 汇总 =====")
    print(f"  窗口：IS≈2024-09~{IS_END}  /  OOS={IS_END}~今")
    print(f"  {'标的':<6}{'IS最优':<20}{'OOS策略收益':>11}{'OOS买入持有':>12}{'跑赢?':>6}")
    beat_cnt = 0; tot = 0
    oos_ret_list = []; oos_bh_list = []
    for r in rows:
        oos_ret = r['oos_best_strat_ret']; bh = r['buy_hold_oos']
        if oos_ret is None or bh is None:
            continue
        tot += 1
        beat = oos_ret > bh
        if beat:
            beat_cnt += 1
        oos_ret_list.append(oos_ret); oos_bh_list.append(bh)
        print(f"  {r['ticker']:<6}{r['best_is']:<20}{oos_ret:>+10.1f}%{bh:>+11.1f}%{'✓' if beat else '✗':>6}")
    if tot:
        print(f"\n  OOS 平均：IS最优策略 {np.mean(oos_ret_list):+.1f}%  vs 买入持有 {np.mean(oos_bh_list):+.1f}%")
        print(f"  IS最优策略在 OOS 跑赢买入持有的标的：{beat_cnt}/{tot}")

    # 另一个视角：固定策略（不做 IS 选优）在 OOS 的平均表现，检验哪个策略 OOS 最稳
    # 重新聚合：对每个策略，统计全部标的的 OOS 表现
    strat_oos = {}
    for TICKER in TICKERS:
        df = _dfs.get(TICKER)
        if df is None:
            continue
        for vname, sig in _sigs[TICKER].items():
            r = _run_on_window(df, sig, IS_END, df.index[-1])
            if r is None:
                continue
            strat_oos.setdefault(vname, []).append(r)
    print("\n  —— 固定策略在 OOS 的平均表现（不做 IS 选优，看哪个最稳）——")
    print(f"  {'策略':<20}{'标的':>4}{'平均收益':>10}{'平均回撤':>10}{'平均夏普':>9}{'跑赢BH':>8}")
    fixed_rows = []
    for vname, lst in strat_oos.items():
        avg_ret = np.mean([r['ret'] for r in lst])
        avg_mdd = np.mean([r['mdd'] for r in lst])
        shs = [r['sh'] for r in lst if r['sh'] is not None]
        avg_sh = np.mean(shs) if shs else None
        cnt = len(lst)
        fixed_rows.append((vname, cnt, avg_ret, avg_mdd, avg_sh))
    fixed_rows.sort(key=lambda x: -x[2])
    for vname, cnt, avg_ret, avg_mdd, avg_sh in fixed_rows:
        print(f"  {vname:<20}{cnt:>4}{avg_ret:>+9.1f}%{avg_mdd:>9.1f}%"
              f"{'' if avg_sh is None else format(round(avg_sh,2),'>9')}")
    os.makedirs('scripts/results', exist_ok=True)
    with open('scripts/results/walkforward_1h_2025.json', 'w', encoding='utf-8') as f:
        json.dump({'is_end': IS_END, 'rows': rows, 'fixed_oos': fixed_rows,
                   'agg': {'oos_ret_avg': float(np.mean(oos_ret_list)) if oos_ret_list else None,
                           'oos_bh_avg': float(np.mean(oos_bh_list)) if oos_bh_list else None,
                           'beat': beat_cnt, 'tot': tot}},
                  f, ensure_ascii=False, indent=1, default=float)
    print(f"\nsaved → scripts/results/walkforward_1h_2025.json")


if __name__ == '__main__':
    main()
