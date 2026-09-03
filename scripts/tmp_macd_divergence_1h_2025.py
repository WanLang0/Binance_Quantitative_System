# -*- coding: utf-8 -*-
"""MACD + 价格形态(顶底背离) + 共振过滤 组合策略研究（QQQ / NVDA，2025-01-01 ~ 今，1h）

口径：
- 数据：Yahoo Finance 1h（美股真实标的 QQQ=纳指100ETF / NVDA=英伟达），经服务器代理(192.168.11.64:7892)
- 核心：MACD 结合价格行为(Price Action)
  顶背离 = 价格创出新高但 MACD(DIF线)未创新高 → 增强做空信号
  底背离 = 价格创出新低但 MACD(DIF线)未创新低 → 增强做多信号
- 共振过滤：死叉/金叉基础上叠加 20日均线、量能放大 作为独立维度过滤
- 变体对比(VARIANTS)：
  1) MACD基准      : 纯金叉/死叉
  2) MACD+背离     : 金叉死叉 与 底/顶背离 取并集(背离补充普通信号)
  3) 背离+均线     : 背离与普通信号均需 close 位于 20日线同侧(趋势过滤)
  4) 背离+量能     : 背离与普通信号均需成交量放大(>1.5倍20期均量)确认
  5) 背离+均线+量能: 三重独立维度共振
- 模式：仅做多(long_only) / 双向模拟(long_short)；止盈止损 不设 / 5%5%
- 结果：全变体对比表 + 每标的最优配置入库历史测试表(strategy_records)
"""
import os, sys, io, time, warnings, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
# 代理：服务器 192.168.11.64:7892（经 warmup + User-Agent 后可稳定拉 Yahoo 1h）
PROXY = "http://192.168.11.64:7892"
os.environ.setdefault("HTTP_PROXY", PROXY)
os.environ.setdefault("HTTPS_PROXY", PROXY)

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ta

_sess = None
def _get_session():
    """带 cookie/crumb + UA + 重试 的 Session，规避 Yahoo Edge 429"""
    global _sess
    if _sess is not None:
        return _sess
    s = requests.Session()
    s.headers['User-Agent'] = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                               '(KHTML, like Gecko) Chrome/120.0 Safari/537.36')
    s.proxies = {'http': PROXY, 'https': PROXY}
    retry = Retry(total=3, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    s.mount('https://', HTTPAdapter(max_retries=retry))
    try:  # 预热拿 cookie/crumb
        s.get('https://query1.finance.yahoo.com/v8/finance/chart/QQQ?range=1d&interval=1d', timeout=10)
    except Exception:
        pass
    _sess = s
    return s
from datetime import datetime
from indicators import TechnicalIndicators
import strategies_store as store

CHANNEL = "Yahoo Finance 美股1h行情(MACD背离/共振研究)"
MARKET_LABEL = "美股真实股票(非币安币) 主网现货"
TICKERS = ['QQQ', 'NVDA', 'AAPL', 'GOOGL',
           'NEM', 'FNV', 'GFI', 'AEM', 'WPM', 'AU', 'KGC', 'RGLD', 'CDE']  # QQQ/NVDA 已测；AAPL=苹果 GOOGL=谷歌；其余为黄金板块
NAME = {'QQQ': '纳指100 ETF', 'NVDA': '英伟达', 'AAPL': '苹果', 'GOOGL': '谷歌',
        'NEM': '纽蒙特黄金', 'FNV': '弗兰科-内华达', 'GFI': '金田', 'AEM': '阿格尼科鹰矿',
        'WPM': '惠顿贵金属', 'AU': '盎格鲁黄金', 'KGC': '金罗斯黄金', 'RGLD': '皇家黄金', 'CDE': '科罗拉多矿'}
MODES = ['long_only', 'long_short']
MODE_NAME = {'long_only': '仅做多', 'long_short': '双向(模拟)'}
TPSL = [(None, None), (0.05, 0.05)]
TPSL_NAME = {(None, None): '不设', (0.05, 0.05): '5%/5%'}
COMM = 0.001

# MACD 参数（与引擎一致）
MACD_P = {"macd": True, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9}

# 变体：名称 -> 是否启用(背离, 均线过滤, 量能过滤)
VARIANTS = {
    'MACD基准':       (False, False, False),
    'MACD+背离':      (True,  False, False),
    '背离+均线':      (True,  True,  False),
    '背离+量能':      (True,  False, True),
    '背离+均线+量能': (True,  True,  True),
}
PIVOT_ORDER = 5       # 背离 pivot 用 5+5 根窗口（1h，约半天）
VOL_MULT = 1.5        # 量能放大倍数(20期均量)


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


def _find_pivots(s, order=PIVOT_ORDER, kind='high'):
    """返回 bool Series：标记窗口 [i-order, i+order] 内的局部极值点(要求为窗口唯一极值)"""
    arr = s.to_numpy()
    piv = np.zeros(len(arr), dtype=bool)
    for i in range(order, len(arr) - order):
        win = arr[i - order:i + order + 1]
        if kind == 'high':
            piv[i] = np.argmax(win) == order and arr[i] == win.max()
            # 排除平台期(与前一根同高)造成的重复，保证更严谨
            piv[i] = piv[i] and arr[i] > arr[i - 1]
        else:
            piv[i] = np.argmin(win) == order and arr[i] == win.min()
            piv[i] = piv[i] and arr[i] < arr[i - 1]
    return pd.Series(piv, index=s.index)


def _divergence(df, macd_col='MACD', order=PIVOT_ORDER):
    """顶背离/底背离。返回 (top_div, bot_div) 布尔 Series（在 pivot 确认点标记）"""
    px_high = _find_pivots(df['high'], order, 'high')
    px_low = _find_pivots(df['low'], order, 'low')
    macd = df[macd_col].to_numpy()
    hi = df['high'].to_numpy(); lo = df['low'].to_numpy()
    top = np.zeros(len(df), dtype=bool); bot = np.zeros(len(df), dtype=bool)
    ph = None; pm = None
    for i in np.where(px_high.to_numpy())[0]:
        p = hi[i]; m = macd[i]
        if ph is not None and p > ph and m < pm:   # 价新高 MACD 未新高 → 顶背离
            top[i] = True
        ph = p; pm = m
    pl = None; pm2 = None
    for i in np.where(px_low.to_numpy())[0]:
        p = lo[i]; m = macd[i]
        if pl is not None and p < pl and m > pm2:  # 价新低 MACD 未新低 → 底背离
            bot[i] = True
        pl = p; pm2 = m
    return pd.Series(top, index=df.index), pd.Series(bot, index=df.index)


def _macd_series(df):
    """标准 MACD 金叉/死叉信号（与 BacktestEngine 一致）"""
    dft = TechnicalIndicators.calculate_macd(df, 12, 26, 9)
    df = df.copy()
    df['MACD'] = dft['MACD']; df['MACD_signal'] = dft['MACD_signal']
    buy = (df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))
    sell = (df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1))
    return buy, sell, df


def _sma(col, period=20):
    return ta.trend.SMAIndicator(col, window=period).sma_indicator()


def build_variant_signals(df, use_div, use_ma, use_vol):
    """按变体组合生成 -1/0/1 信号 Series。返回 (df_transformed, signals)"""
    macd_buy, macd_sell, dfd = _macd_series(df)
    dfd['sma20'] = _sma(dfd['close'], 20)
    dfd['vol_ma20'] = dfd['volume'].rolling(20).mean()
    dfd['vol_up'] = dfd['volume'] > dfd['vol_ma20'] * VOL_MULT

    if use_div:
        top_div, bot_div = _divergence(dfd)
        dfd['top_div'] = top_div; dfd['bot_div'] = bot_div
        buy = macd_buy | bot_div
        sell = macd_sell | top_div
    else:
        buy = macd_buy.copy(); sell = macd_sell.copy()

    # 独立维度共振过滤：均线(价在20日线同侧) + 量能(放量确认)
    close = dfd['close']; sma = dfd['sma20']; vol_up = dfd['vol_up']
    ma_ok_buy = close > sma
    ma_ok_sell = close < sma
    if use_ma:
        buy = buy & ma_ok_buy
        sell = sell & ma_ok_sell
    if use_vol:
        buy = buy & vol_up
        sell = sell & vol_up

    sig = pd.Series(0, index=dfd.index)
    sig[buy] = 1
    sig[sell] = -1
    return dfd, sig


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


store.init_tables()
all_results = {}
for TICKER in TICKERS:
    df = fetch_1h(TICKER)
    if df is None:
        print(f"{TICKER}: 数据获取失败，跳过"); continue
    days = (df.index[-1] - df.index[0]).days
    per = f"{df.index[0]:%Y-%m-%d}~{df.index[-1]:%Y-%m-%d}"
    buy_hold = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    print(f"\n>>> {TICKER}（{NAME[TICKER]}）: {len(df)}根/{days}天 ({per})  买入持有 {buy_hold:+.1f}%")

    out = []
    for vname, (use_div, use_ma, use_vol) in VARIANTS.items():
        dfd, sig = build_variant_signals(df, use_div, use_ma, use_vol)
        n_div = int(sig[sig != 0].sum())
        for mode in MODES:
            for tp, sl in TPSL:
                r = simulate(df, sig, tp, sl, mode)
                if r is None:
                    continue
                out.append({'ticker': TICKER, 'name': NAME[TICKER], 'strat': vname,
                            'mode': MODE_NAME[mode], 'tpsl': TPSL_NAME[(tp, sl)],
                            'period': per, 'days': days, 'n_sig': n_div, **r})
    out.sort(key=lambda x: -x['ret'])
    all_results[TICKER] = out

    print(f"  {'变体':<14}{'模式':<10}{'TP/SL':<7}{'收益':>9}{'回撤':>8}{'笔':>4}{'信号':>5}{'夏普':>7}")
    for o in out[:12]:
        print(f"  {o['strat']:<14}{o['mode']:<10}{o['tpsl']:<7}{o['ret']:>+8.1f}%{o['mdd']:>7.1f}%"
              f"{o['n']:>4}{o['n_sig']:>5}{'' if o['sh'] is None else format(round(o['sh'], 2), '>7')}")

    # 入库：每标的最优「仅做多」变体 + 全场最高(供参考)
    only = [o for o in out if o['mode'] == '仅做多']
    only.sort(key=lambda x: -x['ret'])
    top = only[0] if only else out[0]
    top_global = out[0]
    note = (f"{CHANNEL}，{NAME[TICKER]}美股真实股票{TICKER}，{days}天{top['n']}笔；"
            f"标的自身买入持有{buy_hold:+.1f}%；变体=「{top['strat']}」{top['tpsl']}；"
            + (f"全场最高为双向模拟「{top_global['strat']}」{top_global['ret']:+.1f}%（做空仅参考）"
               if top_global['ret'] > top['ret'] else "仅做多即全场最高"))
    rec = {'symbol': TICKER, 'timeframe': '1h', 'strategy': top['strat'],
           'mode': top['mode'], 'tpsl': top['tpsl'],
           'ret': f"{top['ret']:+.1f}%", 'daily': f"{top['ret'] / max(top['days'], 1):+.2f}%",
           'monthly': f"{top['ret'] / max(top['days'] / 30, 1):+.2f}%",
           'trades': str(top['n']), 'winrate': '—', 'mdd': f"{top['mdd']:.1f}%",
           'stability': f"{top['days']}天{top['n']}笔",
           'market': MARKET_LABEL,
           'source': f"{CHANNEL}（{per}）",
           'note': note,
           'sharpe': None if top['sh'] is None else f"{top['sh']:.2f}",
           'period': per}
    ok, msg = store.add_history(rec)
    print(f"  入库({TICKER}): 「{top['strat']}」{top['mode']} {top['tpsl']} {top['ret']:+.1f}% → {msg}")

os.makedirs('scripts/results', exist_ok=True)
with open('scripts/results/macd_divergence_1h_2025.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=1, default=float)
print('\nsaved → scripts/results/macd_divergence_1h_2025.json')
