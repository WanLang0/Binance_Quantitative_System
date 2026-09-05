# -*- coding: utf-8 -*-
"""美股 MACD 组合策略推荐三档：每笔下单赚钱/亏钱概率统计
复用 tmp_us_macd_div_matrix.py 口径（Yahoo 1h、2024(9月起)/2025/2026YTD 分年、
初始1万、95%仓位、单边手续费0.1%、年末强平）；K线缓存到 scripts/cache/us_*.pkl
"""
import os, sys, io, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ta

PROXY = "http://127.0.0.1:7892"
os.environ.setdefault("HTTP_PROXY", PROXY); os.environ.setdefault("HTTPS_PROXY", PROXY)

COMM = 0.001
INITIAL = 10000.0
YEAR_BARS_MIN = 200
PIVOT_ORDER = 5
VOL_MULT = 1.5
SEGMENTS = {'2024': ('2024-09-06', '2025-01-01'),
            '2025': ('2025-01-01', '2026-01-01'),
            '2026YTD': ('2026-01-01', None)}

TICKERS = ['NVDA','QQQ','TQQQ','AAPL','MSFT','GOOGL','AMZN','META','TSLA',
           'MU','MUU','SNDK','SNXX','SKHYNIX','WDC','STXX',
           'NOK','LITE','MRVL','GLW','COHR','CIEN','ALAB','CRDO',
           'AVGO','AMD','INTC','LRCX','AMAT','TXN','TSM','ASML',
           'HPE','CRM','ORCL','BABA','IBM','NOW','SMCI',
           'CRWD','DDOG','PANW','PLTR','SHOP','TEAM','ADBE','APP','MSTR','NBIS','CRWV','PYPL',
           'SOXL','NVDL','TSLL','MVLL',
           'ASTS','RKLB','SONY',
           'JPM','V','GS','BX','HOOD','SOFI','COIN',
           'LLY','MRK','NVO','MRNA','HIMS','TEM','BNC','XBI',
           'GDX','XLE','BE','FLNC','GEV','VST','RIVN','CAT',
           'COST','HD','DIS','WMT','KO','NFLX','QCOM','KLAC','TER','TTWO','CSCO']
YH_ALIAS = {'SKHYNIX': '000660.KS'}

# 推荐三档（与最优策略页推荐表一致）
CONFIGS = [
    ('MACD+背离', 'long_only', 'tpsl5', '背离 仅做多 止盈止损各5%'),
    ('MACD+背离', 'long_only', 'none', '背离 仅做多 不设'),
    ('MACD+背离+量能+均线', 'long_only', 'none', '背离+量能+均线 仅做多 不设'),
]
TPSL_CFG = {'none': (None, None), 'sl5': (None, 0.05), 'tpsl5': (0.05, 0.05)}
VARIANTS = {
    'MACD+背离': dict(use_div=True, use_ma=False, use_vol=False),
    'MACD+背离+量能+均线': dict(use_div=True, use_ma=True, use_vol=True),
}

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, 'cache')

_sess = None
def _get_session():
    global _sess
    if _sess: return _sess
    s = requests.Session()
    s.headers['User-Agent'] = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                               'Chrome/120.0 Safari/537.36')
    s.proxies = {'http': PROXY, 'https': PROXY}
    retry = Retry(total=3, backoff_factor=1.0, status_forcelist=[429,500,502,503,504])
    s.mount('https://', HTTPAdapter(max_retries=retry))
    try:
        s.get('https://query1.finance.yahoo.com/v8/finance/chart/QQQ?range=1d&interval=1d', timeout=10)
    except Exception:
        pass
    _sess = s
    return s


def fetch_1h_cached(ticker, ttl=6 * 3600):
    p = os.path.join(CACHE_DIR, f'us_{ticker.replace(".", "_")}_1h.pkl')
    if os.path.exists(p) and time.time() - os.path.getmtime(p) < ttl:
        try:
            return pd.read_pickle(p)
        except Exception:
            pass
    lo = (datetime.now() - timedelta(days=728)).strftime('%Y-%m-%d')
    for i in range(4):
        try:
            df = yf.download(ticker, interval='1h', start=lo, end=datetime.now(),
                             progress=False, auto_adjust=True, session=_get_session())
            if df is None or df.empty:
                time.sleep(4); continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[['Open','High','Low','Close','Volume']].copy()
            df.columns = ['open','high','low','close','volume']
            df.index.name = 'timestamp'
            df = df.dropna(subset=['close']); df = df[df['close'] > 0]
            if len(df) >= 200:
                try:
                    df.to_pickle(p)
                except Exception:
                    pass
                return df
            return None
        except Exception as e:
            print('  fetch err:', repr(e)[:70], file=sys.stderr)
        time.sleep(3)
    return None


def _find_pivots(s, order=PIVOT_ORDER, kind='high'):
    arr = s.to_numpy(); piv = np.zeros(len(arr), dtype=bool)
    for i in range(order, len(arr)-order):
        win = arr[i-order:i+order+1]
        if kind=='high':
            piv[i] = np.argmax(win)==order and arr[i]==win.max()
            piv[i] = piv[i] and arr[i]>arr[i-1]
        else:
            piv[i] = np.argmin(win)==order and arr[i]==win.min()
            piv[i] = piv[i] and arr[i]<arr[i-1]
    return pd.Series(piv, index=s.index)


def _divergence(df, macd_col='MACD', order=PIVOT_ORDER):
    px_high = _find_pivots(df['high'], order, 'high')
    px_low  = _find_pivots(df['low'],  order, 'low')
    macd = df[macd_col].to_numpy(); hi = df['high'].to_numpy(); lo = df['low'].to_numpy()
    top = np.zeros(len(df), dtype=bool); bot = np.zeros(len(df), dtype=bool)
    ph=None; pm=None
    for i in np.where(px_high.to_numpy())[0]:
        p=hi[i]; m=macd[i]
        if ph is not None and p>ph and m<pm: top[i]=True
        ph=p; pm=m
    pl=None; pm2=None
    for i in np.where(px_low.to_numpy())[0]:
        p=lo[i]; m=macd[i]
        if pl is not None and p<pl and m>pm2: bot[i]=True
        pl=p; pm2=m
    return pd.Series(top, index=df.index), pd.Series(bot, index=df.index)


def build_variant_signals(df, use_div, use_ma, use_vol):
    macd = ta.trend.MACD(df['close'])
    df = df.copy()
    df['MACD']=macd.macd(); df['MACD_signal']=macd.macd_signal()
    macd_buy = (df['MACD']>df['MACD_signal']) & (df['MACD'].shift(1)<=df['MACD_signal'].shift(1))
    macd_sell = (df['MACD']<df['MACD_signal']) & (df['MACD'].shift(1)>=df['MACD_signal'].shift(1))
    if use_div:
        top_div, bot_div = _divergence(df)
        buy = macd_buy | bot_div; sell = macd_sell | top_div
    else:
        buy = macd_buy.copy(); sell = macd_sell.copy()
    sma = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
    vol_ma = df['volume'].rolling(20).mean()
    vol_up = df['volume'] > vol_ma*VOL_MULT
    if use_ma:
        buy = buy & (df['close']>sma); sell = sell & (df['close']<sma)
    if use_vol:
        buy = buy & vol_up; sell = sell & vol_up
    sig = pd.Series(0, index=df.index); sig[buy]=1; sig[sell]=-1
    return sig


def simulate_trades(df, signals, tp=None, sl=None, mode='long_only',
                    initial=INITIAL, comm=COMM):
    close = df['close'].to_numpy(); sig = signals.to_numpy()
    idx = df.index
    cash = initial; units = 0.0; entry = 0.0; side = 0; cost = 0.0
    trades = []

    def close_pos(price):
        nonlocal cash, units, side
        if side > 0:
            proceeds = units * price * (1 - comm)
        else:
            proceeds = units * entry + (entry - price) * units - units * price * comm
        ret = (proceeds - cost) / cost * 100
        d = '多' if side > 0 else '空'
        cash += proceeds
        units = 0.0; side = 0
        return ret, d

    for i in range(len(df)):
        price = close[i]
        if not np.isfinite(price) or price <= 0:
            continue
        s = int(sig[i]) if i > 0 else 0
        if side != 0 and entry > 0:
            r = (price - entry) / entry if side > 0 else (entry - price) / entry
            if (tp and r >= tp) or (sl and r <= -sl):
                ret, d = close_pos(price)
                trades.append((ret, d))
                continue
        eq = cash + (units * price if side > 0 else units * entry + (entry - price) * units if side < 0 else 0)
        if eq <= 0:
            break
        if s == 1 and side <= 0:
            if side < 0:
                ret, d = close_pos(price)
                trades.append((ret, d))
            u = (cash * 0.95) / (price * (1 + comm))
            if u > 0:
                cash -= u * price * (1 + comm)
                cost = u * price * (1 + comm)
                units = u; entry = price; side = 1
        elif s == -1 and side >= 0:
            if side > 0:
                ret, d = close_pos(price)
                trades.append((ret, d))
            if mode == 'long_short':
                u = (cash * 0.95) / price
                if u > 0:
                    cash -= u * price * (1 + comm)
                    cost = u * price * (1 + comm)
                    units = u; entry = price; side = -1
    if side != 0 and len(df) > 0:
        ret, d = close_pos(close[-1])
        trades.append((ret, d))
    return trades


def main():
    stats = {c[3]: {'rets': [], 'year': {}, 'dir': Counter(), 'dir_win': Counter()} for c in CONFIGS}
    for k, T in enumerate(TICKERS, 1):
        yh = YH_ALIAS.get(T, T)
        df = fetch_1h_cached(yh)
        if df is None:
            print(f'[{k}/{len(TICKERS)}] {T:8} 无数据'); continue
        for vname, cfg in VARIANTS.items():
            need = any(c[0] == vname for c in CONFIGS)
            if not need:
                continue
            try:
                sig = build_variant_signals(df, cfg['use_div'], cfg['use_ma'], cfg['use_vol'])
            except Exception:
                continue
            for yr, (a, b) in SEGMENTS.items():
                d = df[df.index >= a]
                if b: d = d[d.index < b]
                if len(d) < YEAR_BARS_MIN:
                    continue
                for cv, cmode, ctpsl, cname in CONFIGS:
                    if cv != vname:
                        continue
                    tp, sl = TPSL_CFG[ctpsl]
                    tr = simulate_trades(d, sig.loc[d.index], tp=tp, sl=sl, mode=cmode)
                    st = stats[cname]
                    st['rets'].extend(tr)
                    py = st['year'].setdefault(yr, [0, 0])
                    py[0] += len(tr)
                    py[1] += sum(1 for r, _ in tr if r > 0)
                    for r, dd in tr:
                        st['dir'][dd] += 1
                        if r > 0:
                            st['dir_win'][dd] += 1
        n0 = sum(len(v) for v in stats.values())
        print(f'[{k}/{len(TICKERS)}] {T:8} 累计{sum(s["dir"]["多"]+s["dir"]["空"] for s in stats.values())}笔', flush=True)

    print('\n========== 美股（Yahoo 1h · 推荐三档） ==========')
    for cname, st in stats.items():
        rets = [r for r, _ in st['rets']]
        n = len(rets)
        if n == 0:
            continue
        wins = [r for r in rets if r > 0]
        losses = [r for r in rets if r <= 0]
        aw = np.mean(wins) if wins else 0
        al = np.mean(losses) if losses else 0
        print('\n【%s】 总%d笔' % (cname, n))
        print('  总胜率: %d/%d = %.1f%%' % (len(wins), n, len(wins) / n * 100))
        print('  均盈利单 %+.2f%% / 均亏损单 %+.2f%% / 盈亏比 %.2f' % (aw, al, (aw / abs(al)) if al else float('inf')))
        print('  最大单笔盈利 %+.1f%% / 最大单笔亏损 %.1f%%' % (max(wins) if wins else 0, min(losses) if losses else 0))
        print('  分年胜率:', '  '.join('%s:%.1f%%(%d/%d)' % (y, w[1] / w[0] * 100, w[1], w[0])
                                       for y, w in sorted(st['year'].items())))
        print('  分方向胜率:', '  '.join('%s:%.1f%%(%d/%d)' % (dd, st['dir_win'][dd] / st['dir'][dd] * 100,
                                                        st['dir_win'][dd], st['dir'][dd]) for dd in st['dir']))


if __name__ == '__main__':
    main()
