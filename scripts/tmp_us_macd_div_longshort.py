# -*- coding: utf-8 -*-
"""MACD+背离 3变体 · 多空双向 · 无止盈止损 双窗口回测
窗口A: 2024-09-06 ~ 2025-12-31（Yahoo 1h 730天限制下的2024-2025窗口）
窗口B: 2025-01-01 ~ 今（2025-2026窗口）
模式=long_short：+1信号平空开多，-1信号平多开空。手续费0.1%。
每个标的只拉一次数据后切片。
"""
import os, sys, io, time, warnings, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
PROXY = "http://127.0.0.1:7892"
os.environ.setdefault("HTTP_PROXY", PROXY); os.environ.setdefault("HTTPS_PROXY", PROXY)

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ta
from datetime import datetime, timedelta

COMM = 0.001
PIVOT_ORDER = 5
VOL_MULT = 1.5
WIN_A = ('2024-09-06', '2026-01-01')
WIN_B = ('2025-01-01', None)

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

_sess = None
def _get_session():
    global _sess
    if _sess: return _sess
    s = requests.Session()
    s.headers['User-Agent'] = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36')
    s.proxies = {'http': PROXY, 'https': PROXY}
    retry = Retry(total=3, backoff_factor=1.0, status_forcelist=[429,500,502,503,504])
    s.mount('https://', HTTPAdapter(max_retries=retry))
    try:
        s.get('https://query1.finance.yahoo.com/v8/finance/chart/QQQ?range=1d&interval=1d', timeout=10)
    except Exception:
        pass
    _sess = s
    return s

def fetch_1h(ticker, tries=4):
    lo = (datetime.now() - timedelta(days=728)).strftime('%Y-%m-%d')
    for i in range(tries):
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
            df = df.dropna(subset=['close']); df = df[df['close']>0]
            if len(df) >= 200:
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

VARIANTS = {
    'MACD+背离':       dict(use_div=True,  use_ma=False, use_vol=False),
    'MACD+背离+量能':   dict(use_div=True,  use_ma=False, use_vol=True),
    'MACD+背离+量能+均线': dict(use_div=True,  use_ma=True,  use_vol=True),
}
VLIST = list(VARIANTS.keys())

def _close(cash, units, entry, price, side, comm=COMM):
    if side>0: return cash + units*price*(1-comm)
    return cash + units*entry + (entry-price)*units - units*price*comm

def simulate_ls(df, signals, initial=10000.0, comm=COMM):
    """多空双向：+1平空开多；-1平多开空。无TP/SL。"""
    cash=initial; units=0.0; entry=0.0; side=0; n=0; nl=0; ns=0; eq_pts=[]
    for i,(ts,row) in enumerate(df.iterrows()):
        price=row['close']
        if not np.isfinite(price) or price<=0: continue
        sig=int(signals.iloc[i]) if i>0 else 0
        eq=cash+(units*price if side>0 else units*entry+(entry-price)*units if side<0 else 0)
        if eq<=0: return None
        eq_pts.append((ts,eq))
        if sig==1 and side<=0:
            if side<0:
                cash=_close(cash,units,entry,price,side); n+=1; ns+=1; side=0; units=0
            u=(cash*0.95)/(price*(1+comm))
            if u>0: cash-=u*price*(1+comm); units=u; entry=price; side=1; nl+=1
        elif sig==-1 and side>=0:
            if side>0:
                cash=_close(cash,units,entry,price,side); n+=1; side=0; units=0
            u=(cash*0.95)/price
            if u>0: cash-=u*price*(1+comm); units=u; entry=price; side=-1; ns+=1
    if side!=0 and len(df)>0:
        price=df['close'].iloc[-1]
        cash=_close(cash,units,entry,price,side); n+=1
        eq_pts.append((df.index[-1],cash))
    if n==0: return None
    eq=pd.Series(dict(eq_pts)).sort_index()
    peak=eq.cummax(); mdd=((eq-peak)/peak*100).min()
    rr=eq.pct_change().dropna()
    sh=float(rr.mean()/rr.std()*np.sqrt(1764)) if len(rr)>=10 and rr.std()>0 else None
    return {'ret':(eq.iloc[-1]/initial-1)*100,'mdd':mdd,'n':n,'sh':sh,'nl':nl,'ns':ns}

def run_window(df, w):
    a, b = w
    d = df[df.index >= a]
    if b: d = d[d.index < b]
    if len(d) < 200: return None
    row = {}
    for v,cfg in VARIANTS.items():
        try:
            sig = build_variant_signals(d, cfg['use_div'], cfg['use_ma'], cfg['use_vol'])
            row[v] = simulate_ls(d, sig)
        except Exception:
            row[v] = None
    row['_bh'] = (d['close'].iloc[-1]/d['close'].iloc[0]-1)*100
    row['_nbar'] = len(d)
    return row

def summarize(res, label):
    print(f"\n########## {label}  多空双向 无TP/SL ##########")
    stat={}
    for v in VLIST:
        rs=[res[t][v] for t in res if res[t][v]]
        if not rs: continue
        rets=[r['ret'] for r in rs]; mdd=[r['mdd'] for r in rs]; ns=[r['n'] for r in rs]
        sh=[r['sh'] for r in rs if r['sh'] is not None]
        stat[v]=dict(cnt=len(rs), avg=float(np.mean(rets)), med=float(np.median(rets)),
                     pos=sum(1 for x in rets if x>0)/len(rs), avg_mdd=float(np.mean(mdd)),
                     over40=sum(1 for x in mdd if x<=-40), avg_n=float(np.mean(ns)),
                     avg_sh=float(np.mean(sh)) if sh else None)
    bh=[res[t]['_bh'] for t in res]
    stat['_BH']=dict(cnt=len(bh), avg=float(np.mean(bh)), pos=sum(1 for x in bh if x>0)/len(bh))
    print(f"标的数={len(res)}  BH平均 {stat['_BH']['avg']:+.1f}%")
    for v in VLIST:
        s=stat[v]
        shs=f"{s['avg_sh']:+.2f}" if s['avg_sh'] is not None else "N/A"
        print(f"  {v:<22} 平均 {s['avg']:+.1f}%  中位 {s['med']:+.1f}%  胜率 {s['pos']:.0%}  "
              f"回撤均值 {s['avg_mdd']:.1f}%  >40%:{s['over40']}/{s['cnt']}  笔数 {s['avg_n']:.0f}  夏普 {shs}")
    for v in VLIST:
        beat=sum(1 for t in res if res[t][v] and res[t][v]['ret']>res[t]['_bh'])
        print(f"  跑赢BH {v}: {beat}/{len(res)}")
    for v in VLIST:
        rows=sorted(((res[t][v]['ret'],t) for t in res if res[t][v]), reverse=True)
        print(f"  Top5[{v}]: " + "  ".join(f"{t}{r:+.0f}%" for r,t in rows[:5]))
        rows2=sorted(((res[t][v]['ret'],t) for t in res if res[t][v]), reverse=False)
        print(f"  Bot3[{v}]: " + "  ".join(f"{t}{r:+.0f}%" for r,t in rows2[:3]))
    return stat

def main():
    resA={}; resB={}; failed=[]
    for T in TICKERS:
        yh = YH_ALIAS.get(T, T)
        df = fetch_1h(yh)
        if df is None:
            failed.append(T); print(f"{T}: 无数据，跳过"); continue
        ra = run_window(df, WIN_A)
        rb = run_window(df, WIN_B)
        if ra: resA[T]=ra
        if rb: resB[T]=rb
        s = f"{T:6} A:{'✓' if ra else '✗'} B:{'✓' if rb else '✗'}"
        if ra and ra[VLIST[0]]: s += f"  A {ra[VLIST[0]]['ret']:+.1f}%(多{ra[VLIST[0]]['nl']}/空{ra[VLIST[0]]['ns']})"
        if rb and rb[VLIST[0]]: s += f"  | B {rb[VLIST[0]]['ret']:+.1f}%(多{rb[VLIST[0]]['nl']}/空{rb[VLIST[0]]['ns']})"
        print(s)
    print(f"\n无数据跳过: {failed}")

    statA = summarize(resA, "窗口A: 2024-09-06~2025-12-31 (2024-2025)")
    statB = summarize(resB, "窗口B: 2025-01-01~今 (2025-2026)")

    print("\n—— 每标的明细（A | B，多/空笔数）——")
    for t in sorted(set(resA)|set(resB)):
        ra=resA.get(t); rb=resB.get(t)
        parts=[]
        for v in VLIST:
            a1=ra.get(v) if ra else None; b1=rb.get(v) if rb else None
            sa=f"{a1['ret']:+.1f}%/mdd{a1['mdd']:.0f}%/L{a1['nl']}S{a1['ns']}" if a1 else "无"
            sb=f"{b1['ret']:+.1f}%/mdd{b1['mdd']:.0f}%/L{b1['nl']}S{b1['ns']}" if b1 else "无"
            parts.append(f"{sa} | {sb}")
        bha=f"{ra['_bh']:+.1f}%" if ra else "无"; bhb=f"{rb['_bh']:+.1f}%" if rb else "无"
        print(f"{t:6} BH {bha}|{bhb}  " + "  ".join(parts))

    os.makedirs('scripts/results', exist_ok=True)
    out='scripts/results/us_macd_div_1h_longshort.json'
    with open(out,'w',encoding='utf-8') as f:
        json.dump({'mode':'long_short','tp':None,'sl':None,'winA':WIN_A,'winB':WIN_B,'variants':VLIST,
                   'results_A':resA,'results_B':resB,'statA':statA,'statB':statB,
                   'failed':failed}, f, ensure_ascii=False, indent=1, default=float)
    print(f"\nsaved → {out}")

if __name__=='__main__':
    main()
