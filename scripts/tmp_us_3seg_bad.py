# -*- coding: utf-8 -*-
"""综合量化清单 95 只美股：2024(9-12月)/2025/2026(至今) 三段 × 3策略（仅做多无TP/SL）
目标：找出三段表现都不好的股票。
判据：
  A档(严格)：每个有数据的年份窗口，三个策略收益全部为负
  B档(均值)：每个有数据的年份窗口，三策略平均收益为负
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
SEGS = {'2024': ('2024-09-06', '2025-01-01'),
        '2025': ('2025-01-01', '2026-01-01'),
        '2026': ('2026-01-01', None)}

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
NAME = dict([
    ('NVDA','英伟达'),('QQQ','纳指100 ETF'),('TQQQ','纳指100 3x'),('AAPL','苹果'),('MSFT','微软'),
    ('GOOGL','谷歌'),('AMZN','亚马逊'),('META','Meta'),('TSLA','特斯拉'),('MU','美光'),('MUU','美光2x'),
    ('SNDK','闪迪'),('SNXX','闪迪2x'),('SKHYNIX','SK海力士'),('WDC','西部数据'),('STXX','希捷'),
    ('NOK','诺基亚'),('LITE','Lumentum'),('MRVL','迈威尔'),('GLW','康宁'),('COHR','Coherent'),
    ('CIEN','Ciena'),('ALAB','Astera Labs'),('CRDO','Credo'),('AVGO','博通'),('AMD','超威'),
    ('INTC','英特尔'),('LRCX','拉姆研究'),('AMAT','应用材料'),('TXN','德州仪器'),('TSM','台积电'),
    ('ASML','阿斯麦'),('HPE','慧与'),('CRM','赛富时'),('ORCL','甲骨文'),('BABA','阿里'),('IBM','IBM'),
    ('NOW','ServiceNow'),('SMCI','超微'),('CRWD','CrowdStrike'),('DDOG','Datadog'),('PANW','Palo Alto'),
    ('PLTR','Palantir'),('SHOP','Shopify'),('TEAM','Atlassian'),('ADBE','Adobe'),('APP','AppLovin'),
    ('MSTR','MicroStrategy'),('NBIS','Nebius'),('CRWV','CoreWeave'),('PYPL','PayPal'),('SOXL','半导体3x'),
    ('NVDL','英伟达2x'),('TSLL','特斯拉2x'),('MVLL','迈威尔2x'),('ASTS','AST SpaceM'),('RKLB','Rocket Lab'),
    ('SONY','索尼'),('JPM','摩根大通'),('V','Visa'),('GS','高盛'),('BX','黑石'),('HOOD','Robinhood'),
    ('SOFI','SoFi'),('COIN','Coinbase'),('LLY','礼来'),('MRK','默沙东'),('NVO','诺和诺德'),('MRNA','Moderna'),
    ('HIMS','Hims'),('TEM','Tempus'),('BNC','Bannix'),('XBI','生物科技ETF'),('GDX','黄金矿业'),
    ('XLE','能源ETF'),('BE','Bloom Energy'),('FLNC','Fluence'),('GEV','GE Vernova'),('VST','Vistra'),
    ('RIVN','Rivian'),('CAT','卡特彼勒'),('COST','开市客'),('HD','家得宝'),('DIS','迪士尼'),('WMT','沃尔玛'),
    ('KO','可口可乐'),('NFLX','奈飞'),('QCOM','高通'),('KLAC','KLA'),('TER','泰瑞达'),('TTWO','Take-Two'),
    ('CSCO','思科'),
])

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
            if len(df) >= 200: return df
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

def simulate_long(df, signals, initial=10000.0, comm=COMM):
    cash=initial; units=0.0; entry=0.0; side=0; n=0; eq_pts=[]
    for i,(ts,row) in enumerate(df.iterrows()):
        price=row['close']
        if not np.isfinite(price) or price<=0: continue
        sig=int(signals.iloc[i]) if i>0 else 0
        eq=cash+units*price
        if eq<=0: return None
        eq_pts.append((ts,eq))
        if sig==1 and side==0:
            u=(cash*0.95)/(price*(1+comm))
            if u>0: cash-=u*price*(1+comm); units=u; entry=price; side=1
        elif sig==-1 and side>0:
            cash=cash+units*price*(1-comm); side=0; units=0; n+=1
    if side!=0 and len(df)>0:
        price=df['close'].iloc[-1]
        cash=cash+units*price*(1-comm); n+=1
        eq_pts.append((df.index[-1],cash))
    if n==0: return None
    eq=pd.Series(dict(eq_pts)).sort_index()
    peak=eq.cummax(); mdd=((eq-peak)/peak*100).min()
    rr=eq.pct_change().dropna()
    sh=float(rr.mean()/rr.std()*np.sqrt(1764)) if len(rr)>=10 and rr.std()>0 else None
    return {'ret':(eq.iloc[-1]/initial-1)*100,'mdd':mdd,'n':n,'sh':sh}

def main():
    allres={}
    for T in TICKERS:
        yh = YH_ALIAS.get(T, T)
        df = fetch_1h(yh)
        if df is None:
            print(f"{T:6} 无数据"); continue
        seg={}
        for yr,(a,b) in SEGS.items():
            d = df[df.index>=a]
            if b: d = d[d.index<b]
            if len(d) < 200:
                seg[yr]=None; continue
            row={}
            for v,cfg in VARIANTS.items():
                try:
                    sig = build_variant_signals(d, cfg['use_div'], cfg['use_ma'], cfg['use_vol'])
                    r = simulate_long(d, sig)
                    row[v] = r['ret'] if r else None
                    row[v+'_mdd'] = r['mdd'] if r else None
                except Exception:
                    row[v]=None; row[v+'_mdd']=None
            row['_bh']=(d['close'].iloc[-1]/d['close'].iloc[0]-1)*100
            row['_nbar']=len(d)
            seg[yr]=row
        allres[T]=seg
        def _f(x): return f"{x:+.0f}%" if x is not None else "—"
        msg=f"{T:6}"
        for yr in SEGS:
            s=seg[yr]
            if not s: msg+=f"  {yr}:无数据"; continue
            avg=np.mean([s[v] for v in VLIST if s[v] is not None]) if any(s[v] is not None for v in VLIST) else None
            msg+=f"  {yr}:avg{_f(avg)}/BH{_f(s['_bh'])}"
        print(msg)

    # 判定
    print("\n\n########## 三年表现都不好的股票 ##########")
    badA=[]; badB=[]
    for T,seg in allres.items():
        yrs=[y for y in SEGS if seg.get(y)]
        if not yrs: continue
        strict = all(all((seg[y][v] is not None and seg[y][v]<0) for v in VLIST) for y in yrs)
        avg_neg = all((np.mean([seg[y][v] for v in VLIST if seg[y][v] is not None])<0) if any(seg[y][v] is not None for v in VLIST) else False for y in yrs)
        if strict: badA.append(T)
        elif avg_neg: badB.append(T)
    print(f"A档(每年三策略全亏): {len(badA)}只 -> {badA}")
    print(f"B档(每年平均亏损):   {len(badB)}只 -> {badB}")

    # 明细表（按三年平均收益排序）
    print("\n—— A/B档明细（按全期平均升序）——")
    def total_avg(T):
        seg=allres[T]; vals=[]
        for y in SEGS:
            s=seg.get(y)
            if s:
                vs=[s[v] for v in VLIST if s[v] is not None]
                if vs: vals.append(np.mean(vs))
        return np.mean(vals) if vals else 0
    for T in sorted(badA+badB, key=total_avg):
        seg=allres[T]; print(f"\n{T:6}({NAME.get(T,T)})")
        for y in SEGS:
            s=seg.get(y)
            if not s: print(f"   {y}: 无数据(<200bar)"); continue
            def _f2(x): return f"{x:+7.1f}%" if x is not None else "   无交易"
            print(f"   {y}: " + "  ".join(f"{v}:{_f2(s[v])}" for v in VLIST) + f"  BH:{_f2(s['_bh'])}")

    os.makedirs('scripts/results', exist_ok=True)
    out='scripts/results/us_macd_div_3seg_bad.json'
    with open(out,'w',encoding='utf-8') as f:
        json.dump({'segments':{k:list(v) for k,v in SEGS.items()},'variants':VLIST,
                   'results':allres,'badA':badA,'badB':badB}, f, ensure_ascii=False, indent=1, default=float)
    print(f"\nsaved → {out}")

if __name__=='__main__':
    main()
