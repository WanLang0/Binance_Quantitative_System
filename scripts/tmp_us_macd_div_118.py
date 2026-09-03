# -*- coding: utf-8 -*-
"""MACD+背离 变体全量回测：92 只币安美股永续（Yahoo 1h，2025-01-01~今，仅做多不设止盈止损）

3 个策略变体（第3个「量能+量能」为用户笔误，已修正为「量能+均线过滤」）：
  1) MACD+背离                use_div=True,  use_ma=False, use_vol=False
  2) MACD+背离+量能            use_div=True,  use_ma=False, use_vol=True
  3) MACD+背离+量能+均线        use_div=True,  use_ma=True,  use_vol=True
模式=仅做多，TP/SL=不设，手续费 0.1%，初始资金 10000。
输出：每标的每策略 收益/回撤/笔数/夏普；跨标的均值/胜率/稳健性排名。
说明：CXMT/UNITREE/BRKB 在 Yahoo 无可用的 1h 行情（未上市/数据不可用），已跳过，实测 92 只。
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
from datetime import datetime

COMM = 0.001
PIVOT_ORDER = 5
VOL_MULT = 1.5
START = '2025-01-01'
CHANNEL = "Yahoo Finance 美股1h行情"

# 92 只币安美股永续（Yahoo 可测）
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

# Yahoo 别名映射（币安特殊代码 -> Yahoo ticker）
YH_ALIAS = {
    'SKHYNIX': '000660.KS',
}
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

def fetch_1h(ticker, start=START, tries=5):
    for i in range(tries):
        try:
            df = yf.download(ticker, interval='1h', start=start, end=datetime.now(),
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
        except Exception as e:
            print('  fetch err:', repr(e)[:70], file=sys.stderr)
        time.sleep(3)
    return None

# ---- 背离检测 ----
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

def _macd_series(df, fast=12, slow=26, signal=9):
    macd = ta.trend.MACD(df['close'], window_fast=fast, window_slow=slow, window_sign=signal)
    df = df.copy()
    df['MACD']=macd.macd(); df['MACD_signal']=macd.macd_signal(); df['hist']=macd.macd_diff()
    return df

def _sma(col, period=20):
    return ta.trend.SMAIndicator(col, window=period).sma_indicator()

def build_variant_signals(df, use_div, use_ma, use_vol):
    dfd = _macd_series(df)
    macd_buy = (dfd['MACD']>dfd['MACD_signal']) & (dfd['MACD'].shift(1)<=dfd['MACD_signal'].shift(1))
    macd_sell = (dfd['MACD']<dfd['MACD_signal']) & (dfd['MACD'].shift(1)>=dfd['MACD_signal'].shift(1))
    if use_div:
        top_div, bot_div = _divergence(dfd)
        buy = macd_buy | bot_div; sell = macd_sell | top_div
    else:
        buy = macd_buy.copy(); sell = macd_sell.copy()
    dfd['sma20'] = _sma(dfd['close'], 20)
    dfd['vol_ma20'] = dfd['volume'].rolling(20).mean()
    dfd['vol_up'] = dfd['volume'] > dfd['vol_ma20']*VOL_MULT
    close = dfd['close']; sma = dfd['sma20']; vol_up = dfd['vol_up']
    if use_ma:
        buy = buy & (close>sma); sell = sell & (close<sma)
    if use_vol:
        buy = buy & vol_up; sell = sell & vol_up
    sig = pd.Series(0, index=dfd.index); sig[buy]=1; sig[sell]=-1
    return sig

VARIANTS = {
    'MACD+背离':       dict(use_div=True,  use_ma=False, use_vol=False),
    'MACD+背离+量能':   dict(use_div=True,  use_ma=False, use_vol=True),
    'MACD+背离+量能+均线': dict(use_div=True, use_ma=True,  use_vol=True),
}
VLIST = list(VARIANTS.keys())

def _close(cash, units, entry, price, side, comm=COMM):
    if side>0: return cash + units*price*(1-comm)
    return cash + units*entry + (entry-price)*units - units*price*comm

def simulate(df, signals, tp=None, sl=None, mode='long_only', initial=10000.0, comm=COMM):
    cash=initial; units=0.0; entry=0.0; side=0; n=0; eq_pts=[]
    for i,(ts,row) in enumerate(df.iterrows()):
        price=row['close']
        if not np.isfinite(price) or price<=0: continue
        sig=int(signals.iloc[i]) if i>0 else 0
        if side!=0 and entry>0:
            r=(price-entry)/entry if side>0 else (entry-price)/entry
            if (tp and r>=tp) or (sl and r<=-sl):
                cash=_close(cash,units,entry,price,side); side=0; units=0; n+=1
                eq_pts.append((ts,cash)); continue
        eq=cash+(units*price if side>0 else units*entry+(entry-price)*units if side<0 else 0)
        if eq<=0: return None
        eq_pts.append((ts,eq))
        if sig==1 and side<=0:
            if side<0:
                cash=_close(cash,units,entry,price,side); n+=1; side=0; units=0
            u=(cash*0.95)/(price*(1+comm))
            if u>0: cash-=u*price*(1+comm); units=u; entry=price; side=1
        elif sig==-1 and side>=0:
            if side>0:
                cash=_close(cash,units,entry,price,side); n+=1; side=0; units=0
            if mode=='long_short':
                u=(cash*0.95)/price
                if u>0: cash-=u*price*(1+comm); units=u; entry=price; side=-1
    if side!=0 and len(df)>0:
        price=df['close'].iloc[-1]
        cash=_close(cash,units,entry,price,side); n+=1
        eq_pts.append((df.index[-1],cash))
    if n==0: return None
    eq=pd.Series(dict(eq_pts)).sort_index()
    peak=eq.cummax(); mdd=((eq-peak)/peak*100).min()
    rr=eq.pct_change().dropna()
    sh=float(rr.mean()/rr.std()*np.sqrt(1764)) if len(rr)>=10 and rr.std()>0 else None
    return {'ret':(eq.iloc[-1]/initial-1)*100,'mdd':mdd,'n':n,'sh':sh}

def main():
    results={}; failed=[]
    for T in TICKERS:
        yh = YH_ALIAS.get(T, T)
        df = fetch_1h(yh)
        if df is None:
            failed.append(T); print(f"{T}: 获取失败，跳过"); continue
        sigs={}
        for v,cfg in VARIANTS.items():
            try: sigs[v]=build_variant_signals(df, cfg['use_div'], cfg['use_ma'], cfg['use_vol'])
            except Exception as e: print(f"{T} {v} 信号出错: {e}")
        row={}
        for v,cfg in VARIANTS.items():
            if v not in sigs: row[v]=None; continue
            r=simulate(df, sigs[v])
            row[v]=r
        # 买入持有基准
        bh=(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100
        row['_bh']=bh; row['_nbar']=len(df)
        results[T]=row
        r0=row[VLIST[0]]
        s0=f"{r0['ret']:+.1f}%/n{r0['n']}" if r0 else "无交易"
        print(f"{T:6}({NAME.get(T,T)}) 总bar{len(df)} BH{bh:+.1f}%  | {VLIST[0]}:{s0}")
    print(f"\n获取失败跳过: {failed}")

    # 汇总
    print("\n\n########## 策略汇总 ##########")
    print(f"标的数={len(results)}  窗口 {START} ~ 今  模式=仅做多  手续费{COMM:.1%}")
    print(f"{'标的':<6}{'BH%':>8}", end='')
    for v in VLIST: print(f"{v:>22}", end='')
    print()
    # 每个策略的跨标的统计
    stat={}
    for v in VLIST:
        rets=[results[t][v]['ret'] for t in results if results[t][v]]
        mdd=[results[t][v]['mdd'] for t in results if results[t][v]]
        ns=[results[t][v]['n'] for t in results if results[t][v]]
        sh=[results[t][v]['sh'] for t in results if results[t][v] and results[t][v]['sh'] is not None]
        sh_n=len([x for x in sh if x is not None])
        pos=sum(1 for x in rets if x>0)
        stat[v]=dict(cnt=len(rets), avg=np.mean(rets), med=np.median(rets), pos=pos/len(rets) if rets else 0,
                     avg_mdd=np.mean(mdd), avg_n=np.mean(ns), avg_sh=np.mean(sh) if sh else None, n_sh=sh_n)
    bh_list=[results[t]['_bh'] for t in results]
    stat['_BH']=dict(cnt=len(bh_list), avg=np.mean(bh_list), pos=sum(1 for x in bh_list if x>0)/len(bh_list))
    print(f"\n跨标的平均：BH {stat['_BH']['avg']:+.1f}%")
    for v in VLIST:
        s=stat[v]
        shs=f"{s['avg_sh']:+.2f}" if s['avg_sh'] is not None else "N/A"
        print(f"  {v:<22} 平均收益 {s['avg']:+.1f}%  中位 {s['med']:+.1f}%  胜率 {s['pos']:.0%}  "
              f"平均回撤 {s['avg_mdd']:.1f}% 平均笔数 {s['avg_n']:.1f} 平均夏普 {shs}")

    # 每标的每策略详细
    print("\n—— 每标的明细 ——")
    for t in sorted(results):
        row=results[t]
        print(f"{t:6}({NAME.get(t,t)}) BH{row['_bh']:+7.1f}%  ", end='')
        for v in VLIST:
            r=row[v]
            if r: print(f"{v} {r['ret']:+.1f}%/mdd{r['mdd']:.0f}%/n{r['n']}/sh{(r['sh'] if r['sh'] is None else round(r['sh'],2))}  | ", end='')
            else: print(f"{v} 无  | ", end='')
        print()

    # 稳健性：每策略跑赢BH的标的数
    print("\n—— 跑赢买入持平对比 ——")
    for v in VLIST:
        beat=sum(1 for t in results if results[t][v] and results[t][v]['ret']>results[t]['_bh'])
        tot=sum(1 for t in results if results[t][v])
        print(f"  {v:<24} 跑赢BH {beat}/{tot}")

    os.makedirs('scripts/results', exist_ok=True)
    with open('scripts/results/us_macd_div_1h_2025_2026.json','w',encoding='utf-8') as f:
        json.dump({'channel':CHANNEL,'start':START,'variants':VLIST,'results_per_ticker':results,
                   'stat':stat,'failed':failed}, f, ensure_ascii=False, indent=1, default=float)
    print(f"\nsaved → scripts/results/us_macd_div_1h_2025_2026.json")

if __name__=='__main__':
    main()
