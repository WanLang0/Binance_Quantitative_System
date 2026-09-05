# -*- coding: utf-8 -*-
"""美股 × 三种MACD组合策略 全配置分年回测矩阵（最优策略页表格展示）

与虚拟币 top20_macd_div_summary_2024_2026.json 完全同构：
  3变体 × (仅做多/双向多空) × (不设/只止损5%/止盈止损各5%) = 18组配置
  分2024(自9-06上市起)/2025/2026YTD，输出 ret/med/pos/cnt/mdd/mdd_max/sh/beat/n + 三年合计

用法：
  python scripts/tmp_us_macd_div_matrix.py run    # 拉数+全量回测，存明细+summary
  python scripts/tmp_us_macd_div_matrix.py agg    # 仅从明细JSON重聚合summary（不重拉数据）
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
INITIAL = 10000.0
PERIODS_PER_YEAR = 1764          # 美股1h年化因子（252日×7小时，与美股历史回测一致）
PIVOT_ORDER = 5
VOL_MULT = 1.5
YEAR_BARS_MIN = 200
SEGMENTS = {'2024': ('2024-09-06', '2025-01-01'),
            '2025': ('2025-01-01', '2026-01-01'),
            '2026YTD': ('2026-01-01', None)}
TPSL_CFG = {'不设': (None, None), '只止损5%': (None, 0.05), '止盈止损各5%': (0.05, 0.05)}
MODES = {'仅做多': 'long_only', '双向多空': 'long_short'}

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

VARIANTS = {
    'MACD+背离':           dict(use_div=True,  use_ma=False, use_vol=False),
    'MACD+背离+量能':       dict(use_div=True,  use_ma=False, use_vol=True),
    'MACD+背离+量能+均线':   dict(use_div=True,  use_ma=True,  use_vol=True),
}
VLIST = list(VARIANTS.keys())

OUT_DETAIL = 'scripts/results/us_macd_div_matrix_detail_2024_2026.json'
OUT_SUMMARY = 'scripts/results/us_macd_div_summary_2024_2026.json'

# ---------------- Yahoo 1h 数据（与 tmp_us_3seg_bad.py 同口径） ----------------
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

# ---------------- 信号（与 tmp_us_3seg_bad.py 同口径） ----------------
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

# ---------------- 模拟（与虚拟币 tmp_top20_macd_div_1h_2024_2026.py 同口径） ----------------
def _close(cash, units, entry, price, side, comm=COMM):
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm

def simulate(df, signals, tp=None, sl=None, mode='long_only', initial=INITIAL, comm=COMM):
    idx = df.index.to_numpy(); close = df['close'].to_numpy(); sig = signals.to_numpy()
    cash = initial; units = 0.0; entry = 0.0; side = 0; n = 0
    eq_t = []; eq_v = []
    for i in range(len(df)):
        price = close[i]
        if not np.isfinite(price) or price <= 0:
            continue
        s = int(sig[i]) if i > 0 else 0
        if side != 0 and entry > 0:
            r = (price - entry) / entry if side > 0 else (entry - price) / entry
            if (tp and r >= tp) or (sl and r <= -sl):
                cash = _close(cash, units, entry, price, side); side = 0; units = 0; n += 1
                eq_t.append(idx[i]); eq_v.append(cash); continue
        eq = cash + (units * price if side > 0 else units * entry + (entry - price) * units if side < 0 else 0)
        if eq <= 0:
            return None
        eq_t.append(idx[i]); eq_v.append(eq)
        if s == 1 and side <= 0:
            if side < 0:
                cash = _close(cash, units, entry, price, side); n += 1; side = 0; units = 0
            u = (cash * 0.95) / (price * (1 + comm))
            if u > 0:
                cash -= u * price * (1 + comm); units = u; entry = price; side = 1
        elif s == -1 and side >= 0:
            if side > 0:
                cash = _close(cash, units, entry, price, side); n += 1; side = 0; units = 0
            if mode == 'long_short':
                u = (cash * 0.95) / price
                if u > 0:
                    cash -= u * price * (1 + comm); units = u; entry = price; side = -1
    if side != 0 and len(df) > 0:
        cash = _close(cash, units, entry, close[-1], side)
        n += 1
        eq_t.append(idx[-1]); eq_v.append(cash)
    if n == 0:
        return None
    eq = pd.Series(eq_v, index=pd.DatetimeIndex(eq_t)).sort_index()
    peak = eq.cummax(); mdd = ((eq - peak) / peak * 100).min()
    rr = eq.pct_change().dropna()
    sh = float(rr.mean() / rr.std() * np.sqrt(PERIODS_PER_YEAR)) if len(rr) >= 10 and rr.std() > 0 else None
    return {'ret': (eq.iloc[-1] / initial - 1) * 100, 'mdd': mdd, 'n': n, 'sh': sh}

# ---------------- 全量回测 ----------------
def run_all():
    # detail 结构: {ticker: {year: {mode|'bh': {...}}}}
    detail = {}
    failed = []
    for k, T in enumerate(TICKERS, 1):
        yh = YH_ALIAS.get(T, T)
        df = fetch_1h(yh)
        if df is None:
            print(f"[{k}/{len(TICKERS)}] {T:8} 无数据"); failed.append(T); continue
        # 信号在整段历史连续计算（含预热），再分年切片模拟
        sigs = {}
        for v, cfg in VARIANTS.items():
            try:
                sigs[v] = build_variant_signals(df, cfg['use_div'], cfg['use_ma'], cfg['use_vol'])
            except Exception as e:
                print(f"  {T} {v} 信号出错: {e}")
        row_t = {}
        for yr, (a, b) in SEGMENTS.items():
            d = df[df.index >= a]
            if b: d = d[d.index < b]
            if len(d) < YEAR_BARS_MIN:
                continue
            row = {}
            for mname, mcode in MODES.items():
                for tname, (tp, sl) in TPSL_CFG.items():
                    cell = {}
                    for v in VLIST:
                        if v not in sigs:
                            cell[v] = None; continue
                        try:
                            cell[v] = simulate(d, sigs[v].loc[d.index], tp=tp, sl=sl, mode=mcode)
                        except Exception:
                            cell[v] = None
                    row[f'{mname}|{tname}'] = cell
            row['_bh'] = (d['close'].iloc[-1] / d['close'].iloc[0] - 1) * 100
            row['_nbar'] = len(d)
            row_t[yr] = row
        detail[T] = row_t
        r0 = row_t.get('2024', {}).get('仅做多|不设', {}).get(VLIST[0])
        print(f"[{k}/{len(TICKERS)}] {T:8} bar{len(df)}  2024仅做多不设·背离: "
              + (f"{r0['ret']:+.1f}%" if r0 else '无'), flush=True)

    os.makedirs('scripts/results', exist_ok=True)
    with open(OUT_DETAIL, 'w', encoding='utf-8') as f:
        json.dump({'segments': {k: list(v) for k, v in SEGMENTS.items()}, 'variants': VLIST,
                   'results': detail, 'failed': failed}, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n明细已保存 → {OUT_DETAIL}")
    return detail, failed

# ---------------- 聚合 summary（与虚拟币 summary 同构） ----------------
def aggregate(detail=None, failed=None):
    if detail is None:
        d = json.load(open(OUT_DETAIL, encoding='utf-8'))
        detail, failed = d['results'], d.get('failed', [])

    strategies = []
    for v in VLIST:
        configs = []
        for mname in MODES:
            for tname in TPSL_CFG:
                key = f'{mname}|{tname}'
                year_runs = []
                for yr in SEGMENTS:
                    cells = [detail[t][yr][key][v] for t in detail
                             if yr in detail.get(t, {}) and key in detail[t][yr]
                             and detail[t][yr][key].get(v)]
                    bh = [detail[t][yr]['_bh'] for t in detail
                          if yr in detail.get(t, {}) and '_bh' in detail[t][yr]
                          and key in detail[t][yr]]
                    tickers = [t for t in detail if yr in detail.get(t, {}) and key in detail[t][yr]]
                    if not cells:
                        year_runs.append({'yr': yr, 'd': None}); continue
                    rets = [c['ret'] for c in cells]
                    mdds = [c['mdd'] for c in cells if c.get('mdd') is not None]
                    ns = [c['n'] for c in cells if c.get('n') is not None]
                    shs = [c['sh'] for c in cells if c.get('sh') is not None]
                    beat = sum(1 for t in tickers if detail[t][yr][key].get(v)
                               and detail[t][yr][key][v]['ret'] > detail[t][yr]['_bh'])
                    year_runs.append({'yr': yr, 'd': {
                        'ret': round(float(np.mean(rets)), 1),
                        'med': round(float(np.median(rets)), 1),
                        'pos': int(sum(x > 0 for x in rets)),
                        'cnt': len(rets),
                        'mdd': round(float(np.mean(mdds)), 1) if mdds else None,
                        'mdd_max': round(float(min(mdds)), 1) if mdds else None,
                        'sh': round(float(np.mean(shs)), 2) if shs else None,
                        'beat': int(beat),
                        'n': int(sum(ns)),
                        'n_avg': round(float(np.mean(ns)), 1),
                    }})
                # 三年合计
                tot = 1.0
                all_n, all_mdd_avg, all_mdd_max, shs_all = 0, [], [], []
                for entry in year_runs:
                    d_ = entry['d']
                    if not d_:
                        continue
                    tot *= (1 + d_['ret'] / 100)
                    all_n += d_['n']
                    if d_.get('mdd') is not None:
                        all_mdd_avg.append(d_['mdd'])
                    if d_.get('mdd_max') is not None:
                        all_mdd_max.append(d_['mdd_max'])
                    if d_.get('sh') is not None:
                        shs_all.append(d_['sh'])
                cfg = {'tf': '1h', 'mode': mname, 'tpsl': tname, 'wiped': 0}
                for entry in year_runs:
                    cfg[entry['yr']] = entry['d']
                cfg['tot'] = round((tot - 1) * 100, 1)
                cfg['n_all'] = int(all_n)
                cfg['mdd_all_avg'] = round(float(np.mean(all_mdd_avg)), 1) if all_mdd_avg else None
                cfg['mdd_all_max'] = round(float(min(all_mdd_max)), 1) if all_mdd_max else None
                cfg['sh_all'] = round(float(np.mean(shs_all)), 2) if shs_all else None
                configs.append(cfg)
        strategies.append({'name': v, 'configs': configs})

    # BH 基准（分年均值）
    bh = {}
    for yr in SEGMENTS:
        vals = [detail[t][yr]['_bh'] for t in detail if yr in detail.get(t, {}) and '_bh' in detail[t][yr]]
        bh[yr] = round(float(np.mean(vals)), 1) if vals else None

    summary = {
        'meta': {
            'updated': datetime.now().strftime('%Y-%m-%d'),
            'title': '美股代币 × 三种MACD组合策略 · 全配置回测矩阵',
            'desc': (f"标的=综合量化美股清单 {len(detail)} 只（Yahoo Finance 1h，FLNC/HD 已移出实盘清单但保留回测）。"
                     "1h × 仅做多/双向多空 × 不设/只止损5%/止盈止损各5% 共6组配置，"
                     "分2024(自9-06上线起)/2025/2026YTD回测；初始1万USDT、95%仓位、单边手续费0.1%、年末强平，"
                     "止损按收盘价判定，双向做空为现金背书模拟。单元格内小字为正收益标的数/有效标的数。"),
            'bh': {'1h': bh},
            'conclusion': ('美股单边牛市下三种组合均大幅跑输买入持有BH；止盈止损各5%严重牺牲趋势收益（收益缩水60-80%）'
                           '仅换来回撤小幅变浅；双向多空因牛市空单拖累全面劣于仅做多；'
                           '量能/均线过滤把下单数从年均近百笔砍到十几笔、收益基本持平但更挑标的（>40%收益只数反而更多）。'
                           '三段检验已剔除持续亏损标的 FLNC/HD。'),
        },
        'strategies': strategies,
    }
    with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=float)
    print(f"summary已保存 → {OUT_SUMMARY}")

    # 控制台摘要
    print("\n## 分年·跨标的均值（仅做多|不设 为例）")
    for s in strategies:
        c0 = s['configs'][0]
        line = f"{s['name']:<20}"
        for yr in SEGMENTS:
            d_ = c0.get(yr)
            line += f"  {yr}:{('%+.1f%%' % d_['ret']) if d_ else '—'}"
        line += f"  三年合计{c0.get('tot', 0):+.1f}%"
        print(line)
    return summary


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'run'
    if mode == 'agg':
        aggregate()
    else:
        det, fail = run_all()
        aggregate(det, fail)
