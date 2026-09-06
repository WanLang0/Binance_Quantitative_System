# -*- coding: utf-8 -*-
"""市值前40合约币（全部40只）× macd+背离+量能 × 三方案 × 2024/2025/2026YTD

方案一 15m 双向多空 止盈止损各5% (tpsl5)
方案二 15m 仅做多   不设止盈止损
方案三 1h  双向多空 只止损5% (sl5)

口径与 tmp_top40next_macd_vol.py 完全一致：
币安USDT永续、2023-10起拉数(预热)、分年度、初始1万、95%仓位、
单边手续费0.1%、年末强平、止盈止损按收盘价、做空现金背书。
输出：收益率 / 最大回撤 / 交易次数。
K线缓存采用增量更新（只拉缓存末根之后的新数据）。
"""
import os, sys, io, time, warnings, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

PROXY = "http://127.0.0.1:7892"
os.environ.setdefault("HTTP_PROXY", PROXY); os.environ.setdefault("HTTPS_PROXY", PROXY)

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone

from divergence_signals import DIVERGENCE_VARIANTS, build_variant_signals

COMM = 0.001
INITIAL = 10000.0
WARMUP_START = '2023-10-01'
YEAR_BARS_MIN = 200
WINDOWS = [
    ('2024', '2024-01-01', '2025-01-01'),
    ('2025', '2025-01-01', '2026-01-01'),
    ('2026YTD', '2026-01-01', datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')),
]
VARIANT = 'macd+背离+量能'
UD, UM, UV = DIVERGENCE_VARIANTS[VARIANT]

# 三方案（按用户指定顺序）
CONFIGS = [
    ('15m', 'long_short', 'tpsl5', '方案一 15m双向 tpsl5'),
    ('15m', 'long_only', 'none', '方案二 15m仅多 不设'),
    ('1h', 'long_short', 'sl5', '方案三 1h双向 sl5'),
]
TPSL_CFG = {'none': (None, None), 'sl5': (None, 0.05), 'tpsl5': (0.05, 0.05)}
PERIODS = {'1h': 8766, '15m': 35040}

STABLES = {'USDT','USDC','USDS','DAI','USDE','USD1','FDUSD','TUSD','USDP','PYUSD','USDD',
           'BUSD','USDF','USDG','BUIDL','USDX','USDSOL'}
WRAPPED = {'WBTC','WETH','WBETH','WSTETH','STETH','RETH','CBETH','CBBTC','BTCB','WEETH',
           'WBNB','WPOL','MATIC','EZETH','RSETH','LSETH'}

OUT_DETAIL = 'scripts/results/top40all_macd_vol_detail_2024_2026.json'
OUT_SUMMARY = 'scripts/results/top40all_macd_vol_summary_2024_2026.json'

_s = requests.Session()
_s.proxies = {'http': PROXY, 'https': PROXY}
_s.headers['User-Agent'] = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                            'Chrome/120.0 Safari/537.36')


def _get_json(url, params=None, tries=4):
    """带重试的 GET（CoinGecko/代理偶发 SSL 断连）"""
    for i in range(tries):
        try:
            r = _s.get(url, params=params, timeout=30)
            return r.json()
        except Exception as e:
            print(f'  GET {url.split("/")[3]} retry{i + 1}: {repr(e)[:60]}', file=sys.stderr)
            time.sleep(3)
    raise RuntimeError(f'请求失败: {url}')


def get_top40_all_bases():
    """市值前40 ∩ 币安USDT永续，剔除稳定币/封装，返回全部40只名单"""
    perps = set()
    info = _get_json('https://fapi.binance.com/fapi/v1/exchangeInfo')
    for x in info['symbols']:
        if (x.get('quoteAsset') == 'USDT' and x.get('contractType') == 'PERPETUAL'
                and x.get('status') == 'TRADING'):
            perps.add(x['baseAsset'])
    top40, seen = [], set()
    for page in (1, 2):
        js = _get_json('https://api.coingecko.com/api/v3/coins/markets',
                       params={'vs_currency': 'usd', 'order': 'market_cap_desc',
                               'per_page': 100, 'page': page})
        if not isinstance(js, list):
            break
        for c in js:
            sym = str(c.get('symbol', '')).upper()
            if sym in seen or sym in STABLES or sym in WRAPPED or sym not in perps:
                continue
            seen.add(sym)
            top40.append((sym, c.get('name', sym), round(c.get('market_cap') or 0)))
            if len(top40) >= 40:
                break
        if len(top40) >= 40:
            break
        time.sleep(1.2)
    return top40


def fetch_klines(base, tf, start=WARMUP_START, tries=4):
    """拉K线并做增量缓存：缓存存在则只拉末根之后的新数据"""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f'{base}_{tf}.pkl')
    old = None
    if os.path.exists(cache):
        try:
            cand = pd.read_pickle(cache)
            if cand is not None and len(cand) >= YEAR_BARS_MIN:
                old = cand
        except Exception:
            old = None
    sym = f"{base}USDT"
    url = 'https://fapi.binance.com/fapi/v1/klines'
    if old is not None and len(old) > 0:
        since = int(old.index[-1].timestamp() * 1000) + 1
    else:
        since = int(pd.Timestamp(start, tz='UTC').timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows = []
    for _ in range(tries):
        try:
            while since < end_ms:
                kl = _s.get(url, params={'symbol': sym, 'interval': tf,
                                         'startTime': since, 'limit': 1500}, timeout=30).json()
                if not isinstance(kl, list) or not kl:
                    break
                rows.extend(kl)
                since = kl[-1][6] + 1
                if len(kl) < 1500:
                    break
                time.sleep(0.25)
            if rows or old is not None:
                break
        except Exception as e:
            print(f"  fetch {base} {tf} err: {repr(e)[:60]}", file=sys.stderr)
        time.sleep(3)
    if not rows:
        return old
    df_new = pd.DataFrame([r[:6] for r in rows],
                          columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df_new['timestamp'] = pd.to_datetime(df_new['ts'], unit='ms', utc=True)
    df_new = df_new.set_index('timestamp')[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df = pd.concat([old, df_new]) if old is not None else df_new
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df = df.dropna(subset=['close'])
    df = df[df['close'] > 0]
    try:
        df.to_pickle(cache)
    except Exception:
        pass
    return df


def _close(cash, units, entry, price, side, comm=COMM):
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm


def simulate(df, signals, tp=None, sl=None, mode='long_only', ppy=8766,
             initial=INITIAL, comm=COMM):
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
    sh = float(rr.mean() / rr.std() * np.sqrt(ppy)) if len(rr) >= 10 and rr.std() > 0 else None
    return {'ret': (eq.iloc[-1] / initial - 1) * 100, 'mdd': mdd, 'n': n, 'sh': sh}


def main():
    print('== 选币：市值前40（全部）∩ 币安USDT永续，剔除稳定币/封装 ==')
    top40 = get_top40_all_bases()
    print(f'前40名单({len(top40)}只): {[s for s, _, _ in top40]}')
    for i, (s, n, m) in enumerate(top40, 1):
        print(f'  {i:2}. {s:8} {n}  市值≈${m/1e9:.1f}B')

    # detail: {base: {tf: {year: {cfgname: {ret,mdd,n,sh}|None}, '_bh','_nbar'}}}
    detail = {b: {} for b, _, _ in top40}
    for k, (base, name, mc) in enumerate(top40, 1):
        for tf in sorted({c[0] for c in CONFIGS}):
            df = fetch_klines(base, tf)
            if df is None or len(df) < YEAR_BARS_MIN:
                print(f'[{k}/{len(top40)}] {base:8} {tf} 数据不足'); continue
            try:
                _, sig = build_variant_signals(df, UD, UM, UV)
            except Exception as e:
                print(f'[{k}/{len(top40)}] {base:8} {tf} 信号出错: {e}'); continue
            tfd = detail[base].setdefault(tf, {})
            for wname, w0, w1 in WINDOWS:
                m = (df.index >= pd.Timestamp(w0, tz='UTC')) & (df.index < pd.Timestamp(w1, tz='UTC'))
                dw = df[m]
                if len(dw) < YEAR_BARS_MIN:
                    continue
                row = {}
                for ctf, cmode, ctpsl, cname in CONFIGS:
                    if ctf != tf:
                        continue
                    tp, sl = TPSL_CFG[ctpsl]
                    try:
                        row[cname] = simulate(dw, sig.loc[dw.index], tp=tp, sl=sl,
                                              mode=cmode, ppy=PERIODS[tf])
                    except Exception:
                        row[cname] = None
                row['_bh'] = (dw['close'].iloc[-1] / dw['close'].iloc[0] - 1) * 100
                row['_nbar'] = len(dw)
                tfd[wname] = row
            r0 = tfd.get('2024', {}).get(CONFIGS[0][3] if tf == '15m' else CONFIGS[2][3])
            print(f'[{k}/{len(top40)}] {base:8} {tf} bar{len(df)} '
                  + (f"2024主配置:{r0['ret']:+.1f}%" if r0 else '2024:无'), flush=True)

    os.makedirs('scripts/results', exist_ok=True)
    with open(OUT_DETAIL, 'w', encoding='utf-8') as f:
        json.dump({'variant': VARIANT, 'configs': [c[3] for c in CONFIGS],
                   'results': detail}, f, ensure_ascii=False, indent=1, default=float)
    print(f'\n明细 → {OUT_DETAIL}')

    # ---- summary ----
    strategies = []
    cfgs = []
    for ctf, cmode, ctpsl, cname in CONFIGS:
        year_runs = []
        for yr, _, _ in WINDOWS:
            cells = [detail[b][ctf][yr][cname] for b in detail
                     if ctf in detail.get(b, {}) and yr in detail[b][ctf]
                     and detail[b][ctf][yr].get(cname)]
            tickers = [b for b in detail if ctf in detail.get(b, {}) and yr in detail[b][ctf]
                       and detail[b][ctf][yr].get(cname)]
            if not cells:
                year_runs.append({'yr': yr, 'd': None}); continue
            rets = [c['ret'] for c in cells]
            mdds = [c['mdd'] for c in cells if c.get('mdd') is not None]
            ns = [c['n'] for c in cells if c.get('n') is not None]
            shs = [c['sh'] for c in cells if c.get('sh') is not None]
            beat = sum(1 for b in tickers if detail[b][ctf][yr][cname]['ret'] > detail[b][ctf][yr]['_bh'])
            year_runs.append({'yr': yr, 'd': {
                'ret': round(float(np.mean(rets)), 1), 'med': round(float(np.median(rets)), 1),
                'pos': int(sum(x > 0 for x in rets)), 'cnt': len(rets),
                'mdd': round(float(np.mean(mdds)), 1) if mdds else None,
                'mdd_max': round(float(min(mdds)), 1) if mdds else None,
                'sh': round(float(np.mean(shs)), 2) if shs else None,
                'beat': int(beat), 'n': int(sum(ns)), 'n_avg': round(float(np.mean(ns)), 1),
            }})
        tot = 1.0; all_n = 0; mdd_a, mdd_m, shs_all = [], [], []
        for e in year_runs:
            if not e['d']:
                continue
            tot *= (1 + e['d']['ret'] / 100)
            all_n += e['d']['n']
            if e['d'].get('mdd') is not None: mdd_a.append(e['d']['mdd'])
            if e['d'].get('mdd_max') is not None: mdd_m.append(e['d']['mdd_max'])
            if e['d'].get('sh') is not None: shs_all.append(e['d']['sh'])
        cfg = {'tf': ctf, 'mode': '双向多空' if cmode == 'long_short' else '仅做多',
               'tpsl': {'none': '不设', 'sl5': '只止损5%', 'tpsl5': '止盈止损各5%'}[ctpsl],
               'name': cname, 'wiped': 0}
        for e in year_runs:
            cfg[e['yr']] = e['d']
        cfg['tot'] = round((tot - 1) * 100, 1)
        cfg['n_all'] = int(all_n)
        cfg['mdd_all_avg'] = round(float(np.mean(mdd_a)), 1) if mdd_a else None
        cfg['mdd_all_max'] = round(float(min(mdd_m)), 1) if mdd_m else None
        cfg['sh_all'] = round(float(np.mean(shs_all)), 2) if shs_all else None
        cfgs.append(cfg)
    strategies.append({'name': VARIANT, 'configs': cfgs})

    bh = {}
    for tf in ('1h', '15m'):
        per_yr = {}
        for yr, _, _ in WINDOWS:
            vals = [detail[b][tf][yr]['_bh'] for b in detail
                    if tf in detail.get(b, {}) and yr in detail[b][tf]]
            per_yr[yr] = round(float(np.mean(vals)), 1) if vals else None
        bh[tf] = per_yr

    summary = {
        'meta': {
            'updated': datetime.now().strftime('%Y-%m-%d'),
            'title': '市值前40（全部40只）× macd+背离+量能 × 三方案',
            'desc': (f"标的=市值前40(CoinGecko,剔除稳定币/封装)∩币安USDT永续 全部{len(top40)}只。"
                     "三方案：方案一15m双向止盈止损5%；方案二15m仅做多不设；方案三1h双向只止损5%。"
                     "口径与前次回测完全一致（2023-10起预热、分2024/2025/2026YTD、"
                     "初始1万、95%仓位、单边手续费0.1%、年末强平、止损按收盘价、做空现金背书）。"),
            'bh': bh,
            'tickers': [s for s, _, _ in top40],
        },
        'strategies': strategies,
    }
    with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=float)
    print(f'summary → {OUT_SUMMARY}')

    print('\n== 汇总（市值前40 × macd+背离+量能） ==')
    for c in cfgs:
        print(f"\n【{c['name']}（{c['tf']} {c['mode']} {c['tpsl']}）】")
        for yr in ('2024', '2025', '2026YTD'):
            d = c.get(yr)
            if not d:
                print(f'  {yr}: 无'); continue
            print(f"  {yr:8} 均收益{d['ret']:+7.1f}%  中位{d['med']:+7.1f}%  正收益{d['pos']:2}/{d['cnt']:2}"
                  f"  | 平均MDD {d['mdd']:6.1f}%  最差MDD {d['mdd_max']:6.1f}%"
                  f"  | 交易次数 总{d['n']:6d} 均值{d['n_avg']:6.1f}  跑赢BH {d['beat']}/{d['cnt']}")
        print(f"  三年合计 {c['tot']:+.1f}%  平均MDD {c['mdd_all_avg']}%  最差MDD {c['mdd_all_max']}%"
              f"  总交易 {c['n_all']}  夏普 {c['sh_all']}")


if __name__ == '__main__':
    main()
