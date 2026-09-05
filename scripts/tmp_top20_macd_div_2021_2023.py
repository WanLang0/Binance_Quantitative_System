# -*- coding: utf-8 -*-
"""年初市值前20合约币 × macd+背离+量能 年度回测（币安USDT永续，2021/2022/2023）

用法：python tmp_top20_macd_div_2021_2023.py 2021|2022|2023
一次运行该年 3 组配置：
  A) 15m 双向多空 止盈止损各5%（tpsl5）
  B) 15m 仅做多 不设止盈止损（none）
  C) 1h  双向多空 只止损5%（sl5）

币种名单：CoinMarketCap 当年年初快照（2021-01-03 / 2022-01-02 / 2023-01-01）
按市值降序，剔除稳定币/封装资产，剔除当年初尚无币安永续或历史K线已不可得的币，
不足20个按市值排名顺延补足（与实盘"市值前20∩币安在市永续"逻辑一致）。
顺延与剔除明细：
  2021: 剔 BSV/XEM(合约2021-03才上线)/CEL/CRO(无合约或历史不可得)、LEO/USDC/USDT/WBTC/Dai/BUSD → 补 DOGE/ATOM/NEO
  2022: 剔 CRO/LEO/USDT/USDC/UST/Dai/BUSD/WBTC → 补 LINK/NEAR/BCH/ATOM/TRX；LUNA 数据到2022-05-13崩盘下架自然截断
  2023: 剔 TON(2024才有合约)/LEO/USDT/USDC/BUSD/Dai/WBTC → 补 LINK/XMR/ATOM/ETC/XLM/BCH
符号映射：SHIB→1000SHIBUSDT（币安合约单位）；MATIC→MATICUSDT（含改名前完整历史）。

口径与 scripts/tmp_top20_macd_div_1h_2024_2026.py 完全一致：
- 信号 divergence_signals.build_variant_signals(use_div=True, use_ma=False, use_vol=True)（macd+背离+量能）
- 数据 币安fapi K线，年初前2个月仅作指标预热，回测窗口=整年
- 模拟 95%资金入场、单边手续费0.1%、止盈止损按收盘价判定、窗口末强制平仓；
       long_short 卖出信号空仓时开空（现金背书、无杠杆、无资金费率）
- 夏普年化 √年周期数（1h→8766，15m→35040）
"""
import os, sys, io, time, warnings, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

YEAR = sys.argv[1] if len(sys.argv) > 1 else '2021'
assert YEAR in ('2021', '2022', '2023'), "参数须为 2021|2022|2023"

PROXY = "http://127.0.0.1:7892"
os.environ.setdefault("HTTP_PROXY", PROXY); os.environ.setdefault("HTTPS_PROXY", PROXY)

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone

from divergence_signals import build_variant_signals

COMM = 0.001
INITIAL = 10000.0
YEAR_BARS_MIN = 200
PPY = {'1h': 8766, '15m': 35040}

# (符号映射前基准币, 当年CMC市值排名), 名单见模块 docstring
TOP20 = {
    '2021': [('BTC', 1), ('ETH', 2), ('LTC', 4), ('XRP', 5), ('DOT', 6), ('BCH', 7),
             ('ADA', 8), ('BNB', 9), ('LINK', 10), ('XLM', 14), ('EOS', 15), ('XMR', 16),
             ('THETA', 17), ('TRX', 18), ('VET', 20), ('XTZ', 21), ('UNI', 23),
             ('DOGE', 26), ('ATOM', 27), ('NEO', 30)],
    '2022': [('BTC', 1), ('ETH', 2), ('BNB', 3), ('SOL', 5), ('ADA', 6), ('XRP', 8),
             ('LUNA', 9), ('DOT', 10), ('AVAX', 11), ('DOGE', 12), ('SHIB', 13),
             ('MATIC', 14), ('UNI', 18), ('ALGO', 19), ('LTC', 20), ('LINK', 21),
             ('NEAR', 24), ('BCH', 25), ('ATOM', 26), ('TRX', 27)],
    '2023': [('BTC', 1), ('ETH', 2), ('BNB', 5), ('XRP', 6), ('DOGE', 8), ('ADA', 9),
             ('MATIC', 10), ('LTC', 12), ('TRX', 13), ('DOT', 14), ('SHIB', 15),
             ('UNI', 16), ('SOL', 17), ('AVAX', 18), ('LINK', 21), ('XMR', 23),
             ('ATOM', 24), ('ETC', 25), ('XLM', 26), ('BCH', 27)],
}
SYM_MAP = {'SHIB': '1000SHIB'}          # 币安合约符号前缀
WARMUP = {'2021': '2020-10-01', '2022': '2021-10-01', '2023': '2022-10-01'}
W0 = f'{YEAR}-01-01'
W1 = f'{int(YEAR) + 1}-01-01'
CFG = {'A_tpsl5_15m': ('15m', 'long_short', 0.05, 0.05),
       'B_none_15m':  ('15m', 'long_only', None, None),
       'C_sl5_1h':    ('1h', 'long_short', None, 0.05)}

_s = requests.Session()
_s.proxies = {'http': PROXY, 'https': PROXY}
_s.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36'


def fapi_symbol(base):
    return f"{SYM_MAP.get(base, base)}USDT"


def fetch_klines_fapi(base, tf, start, end, tries=4):
    """分页拉取币安USDT永续K线（fapi 限1500/次）；历史数据固定，存在即复用本地缓存"""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f'{fapi_symbol(base)}_{tf}_{YEAR}.pkl')
    if os.path.exists(cache):
        try:
            df_c = pd.read_pickle(cache)
            if df_c is not None and len(df_c) >= YEAR_BARS_MIN:
                return df_c
        except Exception:
            pass
    sym = fapi_symbol(base)
    since = int(pd.Timestamp(start, tz='UTC').timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz='UTC').timestamp() * 1000)
    rows = []
    for _ in range(tries):
        try:
            while since < end_ms:
                kl = _s.get('https://fapi.binance.com/fapi/v1/klines',
                            params={'symbol': sym, 'interval': tf,
                                    'startTime': since, 'limit': 1500}, timeout=30).json()
                if not isinstance(kl, list) or not kl:
                    break
                rows.extend(kl)
                since = kl[-1][6] + 1
                if len(kl) < 1500:
                    break
                time.sleep(0.3)
            if rows:
                break
        except Exception as e:
            print(f"  fetch {base} err: {repr(e)[:60]}", file=sys.stderr)
        time.sleep(3)
    if not rows:
        return None
    df = pd.DataFrame([r[:6] for r in rows],
                      columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df = df.set_index('timestamp')[['open', 'high', 'low', 'close', 'volume']].astype(float)
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


def simulate(df, signals, ppy, tp=None, sl=None, mode='long_only', initial=INITIAL, comm=COMM):
    """与 tmp_top20_macd_div_1h_2024_2026.simulate 逐行同口径"""
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
    top20 = TOP20[YEAR]
    print(f"== {YEAR}年初市值前20合约币（CMC快照，剔除稳定币/封装/无合约，顺延补足） ==")
    print("  " + " ".join(f"{b}(#{r})" for b, r in top20))

    # 数据按 TF 拉一遍，两组15m配置共用
    data = {}
    for tf in ('15m', '1h'):
        data[tf] = {}
        print(f"\n== 拉取 {tf} K线（预热起点 {WARMUP[YEAR]}，窗口 {W0}~{W1}） ==")
        for i, (base, rank) in enumerate(top20, 1):
            df = fetch_klines_fapi(base, tf, WARMUP[YEAR], W1)
            if df is None or len(df) < YEAR_BARS_MIN:
                print(f"  [{i}/20] {base}: 数据不足，跳过"); continue
            data[tf][base] = df
            print(f"  [{i}/20] {base:8} {df.index[0]:%Y-%m-%d}~{df.index[-1]:%Y-%m-%d} bar{len(df)}", flush=True)

    # ---- 每组配置回测 ----
    out = {'year': YEAR, 'top20': top20, 'windows': (W0, W1), 'comm': COMM,
           'initial': INITIAL, 'configs': {}}
    for cname, (tf, mode, tp, sl) in CFG.items():
        ppy = PPY[tf]
        R = {}
        for base, rank in top20:
            df = data[tf].get(base)
            if df is None:
                continue
            m = (df.index >= pd.Timestamp(W0, tz='UTC')) & (df.index < pd.Timestamp(W1, tz='UTC'))
            dw = df[m]
            if len(dw) < YEAR_BARS_MIN:
                continue
            try:
                _, sig = build_variant_signals(df, use_div=True, use_ma=False, use_vol=True)
            except Exception as e:
                print(f"  {base} 信号出错: {e}"); continue
            R[base] = {'sim': simulate(dw, sig.loc[dw.index], ppy, tp=tp, sl=sl, mode=mode),
                       'bh': (dw['close'].iloc[-1] / dw['close'].iloc[0] - 1) * 100,
                       'nbar': int(len(dw))}
        out['configs'][cname] = R
        bh = [v['bh'] for v in R.values()]
        rs = [v['sim'] for v in R.values() if v['sim']]
        rets = [x['ret'] for x in rs]; mdds = [x['mdd'] for x in rs]; ns = [x['n'] for x in rs]
        shs = [x['sh'] for x in rs if x['sh'] is not None]
        beat = sum(1 for b, v in R.items() if v['sim'] and v['sim']['ret'] > v['bh'])
        tfd, md, td = {'A_tpsl5_15m': ('15m', '双向多空', '止盈止损各5%'),
                       'B_none_15m': ('15m', '仅做多', '不设止盈止损'),
                       'C_sl5_1h': ('1h', '双向多空', '只止损5%')}[cname]
        print(f"\n##【{YEAR} · {cname}】 {tfd} · {md} · {td}  标的数={len(R)}")
        print(f"   买入持有BH   平均 {np.mean(bh):+8.1f}%  正收益 {sum(x>0 for x in bh)}/{len(bh)}")
        if rs:
            print(f"   macd+背离+量能 平均 {np.mean(rets):+8.1f}%  中位 {np.median(rets):+8.1f}%  "
                  f"胜率 {sum(x>0 for x in rets)}/{len(rets)}  回撤均值 {np.mean(mdds):6.1f}%  "
                  f"回撤最大 {min(mdds):6.1f}%  笔数均值 {np.mean(ns):5.1f}  "
                  f"夏普均值 {np.mean(shs):+.2f}  跑赢BH {beat}/{len(rs)}")
        print("   每币明细（收益%/回撤%/笔数）：")
        for b in sorted(R, key=lambda x: -(R[x]['sim']['ret'] if R[x]['sim'] else -1e9)):
            v = R[b]
            s = v['sim']
            tag = '' if v['nbar'] >= PPY[tfd] * 0.9 else f"（数据至{data[tf][b][data[tf][b].index < pd.Timestamp(W1, tz='UTC')].index[-1]:%m-%d}截断）"
            print(f"     {b:8} BH{v['bh']:+9.1f}%  " +
                  (f"策略 {s['ret']:+8.1f}%/mdd{s['mdd']:.0f}%/n{s['n']:3}" if s else "策略 无交易") + tag)

    os.makedirs('scripts/results', exist_ok=True)
    fn = f'scripts/results/top20_macddivvol_{YEAR}.json'
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f"\nsaved → {fn}")


if __name__ == '__main__':
    main()
