# -*- coding: utf-8 -*-
"""市值前20合约币 × 三种MACD组合策略 回测（币安USDT永续，2024/2025/2026YTD，不设止盈止损）

用法：python tmp_top20_macd_div_1h_2024_2026.py [1h|15m] [long_only|long_short] [none|sl5|tpsl5]
      默认 1h long_only none；sl5=只止损5%；tpsl5=止盈止损各5%（按收盘价判定，与美股脚本口径一致）
      K线数据缓存于 scripts/cache/，同日重复运行不重复拉取

口径（与美股综合量化 tmp_us_macd_div_2024_2025.py 对齐）：
- 标的：CoinGecko 市值排名（剔除稳定币/封装资产）∩ 币安USDT永续，取前20
- 策略：三种MACD固定组合（divergence_signals.DIVERGENCE_VARIANTS）
    1) macd+背离            2) macd+背离+量能          3) macd+背离+均线+量能
- 数据：币安 fapi K线，2023-10-01 起拉取（前2个月仅作指标预热），分年回测
- 模拟：95%资金入场、单边手续费0.1%、不设止盈止损、窗口末强制平仓；
        long_short 模式：卖出信号空仓时模拟开空（现金背书、无杠杆、无资金费率）
- 夏普年化 √年周期数（1h→8766，15m→35040）
"""
import os, sys, io, time, warnings, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

TF = sys.argv[1] if len(sys.argv) > 1 else '1h'
MODE = sys.argv[2] if len(sys.argv) > 2 else 'long_only'
if MODE not in ('long_only', 'long_short'):
    MODE = 'long_only'
MODE_SUFFIX = 'longshort' if MODE == 'long_short' else 'longonly'
# 止盈止损场景：none=不设；sl5=只止损5%；tpsl5=止盈止损各5%
TPSL_ARG = sys.argv[3] if len(sys.argv) > 3 else 'none'
TPSL_CFG = {'none': (None, None), 'sl5': (None, 0.05), 'tpsl5': (0.05, 0.05)}
TP, SL = TPSL_CFG.get(TPSL_ARG, (None, None))
PERIODS_PER_YEAR = {'1h': 8766, '15m': 35040}.get(TF, 8766)

PROXY = "http://127.0.0.1:7892"
os.environ.setdefault("HTTP_PROXY", PROXY); os.environ.setdefault("HTTPS_PROXY", PROXY)

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone

from divergence_signals import DIVERGENCE_VARIANTS, build_variant_signals

COMM = 0.001
INITIAL = 10000.0
WARMUP_START = '2023-10-01'          # 预热起点（不计入回测）
YEAR_BARS_MIN = 200                  # 年度窗口最少bar数
# 分年窗口（2026年到当前时刻）
WINDOWS = [
    ('2024', '2024-01-01', '2025-01-01'),
    ('2025', '2025-01-01', '2026-01-01'),
    ('2026YTD', '2026-01-01', datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')),
]
VLIST = list(DIVERGENCE_VARIANTS.keys())   # macd+背离 / macd+背离+量能 / macd+背离+均线+量能

STABLES = {'USDT','USDC','USDS','DAI','USDE','USD1','FDUSD','TUSD','USDP','PYUSD','USDD',
           'BUSD','USDF','USDG','BUIDL','USDX','USDSOL'}   # 稳定币/收益凭证
WRAPPED = {'WBTC','WETH','WBETH','WSTETH','STETH','RETH','CBETH','CBBTC','BTCB','WEETH',
           'WBNB','WPOL','MATIC','EZETH','RSETH','LSETH'}  # 封装/质押衍生

_s = requests.Session()
_s.proxies = {'http': PROXY, 'https': PROXY}
_s.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36'


def get_top20_perp_bases():
    """CoinGecko市值排名 ∩ 币安USDT永续(在市PERPETUAL)，剔除稳定币/封装资产，取前20"""
    perps = set()
    info = _s.get('https://fapi.binance.com/fapi/v1/exchangeInfo', timeout=30).json()
    for x in info['symbols']:
        if (x.get('quoteAsset') == 'USDT' and x.get('contractType') == 'PERPETUAL'
                and x.get('status') == 'TRADING'):
            perps.add(x['baseAsset'])
    picked, seen = [], set()
    for page in (1, 2):
        js = _s.get('https://api.coingecko.com/api/v3/coins/markets',
                    params={'vs_currency': 'usd', 'order': 'market_cap_desc',
                            'per_page': 100, 'page': page}, timeout=30).json()
        if not isinstance(js, list):
            break
        for c in js:
            sym = str(c.get('symbol', '')).upper()
            if sym in seen or sym in STABLES or sym in WRAPPED or sym not in perps:
                continue
            seen.add(sym)
            picked.append((sym, c.get('name', sym), round(c.get('market_cap') or 0)))
            if len(picked) >= 20:
                return picked, perps
        time.sleep(1.2)
    return picked, perps


def fetch_klines_fapi(base, start=WARMUP_START, tries=4):
    """分页拉取币安USDT永续K线（fapi 限1500/次）；6小时内磁盘缓存直接复用"""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f'{base}_{TF}.pkl')
    if os.path.exists(cache) and time.time() - os.path.getmtime(cache) < 6 * 3600:
        try:
            df_c = pd.read_pickle(cache)
            if df_c is not None and len(df_c) >= YEAR_BARS_MIN:
                return df_c
        except Exception:
            pass
    sym = f"{base}USDT"
    url = 'https://fapi.binance.com/fapi/v1/klines'
    since = int(pd.Timestamp(start, tz='UTC').timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows = []
    for _ in range(tries):
        try:
            while since < end_ms:
                kl = _s.get(url, params={'symbol': sym, 'interval': TF,
                                         'startTime': since, 'limit': 1500}, timeout=30).json()
                if not isinstance(kl, list) or not kl:
                    break
                rows.extend(kl)
                since = kl[-1][6] + 1
                if len(kl) < 1500:
                    break
                time.sleep(0.25)
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


def simulate(df, signals, tp=None, sl=None, mode='long_only', initial=INITIAL, comm=COMM):
    """与 tmp_us_macd_div_2024_2025.py 相同的逐bar模拟；数组循环提速"""
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


def main():
    print("== 选取市值前20的币安USDT永续合约币 ==")
    top20, perps = get_top20_perp_bases()
    print(f"币安USDT永续共{len(perps)}个；入选（按市值降序）：")
    for i, (b, name, mc) in enumerate(top20, 1):
        print(f"  {i:2}. {b:8} {name}  市值≈${mc/1e9:.1f}B")

    results = {w[0]: {} for w in WINDOWS}   # {year: {base: {variant: {...}, '_bh':..}}}
    for idx, (base, name, mc) in enumerate(top20, 1):
        df = fetch_klines_fapi(base)
        if df is None or len(df) < YEAR_BARS_MIN:
            print(f"[{idx}/20] {base}: 数据不足，跳过"); continue
        cov = f"{df.index[0]:%Y-%m-%d}~{df.index[-1]:%Y-%m-%d}"
        # 指标/信号在整段历史上连续计算（含预热），再分年切片模拟
        sigs = {}
        for v, (ud, um, uv) in DIVERGENCE_VARIANTS.items():
            try:
                _, sig = build_variant_signals(df, ud, um, uv)
                sigs[v] = sig
            except Exception as e:
                print(f"  {base} {v} 信号出错: {e}")
        line = f"[{idx}/20] {base} {cov} bar{len(df)}"
        for wname, w0, w1 in WINDOWS:
            m = (df.index >= pd.Timestamp(w0, tz='UTC')) & (df.index < pd.Timestamp(w1, tz='UTC'))
            dw = df[m]
            if len(dw) < YEAR_BARS_MIN:
                continue
            row = {}
            for v in VLIST:
                if v not in sigs:
                    row[v] = None; continue
                row[v] = simulate(dw, sigs[v].loc[dw.index], tp=TP, sl=SL, mode=MODE)
            row['_bh'] = (dw['close'].iloc[-1] / dw['close'].iloc[0] - 1) * 100
            row['_nbar'] = len(dw)
            results[wname][base] = row
        r0 = results['2024'].get(base, {}).get(VLIST[0])
        line += f" | 2024 {VLIST[0]}: " + (f"{r0['ret']:+.1f}%" if r0 else "无")
        print(line, flush=True)

    # ---- 汇总 ----
    print("\n" + "#" * 78)
    tpsl_desc = {'none': '不设止盈止损', 'sl5': '只止损5%', 'tpsl5': '止盈止损各5%'}[TPSL_ARG]
    print(f"## 分年汇总（{TF} · {'双向多空' if MODE == 'long_short' else '仅做多'} · {tpsl_desc} · "
          f"手续费{COMM:.1%} · 初始{INITIAL:.0f}USDT）")
    for wname, _, _ in WINDOWS:
        R = results[wname]
        if not R:
            print(f"\n【{wname}】无数据"); continue
        print(f"\n【{wname}】 标的数={len(R)}")
        bh = [R[t]['_bh'] for t in R]
        print(f"  买入持有BH   平均 {np.mean(bh):+8.1f}%  正收益 {sum(x>0 for x in bh)}/{len(bh)}")
        for v in VLIST:
            rs = [R[t][v] for t in R if R[t][v]]
            if not rs:
                print(f"  {v:<22} 无有效交易"); continue
            rets = [x['ret'] for x in rs]; mdds = [x['mdd'] for x in rs]; ns = [x['n'] for x in rs]
            shs = [x['sh'] for x in rs if x['sh'] is not None]
            beat = sum(1 for t in R if R[t][v] and R[t][v]['ret'] > R[t]['_bh'])
            print(f"  {v:<22} 平均 {np.mean(rets):+8.1f}%  中位 {np.median(rets):+8.1f}%  "
                  f"胜率 {sum(x>0 for x in rets)}/{len(rets)}  回撤均值 {np.mean(mdds):6.1f}%  "
                  f"笔数 {np.mean(ns):5.1f}  夏普均值 {np.mean(shs):+.2f}  跑赢BH {beat}/{len(rs)}")

    print("\n—— 每币种明细（收益%/回撤%/笔数；BH=买入持有） ——")
    for wname, _, _ in WINDOWS:
        R = results[wname]
        if not R:
            continue
        print(f"\n【{wname}】")
        for t in sorted(R):
            row = R[t]
            parts = "  ".join(
                f"{v}:{row[v]['ret']:+7.1f}%/mdd{row[v]['mdd']:.0f}%/n{row[v]['n']}" if row[v]
                else f"{v}:无" for v in VLIST)
            print(f"  {t:8} BH{row['_bh']:+8.1f}%  {parts}")

    os.makedirs('scripts/results', exist_ok=True)
    out = f'scripts/results/top20_macd_div_{TF}_{MODE_SUFFIX}_{TPSL_ARG}_2024_2026.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'channel': f'币安USDT永续{TF} fapi', 'top20': top20, 'windows': WINDOWS,
                   'variants': VLIST, 'mode': MODE, 'tpsl': tpsl_desc, 'comm': COMM,
                   'results': results}, f, ensure_ascii=False, indent=1, default=float)
    print(f"\nsaved → {out}")


if __name__ == '__main__':
    main()
