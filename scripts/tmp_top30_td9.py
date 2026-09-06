# -*- coding: utf-8 -*-
"""市值前30 × 神奇九转(TD Sequential Setup) 回测（币安USDT永续，2024/2025/2026YTD）

用法：python tmp_top30_td9.py [1h|15m] [all|2024|2025|2026YTD]     默认 1h all
      只测单年时预热起点自动前移92天（缓存名带预热起点，不覆盖共享全量缓存），
      若共享全量缓存新鲜则优先复用；6小时内直接复用

九转计数（TD Setup 简化版，即A圈通行的"神奇九转"）：
- 低9(买入)：连续9根 close[i] < close[i-4]；中途断档计数归零
- 高9(卖出)：连续9根 close[i] > close[i-4]
- 信号在第9根收盘确认并按当根收盘价成交（与项目其它信号口径一致）

三种变体：
1) td9基础           —— 仅计数到9即出信号
2) td9+完美          —— DeMark完美化过滤：低9要求第8或第9根最低价 ≤ 第6、7根最低价
                        （高9对称），过滤未创出衰竭极值的假9转
3) td9+完美+反向9    —— 上者基础上要求：本setup开始前 PRIOR_LOOKBACK 根内出现过
                        反向9转（先有反向动能衰竭，反转更可靠）

回测口径与 tmp_top20_macd_div_1h_2024_2026.py 完全一致：
- 标的：市值前30（复用 top40all_macd_vol_summary 今日名单）
- 数据：fapi K线 2023-10-01 起预热，分 2024/2025/2026YTD
- 模拟：初始1万USDT、95%资金入场、单边手续费0.1%、窗口末强平、
        止盈止损按收盘价判定；long_short：卖出信号空仓时开空（现金背书、无杠杆）
- 夏普年化 √年周期数（1h→8766）
"""
import os, sys, io, time, warnings, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

TF = sys.argv[1] if len(sys.argv) > 1 else '1h'
SEL = sys.argv[2] if len(sys.argv) > 2 else 'all'
PERIODS_PER_YEAR = {'1h': 8766, '15m': 35040}.get(TF, 8766)

PROXY = "http://127.0.0.1:7892"
os.environ.setdefault("HTTP_PROXY", PROXY); os.environ.setdefault("HTTPS_PROXY", PROXY)

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone

COMM = 0.001
INITIAL = 10000.0
DEFAULT_WARMUP = '2023-10-01'
YEAR_BARS_MIN = 200
PRIOR_LOOKBACK = 120          # 反向9转回看窗口（1h≈5天，15m≈30小时）
MODES = ('long_only', 'long_short')
TPSL_CFG = {'none': (None, None), 'sl5': (None, 0.05), 'tpsl5': (0.05, 0.05)}
ALL_WINDOWS = [
    ('2024', '2024-01-01', '2025-01-01'),
    ('2025', '2025-01-01', '2026-01-01'),
    ('2026YTD', '2026-01-01', datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')),
]
WINDOWS = [w for w in ALL_WINDOWS if SEL in ('all', w[0])]
assert WINDOWS, f'未知窗口参数: {SEL}'
# 只测单年时，预热起点=首窗口起点前92天（指标无长周期依赖，足够）
WARMUP_START = DEFAULT_WARMUP if SEL == 'all' else \
    (pd.Timestamp(WINDOWS[0][1], tz='UTC') - pd.Timedelta(days=92)).strftime('%Y-%m-%d')
# 变体名 → (完美化过滤, 反向9过滤)
TD9_VARIANTS = {
    'td9基础': (False, False),
    'td9+完美': (True, False),
    'td9+完美+反向9': (True, True),
}

_s = requests.Session()
_s.proxies = {'http': PROXY, 'https': PROXY}
_s.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64.0; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36'


def get_top30():
    """复用今日 top40all 名单的前30（CoinGecko市值 ∩ 币安USDT永续，剔除稳定币/封装）"""
    js = json.load(open('scripts/results/top40all_macd_vol_summary_2024_2026.json', encoding='utf-8'))
    return js['meta']['tickers'][:30]


def fetch_klines_fapi(base, start=WARMUP_START, tries=4):
    """分页拉取币安USDT永续K线；6小时内磁盘缓存直接复用。
    非默认预热起点时缓存名单独命名，且优先复用新鲜的共享全量缓存（不回写）。"""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    fname = f'{base}_{TF}.pkl' if start == DEFAULT_WARMUP else f'{base}_{TF}_from{start}.pkl'
    cands = [os.path.join(cache_dir, fname)]
    if start != DEFAULT_WARMUP:
        cands.append(os.path.join(cache_dir, f'{base}_{TF}.pkl'))
    for cache in cands:
        if os.path.exists(cache) and time.time() - os.path.getmtime(cache) < 6 * 3600:
            try:
                df_c = pd.read_pickle(cache)
                if df_c is not None and len(df_c) >= YEAR_BARS_MIN:
                    return df_c
            except Exception:
                pass
    cache = cands[0]
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


# ---------------- 神奇九转信号 ----------------
def td_setup_counts(close):
    """TD Setup 连续计数。返回 (buy_cnt, sell_cnt)：第i根的当前连续计数"""
    n = len(close)
    buy_cnt = np.zeros(n, dtype=np.int32)
    sell_cnt = np.zeros(n, dtype=np.int32)
    b = s = 0
    for i in range(4, n):
        b = b + 1 if close[i] < close[i - 4] else 0
        s = s + 1 if close[i] > close[i - 4] else 0
        buy_cnt[i] = b
        sell_cnt[i] = s
    return buy_cnt, sell_cnt


def td9_signals(df, use_perfect=True, use_prior=False):
    """生成 -1/0/1 信号：低9→+1，高9→-1（第9根收盘确认）"""
    close = df['close'].to_numpy(); low = df['low'].to_numpy(); high = df['high'].to_numpy()
    buy_cnt, sell_cnt = td_setup_counts(close)
    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    last_buy9 = -(1 << 30)   # 最近一次高9/低9完成的位置（无论是否被过滤均记录）
    last_sell9 = -(1 << 30)
    for i in range(12, n):   # 第9根至少在第12根之后
        if buy_cnt[i] == 9:
            ok = True
            if use_perfect:  # 完美化：第8或第9根最低价 ≤ 第6、7根最低价
                l8, l9 = low[i - 1], low[i]
                ok = (l8 <= low[i - 3] and l8 <= low[i - 2]) or \
                     (l9 <= low[i - 3] and l9 <= low[i - 2])
            if ok and use_prior:  # 本setup起点(i-8)前 PRIOR_LOOKBACK 根内出现过反向9
                ok = (i - 8 - PRIOR_LOOKBACK) <= last_sell9 < (i - 8)
            if ok:
                sig[i] = 1
            last_buy9 = i
        if sell_cnt[i] == 9:
            ok = True
            if use_perfect:
                h8, h9 = high[i - 1], high[i]
                ok = (h8 >= high[i - 3] and h8 >= high[i - 2]) or \
                     (h9 >= high[i - 3] and h9 >= high[i - 2])
            if ok and use_prior:
                ok = (i - 8 - PRIOR_LOOKBACK) <= last_buy9 < (i - 8)
            if ok:
                sig[i] = -1
            last_sell9 = i
    return pd.Series(sig, index=df.index, dtype=int), int((sig != 0).sum())


# ---------------- 模拟（与项目口径一致） ----------------
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


def main():
    top30 = get_top30()
    print(f"== 神奇九转(TD9) × 市值前30 × {TF} · 2024/2025/2026YTD ==")
    print(f"名单: {', '.join(top30)}\n")

    # results[year][base] = {'_bh':.., 'td9基础|mode|tpsl': {...}, 'n_sig': {variant: 信号数}}
    results = {w[0]: {} for w in WINDOWS}
    for k, base in enumerate(top30, 1):
        df = fetch_klines_fapi(base)
        if df is None or len(df) < YEAR_BARS_MIN:
            print(f"[{k}/30] {base}: 数据不足，跳过", flush=True)
            continue
        sigs, n_sigs = {}, {}
        for v, (up, upri) in TD9_VARIANTS.items():
            try:
                sig, ns = td9_signals(df, use_perfect=up, use_prior=upri)
                sigs[v] = sig; n_sigs[v] = ns
            except Exception as e:
                print(f"  {base} {v} 信号出错: {e}")
        line = f"[{k}/30] {base} {df.index[0]:%Y-%m-%d}~{df.index[-1]:%Y-%m-%d} bar{len(df)} 信号数: " \
               + " ".join(f"{v.split('+')[-1]}={n_sigs.get(v, 0)}" for v in TD9_VARIANTS)
        for wname, w0, w1 in WINDOWS:
            m = (df.index >= pd.Timestamp(w0, tz='UTC')) & (df.index < pd.Timestamp(w1, tz='UTC'))
            dw = df[m]
            if len(dw) < YEAR_BARS_MIN:
                continue
            row = {'_bh': (dw['close'].iloc[-1] / dw['close'].iloc[0] - 1) * 100}
            for v in TD9_VARIANTS:
                if v not in sigs:
                    continue
                sw = sigs[v].loc[dw.index]
                for mode in MODES:
                    for tk, (tp, sl) in TPSL_CFG.items():
                        row[f'{v}|{mode}|{tk}'] = simulate(dw, sw, tp=tp, sl=sl, mode=mode)
            results[wname][base] = row
        print(line, flush=True)

    # ---- 汇总 ----
    print("\n" + "#" * 96)
    print(f"## 分年汇总（{TF} · 手续费{COMM:.1%} · 初始{INITIAL:.0f}USDT · 95%仓位）")
    combo_keys = [f'{v}|{m}|{t}' for v in TD9_VARIANTS for m in MODES for t in TPSL_CFG]
    for wname, _, _ in WINDOWS:
        R = results[wname]
        if not R:
            print(f"\n【{wname}】无数据"); continue
        bh = [R[t]['_bh'] for t in R]
        print(f"\n【{wname}】 标的数={len(R)}   买入持有BH 平均 {np.mean(bh):+7.1f}%  正收益 {sum(x > 0 for x in bh)}/{len(bh)}")
        print(f"  {'变体':<18}{'模式':<10}{'止盈止损':<8}{'平均':>9}{'中位':>9}{'胜率':>8}{'MDD均值':>9}{'笔数均':>7}{'夏普均':>8}{'跑赢BH':>8}")
        for ck in combo_keys:
            rs = [R[t][ck] for t in R if R[t].get(ck)]
            if not rs:
                continue
            v, m, tk = ck.split('|')
            rets = [x['ret'] for x in rs]; mdds = [x['mdd'] for x in rs]; ns = [x['n'] for x in rs]
            shs = [x['sh'] for x in rs if x['sh'] is not None]
            beat = sum(1 for t in R if R[t].get(ck) and R[t][ck]['ret'] > R[t]['_bh'])
            print(f"  {v:<18}{'多空' if m == 'long_short' else '仅多':<10}{tk:<8}"
                  f"{np.mean(rets):>+8.1f}%{np.median(rets):>+8.1f}%"
                  f"{sum(x > 0 for x in rets):>5}/{len(rets):<3}{np.mean(mdds):>8.1f}%"
                  f"{np.mean(ns):>6.1f}{(np.mean(shs) if shs else float('nan')):>+7.2f}{beat:>6}/{len(rs):<4}")

    out = f'scripts/results/top30_td9_{TF}_{SEL}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'channel': f'币安USDT永续{TF} fapi', 'tickers': top30, 'windows': WINDOWS,
                   'variants': list(TD9_VARIANTS), 'modes': list(MODES), 'tpsl': list(TPSL_CFG),
                   'prior_lookback': PRIOR_LOOKBACK, 'comm': COMM, 'results': results},
                  f, ensure_ascii=False, indent=1, default=float)
    print(f"\nsaved → {out}")


if __name__ == '__main__':
    main()
