# -*- coding: utf-8 -*-
"""无前视(look-ahead)校验：任何信号上线前必须通过本检查。

原理：把「全量数据计算的信号 S[t]」与「只用 ≤t 数据(前缀截断)重算的信号 S[t]」逐点比对，
若信号依赖 t 之后的K线，前缀重算将无法复现全量值 → 报 FAIL。

覆盖：固定组合背离三种变体 / 标准策略(MACD·RSI·KDJ·布林·EMA·双均线) / TD9。
含负向对照 legacy_div(修复前的背离逻辑,标志在pivot当根) —— 必须FAIL,证明校验有效。

用法：python scripts/check_no_lookahead.py
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from indicators import TechnicalIndicators
from backtest_engine import BacktestEngine
from divergence_signals import compute_divergence_signals, _find_pivots, DIVERGENCE_VARIANTS

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
MAX_BARS = 30000          # 截取最近N根，校验不需全量（指标暖机充足）
N_CUTS = 8                # 每组抽取的截断点数
CASES = [('BTC', '15m'), ('ETH', '15m'), ('BTC', '1h')]

STD_PARAMS = {
    'MACD':        {'macd': True, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9},
    'RSI':         {'rsi': True, 'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70},
    'KDJ':         {'kdj': True, 'kdj_k_period': 9, 'kdj_d_period': 3, 'kdj_j_period': 3,
                    'kdj_buy_threshold': 20, 'kdj_sell_threshold': 80},
    '布林带':       {'boll': True, 'bb_period': 20, 'bb_std': 2.0},
    'EMA':         {'ema': True, 'ema_short': 12, 'ema_long': 26},
    '双均线交叉':   {'ma_cross': True, 'ma_cross_short': 10, 'ma_cross_long': 30},
}


# ---------- 被校验的信号函数：输入df → 输出与df等长的 -1/0/1 Series ----------
def sig_div(name):
    def f(df):
        _, s = compute_divergence_signals(df.copy(), name)
        return s
    return f


def sig_std(params):
    def f(df):
        dfi = TechnicalIndicators.calculate_all_indicators(df.copy(), params)
        return BacktestEngine(timeframe='15m', signal_mode='and').calculate_signals(dfi, params)
    return f


def td_setup_counts(close):
    n = len(close)
    buy_cnt = np.zeros(n, dtype=np.int32); sell_cnt = np.zeros(n, dtype=np.int32)
    b = s = 0
    for i in range(4, n):
        b = b + 1 if close[i] < close[i - 4] else 0
        s = s + 1 if close[i] > close[i - 4] else 0
        buy_cnt[i] = b; sell_cnt[i] = s
    return buy_cnt, sell_cnt


def sig_td9(df, use_perfect=True):
    close = df['close'].to_numpy(); low = df['low'].to_numpy(); high = df['high'].to_numpy()
    buy_cnt, sell_cnt = td_setup_counts(close)
    sig = np.zeros(len(df), dtype=np.int8)
    for i in range(12, len(df)):
        if buy_cnt[i] == 9:
            ok = True
            if use_perfect:
                l8, l9 = low[i - 1], low[i]
                ok = (l8 <= low[i - 3] and l8 <= low[i - 2]) or (l9 <= low[i - 3] and l9 <= low[i - 2])
            if ok:
                sig[i] = 1
        if sell_cnt[i] == 9:
            ok = True
            if use_perfect:
                h8, h9 = high[i - 1], high[i]
                ok = (h8 >= high[i - 3] and h8 >= high[i - 2]) or (h9 >= high[i - 3] and h9 >= high[i - 2])
            if ok:
                sig[i] = -1
    return pd.Series(sig, index=df.index, dtype=int)


def sig_legacy_div(df):
    """负向对照 = 修复前逻辑：pivot 右侧5根确认但标志打在 pivot 当根（含前视，必须FAIL）"""
    dfc = df.copy()
    from indicators import TechnicalIndicators as TI
    dft = TI.calculate_macd(dfc, 12, 26, 9)
    dfc['MACD'] = dft['MACD']
    px_high = _find_pivots(dfc['high'], 5, 'high')
    px_low = _find_pivots(dfc['low'], 5, 'low')
    macd = dfc['MACD'].to_numpy(); hi = dfc['high'].to_numpy(); lo = dfc['low'].to_numpy()
    top = np.zeros(len(dfc), dtype=bool); bot = np.zeros(len(dfc), dtype=bool)
    ph = None; pm = None
    for i in np.where(px_high.to_numpy())[0]:
        p = hi[i]; m = macd[i]
        if ph is not None and p > ph and m < pm:
            top[i] = True
        ph = p; pm = m
    pl = None; pm2 = None
    for i in np.where(px_low.to_numpy())[0]:
        p = lo[i]; m = macd[i]
        if pl is not None and p < pl and m > pm2:
            bot[i] = True
        pl = p; pm2 = m
    s = pd.Series(0, index=dfc.index)
    s[bot] = 1; s[top] = -1
    return s


# ---------- 校验器 ----------
def pick_cuts(full_sig, n, n_sig=12, n_rand=4):
    """截断点选取：必须覆盖全量计算的『信号bar』(sparse信号下随机点几乎都是0==0,无检出力)，
    另加少量随机bar。要求 t ≥ 400 保证前缀有足够暖机。"""
    pos = np.where(full_sig.to_numpy() != 0)[0]
    pos = pos[pos >= 400]
    picks = set()
    if len(pos):
        idxs = np.linspace(0, len(pos) - 1, min(n_sig, len(pos)), dtype=int)
        picks.update(int(pos[j]) for j in idxs)
    rnd = [int(t) for t in np.linspace(max(400, n // 3), n - 30, n_rand, dtype=int)]
    picks.update(rnd)
    return sorted(picks)


def check(name, sig_fn, df):
    """全量 vs 前缀截断 重算比对（截断点=信号bar+随机bar）。返回 (通过点数, 总点数, 首个不一致样本)"""
    full = sig_fn(df)
    cuts = pick_cuts(full, len(df))
    bad = None
    n_ok = 0
    for t in cuts:
        prefix = sig_fn(df.iloc[:t + 1].copy())
        v_full = int(full.iloc[t])
        v_pref = int(prefix.iloc[-1])
        if v_full == v_pref:
            n_ok += 1
        elif bad is None:
            bad = (str(df.index[t]), v_full, v_pref)
    return n_ok, len(cuts), bad


def main():
    results = []   # (信号, 币/周期, ok, total, bad)
    for base, tf in CASES:
        p = os.path.join(CACHE, f'{base}_{tf}.pkl')
        if not os.path.exists(p):
            print(f'{base}_{tf} 无缓存, 跳过'); continue
        df = pd.read_pickle(p)
        df = df.tail(MAX_BARS).copy()
        n = len(df)
        fns = {f'背离:{k}': sig_div(k) for k in DIVERGENCE_VARIANTS}
        fns.update({f'标准:{k}': sig_std(v) for k, v in STD_PARAMS.items()})
        fns['TD9+完美'] = sig_td9
        fns['负向对照:修复前背离'] = sig_legacy_div
        for name, fn in fns.items():
            ok, total, bad = check(name, fn, df)
            results.append((name, f'{base}_{tf}', ok, total, bad))
            print(f'  {name:<18} {base}_{tf}  {ok}/{total} {"PASS" if ok == total else "FAIL"}'
                  + (f'  不一致@{bad[0]} 全量={bad[1]} 前缀={bad[2]}' if bad else ''), flush=True)

    print('\n===== 汇总 =====')
    all_pass = True
    for name in dict.fromkeys(r[0] for r in results):
        rs = [r for r in results if r[0] == name]
        ok = sum(r[2] for r in rs); total = sum(r[3] for r in rs)
        is_neg = name.startswith('负向对照')
        line_pass = (ok == total) if not is_neg else (ok < total)
        if not line_pass:
            all_pass = False
        print(f'{name:<20} {ok}/{total}  {"PASS" if ok == total else "FAIL"}'
              f'{"  (负向对照,预期FAIL→校验有效)" if is_neg and ok < total else ""}')
    print('\n结论:', '全部信号无前视，校验器有效 ✅' if all_pass
          else '存在前视或校验器失效 ❌ —— 修复前禁止上线/采信回测')


if __name__ == '__main__':
    main()
