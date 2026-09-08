# -*- coding: utf-8 -*-
"""诚实口径策略家族筛选：30市值币, 15m双向tpsl5, 均分30份独立复利, 2024-01~2026-08。

所有信号均为无前视口径（已过 check_no_lookahead.py 校验）：
- macd+背离+量能(修复后) —— 验证锚点: 应≈ C口径 -76.6%
- 纯MACD / 双均线10-30 / EMA12-26 / RSI14(30/70) / KDJ(9,3,3,20/80) / 布林带(20,2)
- TD9+完美 / TD9+完美+反向9 (复用 tmp_top30_td9 因果实现)
- Donchian20突破(自建因果信号: 收盘破20根新高买/新低卖)
"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from indicators import TechnicalIndicators
from backtest_engine import BacktestEngine
from divergence_signals import compute_divergence_signals
from tmp_combo_full import BASES, W0, W1, simulate_cont

TOTAL = 10000.0
INIT = TOTAL / len(BASES)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
TP = SL = 0.05
PRIOR_LOOKBACK = 120

# 标准策略参数（与 composite_trader.strategy_params 一致）
STD = {
    '纯MACD':      {'macd': True, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9},
    'RSI14':       {'rsi': True, 'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70},
    'KDJ':         {'kdj': True, 'kdj_k_period': 9, 'kdj_d_period': 3, 'kdj_j_period': 3,
                    'kdj_buy_threshold': 20, 'kdj_sell_threshold': 80},
    '布林带':       {'boll': True, 'bb_period': 20, 'bb_std': 2.0},
    'EMA12-26':    {'ema': True, 'ema_short': 12, 'ema_long': 26},
    '双均线10-30':  {'ma_cross': True, 'ma_cross_short': 10, 'ma_cross_long': 30},
}
MERGED = {k: v for d in STD.values() for k, v in d.items()}   # 一次算齐全部指标


def td_setup_counts(close):
    n = len(close)
    buy_cnt = np.zeros(n, dtype=np.int32); sell_cnt = np.zeros(n, dtype=np.int32)
    b = s = 0
    for i in range(4, n):
        b = b + 1 if close[i] < close[i - 4] else 0
        s = s + 1 if close[i] > close[i - 4] else 0
        buy_cnt[i] = b; sell_cnt[i] = s
    return buy_cnt, sell_cnt


def td9_signals(df, use_perfect=True, use_prior=False):
    """TD9 因果信号（复制自 tmp_top30_td9，无前视）"""
    close = df['close'].to_numpy(); low = df['low'].to_numpy(); high = df['high'].to_numpy()
    buy_cnt, sell_cnt = td_setup_counts(close)
    sig = np.zeros(len(df), dtype=np.int8)
    last_buy9 = -(1 << 30); last_sell9 = -(1 << 30)
    for i in range(12, len(df)):
        if buy_cnt[i] == 9:
            ok = True
            if use_perfect:
                l8, l9 = low[i - 1], low[i]
                ok = (l8 <= low[i - 3] and l8 <= low[i - 2]) or (l9 <= low[i - 3] and l9 <= low[i - 2])
            if ok and use_prior:
                ok = (i - 8 - PRIOR_LOOKBACK) <= last_sell9 < (i - 8)
            if ok:
                sig[i] = 1
            last_buy9 = i
        if sell_cnt[i] == 9:
            ok = True
            if use_perfect:
                h8, h9 = high[i - 1], high[i]
                ok = (h8 >= high[i - 3] and h8 >= high[i - 2]) or (h9 >= high[i - 3] and h9 >= high[i - 2])
            if ok and use_prior:
                ok = (i - 8 - PRIOR_LOOKBACK) <= last_buy9 < (i - 8)
            if ok:
                sig[i] = -1
            last_sell9 = i
    return pd.Series(sig, index=df.index, dtype=int)


def donchian_signals(df, n=20):
    """Donchian 通道突破（因果：比较对象 shift(1)）"""
    hh = df['close'].rolling(n).max().shift(1)
    ll = df['close'].rolling(n).min().shift(1)
    buy = df['close'] > hh
    sell = df['close'] < ll
    s = pd.Series(0, index=df.index)
    s[buy] = 1; s[sell] = -1
    return s


def main():
    tf = sys.argv[1] if len(sys.argv) > 1 else '15m'
    eqs = {}; fls = {}; grid = None
    for k, base in enumerate(BASES, 1):
        p = os.path.join(CACHE, f'{base}_{tf}.pkl')
        if not os.path.exists(p):
            print(f'{base} 无缓存, 跳过'); continue
        df = pd.read_pickle(p)
        df = df[(df.index >= W0 - pd.Timedelta(days=1)) & (df.index < W1)]
        if df is None or len(df) < 500:
            continue
        # 一次算齐标准指标，供全部标准策略复用
        dfi = TechnicalIndicators.calculate_all_indicators(df.copy(), MERGED)
        engine = BacktestEngine(timeframe='15m', signal_mode='and')
        sigs = {}
        _, s = compute_divergence_signals(df.copy(), 'macd+背离+量能')
        sigs['背离+量能(修复后)'] = s
        for name, params in STD.items():
            sigs[name] = engine.calculate_signals(dfi, params)
        sigs['TD9+完美'] = td9_signals(df)
        sigs['TD9+完美+反向9'] = td9_signals(df, use_perfect=True, use_prior=True)
        sigs['Donchian20'] = donchian_signals(df)
        for name, sig in sigs.items():
            if sig is None or sig.empty or int((sig != 0).sum()) == 0:
                continue
            r = simulate_cont(df, sig, TP, SL, 'long_short', INIT)
            if r is None:
                continue
            eq, fl_t, fl_w = r
            eqs.setdefault(name, {})[base] = eq / INIT
            m = fl_t >= W0
            n_, w_ = fls.get(name, (0, 0))
            fls[name] = (n_ + int(m.sum()), w_ + int(fl_w[m].sum()))
            grid = eq.index if grid is None else grid.union(eq.index)
        print(f'[{k}/{len(BASES)}] {base:8} done', flush=True)

    result = {}
    for name, es in eqs.items():
        nav = sum(e.reindex(grid).ffill().fillna(1.0) for e in es.values()) / len(es)
        nav = nav.dropna()
        ret = (nav.iloc[-1] - 1) * 100
        mdd = float(((nav - nav.cummax()) / nav.cummax() * 100).min())
        me = nav.resample('ME').last().dropna()
        mr = me.pct_change().dropna() * 100
        n_, w_ = fls[name]
        per_year = {}
        prev = nav.iloc[0]
        for y, a, b in (('2024', '2024-01-01', '2025-01-01'), ('2025', '2025-01-01', '2026-01-01'),
                        ('2026', '2026-01-01', '2026-09-01')):
            a, b = pd.Timestamp(a, tz='UTC'), pd.Timestamp(b, tz='UTC')
            seg = nav[(nav.index >= a) & (nav.index < b)]
            if len(seg):
                per_year[y] = round((seg.iloc[-1] / prev - 1) * 100, 1)
                prev = seg.iloc[-1]
        result[name] = dict(total=round(ret, 1), mdd=round(mdd, 1),
                            monthly_avg=round(float(mr.mean()), 2),
                            neg_months=int((mr < 0).sum()), months=len(mr),
                            trades=n_, winrate=round(w_ / n_ * 100, 1) if n_ else None,
                            per_year=per_year)

    order = sorted(result.items(), key=lambda kv: kv[1]['total'], reverse=True)
    print(f'\n===== 诚实口径筛选 · 30币 {tf}双向tpsl5 均分30份 · 2024-01~2026-08 =====')
    print(f'{"策略":<18}{"总收益":>10}{"MDD":>9}{"月均":>8}{"亏损月":>8}{"笔数":>8}{"胜率":>8}{"  2024/2025/2026"}')
    for name, d in order:
        py = d['per_year']
        print(f'{name:<18}{d["total"]:>+9.1f}%{d["mdd"]:>8.1f}%{d["monthly_avg"]:>+7.2f}%'
              f'{d["neg_months"]:>5}/{d["months"]:<2}{d["trades"]:>8}{str(d["winrate"])+"%":>8}'
              f'   {py.get("2024","—"):>+8}/{py.get("2025","—"):>+8}/{py.get("2026","—"):>+8}')

    with open(f'scripts/results/honest_screen_{tf}.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1, default=float)
    print(f'\n明细 → scripts/results/honest_screen_{tf}.json')


if __name__ == '__main__':
    main()
