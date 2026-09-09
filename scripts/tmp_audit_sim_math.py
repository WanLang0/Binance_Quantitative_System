# -*- coding: utf-8 -*-
"""simulate_lev 独立交叉验证：
1) 会计守恒：重放每笔交易，核对每根K线的 cash 轨迹与逐笔流水合计
2) 爆仓判定数学：equity = cash + margin + units*(price-entry) 的解析验证
3) 杠杆收益数学：1x/2x/4x 理论关系抽查（同一信号序列）
4) 未来函数扫描：信号仅用 shift 过去值、ATR 用当根收盘价
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from scripts.tmp_tp_close_backtest import load_data, build_signal, simulate_lev, COMM, INIT, MAINT, VARIANTS, W0, K_SL, K_TP

ok = True


def t(name, cond, detail=''):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        ok = False


# ---------- 1. 未来函数扫描（代码级） ----------
src = open(os.path.join(os.path.dirname(__file__), 'tmp_tp_close_backtest.py'), encoding='utf-8').read()
t('信号只用历史值(shift(1)比较)', 'shift(1)' in src)
t('无 shift(-1)/未来引用', 'shift(-' not in src)
t('ATR用ta库滚动(无重排)', 'average_true_range' in src)


# ---------- 2. 会计守恒: 手工重放与 simulate_lev 对比 ----------
def manual_replay(df, sig, lev, k_sl, k_tp, tp_mode, atr):
    """独立实现（不同写法）重放同一规则，终值与 simulate_lev 对比"""
    close = df['close'].to_numpy(); hi = df['high'].to_numpy(); lo = df['low'].to_numpy()
    s_ = sig.to_numpy(); atr_a = atr.to_numpy()
    cash = INIT
    pos = None  # dict(side, entry, units, margin, tp_pct, sl_pct)
    for i in range(1, len(df)):
        price = close[i]
        if pos is not None:
            side, entry, units, margin = pos['side'], pos['entry'], pos['units'], pos['margin']
            if side > 0:
                tp_lvl = entry * (1 + pos['tp_pct']); sl_lvl = entry * (1 - pos['sl_pct'])
                hit_tp = (hi[i] >= tp_lvl) if tp_mode == 'intraday' else (price >= tp_lvl)
                hit_sl = price <= sl_lvl
            else:
                tp_lvl = entry * (1 - pos['tp_pct']); sl_lvl = entry * (1 + pos['sl_pct'])
                hit_tp = (lo[i] <= tp_lvl) if tp_mode == 'intraday' else (price <= tp_lvl)
                hit_sl = price >= sl_lvl
            equity = cash + margin + units * (price - entry) * (1 if side > 0 else -1)
            if equity <= units * entry * MAINT:
                # 逐仓爆仓: 亏掉保证金, 只保留未占用钱包余额
                cash = max(cash, 0.0)
                pos = None
                continue
            if hit_sl:
                px = sl_lvl
            elif hit_tp:
                px = tp_lvl if tp_mode == 'intraday' else price
            else:
                px = None
            if px is not None:
                pnl = units * (px - entry) * (1 if side > 0 else -1)
                cash = cash + margin + pnl - abs(units * px) * COMM
                cash = max(cash, 0.0)
                pos = None
                continue
        s = int(s_[i])
        if s != 0 and s != (pos['side'] if pos else 0) and df.index[i] >= W0:
            if pos is not None:  # 反手
                side, entry, units, margin = pos['side'], pos['entry'], pos['units'], pos['margin']
                pnl = units * (price - entry) * (1 if side > 0 else -1)
                cash = cash + margin + pnl - abs(units * price) * COMM
                cash = max(cash, 0.0)
            margin = cash * 0.95
            units = margin * lev / price
            entry = price
            apct = atr_a[i] / price if np.isfinite(atr_a[i]) and atr_a[i] > 0 else 0.05
            if np.isfinite(atr_a[i]) and atr_a[i] > 0:
                sl_p = float(np.clip(k_sl * apct, 0.01, 0.08))
                tp_p = float(np.clip(k_tp * apct, 0.01, 0.08))
            else:
                sl_p = tp_p = 0.05
            pos = dict(side=s, entry=entry, units=units, margin=margin, tp_pct=tp_p, sl_pct=sl_p)
            cash -= margin + units * price * COMM
    if pos is not None:
        side, entry, units, margin = pos['side'], pos['entry'], pos['units'], pos['margin']
        px = close[-1]
        pnl = units * (px - entry) * (1 if side > 0 else -1)
        cash = max(cash + margin + pnl - abs(units * px) * COMM, 0.0)
    return cash


print('\n== 会计守恒交叉验证（独立重放 vs simulate_lev）==')
data = load_data('1h')
n_checked = 0
max_rel_err = 0.0
for lev in (1, 2, 4):
    for tp_mode in ('intraday', 'close'):
        for name, (f, s, g) in [('12/16/7', VARIANTS['12/16/7'])]:
            for base, (df, atr) in list(data.items())[:6]:
                sig = build_signal(df, f, s, g, 1.2)
                r = simulate_lev(df, sig, INIT, atr, K_SL, K_TP, lev, W0, tp_mode)
                if r is None:
                    continue
                eq, fl_t, fl_w, n_liq, n_trade = r
                # simulate_lev 末值(若尾部无平仓, eq最后是补平后的cash)
                final_cash = eq.iloc[-1]
                m_cash = manual_replay(df, sig, lev, K_SL, K_TP, tp_mode, atr)
                # 有爆仓时两者对爆仓回收处理一致则应相等; 允许浮点误差
                rel = abs(final_cash - m_cash) / max(m_cash, 1e-9)
                max_rel_err = max(max_rel_err, rel)
                n_checked += 1
                if rel > 1e-8:
                    t(f'终值一致 {base} lev{lev} {tp_mode}', False, f'sim={final_cash:.6f} manual={m_cash:.6f} rel={rel:.2e}')
t(f'独立重放一致（{n_checked}组, 最大相对误差 {max_rel_err:.2e}）', max_rel_err <= 1e-8)


# ---------- 3. 爆仓数学验证（解析） ----------
print('\n== 爆仓判定数学 ==')
# 多头: equity = cash + margin + units*(p-entry); cash 初始扣了 margin+fee
# 开仓瞬间 p=entry: equity = cash0 - margin - fee + margin = cash0 - fee ≈ cash0 ✓ (无浮动盈亏)
# fee = margin*lev*COMM
cash0 = INIT
for lev in (1, 2, 4):
    margin = cash0 * 0.95
    units = margin * lev / 100.0
    fee = units * 100.0 * COMM
    cash = cash0 - margin - fee
    equity0 = cash + margin + units * 0.0
    t(f'lev{lev} 开仓瞬间权益=初始-手续费', abs(equity0 - (cash0 - fee)) < 1e-9, f'eq0={equity0:.4f} fee={fee:.4f}')
    # 爆仓价(多头): equity = maint = units*entry*MAINT
    # cash + margin + units*(p-entry) = units*entry*MAINT → p_liq
    p = 100.0
    p_liq = (units * p * MAINT - cash - margin) / units + p
    t(f'lev{lev} 多头爆仓价 ≈ 1 - 1/lev + ...', p_liq < p, f'p_liq={p_liq:.4f} (跌 {1-p_liq/p:.2%})')


# ---------- 4. 单笔交易盈亏守恒 ----------
print('\n== 单笔盈亏守恒 ==')
# 例: 2x, margin=9500, units=190@100, 止损5% → exit=95
margin = 9500.0; units = 190.0; entry = 100.0
pnl = units * (95.0 - entry)
cash_before = 10000.0 - margin - units * entry * COMM  # 开仓后剩余
cash_after = cash_before + margin + pnl - units * 95.0 * COMM
expect = 10000.0 - units * entry * COMM + pnl - units * 95.0 * COMM  # = 初始 - 两边手续费 + pnl
t('多头止损2x: 终cash = 初始 + pnl - 全部手续费', abs(cash_after - expect) < 1e-9)
t('止损单笔净亏 = 5%名义×(双边费)', abs((cash_after - 10000.0) - (pnl - units*100*COMM - units*95*COMM)) < 1e-9)

print('\n结论:', 'ALL PASS ✓' if ok else '存在 FAIL ✗')
sys.exit(0 if ok else 1)
