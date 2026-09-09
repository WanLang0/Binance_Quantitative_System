# -*- coding: utf-8 -*-
"""临时验证: tmp_macd_vol_monthly_pos.simulate vs tmp_tp_close_backtest.simulate_lev
对同一资产/信号/参数下, 比较两者权益曲线终值是否一致, 证明逐月脚本口径与基准引擎等价。
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tmp_macd_vol_monthly_pos as M
import tmp_tp_close_backtest as T

data = T.load_data('1h')
bases = list(data.keys())
print(f'资产池: {len(bases)} 币')

VARIANTS = T.VARIANTS
worst = 0.0
cnt = 0
for name, (f, s, g) in VARIANTS.items():
    for lev in (1, 2, 4):
        for base in bases:
            df, atr = data[base]
            sig = T.build_signal(df, f, s, g, T.VOL_MULT)
            # 逐月脚本 simulate(默认 k_sl=T.K_SL, k_tp=T.K_TP, tp_mode='intraday')
            r_m = M.simulate(df, sig, T.INIT, atr, lev, T.W0)
            # 基准 simulate_lev(显式传同参数)
            r_t = T.simulate_lev(df, sig, T.INIT, atr, T.K_SL, T.K_TP, lev, T.W0, 'intraday')
            if r_m is None or r_t is None:
                if (r_m is None) != (r_t is None):
                    print(f'  !! {name} {lev}x {base}: 一方为None不一致')
                continue
            eq_m = r_m[0]
            eq_t = r_t[0]
            end_m = float(eq_m.iloc[-1])
            end_t = float(eq_t.iloc[-1])
            if end_t != 0:
                rel = abs(end_m - end_t) / abs(end_t)
            else:
                rel = abs(end_m - end_t)
            worst = max(worst, rel)
            if rel > 1e-9:
                print(f'  !! {name} {lev}x {base}: month_sim={end_m:.6f} base_sim={end_t:.6f} rel={rel:.2e}')
            cnt += 1
print(f'\n比较 {cnt} 组, 最大相对误差 = {worst:.3e}')
print('PASS: 逐月脚本权益曲线与基准引擎完全一致' if worst < 1e-9 else 'FAIL: 存在差异')
