# -*- coding: utf-8 -*-
"""验证引擎实现的三种 MACD+量能策略（12/26/9, 12/16/5, 12/16/7）与回测口径逐根一致"""
import os, sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from indicators import TechnicalIndicators
from divergence_signals import compute_divergence_signals, DIVERGENCE_VARIANTS, find_divergence_name, VARIANT_MACD_PARAMS

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
BASES = ['BTC', 'ETH', 'SOL', 'DOGE', 'ICP', 'GRAM']
VARIANTS = {'macd+量能': (12, 26, 9), 'macd 12/16/5+量能': (12, 16, 5), 'macd 12/16/7+量能': (12, 16, 7)}

# 1) 变体注册与参数映射检查
for name, params in VARIANTS.items():
    assert name in DIVERGENCE_VARIANTS, f'{name} 未注册'
    assert find_divergence_name([name]) == name, f'{name} 检索失败'
    assert VARIANT_MACD_PARAMS.get(name, (12, 26, 9)) == params, f'{name} MACD参数错误'
    assert DIVERGENCE_VARIANTS[name] == (False, False, True), f'{name} flags错误'
print('✓ 三种变体注册/检索/MACD参数全部正确（均为 背离=否 均线=否 量能=是）')

# 2) 引擎信号 vs 回测口径逐根一致
for base in BASES:
    p = os.path.join(CACHE, f'{base}_4h.pkl')
    if not os.path.exists(p):
        print(f'- {base} 无缓存, 跳过'); continue
    df = pd.read_pickle(p)
    for name, (f, s, g) in VARIANTS.items():
        _, sig_engine = compute_divergence_signals(df, name)
        # 回测口径：tmp_macd_best_lev.py 的 build_signal
        dft = TechnicalIndicators.calculate_macd(df, f, s, g)
        macd, sigl = dft['MACD'], dft['MACD_signal']
        macd_buy = (macd > sigl) & (macd.shift(1) <= sigl.shift(1))
        macd_sell = (macd < sigl) & (macd.shift(1) >= sigl.shift(1))
        vol_up = df['volume'] > df['volume'].rolling(20).mean() * 1.5
        sig_bt = pd.Series(0, index=df.index)
        sig_bt[(macd_buy & vol_up).fillna(False)] = 1
        sig_bt[(macd_sell & vol_up).fillna(False)] = -1
        assert bool((sig_engine == sig_bt).all()), f'{base} {name} 信号不一致!'
        nb, ns = int((sig_bt == 1).sum()), int((sig_bt == -1).sum())
        print(f'✓ {base:6} {name:20} 逐根一致 (买{nb} 卖{ns}, 共{len(df)}根)')

print('\n全部验证通过：实盘引擎三种策略信号与回测口径完全一致，可直接实盘。')
