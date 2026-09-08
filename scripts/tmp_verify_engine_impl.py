# -*- coding: utf-8 -*-
"""验证引擎实现的 macd+量能 信号与回测冠军脚本(tmp_atr4h_lev)逐根一致 + ATR口径一致"""
import os, sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import ta
from ta.volatility import AverageTrueRange

from indicators import TechnicalIndicators
from divergence_signals import compute_divergence_signals, DIVERGENCE_VARIANTS, find_divergence_name

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
BASES = ['BTC', 'ETH', 'SOL', 'DOGE', 'ICP', 'GRAM']

# 1) 变体注册检查
flags = DIVERGENCE_VARIANTS.get('macd+量能')
assert flags == (False, False, True), f'变体flags错误: {flags}'
assert find_divergence_name(['macd+量能']) == 'macd+量能'
assert find_divergence_name(['EMA', 'macd+量能']) == 'macd+量能'
print('✓ 变体注册/检索正确: macd+量能 -> (背离=否, 均线=否, 量能=是)')

for base in BASES:
    p = os.path.join(CACHE, f'{base}_4h.pkl')
    if not os.path.exists(p):
        print(f'- {base} 无缓存, 跳过'); continue
    df = pd.read_pickle(p)

    # 2) 引擎路径：compute_divergence_signals
    _, sig_engine = compute_divergence_signals(df, 'macd+量能')

    # 3) 回测路径：tmp_atr4h_lev.load 的手写信号
    dft = TechnicalIndicators.calculate_macd(df, 12, 26, 9)
    macd, sigl = dft['MACD'], dft['MACD_signal']
    macd_buy = (macd > sigl) & (macd.shift(1) <= sigl.shift(1))
    macd_sell = (macd < sigl) & (macd.shift(1) >= sigl.shift(1))
    vol_up = df['volume'] > df['volume'].rolling(20).mean() * 1.5
    sig_bt = pd.Series(0, index=df.index)
    sig_bt[(macd_buy & vol_up).fillna(False)] = 1
    sig_bt[(macd_sell & vol_up).fillna(False)] = -1

    same = (sig_engine == sig_bt).all()
    nb = int((sig_bt == 1).sum()); ns = int((sig_bt == -1).sum())
    assert bool(same), f'{base} 信号不一致!'
    print(f'✓ {base:6} 信号逐根一致 (买{nb} 卖{ns}, 共{len(df)}根)')

    # 4) ATR 口径一致：引擎 calculate_atr vs 回测 AverageTrueRange
    atr_engine = TechnicalIndicators.calculate_atr(df, 14)
    atr_bt = AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
    ok = np.allclose(atr_engine.dropna().values, atr_bt.dropna().values)
    assert ok, f'{base} ATR不一致!'
print('✓ ATR14 口径一致（引擎 calculate_atr ≡ 回测 AverageTrueRange）')
print('\n全部验证通过：实盘引擎 macd+量能 信号/ATR 与回测冠军口径完全一致。')
