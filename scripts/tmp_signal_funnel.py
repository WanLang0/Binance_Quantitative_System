# -*- coding: utf-8 -*-
"""信号漏斗分解：定位下单少的瓶颈层
统计链条：原始MACD交叉 -> 过量能过滤(1.5x20均量) -> 实际可入场(仓状态机约束) -> 1h 30币合计
口径与 tmp_macd_3param_positions.py 一致。
"""
import os, sys, io, json, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange

from indicators import TechnicalIndicators
from tmp_combo_full import BASES, W0, W1

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
CLIP_HI = 0.08
PER, K_SL, K_TP = 14, 1.5, 2.0
MAINT = 0.005
VARIANTS = {'12/26/9': (12, 26, 9), '12/16/5': (12, 16, 5), '12/16/7': (12, 16, 7)}


def load_data(tf):
    data = {}
    for base in BASES:
        p = os.path.join(CACHE, f'{base}_{tf}.pkl')
        if not os.path.exists(p):
            continue
        df = pd.read_pickle(p)
        df = df[df.index < W1]
        if len(df) < 300:
            continue
        atr = AverageTrueRange(df['high'], df['low'], df['close'], PER).average_true_range()
        data[base] = (df, atr)
    return data


def main():
    data = load_data('1h')
    ncoins = len(data)
    days = (W1 - W0).days
    print(f'1h · {ncoins} 币 · 窗口 {W0.date()} ~ {W1.date()}（{days} 天）\n')

    out = {}
    for name, (f, s, g) in VARIANTS.items():
        raw_cross = vol_pass = entry_sig = 0
        vol_pass_pct_vals = []
        for base, (df, atr) in data.items():
            dft = TechnicalIndicators.calculate_macd(df, f, s, g)
            macd_buy = (dft['MACD'] > dft['MACD_signal']) & (dft['MACD'].shift(1) <= dft['MACD_signal'].shift(1))
            macd_sell = (dft['MACD'] < dft['MACD_signal']) & (dft['MACD'].shift(1) >= dft['MACD_signal'].shift(1))
            vol_up = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).fillna(False)
            # 量能强度分布（过滤阈值处通过率）
            m = df['volume'] / df['volume'].rolling(20).mean()
            vol_pass_pct_vals.extend(m.dropna().tolist())

            w = df.index >= W0
            raw_cross += int((macd_buy | macd_sell)[w].sum())
            vol_pass += int(((macd_buy | macd_sell) & vol_up)[w].sum())
            # 仓状态机下真正入场的信号（复盘与positions脚本同逻辑）
            idx = df.index.to_numpy(); close = df['close'].to_numpy()
            hi = df['high'].to_numpy(); lo = df['low'].to_numpy()
            sig = pd.Series(0, index=df.index)
            sig[(macd_buy & vol_up).fillna(False)] = 1
            sig[(macd_sell & vol_up).fillna(False)] = -1
            sig = sig.to_numpy()
            atr_a = atr.to_numpy()
            units = 0.0; entry = 0.0; side = 0; cur_tp = cur_sl = 0.0
            for i in range(len(df)):
                price = close[i]
                if not np.isfinite(price) or price <= 0:
                    continue
                sv = int(sig[i]) if i > 0 else 0
                if side != 0 and entry > 0:
                    if side > 0:
                        tp_lvl, sl_lvl = entry * (1 + cur_tp), entry * (1 - cur_sl)
                        hit_tp_i = hi[i] >= tp_lvl; hit_sl_c = price <= sl_lvl
                    else:
                        tp_lvl, sl_lvl = entry * (1 - cur_tp), entry * (1 + cur_sl)
                        hit_tp_i = lo[i] <= tp_lvl; hit_sl_c = price >= sl_lvl
                    maint = units * entry * MAINT
                    equity = (units * (price - entry) if side > 0 else units * (entry - price)) + 1.0
                    if equity <= maint or hit_sl_c or hit_tp_i:
                        units = 0.0; side = 0
                if sv == 1 and side <= 0 and idx[i] >= W0:
                    if side == 0: entry_sig += 1
                    entry = price; side = 1
                    if np.isfinite(atr_a[i]) and atr_a[i] > 0:
                        apct = atr_a[i] / price
                        cur_sl = float(np.clip(K_SL * apct, 0.01, CLIP_HI))
                        cur_tp = float(np.clip(K_TP * apct, 0.01, CLIP_HI))
                    else:
                        cur_sl = cur_tp = 0.05
                    units = 1.0
                elif sv == -1 and side >= 0 and idx[i] >= W0:
                    if side == 0: entry_sig += 1
                    entry = price; side = -1
                    if np.isfinite(atr_a[i]) and atr_a[i] > 0:
                        apct = atr_a[i] / price
                        cur_sl = float(np.clip(K_SL * apct, 0.01, CLIP_HI))
                        cur_tp = float(np.clip(K_TP * apct, 0.01, CLIP_HI))
                    else:
                        cur_sl = cur_tp = 0.05
                    units = 1.0

        m = pd.Series(vol_pass_pct_vals)
        out[name] = dict(raw=raw_cross, vol_pass=vol_pass, entries=entry_sig)
        print(f'===== {name} (1h) =====')
        print(f'  原始MACD交叉(金叉+死叉) : {raw_cross:6} 次 · 单币 {raw_cross/days/ncoins*365/365:.2f} 次/天 · 组合 {raw_cross/days:.2f} 次/天')
        print(f'  过量能过滤(1.5x20均量)后 : {vol_pass:6} 次 · 保留率 {vol_pass/raw_cross*100:5.1f}% · 组合 {vol_pass/days:.2f} 次/天')
        print(f'  仓状态机实际开仓(空仓->开): {entry_sig:6} 次 · 组合 {entry_sig/days:.2f} 次/天 · 单币 {entry_sig/days/ncoins:.3f} 次/天')
        print(f'  量比分布: 中位 {m.median():.2f}x · P75 {m.quantile(0.75):.2f}x · >=1.5x占比 {(m>=1.5).mean()*100:.1f}%\n')

    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'macd_3param_funnel.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f'明细 → {p}')


if __name__ == '__main__':
    main()
