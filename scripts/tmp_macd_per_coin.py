# -*- coding: utf-8 -*-
"""逐币种收益/回撤拆解：验证"持仓份额小但收益高"是否合理
为每个币种独立复利（INIT=10000/30），输出单币收益、MDD、交易笔数、胜率、
对组合的贡献占比，看组合收益是否被少数赢家币撑起。
口径与 tmp_vol12_backtest 一致：1.5x 量能、ATR止盈止损、C判定、30币均分独立复利。
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from tmp_vol12_backtest import VARIANTS, load_data, simulate_lev, INIT, K_SL, K_TP
from tmp_combo_full import W0, BASES


def run_per_coin(data, f, s, g, lev):
    """返回每币的 dict：终值倍数、MDD、笔数、胜率"""
    res = {}
    for base, (df, atr) in data.items():
        sig = build_signal(df, f, s, g)
        r = simulate_lev(df, sig, INIT, atr, K_SL, K_TP, lev, W0)
        if r is None:
            continue
        eq, fl_t, fl_w, n_liq, n_trade = r
        m = fl_t >= W0
        n = int(m.sum())
        w = int(fl_w[m].sum())
        e = eq / INIT
        final = float(e.iloc[-1])
        mdd = float(((e - e.cummax()) / e.cummax() * 100).min())
        res[base] = {'final': final, 'ret': (final - 1) * 100, 'mdd': mdd,
                     'trades': n, 'winrate': (w / n * 100 if n else None)}
    return res


def build_signal(df, f, s, g):
    from indicators import TechnicalIndicators
    dft = TechnicalIndicators.calculate_macd(df, f, s, g)
    macd_buy = (dft['MACD'] > dft['MACD_signal']) & (dft['MACD'].shift(1) <= dft['MACD_signal'].shift(1))
    macd_sell = (dft['MACD'] < dft['MACD_signal']) & (dft['MACD'].shift(1) >= dft['MACD_signal'].shift(1))
    vol_up = df['volume'] > df['volume'].rolling(20).mean() * 1.5
    sig = pd.Series(0, index=df.index)
    sig[(macd_buy & vol_up).fillna(False)] = 1
    sig[(macd_sell & vol_up).fillna(False)] = -1
    return sig


def main():
    import pandas as pd
    data = load_data('1h')
    print(f'1h 周期 {len(data)} 币 · 量能1.5x 基准 · 30币均分独立复利(每币 INIT=10000/30={INIT:.0f})', flush=True)

    configs = [('12/16/7', (12, 16, 7), 2), ('12/16/7', (12, 16, 7), 1),
               ('12/16/5', (12, 16, 5), 2)]
    out = {}
    for name, (f, s, g), lev in configs:
        res = run_per_coin(data, f, s, g, lev)
        # 排序
        by_ret = sorted(res.items(), key=lambda x: -x[1]['ret'])
        finals = np.array([v['final'] for v in res.values()])
        nav = finals.mean()   # 组合 = 30币终值平均（等权）。不计复利再平衡口径
        wins = [b for b, v in res.items() if v['ret'] > 0]
        losses = [b for b, v in res.items() if v['ret'] <= 0]
        total_abs = sum([v['ret'] for v in res.values() if v['ret'] > 0])
        out[f'{name}_{lev}x'] = res

        print(f'\n===== {name} · {lev}x =====', flush=True)
        print(f'  组合等权平均(终值倍数): {nav:.2f}x · 单币收益>0: {len(wins)}/{len(res)} · 亏损: {len(losses)}', flush=True)
        print(f'  {len(wins)}个赢家币贡献正收益合计: +{total_abs:.0f}%', flush=True)
        print('  --- 全部币种按收益排序(币/收益/MDD/笔数/胜率) ---', flush=True)
        for b, v in by_ret:
            print(f'  {b:>6}: {v["ret"]:+9.1f}%  MDD{v["mdd"]:7.1f}%  笔{v["trades"]:>4}  胜{v["winrate"] if v["winrate"] else 0:5.1f}%', flush=True)
        print('  --- 组合 = 上述30币终值倍数取平均 ---', flush=True)

    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results',
                     'macd_per_coin_breakdown.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f'\n明细 → {p}', flush=True)


if __name__ == '__main__':
    main()
