# -*- coding: utf-8 -*-
"""macd+背离+量能 × 前30币 × 三方案 × 杠杆倍率敏感性测试

核心问题：启用合约杠杆，用几倍杠杆 / 如何有条件用杠杆更安全收益更高。

杠杆模型（逐仓、单一币种、无风险保证金）：
- 名义仓位 = 保证金 × leverage；收益按名义仓位算 → 收益随杠杆近似线性放大
- 手续费 = 名义 × 0.1%（杠杆越高手续费占比越大 → 收益次线性，存在最优杠杆）
- 权益(保证金) = 现金 + 仓位浮动盈亏；回撤在保证金口径上计算
- 爆仓：逐仓，维持保证金率=0.5%，保证金=杠杆的倒数(1/L)。
   用K线high/low判断：若最低价触到"维持保证金线"，按该方向仓位归零。
   - 多单爆仓价 = entry*(1 - (1/L - mmr))      mmr=0.005
   - 空单爆仓价 = entry*(1 + (1/L - mmr))
- 两种停损口径：
   A) 价格停损5%（sltool='price'）→ 权益止损 = 5%×杠杆（最危险，杠杆越大权益止损失控）
   B) 权益停损5%（sltool='equity'）→ 价格停损 = 5%/杠杆（把权益回撤锁在5%，杠杆越大越贴近爆仓）
- 止盈统一价格口径：tpsl5 的止盈=5%价格；不设=无。

输出：收益率 / 最大回撤(保证金口径) / 爆仓次数 / 交易次数，按杠杆 1x..8x。
"""
import os, sys, io, warnings, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

PROXY = "http://127.0.0.1:7892"
os.environ.setdefault("HTTP_PROXY", PROXY); os.environ.setdefault("HTTPS_PROXY", PROXY)

import numpy as np
import pandas as pd
from datetime import datetime, timezone

from divergence_signals import DIVERGENCE_VARIANTS, build_variant_signals
from tmp_top40_all_macd_vol import COMM, INITIAL, YEAR_BARS_MIN, TPSL_CFG, PERIODS, fetch_klines

VARIANT = 'macd+背离+量能'
UD, UM, UV = DIVERGENCE_VARIANTS[VARIANT]
LEVERS = [1, 2, 3, 4, 5, 6, 8]
MMR = 0.005          # 逐仓维持保证金率 0.5%
CONFIG_NAMES = ['方案一 15m双向 tpsl5', '方案二 15m仅多 不设', '方案三 1h双向 sl5']
CONFIGS = [('15m', 'long_short', 'tpsl5'), ('15m', 'long_only', 'none'), ('1h', 'long_short', 'sl5')]
SLC = {  # {cfgname: 基准价格停损}
    '方案一 15m双向 tpsl5': 0.05,
    '方案二 15m仅多 不设': None,
    '方案三 1h双向 sl5': 0.05,
}
SLTOOLS = ['price', 'equity']
ANALYSIS_START = pd.Timestamp('2024-01-01', tz='UTC')

# ---- 前30名单（从 summary 读，与牛熊/年度一致）----
summary = json.load(open('scripts/results/top40all_macd_vol_summary_2024_2026.json', encoding='utf-8'))
top30 = summary['meta']['tickers'][:30]
print(f'前30名单: {top30}')


def simulate_lever(df, signals, lev, slt, tp=None, sl_price=None, mode='long_only',
                   ppy=35040, initial=INITIAL, comm=COMM):
    """单币含杠杆模拟（逐仓、单一持仓、无风险保证金）。
    记账：cash=可用余额；collateral=锁定的保证金；pnl=未实现盈亏；equity=cash+collateral+pnl。
    slt: 'price'(基准价格停损) | 'equity'(权益5%=价格5%/杠杆)。返回 dict 或 None。"""
    idx = df.index.to_numpy(); close = df['close'].to_numpy()
    high = df['high'].to_numpy(); low = df['low'].to_numpy()
    sig = signals.to_numpy()
    cash = initial; collateral = 0.0; pnl = 0.0
    units = 0.0; entry = 0.0; side = 0; n = 0; liq_ct = 0
    eq_t = []; eq_v = []
    mmr_frac = 1.0 / lev - MMR          # 多头/空头对称的爆仓价格距离
    no_pos = lambda: units == 0.0 or entry <= 0

    def _equity():
        return cash + collateral + pnl

    def _close_pos(price):
        """平仓：返还 collateral + pnl，扣平仓名义手续费。返回平仓后 cash。"""
        nonlocal cash, collateral, pnl, units, entry, side
        fee = (units * price) * comm
        cash += collateral + pnl - fee
        collateral = 0.0; pnl = 0.0; units = 0.0; entry = 0.0; side = 0
        return cash

    for i in range(len(df)):
        price = close[i]; hi = high[i]; lo = low[i]
        if not np.isfinite(price) or price <= 0:
            continue
        s = int(sig[i]) if i > 0 else 0
        # 持仓时更新未实现盈亏
        if not no_pos():
            # 先按当前收盘更新 pnl（用于停损/权益判断）
            pnl = units * (price - entry) if side > 0 else units * (entry - price)
            r = (price - entry) / entry if side > 0 else (entry - price) / entry
            eff_sl = None
            if slt == 'equity' and sl_price:
                eff_sl = sl_price / lev          # 权益停损 → 价格距离
            hit = False
            if slt == 'price' and sl_price and r <= -sl_price:
                hit = True
            elif slt == 'equity' and eff_sl and r <= -eff_sl:
                hit = True
            if not hit and tp and r >= tp:
                hit = True
            if hit:
                _close_pos(price); n += 1
                eq_t.append(idx[i]); eq_v.append(_equity()); continue
            # 爆仓：关 high/low 触及维持线
            if side > 0 and lo <= entry * (1 - mmr_frac):
                cash += 0              # 保证金全部损失(保守，忽略维持保证金余额)
                collateral = 0.0; pnl = 0.0; units = 0.0; entry = 0.0; side = 0
                n += 1; liq_ct += 1
                eq_t.append(idx[i]); eq_v.append(_equity()); continue
            if side < 0 and hi >= entry * (1 + mmr_frac):
                cash += 0
                collateral = 0.0; pnl = 0.0; units = 0.0; entry = 0.0; side = 0
                n += 1; liq_ct += 1
                eq_t.append(idx[i]); eq_v.append(_equity()); continue
        eq = _equity()
        if eq <= 0:
            return None
        eq_t.append(idx[i]); eq_v.append(eq)
        # 开仓：反向时先平仓（保证金口径）
        if s == 1 and side <= 0:
            if side < 0:
                _close_pos(price); n += 1
                eq = _equity()
            margin = eq * 0.95
            notional = margin * lev
            fee = notional * comm
            if margin > 0 and notional > 0 and price > 0:
                cash = cash + collateral - margin - fee   # 释放旧保证金→锁新保证金
                collateral = margin
                units = notional / price; entry = price; side = 1; pnl = 0.0
        elif s == -1 and side >= 0:
            if side > 0:
                _close_pos(price); n += 1
                eq = _equity()
            if mode == 'long_short':
                margin = eq * 0.95
                notional = margin * lev
                fee = notional * comm
                if margin > 0 and notional > 0 and price > 0:
                    cash = cash + collateral - margin - fee
                    collateral = margin
                    units = notional / price; entry = price; side = -1; pnl = 0.0
    if not no_pos():
        _close_pos(close[-1]); n += 1
        eq_t.append(idx[-1]); eq_v.append(_equity())
    if n == 0:
        return None
    eq = pd.Series(eq_v, index=pd.DatetimeIndex(eq_t)).sort_index()
    peak = eq.cummax(); mdd = ((eq - peak) / peak * 100).min()
    rr = eq.pct_change().dropna()
    sh = float(rr.mean() / rr.std() * np.sqrt(ppy)) if len(rr) >= 10 and rr.std() > 0 else None
    return {'ret': (eq.iloc[-1] / initial - 1) * 100, 'mdd': mdd, 'n': n,
            'liq': liq_ct, 'sh': sh}


def run_by_year(base):
    """单币：缓存三tf全量信号，按 [2024,2025,2026YTD] 分段，返回二维结果。
    返回 {tf: {year: sig}}"""
    out = {}
    for tf in sorted({c[0] for c in CONFIGS}):
        df = fetch_klines(base, tf)
        if df is None or len(df) < YEAR_BARS_MIN:
            continue
        try:
            _, sig = build_variant_signals(df, UD, UM, UV)
        except Exception:
            continue
        out[tf] = {'df': df, 'sig': sig}
    return out


years = [('2024', '2024-01-01', '2025-01-01'), ('2025', '2025-01-01', '2026-01-01'),
         ('2026YTD', '2026-01-01', datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M'))]

# res[cfgname][slt][lev] = {'tot':几何平均收益%, 'avg_ret':均收益%, 'pos':'x/y',
#                           'avg_mdd':%, 'max_mdd':%, 'liq':总爆仓, 'n':总交易,
#                           'n_avg':, 'runs':样本数}
from collections import defaultdict
res = {cn: {st: {lev: defaultdict(list) for lev in LEVERS} for st in SLTOOLS} for cn in CONFIG_NAMES}

for k, base in enumerate(top30, 1):
    tb = run_by_year(base)
    if not tb:
        continue
    for ctf, cmode, ctpsl in CONFIGS:
        tfd = tb.get(ctf)
        if not tfd:
            continue
        df, sig = tfd['df'], tfd['sig']
        for yr, w0, w1 in years:
            m = (df.index >= pd.Timestamp(w0, tz='UTC')) & (df.index < pd.Timestamp(w1, tz='UTC'))
            dw = df[m]; sw = sig.loc[dw.index]
            if len(dw) < YEAR_BARS_MIN:
                continue
            base_sl = SLC[CONFIG_NAMES[CONFIGS.index((ctf, cmode, ctpsl))]]
            tp, _ = TPSL_CFG[ctpsl]
            for lev in LEVERS:
                for slt in SLTOOLS:
                    r = simulate_lever(dw, sw, lev, slt, tp=tp,
                                       sl_price=base_sl, mode=cmode, ppy=PERIODS[ctf])
                    if r:
                        cnt = res[CONFIG_NAMES[CONFIGS.index((ctf, cmode, ctpsl))]][slt][lev]
                        cnt['rets'].append(r['ret']); cnt['mdds'].append(r['mdd'])
                        cnt['ns'].append(r['n']); cnt['liqs'].append(r['liq'])
    print(f'[{k}/{len(top30)}] {base} done', flush=True)

print('\n== 杠杆敏感性（前30 × macd+背离+量能，逐仓，2024-2026YTD 三段复利） ==')
print('口径：收益=保证金口径几何平均；MDD=保证金口径；爆仓=触及维持线归零次数(逐仓)')
print('停损商品：price=基准5%价格停损；equity=权益5%停损(价格=5%/杠杆)')

out_rows = []
for cn in CONFIG_NAMES:
    print(f'\n【{cn}】')
    row_base = {'name': cn, 'lev': {}}
    for slt in SLTOOLS:
        print(f'  -- 停损口径 = {slt} --')
        print(f'  {"杠杆":>3} {"均收益":>9} {"几何":>9} {"正收益":>7} {"均MDD":>9} {"最差MDD":>9} {"爆仓":>5} {"交易/均":>9}')
        for lev in LEVERS:
            acc = res[cn][slt][lev]
            if not acc['rets']:
                continue
            geo = (np.prod([1 + x / 100 for x in acc['rets']]) ** (1 / len(acc['rets'])) - 1) * 100
            avg_ret = float(np.mean(acc['rets'])); avg_mdd = float(np.mean(acc['mdds']))
            max_mdd = float(min(acc['mdds'])); liq = int(sum(acc['liqs']))
            n_avg = int(np.mean(acc['ns']))
            print(f'  {lev:>3}x {avg_ret:>+9.1f}% {geo:>+9.1f}% {sum(x>0 for x in acc["rets"]):>4}/{len(acc["rets"]):<3} '
                  f'{avg_mdd:>+9.1f}% {max_mdd:>+9.1f}% {liq:>5} {n_avg:>9}')
            row_base['lev'][f'{lev}x_{slt}'] = {'avg_ret': round(avg_ret, 1), 'geo': round(geo, 1),
                                                'pos': f'{sum(x>0 for x in acc["rets"])}/{len(acc["rets"])}',
                                                'avg_mdd': round(avg_mdd, 1), 'max_mdd': round(max_mdd, 1),
                                                'liq': liq, 'n_avg': n_avg}
    out_rows.append(row_base)

os.makedirs('scripts/results', exist_ok=True)
with open('scripts/results/top30_leverage_grid.json', 'w', encoding='utf-8') as f:
    json.dump({'variant': VARIANT, 'meta': {
                   'note': f'逐仓,维持保证金{MMR*100:.2f}%,95%保证金,手续费0.1%/边',
                   'basis': '价格停损price=基准5%; 权益停损equity=5%/杠杆',
                   'tickers': top30, 'configs': CONFIG_NAMES}, 'rows': out_rows},
              f, ensure_ascii=False, indent=1, default=float)
print('\nsummary → scripts/results/top30_leverage_grid.json')
