# -*- coding: utf-8 -*-
"""1h双向sl5（前20）：逐笔交易的盘中最大不利偏移(MAE)分析
回答：收盘价判定的止损，盘中插针会打穿多深？影响杠杆决策。

方法：读Excel逐笔明细（开/平仓时间、方向、收益率=收盘价口径），读K线缓存，
取每笔持仓期间的最低价(多单)/最高价(空单)相对开仓价的最差偏移。
"""
import os, sys, io, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(HERE, 'results', 'top20_macd_vol_trades_detail_2024_2026.xlsx')
CACHE = os.path.join(HERE, 'cache')

PROXY = "http://127.0.0.1:7892"
os.environ.setdefault("HTTP_PROXY", PROXY); os.environ.setdefault("HTTPS_PROXY", PROXY)

_s = requests.Session()
_s.proxies = {'http': PROXY, 'https': PROXY}
_s.headers['User-Agent'] = 'Mozilla/5.0'


def fetch_1h(base, tries=3):
    """读缓存(不限时效，K线历史不变)或重拉（2023-10起，含时间列）"""
    p = os.path.join(CACHE, f'{base}_1h.pkl')
    if os.path.exists(p):
        try:
            df = pd.read_pickle(p)
            if df is not None and len(df) > 3000:
                return df
        except Exception:
            pass
    sym = f"{base}USDT"
    since = int(pd.Timestamp('2023-10-01', tz='UTC').timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows = []
    for _ in range(tries):
        try:
            while since < end_ms:
                kl = _s.get('https://fapi.binance.com/fapi/v1/klines',
                            params={'symbol': sym, 'interval': '1h',
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
            print(f'  fetch {base} err: {repr(e)[:50]}', file=sys.stderr)
        time.sleep(3)
    if not rows:
        return None
    df = pd.DataFrame([r[:6] for r in rows], columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df = df.set_index('timestamp')[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    try:
        df.to_pickle(p)
    except Exception:
        pass
    return df


def main():
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    ws = wb['1h_双向多空_只止损5%']
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    col = {h: i for i, h in enumerate(header)}
    trades = []
    for r in rows:
        if r[col['开仓时间']] is None:
            continue
        t0, t1 = r[col['开仓时间']], r[col['平仓时间']]
        if isinstance(t0, str):
            t0 = datetime.fromisoformat(t0)
        if isinstance(t1, str):
            t1 = datetime.fromisoformat(t1)
        trades.append((str(r[col['币种']]), t0, t1, str(r[col['方向']]),
                       float(r[col['开仓价格']]), float(r[col['收益率(%)']])))
    print(f'逐笔明细: {len(trades)} 笔')

    klines = {}
    for b in sorted({t[0] for t in trades}):
        klines[b] = fetch_1h(b)
        if klines[b] is None:
            print(f'  {b} K线缺失!')

    recs = []
    miss = 0
    for b, t0, t1, d, entry, ret in trades:
        df = klines.get(b)
        if df is None:
            miss += 1
            continue
        m = (df.index >= pd.Timestamp(t0)) & (df.index <= pd.Timestamp(t1))
        if m.sum() == 0:
            miss += 1
            continue
        seg = df.loc[m]
        if d == '做多':
            worst = seg['low'].min()
            mae = (worst - entry) / entry * 100      # 负值，盘中最深浮亏
        else:
            worst = seg['high'].max()
            mae = (entry - worst) / entry * 100      # 负值
        recs.append((ret, mae, d, b))

    rets = np.array([r[0] for r in recs])
    maes = np.array([r[1] for r in recs])
    n = len(recs)
    print(f'匹配成功 {n} 笔（缺失 {miss}）\n')

    print('== 全部交易：盘中最大不利偏移 MAE（相对开仓价）==')
    for q in (50, 90, 99, 100):
        v = np.percentile(maes, q) if q < 100 else maes.min()
        print('  P%-3d MAE = %6.1f%%' % (q, v))

    # 收盘价亏损单 vs 盘中穿透
    loss = rets <= 0
    print('\n== 亏钱单（收盘价口径 n=%d）的盘中穿透 ==\n' % loss.sum())
    lm = maes[loss]
    for th in (-10, -15, -20, -30):
        cnt = (lm < th).sum()
        print('  盘中曾低于 %d%%: %d 笔 (%.1f%%)' % (th, cnt, cnt / len(lm) * 100))

    # 收盘价接近-5%止损的单（触发止损），盘中实际多深
    sl_trig = (rets > -6.5) & (rets < -3.5)
    if sl_trig.sum():
        print('\n== 触发止损5%%的单（收盘价-3.5~-6.5, n=%d）盘中深度 ==' % sl_trig.sum())
        m2 = maes[sl_trig]
        print('  MAE 中位 %.1f%% | P90 %.1f%% | 最深 %.1f%%' % (
            np.percentile(m2, 50), np.percentile(m2, 90), m2.min()))
        for th in (-10, -15, -20):
            print('  盘中曾低于 %d%%: %d 笔 (%.1f%%)' % (th, (m2 < th).sum(), (m2 < th).mean() * 100))

    # 杠杆推算：单份权益损失 = 名义亏损 × 杠杆
    print('\n== 杠杆含义（95%仓位下单笔最差盘中浮亏 → 该份权益损失）==')
    worst_mae = maes.min()
    for lev in (1, 2, 3):
        # 名义亏损 = |MAE|；投入该份的权益 = 名义/lev
        eq_loss = (1 - (1 + worst_mae / 100) ** lev) * 100 if worst_mae > -100 else 100
        print('  杠杆%dx: 最深MAE %.1f%% → 该份权益浮亏 %.0f%%%s' % (
            lev, worst_mae, eq_loss, '  ← 爆仓' if eq_loss >= 95 else ''))

    # 极端案例
    idx = maes.argsort()[:5]
    print('\n== 盘中最深5笔 ==')
    for i in idx:
        print('  %s %s 收盘价收益%+.1f%% 盘中最深%.1f%%' % (recs[i][3], recs[i][2], rets[i], maes[i]))


if __name__ == '__main__':
    main()
