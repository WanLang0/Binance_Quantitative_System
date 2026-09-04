# -*- coding: utf-8 -*-
"""macd+背离+量能 指定3组配置：逐笔交易明细回测 → Excel

配置（用户指定）：
  1) 15m 仅做多   不设
  2) 15m 双向多空 止盈止损各5%
  3) 1h  双向多空 只止损5%

每笔记录：币种/年份/方向/开仓时间/开仓价格/平仓时间/平仓价格/持仓时长/平仓原因/
投入资金/净盈亏/收益率/平仓后权益。口径与 tmp_top20_macd_div_1h_2024_2026.py 完全一致
（初始1万USDT/95%仓位/单边手续费0.1%/年末强平/止损按收盘价判定/做空现金背书无资金费率），
仅新增逐笔记录。输出：scripts/results/top20_macd_vol_trades_detail_2024_2026.xlsx
"""
import os, sys, io, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

PROXY = "http://127.0.0.1:7892"
os.environ.setdefault("HTTP_PROXY", PROXY); os.environ.setdefault("HTTPS_PROXY", PROXY)

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone

from divergence_signals import DIVERGENCE_VARIANTS, build_variant_signals

VARIANT = 'macd+背离+量能'
CONFIGS = [   # (tf, mode, tpsl)
    ('15m', 'long_only', 'none'),
    ('15m', 'long_short', 'tpsl5'),
    ('1h', 'long_short', 'sl5'),
]
TPSL_CFG = {'none': (None, None), 'sl5': (None, 0.05), 'tpsl5': (0.05, 0.05)}
TPSL_DESC = {'none': '不设', 'sl5': '只止损5%', 'tpsl5': '止盈止损各5%'}
MODE_DESC = {'long_only': '仅做多', 'long_short': '双向多空'}
COMM = 0.001
INITIAL = 10000.0
WARMUP_START = '2023-10-01'
YEAR_BARS_MIN = 200
WINDOWS = [
    ('2024', '2024-01-01', '2025-01-01'),
    ('2025', '2025-01-01', '2026-01-01'),
    ('2026YTD', '2026-01-01', datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')),
]
# 与此前回测完全相同的市值前20标的（CoinGecko排名∩币安USDT永续，按市值降序）
TOP20 = ['BTC', 'ETH', 'BNB', 'XRP', 'SOL', 'TRX', 'HYPE', 'ZEC', 'DOGE', 'XMR',
         'LINK', 'ADA', 'XLM', 'BCH', 'CC', 'LTC', 'UNI', 'GRAM', 'HBAR', 'AVAX']

_s = requests.Session()
_s.proxies = {'http': PROXY, 'https': PROXY}
_s.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36'


def fetch_klines_fapi(base, tf, start=WARMUP_START, tries=4):
    """分页拉取币安USDT永续K线；6小时内磁盘缓存复用（与原脚本一致）"""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f'{base}_{tf}.pkl')
    if os.path.exists(cache) and time.time() - os.path.getmtime(cache) < 6 * 3600:
        try:
            df_c = pd.read_pickle(cache)
            if df_c is not None and len(df_c) >= YEAR_BARS_MIN:
                return df_c
        except Exception:
            pass
    sym = f"{base}USDT"
    url = 'https://fapi.binance.com/fapi/v1/klines'
    since = int(pd.Timestamp(start, tz='UTC').timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows = []
    for _ in range(tries):
        try:
            while since < end_ms:
                kl = _s.get(url, params={'symbol': sym, 'interval': tf,
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
            print(f"  fetch {base} err: {repr(e)[:60]}", file=sys.stderr)
        time.sleep(3)
    if not rows:
        return None
    df = pd.DataFrame([r[:6] for r in rows],
                      columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df = df.set_index('timestamp')[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df = df.dropna(subset=['close'])
    df = df[df['close'] > 0]
    try:
        df.to_pickle(cache)
    except Exception:
        pass
    return df


def _close(cash, units, entry, price, side, comm=COMM):
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm


def simulate_trades(df, signals, base, year, tp=None, sl=None, mode='long_only',
                    initial=INITIAL, comm=COMM):
    """与原 simulate() 逻辑完全一致，额外逐笔记账；返回 (聚合结果, 交易明细list)"""
    idx = df.index.to_numpy(); close = df['close'].to_numpy(); sig = signals.to_numpy()
    cash = initial; units = 0.0; entry = 0.0; side = 0; n = 0
    eq_t = []; eq_v = []
    trades = []
    entry_time = None; cash_at_entry = initial
    for i in range(len(df)):
        price = close[i]
        if not np.isfinite(price) or price <= 0:
            continue
        s = int(sig[i]) if i > 0 else 0
        if side != 0 and entry > 0:
            r = (price - entry) / entry if side > 0 else (entry - price) / entry
            if (tp and r >= tp) or (sl and r <= -sl):
                reason = ('止盈' if (tp and r >= tp) else '止损')
                cash_new = _close(cash, units, entry, price, side)
                trades.append(_mk(base, year, side, entry_time, entry, idx[i], price,
                                  reason, cash_at_entry, cash_new, cash))
                cash = cash_new; side = 0; units = 0; n += 1
                eq_t.append(idx[i]); eq_v.append(cash); continue
        eq = cash + (units * price if side > 0 else units * entry + (entry - price) * units if side < 0 else 0)
        if eq <= 0:
            return None, trades
        eq_t.append(idx[i]); eq_v.append(eq)
        if s == 1 and side <= 0:
            if side < 0:
                cash_new = _close(cash, units, entry, price, side)
                trades.append(_mk(base, year, side, entry_time, entry, idx[i], price,
                                  '信号反转', cash_at_entry, cash_new, cash))
                cash = cash_new; n += 1; side = 0; units = 0
            u = (cash * 0.95) / (price * (1 + comm))
            if u > 0:
                cash -= u * price * (1 + comm); units = u; entry = price; side = 1
                entry_time = idx[i]; cash_at_entry = cash + u * price * (1 + comm)
        elif s == -1 and side >= 0:
            if side > 0:
                cash_new = _close(cash, units, entry, price, side)
                trades.append(_mk(base, year, side, entry_time, entry, idx[i], price,
                                  '信号反转', cash_at_entry, cash_new, cash))
                cash = cash_new; n += 1; side = 0; units = 0
            if mode == 'long_short':
                u = (cash * 0.95) / price
                if u > 0:
                    cash -= u * price * (1 + comm); units = u; entry = price; side = -1
                    entry_time = idx[i]; cash_at_entry = cash + u * price * (1 + comm)
    if side != 0 and len(df) > 0:
        cash_new = _close(cash, units, entry, close[-1], side)
        trades.append(_mk(base, year, side, entry_time, entry, idx[-1], close[-1],
                          '年末强平', cash_at_entry, cash_new, cash))
        cash = cash_new; n += 1
        eq_t.append(idx[-1]); eq_v.append(cash)
    if n == 0:
        return None, trades
    eq = pd.Series(eq_v, index=pd.DatetimeIndex(eq_t)).sort_index()
    agg = {'ret': (eq.iloc[-1] / initial - 1) * 100, 'n': n}
    return agg, trades


def _mk(base, year, side, e_t, e_p, x_t, x_p, reason, cash_entry, cash_exit, cash_before_close):
    """组装一笔交易记录（中文表头）"""
    pnl = cash_exit - cash_entry
    hours = (pd.Timestamp(x_t) - pd.Timestamp(e_t)).total_seconds() / 3600
    return {
        '币种': base, '年份': year, '方向': '做多' if side > 0 else '做空',
        '开仓时间': str(pd.Timestamp(e_t)), '开仓价格': round(float(e_p), 6),
        '平仓时间': str(pd.Timestamp(x_t)), '平仓价格': round(float(x_p), 6),
        '持仓时长(小时)': round(hours, 2), '平仓原因': reason,
        '投入资金(USDT)': round(cash_entry, 2), '净盈亏(USDT)': round(pnl, 2),
        '收益率(%)': round(pnl / cash_entry * 100, 3) if cash_entry > 0 else 0.0,
        '平仓后权益(USDT)': round(cash_exit, 2),
    }


def main():
    print("== macd+背离+量能 · 3组配置逐笔明细回测 ==")
    print(f"标的：市值前20（{len(TOP20)}个），窗口：2024 / 2025 / 2026YTD")
    tfs = sorted({c[0] for c in CONFIGS})
    data = {tf: {} for tf in tfs}      # tf -> base -> df
    sigs = {tf: {} for tf in tfs}      # tf -> base -> signals
    ud, um, uv = DIVERGENCE_VARIANTS[VARIANT]
    for tf in tfs:
        for i, base in enumerate(TOP20, 1):
            df = fetch_klines_fapi(base, tf)
            if df is None or len(df) < YEAR_BARS_MIN:
                print(f"[{tf} {i}/20] {base}: 数据不足，跳过", flush=True)
                continue
            data[tf][base] = df
            try:
                _, sig = build_variant_signals(df, ud, um, uv)
                sigs[tf][base] = sig
            except Exception as e:
                print(f"[{tf} {i}/20] {base} 信号出错: {e}", flush=True)
                continue
            print(f"[{tf} {i}/20] {base} bar{len(df)} {df.index[0]:%Y-%m-%d}~{df.index[-1]:%Y-%m-%d}", flush=True)

    sheets = {}
    for tf, mode, tpsl in CONFIGS:
        tp, sl = TPSL_CFG[tpsl]
        name = f"{tf}_{MODE_DESC[mode]}_{TPSL_DESC[tpsl]}"
        rows = []
        for base in TOP20:
            df = data[tf].get(base)
            if df is None or base not in sigs[tf]:
                continue
            for wname, w0, w1 in WINDOWS:
                m = (df.index >= pd.Timestamp(w0, tz='UTC')) & (df.index < pd.Timestamp(w1, tz='UTC'))
                dw = df[m]
                if len(dw) < YEAR_BARS_MIN:
                    continue
                agg, tr = simulate_trades(dw, sigs[tf][base].loc[dw.index], base, wname,
                                          tp=tp, sl=sl, mode=mode)
                rows.extend(tr)
        tdf = pd.DataFrame(rows)
        sheets[name] = tdf
        rets = {y: round(tdf[tdf['年份'] == y]['净盈亏(USDT)'].sum(), 1)
                for y in ('2024', '2025', '2026YTD') if len(tdf[tdf['年份'] == y])}
        print(f"\n【{name}】共 {len(tdf)} 笔；各年净盈亏合计(USDT, 各币独立1万起始加总)：{rets}", flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results',
                       'top20_macd_vol_trades_detail_2024_2026.xlsx')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        for name, tdf in sheets.items():
            tdf.to_excel(w, sheet_name=name[:31], index=False)
        # 汇总sheet：按 配置×年份 统计笔数/胜率/净盈亏
        sm = []
        for name, tdf in sheets.items():
            for y in ('2024', '2025', '2026YTD'):
                r = tdf[tdf['年份'] == y]
                if not len(r):
                    continue
                sm.append({'配置': name, '年份': y, '笔数': len(r),
                           '胜率(%)': round((r['净盈亏(USDT)'] > 0).mean() * 100, 1),
                           '净盈亏合计(USDT)': round(r['净盈亏(USDT)'].sum(), 2),
                           '单笔均值(USDT)': round(r['净盈亏(USDT)'].mean(), 2),
                           '平均持仓(小时)': round(r['持仓时长(小时)'].mean(), 1)})
        pd.DataFrame(sm).to_excel(w, sheet_name='汇总', index=False)
    print(f"\nsaved → {out}")
    for name, tdf in sheets.items():
        print(f"  sheet[{name[:31]}] {len(tdf)} 行")


if __name__ == '__main__':
    main()
