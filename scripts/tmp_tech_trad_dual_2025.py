# -*- coding: utf-8 -*-
"""科技/传统 两板块代表性龙头股 双向(做多+做空) 综合量化优先匹配回测（份额数 1/2/10 对比）

口径：
- 数据：Yahoo Finance 1h（美股真实个股），经代理，与历史测试同渠道
- 双向最优策略：每只股票在「双向(long_short)」口径下，7 种策略 × 2 档止盈止损 全周期扫描，
  取收益最高者（signal_mode='or'，与 tmp_us_sectors_1h_2025.py 双向逻辑一致）
- 做多最优策略（对照）：signal_mode='and'，long_only 口径
- 量化优先匹配：总资金 1万 USDT 均分为 n 份份额，最多同时持仓 n 只，先触发先得，平仓后份额回收
- 测试区间：2025-04 / 05 / 07 / 08 各自独立回测（月初满资金+n份份额，月底强制平仓）
- 对比：份额数 1（单股）/ 2（两股）/ 10（综合量化）× 做多 vs 双向
"""
import os, sys, io, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7892")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7892")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from indicators import TechnicalIndicators
from backtest_engine import BacktestEngine

TOTAL_FUND = 10000.0
BUY_PCT = 0.95          # 每份份额实际投入 95%
COMM = 0.001            # 单边 0.1% 手续费
SHARE_COUNTS = [1, 2, 10]

# 20 只代表性龙头（科技 / 传统）
TECH = [
    ('NVDA', '英伟达', '资讯科技/AI芯片'),
    ('AAPL', '苹果', '资讯科技'),
    ('MSFT', '微软', '资讯科技'),
    ('GOOGL', '谷歌', '通讯服务'),
    ('META', 'Meta', '通讯服务'),
    ('MU', '美光', '存储'),
    ('AVGO', '博通', 'AI芯片'),
    ('TSM', '台积电', '半导体'),
    ('CRM', '赛富时', '云计算'),
    ('EQIX', '易昆尼克斯', '数据中心'),
]
TRAD = [
    ('JPM', '摩根大通', '金融'),
    ('LLY', '礼来', '医疗保健'),
    ('WMT', '沃尔玛', '必需消费'),
    ('PG', '宝洁', '必需消费'),
    ('XOM', '埃克森美孚', '能源'),
    ('AMZN', '亚马逊', '非必需消费'),
    ('TSLA', '特斯拉', '非必需消费'),
    ('BA', '波音', '工业'),
    ('RTX', '雷神技术', '航天/军工'),
    ('NEM', '纽蒙特', '黄金/原材料'),
]

# 策略名 -> 指标参数（与历史测试脚本 COMBOS 一致）
RSI = {"rsi": True, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70}
KDJ = {"kdj": True, "kdj_k_period": 9, "kdj_d_period": 3, "kdj_j_period": 3,
       "kdj_buy_threshold": 20, "kdj_sell_threshold": 80}
MACD = {"macd": True, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9}
EMA = {"ema": True, "ema_short": 12, "ema_long": 26, "ema_periods": [12, 26]}
DMA = {"ma_cross": True, "ma_cross_short": 10, "ma_cross_long": 30, "ma_cross_periods": [10, 30]}
BOLL = {"boll": True, "bb_period": 20, "bb_std": 2.0}


def _merge(*ds):
    m = {}
    for d in ds:
        m.update(d)
    return m


COMBOS = {'KDJ': KDJ, 'RSI': RSI, 'MACD': MACD, 'EMA': EMA, '双均线': DMA, '布林带': BOLL,
          'RSI+MACD': _merge(RSI, MACD)}
TPSL = [(None, None), (0.05, 0.05)]

engine_or = BacktestEngine(signal_mode='or')
engine_and = BacktestEngine(signal_mode='and')


def _norm_tz(idx):
    if idx.tz is None:
        return idx.tz_localize('America/New_York')
    try:
        return idx.tz_convert('America/New_York')
    except Exception:
        return idx


def fetch_1h(ticker, start='2025-01-01', tries=5):
    for _ in range(tries):
        try:
            df = yf.download(ticker, interval='1h', start=start, end=datetime.now(),
                             progress=False, auto_adjust=True)
            if df is None or df.empty:
                time.sleep(3)
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.index.name = 'timestamp'
            df.index = _norm_tz(df.index)
            df = df.dropna(subset=['close'])
            df = df[df['close'] > 0]
            if len(df) >= 200:
                return df
        except Exception as e:
            print(f'  fetch err {ticker}: {e}')
        time.sleep(3)
    return None


def _close(cash, units, entry, price, side, comm=0.001):
    """平仓后的现金（与 tmp_us_sectors_1h_2025.py 双向口径一致）"""
    if side > 0:
        return cash + units * price * (1 - comm)
    return cash + units * entry + (entry - price) * units - units * price * comm


def single_simulate(df, signals, tp, sl, mode, initial=10000.0, comm=0.001):
    """单标的全周期回测（做多/双向），与 tmp_us_sectors_1h_2025.py 的 simulate 一致。"""
    cash = initial
    units = 0.0
    entry = 0.0
    side = 0
    n = 0
    eq_pts = []
    for i, (ts, row) in enumerate(df.iterrows()):
        price = row['close']
        if not np.isfinite(price) or price <= 0:
            continue
        sig = int(signals.iloc[i]) if i > 0 else 0
        if side != 0 and entry > 0:
            r = (price - entry) / entry if side > 0 else (entry - price) / entry
            if (tp and r >= tp) or (sl and r <= -sl):
                cash = _close(cash, units, entry, price, side)
                side = 0
                units = 0
                n += 1
                eq_pts.append((ts, cash))
                continue
        eq = cash + (units * price if side > 0 else units * entry + (entry - price) * units if side < 0 else 0)
        if eq <= 0:
            return None
        eq_pts.append((ts, eq))
        if sig == 1 and side <= 0:
            if side < 0:
                cash = _close(cash, units, entry, price, side)
                n += 1
                side = 0
                units = 0
            u = (cash * 0.95) / (price * (1 + comm))
            if u > 0:
                cash -= u * price * (1 + comm)
                units = u
                entry = price
                side = 1
        elif sig == -1 and side >= 0:
            if side > 0:
                cash = _close(cash, units, entry, price, side)
                n += 1
                side = 0
                units = 0
            if mode == 'long_short':
                u = (cash * 0.95) / price
                if u > 0:
                    cash -= u * price * (1 + comm)
                    units = u
                    entry = price
                    side = -1
    if side != 0 and len(df) > 0:
        price = df['close'].iloc[-1]
        cash = _close(cash, units, entry, price, side)
        n += 1
        eq_pts.append((df.index[-1], cash))
    if n == 0:
        return None
    eq = pd.Series(dict(eq_pts)).sort_index()
    return {'ret': (eq.iloc[-1] / initial - 1) * 100, 'n': n}


def scan_best(df):
    """对单只股票全周期扫描，返回 (双向最优, 做多最优)，各为 (策略名, tp, sl)。"""
    best_dual = None
    best_long = None
    for strat, ip in COMBOS.items():
        dft = TechnicalIndicators.calculate_all_indicators(df, ip)
        s_or = engine_or.calculate_signals(dft, ip)
        s_and = engine_and.calculate_signals(dft, ip)
        for tp, sl in TPSL:
            r = single_simulate(df, s_or, tp, sl, 'long_short')
            if r and (best_dual is None or r['ret'] > best_dual[0]):
                best_dual = (r['ret'], strat, tp, sl)
            r2 = single_simulate(df, s_and, tp, sl, 'long_only')
            if r2 and (best_long is None or r2['ret'] > best_long[0]):
                best_long = (r2['ret'], strat, tp, sl)
    def _norm(strat, tp, sl):
        return strat, (tp or 0.0), (sl or 0.0)
    dual = _norm(*best_dual[1:]) if best_dual else ('EMA', 0.0, 0.0)
    long_ = _norm(*best_long[1:]) if best_long else ('EMA', 0.0, 0.0)
    return dual, long_


def simulate_portfolio(dfs, signals, tickers, tps, sls, month_start, month_end,
                       share_count, allow_short):
    """优先匹配回测：单月，最多同时持仓 share_count 只。返回 (已实现盈亏, 交易列表, 最大回撤)。"""
    share_value = TOTAL_FUND / share_count
    all_ts = set()
    for t in tickers:
        sub = dfs[t].index
        sub = sub[(sub >= month_start) & (sub <= month_end)]
        all_ts.update(sub)
    timeline = sorted(all_ts)

    available = share_count
    positions = {}      # ticker -> {cash, units, entry, side, in_px}
    closed_pnl = 0.0
    trades = []
    equity_curve = []

    for ts in timeline:
        # 1) 处理持仓：止盈 / 止损 / 反向信号
        for t in list(positions.keys()):
            if ts not in dfs[t].index:
                continue
            price = float(dfs[t].loc[ts, 'close'])
            pos = positions[t]
            sig = signals[t].get(ts, 0)
            r = (price - pos['entry']) / pos['entry'] if pos['side'] > 0 else (pos['entry'] - price) / pos['entry']
            reason = None
            if tps[t] > 0 and r >= tps[t]:
                reason = '止盈'
            elif sls[t] > 0 and r <= -sls[t]:
                reason = '止损'
            elif pos['side'] > 0 and sig == -1:
                reason = '反向平多'
            elif pos['side'] < 0 and sig == 1:
                reason = '反向平空'
            if reason:
                pos['cash'] = _close(pos['cash'], pos['units'], pos['entry'], price, pos['side'], COMM)
                pnl = pos['cash'] - share_value
                closed_pnl += pnl
                trades.append((t, pos['side'], pos['entry'], price, reason, pnl, r * 100))
                del positions[t]
                available += 1
        # 2) 开仓：先触发先得
        for t in tickers:
            if t in positions or available <= 0:
                continue
            if ts not in dfs[t].index:
                continue
            price = float(dfs[t].loc[ts, 'close'])
            sig = signals[t].get(ts, 0)
            if sig == 1:
                u = (share_value * BUY_PCT) / (price * (1 + COMM))
                if u > 0:
                    cash = share_value - u * price * (1 + COMM)
                    positions[t] = {'cash': cash, 'units': u, 'entry': price, 'side': 1, 'in_px': price}
                    available -= 1
            elif sig == -1 and allow_short:
                u = (share_value * BUY_PCT) / price
                if u > 0:
                    cash = share_value - u * price * (1 + COMM)
                    positions[t] = {'cash': cash, 'units': u, 'entry': price, 'side': -1, 'in_px': price}
                    available -= 1
        # 3) 净值（含持仓浮动盈亏）
        floating = 0.0
        for t, pos in positions.items():
            px = float(dfs[t].loc[ts, 'close']) if ts in dfs[t].index else pos['entry']
            floating += _close(pos['cash'], pos['units'], pos['entry'], px, pos['side'], COMM) - share_value
        equity_curve.append(TOTAL_FUND + closed_pnl + floating)

    # 4) 月末强制平仓
    for t in list(positions.keys()):
        df = dfs[t]
        sub = df.index[df.index <= month_end]
        last_close = float(df.loc[sub[-1], 'close']) if len(sub) else positions[t]['entry']
        pos = positions[t]
        pos['cash'] = _close(pos['cash'], pos['units'], pos['entry'], last_close, pos['side'], COMM)
        pnl = pos['cash'] - share_value
        closed_pnl += pnl
        r = (last_close - pos['entry']) / pos['entry'] if pos['side'] > 0 else (pos['entry'] - last_close) / pos['entry']
        trades.append((t, pos['side'], pos['entry'], last_close, '月末平仓', pnl, r * 100))
        available += 1
    equity_curve.append(TOTAL_FUND + closed_pnl)

    # 5) 最大回撤（净值曲线峰谷）
    peak = TOTAL_FUND
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return closed_pnl, trades, max_dd


def build_dataset(stocks):
    """拉数据 + 扫描双向/做多最优策略 + 计算两套信号。"""
    dfs, names = {}, {}
    dual_sigs, dual_tps, dual_sls = {}, {}, {}
    long_sigs, long_tps, long_sls = {}, {}, {}
    tickers = []
    for t, name, sector in stocks:
        df = fetch_1h(t)
        if df is None:
            print(f'  {t}（{name}）: 数据获取失败，跳过')
            continue
        (d_strat, d_tp, d_sl), (l_strat, l_tp, l_sl) = scan_best(df)

        dip = COMBOS.get(d_strat)
        if dip is None:
            print(f'  {t}: 双向策略「{d_strat}」无参数映射，跳过')
            continue
        lip = COMBOS.get(l_strat)
        if lip is None:
            lip = dip

        ddft = TechnicalIndicators.calculate_all_indicators(df, dip)
        d_s = engine_or.calculate_signals(ddft, dip)
        ldft = TechnicalIndicators.calculate_all_indicators(df, lip)
        l_s = engine_and.calculate_signals(ldft, lip)

        dfs[t] = df
        names[t] = name
        dual_sigs[t] = {ts: int(v) for ts, v in d_s.items()}
        dual_tps[t], dual_sls[t] = d_tp, d_sl
        long_sigs[t] = {ts: int(v) for ts, v in l_s.items()}
        long_tps[t], long_sls[t] = l_tp, l_sl
        tickers.append(t)
        print(f'  {t:<6}{name:<6} 双向最优={d_strat}({_tpsl(d_tp, d_sl)})  做多最优={l_strat}({_tpsl(l_tp, l_sl)})')
    return (dfs, names, tickers,
            dual_sigs, dual_tps, dual_sls, long_sigs, long_tps, long_sls)


def _tpsl(tp, sl):
    if tp <= 0 and sl <= 0:
        return '不设'
    return f'{tp*100:.0f}%/{sl*100:.0f}%'


def main():
    groups = {'科技产业': TECH, '传统产业': TRAD}
    months = {
        '4月': (pd.Timestamp('2025-04-01', tz='America/New_York'),
                pd.Timestamp('2025-04-30 23:59:59', tz='America/New_York')),
        '5月': (pd.Timestamp('2025-05-01', tz='America/New_York'),
                pd.Timestamp('2025-05-31 23:59:59', tz='America/New_York')),
        '7月': (pd.Timestamp('2025-07-01', tz='America/New_York'),
                pd.Timestamp('2025-07-31 23:59:59', tz='America/New_York')),
        '8月': (pd.Timestamp('2025-08-01', tz='America/New_York'),
                pd.Timestamp('2025-08-31 23:59:59', tz='America/New_York')),
    }

    datasets = {}
    for gname, stocks in groups.items():
        print(f'\n拉取并扫描 {gname}（双向 + 做多最优策略）...')
        datasets[gname] = build_dataset(stocks)
        print(f'  已就绪 {len(datasets[gname][2])} 只标的')

    month_names = list(months.keys())
    # results[mode][sc][gname][mname] = (ret%, max_dd%, wins, n_trades)
    modes = {'做多': False, '双向': True}
    results = {m: {} for m in modes}

    for mode_name, allow_short in modes.items():
        for sc in SHARE_COUNTS:
            results[mode_name][sc] = {}
            for gname, ds in datasets.items():
                dfs, names, tickers, dsig, dtp, dsl, lsig, ltp, lsl = ds
                if not tickers:
                    continue
                sigs = dsig if allow_short else lsig
                tps = dtp if allow_short else ltp
                sls = dsl if allow_short else lsl
                results[mode_name][sc][gname] = {}
                for mname, (ms, me) in months.items():
                    pnl, trades, max_dd = simulate_portfolio(
                        dfs, sigs, tickers, tps, sls, ms, me, sc, allow_short)
                    ret = pnl / TOTAL_FUND * 100
                    wins = sum(1 for x in trades if x[5] > 0)
                    results[mode_name][sc][gname][mname] = (ret, max_dd * 100, wins, len(trades))

    # 打印对比表
    for mode_name in modes:
        print('\n' + '=' * 100)
        print(f'【{mode_name}口径】份额数 1 / 2 / 10 对比')
        print('=' * 100)
        for sc in SHARE_COUNTS:
            print(f'\n--- 份额数 = {sc} ---')
            for gname in groups:
                print(f'  【{gname}】')
                header = '    月份    ' + ''.join(f'{m:>12}' for m in month_names)
                print(header)
                rets, dds = [], []
                for mname in month_names:
                    ret, dd, wins, n = results[mode_name][sc][gname][mname]
                    rets.append(ret)
                    dds.append(dd)
                row = '    收益率  ' + ''.join(f'{results[mode_name][sc][gname][m][0]:>+11.2f}%' for m in month_names)
                row += f'{np.mean(rets):>+11.2f}%'
                print(row)
                row = '    回撤    ' + ''.join(f'{results[mode_name][sc][gname][m][1]:>11.2f}%' for m in month_names)
                row += f'{np.mean(dds):>11.2f}%'
                print(row)
                row = '    胜率    ' + ''.join(f'{results[mode_name][sc][gname][m][2]:>8}/{results[mode_name][sc][gname][m][3]:<3}' for m in month_names)
                print(row)

    # 汇总对比（月均收益 / 月均回撤 / 最差回撤 / 收益回撤比）
    print('\n' + '=' * 100)
    print('汇总对比（4个月均值）')
    print('=' * 100)
    print(f'{"口径":<6}{"板块":<8}{"份额":>5}{"月均收益":>11}{"月均回撤":>11}{"最差回撤":>11}{"收益回撤比":>11}')
    for mode_name in modes:
        for gname in groups:
            for sc in SHARE_COUNTS:
                rets = [results[mode_name][sc][gname][m][0] for m in month_names]
                dds = [results[mode_name][sc][gname][m][1] for m in month_names]
                avg_ret = np.mean(rets)
                avg_dd = np.mean(dds)
                worst_dd = max(dds)
                ratio = avg_ret / avg_dd if avg_dd > 0 else 0.0
                label = {1: '单股', 2: '两股', 10: '综合'}[sc]
                print(f'{mode_name:<6}{gname:<8}{label:>4}({sc}){avg_ret:>+9.2f}%{avg_dd:>10.2f}%{worst_dd:>10.2f}%{ratio:>11.2f}')


if __name__ == '__main__':
    main()