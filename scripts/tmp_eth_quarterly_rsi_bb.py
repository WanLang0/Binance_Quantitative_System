# -*- coding: utf-8 -*-
"""ETH 1h RSI+布林带(OR) 止盈3%/止损3% 按季度回测 2023/2024/2025"""
import os, sys, io, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
os.environ.setdefault("HTTP_PROXY", "http://192.168.11.188:7892")
os.environ.setdefault("HTTPS_PROXY", "http://192.168.11.188:7892")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datetime import datetime
import pandas as pd
from data_fetcher import BinanceDataFetcher
from indicators import TechnicalIndicators
from backtest_engine import BacktestEngine

SYMBOL = "ETH/USDT"
TIMEFRAME = "1h"
INITIAL_CAPITAL = 10000
COMMISSION = 0.001
STOP_LOSS = 0.03
TAKE_PROFIT = 0.03
WEEKLY_CLOSE = False
SIGNAL_MODE = 'or'

PARAMS = {
    "rsi": True, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70,
    "boll": True, "bb_period": 20, "bb_std": 2,
}

out_dir = os.path.join(ROOT, "results", "ETH_按季度_RSI_BB")
os.makedirs(out_dir, exist_ok=True)

# 年份 -> (拉取开始, 拉取结束)
year_ranges = {
    2023: ("2023-01-01", "2024-01-01"),
    2024: ("2024-01-01", "2025-01-01"),
    2025: ("2025-01-01", "2026-01-01"),
}
# 季度 -> (月份起点, 月份止点)
quarters = {"Q1": (1, 4), "Q2": (4, 7), "Q3": (7, 10), "Q4": (10, 13)}

fetcher = BinanceDataFetcher(); fetcher.set_market_type('spot')

def build_pairs(trades_df):
    pairs, open_buy = [], None
    for _, row in trades_df.iterrows():
        a = row['action']
        if a == 'BUY':
            open_buy = row
        elif a in ('SELL','TAKE_PROFIT','STOP_LOSS','WEEKLY_CLOSE') and open_buy is not None:
            cost = open_buy['shares']*open_buy['price']*(1+COMMISSION)
            revenue = row['shares']*row['price']*(1-COMMISSION)
            pnl = revenue - cost
            reason = '止损' if a=='STOP_LOSS' else ('止盈' if a=='TAKE_PROFIT' else ('周末清仓' if a=='WEEKLY_CLOSE' else '信号卖出'))
            pairs.append({'买入时间': open_buy['timestamp'], '卖出时间': row['timestamp'],
                          '买入价格': round(open_buy['price'],2), '卖出价格': round(row['price'],2),
                          '数量': round(row['shares'],6), '盈亏(USDT)': round(pnl,2),
                          '盈亏%': round(pnl/cost*100,2), '平仓原因': reason,
                          '持仓小时': round((row['timestamp']-open_buy['timestamp']).total_seconds()/3600,2)})
            open_buy = None
    return pd.DataFrame(pairs)

summary_rows = []
all_pairs = []

for year, (s, e) in year_ranges.items():
    print(f"\n=== {year} 拉取数据中... ===", flush=True)
    df = None
    for attempt in range(4):
        try:
            df = fetcher.fetch_historical_data(SYMBOL, s, e, TIMEFRAME)
            if df is not None and not df.empty:
                break
        except Exception as ex:
            print(f" 拉取失败(尝试{attempt+1}): {ex}", flush=True)
            time.sleep(5)
    if df is None or df.empty:
        print(f" {year} 无数据", flush=True); continue
    print(f" {len(df)} 根 ({df.index[0]} ~ {df.index[-1]})", flush=True)

    # 整年统一计算指标
    dft = TechnicalIndicators.calculate_all_indicators(df, PARAMS)

    for qlabel, (ms, me) in quarters.items():
        mfilter = (dft.index.month >= ms) & (dft.index.month < me)
        # 季度首日指标已含前期预热，直接切片回测
        seg = dft[mfilter]
        if seg.empty:
            summary_rows.append({'年份': year, '季度': qlabel, 'K线数': 0})
            continue
        engine = BacktestEngine(INITIAL_CAPITAL, COMMISSION, take_profit=TAKE_PROFIT,
                                stop_loss=STOP_LOSS, timeframe=TIMEFRAME, weekly_close=WEEKLY_CLOSE,
                                signal_mode=SIGNAL_MODE)
        res = engine.run_backtest(seg, PARAMS)
        trades_df = res['trades'].copy()
        if not trades_df.empty:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        pairs = build_pairs(trades_df)
        total_pnl = pairs['盈亏(USDT)'].sum() if not pairs.empty else 0
        tp = int((trades_df['action']=='TAKE_PROFIT').sum()) if not trades_df.empty else 0
        sl = int((trades_df['action']=='STOP_LOSS').sum()) if not trades_df.empty else 0
        win = len(pairs[pairs['盈亏(USDT)']>0]) if not pairs.empty else 0
        wr = round(win/len(pairs)*100,1) if len(pairs) else 0
        row = {'年份': year, '季度': qlabel, 'K线数': len(seg), '收益率%': round(res['total_return'],2),
               '总盈亏(USDT)': round(total_pnl,2), '交易数': len(pairs), '止盈': tp, '止损': sl,
               '胜率%': wr, '信号逻辑': 'OR'}
        summary_rows.append(row)
        print(f"  {year} {qlabel}: 收益{row['收益率%']:>7}% 盈亏{row['总盈亏(USDT)']:>8}U 交易{len(pairs):>3} 止盈{tp}/止损{sl} 胜率{wr:>5}%", flush=True)
        if not pairs.empty:
            pairs['年份'] = year; pairs['季度'] = qlabel
            all_pairs.append(pairs)

if summary_rows:
    sdf = pd.DataFrame(summary_rows)
    pvt = sdf.pivot(index='年份', columns='季度', values='收益率%')
    print("\n==== 季度收益率矩阵 (%) ====", flush=True)
    print(pvt.to_string(), flush=True)
    print("\n==== 季度总盈亏矩阵 (USDT) ====", flush=True)
    print(sdf.pivot(index='年份', columns='季度', values='总盈亏(USDT)').to_string(), flush=True)

    fn = os.path.join(out_dir, f"ETH_1h_按季度_RSI_BB_止损3止盈3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    with pd.ExcelWriter(fn, engine='openpyxl') as w:
        sdf.to_excel(w, sheet_name='季度汇总', index=False)
        # 透视表
        sdf.pivot(index='年份', columns='季度', values='收益率%').to_excel(w, sheet_name='收益率矩阵')
        sdf.pivot(index='年份', columns='季度', values='总盈亏(USDT)').to_excel(w, sheet_name='盈亏矩阵')
        if all_pairs:
            pd.concat(all_pairs).to_excel(w, sheet_name='交易明细', index=False)
    print(f"\n✅ 已保存: {fn}", flush=True)
