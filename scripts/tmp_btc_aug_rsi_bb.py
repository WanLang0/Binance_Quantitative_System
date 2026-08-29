# -*- coding: utf-8 -*-
"""BTC 1h 8月1-24日 RSI+布林带, 止盈3%/止损3%, OR逻辑"""
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

SYMBOLS = ["ETH/USDT"]
TIMEFRAME = "1h"
INITIAL_CAPITAL = 10000
COMMISSION = 0.001
STOP_LOSS = 0.03
TAKE_PROFIT = 0.03
WEEKLY_CLOSE = False
SIGNAL_MODE = 'or'  # 系统已切换为 OR 逻辑

PARAMS = {
    "rsi": True, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70,
    "boll": True, "bb_period": 20, "bb_std": 2,
}

out_dir = os.path.join(ROOT, "results", "ETH_8月")
os.makedirs(out_dir, exist_ok=True)

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

for sym in SYMBOLS:
    print(f"=== {sym} ===", flush=True)
    df = None
    for attempt in range(4):
        try:
            df = fetcher.fetch_historical_data(sym, "2026-08-01", "2026-08-25", TIMEFRAME)
            if df is not None and not df.empty:
                break
        except Exception as e:
            print(f" 拉取失败(尝试{attempt+1}): {e}", flush=True)
            time.sleep(5)
    if df is None or df.empty:
        print(" 无数据", flush=True); continue
    print(f" {len(df)} 根 ({df.index[0]} ~ {df.index[-1]})", flush=True)

    dft = TechnicalIndicators.calculate_all_indicators(df, PARAMS)
    engine = BacktestEngine(INITIAL_CAPITAL, COMMISSION, take_profit=TAKE_PROFIT,
                            stop_loss=STOP_LOSS, timeframe=TIMEFRAME, weekly_close=WEEKLY_CLOSE,
                            signal_mode=SIGNAL_MODE)
    res = engine.run_backtest(dft, PARAMS)

    trades_df = res['trades'].copy()
    if not trades_df.empty:
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    pairs = build_pairs(trades_df)
    total_pnl = pairs['盈亏(USDT)'].sum() if not pairs.empty else 0
    tp = int((trades_df['action']=='TAKE_PROFIT').sum()) if not trades_df.empty else 0
    sl = int((trades_df['action']=='STOP_LOSS').sum()) if not trades_df.empty else 0

    print(f" K线 {len(df)} | 收益 {res['total_return']:.2f}% | 总盈亏 {total_pnl:.2f}U | 交易{len(pairs)} | 止盈{tp}/止损{sl} | 胜率{res['win_rate']:.1f}%", flush=True)

    fn = os.path.join(out_dir, f"ETH_1h_8月_RSI_BB_止损3止盈3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    with pd.ExcelWriter(fn, engine='openpyxl') as w:
        # 汇总
        pd.DataFrame([{'币种': sym, 'K线数': len(df), '收益率%': round(res['total_return'],2),
                       '总盈亏(USDT)': round(total_pnl,2), '交易数': len(pairs), '止盈': tp, '止损': sl,
                       '胜率%': round(res['win_rate'],2), '信号逻辑': 'OR'}]).to_excel(w, sheet_name='策略对比', index=False)
        if not pairs.empty:
            pairs.to_excel(w, sheet_name='交易明细', index=False)
        df.to_excel(w, sheet_name='K线数据', index=False)
    print(f"✅ 已保存: {fn}", flush=True)
