# -*- coding: utf-8 -*-
"""
ETH/USDT 现货 回测（每周强制清仓版）
策略：EMA(12/26)+MACD(12/26/9), 15m 周期
范围：2026-01-01 ~ 2026-08-24
初始资金 10000，手续费 0.1%，止损 5%
每自然周（周一~周日）周日24点强制清仓，按周统计盈亏，输出 Excel（含买卖点）
"""
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

os.environ.setdefault("HTTP_PROXY", "http://192.168.11.188:7892")
os.environ.setdefault("HTTPS_PROXY", "http://192.168.11.188:7892")

import pandas as pd
from datetime import datetime

from data_fetcher import BinanceDataFetcher
from indicators import TechnicalIndicators
from backtest_engine import BacktestEngine

SYMBOL = "ETH/USDT"
TIMEFRAME = "15m"
START = "2026-01-01"
END = "2026-08-24"
INITIAL_CAPITAL = 10000
COMMISSION = 0.001
STOP_LOSS = 0.05

indicator_params = {
    "ema": True, "ema_short": 12, "ema_long": 26, "ema_periods": [12, 26],
    "macd": True, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
}

fetcher = BinanceDataFetcher()
fetcher.set_market_type('spot')
print(f"拉取 {SYMBOL} {TIMEFRAME} {START} ~ {END} ...")
df = fetcher.fetch_historical_data(SYMBOL, START, END, TIMEFRAME)
print(f"K线数: {len(df)}, 范围 {df.index[0]} ~ {df.index[-1]}")

dft = TechnicalIndicators.calculate_all_indicators(df, indicator_params)
engine = BacktestEngine(INITIAL_CAPITAL, COMMISSION, take_profit=None,
                        stop_loss=STOP_LOSS, timeframe=TIMEFRAME, weekly_close=True)
results = engine.run_backtest(dft, indicator_params)

trades_df = results['trades'].copy()
trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
actions = trades_df['action'].value_counts().to_dict()
print("动作统计:", actions)

# 配对买卖：BUY ->（SELL/TAKE_PROFIT/STOP_LOSS/WEEKLY_CLOSE）
def build_pairs(tr):
    pairs = []
    open_buy = None
    for _, row in tr.iterrows():
        a = row['action']
        if a == 'BUY':
            open_buy = row
        elif a in ('SELL', 'TAKE_PROFIT', 'STOP_LOSS', 'WEEKLY_CLOSE') and open_buy is not None:
            cost = open_buy['shares'] * open_buy['price'] * (1 + COMMISSION)
            revenue = row['shares'] * row['price'] * (1 - COMMISSION)
            pnl = revenue - cost
            pnl_pct = pnl / cost * 100
            pairs.append({
                '买入时间': open_buy['timestamp'],
                '卖出时间': row['timestamp'],
                '买入价格': round(open_buy['price'], 2),
                '卖出价格': round(row['price'], 2),
                '数量': round(row['shares'], 6),
                '买入金额': round(cost, 2),
                '卖出金额': round(revenue, 2),
                '盈亏(USDT)': round(pnl, 2),
                '盈亏%': round(pnl_pct, 2),
                '平仓原因': '止损' if a == 'STOP_LOSS' else ('周末清仓' if a == 'WEEKLY_CLOSE' else ('止盈' if a == 'TAKE_PROFIT' else '信号卖出')),
                '持仓天数': round((row['timestamp'] - open_buy['timestamp']).total_seconds() / 86400.0, 2),
            })
            open_buy = None
    return pd.DataFrame(pairs)

pairs = build_pairs(trades_df)
print(f"完整交易对数: {len(pairs)}")

# 按自然周（周一~周日，以卖出时间所在周）聚合
def week_start(ts):
    # pandas Timestamp, 周一=0, 周日=6
    return pd.Timestamp(ts.date()) - pd.Timedelta(days=ts.weekday())

pairs['周起始'] = pairs['卖出时间'].apply(week_start)
pairs['周结束'] = pairs['周起始'] + pd.Timedelta(days=6)

weekly = pairs.groupby('周起始').agg(
    周交易数=('盈亏(USDT)', 'count'),
    周盈亏USDT=('盈亏(USDT)', 'sum'),
    周盈亏PCT=('盈亏(USDT)', lambda x: round(x.sum() / INITIAL_CAPITAL * 100, 2)),
    盈利笔数=('盈亏(USDT)', lambda x: int((x > 0).sum())),
    亏损笔数=('盈亏(USDT)', lambda x: int((x < 0).sum())),
    平均持仓天数=('持仓天数', 'mean'),
).reset_index()
weekly['周结束'] = weekly['周起始'] + pd.Timedelta(days=6)
weekly['周结束'] = weekly['周结束'].dt.strftime('%Y-%m-%d')
weekly['周起始'] = weekly['周起始'].dt.strftime('%Y-%m-%d')
weekly = weekly[['周起始', '周结束', '周交易数', '周盈亏USDT', '周盈亏PCT', '盈利笔数', '亏损笔数', '平均持仓天数']]
weekly['周盈亏USDT'] = weekly['周盈亏USDT'].round(2)
weekly['平均持仓天数'] = weekly['平均持仓天数'].round(2)

# 总体
total_pnl = pairs['盈亏(USDT)'].sum() if not pairs.empty else 0
print(f"\n=== 总体 ===")
print(f"总收益率: {results['total_return']:.2f}%")
print(f"最终资金: {results['final_equity']:.2f}")
print(f"最大回撤: {results['max_drawdown']:.2f}%")
print(f"夏普: {results['sharpe_ratio']:.2f}")
print(f"胜率: {results['win_rate']:.2f}%")
print(f"总盈亏: {total_pnl:.2f} USDT")

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                        f"ETH_weekly_clear_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

overview = pd.DataFrame({
    '项目': ['交易对', '周期', '开始', '结束', '策略', '初始资金', '手续费', '止损', '周末清仓',
             'K线数', '总收益率%', '最终资金', '最大回撤%', '夏普', '胜率%', '总交易数', '总盈亏USDT'],
    '数值': [SYMBOL, TIMEFRAME, START, END, 'EMA(12/26)+MACD(12/26/9)', INITIAL_CAPITAL,
            f"{COMMISSION*100}%", f"{STOP_LOSS*100}%", '是(周日24点)',
            len(df), round(results['total_return'], 2), round(results['final_equity'], 2),
            round(results['max_drawdown'], 2), round(results['sharpe_ratio'], 2),
            round(results['win_rate'], 2), len(pairs), round(total_pnl, 2)]
})

with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    overview.to_excel(writer, sheet_name='总览', index=False)
    weekly.to_excel(writer, sheet_name='按周盈亏', index=False)
    pairs.to_excel(writer, sheet_name='买卖点记录', index=False)

print(f"\n✅ 已输出: {out_path}")
