# -*- coding: utf-8 -*-
"""
BTC/USDT 现货 1m 2026H1 策略组合批量对比
条件：初始资金10000, 手续费0.1%, 止损5%, 止盈10%, 周末不清仓
仅测试多策略组合（不测单个策略）
"""
import os, sys, io, itertools, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7892")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7892")

from datetime import datetime
import pandas as pd
from data_fetcher import BinanceDataFetcher
from indicators import TechnicalIndicators
from backtest_engine import BacktestEngine

SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"
START = "2026-01-01"
END = "2026-06-30"
INITIAL_CAPITAL = 10000
COMMISSION = 0.001
STOP_LOSS = 0.05
TAKE_PROFIT = 0.10
WEEKLY_CLOSE = False

# 5 个可叠加策略
ALL_INDICATORS = ["RSI", "KDJ", "布林带", "EMA", "MACD"]

# 每个策略的默认参数构建函数
def build_params(combo):
    p = {}
    if "RSI" in combo:
        p.update({"rsi": True, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70})
    if "KDJ" in combo:
        p.update({"kdj": True, "kdj_k_period": 9, "kdj_d_period": 3, "kdj_j_period": 3,
                  "kdj_buy_threshold": 20, "kdj_sell_threshold": 80})
    if "布林带" in combo:
        p.update({"boll": True, "bb_period": 20, "bb_std": 2.0})
    if "EMA" in combo:
        p.update({"ema": True, "ema_short": 12, "ema_long": 26, "ema_periods": [12, 26]})
    if "MACD" in combo:
        p.update({"macd": True, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9})
    return p

# 生成所有多策略组合（数量 >= 2）
COMBOS = []
for r in range(2, len(ALL_INDICATORS) + 1):
    for combo in itertools.combinations(ALL_INDICATORS, r):
        COMBOS.append(list(combo))

print(f"策略组合数: {len(COMBOS)}")

# 拉取数据（一次拉取，所有组合共用）
fetcher = BinanceDataFetcher()
fetcher.set_market_type('spot')
print(f"拉取 {SYMBOL} {TIMEFRAME} {START} ~ {END} ...")
df = fetcher.fetch_historical_data(SYMBOL, START, END, TIMEFRAME)
if df is None or df.empty:
    print("数据为空，退出")
    sys.exit(1)
print(f"K线数: {len(df)}, 范围 {df.index[0]} ~ {df.index[-1]}")

def week_start(ts):
    return pd.Timestamp(ts.date()) - pd.Timedelta(days=ts.weekday())

def build_pairs(trades_df):
    """BUY -> (SELL/TAKE_PROFIT/STOP_LOSS/WEEKLY_CLOSE) 配对"""
    pairs = []
    open_buy = None
    for _, row in trades_df.iterrows():
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
                '平仓原因': '止损' if a == 'STOP_LOSS' else ('止盈' if a == 'TAKE_PROFIT' else ('周末清仓' if a == 'WEEKLY_CLOSE' else '信号卖出')),
                '持仓小时': round((row['timestamp'] - open_buy['timestamp']).total_seconds() / 3600.0, 2),
            })
            open_buy = None
    return pd.DataFrame(pairs)

# 逐组合回测
summary = []
weekly_all = []
detail_all = []
for combo in COMBOS:
    combo_name = "+".join(combo)
    ip = build_params(combo)
    dft = TechnicalIndicators.calculate_all_indicators(df, ip)
    engine = BacktestEngine(INITIAL_CAPITAL, COMMISSION, take_profit=TAKE_PROFIT,
                            stop_loss=STOP_LOSS, timeframe=TIMEFRAME, weekly_close=WEEKLY_CLOSE)
    results = engine.run_backtest(dft, ip)

    trades_df = results['trades'].copy()
    if not trades_df.empty:
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    pairs = build_pairs(trades_df)

    total_pnl = pairs['盈亏(USDT)'].sum() if not pairs.empty else 0
    # 止盈/止损次数
    tp_cnt = int((trades_df['action'] == 'TAKE_PROFIT').sum()) if not trades_df.empty else 0
    sl_cnt = int((trades_df['action'] == 'STOP_LOSS').sum()) if not trades_df.empty else 0

    summary.append({
        '策略组合': combo_name,
        '策略数': len(combo),
        '总收益率%': round(results['total_return'], 2),
        '最终资金': round(results['final_equity'], 2),
        '最大回撤%': round(results['max_drawdown'], 2),
        '夏普': round(results['sharpe_ratio'], 2),
        '胜率%': round(results['win_rate'], 2),
        '交易数': len(pairs),
        '止盈次数': tp_cnt,
        '止损次数': sl_cnt,
        '总盈亏(USDT)': round(total_pnl, 2),
    })
    print(f"  {combo_name}: 收益{results['total_return']:.2f}% 交易{len(pairs)} 止盈{tp_cnt} 止损{sl_cnt}")

    # 按周统计（周末不清仓，仅按卖出时间所在周汇总盈亏）
    if not pairs.empty:
        pairs['周起始'] = pairs['卖出时间'].apply(week_start)
        weekly = pairs.groupby('周起始').agg(
            周交易数=('盈亏(USDT)', 'count'),
            周盈亏USDT=('盈亏(USDT)', 'sum'),
            盈利笔数=('盈亏(USDT)', lambda x: int((x > 0).sum())),
            亏损笔数=('盈亏(USDT)', lambda x: int((x < 0).sum())),
        ).reset_index()
        weekly['策略组合'] = combo_name
        weekly['周起始'] = weekly['周起始'].dt.strftime('%Y-%m-%d')
        weekly['周盈亏USDT'] = weekly['周盈亏USDT'].round(2)
        weekly_all.append(weekly)

    pairs['策略组合'] = combo_name
    detail_all.append(pairs)

# 汇总排序
summary_df = pd.DataFrame(summary).sort_values('总收益率%', ascending=False).reset_index(drop=True)
summary_df.insert(0, '排名', range(1, len(summary_df) + 1))
print("\n=== 收益率 Top 8 ===")
print(summary_df.head(8)[['排名', '策略组合', '总收益率%', '交易数', '止盈次数', '止损次数', '最大回撤%', '夏普', '胜率%']].to_string(index=False))

weekly_df = pd.concat(weekly_all, ignore_index=True) if weekly_all else pd.DataFrame()
detail_df = pd.concat(detail_all, ignore_index=True) if detail_all else pd.DataFrame()

# 输出
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "1min_strategy_test_btc")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"BTC_USDT_1m_2026H1_止损5_止盈10_周末不清仓_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

overview = pd.DataFrame({
    '项目': ['交易对', '周期', '开始', '结束', '初始资金', '手续费', '止损', '止盈', '周末清仓', 'K线数', '策略组合数'],
    '数值': [SYMBOL, TIMEFRAME, START, END, INITIAL_CAPITAL, f"{COMMISSION*100}%",
            f"{STOP_LOSS*100}%", f"{TAKE_PROFIT*100}%", '否', len(df), len(COMBOS)]
})

with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    overview.to_excel(writer, sheet_name='测试条件', index=False)
    summary_df.to_excel(writer, sheet_name='策略对比', index=False)
    weekly_df.to_excel(writer, sheet_name='按周盈亏', index=False)
    detail_df.to_excel(writer, sheet_name='交易明细', index=False)

print(f"\n✅ 已输出: {out_path}")
