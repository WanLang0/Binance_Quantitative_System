# -*- coding: utf-8 -*-
"""交易系统一致性验证：
  1) 手动裸算KDJ信号（ta库独立实现）vs 系统信号（TechnicalIndicators+BacktestEngine OR）逐根对比
  2) get_ohlcv 丢弃未收盘K线单测（mock fetch_ohlcv）
  3) auto_futures._apply_orders 信号→下单状态机（mock trader 记录调用序列）
"""
import os, sys, io, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7892")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7892")

import pandas as pd
import numpy as np

def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'}  {name}")
    assert cond, name

# ============ 1) 手动 vs 系统：ETH 4h KDJ 信号逐根对比 ============
from data_fetcher import BinanceDataFetcher
from indicators import TechnicalIndicators
from backtest_engine import BacktestEngine
import ta.trend

fetcher = BinanceDataFetcher(); fetcher.set_market_type('spot')
df = None
for i in range(4):
    df = fetcher.fetch_historical_data("ETH/USDT", "2025-01-01", "2025-12-31", "4h")
    if df is not None and len(df) > 500:
        break
    time.sleep(3)
check(f'拉取ETH 4h数据({len(df) if df is not None else 0}根)', df is not None and len(df) > 500)

ip = {"kdj": True, "kdj_k_period": 9, "kdj_d_period": 3, "kdj_j_period": 3,
      "kdj_buy_threshold": 20, "kdj_sell_threshold": 80}

# 系统路径
dft = TechnicalIndicators.calculate_all_indicators(df, ip)
sys_signals = BacktestEngine(timeframe='4h', signal_mode='or').calculate_signals(dft, ip)

# 手动路径：独立用 ta 库重算 K/D（不经过我们的任何模块），按系统真实规则：
#   买 = K上穿D 且 K<20；卖 = K下穿D 且 K>80（不是J值阈值！）
from ta.momentum import StochasticOscillator
so = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=9, smooth_window=3, fillna=False)
k = so.stoch()
d_ = so.stoch_signal()
manual = pd.Series(0, index=df.index)
for i in range(1, len(k)):
    if pd.notna(k.iloc[i - 1]) and pd.notna(k.iloc[i]) and pd.notna(d_.iloc[i - 1]) and pd.notna(d_.iloc[i]):
        if k.iloc[i - 1] <= d_.iloc[i - 1] and k.iloc[i] > d_.iloc[i] and k.iloc[i] < 20:
            manual.iloc[i] = 1
        elif k.iloc[i - 1] >= d_.iloc[i - 1] and k.iloc[i] < d_.iloc[i] and k.iloc[i] > 80:
            manual.iloc[i] = -1

same = (sys_signals == manual).all()
diff_idx = [i for i in range(len(sys_signals)) if sys_signals.iloc[i] != manual.iloc[i]]
check(f'系统信号与手动裸算逐根一致（{int((sys_signals != 0).sum())}个信号，差异{len(diff_idx)}处）', bool(same))
if not same:
    print('  差异位置:', df.index[diff_idx[:5]])

# ============ 2) get_ohlcv 丢弃未收盘K线 ============
from demo_trader import DemoTrader

class MockTrader(DemoTrader):
    def __init__(self, fake_candles):
        self.fake = fake_candles
    def _noop(self, *a, **k):
        pass

now_ms = int(time.time() * 1000)
tf = 4 * 3600 * 1000
closed_ts = (now_ms // tf) * tf - tf          # 上一根已收盘4hK线起点
open_ts = (now_ms // tf) * tf                 # 当前进行中4hK线起点
mk = lambda ts: [ts, 100, 110, 95, 105, 10]

t = MockTrader([mk(closed_ts - tf), mk(closed_ts), mk(open_ts)])
t.exchange = type('E', (), {'fetch_ohlcv': staticmethod(lambda *a, **k: t.fake)})()
candles, err = t.get_ohlcv('ETH/USDT', '4h', limit=10)
check('未收盘尾根被丢弃', err is None and len(candles) == 2 and candles[-1]['ts'] == closed_ts)

t2 = MockTrader([mk(closed_ts - 2 * tf), mk(closed_ts - tf), mk(closed_ts)])
t2.exchange = type('E', (), {'fetch_ohlcv': staticmethod(lambda *a, **k: t2.fake)})()
c2, e2 = t2.get_ohlcv('ETH/USDT', '4h', limit=10)
check('全收盘K线不误删', e2 is None and len(c2) == 3)

from futures_trader import FuturesTrader
class MockFut(FuturesTrader):
    def __init__(self, fake_candles):
        self.fake = fake_candles
tf3 = MockFut([mk(closed_ts), mk(open_ts)])
tf3.exchange = type('E', (), {'fetch_ohlcv': staticmethod(lambda *a, **k: tf3.fake)})()
c3, e3 = tf3.get_ohlcv('ETH/USDT', '4h', limit=10)
check('合约侧同样丢弃未收盘', e3 is None and len(c3) == 1)

# ============ 3) 信号→下单状态机（auto_futures._apply_orders） ============
from auto_futures import AutoFutures

calls = []
class RecTrader:
    testnet = True
    def round_amount(self, sym, amt):
        return round(amt, 3)
    def get_ticker(self, sym):
        return 2000.0, None
    def cancel_all_orders(self, sym):
        return None, None
    def place_order(self, sym, side, type_, qty, params=None, reduce_only=False):
        calls.append(('open' if not reduce_only else 'reduce', side, qty))
        return {'id': 'x1'}, None
    def close_position(self, sym, side, qty, price=None):
        calls.append(('close', side, qty))
        return {'id': 'x2'}, None

eng = AutoFutures.__new__(AutoFutures)   # 跳过 __init__（不连网）
from datetime import datetime as _dt
import threading
eng._lock = threading.Lock()
eng.status = {'side': 'none', 'position': 0, 'buy_count': 0, 'sell_count': 0, 'signal': 0,
              'last_error': '', 'grid': {'levels': [], 'filled': 0}, 'log': []}
eng.leverage = 1
eng.trader = RecTrader()
eng._log = lambda msg: None
eng.save_state = lambda: None
eng._refresh_last_price = lambda sym: 2000.0
eng._place_stop_protection = lambda *a, **k: calls.append(('stop_order', a[2], a[3]))
eng._task_id = None

def set_sig(v):
    eng.status['signal'] = v

# 序列: 1,1,-1,-1,1 → 开多、平多、开空、平空(不反手开多)…第5个1时若持空只平空
# 实际链路: 1开多 / 1持续 / -1平多 / -1开空 / 1平空(仅平不反手)
seq = [1, 1, -1, -1, 1]
for s in seq:
    set_sig(s)
    eng._apply_orders('ETH/USDT', 1000)
opens_buy = [c for c in calls if c[0] == 'open' and c[1] == 'buy']
opens_sell = [c for c in calls if c[0] == 'open' and c[1] == 'sell']
reduces = [c for c in calls if c[0] == 'reduce']
stops = [c for c in calls if c[0] == 'stop_order']
check('双向: 开多1次+开空1次', len(opens_buy) == 1 and len(opens_sell) == 1)
check('双向: 平多1次+平空1次(reduceOnly)', len(reduces) == 2)
check('双向: 平空不反手开多(1信号持空只平)', len(opens_buy) == 1)
check('每次开仓挂保护单(2次开仓2单)', len(stops) == 2)

# 仅做多模式: 1,1,-1,-1 → 开多1次, 平多1次, 不开空
calls.clear()
eng.status.update({'side': 'none', 'position': 0, 'long_only': True})
for s in [1, 1, -1, -1]:
    set_sig(s)
    eng._apply_orders('ETH/USDT', 1000)
ob = [c for c in calls if c[0] == 'open' and c[1] == 'buy']
os_ = [c for c in calls if c[0] == 'open' and c[1] == 'sell']
rd = [c for c in calls if c[0] == 'reduce']
check('仅做多: 卖出信号只平多不开空', len(ob) == 1 and len(os_) == 0 and len(rd) == 1)

# 空头: -1(开空) → 1(平空)
calls.clear()
eng.status.update({'side': 'none', 'position': 0, 'long_only': False})
for s in [-1, 1]:
    set_sig(s)
    eng._apply_orders('ETH/USDT', 1000)
os2 = [c for c in calls if c[0] == 'open' and c[1] == 'sell']
rd2 = [c for c in calls if c[0] == 'reduce']
check('双向: 开空1次后平空1次', len(os2) == 1 and len(rd2) == 1)

print('\n全部通过')
