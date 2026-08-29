# -*- coding: utf-8 -*-
"""离线验证：AutoFutures 三种模式状态字段与合约下单逻辑（不联网）"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auto_futures import AutoFutures

class MockFuturesTrader:
    """模拟合约交易器（不联网）"""
    def __init__(self):
        self._balance = [{'asset': 'USDT', 'free': 5000, 'used': 0, 'total': 5000}]
        self._positions = []
        self._price = 60000
        self._orders = []
    def is_configured(self): return True
    def get_balance(self): return self._balance, None
    def get_positions(self, symbol='BTC/USDT'):
        if not self._positions:
            return [], None
        return self._positions, None
    def get_ticker(self, symbol): return self._price, None
    def fetch_ticker(self, symbol): return self._price, None
    def get_ohlcv(self, symbol, timeframe='15m', limit=100):
        import pandas as pd, numpy as np
        n = 200
        ts = pd.date_range('2024-01-01', periods=n, freq='15min')
        close = np.linspace(100, 200, n)
        return [{'ts': int(t.timestamp()*1000), 'open': c, 'high': c+1, 'low': c-1, 'close': c, 'volume': 100} for t, c in zip(ts, close)], None
    def get_symbol_amount_precision(self, symbol): return 0
    def round_amount(self, symbol, amount): return round(amount, 3)
    def set_leverage(self, lev, symbol): return True, None
    def place_order(self, symbol, side, order_type, quantity, price=None, reduce_only=False):
        self._orders.append({'side': side, 'qty': quantity, 'reduce_only': reduce_only})
        self._positions = []
        if not reduce_only:
            pos_side = 'long' if side == 'buy' else 'short'
            self._positions = [{
                'symbol': symbol, 'side': pos_side, 'contracts': quantity,
                'entry_price': self._price, 'mark_price': self._price,
                'unrealized_pnl': 0, 'liquidation_price': 0,
                'leverage': 5, 'margin': quantity * self._price / 5,
            }]
        return {'filled': quantity}, None
    def get_open_orders(self): return [], None
    def fetch_positions(self, symbol=None): return self._positions, None

# 用 __new__ 绕过 __init__，注入 mock
af = AutoFutures.__new__(AutoFutures)
af.api_key = 'demo'
af.api_secret = 'demo'
af.leverage = 5
af.trader = MockFuturesTrader()
af._lock = __import__('threading').Lock()
af.reset_status()

# 验证状态字段
assert af.status['mode'] == 'standard'
assert af.status['leverage'] == 5
assert af.status['side'] == 'none'
assert af.status['unrealized_pnl'] == 0
assert af.status['liquidation_price'] == 0
print("OK: 合约状态字段正确")

# 验证标准模式下开多逻辑
af.status['signal'] = 1  # 做多
af.status['qty_usdt'] = 1000
af.status['interval'] = 30
res = af._apply_orders('BTC/USDT', 1000)
assert af.status['side'] == 'long'
assert af.status['position'] > 0
print(f"OK: 标准模式开多 -> side={af.status['side']}, 张数={af.status['position']}")

# 验证平多逻辑
af.status['signal'] = -1
res = af._apply_orders('BTC/USDT', 1000)
assert af.status['side'] == 'none' and af.status['position'] == 0, af.status
print("OK: 标准模式平多 -> 空仓")

# 验证开空逻辑
af.status['signal'] = -1
res = af._apply_orders('BTC/USDT', 1000)
assert af.status['side'] == 'short'
assert af.status['position'] > 0
print(f"OK: 标准模式开空 -> side={af.status['side']}, 张数={af.status['position']}")

# 验证平空逻辑
af.status['signal'] = 1
res = af._apply_orders('BTC/USDT', 1000)
assert af.status['side'] == 'none' and af.status['position'] == 0
print("OK: 标准模式平空 -> 空仓")

# 验证网格模式（建仓后止盈清空）
af.status['side'] = 'none'
af.status['position'] = 0
ok, msg = af._grid_buy('BTC/USDT', 60000, 1000, 0.01, 12)
assert ok and af.status['grid']['filled'] == 1
assert af.status['side'] == 'long'
print(f"OK: 网格建仓 -> 持仓格数={af.status['grid']['filled']}")
# 价格上涨1%触发卖出
ok, msg = af._grid_sell('BTC/USDT', 60000 * 1.01, 0.01)
assert af.status['grid']['filled'] == 0
print(f"OK: 网格止盈 -> 持仓格数={af.status['grid']['filled']}")

print("\n=== 全部合约离线测试通过 ===")
