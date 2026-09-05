# -*- coding: utf-8 -*-
"""验证 get_positions 空单contracts归一化 + 外部平仓防抖"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# ---- 1. get_positions 归一化 ----
from futures_trader import FuturesTrader

t = FuturesTrader.__new__(FuturesTrader)  # 跳过 __init__（不建交易所）
fake = [
    # 单向模式：空单 contracts 为负
    {'symbol': 'CC/USDT:USDT', 'contracts': -2935, 'side': 'short', 'entryPrice': 0.1079,
     'contractSize': 1, 'unrealizedPnl': -1.2, 'liquidationPrice': 0.2, 'leverage': 1,
     'initialMargin': 316.67},
    # 多单正数
    {'symbol': 'BTC/USDT:USDT', 'contracts': 0.003, 'side': 'long', 'entryPrice': 79000,
     'contractSize': 1, 'unrealizedPnl': 5, 'liquidationPrice': 40000, 'leverage': 1,
     'initialMargin': 237},
    # 无仓位记录（contracts=0/None）
    {'symbol': 'ETH/USDT:USDT', 'contracts': 0, 'side': None},
    {'symbol': 'XRP/USDT:USDT', 'contracts': None, 'side': None},
    # side 缺失的负数（对冲模式边界）
    {'symbol': 'DOGE/USDT:USDT', 'contracts': -100, 'side': None, 'entryPrice': 0.1,
     'contractSize': 1, 'leverage': 1, 'initialMargin': 10},
]
t.exchange = MagicMock()
t.exchange.fetch_positions.return_value = fake
res, err = t.get_positions()
print('返回持仓数:', len(res), '(应为3)')
for p in res:
    print('  %s side=%s contracts=%s' % (p['symbol'], p['side'], p['contracts']))
ok1 = (len(res) == 3
       and res[0]['symbol'] == 'CC/USDT:USDT' and res[0]['side'] == 'short' and res[0]['contracts'] == 2935
       and res[1]['symbol'] == 'BTC/USDT:USDT' and res[1]['side'] == 'long'
       and res[2]['symbol'] == 'DOGE/USDT:USDT' and res[2]['side'] == 'short' and res[2]['contracts'] == 100)
print('归一化验证:', 'PASS' if ok1 else 'FAIL')

# ---- 2. 外部平仓防抖逻辑（直接模拟判定分支） ----
from composite_trader import CompositeTrader


def check_debounce(last_trade_time):
    """复刻 _refresh 中的防抖判定"""
    lt = (last_trade_time or {}).get('time') if isinstance(last_trade_time, dict) else (last_trade_time or {}).get('time')
    try:
        recent = bool(lt) and (datetime.now() - datetime.fromisoformat(lt)).total_seconds() < 120
    except Exception:
        recent = False
    return recent


print('\n防抖验证:')
print('  30秒前开仓 → 保持状态(防抖):', 'PASS' if check_debounce({'time': (datetime.now() - timedelta(seconds=30)).isoformat()}) else 'FAIL')
print('  3分钟前开仓 → 正常判定外部平仓:', 'PASS' if not check_debounce({'time': (datetime.now() - timedelta(minutes=3)).isoformat()}) else 'FAIL')
print('  无记录 → 正常判定:', 'PASS' if not check_debounce({}) else 'FAIL')
print('  无效时间串 → 正常判定(不误防抖):', 'PASS' if not check_debounce({'time': 'bad-string'}) else 'FAIL')
