# -*- coding: utf-8 -*-
"""实测：开仓后 _fill_info_of 的成交均价/数量 vs 交易所持仓均价 对照"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from futures_trader import FuturesTrader
from composite_trader import CompositeTrader

KEY = os.environ['TEST_KEY']
SEC = os.environ['TEST_SEC']
SYM = 'BTC/USDT'
QTY = 0.001

t = FuturesTrader(KEY, SEC, proxy=None, testnet=True, leverage=1)
t.set_leverage(1, SYM)

eng = CompositeTrader.__new__(CompositeTrader)
eng.trader = t
eng.leverage = 1

# 下单前参考价
ref_price = None
p, _ = t.get_ticker(SYM)
ref_price = p
print(f'下单前参考价(last): {ref_price}')

# 市价开空
order, err = t.place_order(SYM, 'sell', 'market', QTY)
print('开空下单:', 'OK' if order else f'FAIL {err}')
time.sleep(1)

# 修复后的取值逻辑
fill_price, filled = eng._fill_info_of(order, ref_price)
print(f'成交回报: average/price={fill_price}  filled={filled}')
print(f'ccxt order 原始: average={order.get("average")} price={order.get("price")} filled={order.get("filled")}')

# 交易所持仓真实均价
time.sleep(1)
pos, _ = t.get_positions()
me = next((x for x in pos if x['symbol_base'] == SYM), None)
if me:
    print(f'\n交易所持仓: side={me["side"]} qty={me["contracts"]} entryPrice={me["entry_price"]}')
    diff = abs(fill_price - me['entry_price'])
    print(f'均价差异: {diff:.6f} ({diff / me["entry_price"] * 100:.4f}%)')
    print('结论:', '完全一致 PASS' if diff < 0.5 else '有差异（下一轮轮询会被交易所值覆盖）')
else:
    print('未查到持仓')

# 清理
o, err = t.place_order(SYM, 'buy', 'market', QTY, reduce_only=True)
print('\n清理平空:', 'OK' if o else f'FAIL {err}')
