# -*- coding: utf-8 -*-
"""验证 futures_trader 持仓模式适配（无需网络/密钥）：
1) 单向模式：开仓不带参数、平仓带 reduceOnly（原行为不变）
2) 双向模式：开/平仓均带正确 positionSide 且不传 reduceOnly
3) 模式探测失败但账户实为双向：首单 -4061 → 自动翻转缓存重试成功
4) 止损保护单 place_stop_order 同样适配
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from futures_trader import FuturesTrader


class FakeEx:
    def __init__(self, dual, query_ok=True, reject_no_ps=False):
        self.dual, self.query_ok, self.reject_no_ps = dual, query_ok, reject_no_ps
        self.calls = []

    def fapi_private_v2_get_position_side_dual(self):
        if not self.query_ok:
            raise RuntimeError('network down')
        return {'dualSidePosition': self.dual}

    def price_to_precision(self, symbol, price):
        return str(price)

    def create_order(self, symbol, type_, side, qty, price, params):
        self.calls.append((side, dict(params)))
        if self.reject_no_ps and 'positionSide' not in params:
            raise Exception('binanceusdm {"code":-4061,"msg":"Order\'s position side does not match user\'s setting."}')
        return {'id': '1', 'average': 100.0, 'filled': qty}


def check(name, cond, detail=''):
    print(('PASS' if cond else 'FAIL'), name, detail)
    if not cond:
        sys.exit(1)


t = FuturesTrader(api_key='k', api_secret='s')

# 1) 单向模式
t.exchange = FakeEx(dual=False)
_, e = t.place_order('BTC/USDT', 'buy', 'market', 1); check('单向开多', e is None and t.exchange.calls[-1][1] == {})
_, e = t.place_order('BTC/USDT', 'sell', 'market', 1, reduce_only=True)
check('单向平多(reduceOnly)', e is None and t.exchange.calls[-1][1] == {'reduceOnly': True})
_, e = t.place_stop_order('BTC/USDT', 'sell', 95, 1)
check('单向止损单', e is None and t.exchange.calls[-1][1].get('reduceOnly') is True
      and t.exchange.calls[-1][1]['type'] == 'STOP_MARKET')

# 2) 双向模式（探测成功）
t2 = FuturesTrader(api_key='k', api_secret='s')
t2.exchange = FakeEx(dual=True)
_, e = t2.place_order('BTC/USDT', 'buy', 'market', 1)
check('双向开多(LONG)', e is None and t2.exchange.calls[-1][1] == {'positionSide': 'LONG'})
_, e = t2.place_order('BTC/USDT', 'sell', 'market', 1)
check('双向开空(SHORT)', e is None and t2.exchange.calls[-1][1] == {'positionSide': 'SHORT'})
_, e = t2.place_order('BTC/USDT', 'sell', 'market', 1, reduce_only=True)
check('双向平多(sell+LONG)', e is None and t2.exchange.calls[-1][1] == {'positionSide': 'LONG'})
_, e = t2.place_order('BTC/USDT', 'buy', 'market', 1, reduce_only=True)
check('双向平空(buy+SHORT)', e is None and t2.exchange.calls[-1][1] == {'positionSide': 'SHORT'})
_, e = t2.place_stop_order('BTC/USDT', 'buy', 105, 1)
check('双向止损平空(buy+SHORT)', e is None and t2.exchange.calls[-1][1].get('positionSide') == 'SHORT'
      and 'reduceOnly' not in t2.exchange.calls[-1][1])

# 3) 探测失败 + 账户实为双向 → -4061 自愈
t3 = FuturesTrader(api_key='k', api_secret='s')
t3.exchange = FakeEx(dual=True, query_ok=False, reject_no_ps=True)
_, e = t3.place_order('BTC/USDT', 'buy', 'market', 1)
check('-4061自愈重试', e is None and t3.exchange.calls[-1][1] == {'positionSide': 'LONG'},
      f'err={e}')
_, e = t3.place_order('BTC/USDT', 'sell', 'market', 1, reduce_only=True)
check('自愈后缓存生效', e is None and t3.exchange.calls[-1][1] == {'positionSide': 'LONG'})

print('\nall passed')
