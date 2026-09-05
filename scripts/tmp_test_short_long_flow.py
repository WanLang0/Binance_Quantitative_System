# -*- coding: utf-8 -*-
"""测试网多空单全流程实测 v2：
开空 → 按引擎真实匹配逻辑(symbol_base双键)验证 → 平空 → 开多 → 验证 → 平多
"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from futures_trader import FuturesTrader

KEY = os.environ['TEST_KEY']
SEC = os.environ['TEST_SEC']
SYM = 'BTC/USDT'          # 引擎配置格式（不带 :USDT 后缀，复现真实场景）
QTY = 0.001

t = FuturesTrader(KEY, SEC, proxy=None, testnet=True, leverage=1)
ok, err = t.set_leverage(1, SYM)
print('set_leverage:', ok, err or '')


def engine_lookup(pos_list, sym):
    """复刻 composite_trader._refresh_positions_and_balance 的双键匹配"""
    pos_by_symbol = {}
    for p in pos_list:
        pos_by_symbol[p.get('symbol')] = p
        base = p.get('symbol_base') or (p.get('symbol') or '').rsplit(':', 1)[0]
        if base and base != p.get('symbol'):
            pos_by_symbol.setdefault(base, p)
    return pos_by_symbol.get(sym)


def show(tag):
    pos_list, err = t.get_positions()
    me = engine_lookup(pos_list, SYM)
    print(f'\n[{tag}] err={err}')
    if me:
        print(f"  引擎匹配到: side={me['side']} contracts={me['contracts']} entry={me['entry_price']}")
    else:
        print('  引擎匹配: 无持仓')
    return me


results = []

o, err = t.place_order(SYM, 'sell', 'market', QTY)
print('\n开空下单:', 'OK' if o else f'FAIL {err}')
time.sleep(2)
me = show('开空后')
results.append(('空单识别(side=short)', me is not None and me['side'] == 'short' and me['contracts'] > 0))

o, err = t.place_order(SYM, 'buy', 'market', QTY, reduce_only=True)
print('\n平空下单:', 'OK' if o else f'FAIL {err}')
time.sleep(2)
me = show('平空后')
results.append(('空仓清零', me is None))

o, err = t.place_order(SYM, 'buy', 'market', QTY)
print('\n开多下单:', 'OK' if o else f'FAIL {err}')
time.sleep(2)
me = show('开多后')
results.append(('多单识别(side=long)', me is not None and me['side'] == 'long' and me['contracts'] > 0))

o, err = t.place_order(SYM, 'sell', 'market', QTY, reduce_only=True)
print('\n平多下单:', 'OK' if o else f'FAIL {err}')
time.sleep(2)
me = show('平多后')
results.append(('多仓清零', me is None))

bal, _ = t.get_balance()
usdt = next((b for b in bal if b['asset'] == 'USDT'), None)
print('\n' + '=' * 50)
for name, okc in results:
    print(f"  {name}: {'PASS' if okc else 'FAIL'}")
print('测试后USDT余额:', usdt['free'] if usdt else '?')
print('全部通过' if all(x[1] for x in results) else '存在失败项')
