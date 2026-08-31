# -*- coding: utf-8 -*-
"""补充入库：美股真实股票主网合约双向（做多+做空）最佳记录

背景：用户确认这些标的（MU/QQQ/AAPL/NVDA）主网均有对应 USDT 永续合约，
实盘可通过合约真正做空，因此双向模拟结果可作为正式参考入库。
本脚本从已保存的扫描结果中取每个标的的【双向(模拟)】最高记录，标注合约口径入库。
"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import strategies_store as store

CHANNEL = "Yahoo Finance 美股1h行情"
# 主网合约符号（USDT永续），各标的映射
SYM = {'MU': 'MU/USDT:USDT', 'QQQ': 'QQQ/USDT:USDT', 'AAPL': 'AAPL/USDT:USDT', 'NVDA': 'NVDA/USDT:USDT'}
NAME = {'MU': '美光', 'QQQ': '纳指100', 'AAPL': '苹果', 'NVDA': '英伟达'}
MARKET = '主网合约(美股真实股票行情)'

store.init_tables()

# 读取两批结果
with open('scripts/results/mu_yahoo_1h_2025.json', encoding='utf-8') as f:
    MU = json.load(f)
with open('scripts/results/us_stocks_yahoo_1h_2025.json', encoding='utf-8') as f:
    US = json.load(f)

# 汇总所有标的结果
all_res = {}
for o in MU['results']:
    all_res.setdefault('MU', []).append(o)
for t in ['QQQ', 'AAPL', 'NVDA']:
    all_res.setdefault(t, []).extend(US[t])

for ticker, out in all_res.items():
    # 取双向(模拟)模式中收益最高者
    long_short = [o for o in out if o['mode'] == '双向(模拟)']
    if not long_short:
        continue
    long_short.sort(key=lambda x: -x['ret'])
    top = long_short[0]
    n = top['n']
    wins = sum(1 for t in top['trades'] if t['ret'] > 0)
    winrate = f"{wins / len(top['trades']) * 100:.1f}%" if top['trades'] else '—'
    sym = SYM[ticker]
    days = top['days']
    per = top['period']
    buyhold = None
    if ticker == 'MU':
        buyhold = MU['buyhold']
    note = (f"{CHANNEL}，{NAME[ticker]}美股真实股票，{days}天{top['n']}笔；"
            f"主网USDT永续合约实盘可双向做空（做多+做空）；"
            + (f"标的自身买入持有{buyhold:+.1f}%；" if buyhold is not None else "")
            + f"双向做空贡献主要利润或规避下行，仅做多版见同源记录")
    rec = {'symbol': sym, 'timeframe': '1h', 'strategy': top['strat'],
           'mode': '双向', 'tpsl': top['tpsl'],
           'ret': f"{top['ret']:+.1f}%", 'daily': f"{top['ret'] / max(top['days'], 1):+.2f}%",
           'trades': str(top['n']), 'winrate': winrate, 'mdd': f"{top['mdd']:.1f}%",
           'stability': f"{top['days']}天{top['n']}笔",
           'market': MARKET,
           'source': f"{CHANNEL}（{per}）合约双向",
           'note': note,
           'sharpe': None if top['sh'] is None else f"{top['sh']:.2f}",
           'period': per}
    ok, msg = store.add_history(rec)
    print(f"入库({ticker} 双向): {top['strat']} {top['tpsl']} {top['ret']:+.1f}% → {msg}")
    if ok:
        with store._conn() as c:
            row = c.execute("SELECT id FROM strategy_records WHERE symbol=? AND strategy=? AND mode=? "
                            "AND tpsl=? AND source=?",
                            (sym, top['strat'], top['mode'], top['tpsl'], rec['source'])).fetchone()
            if row:
                c.execute("UPDATE strategy_records SET trades_detail=? WHERE id=?",
                          (json.dumps({'trades': top['trades'], 'n': len(top['trades']),
                                       'ret_total': round(top['ret'], 1)}, ensure_ascii=False), row[0]))
print("\n完成")
