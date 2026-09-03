# -*- coding: utf-8 -*-
import sqlite3, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
c = sqlite3.connect('data/strategy_records.db')
c.row_factory = sqlite3.Row
new = [r[0].split('/')[0].split('~')[0].upper() for r in c.execute("SELECT DISTINCT symbol FROM strategy_records")]
targets = 'ABT AEM ALAB AMAT AMD AMGN ASTS AU BABA CDE CGNX CIEN CNQ COHR COST CRDO DIS DVN GD GE GILD GLW HD IBM INTC KGC KMI LRCX MCD MCHP MPC MRVL NOW NVO OKE ORCL OXY RGLD RMBS SATL SATS SERVE SIMO SMCI STM TDG TJX TMO TRGP TSAT TXN VLO VSAT WMB WPM'.split()
print("本轮应测 53 只(剔除SATS/SERVE)，\n数据库去重标的总数:", len(new))
print("\n各标的入库详情:")
for s in sorted(targets):
    if s in ('SATS','SERVE'):
        continue
    row = c.execute("SELECT strategy,mode,tpsl,ret,trades,winrate,mdd,sharpe,source FROM strategy_records WHERE symbol=? ORDER BY id DESC LIMIT 1",(s,)).fetchone()
    if row:
        print(f"  {s:<5} {row['strategy']:<8} {row['mode']:<6} {row['tpsl']:<5} {row['ret']:>8} 笔{row['trades']:>4} 胜率{row['winrate']:>6} 回撤{row['mdd']:>7} 夏普{row['sharpe']}")
    else:
        print(f"  {s:<5} ❌ 未入库")
