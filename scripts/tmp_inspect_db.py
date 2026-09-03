# -*- coding: utf-8 -*-
import sqlite3
con = sqlite3.connect('data/strategy_records.db')
cur = con.cursor()
cur.execute("SELECT DISTINCT symbol FROM strategy_records WHERE market LIKE '%美股%' OR source LIKE '%Yahoo%' OR source LIKE '%QQQ%'")
rows = sorted(r[0] for r in cur.fetchall())
def is_us(s):
    if '.KS' in s:  return False   # 韩股
    if '/USDT' in s: return False  # 合约存根
    return True
clean = [s for s in rows if is_us(s)]
print("美股/Yahoo 总符号 =", len(rows))
print("剔除后纯净美股 ticker 数 =", len(clean))
print(clean)
