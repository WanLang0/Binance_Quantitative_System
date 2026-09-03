# -*- coding: utf-8 -*-
import sqlite3, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
c = sqlite3.connect('data/strategy_records.db')
c.row_factory = sqlite3.Row
rows = c.execute("SELECT DISTINCT symbol FROM strategy_records ORDER BY symbol").fetchall()
syms = sorted({r['symbol'].split('/')[0].split('~')[0].upper() for r in rows})
print("共 %d 个去重标的:" % len(syms))
print(",".join(syms))
