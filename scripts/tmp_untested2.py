# -*- coding: utf-8 -*-
import sqlite3, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
c = sqlite3.connect('data/strategy_records.db')
tested = {r[0].split('/')[0].split('~')[0].upper() for r in c.execute("SELECT DISTINCT symbol FROM strategy_records")}
all_syms = set()
for line in """
RMBS, SIMO, MRVL, GLW, COHR, CIEN, ALAB, CRDO, AMD, INTC, LRCX, AMAT, TXN,
ORCL, BABA, IBM, NOW, MCHP, SMCI, GE, ASTS, SATS, VSAT, SATL, TSAT, STM, CGNX,
SERVE, GD, TDG, MPC, CNQ, VLO, WMB, KMI, TRGP, OKE, OXY, DVN, AEM, WPM, AU,
KGC, RGLD, CDE, AMGN, TMO, ABT, GILD, NVO, COST, HD, DIS, MCD, TJX
""".split(','):
    s = line.strip()
    if s and s.upper() not in tested:
        all_syms.add(s.upper())
print("唯一未测试标的数:", len(all_syms))
print(",".join(sorted(all_syms)))
