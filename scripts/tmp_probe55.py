# -*- coding: utf-8 -*-
import os, io, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7892'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7892'
import yfinance as yf
from datetime import datetime
syms = 'ABT AEM ALAB AMAT AMD AMGN ASTS AU BABA CDE CGNX CIEN CNQ COHR COST CRDO DIS DVN GD GE GILD GLW HD IBM INTC KGC KMI LRCX MCD MCHP MPC MRVL NOW NVO OKE ORCL OXY RGLD RMBS SATL SATS SERVE SIMO SMCI STM TDG TJX TMO TRGP TSAT TXN VLO VSAT WMB WPM'.split()
fail = []
for s in syms:
    try:
        df = yf.download(s, interval='1h', start='2025-01-01', end=datetime.now(), progress=False, auto_adjust=True)
        ok = df is not None and len(df) >= 200
        print(('OK  ' if ok else 'FAIL') + s, len(df) if df is not None else 0)
        if not ok:
            fail.append(s)
    except Exception as e:
        print('ERR', s, e); fail.append(s)
print('FAIL:', fail)
