# -*- coding: utf-8 -*-
import os, io, sys, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd, numpy as np

def test(label, proxy):
    PROXY = proxy
    os.environ["HTTP_PROXY"] = PROXY
    os.environ["HTTPS_PROXY"] = PROXY
    s = requests.Session()
    s.headers['User-Agent'] = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                               '(KHTML, like Gecko) Chrome/120.0 Safari/537.36')
    s.proxies = {'http': PROXY, 'https': PROXY}
    retry = Retry(total=2, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    s.mount('https://', HTTPAdapter(max_retries=retry))
    try:
        s.get('https://query1.finance.yahoo.com/v8/finance/chart/QQQ?range=1d&interval=1d', timeout=8)
    except Exception as e:
        print(f"[{label}] warmup fail: {repr(e)[:80]}")
    try:
        df = yf.download('QQQ', interval='1h', start='2025-01-01', progress=False, auto_adjust=True, session=s)
        n = 0 if df is None else len(df)
        print(f"[{label}] rows={n}")
    except Exception as e:
        print(f"[{label}] yf err: {repr(e)[:80]}")
    # 干净退出环境变量
    os.environ.pop("HTTP_PROXY", None); os.environ.pop("HTTPS_PROXY", None)

test("127.0.0.1:7892", "http://127.0.0.1:7892")
test("192.168.11.64:7892", "http://192.168.11.64:7892")
