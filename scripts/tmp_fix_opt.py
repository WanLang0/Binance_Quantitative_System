# -*- coding: utf-8 -*-
"""修正: 把误归到 usdt/ 的 opt 文件按文件名内币种分发到各自目录，并删除空 usdt/"""
import os, sys, io, shutil, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res = os.path.join(ROOT, "results")

usdt_dir = os.path.join(res, "usdt")
if os.path.isdir(usdt_dir):
    for f in glob.glob(os.path.join(usdt_dir, "opt_*.xlsx")):
        name = os.path.basename(f)
        # 形如 opt_20260825_220901_BTC_USDT.xlsx -> 币种在倒数第二段
        parts = name.replace(".xlsx","").split("_")
        coin = parts[-2].lower()  # BTC/ETH/XRP/BNB
        target = os.path.join(res, coin)
        os.makedirs(target, exist_ok=True)
        dst = os.path.join(target, name)
        shutil.move(f, dst)
        print(f"  + {name} -> {coin}/", flush=True)
    if not os.listdir(usdt_dir):
        shutil.rmtree(usdt_dir)
        print("🗑 已删除空目录 usdt/", flush=True)

print("\n✅ 修正完成", flush=True)
