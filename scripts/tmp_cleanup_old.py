# -*- coding: utf-8 -*-
"""eth/ 已完整，直接删除 ETH 冗余源文件夹；并清理所有已清空的旧币种源文件夹"""
import os, sys, io, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res = os.path.join(ROOT, "results")

# ---- 1. ETH 冗余源文件夹（eth/ 已含全部内容，删除副本）----
eth_src = [
    "ETH_15m_分年度_OR_止损5_止盈5", "ETH_15m_分年度_止损5_止盈10",
    "15min_strategy_test_eth", "1min_strategy_test_eth",
    "ETH_1h_分年度_OR_止损5_止盈5", "ETH_1h_2026临时补充",
]
for name in eth_src:
    p = os.path.join(res, name)
    if os.path.isdir(p):
        shutil.rmtree(p)
        print(f"🗑 删除冗余ETH源目录: {name}", flush=True)

# ---- 2. 清理其他已清空的旧源文件夹 ----
old = [
    "15min_strategy_test_btc","1min_strategy_test_btc","15min_strategy_test_xrp",
    "15min_strategy_test_bnb","15min_strategy_test_mu","1min_strategy_test_mu",
    "1min_strategy_test_mu_tp10","15min_strategy_test_lite","15min_strategy_test_lite_tp10",
    "15min_strategy_test_liteb_tp10","1min_strategy_test_lite_tp10",
    "15min_strategy_test_muub","1min_strategy_test_muub",
    "15min_strategy_test_nvda","15min_strategy_test_nvdab",
    "15min_strategy_test_sndkb","15min_strategy_test_spcxb",
    "5min_strategy_test","combo_optimize_2026H1",
]
for name in old:
    p = os.path.join(res, name)
    if os.path.isdir(p):
        remaining = [f for f in os.listdir(p) if not f.startswith(".")]
        if not remaining:
            shutil.rmtree(p); print(f"🗑 删除空目录 {name}", flush=True)
        else:
            print(f"⚠ 非空保留: {name} -> {remaining}", flush=True)

print("\n✅ 清理完成", flush=True)
