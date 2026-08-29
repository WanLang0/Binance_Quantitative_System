# -*- coding: utf-8 -*-
"""按币种归纳 results/ 下所有测试数据，仿照 eth 目录结构。
基准路径: 项目根目录（脚本上一级），而非脚本所在目录。"""
import os, sys, io, shutil, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
res = os.path.join(ROOT, "results")

def mk(*p):
    d = os.path.join(res, *p)
    os.makedirs(d, exist_ok=True)
    return d

def move_folder_contents(src_pattern, target_dir):
    folders = glob.glob(os.path.join(res, src_pattern))
    count = 0
    for folder in folders:
        os.makedirs(target_dir, exist_ok=True)
        for f in glob.glob(os.path.join(folder, "*.xlsx")):
            dst = os.path.join(target_dir, os.path.basename(f))
            if os.path.abspath(dst) != os.path.abspath(f):
                shutil.move(f, dst)
                count += 1
                print(f"  + {os.path.basename(f)}  (来自 {os.path.basename(folder)})", flush=True)
    if count == 0:
        print(f"  (无文件，源: {src_pattern})", flush=True)
    return count

moves = [
    ("15min_strategy_test_btc",        mk("btc", "15m")),
    ("1min_strategy_test_btc",         mk("btc", "1m")),
    ("15min_strategy_test_xrp",        mk("xrp", "15m")),
    ("15min_strategy_test_bnb",        mk("bnb", "15m")),
    ("15min_strategy_test_mu",         mk("mu", "15m")),
    ("1min_strategy_test_mu",          mk("mu", "1m")),
    ("1min_strategy_test_mu_tp10",     mk("mu", "1m")),
    ("15min_strategy_test_lite",       mk("lite", "15m")),
    ("15min_strategy_test_lite_tp10",  mk("lite", "15m")),
    ("15min_strategy_test_liteb_tp10", mk("lite", "15m")),
    ("1min_strategy_test_lite_tp10",   mk("lite", "1m")),
    ("15min_strategy_test_muub",       mk("muub", "15m")),
    ("1min_strategy_test_muub",        mk("muub", "1m")),
    ("15min_strategy_test_nvda",       mk("美股代币", "15m")),
    ("15min_strategy_test_nvdab",      mk("美股代币", "15m")),
    ("15min_strategy_test_sndkb",      mk("美股代币", "15m")),
    ("15min_strategy_test_spcxb",      mk("美股代币", "15m")),
]

for pat, target in moves:
    print(f"== {pat} ==", flush=True)
    move_folder_contents(pat, target)

# 优化结果按币种拆分 + 汇总到共享
print("== combo_optimize 优化结果 ==", flush=True)
opt_dir = os.path.join(res, "combo_optimize_2026H1")
if os.path.isdir(opt_dir):
    for f in glob.glob(os.path.join(opt_dir, "opt_*.xlsx")):
        name = os.path.basename(f)
        coin = name.split("_")[-1].replace(".xlsx","").lower()
        target = mk(coin)
        dst = os.path.join(target, name)
        if os.path.abspath(dst) != os.path.abspath(f):
            shutil.move(f, dst)
            print(f"  + {name} -> {coin}/", flush=True)
    for f in glob.glob(os.path.join(opt_dir, "*.xlsx")):
        if "opt_" not in os.path.basename(f):
            target = mk("共享_全局")
            dst = os.path.join(target, os.path.basename(f))
            if os.path.abspath(dst) != os.path.abspath(f):
                shutil.move(f, dst)
                print(f"  + {os.path.basename(f)} -> 共享_全局/", flush=True)

# 全局汇总
print("== 共享全局 ==", flush=True)
shared_dir = mk("共享_全局")
for name in ["各行情最佳策略(不清仓).xlsx", "新币种最优策略汇总.xlsx"]:
    src = os.path.join(res, name)
    if os.path.exists(src):
        shutil.move(src, os.path.join(shared_dir, name))
        print(f"  + {name}", flush=True)
for f in glob.glob(os.path.join(res, "5min_strategy_test", "*.xlsx")):
    shutil.move(f, os.path.join(mk("共享_全局", "5m"), os.path.basename(f)))
    print(f"  + 5m/{os.path.basename(f)}", flush=True)

print("\n✅ 移动完成", flush=True)
