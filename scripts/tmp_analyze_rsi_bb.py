# -*- coding: utf-8 -*-
"""分析 RSI+布林带(OR/AND) 在不同周期/时段/止盈止损下的表现，归纳适合的行情"""
import os, sys, io, glob, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

results_dir = os.path.join(ROOT, "results")

# 收集所有含"全时段汇总"或分年度的 xlsx
candidates = glob.glob(os.path.join(results_dir, "**", "*汇总*.xlsx"), recursive=True) + \
             glob.glob(os.path.join(results_dir, "**", "*分年度*.xlsx"), recursive=True) + \
             glob.glob(os.path.join(results_dir, "**", "*8月*.xlsx"), recursive=True)

target_col = "策略名"
records = []
def norm(x):
    s = str(x).replace(" ", "")
    if "布林" in s:
        s = s.replace("布林", "BOLL")
    return s

for path in sorted(candidates):
    try:
        xl = pd.ExcelFile(path)
        # 找到策略对比 sheet（可能名字不同，取第一个含策略名列的 sheet）
        sheet = None
        for s in xl.sheet_names:
            try:
                df = xl.parse(s)
                if "策略名" in str(list(df.columns)) or "策略组合" in str(list(df.columns)):
                    sheet = s; break
            except Exception:
                continue
        if sheet is None:
            continue
        df = xl.parse(sheet)
        if "策略名" not in df.columns:
            # 尝试策略组合列
            if "策略组合" in df.columns:
                df = df.rename(columns={"策略组合": "策略名"})
            else:
                continue
        # 提取含 RSI 和 BOLL 的纯 RSI+布林带（2策略）组合
        for _, row in df.iterrows():
            name = norm(row.get("策略名", ""))
            n_strat = row.get("策略数", None)
            if "RSI" in name and "BOLL" in name:
                if n_strat is not None and str(n_strat).strip() not in ("", "nan"):
                    try:
                        if int(n_strat) != 2:
                            continue
                    except Exception:
                        pass
                rec = {
                    "file": os.path.basename(path),
                    "时段": row.get("时段", ""),
                    "策略": row.get("策略名", ""),
                    "总收益率%": row.get("总收益率%", None),
                    "整段最大回撤%": row.get("整段最大回撤%", None),
                    "周均盈利%": row.get("周均盈利%", None),
                    "交易数": row.get("交易数", None),
                    "止盈次数": row.get("止盈次数", None),
                    "止损次数": row.get("止损次数", None),
                    "胜率%": row.get("胜率%", None),
                    "夏普": row.get("夏普", None),
                }
                records.append(rec)
    except Exception as e:
        print(f"[err] {os.path.basename(path)}: {e}")

print(f"共找到 {len(records)} 条 RSI+布林带(2策略) 记录\n")
# 按文件分组打印
by_file = {}
for r in records:
    by_file.setdefault(r["file"], []).append(r)

for f in sorted(by_file):
    print("="*70)
    print(f"## {f}")
    for r in by_file[f]:
        print(f"  {r['时段']:<12} 收益={r['总收益率%']:>7}%  回撤={r['整段最大回撤%']:>7}%  周均={r['周均盈利%']:>6}%  交易={r['交易数']:>5}  止盈={r['止盈次数']}/止损={r['止损次数']}  胜率={r['胜率%']:>6}%")

# 保存摘要
with open(os.path.join(ROOT, "results", "RSI_BB_行情分析摘要.json"), "w", encoding="utf-8") as fp:
    json.dump(records, fp, ensure_ascii=False, indent=2)
print(f"\n摘要已保存 results/RSI_BB_行情分析摘要.json")
