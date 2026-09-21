# -*- coding: utf-8 -*-
"""读取 ndx_martin_research.json, 输出净值曲线对比图 + 完整指标表"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
with open(os.path.join(RES, "ndx_martin_research.json"), encoding="utf-8") as f:
    R = json.load(f)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------- equity chart (NDX daily) ----------------
det = json.load(open(os.path.join(RES, "ndx_martin_research.json"), encoding="utf-8"))
# equity curves are under details_daily -> {name: {metrics, trades, equity}}
# 注意: details_daily 里 equity 在 run_all 时已写入
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1]})
colors = {"BH": "#999999", "DCA": "#2ca02c", "M0": "#d62728", "A": "#1f77b4",
          "A_L": "#9467bd", "B": "#8c564b", "C": "#ff7f0e", "D": "#17becf",
          "E": "#e377c2"}
main = ["BH", "DCA", "M0", "A", "C", "E"]          # 图上保留 6 条, 避免过挤
for name in main:
    eq = pd.DataFrame(det["details_daily"][name]["equity"])
    eq["date"] = pd.to_datetime(eq["date"])
    axes[0].plot(eq["date"], eq["equity"], label=name, lw=1.6 if name == "E" else 1.1,
                 color=colors.get(name))
axes[0].set_yscale("log")
axes[0].set_title("NDX Daily 2010-2026: Buy&Hold / DCA / Martin variants (log)")
axes[0].legend(ncol=6, fontsize=9)
axes[0].grid(alpha=0.3)

eqe = pd.DataFrame(det["details_daily"]["E"]["equity"])
eqe["date"] = pd.to_datetime(eqe["date"])
axes[1].fill_between(eqe["date"], eqe["invested_frac"], 0, color="#e377c2", alpha=0.6)
axes[1].set_title("E (Reversal Martin) invested fraction of equity")
axes[1].grid(alpha=0.3)

out = os.path.join(RES, "ndx_martin_equity.png")
fig.tight_layout()
fig.savefig(out, dpi=130)
print("chart ->", out)

# ---------------- tables ----------------
COLS = ["total_return", "cagr", "sharpe", "max_dd", "mdd_duration_years",
        "win_rate", "avg_win", "avg_loss", "trades", "avg_hold_days",
        "max_hold_days", "max_layers_used", "time_in_market_pct"]
HEADER = ["ret", "cagr", "sharpe", "mdd", "mdd_yrs", "win%", "avgW", "avgL",
          "n", "hold_avg", "hold_max", "layers", "in_mkt%"]


def show(section_dict, title):
    print("\n=== " + title + " ===")
    rows = []
    for name, m in section_dict.items():
        rows.append([name] + [m.get(c, "-") for c in COLS])
    df = pd.DataFrame(rows, columns=["strat"] + HEADER)
    fmt = {"ret": "{:+.1%}", "cagr": "{:+.1%}", "mdd": "{:+.1%}",
           "win%": "{:.0%}", "avgW": "{:+.2%}", "avgL": "{:+.2%}",
           "in_mkt%": "{:.0%}", "sharpe": "{:.2f}"}
    for c, f in fmt.items():
        if c in df:
            df[c] = df[c].map(lambda v: f.format(v) if isinstance(v, (int, float)) else v)
    print(df.to_string(index=False))


show(R["daily"], "NDX DAILY 2010-2026 (16.7y)")
show(R["stress"]["GFC_2007_2009"], "STRESS: GFC 2007-2009")
show(R["stress"]["BEAR_2022"], "STRESS: BEAR 2022")
show(R["h4"], "QQQ 4H last ~2.5y (2025-03 -> 2026-09)")
