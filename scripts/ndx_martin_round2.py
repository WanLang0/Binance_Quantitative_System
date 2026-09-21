# -*- coding: utf-8 -*-
"""
NDX-SRM Round 2 — CE 组合验证
  ① 因子拆解: C(门) / E(确认) / E_ng(无门) / CE / CE-L —— 检验两个组件的独立增益
  ② 成本敏感度: 0/5/10/20/50 bps 单边
  ③ 参数邻域: RSI {30,32,35,38,40} x ATR间距 4档 = 20 cells, 稳定性审计
  ④ Walk-forward: 3 窗 train 选参(冻结) -> OOS 验证, 附 fixed(35,[1,2,3]) 基准

统一成本默认 10 bps/边 (因子表另给 0 bps 列以对齐 Round1)。
输出: scripts/results/ndx_martin_round2.json + 控制台表格
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ndx_martin_research import (MartinEngine, perf_metrics, add_indicators,
                                 fetch_yahoo, bh_backtest, RESULTS)

OUT = os.path.join(RESULTS, "ndx_martin_round2.json")
BASE_COST = 10  # bps 单边, 除特别注明外全部使用


def mk_cfg(**kw):
    """CE 基础配置: 金叉门 + SRM 入场 + 反转确认加仓 + 2ATR TP + -12% SL + 15日强平"""
    cfg = dict(kind="martin", trend_gate="golden", rsi_entry=35, dd_entry=.03,
               weights=[.20, .20, .25, .35], spacing="atr",
               atr_mult=[1.0, 2.0, 3.0], max_layers=4,
               tp="atr", tp_mult=2.0, sl_pct=.12, max_hold=15,
               min_gap_bars=2, reversal_confirm=True, rev_rsi=[40, 32, 28],
               cost_bps=BASE_COST, start_bar=0)
    cfg.update(kw)
    return cfg


def run_one(df, cfg, label):
    eng = MartinEngine(df, cfg)
    eq, trades = eng.run()
    m = perf_metrics(eq, trades, label=label)
    m["invested_frac_avg"] = round(float(eq["invested_frac"].mean()), 4)
    m["time_in_market_pct"] = round(float((eq["invested_frac"] > 0.01).mean()), 4)
    return m, eq, trades


def row(name, m):
    return {"strat": name, "ret": m["total_return"], "cagr": m["cagr"],
            "sharpe": m["sharpe"], "mdd": m["max_dd"],
            "uw_max_d": m["max_underwater_bars"], "uw_pct": m["pct_time_underwater"],
            "win%": m["win_rate"], "avgW": m["avg_win"], "avgL": m["avg_loss"],
            "streak": m["max_loss_streak"], "slStreak": m["max_sl_streak"],
            "slRec_d": m["avg_sl_recovery_days"],
            "n": m["closed_trades"], "in_mkt%": m["time_in_market_pct"]}


def print_table(rows, title):
    print("\n=== " + title + " ===")
    df = pd.DataFrame(rows)
    fmt = {"ret": "{:+.1%}", "cagr": "{:+.1%}", "mdd": "{:+.1%}", "uw_pct": "{:.0%}",
           "win%": "{:.0%}", "avgW": "{:+.2%}", "avgL": "{:+.2%}",
           "in_mkt%": "{:.0%}", "sharpe": "{:.2f}", "slRec_d": "{:.0f}"}
    for c, f in fmt.items():
        if c in df:
            df[c] = df[c].map(lambda v: f.format(v) if isinstance(v, (int, float)) else v)
    print(df.to_string(index=False))


def main():
    t0 = __import__("time").time()
    out = {"meta": {"cost_bps_default": BASE_COST,
                    "note": "cost=单边bps; E_ng=E去金叉门; CE=金叉门+反转确认; CE-L=CE已是15日强平, L-Off=取消强平对照"}}

    ndx_all = add_indicators(fetch_yahoo("^NDX"))
    ndx = ndx_all[ndx_all["date"] >= "2010-01-01"].reset_index(drop=True)
    print(f"NDX daily bars={len(ndx)}  {ndx['date'].iloc[0].date()} -> {ndx['date'].iloc[-1].date()}")

    # ---------------- ① 因子拆解 ----------------
    fac_cfgs = {
        "C   (gate only)":        mk_cfg(reversal_confirm=False),
        "E   (confirm only)":     mk_cfg(trend_gate="ma200"),
        "E_ng (no gate)":         mk_cfg(trend_gate=None),
        "CE  (gate+confirm)":     mk_cfg(),
        "CE-Loff (no time-stop)": mk_cfg(max_hold=None),
    }
    rows0, rows10, fac_detail = [], [], {}
    for name, cfg in fac_cfgs.items():
        c0 = dict(cfg, cost_bps=0)
        m0, _, _ = run_one(ndx, c0, name)
        m1, _, _ = run_one(ndx, cfg, name)
        rows0.append(row(name, m0))
        rows10.append(row(name, m1))
        fac_detail[name] = {"0bps": m0, "10bps": m1}
    bh_eq, bh_tr = bh_backtest(ndx)
    bh_m = perf_metrics(bh_eq, bh_tr, label="BH")
    bh_m["time_in_market_pct"] = 1.0
    rows10.insert(0, row("BH", bh_m))
    print_table(rows0, "① 因子拆解 @ 0 bps (对齐 Round1)")
    print_table(rows10, "① 因子拆解 @ 10 bps 单边")
    out["factors"] = fac_detail

    # ---------------- ② 成本敏感度 ----------------
    cost_rows = {"E": [], "CE": []}
    cost_detail = {}
    for bps in [0, 5, 10, 20, 50]:
        for tag, cfg in [("E", mk_cfg(trend_gate="ma200", cost_bps=bps)),
                         ("CE", mk_cfg(cost_bps=bps))]:
            m, _, _ = run_one(ndx, cfg, f"{tag}@{bps}bps")
            cost_rows[tag].append({"strat": f"{tag} @{bps}bps", "ret": m["total_return"],
                                   "cagr": m["cagr"], "sharpe": m["sharpe"],
                                   "mdd": m["max_dd"], "n": m["closed_trades"],
                                   "avgL": m["avg_loss"]})
            cost_detail[f"{tag}_{bps}bps"] = m
    print_table(cost_rows["E"], "② 成本敏感度 — E (RSI门版)")
    print_table(cost_rows["CE"], "② 成本敏感度 — CE")
    out["cost"] = cost_detail

    # ---------------- ③ 参数邻域 (CE, 10bps) ----------------
    rsi_levels = [30, 32, 35, 38, 40]
    spacing_levels = [[.75, 1.0, 1.25], [1.0, 2.0, 3.0], [1.5, 2.0, 2.5], [2.5, 3.0, 3.5]]
    grid = []
    grid_detail = {}
    for rs in rsi_levels:
        for sp in spacing_levels:
            cfg = mk_cfg(rsi_entry=rs, atr_mult=sp)
            m, _, _ = run_one(ndx, cfg, f"rsi{rs}_sp{sp[0]}")
            tag = f"rsi{rs}|sp{sp}"
            grid.append({"rsi": rs, "spacing": str(sp), "sharpe": m["sharpe"],
                         "cagr": m["cagr"], "mdd": m["max_dd"], "n": m["closed_trades"]})
            grid_detail[tag] = m
    g = pd.DataFrame(grid)
    n_ok = int((g["sharpe"] >= 1.0).sum())
    med = float(g["sharpe"].median())
    print("\n=== ③ 参数邻域 (CE @10bps): 20 cells ===")
    print(g.pivot(index="rsi", columns="spacing", values="sharpe").to_string())
    print(f"cells Sharpe>=1.0: {n_ok}/20   median Sharpe={med:.2f}   "
          f"min={g['sharpe'].min():.2f}  max={g['sharpe'].max():.2f}")
    print("per-rsi mean sharpe:   " + str({rs: round(float(g[g['rsi'] == rs]['sharpe'].mean()), 2) for rs in rsi_levels}))
    print("per-spacing mean sharpe: " + {str(sp): round(float(g[g['spacing'] == str(sp)]['sharpe'].mean()), 2) for sp in spacing_levels}.__str__())
    mdd_bad = int((g["mdd"] < -0.15).sum())
    print(f"cells MDD<-15%: {mdd_bad}/20")
    out["neighborhood"] = {"grid": grid, "summary": {
        "n_sharpe_ge_1": n_ok, "median_sharpe": med,
        "min_sharpe": float(g["sharpe"].min()), "max_sharpe": float(g["sharpe"].max()),
        "n_mdd_lt_15": mdd_bad}}

    # ---------------- ④ Walk-forward ----------------
    dates = ndx["date"].reset_index(drop=True)

    def idx_of(s):
        hit = dates[dates >= pd.Timestamp(s)]
        return int(hit.index[0]) if len(hit) else len(ndx) - 1

    wf_windows = [("WF1", "2010-01-01", "2017-12-31", "2018-01-01", "2020-12-31"),
                  ("WF2", "2010-01-01", "2020-12-31", "2021-01-01", "2023-12-31"),
                  ("WF3", "2010-01-01", "2023-12-31", "2024-01-01", "2026-12-31")]
    spacing_levels_ = [[.75, 1.0, 1.25], [1.0, 2.0, 3.0], [1.5, 2.0, 2.5], [2.5, 3.0, 3.5]]
    wf_rows, wf_detail = [], {}
    for tag, tr_lo, tr_hi, te_lo, te_hi in wf_windows:
        i0, i1 = idx_of(tr_lo), idx_of(tr_hi)
        j0, j1 = idx_of(te_lo), min(idx_of(te_hi), len(ndx) - 1)
        df_tr = ndx.iloc[:i1 + 1].reset_index(drop=True)

        best, best_key = None, None
        for rs in rsi_levels:
            for sp in spacing_levels_:
                m, _, _ = run_one(df_tr, mk_cfg(rsi_entry=rs, atr_mult=sp, start_bar=i0), "tr")
                key = (m["sharpe"], m["cagr"])
                if m["max_dd"] >= -0.15 and (best is None or key > best):
                    best, best_key = key, (rs, sp)
        if best_key is None:   # 全部违反 MDD 约束时退化为纯 sharpe
            for rs in rsi_levels:
                for sp in spacing_levels_:
                    m, _, _ = run_one(df_tr, mk_cfg(rsi_entry=rs, atr_mult=sp, start_bar=i0), "tr")
                    if best is None or (m["sharpe"], m["cagr"]) > best:
                        best, best_key = (m["sharpe"], m["cagr"]), (rs, sp)
        rs_sel, sp_sel = best_key

        df_te = ndx.iloc[:j1 + 1].reset_index(drop=True)
        m_fix, _, _ = run_one(df_te, mk_cfg(start_bar=j0), f"{tag} OOS fixed(35,1/2/3)")
        m_sel, _, _ = run_one(df_te, mk_cfg(rsi_entry=rs_sel, atr_mult=sp_sel, start_bar=j0),
                              f"{tag} OOS selected")
        bh_eq, bh_tr = bh_backtest(ndx.iloc[j0:j1 + 1].reset_index(drop=True))
        bh_m = perf_metrics(bh_eq, bh_tr, label=f"{tag} OOS BH")

        wf_rows.append({"wf": tag, "oos": f"{te_lo[:4]}-{te_hi[:4]}",
                        "selected": f"rsi{rs_sel}|sp{sp_sel}",
                        "fix_sharpe": m_fix["sharpe"], "fix_cagr": m_fix["cagr"],
                        "fix_mdd": m_fix["max_dd"], "fix_n": m_fix["closed_trades"],
                        "sel_sharpe": m_sel["sharpe"], "sel_cagr": m_sel["cagr"],
                        "sel_mdd": m_sel["max_dd"], "sel_n": m_sel["closed_trades"],
                        "bh_sharpe": bh_m["sharpe"], "bh_cagr": bh_m["cagr"]})
        wf_detail[tag] = {"selected_params": {"rsi": rs_sel, "spacing": sp_sel},
                          "oos_fixed": m_fix, "oos_selected": m_sel, "oos_bh": bh_m}
        print(f"[WF] {tag} train={tr_lo[:4]}-{tr_hi[:4]} selected rsi={rs_sel} sp={sp_sel}")

    wfdf = pd.DataFrame(wf_rows)
    print("\n=== ④ Walk-forward OOS (CE @10bps) ===")
    f2 = {"fix_cagr": "{:+.1%}", "sel_cagr": "{:+.1%}", "bh_cagr": "{:+.1%}",
          "fix_mdd": "{:+.1%}", "sel_mdd": "{:+.1%}",
          "fix_sharpe": "{:.2f}", "sel_sharpe": "{:.2f}", "bh_sharpe": "{:.2f}"}
    wfd = wfdf.copy()
    for c, f in f2.items():
        wfd[c] = wfd[c].map(lambda v: f.format(v))
    print(wfd.to_string(index=False))
    out["walk_forward"] = wf_detail

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f"\nsaved -> {OUT}  ({__import__('time').time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
