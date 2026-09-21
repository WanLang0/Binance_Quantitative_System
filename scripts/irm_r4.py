# -*- coding: utf-8 -*-
"""
R4 — IRM-30 研究冻结前的最后四项验证
  ① QQQ 全真口径: 全历史 QQQ, 成本扫 {2,5,10,20}bps + AdjClose(TR) 口径对照
  ② Cash Yield 分解: {0,2,4,5,6}% x 7 市场 -> "交易Alpha" 与 "现金收益" 彻底拆开
  ③ 延迟成交: T+1 open / T+1 close / T+2 open (全历史, 7市场)
  ④ Muon x IRM: 日/周 Pearson + Spearman + 下行相关 + 尾部行为 + 组合(90/75/50) + NAV图
  ⑤ 逐笔审计 CSV: 每市场全量 (含每层成交与信号上下文)

固定 IRM 参数 (R3 冻结): RSI30/dd3%/sp2.5-3-3.5/TP2ATR/SL-12%/15d/20-20-25-35, 无趋势门。
10bps 默认成本 (①除外), cash 4% 默认 (②除外)。
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ndx_martin_research import (MartinEngine, perf_metrics, add_indicators,
                                 fetch_yahoo, RESULTS)

OUT = os.path.join(RESULTS, "irm_r4.json")
CSV_DIR = os.path.join(RESULTS, "irm_audit")
os.makedirs(CSV_DIR, exist_ok=True)
COST = 10
MARKETS = [("^NDX", "NDX"), ("QQQ", "QQQ"), ("^GSPC", "SPX"), ("^GDAXI", "DAX"),
           ("^STOXX50E", "SX5E"), ("^HSI", "HSI"), ("^N225", "N225")]


def base_cfg(**kw):
    cfg = dict(kind="martin", trend_gate=None, rsi_entry=30, dd_entry=.03,
               weights=[.20, .20, .25, .35], spacing="atr",
               atr_mult=[2.5, 3.0, 3.5], max_layers=4,
               tp="atr", tp_mult=2.0, sl_pct=.12, max_hold=15,
               min_gap_bars=2, reversal_confirm=True, rev_rsi=[40, 32, 28],
               cost_bps=COST, cash_rate=0.04, exec="open", start_bar=0)
    cfg.update(kw)
    return cfg


def run_one(df, cfg, label):
    eng = MartinEngine(df, cfg)
    eq, trades = eng.run()
    m = perf_metrics(eq, trades, label=label)
    m["time_in_market_pct"] = round(float((eq["invested_frac"] > 0.01).mean()), 4)
    return m, eq, trades


def fmt_row(name, m):
    return {"strat": name, "cagr": m["cagr"], "sharpe": m["sharpe"], "mdd": m["max_dd"],
            "win%": m["win_rate"], "avgW": m["avg_win"], "avgL": m["avg_loss"],
            "n": m["closed_trades"], "in_mkt%": m["time_in_market_pct"]}


def ptable(rows, title):
    print("\n=== " + title + " ===")
    df = pd.DataFrame(rows)
    fmt = {"cagr": "{:+.2%}", "mdd": "{:+.1%}", "win%": "{:.0%}", "avgW": "{:+.2%}",
           "avgL": "{:+.2%}", "in_mkt%": "{:.0%}", "sharpe": "{:.2f}"}
    for c, f in fmt.items():
        if c in df:
            df[c] = df[c].map(lambda v: f.format(v) if isinstance(v, (int, float)) else v)
    print(df.to_string(index=False))


def export_audit_csv(name, trades):
    """⑤ 逐笔审计: 每笔一行 + 层明细两表"""
    rows = []
    for t in trades:
        rows.append({
            "entry_date": t["entry_date"], "exit_date": t["exit_date"],
            "status": t["status"], "reason": t["reason"], "layers": t["layers"],
            "avg_cost": t["avg_cost"], "exit_price": t["exit_price"],
            "pnl_pct": t["pnl_pct"], "hold_days": t["bars_held"],
            "entry_rsi": t["entry_rsi"], "entry_dd20": t["entry_dd20"],
            "entry_atr_pct": t["entry_atr_pct"],
            "tp_trigger": t["tp_trigger"], "sl_trigger": t["sl_trigger"],
            "invested": t["total_invested"],
        })
    pd.DataFrame(rows).to_csv(os.path.join(CSV_DIR, f"{name}_trades.csv"),
                              index=False, encoding="utf-8-sig")
    lrows = []
    for t in trades:
        for lf in t["layer_fills"]:
            lrows.append({"entry_date": t["entry_date"], "pnl_pct": t["pnl_pct"],
                          "status": t["status"]} | lf)
    pd.DataFrame(lrows).to_csv(os.path.join(CSV_DIR, f"{name}_layers.csv"),
                               index=False, encoding="utf-8-sig")


def main():
    t0 = time.time()
    out = {"meta": {"fixed": "RSI30/dd3%/sp2.5-3-3.5/TP2ATR/SL12%/15d/20-20-25-35/no-gate",
                    "cost_bps": COST, "cash_rate_default": 4.0}}
    data = {}
    for sym, name in MARKETS:
        raw = fetch_yahoo(sym)
        data[name] = add_indicators(raw).dropna(subset=["ma200"]).reset_index(drop=True)
        export_audit_csv(name, run_one(data[name], base_cfg(), name)[2])
    print("[audit csv] ->", CSV_DIR)

    # ---------------- ① QQQ 全真口径 ----------------
    qqq = data["QQQ"]
    qrows, q_detail = [], {}
    for bps in [2, 5, 10, 20]:
        m, _, _ = run_one(qqq, base_cfg(cost_bps=bps), f"QQQ@{bps}bps")
        qrows.append(fmt_row(f"QQQ price @{bps}bps", m))
        q_detail[f"price_{bps}bps"] = m
    m, _, _ = run_one(qqq, base_cfg(cost_bps=2), "QQQ TR")
    qrows.append(fmt_row("QQQ adjClose(TR) @2bps", m))
    q_detail["tr_2bps"] = m
    ptable(qrows, "① QQQ 全历史 1999-03→: 交易口径 + TR 口径")
    out["qqq_real"] = q_detail

    # ---------------- ② Cash Yield 分解 ----------------
    cash_rows, cash_detail = [], {}
    for rate in [0, 2, 4, 5, 6]:
        row = {"strat": f"cash={rate}%"}
        for name, df in data.items():
            m, _, _ = run_one(df, base_cfg(cash_rate=rate / 100), f"{name}@cash{rate}")
            row[name] = m["cagr"]
            cash_detail[f"{name}_{rate}"] = m["cagr"]
        cash_rows.append(row)
    ptable(cash_rows, "② CAGR 分解: 交易Alpha(cash=0) + 现金收益(4-6%) x 7市场")
    out["cash_decomp_cagr"] = cash_detail

    # ---------------- ③ 延迟成交 ----------------
    lag_rows, lag_detail = [], {}
    for name, df in data.items():
        row = {"mkt": name}
        for ex, tag in [("open", "T+1open"), ("close", "T+1close"), ("open2", "T+2open")]:
            m, _, _ = run_one(df, base_cfg(exec=ex), f"{name}|{tag}")
            row[tag] = m["sharpe"]
            lag_detail[f"{name}_{tag}"] = {"sharpe": m["sharpe"], "cagr": m["cagr"],
                                           "mdd": m["max_dd"], "n": m["closed_trades"]}
        lag_rows.append(row)
    ptable(lag_rows, "③ 延迟成交 Sharpe (10bps, cash4%): 信号执行时点敏感度")
    out["exec_lag"] = lag_detail

    # ---------------- ④ Muon x IRM ----------------
    try:
        import momentum_service as ms
        assert ms.available(), "engine import failed"
        dfs, funds, ob, w1 = ms.load_data()
        extra = None
        if ob:
            from tmp_combo_full import W0 as _W0
            from tmp_r7_universe_audit import pool_at as _pool_at, build_gate as _build_gate
            dates = sorted(set().union(*[set(d.index) for d in dfs.values()]))
            grid = [d for d in dates if _W0 <= d < w1][::ms.DEFAULTS['rebal_n']]
            pools5 = {t: _pool_at(dfs, ob, t, 'U8') for t in grid}
            gate = _build_gate(dfs, pools5, ma_n=60)
            extra = (lambda t: _pool_at(dfs, ob, t, 'U8'), gate, w1)
        eq, *_rest = ms._run_backtest(dfs, funds, ms.DEFAULTS, ms.DEFAULTS['slippage'],
                                      extra=extra)
        muon_nav = (eq / float(eq.iloc[0])).rename("muon")
        muon_nav.index = pd.to_datetime(muon_nav.index).tz_localize(None)
        print(f"\n[Muon] bars={len(muon_nav)} {muon_nav.index.min()} → {muon_nav.index.max()} "
              f"ret={(muon_nav.iloc[-1]-1)*100:+.1f}%")
    except Exception as e:
        print(f"\n[Muon] UNAVAILABLE: {e} — ④ 跳过组合部分")
        muon_nav = None
        out["portfolio"] = {"error": str(e)}

    if muon_nav is not None:
        m_irm, eq_irm, _ = run_one(data["QQQ"], base_cfg(), "IRM(QQQ)")
        irm_nav = pd.Series((eq_irm["equity"] / eq_irm["equity"].iloc[0]).values,
                            index=pd.to_datetime(eq_irm["date"]), name="irm")
        # IRM 空仓日也有 cash 计息 -> 都是真净值口径, 直接对齐相加
        both = pd.concat([muon_nav, irm_nav], axis=1).dropna()
        r_m = both["muon"].pct_change()
        r_i = both["irm"].pct_change()

        def corr_pack(x, y):
            d = pd.concat([x, y], axis=1).dropna()
            pear = float(d.iloc[:, 0].corr(d.iloc[:, 1]))
            spear = float(d.iloc[:, 0].corr(d.iloc[:, 1], method="spearman"))
            return round(pear, 3), round(spear, 3)

        p_d, s_d = corr_pack(r_m, r_i)
        p_w, s_w = corr_pack(r_m.resample("W").sum(), r_i.resample("W").sum())
        # 下行相关: Muon 负收益日
        dn = r_m < 0
        p_dn, s_dn = corr_pack(r_m[dn], r_i[dn])
        # 尾部: Muon 最差 5% 日 / CRASH 日 IRM 表现
        q05 = r_m.quantile(0.05)
        tail = r_m[r_m <= q05]
        tail_irm = r_i[tail.index]
        tail_tbl = pd.DataFrame({"muon": tail, "irm": tail_irm})
        out["portfolio"] = {
            "overlap_days": int(len(both)),
            "span": [str(both.index[0].date()), str(both.index[-1].date())],
            "pearson_daily": p_d, "spearman_daily": s_d,
            "pearson_weekly": p_w, "spearman_weekly": s_w,
            "pearson_downside": p_dn, "spearman_downside": s_dn,
            "muon_tail5pct_mean": round(float(tail.mean()), 4),
            "irm_in_muon_tail_mean": round(float(tail_irm.mean()), 4),
            "irm_pos_rate_in_muon_tail": round(float((tail_irm > 0).mean()), 3),
            "muon_worst_days": {str(k.date()): {"muon": round(float(v.muon), 4),
                                                "irm": round(float(v.irm), 4)}
                                for k, v in tail_tbl.nsmallest(10, "muon").iterrows()},
        }
        print(f"\n=== ④ Muon x IRM (重叠 {len(both)} 日) ===")
        print(f"day  pearson={p_d} spearman={s_d} | weekly {p_w}/{s_w} | downside {p_dn}/{s_dn}")
        print(f"Muon最差5%日: muon均值{tail.mean():+.3%} vs IRM均值{tail_irm.mean():+.3%} "
              f"(IRM为正比例 {(tail_irm>0).mean():.0%})")

        # 组合: 权重 w Muon + (1-w) IRM, 每日再平衡近似
        port_rows = []
        for w in [1.0, 0.9, 0.75, 0.5, 0.0]:
            rp = w * r_m + (1 - w) * r_i
            nav = (1 + rp).cumprod()
            yrs = len(nav) / 365
            cagr = nav.iloc[-1] ** (1 / yrs) - 1
            sharpe = rp.mean() / rp.std() * np.sqrt(365)
            dd = float((nav / nav.cummax() - 1).min())
            port_rows.append({"port": f"M{w:.0%}/I{1-w:.0%}", "cagr": cagr,
                              "sharpe": round(sharpe, 2), "mdd": dd})
        ptable(port_rows, "④ 组合 (日再平衡近似, 净值均为含成本口径)")
        out["portfolio"]["blends"] = port_rows

        # NAV 图
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 5))
            for col in ["muon", "irm"]:
                ax.plot(both.index, both[col], label=col, lw=1.2)
            for w, c in [(0.9, "#2ca02c"), (0.75, "#ff7f0e"), (0.5, "#d62728")]:
                ax.plot(both.index, ((1 + w * r_m + (1 - w) * r_i).fillna(0)).cumprod(),
                        label=f"blend {w:.0%}/{1-w:.0%}", lw=1.0, alpha=0.8, color=c)
            ax.set_title("Muon (crypto L/S momentum) x IRM-30 (QQQ) NAV")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(RESULTS, "irm_muon_blend.png"), dpi=130)
            print("chart -> irm_muon_blend.png")
        except Exception as e:
            print("chart skip:", e)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f"\nsaved -> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
