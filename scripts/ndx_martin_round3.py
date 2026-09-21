# -*- coding: utf-8 -*-
"""
R3 — Cross-Market Validation (IRM-30 候选)
  固定参数, 所有市场不重优化:
    RSI30 入场 + 20日高回撤>=3% + sp 2.5/3.0/3.5 ATR + TP=2ATR + SL=-12% + 15日强平
    weights 20/20/25/35, 反转确认加仓 (RSI阈值40/32/28 / engulf / 站回MA20)
  市场: ^NDX, QQQ, ^GSPC, ^GDAXI, ^STOXX50E, ^HSI, ^N225 (Yahoo, 全部可用历史)

实验:
  ① 跨市场主表: CE 与 E_ng 两结构 x 7 市场
  ② 四象限 (每市场): {单仓100% / 马丁分层} x {无入场确认 / 有入场确认}
     -> 回答 "到底是什么在赚钱"
  ③ 行情归因: Bull/Bear/Sideways (MA200坡度+价格位) x HiVol/LoVol (ATR%中位数)
     -> CE 策略收益在各行情状态的分布

成本统一 10bps/边, 现金 4%/年。
输出: scripts/results/ndx_martin_round3.json
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
                                 fetch_yahoo, bh_backtest, RESULTS)

OUT = os.path.join(RESULTS, "ndx_martin_round3.json")
COST = 10
MARKETS = [("^NDX", "NDX"), ("QQQ", "QQQ"), ("^GSPC", "SPX"), ("^GDAXI", "DAX"),
           ("^STOXX50E", "SX5E"), ("^HSI", "HSI"), ("^N225", "N225")]


def base_cfg(**kw):
    cfg = dict(kind="martin", trend_gate=None, rsi_entry=30, dd_entry=.03,
               weights=[.20, .20, .25, .35], spacing="atr",
               atr_mult=[2.5, 3.0, 3.5], max_layers=4,
               tp="atr", tp_mult=2.0, sl_pct=.12, max_hold=15,
               min_gap_bars=2, reversal_confirm=True, rev_rsi=[40, 32, 28],
               cost_bps=COST, start_bar=0)
    cfg.update(kw)
    return cfg


def run_one(df, cfg, label):
    eng = MartinEngine(df, cfg)
    eq, trades = eng.run()
    m = perf_metrics(eq, trades, label=label)
    m["time_in_market_pct"] = round(float((eq["invested_frac"] > 0.01).mean()), 4)
    return m, eq, trades


def ptable(rows, title, cols=("strat", "cagr", "sharpe", "mdd", "win%", "avgW",
                              "avgL", "n", "in_mkt%")):
    print("\n=== " + title + " ===")
    df = pd.DataFrame([{c: r[c] for c in cols} for r in rows])
    fmt = {"cagr": "{:+.1%}", "mdd": "{:+.1%}", "win%": "{:.0%}", "avgW": "{:+.2%}",
           "avgL": "{:+.2%}", "in_mkt%": "{:.0%}", "sharpe": "{:.2f}"}
    for c, f in fmt.items():
        if c in df:
            df[c] = df[c].map(lambda v: f.format(v) if isinstance(v, (int, float)) else v)
    print(df.to_string(index=False))


def row_of(name, m):
    return {"strat": name, "ret": m["total_return"], "cagr": m["cagr"],
            "sharpe": m["sharpe"], "sortino": m["sortino"], "mdd": m["max_dd"],
            "uw_max_d": m["max_underwater_bars"], "uw_pct": m["pct_time_underwater"],
            "win%": m["win_rate"], "avgW": m["avg_win"], "avgL": m["avg_loss"],
            "streak": m["max_loss_streak"], "n": m["closed_trades"],
            "hold_avg": m["avg_hold_days"], "in_mkt%": m["time_in_market_pct"]}


# ---------------- regime attribution ----------------
def regime_labels(df):
    """用全样本 MA200 斜率+价格位划分 Bull/Bear/Side; ATR% 中位数分 Hi/LoVol。
    标签只用当日及之前信息构造的均线状态, 归因是描述性统计, 不参与交易。"""
    c = df["close"]
    slope = df["ma200"] / df["ma200"].shift(63) - 1
    bull = (c > df["ma200"]) & (slope > 0.02)
    bear = (c < df["ma200"]) & (slope < -0.02)
    reg = np.where(bull, "Bull", np.where(bear, "Bear", "Side"))
    medv = float(df["atr_pct"].median())
    vol = np.where(df["atr_pct"] > medv, "HiVol", "LoVol")
    return pd.Series(reg, index=df.index), pd.Series(vol, index=df.index), medv


def regime_attribution(eq, df, reg, vol):
    e = eq.copy()
    e["ret"] = e["equity"].pct_change().fillna(0)
    e["reg"] = reg.values[:len(e)]
    e["vol"] = vol.values[:len(e)]
    out = {}
    for key in ["reg", "vol"]:
        grp = e.groupby(key)["ret"].agg(["sum", "count", "mean"])
        pos = e.assign(pos=(e["ret"] > 0)).groupby(key)["pos"].mean()
        out[key] = {str(k): {"ret_sum": round(float(v["sum"]), 4),
                             "days": int(v["count"]),
                             "pos_day%": round(float(pos[k]), 3)}
                    for k, v in grp.iterrows()}
    return out


def main():
    t0 = time.time()
    out = {"meta": {"fixed_params": {"rsi": 30, "dd": 0.03, "spacing_atr": [2.5, 3, 3.5],
                                     "tp_atr": 2.0, "sl": -0.12, "max_hold": 15,
                                     "weights": [0.2, 0.2, 0.25, 0.35],
                                     "addon_confirm": True, "cost_bps": COST},
                    "no_reopt": True}}

    # ① + ② per market
    main_rows, quad_rows = [], []
    out["markets"] = {}
    for sym, name in MARKETS:
        try:
            raw = fetch_yahoo(sym)
        except Exception as e:
            print(f"[{name}] fetch failed: {e}")
            continue
        df = add_indicators(raw).dropna(subset=["ma200"]).reset_index(drop=True)
        reg, vol, medv = regime_labels(df)

        # --- ① CE (martin + addon-confirm, 无趋势门 = E_ng 结构, R2 显示门无独立收益增益)
        m_ce, eq_ce, _ = run_one(df, base_cfg(), f"{name}|CE")
        # --- E_ng 对照 (RSI35 加仓确认阈值版 = R2 原版, 检验 rsi30 迁移是否稳)
        m_e35, _, _ = run_one(df, base_cfg(rsi_entry=35, rev_rsi=[40, 32, 28]), f"{name}|E35")
        m_bh, eq_bh, tr_bh = run_one.__wrapped__(df, {"kind": "bh"}, name) if False else (None, None, None)
        bh_eq, bh_tr = bh_backtest(df)
        m_bh = perf_metrics(bh_eq, bh_tr, label=f"{name}|BH")
        m_bh["time_in_market_pct"] = 1.0

        for r in (row_of("CE", m_ce), row_of("E35", m_e35), row_of("BH", m_bh)):
            main_rows.append({"mkt": name} | r)
        out["markets"][name] = {"CE": m_ce, "E35": m_e35, "BH": m_bh,
                                "atr_pct_median": medv,
                                "span": [str(df['date'].iloc[0].date()), str(df['date'].iloc[-1].date())],
                                "bars": len(df)}

        # --- ② 四象限 ( Martin x 入场确认 ), 全部用同一入场条件 RSI30+dd3%
        quad_defs = {
            "A single_noconf": base_cfg(max_layers=1, weights=[1.0], reversal_confirm=False,
                                        entry_confirm=False),
            "B martin_noconf": base_cfg(reversal_confirm=False, entry_confirm=False),
            "C single_conf":   base_cfg(max_layers=1, weights=[1.0], reversal_confirm=False,
                                        entry_confirm=True),
            "D martin_conf":   base_cfg(entry_confirm=True),
        }
        for qname, qcfg in quad_defs.items():
            mq, _, _ = run_one(df, qcfg, f"{name}|{qname}")
            quad_rows.append({"mkt": name, "quad": qname.split("_")[0],
                              "variant": "_".join(qname.split("_")[1:])} | row_of(qname, mq))
            out["markets"][name][f"quad_{qname}"] = mq

        # --- ③ 行情归因 (CE)
        out["markets"][name]["regime_ce"] = regime_attribution(eq_ce, df, reg, vol)

    # ---------------- tables ----------------
    df_main = pd.DataFrame(main_rows)
    fmt = {"cagr": "{:+.1%}", "mdd": "{:+.1%}", "win%": "{:.0%}", "avgW": "{:+.2%}",
           "avgL": "{:+.2%}", "in_mkt%": "{:.0%}", "sharpe": "{:.2f}"}
    show = df_main[["mkt", "strat", "cagr", "sharpe", "mdd", "win%", "avgW", "avgL",
                    "n", "in_mkt%", "hold_avg", "streak"]].copy()
    for c, f in fmt.items():
        show[c] = show[c].map(lambda v: f.format(v) if isinstance(v, (int, float)) else v)
    print("\n===== ① 跨市场主表 (固定 RSI30/sp2.5-3-3.5/TP2ATR/SL-12%/15d, 10bps) =====")
    print(show.to_string(index=False))

    dfq = pd.DataFrame(quad_rows)
    showq = dfq[["mkt", "quad", "variant", "cagr", "sharpe", "mdd", "win%", "avgL",
                 "n", "in_mkt%"]].copy()
    for c, f in fmt.items():
        if c in showq:
            showq[c] = showq[c].map(lambda v: f.format(v) if isinstance(v, (int, float)) else v)
    print("\n===== ② 四象限: A单仓无确认 / B马丁无确认 / C单仓+确认 / D马丁+确认 =====")
    print(showq.to_string(index=False))

    # 象限均值
    qavg = dfq.groupby("quad")[["sharpe", "cagr", "mdd", "avgL", "win%"]].mean()
    print("\n--- 四象限跨市场均值 ---")
    print(qavg.to_string(float_format=lambda x: f"{x:.3f}"))

    # regime 表
    print("\n===== ③ CE 行情归因 (ret_sum = 该状态内策略日收益合计) =====")
    for name in out["markets"]:
        rg = out["markets"][name].get("regime_ce", {})
        if not rg:
            continue
        parts = []
        for k, v in rg.get("reg", {}).items():
            parts.append(f"{k}:{v['ret_sum']:+.1%}/{v['days']}d")
        parts_v = []
        for k, v in rg.get("vol", {}).items():
            parts_v.append(f"{k}:{v['ret_sum']:+.1%}")
        print(f"{name:5s} | " + "  ".join(parts) + "   ||  " + "  ".join(parts_v))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f"\nsaved -> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
