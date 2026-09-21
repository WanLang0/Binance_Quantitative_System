# -*- coding: utf-8 -*-
"""
NDX Swing Martin V1 研究 — A/B/C/D 四版马丁 + SRM/Reversal 变体 + 三类对照组

数据:  Yahoo Finance EOD, NDX (^NDX) 2007-10 至今, QQQ 近~2.5年做 1H->4H 窗口验证
执行:  信号当日收盘确认 -> 次日开盘成交 (无未来函数)
资金:  单一资金池 100%, 股数整数计, 未用资金吃 4%/年现金利息 (模拟货基)
仓位: 马丁层权重按目标名义/资金占比执行, 不用杠杆 (总仓<=100%名义)

组合清单:
  BH   Buy & Hold (全仓持有)
  DCA  每月定投 (等额,月初首个交易日)
  M0   传统马丁: 无趋势门, 首笔25%, 每跌3%加25%(2层), 成本+3%TP, -12%组合SL, 长持
  A    经典: MA200>0 门, RSI14<40 且 20日高回撤>=3% 首笔, 20/20/25/35,
              -3/-6/-10% 加仓, 成本+3%TP, -12%SL, 长持
  A_L  A + 最长持仓15个交易日强制平仓 (用户"短中期"定义)
  B    ATR马丁: 同A门, 加仓距离 = 1.0/2.0/3.0 x ATR14(%,信号日), 成本+1.5ATR TP, -12%SL
  C    趋势马丁: MA50>MA200 才允许开新仓(已有仓照常管理), 其余同A
  D    SRM: MA200>0 且 RSI14<35 且 回撤>3% 首笔20%, -1/-2/-3 ATR 加 20/25/35,
              TP=成本+2ATR, SL=-12%, 最长15个交易日
  E    Reversal: D 的加仓条件 + 反转确认(RSI阈值递减 / 4H bull engulfing近似 /
              站回MA20), 未确认则不加仓, 其余同D

输出: scripts/results/ndx_martin_research.json + 控制台汇总表
"""
import json
import math
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
RESULTS = os.path.join(HERE, "results")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

TRADING_DAYS = 252
CASH_RATE = 0.04          # 现金年化 (货基近似)
LONG_HOLD_CAP = 15        # 短中期最长持仓交易日 (A_L/D/E 用)


# ---------------------------------------------------------------- data
def fetch_yahoo(symbol, start="2007-01-01"):
    fp = os.path.join(CACHE, f"yahoo_{symbol.replace('^', '_')}.csv")
    if os.path.exists(fp):
        df = pd.read_csv(fp, parse_dates=["date"])
    else:
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol
            + "?period1=" + str(int(pd.Timestamp(start).timestamp()))
            + "&period2=" + str(int(time.time()))
            + "&interval=1d&events=history"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = {}
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = json.loads(r.read().decode("utf-8"))
        res = raw["chart"]["result"][0]
        ts = res["timestamp"]
        q = res["indicators"]["quote"][0]
        adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
        df = pd.DataFrame({
            "date": pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").tz_localize(None).normalize(),
            "open": q["open"], "high": q["high"], "low": q["low"],
            "close": q["close"], "adjclose": adj, "volume": q["volume"],
        })
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        df.to_csv(fp, index=False)
    df["adj_factor"] = df["adjclose"] / df["close"]
    for c in ["open", "high", "low"]:
        df[c] = df[c] * df["adj_factor"]
    df["close"] = df["adjclose"]
    df = df.drop(columns=["adjclose", "adj_factor"])
    return df.dropna().reset_index(drop=True)


def fetch_yahoo_intraday_1h(symbol, days_back=720):
    """Yahoo 1h 数据最多回溯 ~730 天。"""
    start = pd.Timestamp.utcnow() - pd.Timedelta(days=days_back)
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol
        + "?period1=" + str(int(start.timestamp()))
        + "&period2=" + str(int(time.time()))
        + "&interval=1h&events=history&includeAdjustedClose=true"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = json.loads(r.read().decode("utf-8"))
    res = raw["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
    df = pd.DataFrame({
        "dt": pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").tz_localize(None),
        "open": q["open"], "high": q["high"], "low": q["low"],
        "close": q["close"], "adjclose": adj,
    })
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    af = df["adjclose"] / df["close"]
    for c in ["open", "high", "low"]:
        df[c] = df[c] * af
    df["close"] = df["adjclose"]
    return df.drop(columns=["adjclose"]).dropna().reset_index(drop=True)


def resample_4h(df1h):
    """美东 4h 桶: 10:00 / 14:00 结束的桶 (9:30 开盘 -> 3 bars/日), 标签用桶结束时间."""
    g = df1h.set_index("dt")
    agg = g.resample("4h", origin="start_day", offset="6h").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
    ).dropna(subset=["close"]).reset_index()
    agg = agg[agg["dt"].dt.time.isin([pd.Timestamp("10:00").time(), pd.Timestamp("14:00").time()])]
    agg = agg.rename(columns={"dt": "date"})
    return agg.reset_index(drop=True)


def add_indicators(df):
    c = df["close"]
    df["ma20"] = c.rolling(20).mean()
    df["ma50"] = c.rolling(50).mean()
    df["ma200"] = c.rolling(200).mean()
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)
    df["roll_max20"] = c.rolling(20).max()
    df["dd20"] = c / df["roll_max20"] - 1.0
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - c.shift()).abs(),
        (df["low"] - c.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["atr_pct"] = df["atr"] / c
    df["engulf"] = (c > df["open"]) & (c.shift() < df["open"].shift()) & (c > df["open"].shift())
    return df


# ---------------------------------------------------------------- engine
def perf_metrics(equity, trades, bars_per_year=TRADING_DAYS, label=""):
    eq = equity["equity"].values
    ret = pd.Series(eq).pct_change().fillna(0)
    yrs = len(eq) / bars_per_year
    cagr = (eq[-1] / eq[0]) ** (1 / yrs) - 1 if yrs > 0 and eq[-1] > 0 else -1.0
    vol = ret.std() * math.sqrt(bars_per_year)
    sharpe = ret.mean() / ret.std() * math.sqrt(bars_per_year) if ret.std() > 0 else 0.0
    downside = ret[ret < 0].std() * math.sqrt(bars_per_year)
    sortino = ret.mean() / downside if downside and downside > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    dd_series = eq / peak - 1
    mdd = float(dd_series.min())
    mdd_start = int(np.argmax(dd_series == mdd))
    mdd_peak = int(np.argmax(eq[:mdd_start + 1]))
    recovery = (mdd_start - mdd_peak) / bars_per_year if mdd < 0 else 0.0

    # ---- underwater / loss-streak 分析 (用户要求的 Recovery Time 审计项)
    underwater = dd_series < -1e-9
    uw_spells = []
    in_uw = False
    l0 = 0
    for j in range(len(underwater)):
        if underwater[j] and not in_uw:
            in_uw, l0 = True, j
        elif not underwater[j] and in_uw:
            in_uw = False
            uw_spells.append(j - l0)
    if in_uw:
        uw_spells.append(len(underwater) - l0)
    max_uw_bars = max(uw_spells) if uw_spells else 0
    avg_uw_bars = float(np.mean(uw_spells)) if uw_spells else 0.0
    pct_time_uw = float(underwater.mean())

    closed = [t for t in trades if t["status"] in ("tp", "sl", "time_exit", "be")]
    wins = [t for t in closed if t["pnl_pct"] > 0]
    losses = [t for t in closed if t["pnl_pct"] <= 0]
    win_rate = len(wins) / len(closed) if closed else 0.0
    avg_w = float(np.mean([t["pnl_pct"] for t in wins])) if wins else 0.0
    avg_l = float(np.mean([t["pnl_pct"] for t in losses])) if losses else 0.0

    # 连续亏损 / SL 连击 / SL 后恢复时间 (下笔盈利交易的开仓距 SL 平仓的 bars)
    streak = max_streak = 0
    sl_streak = max_sl_streak = 0
    sl_recoveries = []
    last_sl_exit = None
    for t in closed:
        if t["pnl_pct"] <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
        if t["status"] == "sl":
            sl_streak += 1
            max_sl_streak = max(max_sl_streak, sl_streak)
            last_sl_exit = t["exit_date"]
        else:
            sl_streak = 0
            if last_sl_exit is not None:
                sl_recoveries.append(t["entry_date"])
                last_sl_exit = None
    sl_rec_days = []
    if sl_recoveries:
        dt = pd.to_datetime(pd.Series(sl_recoveries))
        # 近似: 用 trades 里对应 exit 与下一 entry 的日历日差
    # 更直接: 在 closed 序列上找 sl -> 下一笔 win 的 entry 间隔
    dates_idx = {str(t["exit_date"]): k for k, t in enumerate(closed)}
    rec_bars = []
    pending_sl_end = None
    for t in closed:
        if t["status"] == "sl":
            pending_sl_end = t
            continue
        if pending_sl_end is not None and t["pnl_pct"] > 0:
            d0 = pd.Timestamp(pending_sl_end["exit_date"])
            d1 = pd.Timestamp(t["entry_date"])
            rec_bars.append((d1 - d0).days)
            pending_sl_end = None
        elif pending_sl_end is not None and t["pnl_pct"] <= 0:
            pending_sl_end = t if t["status"] == "sl" else pending_sl_end

    open_trades = [t for t in trades if t["status"] == "open"]
    return {
        "label": label,
        "final_equity": round(float(eq[-1]), 4),
        "total_return": round(eq[-1] / eq[0] - 1, 4),
        "cagr": round(cagr, 4),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "vol": round(vol, 4),
        "max_dd": round(mdd, 4),
        "mdd_duration_years": round(recovery, 2),
        "max_underwater_bars": int(max_uw_bars),
        "avg_underwater_bars": round(avg_uw_bars, 1),
        "pct_time_underwater": round(pct_time_uw, 4),
        "trades": len(trades),
        "closed_trades": len(closed),
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_w, 4),
        "avg_loss": round(avg_l, 4),
        "expectancy_per_trade": round((eq[-1] / eq[0]) ** (1 / max(len(closed), 1)) - 1, 5),
        "avg_hold_days": round(float(np.mean([t["bars_held"] for t in closed])), 1) if closed else 0,
        "max_hold_days": int(max([t["bars_held"] for t in closed], default=0)),
        "max_layers_used": int(max([t["layers"] for t in trades], default=0)),
        "max_loss_streak": int(max_streak),
        "max_sl_streak": int(max_sl_streak),
        "avg_sl_recovery_days": round(float(np.mean(rec_bars)), 1) if rec_bars else 0.0,
        "max_sl_recovery_days": int(max(rec_bars)) if rec_bars else 0,
        "still_open_at_end": len(open_trades),
        "open_unrealized_pct": round(float(np.mean([t["pnl_pct"] for t in open_trades])), 4) if open_trades else 0.0,
    }


class MartinEngine:
    """日线/4H通用马丁引擎。bar 时间单位 = 1根K线 (日线=1交易日)。

    核心规则:
      entry 信号在 bar t 收盘确认 -> bar t+1 开盘成交
      管理层检查 (加仓/TP/SL/time) 用 bar t+1 的 H/L, 成交价 t+1 开盘
      layer distances 基于第一笔实际成交价; TP 基于组合平均成本
      TP 触发价 >= 成本*(1+tp): 用开盘价成交, 否则用触发价 (保守)
      SL 触发价 <= 成本*(1-sl): 优先于 TP 检查 (保守)
    """

    def __init__(self, df, cfg):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.cost = cfg.get("cost_bps", 0.0) / 10000.0   # 单边成本
        self.equity = cfg.get("init_equity", 1_000_000.0)
        self.shares = 0.0
        self.cash = self.equity
        self.layers = []            # [(market_px, shares, cash_spent)]
        self.layer_log = []         # 每层成交审计: 成交价/信号日上下文/确认类型
        self.entry_bar = None
        self.trades = []
        self.equity_curve = []
        self.pending_entry = None
        self.pending_bar = None
        self.risk_off = False       # SL 后冷却

    # -- helpers
    def avg_cost(self):
        """每股份的总成本 (含手续费), 用于 TP/SL 触发价"""
        if not self.layers:
            return 0.0
        q = sum(s for _, s, _ in self.layers)
        return sum(c for _, _, c in self.layers) / q

    def market_value(self, price):
        return self.shares * price

    def snapshot(self, i, price):
        return self.cash + self.market_value(price)

    def close_all(self, i, price, status, reason=""):
        qty = self.shares
        if qty > 0:
            proceeds = qty * price * (1 - self.cost)
            self.cash += proceeds
            cost_basis = sum(c for _, _, c in self.layers)
            pnl_pct = (proceeds - cost_basis) / cost_basis if cost_basis else 0.0
            t0 = self.entry_bar
            r0 = self.df.iloc[t0]
            avg = self.avg_cost()
            self.trades.append({
                "entry_date": str(self.df["date"].iloc[self.entry_bar].date()) if self.entry_bar is not None else "",
                "exit_date": str(self.df["date"].iloc[i].date()),
                "layers": len(self.layers),
                "avg_cost": round(self.avg_cost(), 2),
                "exit_price": round(price, 2),
                "pnl_pct": round(pnl_pct, 4),
                "bars_held": i - self.entry_bar,
                "status": status,
                "reason": reason,
                # ---- audit context ----
                "entry_rsi": round(float(r0["rsi"]), 1) if not np.isnan(r0["rsi"]) else None,
                "entry_dd20": round(float(r0["dd20"]), 4),
                "entry_atr_pct": round(float(r0["atr_pct"]), 4),
                "tp_trigger": round(avg * (1 + self.tp_distance(t0)), 2),
                "sl_trigger": round(avg * (1 - self.cfg["sl_pct"]), 2),
                "total_invested": round(cost_basis, 2),
                "layer_fills": [dict(x) for x in self.layer_log],
            })
        self.shares = 0.0
        self.layers = []
        self.entry_bar = None
        self.layer_log = []
        self.pending_entry = None
        self.pending_bar = None
        if status == "sl":
            self.risk_off = True

    # -- signal functions (all evaluated on bar t, close-confirmed)
    def entry_ok(self, t):
        c = self.cfg
        r = self.df.iloc[t]
        if not c.get("entry", True):
            return False
        if c.get("trend_gate", "ma200") == "ma200" and not (r["close"] > r["ma200"]):
            return False
        if c.get("trend_gate") == "golden" and not (r["ma50"] > r["ma200"]):
            return False
        if c.get("rsi_entry") is not None and not (r["rsi"] < c["rsi_entry"]):
            return False
        if c.get("dd_entry") is not None and not (r["dd20"] <= -c["dd_entry"]):
            return False
        if c.get("entry_confirm"):
            rsi_up = t > 0 and r["rsi"] > self.df["rsi"].iloc[t - 1]
            if not (bool(r["engulf"]) or rsi_up):
                return False
        return True

    def layer_offsets(self, t):
        """返回 [ (dist_pct_from_entry(正数=更低), weight) ... ] for layers 2..N"""
        c = self.cfg
        r = self.df.iloc[t]
        if c.get("spacing") == "atr":
            k = c["atr_mult"]            # list like [1.0, 2.0, 3.0]
            ap = r["atr_pct"]
            offs = [k[j] * ap for j in range(len(k))]
        else:
            offs = c["fixed_steps"]      # e.g. [0.03, 0.06, 0.10]
        return offs

    def tp_distance(self, t):
        c = self.cfg
        r = self.df.iloc[t]
        if c.get("tp") == "atr":
            return c["tp_mult"] * r["atr_pct"]
        return c["tp_pct"]               # 0.03

    def next_entry_price_ok(self, t):
        pass  # placeholder

    def confirm_reversal(self, t):
        """E 版: 加仓需要反转确认。返回该层所需的确认函数结果。"""
        r = self.df.iloc[t]
        return (
            r["rsi"] < self.cfg["rev_rsi"][min(len(self.layers) - 1, len(self.cfg["rev_rsi"]) - 1)]
            or bool(r["engulf"])
            or (not np.isnan(r["ma20"]) and r["close"] > r["ma20"])
        )

    # -- main loop
    def run(self):
        df = self.df
        c = self.cfg
        n = len(df)
        cash_rate_bar = c.get("cash_rate", CASH_RATE) / TRADING_DAYS
        last_entry_bar = None
        min_gap = c.get("min_gap_bars", 2)
        start = c.get("start_bar", 0)      # walk-forward: 指标可预热, 但只在 start 后交易

        for i in range(start, n):
            r = df.iloc[i]
            o, h, l = r["open"], r["high"], r["low"]

            # ---- execute pending entry (exec mode: T+1 open[默认] / T+1 close / T+2 open)
            if self.pending_entry is not None and not self.risk_off:
                mode = self.cfg.get("exec", "open")
                exec_now = (i - self.pending_bar) >= (2 if mode == "open2" else 1)
                px = o if mode in ("open", "open2") else r["close"]
                if exec_now and px > 0:
                    w = self.pending_entry
                    alloc = min(self.equity * w, self.cash)
                    if alloc > self.cash * 0.5:      # 防现金不足时跳票
                        alloc = self.cash
                    if alloc > 0:
                        fill_px = px * (1 + self.cost)      # 买入成本
                        sh = alloc / fill_px
                        self.shares += sh
                        self.cash -= alloc
                        self.layers.append((px, sh, alloc))
                        sig = self.df.iloc[self.pending_bar]
                        if self.entry_bar is None:
                            self.entry_bar = i
                            last_entry_bar = i
                        self.layer_log.append({
                            "layer": len(self.layers),
                            "sig_date": str(sig["date"].date()),
                            "fill_date": str(r["date"].date()),
                            "fill_px": round(px, 2),
                            "weight": w,
                            "alloc": round(alloc, 2),
                            "sig_rsi": round(float(sig["rsi"]), 1),
                            "sig_dd20": round(float(sig["dd20"]), 4),
                            "sig_atr_pct": round(float(sig["atr_pct"]), 4),
                        })
                    self.pending_entry = None
                    self.pending_bar = None

            # ---- manage open position (checks use today's H/L)
            if self.layers:
                avg = self.avg_cost()
                tp_px = avg * (1 + self.tp_distance(self.entry_bar or 0))
                sl_px = avg * (1 - c["sl_pct"])
                # time exit
                held = i - self.entry_bar
                if c.get("max_hold") and held >= c["max_hold"]:
                    self.close_all(i, o, "time_exit", "max_hold")
                else:
                    hit_sl = l <= sl_px
                    hit_tp = h >= tp_px
                    if hit_sl and hit_tp:
                        self.close_all(i, o, "sl", "both_hit_open")   # 保守
                    elif hit_sl:
                        self.close_all(i, sl_px, "sl", "portfolio_sl")
                    elif hit_tp:
                        fill = o if o >= tp_px else tp_px
                        self.close_all(i, fill, "tp", "take_profit")

            # ---- add-on check (signal from previous bar, execute today open already done;
            #      here we set pending for tomorrow based on today's bar)
            if (
                self.layers and self.pending_entry is None and not self.risk_off
                and len(self.layers) < c["max_layers"]
                and c.get("entry", True) is not False
                and self.entry_bar is not None
                and (i - (last_entry_bar or 0)) >= min_gap
                and (c.get("trend_gate") != "golden" or df.iloc[i]["ma50"] > df.iloc[i]["ma200"])
                and not c.get("no_addons", False)
            ):
                entry_px = self.layers[0][0]
                req_drop = self.layer_offsets(self.entry_bar)[len(self.layers) - 1]
                trig = entry_px * (1 - req_drop)
                ok = l <= trig
                if ok and c.get("reversal_confirm"):
                    ok = self.confirm_reversal(i)
                if ok:
                    self.pending_entry = c["weights"][len(self.layers)]
                    self.pending_bar = i

            # ---- fresh entry check (based on today's close, executed tomorrow)
            if (
                not self.layers and self.pending_entry is None and not self.risk_off
                and self.entry_ok(i)
            ):
                self.pending_entry = c["weights"][0]
                self.pending_bar = i

            # ---- book equity at close
            eq = self.snapshot(i, r["close"])
            self.equity = eq
            self.cash *= 1 + cash_rate_bar
            self.equity_curve.append({
                "date": r["date"], "equity": eq,
                "invested_frac": round(self.market_value(r["close"]) / eq, 4) if eq else 0,
            })

            # risk_off 冷却: 跌破 MA200 x 5% 才重新允许开仓
            if self.risk_off and self.cfg.get("cool_reset", True):
                if r["close"] < r["ma200"] * 0.95:
                    self.risk_off = False

        # end: close open position at last close for stats
        if self.layers:
            self.close_all(n - 1, df["close"].iloc[-1], "open", "end_of_data")
            # restore status open for honesty
            self.trades[-1]["status"] = "open_end"
        eq_df = pd.DataFrame(self.equity_curve)
        return eq_df, self.trades


# ---------------------------------------------------------------- baselines
def bh_backtest(df):
    eq0, eq, shares = 1_000_000.0, [], None
    for _, r in df.iterrows():
        if shares is None:
            shares = eq0 / r["open"]
        eq.append({"date": r["date"], "equity": shares * r["close"]})
    return pd.DataFrame(eq), []


def dca_backtest(df):
    eq0 = 1_000_000.0
    monthly_cash = eq0 * 0.005        # 每月投入 0.5% 初始资金 (≈定投工资流)
    cash, shares, eq, last_month = eq0 * 0.98, 0.0, [], None
    for _, r in df.iterrows():
        m = r["date"].to_period("M")
        if m != last_month:
            cash += monthly_cash
            shares += monthly_cash / r["open"]
            last_month = m
        eq.append({"date": r["date"], "equity": cash + shares * r["close"]})
    return pd.DataFrame(eq), []


# ---------------------------------------------------------------- strategy configs
def build_configs():
    return {
        "BH":  {"kind": "baseline_bh"},
        "DCA": {"kind": "baseline_dca"},
        "M0": dict(kind="martin", trend_gate=None, entry=True, rsi_entry=None, dd_entry=None,
                   weights=[.25, .25, .25, .25], fixed_steps=[.03, .06, .10], max_layers=4,
                   tp="pct", tp_pct=.03, sl_pct=.12, max_hold=None, min_gap_bars=2),
        "A":  dict(kind="martin", trend_gate="ma200", rsi_entry=40, dd_entry=.03,
                   weights=[.20, .20, .25, .35], fixed_steps=[.03, .06, .10], max_layers=4,
                   tp="pct", tp_pct=.03, sl_pct=.12, max_hold=None, min_gap_bars=2),
        "A_L": dict(kind="martin", trend_gate="ma200", rsi_entry=40, dd_entry=.03,
                    weights=[.20, .20, .25, .35], fixed_steps=[.03, .06, .10], max_layers=4,
                    tp="pct", tp_pct=.03, sl_pct=.12, max_hold=LONG_HOLD_CAP, min_gap_bars=2),
        "B":  dict(kind="martin", trend_gate="ma200", rsi_entry=40, dd_entry=.03,
                   weights=[.20, .20, .25, .35], spacing="atr", atr_mult=[1.0, 2.0, 3.0],
                   max_layers=4, tp="atr", tp_mult=1.5, sl_pct=.12, max_hold=None, min_gap_bars=2),
        "C":  dict(kind="martin", trend_gate="golden", rsi_entry=40, dd_entry=.03,
                   weights=[.20, .20, .25, .35], fixed_steps=[.03, .06, .10], max_layers=4,
                   tp="pct", tp_pct=.03, sl_pct=.12, max_hold=None, min_gap_bars=2),
        "D":  dict(kind="martin", trend_gate="ma200", rsi_entry=35, dd_entry=.03,
                   weights=[.20, .20, .25, .35], spacing="atr", atr_mult=[1.0, 2.0, 3.0],
                   max_layers=4, tp="atr", tp_mult=2.0, sl_pct=.12,
                   max_hold=LONG_HOLD_CAP, min_gap_bars=2),
        "E":  dict(kind="martin", trend_gate="ma200", rsi_entry=35, dd_entry=.03,
                   weights=[.20, .20, .25, .35], spacing="atr", atr_mult=[1.0, 2.0, 3.0],
                   max_layers=4, tp="atr", tp_mult=2.0, sl_pct=.12,
                   max_hold=LONG_HOLD_CAP, min_gap_bars=2,
                   reversal_confirm=True, rev_rsi=[40, 32, 28]),
    }


def run_all(df, cfgs, tag):
    rows, details = [], {}
    for name, cfg in cfgs.items():
        if cfg.get("kind") == "baseline_bh":
            eq, trades = bh_backtest(df)
        elif cfg.get("kind") == "baseline_dca":
            eq, trades = dca_backtest(df)
        else:
            eng = MartinEngine(df, cfg)
            eq, trades = eng.run()
        m = perf_metrics(eq, trades, label=f"{tag}|{name}")
        m["invested_frac_avg"] = round(float(eq["invested_frac"].mean()), 4) if "invested_frac" in eq else 1.0
        m["time_in_market_pct"] = round(float((eq["invested_frac"] > 0.01).mean()), 4) if "invested_frac" in eq else 1.0
        rows.append(m)
        details[name] = {
            "metrics": m,
            "trades": trades,
            "equity": eq.assign(date=eq["date"].astype(str)).to_dict("records"),
        }
    return rows, details


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    out_path = os.path.join(RESULTS, "ndx_martin_research.json")
    results = {"meta": {"generated": pd.Timestamp.now().isoformat(), "engine": "MartinEngine v1",
                        "cash_rate": CASH_RATE, "exec": "signal close t -> next open t+1"}}

    print("== 1) NDX daily 2007-now ==")
    ndx = add_indicators(fetch_yahoo("^NDX"))
    ndx = ndx[ndx["date"] >= "2010-01-01"].reset_index(drop=True)
    print(f"   bars={len(ndx)}  {ndx['date'].iloc[0].date()} -> {ndx['date'].iloc[-1].date()}")
    cfgs = build_configs()
    rows, det = run_all(ndx, cfgs, "NDX_D")
    results["daily"] = {r["label"].split("|")[1]: r for r in rows}
    results["details_daily"] = {k: v["metrics"] | {"trades": v["trades"],
                                                   "equity": v["equity"]}
                                for k, v in det.items()}

    print("== 2) stress windows: GFC 2007-2009 / bear 2022 (daily) ==")
    ndx_all = add_indicators(fetch_yahoo("^NDX"))
    results["stress"] = {}
    for tag, lo, hi in [("GFC_2007_2009", "2007-10-01", "2009-12-31"),
                        ("BEAR_2022", "2021-11-20", "2022-12-31")]:
        sl = ndx_all[(ndx_all["date"] >= lo) & (ndx_all["date"] <= hi)].reset_index(drop=True)
        sub_cfgs = {k: v for k, v in build_configs().items()
                    if k in ("BH", "M0", "A", "C", "E", "D")}
        rows_s, _ = run_all(sl, sub_cfgs, tag)
        results["stress"][tag] = {r["label"].split("|")[1]: r for r in rows_s}

    print("== 3) 4H window: QQQ 1h -> 4h (~last 2.5y) ==")
    try:
        q1h = fetch_yahoo_intraday_1h("QQQ", days_back=700)
        q4h = add_indicators(resample_4h(q1h))
        q4h = q4h.dropna(subset=["ma200"]).reset_index(drop=True)
        print(f"   4h bars={len(q4h)}  {q4h['date'].iloc[0]} -> {q4h['date'].iloc[-1]}")
        rows4, det4 = run_all(q4h, {k: v for k, v in build_configs().items()}, "QQQ_4H")
        results["h4"] = {r["label"].split("|")[1]: r for r in rows4}
    except Exception as e:
        print("   4H skip:", e)
        results["h4"] = {}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\nsaved -> {out_path}   ({time.time()-t0:.1f}s)")

    # ---- console tables
    cols = ["label", "total_return", "cagr", "sharpe", "max_dd", "mdd_duration_years",
            "win_rate", "avg_win", "avg_loss", "trades", "avg_hold_days", "max_hold_days",
            "max_layers_used", "time_in_market_pct"]
    for section in ["daily", "h4"]:
        if not results.get(section):
            continue
        print(f"\n----- {section} -----")
        table = pd.DataFrame([results[section][k] for k in results[section]])
        avail = [c for c in cols if c in table.columns]
        print(table[avail].to_string(index=False))

    print("\n----- trade samples (NDX_D, best martin variant) -----")
    best = max([k for k in results["details_daily"] if k not in ("BH", "DCA")],
               key=lambda k: results["details_daily"][k]["cagr"])
    for t in results["details_daily"][best]["trades"][:8]:
        print(t)
    print(f"best martin variant by CAGR: {best}")


if __name__ == "__main__":
    main()
