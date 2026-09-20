# -*- coding: utf-8 -*-
"""横截面动量回测服务（研究/回测代码，归属 scripts/，不入生产依赖链）。

封装同目录研究引擎 tmp_xsec_mom_longshort.py，供 Web 回测页 /momentum 调用。
实盘执行代码 momentum_live_trader.py 位于项目根目录，不依赖本模块及 scripts/ 下任何文件。

最优口径（默认 = V2@U8-ER「Muon」冻结规格）:
  U8 PIT 宇宙（ADV30≥100M · 上市≥365d · 日成交额≥10M · ADV30 Top50）
  R14 排名 · Long Top20% 等权 · Short Bot20% ExpRank α=0.3
  dollar-neutral 50/50 · MA60 gate（OFF 全平）· 5D 网格 T+5 open 执行
  真实 Funding(逐8h事件) · 25bps 成本（5bps fee + 20bps 滑点）· 1x

本模块只做「读取缓存 → 跑回测 → 组装展示数据」。
"""
import json
import os
import sys

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import numpy as np
import pandas as pd

# 研究引擎位于 scripts/tmp_xsec_mom_longshort.py，且依赖本机 scripts/cache/*.pkl
# 回测缓存——两者均不入库（.gitignore 排除），生产/服务器拉代码后可能不存在。
# 缺失时不阻断 app 启动：回测页显示提示，实盘页 /crypto-momentum 不受影响。
try:
    from tmp_xsec_mom_longshort import (
        load as _load,
        backtest as _backtest,
        stat_core as _stat_core,
        INIT,
        monthly_breakdown as _monthly_breakdown,
    )
    from tmp_r7_universe_audit import pool_at as _pool_at, build_gate as _build_gate
    _ENGINE_OK = True
except ImportError:
    _load = _backtest = _stat_core = _monthly_breakdown = None
    _pool_at = _build_gate = None
    INIT = 10000.0
    _ENGINE_OK = False

MISSING_MSG = ("本机缺少回测研究引擎（scripts/tmp_xsec_mom_longshort.py）或"
               "日线缓存（scripts/cache/*.pkl），回测功能不可用。"
               "实盘页「横截面动量量化」不受影响，可正常使用。")


def available():
    return _ENGINE_OK

# 默认参数 = V2@U8-ER「Muon」冻结规格（R28-D 对账口径）
DEFAULTS = dict(
    rebal_n=5,
    top_frac=0.20,
    liq_min=100_000_000.0,
    leverage=1.0,
    slippage=0.002,       # 20bps 滑点（总成本 25bps = 5 fee + 20 slip）
    mode='r14',
    vol_max=None,
    abs_mom=False,
    long_only=False,
    funding_mode='8h',
    min_len=35,
    universe='U8',        # 'U8' = PIT 动态宇宙 + MA60 gate + ExpRank；'fixed' = 旧 30 币口径
    short_weight='exp',   # 短腿 ExpRank α=0.3（仅 U8 口径生效）
)

SLIP_GRID = [('0', 0.0), ('10', 0.001), ('20', 0.002), ('30', 0.003), ('40', 0.004)]

_data_cache = {}


def _load_universe_meta():
    """加载 PIT 上下文：全宇宙日线（含 liq/r14 列）+ onboard 上市日期。

    U8 冻结口径依赖两件东西：
    - load_all 全币缓存（fixed 口径只用 30 币，U8 需要 ADV30 Top50 扫描）
    - cache/onboard.json（上市日期，用于 age≥365d 过滤；live trader 同款）
    任一缺失则回退 fixed 口径。"""
    from tmp_fund_xsec_r5 import load_all
    from tmp_combo_full import W1 as _W1
    dfs, funds = load_all(min_len=DEFAULTS['min_len'])
    for df in dfs.values():
        # load_all 只带 liq 列，补齐 signal() 所需的动量/波动率列
        if 'r7' not in df.columns:
            df['r7'] = df['close'] / df['close'].shift(7) - 1.0
        if 'r14' not in df.columns:
            df['r14'] = df['close'] / df['close'].shift(14) - 1.0
        if 'r30' not in df.columns:
            df['r30'] = df['close'] / df['close'].shift(30) - 1.0
        if 'vol20' not in df.columns:
            df['vol20'] = df['close'].pct_change().rolling(20).std() * np.sqrt(365)
    ob = {}
    ob_path = os.path.join(_SCRIPTS, 'cache', 'onboard.json')
    if os.path.exists(ob_path):
        with open(ob_path, encoding='utf-8') as f:
            ob = json.load(f)
    return dfs, funds, ob, _W1


def load_data(exclude=None):
    """加载回测数据（带缓存）。universe='U8' 用全币缓存 + onboard.json。"""
    if not _ENGINE_OK:
        raise RuntimeError(MISSING_MSG)
    key = (DEFAULTS['min_len'], DEFAULTS['funding_mode'], DEFAULTS['universe'],
           tuple(exclude or []))
    if key not in _data_cache:
        if DEFAULTS['universe'] == 'U8':
            dfs, funds, ob, w1 = _load_universe_meta()
            if exclude:
                dfs = {b: d for b, d in dfs.items() if b not in exclude}
                funds = {b: f for b, f in funds.items() if b not in exclude}
            _data_cache[key] = (dfs, funds, ob, w1)
        else:
            dfs, funds = _load(min_len=key[0], funding_mode=key[1], exclude=exclude)
            _data_cache[key] = (dfs, funds, {}, None)
    return _data_cache[key]


def _run_backtest(dfs, funds, params, slippage, extra=None):
    """extra: (pool_fn, gate, w1) —— U8 口径的 PIT 池 / MA60 gate / 窗口右端。"""
    kw = dict(
        use_funding=True, exec_lag=1, mode=params['mode'],
        liq_filter='absolute', long_only=params['long_only'],
        leverage=params['leverage'], rebal_n=params['rebal_n'],
        top_frac=params['top_frac'], vol_max=params['vol_max'],
        abs_mom=params['abs_mom'], slippage=slippage,
        liq_min=params['liq_min'],
    )
    if extra is not None:
        pool_fn, gate, w1 = extra
        kw.update(pool_fn=pool_fn, gate=gate,
                  short_weight_mode=params.get('short_weight', 'exp'))
        if w1 is not None:
            kw['t1'] = w1
    eq, nt, avge, legs, tlog, liq_date, min_buffer, audit = _backtest(dfs, funds, **kw)
    return eq, nt, avge, tlog, liq_date, min_buffer, audit


def _trad_stats(audit, eqn):
    avg_eq = float(eqn.mean() * INIT)
    years = (eqn.index[-1] - eqn.index[0]).days / 365.0
    annual_turnover = (audit['turnover_amt'] / avg_eq / years) if (avg_eq > 0 and years > 0) else 0.0
    days = [t['days'] for t in audit['closed_trades'] if t.get('days') is not None]
    return dict(annual_turnover=round(float(annual_turnover), 1),
                avg_hold_days=round(float(np.mean(days)), 1) if days else 0.0)


def _yearly(eqn):
    out = {}
    for y in ('2024', '2025', '2026'):
        a = pd.Timestamp(f'{y}-01-01', tz='UTC')
        b = pd.Timestamp(f'{int(y) + 1}-01-01', tz='UTC')
        seg = eqn[(eqn.index >= a) & (eqn.index < b)]
        if len(seg) == 0:
            out[y] = None
            continue
        prev = eqn[eqn.index < a]
        bn = prev.iloc[-1] if len(prev) else 1.0
        ret = float((seg.iloc[-1] / bn - 1) * 100)
        mdd = float(((seg - seg.cummax()) / seg.cummax() * 100).min())
        dr = seg.pct_change().dropna()
        sh = float(dr.mean() / dr.std() * np.sqrt(365)) if dr.std() > 0 else 0.0
        out[y] = dict(ret=round(ret, 1), mdd=round(mdd, 1), sharpe=round(sh, 2))
    return out


def run(params):
    """执行回测并返回展示数据。params 为与 DEFAULTS 同结构的字典。"""
    use_u8 = (params.get('universe') == 'U8') and _pool_at is not None
    dfs, funds, ob, w1 = load_data()
    extra = None
    if use_u8:
        if not ob:
            # 无 onboard.json → 上市天数不可判 → 回退 fixed 口径
            use_u8 = False
        else:
            from tmp_combo_full import W0 as _W0
            # 调仓网格（每 rebal_n 个交易日）与池构建——与 R7/R28-D 同构
            dates = sorted(set().union(*[set(d.index) for d in dfs.values()]))
            grid = [d for d in dates if _W0 <= d < w1][::params['rebal_n']]
            pools5 = {t: _pool_at(dfs, ob, t, 'U8') for t in grid}
            gate = _build_gate(dfs, pools5, ma_n=60)
            extra = (lambda t: _pool_at(dfs, ob, t, 'U8'), gate, w1)
    eq, nt, avge, tlog, liq_date, min_buffer, audit = _run_backtest(
        dfs, funds, params, params['slippage'], extra=extra)
    eqn = eq / INIT
    ret, ann, mdd, sharpe = _stat_core(eqn)
    net = float(eq.iloc[-1] - INIT)

    # —— 逐币贡献（最终快照，含浮动盈亏；排序取贡献最大的币） ——
    snap = dict(audit['asset_daily'][-1][1])
    ranked = sorted(snap.items(), key=lambda x: -x[1])
    coins = [dict(base=b, pnl=round(float(v), 2),
                  share=round(float(v / net * 100), 1) if net else 0.0)
             for b, v in ranked]
    top1_coin = ranked[0][0] if ranked else None
    top1_share = round(float(ranked[0][1] / net * 100), 1) if (ranked and net) else 0.0
    top5_share = round(float(sum(v for _, v in ranked[:5]) / net * 100), 1) if net else 0.0

    # —— 去 ZEC 稳健性 ——
    dfs_z, funds_z, ob_z, w1_z = load_data(exclude=['ZEC'])
    # U8 口径下排除 ZEC 后需重建池/gate（ZEC 可能影响 ADV30 排名与等权指数）
    extra_z = None
    if extra is not None and ob_z:
        from tmp_combo_full import W0 as _W0
        dates_z = sorted(set().union(*[set(d.index) for d in dfs_z.values()]))
        grid_z = [d for d in dates_z if _W0 <= d < w1_z][::params['rebal_n']]
        pools_z = {t: _pool_at(dfs_z, ob_z, t, 'U8') for t in grid_z}
        extra_z = (lambda t: _pool_at(dfs_z, ob_z, t, 'U8'),
                   _build_gate(dfs_z, pools_z, ma_n=60), w1_z)
    eq_z, _, _, _, _, _, _ = _run_backtest(dfs_z, funds_z, params, params['slippage'],
                                           extra=extra_z)
    drop_zec = _stat_core(eq_z / INIT)[0]

    # —— 滑点梯度（0/10/20/30/40bps，U8 口径下同 extra 池/gate，仅变滑点） ——
    slip_gradient = []
    for tag, bps in SLIP_GRID:
        eq_s, _, _, _, _, _, _ = _run_backtest(dfs, funds, params, bps, extra=extra)
        r = _stat_core(eq_s / INIT)
        slip_gradient.append(dict(tag=tag, ret=r[0], sharpe=r[3], mdd=r[2]))

    # —— 逐月分列（总收益 = 多头 + 空头 + Funding + Fee） ——
    monthly_rows = []
    for m, tot, lg, sg, fund, fee in _monthly_breakdown(audit['leg_daily'], eq):
        monthly_rows.append(dict(month=m, ret=tot, long=lg, short=sg, fund=fund, fee=fee))

    mret = [r['ret'] for r in monthly_rows]
    month_stats = dict(
        win_rate=round(float((np.array(mret) > 0).mean() * 100), 1) if mret else 0.0,
        best=round(float(max(mret)), 1) if mret else 0.0,
        worst=round(float(min(mret)), 1) if mret else 0.0,
        std=round(float(np.std(mret, ddof=1)), 1) if len(mret) > 1 else 0.0,
        worst3=round(float(sum(sorted(mret)[:3])), 1) if mret else 0.0,
    )

    ts = _trad_stats(audit, eqn)
    equity_points = [[int(pd.Timestamp(t).timestamp()), float(v)] for t, v in eq.items()]

    return dict(
        stat=dict(
            ret=ret, cagr=ann, mdd=mdd, sharpe=sharpe,
            trades=int(nt), avg_elig=round(float(avge), 1),
            top1_coin=top1_coin, top1_share=top1_share, top5_share=top5_share,
            drop_zec=drop_zec,
            liq_date=str(liq_date) if liq_date else None,
            buffer=None if min_buffer is None else round(float(min_buffer), 4),
            annual_turnover=ts['annual_turnover'], avg_hold_days=ts['avg_hold_days'],
        ),
        equity_points=equity_points,
        initial=INIT,
        monthly=monthly_rows,
        month_stats=month_stats,
        coins=coins,
        slip_gradient=slip_gradient,
        yearly=_yearly(eqn),
    )
