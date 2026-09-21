# -*- coding: utf-8 -*-
"""跨资产观察器（R32-D / Muon-X 观察期，Research 状态，不参与实盘资金决策）。

每次 Muon 调仓时追加一条 JSONL 记录（data/xasset_observer.jsonl）：
  date, muon_state, qqq_state, qqq_ma50, qqq_ma200, spread, qqq_position,
  muon_ret_30d, qqq_ret_30d, muon_dd, qqq_dd, comb_dd, overlap_dd_risk

设计原则（用户冻结）：
  - 观察 ≠ 执行：本模块只记录，不产生任何仓位变化
  - 失败静默：任何异常不影响交易主流程
  - 数据源：Yahoo QQQ 日线（~24:00 UTC 收盘追加前一日，隔夜数据最新 T-1）
  - 观察项（R34 审计时回答）：①OFF 期 regime 分布 ②whipsaw 实际频率/成本
    ③Muon ON 大跌时 QQQ 是否同步暴露（overlap_dd_risk）
"""
import os
import json
import time
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OBS_FP = os.path.join(DATA_DIR, 'xasset_observer.jsonl')
CACHE_FP = os.path.join(DATA_DIR, 'qqq_daily_cache.csv')

# 与 R32-D 完全一致的信号（冻结口径，不自创参数）
FAST, SLOW = 50, 200


def _fetch_qqq_daily():
    """Yahoo chart API 拉 QQQ 日线（5 年），带本地 CSV 增量缓存。返回 DataFrame(index=UTC date)。"""
    import io

    import pandas as pd

    cached = None
    if os.path.exists(CACHE_FP):
        try:
            cached = pd.read_csv(CACHE_FP, parse_dates=['date']).set_index('date')['close']
        except Exception:
            cached = None

    url = ('https://query1.finance.yahoo.com/v8/finance/chart/QQQ'
           '?range=5y&interval=1d')
    fresh = None
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
        ts = data['chart']['result'][0]['timestamp']
        closes = data['chart']['result'][0]['indicators']['quote'][0]['close']
        fresh = pd.Series(closes, index=pd.to_datetime(ts, unit='s', utc=True).tz_convert(None).normalize(),
                          name='close').dropna()
    except Exception:
        fresh = None

    if fresh is not None and cached is not None:
        merged = pd.concat([cached[~cached.index.isin(fresh.index)], fresh]).sort_index()
    else:
        merged = fresh if fresh is not None else cached
    if merged is None or len(merged) < SLOW + 30:
        return None

    try:
        merged.to_frame().to_csv(CACHE_FP, index_label='date', date_format='%Y-%m-%d')
    except Exception:
        pass
    return merged


def observe(muon_state, muon_dd=None, muon_ret_30d=None, muon_equity=None, now_ts=None):
    """写入一条观察记录。muon_state: True=ON / False=OFF / None=未知。
    返回写入的记录 dict（异常时返回 None）。"""
    try:
        import pandas as pd

        px = _fetch_qqq_daily()
        if px is None:
            return None
        ma50 = px.rolling(FAST, min_periods=FAST).mean()
        ma200 = px.rolling(SLOW, min_periods=SLOW).mean()
        last = px.index[-1]
        f, s, c = float(ma50.iloc[-1]), float(ma200.iloc[-1]), float(px.iloc[-1])
        qqq_state = bool(f > s) if pd.notna(f) and pd.notna(s) else None
        qqq_pos = 'QQQ' if qqq_state else 'Cash'
        spread = (f / s - 1) * 100

        # 30 日收益与回撤
        q30 = px.iloc[-1] / px.iloc[-31] - 1 if len(px) >= 31 else None
        eq = (1 + px.pct_change().fillna(0)).cumprod()
        dd = eq.iloc[-1] / eq.cummax().iloc[-1] - 1

        rec = {
            'ts': int(now_ts if now_ts is not None else time.time()),
            'date': str(last.date()),
            'muon_state': muon_state,
            'qqq_state': qqq_state,
            'qqq_position': qqq_pos,          # R32-D 口径下的 SHOULD 仓位（仅记录，不执行）
            'qqq_close': round(c, 2),
            'qqq_ma50': round(f, 2),
            'qqq_ma200': round(s, 2),
            'ma_spread_pct': round(float(spread), 3),
            'qqq_ret_30d': None if q30 is None else round(q30 * 100, 2),
            'qqq_dd_pct': round(dd * 100, 2),
            'muon_ret_30d': None if muon_ret_30d is None else round(muon_ret_30d * 100, 2),
            'muon_dd_pct': None if muon_dd is None else round(muon_dd * 100, 2),
            'muon_equity': None if muon_equity is None else round(float(muon_equity), 2),
            # 组合视角：Muon ON 期若叠加 QQQ（R32-D 只在 OFF 生效，ON 期 QQQ 恒 0，此字段恒 False；
            # 但若未来 Muon ON 与金叉同时发生且用户手动持有 QQQ，此字段才有意义）
            'overlap_dd_risk': bool(muon_state and qqq_state and (dd is not None) and (dd < -0.10)),
        }
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(OBS_FP, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        return rec
    except Exception:
        return None


def read_records(fp=OBS_FP):
    """读取全部观察记录（观察期审计用）。"""
    if not os.path.exists(fp):
        return []
    out = []
    with open(fp, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def summarize(records=None):
    """观察期汇总：OFF 期 regime 分布 + whipsaw 计数 + overlap 风险计数。"""
    import pandas as pd

    recs = records if records is not None else read_records()
    if not recs:
        return {'n': 0}
    df = pd.DataFrame(recs).drop_duplicates(subset='date', keep='last')
    off = df[df['muon_state'] == False]  # noqa: E712
    return {
        'n': len(df),
        'days': (pd.to_datetime(df['date']).max() - pd.to_datetime(df['date']).min()).days,
        'muon_off_pct': round(len(off) / len(df) * 100, 1),
        'off_golden_pct': round((off['qqq_state'] == True).mean() * 100, 1) if len(off) else None,   # noqa: E712
        'off_death_pct': round((off['qqq_state'] == False).mean() * 100, 1) if len(off) else None,   # noqa: E712
        'qqq_whipsaws': int((df['qqq_state'].astype(float).diff().abs() > 0).sum()),
        'overlap_risk_days': int(df['overlap_dd_risk'].sum()),
        'ma_spread_now': df['ma_spread_pct'].iloc[-1],
    }
