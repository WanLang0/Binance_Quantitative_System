# -*- coding: utf-8 -*-
"""
MACD + 价格形态(顶底背离) + 共振过滤 —— 固定组合策略信号计算模块。

三种固定组合策略（用户按固定组合单独实现）：
- macd+背离                : MACD金叉/死叉 与 底/顶背离 取并集（背离补充普通信号）
- macd+背离+量能            : 上者基础上 叠加「成交量放大(>1.5倍20期均量)」共振过滤
- macd+背离+均线+量能       : 上者基础上 叠加「收盘价位于20日均线同侧(趋势过滤)」共振过滤
- macd+量能                 : 纯 MACD金叉/死叉 × 放量过滤（无背离，2024-2026回测冠军：
                              4h·双向·ATR动态止盈止损·C判定 = +80.5%/MDD-4.7%/1x）
- macd 12/16/5+量能         : 同框架，MACD 12/16/5 + 1.2x量能（1h稳健型，1.2x口径收益全面优于1.5x）
- macd 12/16/7+量能         : 同框架，MACD 12/16/7 + 1.2x量能（新冠军，1.2x口径更优）

信号构造（无前视口径，与实盘引擎严格一致）：
- 买入 = (MACD金叉 | 底背离) [& 均线过滤] [& 量能过滤]
- 卖出 = (MACD死叉 | 顶背离) [& 均线过滤] [& 量能过滤]
- 背离信号在 pivot 确认根（i+PIVOT_ORDER）标记，非 pivot 当根——修复前视偏差（回测提前5根
  用未来数据入场，实盘引擎读最后一根K线时背离永不成立）。

本模块被 auto_trader / auto_futures / composite_trader 复用，
使实盘引擎的信号与历史回测严格一致，避免"回测赚钱、实盘变样"。
新增信号必须通过 scripts/check_no_lookahead.py 无前视校验。
"""
import numpy as np
import pandas as pd
import ta

from indicators import TechnicalIndicators

# 背离 pivot 用 5+5 根窗口（1h，约半天）
PIVOT_ORDER = 5
# 量能放大倍数（相对 20 期均量）；标准策略用 1.5x，优化变体经 VARIANT_VOL_MULT 单独覆盖
VOL_MULT = 1.5
# MACD 参数（与引擎一致）
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
# 均线周期（共振过滤用）
SMA_PERIOD = 20
VOL_PERIOD = 20

# 固定组合策略名 → (是否启用背离, 是否启用均线过滤, 是否启用量能过滤)
DIVERGENCE_VARIANTS = {
    'macd+背离':             (True,  False, False),
    'macd+背离+量能':         (True,  False, True),
    'macd+背离+均线+量能':    (True,  True,  True),
    'macd+量能':              (False, False, True),
}

# 两种优化参数变体（2024-2026回测，30币均分独立复利池，ATR14 C判定，1x，1.2x量能口径）：
# - 12/16/5：稳健型。信号最密、胜率最高（总量+188.4%/MDD-9.5%），逐年分布平滑
# - 12/16/7：进攻型（新冠军）。总量+231.3%/MDD-8.3%，年度分布最优
# 均为「纯 MACD金叉/死叉 × 1.2倍20期均量过滤」框架，仅 MACD 参数不同
DIVERGENCE_VARIANTS.update({
    'macd 12/16/5+量能': (False, False, True),
    'macd 12/16/7+量能': (False, False, True),
})

# 策略名 → MACD 参数覆盖（默认 MACD_FAST/SLOW/SIGNAL）。新变体用各自回测冠军参数
VARIANT_MACD_PARAMS = {
    'macd 12/16/5+量能': (12, 16, 5),
    'macd 12/16/7+量能': (12, 16, 7),
}

# 策略名 → 量能放大倍数（相对20期均量）。默认 VOL_MULT=1.5 应用于 macd+量能 等标准策略；
# 优化变体 12/16/5、12/16/7 经回测验证 1.2x 口径收益全面优于 1.5x，故单独覆盖为 1.2。
VARIANT_VOL_MULT = {
    'macd 12/16/5+量能': 1.2,
    'macd 12/16/7+量能': 1.2,
}


def _vol_mult(name):
    """返回策略名对应的量能放大倍数；未覆盖时回退全局 VOL_MULT（1.5x）"""
    return VARIANT_VOL_MULT.get(_normalize(name), VOL_MULT)


def _normalize(name):
    """策略名归一化：统一小写并去除空格，便于与 DIVERGENCE_VARIANTS 匹配"""
    return (name or '').strip().lower()


def is_divergence_variant(name):
    """判断一个策略名是否为固定组合背离策略"""
    return _normalize(name) in DIVERGENCE_VARIANTS


def any_divergence_variant(names):
    """判断一个策略名列表（多选组合）中是否包含任一固定组合背离策略"""
    for n in (names or []):
        if is_divergence_variant(n):
            return True
    return False


def find_divergence_name(names):
    """从策略名列表中返回第一个命中的固定组合背离策略名；无则返回 None"""
    for n in (names or []):
        if is_divergence_variant(n):
            return _normalize(n)
    return None


def _find_pivots(s, order=PIVOT_ORDER, kind='high'):
    """返回 bool Series：标记窗口 [i-order, i+order] 内的局部极值点(要求为窗口唯一极值)"""
    arr = s.to_numpy()
    piv = np.zeros(len(arr), dtype=bool)
    for i in range(order, len(arr) - order):
        win = arr[i - order:i + order + 1]
        if kind == 'high':
            piv[i] = np.argmax(win) == order and arr[i] == win.max()
            # 排除平台期(与前一根同高)造成的重复，保证更严谨
            piv[i] = piv[i] and arr[i] > arr[i - 1]
        else:
            piv[i] = np.argmin(win) == order and arr[i] == win.min()
            piv[i] = piv[i] and arr[i] < arr[i - 1]
    return pd.Series(piv, index=s.index)


def _divergence(df, macd_col='MACD', order=PIVOT_ORDER):
    """顶背离/底背离。返回 (top_div, bot_div) 布尔 Series。

    无前视口径：pivot 在 i 处成立需要 [i-order, i+order] 窗口（即 i+order 根收盘后才可知），
    因此背离标志统一右移 order 根、标记在"确认根" i+order 上——
    回测在确认根收盘成交、实盘引擎在最后一根已收盘K线上读到同样的标志，
    两者时间轴严格一致（修复前标志在 pivot 当根 i，回测等于提前 order 根用未来数据抄底摸顶）。"""
    px_high = _find_pivots(df['high'], order, 'high')
    px_low = _find_pivots(df['low'], order, 'low')
    macd = df[macd_col].to_numpy()
    hi = df['high'].to_numpy(); lo = df['low'].to_numpy()
    top = np.zeros(len(df), dtype=bool); bot = np.zeros(len(df), dtype=bool)
    ph = None; pm = None
    for i in np.where(px_high.to_numpy())[0]:
        p = hi[i]; m = macd[i]
        if ph is not None and p > ph and m < pm:   # 价新高 MACD 未新高 → 顶背离
            top[i] = True
        ph = p; pm = m
    pl = None; pm2 = None
    for i in np.where(px_low.to_numpy())[0]:
        p = lo[i]; m = macd[i]
        if pl is not None and p < pl and m > pm2:  # 价新低 MACD 未新低 → 底背离
            bot[i] = True
        pl = p; pm2 = m
    # 关键：右移到确认根（i+order 收盘时 pivot 才被确认），消除前视偏差
    top = np.concatenate([np.zeros(order, dtype=bool), top[:-order]])
    bot = np.concatenate([np.zeros(order, dtype=bool), bot[:-order]])
    return pd.Series(top, index=df.index), pd.Series(bot, index=df.index)


def build_variant_signals(df, use_div, use_ma, use_vol, macd_params=None, vol_mult=None):
    """按变体组合生成 -1/0/1 信号 Series。返回 (df_transformed, signals)
    macd_params: 可选 (fast, slow, signal) 元组，覆盖默认 MACD 参数（优化变体用）
    vol_mult: 可选量能放大倍数（优化变体分别用 1.2x/1.5x）；缺省用全局 VOL_MULT"""
    fast, slow, signal = macd_params or (MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    vol_mult = vol_mult if vol_mult is not None else VOL_MULT
    # 标准 MACD（与 BacktestEngine 口径一致）
    dft = TechnicalIndicators.calculate_macd(df, fast, slow, signal)
    dfd = df.copy()
    dfd['MACD'] = dft['MACD']; dfd['MACD_signal'] = dft['MACD_signal']
    macd_buy = (dfd['MACD'] > dfd['MACD_signal']) & (dfd['MACD'].shift(1) <= dfd['MACD_signal'].shift(1))
    macd_sell = (dfd['MACD'] < dfd['MACD_signal']) & (dfd['MACD'].shift(1) >= dfd['MACD_signal'].shift(1))

    # 共振过滤指标：20日均线 + 20期均量放量
    dfd['sma20'] = ta.trend.SMAIndicator(dfd['close'], window=SMA_PERIOD).sma_indicator()
    dfd['vol_ma20'] = dfd['volume'].rolling(VOL_PERIOD).mean()
    dfd['vol_up'] = dfd['volume'] > dfd['vol_ma20'] * vol_mult

    if use_div:
        top_div, bot_div = _divergence(dfd)
        dfd['top_div'] = top_div; dfd['bot_div'] = bot_div
        buy = macd_buy | bot_div
        sell = macd_sell | top_div
    else:
        buy = macd_buy.copy(); sell = macd_sell.copy()

    # 独立维度共振过滤：均线(价在20日线同侧) + 量能(放量确认)
    close = dfd['close']; sma = dfd['sma20']; vol_up = dfd['vol_up']
    ma_ok_buy = close > sma
    ma_ok_sell = close < sma
    if use_ma:
        buy = buy & ma_ok_buy
        sell = sell & ma_ok_sell
    if use_vol:
        buy = buy & vol_up
        sell = sell & vol_up

    sig = pd.Series(0, index=dfd.index)
    sig[buy] = 1
    sig[sell] = -1
    return dfd, sig


def compute_divergence_signals(df, name):
    """按固定组合策略名计算 -1/0/1 信号 Series。

    Args:
        df: 含 open/high/low/close/volume 列、以时间为索引的 DataFrame
        name: 固定组合策略名（如 'macd+背离+量能'）

    Returns:
        (df_transformed, signals)；若非固定组合背离策略则返回 (None, None)
    """
    flags = DIVERGENCE_VARIANTS.get(_normalize(name))
    if flags is None:
        return None, None
    use_div, use_ma, use_vol = flags
    macd_params = VARIANT_MACD_PARAMS.get(_normalize(name))
    vol_mult = _vol_mult(name)
    return build_variant_signals(df, use_div, use_ma, use_vol, macd_params, vol_mult)
