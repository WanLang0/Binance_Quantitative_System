# -*- coding: utf-8 -*-
"""每日虚拟币信号/异动汇总邮件模块（每天 18:30 自动发送）

功能：在每天 18:30，对「市值前20虚拟币」用固定的 MACD 链复组合策略
（macd+背离+量能 等，与回测/实盘引擎同口径）扫描当日信号并汇总：
- 当日触发的买点/卖点信号（按 1h 已收盘K线判定）
- 各币最近 24h 涨跌幅（异动榜单）
- 无信号/无变化也会发一封「今日无信号」邮件

与主交易引擎完全解耦：不操作 API 下单，只用 ccxt.binanceusdm 拉公开K线，
通过 .env 配置的代理访问币安；邮件经 mailer.send_async 发送（同样走 QQ SMTP）。
"""
import os
import json
import threading
import time
import logging
from datetime import datetime, timedelta

import pandas as pd

from divergence_signals import compute_divergence_signals, find_divergence_name

logger = logging.getLogger('daily_signal_report')

# 数据根目录（磁盘状态/配置）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, 'data', 'daily_signal_report.json')
# 发送互斥锁：串行化「调度器定时发送」与「设置页立即发送」，避免并发双发邮件。
# 只在本模块内部使用、单锁无嵌套，不存在死锁路径。
_send_lock = threading.Lock()
_mailer = None


# ---------- 依赖注入（避免循环 import，由 app.py 在启动时注入） ----------
def _inject(mailer_module, auth_module):
    """注入 mailer / auth 模块（app.py 启动时调用一次）"""
    global _mailer
    _mailer = mailer_module


def _get_mailer():
    global _mailer
    if _mailer is None:
        import mailer as m
        _mailer = m
    return _mailer


# 市值前20虚拟币（与综合量化 CRYPTO_COMPOSITE_SYMBOLS 前20一致，CoinGecko 按市值剔除稳定币）
TOP20_SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'SOL/USDT',
    'TRX/USDT', 'HYPE/USDT', 'ZEC/USDT', 'DOGE/USDT', 'XMR/USDT',
    'LINK/USDT', 'ADA/USDT', 'XLM/USDT', 'BCH/USDT', 'CC/USDT',
    'LTC/USDT', 'UNI/USDT', 'GRAM/USDT', 'HBAR/USDT', 'AVAX/USDT',
]
NAME_MAP = {
    'BTC': '比特币', 'ETH': '以太坊', 'BNB': '币安币', 'XRP': '瑞波币', 'SOL': 'Solana',
    'TRX': '波场', 'HYPE': 'Hyperliquid', 'ZEC': 'Zcash', 'DOGE': '狗狗币', 'XMR': '门罗币',
    'LINK': 'Chainlink', 'ADA': '艾达币', 'XLM': '恒星币', 'BCH': '比特现金', 'CC': 'Canton',
    'LTC': '莱特币', 'UNI': 'Uniswap', 'GRAM': 'Gram', 'HBAR': 'Hedera', 'AVAX': '雪崩',
}
# 固定组合策略（与回测一致）；优先用「macd+背离+量能」作为主扫描策略
SCAN_STRATEGIES = ['macd+背离+量能', 'macd+背离', 'macd+背离+均线+量能']


# ---------- ccxt 数据源（合约 K 线，与回测口径一致） ----------
def _load_env_proxy():
    """读取 .env 中的 HTTP_PROXY/HTTPS_PROXY（与 app.py 的 _load_env_file 同口径）"""
    env_path = os.path.join(BASE_DIR, '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, val = line.partition('=')
                key, val = key.strip(), val.strip().strip('"').strip("'")
                if val and not os.environ.get(key):
                    os.environ[key] = val
    return (os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy') or None,
            os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy') or None)


class _Exchange:
    """极简 ccxt 合约行情拉取器（只读公开K线/价格，不触碰私钥/下单）"""

    def __init__(self):
        import ccxt
        http_proxy, https_proxy = _load_env_proxy()
        config = {
            'enableRateLimit': True,
            'timeout': 15000,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
            },
        }
        proxy = https_proxy or http_proxy
        if proxy:
            config['proxies'] = {'http': proxy, 'https': proxy}
        self._ex = ccxt.binanceusdm(config)
        self._ex.options['fetchCurrencies'] = False
        self._ex.options['warnOnFetchOpenOrdersWithoutSymbol'] = False

    def get_ohlcv(self, symbol, timeframe='1h', limit=300):
        """拉取最近 limit 根已收盘K线（丢弃进行中的未收盘K线）
        返回 [{ts, open, high, low, close, volume}...] 或 (None, err)"""
        try:
            candles = self._ex.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not candles:
                return None, "无K线数据"
            tf_ms = {'1m': 60_000, '3m': 180_000, '5m': 300_000, '15m': 900_000,
                     '30m': 1_800_000, '1h': 3_600_000, '2h': 7_200_000,
                     '4h': 14_400_000, '6h': 21_600_000, '12h': 43_200_000,
                     '1d': 86_400_000}.get(timeframe, 0)
            if tf_ms and candles[-1][0] + tf_ms > int(time.time() * 1000):
                candles = candles[:-1]  # 丢弃未收盘的进行中K线
            if not candles:
                return None, "无K线数据"
            data = [{
                'ts': int(c[0]), 'open': float(c[1]), 'high': float(c[2]),
                'low': float(c[3]), 'close': float(c[4]), 'volume': float(c[5]),
            } for c in candles]
            return data, None
        except Exception as e:
            return None, str(e)

    def get_ticker(self, symbol):
        """获取最新价 + 24h 涨跌幅，返回 (price, change_pct, err)。失败返回 (None, None, err)"""
        try:
            t = self._ex.fetch_ticker(symbol)
            price = t.get('last')
            change = t.get('percentage')  # 已乘100的百分数（如 +2.5）
            if change is None and t.get('close') and t.get('open'):
                change = (t['close'] - t['open']) / t['open'] * 100
            return price, change, None
        except Exception as e:
            return None, None, str(e)


# ---------- 信号扫描 ----------
def _to_df(candles):
    """K线列表 → 含 open/high/low/close/volume、按时间索引的 DataFrame（供 compute_divergence_signals）"""
    if not candles:
        return None
    df = pd.DataFrame(candles)
    df['timestamp'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.set_index('timestamp')
    return df


def scan_symbol(ex, symbol, strategies, timeframe='1h'):
    """扫描单一币种当日信号。
    返回 {'symbol','name','price','change_pct','signals':[{strategy,dt,dir}...],'err'}"""
    name = NAME_MAP.get(symbol.split('/')[0], symbol.split('/')[0])
    price, change, terr = ex.get_ticker(symbol)
    candles, cerr = ex.get_ohlcv(symbol, timeframe=timeframe, limit=300)
    if cerr or not candles:
        return {'symbol': symbol, 'name': name, 'price': price, 'change_pct': change,
                'signals': [], 'err': cerr or '无K线'}
    df = _to_df(candles)
    if df is None or df.empty:
        return {'symbol': symbol, 'name': name, 'price': price, 'change_pct': change,
                'signals': [], 'err': '无K线'}

    # 当日 00:00 边界：把「本地自然日」起始转换为 UTC 时间戳再与K线(UTC)比较，
    # 避免本地时区(如东八区)与交易所 UTC 的偏移导致切掉/漏掉当日信号。
    # 方式：取本地当日0点，用 time.mktime 解析为本地时间戳，再减去本地时区偏移得到UTC。
    now_local = datetime.now()
    day_start_local = datetime(now_local.year, now_local.month, now_local.day, 0, 0, 0)
    # 本地0点对应的 UTC 时间戳（本地是UTC+N，则 UTC_ts = local_ts - N*3600）
    day_start_utc_ts = time.mktime(day_start_local.timetuple()) - now_local.tzinfo.utcoffset(now_local).total_seconds() if now_local.tzinfo else time.mktime(day_start_local.timetuple())
    day_start = datetime.utcfromtimestamp(day_start_utc_ts)

    all_signals = []
    for strat in strategies:
        dflag = find_divergence_name([strat])
        if not dflag:
            continue
        _dft, sig = compute_divergence_signals(df, dflag)
        if sig is None or sig.empty:
            continue
        # 遍历当日范围内的信号（sig 与 df 索引对齐，时间为 UTC）
        for ts, v in sig.items():
            if v == 0:
                continue
            if ts.tzinfo:
                ts_naive = ts.tz_localize(None)
            else:
                ts_naive = ts
            if ts_naive >= day_start:
                all_signals.append({
                    'strategy': strat,
                    'dt': ts_naive.strftime('%H:%M'),
                    'dir': '买' if v == 1 else '卖',
                })
    return {'symbol': symbol, 'name': name, 'price': price, 'change_pct': change,
            'signals': all_signals, 'err': None}


# ---------- 邮件正文 ----------
def build_report_body(day, rows, total_signals, err_count):
    """组装邮件正文（纯文本）"""
    lines = []
    lines.append(f"📧 币安量化 · 虚拟币信号/异动日报")
    lines.append(f"📅 日期：{day}")
    lines.append(f"⏰ 生成：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("【本日 MACD 链复信号汇总】")
    if total_signals == 0:
        lines.append("今日未触发任何买点/卖点信号，市场相对平静。")
    else:
        lines.append(f"今日共命中 {total_signals} 次信号：")
    for r in rows:
        sigs = r.get('signals') or []
        change = r.get('change_pct')
        change_txt = f"{change:+.2f}%" if change is not None else "—"
        line = f"• {r['symbol']}（{r['name']}） 24h: {change_txt}"
        if r.get('err'):
            line += "（数据异常）"
        if sigs:
            detail = ", ".join(f"{s['strategy']}:{s['dir']}@{s['dt']}点" for s in sigs)
            line += f"  信号[ {detail} ]"
        else:
            line += ""
        lines.append(line)
    lines.append("")
    if err_count:
        lines.append(f"⚠ 有 {err_count} 个币种数据获取失败，已跳过。")
    lines.append("")
    lines.append("—— 本邮件由币安量化系统每日自动发送 ——")
    return "\n".join(lines)


# ---------- 状态持久化（幂等：同一天内不重复发送） ----------
# 默认配置：开关关闭 + 每天 18:30 发送
DEFAULT_CONFIG = {'enabled': False, 'time': '18:30'}


def _load_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_state(state):
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"保存日报状态失败: {e}")


def get_config():
    """读取日报开关配置，返回 {enabled, time, last_sent_date, last_total_signals}"""
    state = _load_state()
    cfg = dict(DEFAULT_CONFIG)
    saved = state.get('config') or {}
    if isinstance(saved, dict):
        cfg.update(saved)
    # 兼容旧状态文件：若顶层已有 enabled/time 字段则迁移读取
    cfg['enabled'] = bool(saved.get('enabled', state.get('enabled', cfg['enabled'])))
    cfg['time'] = saved.get('time', state.get('time', cfg['time']))
    cfg.setdefault('last_sent_date', state.get('last_sent_date'))
    cfg.setdefault('last_total_signals', state.get('total_signals'))
    return cfg


def set_config(enabled=None, report_time=None):
    """更新日报开关配置，返回 (是否成功, 提示)。
    enabled: True/False；report_time: 'HH:MM' 24小时制"""
    state = _load_state()
    cfg = dict(DEFAULT_CONFIG)
    existing = state.get('config') or {}
    if isinstance(existing, dict):
        cfg.update(existing)
    if enabled is not None:
        cfg['enabled'] = bool(enabled)
    if report_time is not None:
        report_time = (report_time or '').strip()
        if not report_time:
            return False, '发送时间不能为空'
        try:
            hh, mm = report_time.split(':')
            hh, mm = int(hh), int(mm)
            if not (0 <= hh <= 23 and 0 <= mm <= 59):
                raise ValueError
        except (ValueError, TypeError):
            return False, '时间格式应为 HH:MM（24小时制），例如 18:30'
        cfg['time'] = f"{hh:02d}:{mm:02d}"
    state['config'] = cfg
    _save_state(state)
    return True, f'日报已{"开启" if cfg["enabled"] else "关闭"}，发送时间 {cfg["time"]}'


def _has_sent_today(state, day):
    return state.get('last_sent_date') == day


def run_daily_report(force=False):
    """执行一次当日信号扫描并发送邮件（开关关闭时跳过；若当天已发且非 force 则跳过）。
    发送全程持模块级互斥锁：防止「调度器触发」与「设置页立即发送」并发导致重复发信。
    返回 (是否发送, 提示)"""
    if not _send_lock.acquire(blocking=False):
        return False, '日报正在发送中，请稍候（避免重复发信）'
    try:
        cfg = get_config()
        if not cfg.get('enabled') and not force:
            return False, '每日日报开关未开启'
        mailer = _get_mailer()
        if not mailer:
            return False, 'mailer 未初始化'
        if not mailer.is_configured():
            return False, '未配置邮箱（设置页绑定 QQ 邮箱后方可发送）'

        today = datetime.now().strftime('%Y-%m-%d')
        state = _load_state()
        if not force and _has_sent_today(state, today):
            return False, f'{today} 日报已发送，跳过'

        ex = _Exchange()
        rows, total_signals, err_count = [], 0, 0
        for sym in TOP20_SYMBOLS:
            try:
                r = scan_symbol(ex, sym, SCAN_STRATEGIES)
            except Exception as e:
                r = {'symbol': sym, 'name': NAME_MAP.get(sym.split('/')[0], sym),
                     'price': None, 'change_pct': None, 'signals': [], 'err': str(e)}
            if r.get('err'):
                err_count += 1
            total_signals += len(r.get('signals') or [])
            rows.append(r)

        body = build_report_body(today, rows, total_signals, err_count)
        subject = f"📧 币安量化 · 虚拟币信号/异动日报 {today}"
        ok, tip = mailer.send_email(subject, body)
        if ok:
            _save_state({**state, 'last_sent_date': today, 'total_signals': total_signals})
        return ok, tip
    finally:
        _send_lock.release()


# ---------- 调度器（守护线程，每天 18:30 触发） ----------
def _parse_time_str(tstr):
    """解析 'HH:MM' 为 (hour, minute)，非法返回 None"""
    try:
        hh, mm = (tstr or '').split(':')
        hh, mm = int(hh), int(mm)
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return hh, mm
    except (ValueError, TypeError):
        pass
    return None


def _next_trigger_dt(now=None, tstr='18:30'):
    """计算下一个指定时间(hh:mm)触发时刻（若今天已过该时刻则顺延到明天）"""
    now = now or datetime.now()
    hm = _parse_time_str(tstr)
    if hm is None:
        hm = (18, 30)
    target = now.replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)
    return target


def _scheduler_loop():
    # 分段睡眠（每段最多60s）：中途修改发送时间/开关可在1分钟内生效，
    # 避免「一觉睡到旧触发时刻」导致改完时间仍在旧时间发送。
    while True:
        try:
            now = datetime.now()
            cfg = get_config()
            if not cfg.get('enabled'):
                time.sleep(60)  # 未开启：轻量轮询配置
                continue
            nxt = _next_trigger_dt(now, cfg.get('time', '18:30'))
            sleep_s = (nxt - now).total_seconds()
            if sleep_s > 0:
                time.sleep(min(sleep_s, 60))  # 分段睡，醒来重读配置重算触发点
                continue
            # 到达触发时间，执行日报（force 关闭时同一天自动幂等）
            ok, tip = run_daily_report(force=False)
            logger.info(f"每日日报执行: {tip}")
            time.sleep(60)  # 触发后短暂歇息，防止时间边界抖动重复触发
        except Exception as e:
            logger.warning(f"日报调度异常: {e}")
            time.sleep(60)


def start_scheduler():
    """启动守护调度线程（app.py 启动时调用一次）"""
    t = threading.Thread(target=_scheduler_loop, daemon=True, name='daily-signal-report')
    t.start()
    return t
