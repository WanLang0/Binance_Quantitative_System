# -*- coding: utf-8 -*-
"""定时持仓报告邮件模块（默认每天 18:30 自动发送）

功能：在设定时刻，把「此时此刻的真实持仓情况」通过邮件发送：
- 各账户（按运行中引擎的交易器去重分组）的合约持仓：
  币种、方向、数量、开仓均价、标记价、未实现盈亏、收益率、强平价
- 账户 USDT 权益/可用/占用保证金
- 运行中的量化任务列表概览

持仓数据由 app.py 注入的 positions_provider 提供（查询运行中引擎的
trader，即交易所真实持仓，非内存快照）；邮件经 mailer.send_async 发送
（QQ SMTP，HTML 样式，与成交通知邮件同风格）。
"""
import os
import json
import threading
import time
import logging
from datetime import datetime, timedelta

logger = logging.getLogger('daily_signal_report')

# 数据根目录（磁盘状态/配置）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, 'data', 'daily_signal_report.json')
# 发送互斥锁：串行化「调度器定时发送」与「设置页立即发送」，避免并发双发邮件。
# 只在本模块内部使用、单锁无嵌套，不存在死锁路径。
_send_lock = threading.Lock()
_mailer = None
# 持仓快照数据源（app.py 启动时注入；返回 {'groups': [...], 'task_count': int}）
_positions_provider = None


# ---------- 依赖注入（避免循环 import，由 app.py 在启动时注入） ----------
def _inject(mailer_module, auth_module=None, positions_provider=None):
    """注入 mailer 模块与持仓快照数据源（app.py 启动时调用一次）"""
    global _mailer, _positions_provider
    _mailer = mailer_module
    if positions_provider:
        _positions_provider = positions_provider


def _get_mailer():
    global _mailer
    if _mailer is None:
        import mailer as m
        _mailer = m
    return _mailer


# ---------- 邮件正文（HTML） ----------
def build_positions_html(snapshot):
    """组装持仓报告邮件正文（HTML，样式与成交通知邮件一致：多=绿/空=红）"""
    GREEN, RED, TXT, MUTED, BORDER, BG = '#16a34a', '#dc2626', '#1e293b', '#8a94a6', '#e3e8ef', '#f7f9fc'
    groups = (snapshot or {}).get('groups') or []
    task_count = (snapshot or {}).get('task_count') or 0
    now_txt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def pnl_color(v):
        return GREEN if v > 0 else (RED if v < 0 else MUTED)

    def fmt(v, nd=2):
        try:
            return f"{float(v):,.{nd}f}"
        except (TypeError, ValueError):
            return '—'

    html = f"""
<div style="font-family:-apple-system,'Segoe UI','Microsoft YaHei',sans-serif;max-width:640px;margin:0 auto;border:1px solid {BORDER};border-radius:10px;overflow:hidden">
  <div style="background:#1e2530;padding:14px 18px">
    <div style="color:#fff;font-size:17px;font-weight:600">💰 币安量化 · 持仓报告</div>
    <div style="color:#aab4c4;font-size:12px;margin-top:4px">⏰ {now_txt} · 运行中任务 {task_count} 个</div>
  </div>
  <div style="padding:16px 18px;background:#fff">"""

    if not groups:
        html += f"""
    <div style="text-align:center;padding:26px 10px;color:{MUTED};font-size:14px">
      当前无运行中的量化任务，无持仓可报。<br>启动综合量化/自动合约任务后，此处将展示账户真实持仓。
    </div>"""
    else:
        for g in groups:
            tasks = '、'.join(g.get('tasks') or []) or '（无任务名）'
            network = g.get('network') or ''
            usdt = g.get('usdt') or {}
            positions = g.get('positions') or []
            total_upnl = sum(float(p.get('unrealized_pnl') or 0) for p in positions)
            up_color = pnl_color(total_upnl)
            html += f"""
    <div style="border:1px solid {BORDER};border-radius:8px;overflow:hidden;margin-bottom:14px">
      <div style="background:{BG};padding:10px 14px;border-bottom:1px solid {BORDER}">
        <b style="color:{TXT};font-size:14px">{tasks}</b>
        <span style="color:{MUTED};font-size:12px;margin-left:8px">{'· ' + network if network else ''} · 持仓 {len(positions)} 个</span>
      </div>
      <div style="padding:10px 14px;font-size:13px;color:{TXT}">
        <div>USDT 权益 <b>{fmt(usdt.get('total'))}</b>
             <span style="color:{MUTED}">| 可用 {fmt(usdt.get('free'))}</span>
             <span style="color:{MUTED}">| 占用保证金 {fmt(usdt.get('used'))}</span>
             <span style="margin-left:10px">未实现盈亏
               <b style="color:{up_color}">{total_upnl:+,.2f} U</b></span>
      </div>"""
            if g.get('err'):
                html += f"""
      <div style="padding:6px 14px 10px;font-size:12px;color:{RED}">⚠ 持仓查询失败：{g['err']}</div>"""
            if positions:
                head = (f"<tr style='background:{BG};color:{MUTED};font-size:12px'>"
                        f"<td style='padding:6px 10px'>币种</td><td style='padding:6px 10px'>方向</td>"
                        f"<td style='padding:6px 10px'>数量</td><td style='padding:6px 10px'>开仓均价</td>"
                        f"<td style='padding:6px 10px'>标记价</td><td style='padding:6px 10px'>未实现盈亏</td>"
                        f"<td style='padding:6px 10px'>收益率</td><td style='padding:6px 10px'>强平价</td></tr>")
                rows = [head]
                for p in positions:
                    side = p.get('side') or '?'
                    side_txt = '做多' if side == 'long' else ('做空' if side == 'short' else side)
                    side_color = GREEN if side == 'long' else (RED if side == 'short' else TXT)
                    upnl = float(p.get('unrealized_pnl') or 0)
                    ppct = float(p.get('pnl_pct') or 0)
                    rows.append(
                        f"<tr style='border-bottom:1px solid {BORDER};font-size:13px'>"
                        f"<td style='padding:7px 10px;font-weight:600'>{(p.get('symbol_base') or p.get('symbol') or '?').split('/')[0]}</td>"
                        f"<td style='padding:7px 10px;color:{side_color};font-weight:600'>{side_txt}</td>"
                        f"<td style='padding:7px 10px'>{fmt(p.get('contracts'), 6).rstrip('0').rstrip('.') or '0'}</td>"
                        f"<td style='padding:7px 10px'>{fmt(p.get('entry_price'), 4)}</td>"
                        f"<td style='padding:7px 10px'>{fmt(p.get('mark_price'), 4)}</td>"
                        f"<td style='padding:7px 10px;color:{pnl_color(upnl)};font-weight:600'>{upnl:+,.2f} U</td>"
                        f"<td style='padding:7px 10px;color:{pnl_color(ppct)}'>{ppct:+.2f}%</td>"
                        f"<td style='padding:7px 10px;color:{MUTED}'>{fmt(p.get('liquidation_price'), 4)}</td></tr>")
                html += (f"<table style='width:100%;border-collapse:collapse;text-align:left'>"
                         + ''.join(rows) + "</table>")
            else:
                html += f"""
      <div style="text-align:center;padding:14px;color:{MUTED};font-size:13px">当前空仓 🎉</div>"""
            html += """
    </div>"""
    html += f"""
  </div>
  <div style="background:#f7f9fc;padding:10px 18px;color:#8a94a6;font-size:12px;text-align:center">
    —— 本邮件由币安量化系统定时发送（数据为发送时刻的交易所真实持仓）——
  </div>
</div>"""
    return html


def _report_subject(snapshot):
    """邮件标题：简短汇总（N仓 + 盈亏）"""
    groups = (snapshot or {}).get('groups') or []
    positions = [p for g in groups for p in (g.get('positions') or [])]
    if not groups:
        return "💰币安量化持仓：无运行任务"
    if not positions:
        return "💰币安量化持仓：空仓"
    total = sum(float(p.get('unrealized_pnl') or 0) for p in positions)
    return f"💰币安量化持仓：{len(positions)}仓 {total:+.2f}U"


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
        logger.warning(f"保存报告状态失败: {e}")


def get_config():
    """读取报告开关配置，返回 {enabled, time, last_sent_date, last_total_signals}(持仓数)"""
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
    """更新报告开关配置，返回 (是否成功, 提示)。
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
    return True, f'持仓报告已{"开启" if cfg["enabled"] else "关闭"}，发送时间 {cfg["time"]}'


def _has_sent_today(state, day):
    return state.get('last_sent_date') == day


def run_daily_report(force=False):
    """执行一次持仓快照并发送邮件（开关关闭时跳过；若当天已发且非 force 则跳过）。
    发送全程持模块级互斥锁：防止「调度器触发」与「设置页立即发送」并发导致重复发信。
    返回 (是否发送, 提示)"""
    if not _send_lock.acquire(blocking=False):
        return False, '报告正在发送中，请稍候（避免重复发信）'
    try:
        cfg = get_config()
        if not cfg.get('enabled') and not force:
            return False, '定时报告开关未开启'
        mailer = _get_mailer()
        if not mailer:
            return False, 'mailer 未初始化'
        if not mailer.is_configured():
            return False, '未配置邮箱（设置页绑定 QQ 邮箱后方可发送）'

        today = datetime.now().strftime('%Y-%m-%d')
        state = _load_state()
        if not force and _has_sent_today(state, today):
            return False, f'{today} 报告已发送，跳过'

        # 此时此刻的持仓快照（app.py 注入的 provider；未注入按无任务处理）
        snapshot = {'groups': [], 'task_count': 0}
        if _positions_provider:
            try:
                snapshot = _positions_provider() or snapshot
            except Exception as e:
                logger.warning(f"持仓快照获取失败: {e}")
                snapshot = {'groups': [], 'task_count': 0}

        body = build_positions_html(snapshot)
        subject = _report_subject(snapshot)
        ok, tip = mailer.send_email(subject, body, html=True)
        if ok:
            pos_count = sum(len(g.get('positions') or []) for g in (snapshot.get('groups') or []))
            _save_state({**state, 'last_sent_date': today, 'total_signals': pos_count})
        return ok, tip
    finally:
        _send_lock.release()


# ---------- 调度器（守护线程，按配置时间触发） ----------
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
            # 到达触发时间，执行报告（force 关闭时同一天自动幂等）
            ok, tip = run_daily_report(force=False)
            logger.info(f"定时持仓报告执行: {tip}")
            time.sleep(60)  # 触发后短暂歇息，防止时间边界抖动重复触发
        except Exception as e:
            logger.warning(f"报告调度异常: {e}")
            time.sleep(60)


def start_scheduler():
    """启动守护调度线程（app.py 启动时调用一次）"""
    t = threading.Thread(target=_scheduler_loop, daemon=True, name='daily-signal-report')
    t.start()
    return t
