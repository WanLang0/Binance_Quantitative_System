# -*- coding: utf-8 -*-
"""邮件通知模块 — 可选的交易提醒。

设计原则：
  1. 配置驱动 — 读取 data/email_config.json，文件不存在就不发邮件
  2. 安全 — 所有发送 try/except，失败只返回 False 不 raise，绝不阻塞交易主链路
  3. 线程安全 — 用独立 SMTP 连接，短连接（发完就关）
  4. 不保存敏感信息 — 配置文件里的密码/授权码由用户自己管理，本模块不做任何额外持久化

配置文件 data/email_config.json 格式：
{
  "enabled": true,
  "smtp_host": "smtp.qq.com",
  "smtp_port": 465,
  "use_ssl": true,
  "username": "your_email@qq.com",
  "password": "your_smtp_auth_code",
  "from_addr": "your_email@qq.com",
  "to_addrs": ["recipient1@qq.com", "recipient2@163.com"],
  "subject_prefix": "[V2量化]"
}

注意：password 是 SMTP 专用授权码（QQ邮箱/163邮箱在设置里开启 SMTP 后生成），不是邮箱登录密码。
"""
import json
import os
import smtplib
import ssl
import traceback
from email.mime.text import MIMEText
from email.header import Header
from datetime import datetime

CONFIG_PATH = os.path.join('data', 'email_config.json')


class EmailNotifier:
    """邮件通知器。构造时读取配置；send_* 方法返回 bool，调用方可忽略。"""

    def __init__(self, config_path=None):
        self.enabled = False
        self.cfg = None
        path = config_path or CONFIG_PATH
        try:
            if not os.path.exists(path):
                return
            with open(path, 'r', encoding='utf-8') as f:
                self.cfg = json.load(f)
            self.enabled = bool(self.cfg.get('enabled', False))
            if not self.cfg.get('smtp_host') or not self.cfg.get('username'):
                self.enabled = False
        except Exception:
            self.enabled = False
            self.cfg = None

    # ---------- 底层发送 ----------
    def _send(self, subject, body):
        """发一封邮件。成功返回 True，任何异常返回 False（不 raise）。"""
        if not self.enabled or not self.cfg:
            return False
        try:
            msg = MIMEText(body, 'plain', 'utf-8')
            subj = f"{self.cfg.get('subject_prefix', '')}{subject}".strip()
            msg['Subject'] = Header(subj, 'utf-8')
            msg['From'] = self.cfg['from_addr']
            msg['To'] = ', '.join(self.cfg['to_addrs'])

            host = self.cfg['smtp_host']
            port = int(self.cfg.get('smtp_port', 465))
            use_ssl = bool(self.cfg.get('use_ssl', True))
            username = self.cfg['username']
            password = self.cfg['password']

            if use_ssl:
                ctx = ssl.create_default_context()
                with smtplib.SMTP_SSL(host, port, context=ctx, timeout=15) as server:
                    server.login(username, password)
                    server.sendmail(self.cfg['from_addr'], self.cfg['to_addrs'], msg.as_string())
            else:
                with smtplib.SMTP(host, port, timeout=15) as server:
                    server.starttls()
                    server.login(username, password)
                    server.sendmail(self.cfg['from_addr'], self.cfg['to_addrs'], msg.as_string())
            return True
        except Exception:
            # 邮件失败不应该影响交易，静默处理（调用方可记录日志）
            return False

    # ---------- 高层接口 ----------
    def notify_rebalance(self, task_name, gate_on, longs, shorts, symbols, equity, total_fund,
                         realized_pnl, n_elig, dispersion):
        """调仓汇总邮件 — 主要触发点，每 5 天一封。"""
        if not self.enabled:
            return False
        ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
        pnl_pct = (equity / total_fund - 1) * 100 if total_fund > 0 else 0.0

        lines = [
            f"【调仓通知】{task_name}",
            f"时间: {ts}",
            f"",
            f"=== 闸门状态 ===",
            f"Gate: {'ON ✅ (正常交易)' if gate_on else 'OFF ❌ (全部空仓)'}",
            f"",
            f"=== 信号 ===",
            f"候选数: {n_elig}",
            f"离散度: {dispersion:.4f}",
            f"多头 ({len(longs)}): {', '.join(longs) if longs else '(空)'}",
            f"空头 ({len(shorts)}): {', '.join(shorts) if shorts else '(空)'}",
            f"",
            f"=== 当前持仓 ===",
        ]
        for s in symbols:
            base = s['symbol'].split('/')[0]
            pos = s.get('position', 0) or 0
            if pos > 0:
                side = '多' if s['side'] == 'long' else '空'
                ep = s.get('entry_price', 0)
                lp = s.get('last_price', 0)
                pnl = s.get('unrealized_pnl', 0) or 0
                pct = s.get('pnl_pct', 0) or 0
                lines.append(f"  {base} {side} {pos}张 @ 开仓{ep:.6f} / 现价{lp:.6f}  浮盈亏{pnl:+.2f}U ({pct:+.2f}%)")
        lines.append(f"")
        lines.append(f"=== 权益 ===")
        lines.append(f"账户权益: {equity:.2f}U (本金{total_fund:.2f}U, 总收益率{pnl_pct:+.2f}%)")
        lines.append(f"累计已实现盈亏: {realized_pnl:+.2f}U")

        return self._send(f"调仓完成 · Gate{'ON' if gate_on else 'OFF'} · 权益{pnl_pct:+.2f}%",
                          '\n'.join(lines))

    def notify_gate_change(self, task_name, gate_on, gate_index=None, gate_ma=None):
        """Gate 状态切换告警 — Gate ON↔OFF 变化时发。"""
        if not self.enabled:
            return False
        ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
        status = "OFF → 全部空仓 ⚠️" if not gate_on else "ON → 恢复开仓 ✅"
        body = (
            f"【闸门切换】{task_name}\n"
            f"时间: {ts}\n"
            f"状态: {status}\n"
            f"等权指数: {gate_index}\n"
            f"MA{60}: {gate_ma}\n"
        )
        return self._send(f"闸门切换 · {status}", body)

    def notify_alert(self, task_name, title, detail):
        """通用告警（连续错误、MDD 超阈值等）。"""
        if not self.enabled:
            return False
        ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
        body = f"【告警】{task_name}\n时间: {ts}\n\n{title}\n{detail}\n"
        return self._send(f"⚠️ {title}", body)

    def notify_start(self, task_name, total_fund, universe_mode, ma_gate, testnet):
        """任务启动通知 — 方便确认配置正确。"""
        if not self.enabled:
            return False
        ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
        body = (
            f"【任务启动】{task_name}\n"
            f"时间: {ts}\n"
            f"初始资金: {total_fund:.2f}U\n"
            f"宇宙模式: {universe_mode}\n"
            f"闸门: MA{ma_gate}\n"
            f"网络: {'测试网' if testnet else '主网'}\n"
            f"邮件通知已启用 ✅\n"
        )
        return self._send(f"🚀 任务启动 · {task_name}", body)


if __name__ == '__main__':
    # 快速自测 — 需要 data/email_config.json 存在
    n = EmailNotifier()
    if n.enabled:
        ok = n._send("[测试] V2 邮件通知自测",
                     "这是一封来自 momentum_live_trader 的测试邮件。\n"
                     f"时间: {datetime.utcnow().isoformat()} UTC\n"
                     "如果你收到了，说明 SMTP 配置正确，可以在实盘里用了。")
        print(f"发送结果: {'成功 ✅' if ok else '失败 ❌'}")
    else:
        print("邮件未启用 — data/email_config.json 不存在或 enabled=false")
