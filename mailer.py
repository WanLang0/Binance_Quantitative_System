# -*- coding: utf-8 -*-
"""邮件告警模块：QQ邮箱 SMTP（自发自收），供引擎异常时离线提醒

用法:
    mailer.send_async(subject, body)   # 后台线程发送，绝不阻塞调用方（引擎主循环用）
    mailer.send_email(subject, body)   # 同步发送，返回 (是否成功, 提示)（测试按钮用）
"""
import smtplib
import threading
from email.header import Header
from email.mime.text import MIMEText

import auth

SMTP_HOST = 'smtp.qq.com'
SMTP_PORT = 465  # SSL
TIMEOUT = 15     # 网络不通时避免长时间挂起


def send_email(subject, body):
    """同步发送告警邮件到已配置的QQ邮箱，返回 (是否成功, 提示)"""
    account = auth.get_any_email_account()
    if not account:
        return False, '未配置邮箱或授权码'
    email, auth_code = account
    msg = MIMEText(body, 'plain', 'utf-8')
    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = email
    msg['To'] = email
    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=TIMEOUT) as s:
            s.login(email, auth_code)
            s.sendmail(email, [email], msg.as_string())
        return True, f'已发送至 {email}'
    except Exception as e:
        return False, f'发送失败: {type(e).__name__}: {e}'


def send_async(subject, body):
    """守护线程发送（引擎故障告警用）：网络不通时最多阻塞 TIMEOUT 秒且不影响交易循环"""
    def _worker():
        try:
            ok, tip = send_email(subject, body)
            print(f"[mailer] {'OK' if ok else 'FAIL'} {subject} -> {tip}")
        except Exception as e:
            print(f"[mailer] 异常: {e}")
    threading.Thread(target=_worker, daemon=True).start()


def is_configured():
    return auth.get_any_email_account() is not None
