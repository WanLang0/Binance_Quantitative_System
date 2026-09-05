# -*- coding: utf-8 -*-
"""生成三种场景的HTML成交邮件预览（本地文件，不发送），供浏览器查看效果"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch
import mailer
from composite_trader import CompositeTrader

captured = []
with patch.object(mailer, 'is_configured', return_value=True), \
     patch.object(mailer, 'send_async', side_effect=lambda s, b, html=False: captured.append((s, b, html))):
    eng = CompositeTrader.__new__(CompositeTrader)
    eng._task_id = '20260905160130'
    eng.status = {'name': '15m双向tpsl5测试', 'account_balance': 1378.45, 'buy_count': 3, 'sell_count': 1}
    eng._log = lambda *a, **k: None
    eng.leverage = 1
    base = {'symbol': 'CC/USDT', 'name': 'Canton', 'strategy': 'macd+背离+量能', 'timeframe': '15m',
            'position': 2935, 'allocated_fund': 316.67, 'buy_balance': 318.92, 'leverage': 1}
    eng._mail_trade(dict(base), '做空', '开仓', 2935, 0.1079, extra='开仓价 0.1079, 杠杆1x')
    eng._mail_trade(dict(base), '平仓', '止盈', 2935, 0.1135, extra='平仓价 0.1135', pnl=15.42)
    eng._mail_trade(dict(base), '平仓', '止损', 2935, 0.1018, extra='平仓价 0.1018', pnl=-6.75)

parts = ['<html><body style="background:#eef1f6;padding:20px">']
for subject, body, is_html in captured:
    parts.append(f'<div style="color:#333;font-size:14px;margin:18px 0 6px">邮件标题：{subject} | html={is_html}</div>')
    parts.append(body)
parts.append('</body></html>')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'mail_preview.html')
open(out, 'w', encoding='utf-8').write(''.join(parts))
print('预览已生成 →', out)
print('场景数:', len(captured), '| 全部html模式:', all(x[2] for x in captured))
for subject, _, _ in captured:
    print(' -', subject)
