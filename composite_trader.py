# -*- coding: utf-8 -*-
"""
美股综合量化交易引擎
一个综合任务 = 同时监控/交易多个美股币对，每个币对可独立配置策略与资金权重。

核心能力：
- 多币种批量拉取实时数据（get_tickers 一次获取全部）
- 逐币对独立策略信号检测（买点/卖点）
- 资金管理：每个币对有独立复利池(allocated_fund × ratio)，买入金额 = buy_balance × 0.95（安全系数）
- 多币对同时出现买点时，按各自权重(ratio) 计算下单金额，互不占用
- 买点开多，卖点平多；可选卖点开空(allow_short)
- 完整交易记录 + 任务状态监控 + 邮件/日志持久化

通过独立后台线程运行，周期批量拉K线 → 计算各币对指标 → 校验信号 → 调用 FuturesTrader 开平仓。
"""
import json
import os
import threading
import time
from datetime import datetime

import pandas as pd

from indicators import TechnicalIndicators
from backtest_engine import BacktestEngine
from futures_trader import FuturesTrader
import mailer


# 状态持久化文件
STATE_FILE = os.path.join('data', 'composite_state.json')
# 量化任务历史列表（用于崩溃后查看/手动恢复）
TASKS_FILE = os.path.join('data', 'composite_tasks.json')
# 任务日志目录（每任务一个文件，供前端切换查看与导出）
LOG_DIR = os.path.join('data', 'logs')
# 心跳告警阈值：连续失败达到该次数触发告警
ALERT_THRESHOLD = 3
# 买入安全系数：实际下单金额 = 计算值 × 0.95
DEFAULT_BUY_PCT = 0.95


# 策略名 → 指标参数（与 app._strategy_params_from_names 保持一致）
def strategy_params(name):
    p = {}
    n = (name or '').strip()
    if n == "RSI":
        p.update({"rsi": True, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70})
    elif n == "KDJ":
        p.update({"kdj": True, "kdj_k_period": 9, "kdj_d_period": 3, "kdj_j_period": 3,
                  "kdj_buy_threshold": 20, "kdj_sell_threshold": 80})
    elif n == "布林带":
        p.update({"boll": True, "bb_period": 20, "bb_std": 2.0})
    elif n == "EMA":
        p.update({"ema": True, "ema_short": 12, "ema_long": 26, "ema_periods": [12, 26]})
    elif n == "MACD":
        p.update({"macd": True, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9})
    elif n == "双均线交叉":
        p.update({"ma_cross": True, "ma_cross_short": 10, "ma_cross_long": 30,
                  "ma_cross_periods": [10, 30]})
    return p


class CompositeTrader:
    """美股综合量化引擎（多币种 + 独立策略 + 资金比例分配，独立后台线程）"""

    def __init__(self, api_key='', api_secret='', proxy=None, trader=None, leverage=5, testnet=True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.proxy = proxy
        self.leverage = leverage
        self.testnet = testnet
        # 可复用外部已建好的合约交易器
        self.trader = trader or FuturesTrader(api_key, api_secret, proxy, testnet, leverage)
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._running = False
        self.log_file = None      # 任务日志文件（start 时按 task_id 生成，磁盘持久化供导出）
        self._task_id = None
        self.reset_status()

    # ---------- 状态管理 ----------
    @property
    def state_file(self):
        return STATE_FILE

    def reset_status(self):
        self.status = {
            'running': False,
            'name': '美股综合量化任务',
            'total_fund': 10000,        # 任务总可用资金（USDT）
            'buy_pct': DEFAULT_BUY_PCT,  # 买入安全系数（实际投入 = 计算值 × 0.95）
            'interval': 30,              # 轮询间隔（秒）
            'leverage': self.leverage,   # 杠杆倍数
            'prioritize': False,         # 量化优先匹配：开启后资金均分为 n 份份额，先触发买点先分配
            'share_count': 0,            # 份额总数 n（默认=股票数）
            'available_shares': 0,       # 候选池当前可用份额
            'symbols': [],               # 币对配置+实时状态列表
            'account_balance': 0.0,      # 账户可用 USDT 余额
            'last_loop_time': None,      # 心跳：最近一次成功轮询时间
            'signal': '等待',            # 任务级最近信号（任一币对最显著）
            'buy_count': 0,              # 全任务累计买入（开仓）次数
            'sell_count': 0,             # 全任务累计卖出（平仓）次数
            'monitor_loop': 0,
            'alerts': [],                # 告警记录（最近20条）
            'log': [],                   # 最近日志（最多 50 条）
            'started_at': None,
            'last_error': None,
            'consecutive_errors': 0,
        }

    def _log(self, msg):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        with self._lock:
            self.status['log'].append(line)
            if len(self.status['log']) > 50:
                self.status['log'] = self.status['log'][-50:]
        # 磁盘持久化（全量，供前端切换任务查看与导出；失败不影响交易）
        if self.log_file:
            try:
                os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(line + "\n")
            except Exception:
                pass

    def _alert(self, msg):
        """告警：写入告警列表（页面红框展示）+ 运行日志 + 磁盘文件"""
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        with self._lock:
            self.status['alerts'].append(line)
            if len(self.status['alerts']) > 20:
                self.status['alerts'] = self.status['alerts'][-20:]
        self._log(f"⚠ 告警: {msg}")
        try:
            os.makedirs(os.path.dirname(TASKS_FILE), exist_ok=True)
            with open(os.path.join('data', 'composite_alerts.log'), 'a', encoding='utf-8') as f:
                f.write(line + "\n")
        except Exception:
            pass

    # ---------- 量化任务历史（崩溃恢复用） ----------
    def _save_task(self):
        """启动时把本次量化配置写入任务列表（保留最近20条），供崩溃后手动恢复"""
        rec = {
            'id': self._task_id,
            'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'name': self.status['name'],
            'total_fund': self.status['total_fund'],
            'interval': self.status['interval'],
            'leverage': self.status['leverage'],
            'buy_pct': self.status['buy_pct'],
            'prioritize': self.status.get('prioritize', False),
            'share_count': self.status.get('share_count', 0),
            'symbols': [{
                'symbol': s['symbol'], 'name': s['name'], 'strategy': s['strategy'],
                'timeframe': s['timeframe'], 'fund_ratio': s['fund_ratio'],
                'long_only': s.get('long_only', True),
                'allow_short': s.get('allow_short', False),
                'take_profit_pct': s.get('take_profit_pct', 0),
                'stop_loss_pct': s.get('stop_loss_pct', 0),
            } for s in self.status['symbols']],
            'status': 'running',
            'last_active': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        tasks = self._read_tasks()
        # 复用旧 task_id（恢复任务）时更新原记录，否则插入新记录，避免恢复时残留旧任务
        replaced = False
        for i, t in enumerate(tasks):
            if t.get('id') == rec['id']:
                tasks[i] = rec
                replaced = True
                break
        if not replaced:
            tasks.insert(0, rec)
        tasks = tasks[:20]
        try:
            os.makedirs(os.path.dirname(TASKS_FILE), exist_ok=True)
            with open(TASKS_FILE, 'w', encoding='utf-8') as f:
                json.dump(tasks, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _update_task_status(self, task_status):
        """停止时更新任务状态，便于任务列表区分 运行中/已停止"""
        tid = getattr(self, '_task_id', None)
        if not tid:
            return
        tasks = self._read_tasks()
        for t in tasks:
            if t.get('id') == tid:
                t['status'] = task_status
                t['last_active'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                break
        try:
            os.makedirs(os.path.dirname(TASKS_FILE), exist_ok=True)
            with open(TASKS_FILE, 'w', encoding='utf-8') as f:
                json.dump(tasks, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    @staticmethod
    def _read_tasks():
        try:
            if os.path.exists(TASKS_FILE):
                with open(TASKS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    # ---------- 状态持久化 ----------
    def save_state(self):
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            data = {
                'name': self.status['name'],
                'total_fund': self.status['total_fund'],
                'interval': self.status['interval'],
                'leverage': self.status['leverage'],
                'prioritize': self.status.get('prioritize', False),
                'share_count': self.status.get('share_count', 0),
                'available_shares': self.status.get('available_shares', 0),
                'symbols': [{
                    'symbol': s['symbol'], 'name': s['name'], 'strategy': s['strategy'],
                    'timeframe': s['timeframe'], 'fund_ratio': s['fund_ratio'],
                    'allocated_fund': s['allocated_fund'], 'buy_balance': s['buy_balance'],
                    'buy_pct': s['buy_pct'],
                    'long_only': s.get('long_only', True),
                    'allow_short': s.get('allow_short', False),
                    'take_profit_pct': s.get('take_profit_pct', 0),
                    'stop_loss_pct': s.get('stop_loss_pct', 0),
                    'side': s.get('side', 'none'), 'position': s.get('position', 0),
                    'entry_price': s.get('entry_price', 0.0),
                    'buy_count': s.get('buy_count', 0), 'sell_count': s.get('sell_count', 0),
                    'shares': s.get('shares', 0),
                } for s in self.status['symbols']],
                'buy_count': self.status['buy_count'],
                'sell_count': self.status['sell_count'],
                'last_loop_time': self.status.get('last_loop_time'),
            }
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def load_state(self):
        try:
            if not os.path.exists(self.state_file):
                return
            with open(self.state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.status['name'] = data.get('name', self.status['name'])
            self.status['total_fund'] = data.get('total_fund', self.status['total_fund'])
            self.status['interval'] = data.get('interval', self.status['interval'])
            self.status['leverage'] = data.get('leverage', self.status['leverage'])
            self.status['prioritize'] = data.get('prioritize', False)
            self.status['share_count'] = data.get('share_count', 0)
            self.status['available_shares'] = data.get('available_shares', 0)
            saved = data.get('symbols', [])
            for i, s in enumerate(self.status['symbols']):
                if i >= len(saved):
                    break
                src = saved[i]
                if s['symbol'] != src.get('symbol'):
                    continue
                s['allocated_fund'] = src.get('allocated_fund', s.get('allocated_fund', self.status['total_fund'] * s['fund_ratio']))
                s['buy_balance'] = src.get('buy_balance', s.get('buy_balance', s['allocated_fund']))
                s['buy_pct'] = src.get('buy_pct', DEFAULT_BUY_PCT)
                s['side'] = src.get('side', 'none')
                s['position'] = src.get('position', 0)
                s['entry_price'] = src.get('entry_price', 0.0)
                s['buy_count'] = src.get('buy_count', s.get('buy_count', 0))
                s['sell_count'] = src.get('sell_count', s.get('sell_count', 0))
                s['shares'] = src.get('shares', 0)
            self.status['buy_count'] = data.get('buy_count', self.status['buy_count'])
            self.status['sell_count'] = data.get('sell_count', self.status['sell_count'])
            self.status['last_loop_time'] = data.get('last_loop_time')
        except Exception:
            pass

    # ---------- 币对配置 ----------
    def _build_symbols(self, symbol_configs, total_fund, buy_pct=DEFAULT_BUY_PCT,
                       prioritize=False, share_count=0):
        """根据前端配置构建币对状态列表。
        默认模式：按各币对 fund_ratio 分配资金池（复利起点）。
        优先匹配模式(prioritize)：将总资金平均划分为 n 份份额，每份 = total_fund / n，
        由触发买点的股票按时间顺序抢占份额（先触发先得），份额用完暂停买入。
        """
        symbols = []
        eff_buy_pct = float(buy_pct or DEFAULT_BUY_PCT) or DEFAULT_BUY_PCT
        valid_cfgs = [c for c in (symbol_configs or []) if (c.get('symbol') or '').strip()]
        n_sym = len(valid_cfgs)
        if n_sym == 0:
            self.status['share_count'] = 0
            self.status['available_shares'] = 0
            return []
        # 份额总数 n：优先匹配下默认=股票数，可配置但不超过股票数
        n = n_sym
        if prioritize:
            n = int(share_count or 0)
            if n < 1 or n > n_sym:
                n = n_sym
        unit = (total_fund / n) if (prioritize and n > 0) else 0.0
        for cfg in valid_cfgs:
            sym = (cfg.get('symbol') or '').strip()
            if prioritize:
                ratio = 1.0 / n if n > 0 else 0.0
                allocated = round(unit, 8)
            else:
                ratio = float(cfg.get('fund_ratio') or 0)
                # 若配置的是百分比(如 25) 或比例(0.25)，统一归一为比例
                if ratio > 1:
                    ratio = ratio / 100.0
                ratio = max(0.0, min(1.0, ratio))
                allocated = round(total_fund * ratio, 8)
            symbols.append({
                'symbol': sym,
                'name': cfg.get('name') or sym.split('/')[0],
                'strategy': cfg.get('strategy') or 'EMA',
                'timeframe': cfg.get('timeframe') or '1h',
                'fund_ratio': ratio,
                'allocated_fund': allocated,
                'buy_balance': allocated,           # 复利池起点 = 分到的本金
                'buy_pct': eff_buy_pct,             # 任务级买入安全系数（如 0.95）
                'long_only': bool(cfg.get('long_only', True)),
                'allow_short': bool(cfg.get('allow_short', False)),
                'take_profit_pct': float(cfg.get('take_profit_pct') or 0),
                'stop_loss_pct': float(cfg.get('stop_loss_pct') or 0),
                # 实时状态
                'side': 'none',
                'position': 0,
                'entry_price': 0.0,
                'last_price': 0.0,
                'last_close': 0.0,
                'signal': '等待',
                'unrealized_pnl': 0.0,
                'liquidation_price': 0.0,
                'margin': 0.0,
                'buy_count': 0,
                'sell_count': 0,
                'last_trade': None,
                'last_error': None,
                'shares': 0,                        # 优先匹配：当前占用的份额数
            })
        # 优先匹配：写入份额池总量与可用数（关闭时清零）
        self.status['prioritize'] = bool(prioritize)
        self.status['share_count'] = n if prioritize else 0
        self.status['available_shares'] = n if prioritize else 0
        return symbols

    def _inv(self, s):
        """币对当前实际投入金额 = 复利池 × 安全系数(95%)"""
        return (s.get('buy_balance') or 0.0) * (s.get('buy_pct', DEFAULT_BUY_PCT) or DEFAULT_BUY_PCT)

    # ---------- 量化优先匹配：份额授予/回收 ----------
    def _grant_share(self, s):
        """优先匹配：为触发买点的股票分配 1 份份额。成功返回 True，份额耗尽返回 False。"""
        if not self.status.get('prioritize'):
            return True
        total = self.status.get('share_count', 0)
        if self.status.get('available_shares', 0) <= 0:
            self._log(f"优先匹配: 份额已用完({total}份)，暂停买入 {s['symbol']}，仅监控卖出")
            return False
        self.status['available_shares'] = self.status.get('available_shares', 0) - 1
        s['shares'] = s.get('shares', 0) + 1
        self._log(f"优先匹配: 分配 1 份份额给 {s['symbol']}（剩余 {self.status['available_shares']}/{total} 份）")
        return True

    def _release_share(self, s):
        """优先匹配：卖出完成后回收该股票占用的份额，重新加入候选池。"""
        if not self.status.get('prioritize'):
            return
        held = s.get('shares', 0) or 0
        if held <= 0:
            return
        total = self.status.get('share_count', 0)
        self.status['available_shares'] = min(total, self.status.get('available_shares', 0) + held)
        s['shares'] = 0
        self._log(f"优先匹配: 回收 {s['symbol']} 的 {held} 份份额（剩余 {self.status['available_shares']}/{total} 份）")

    # ---------- 批量价格获取 ----------
    def _refresh_prices(self):
        """批量获取所有币对当前价格（一次请求），未命中的逐个兜底"""
        syms = [s['symbol'] for s in self.status['symbols'] if s['symbol']]
        if not syms:
            return
        prices = {}
        try:
            prices = self.trader.get_tickers(syms)
        except Exception:
            prices = {}
        for s in self.status['symbols']:
            p = prices.get(s['symbol'])
            if p:
                s['last_price'] = p
            else:
                try:
                    p2, _ = self.trader.get_ticker(s['symbol'])
                    if p2:
                        s['last_price'] = p2
                except Exception:
                    pass

    def _refresh_positions_and_balance(self):
        """一次性刷新全部币对真实持仓方向/张数/均价/未实现盈亏/强平价 与账户 USDT 余额"""
        try:
            bal_list, _ = self.trader.get_balance()
            if bal_list:
                usdt_asset = next((b for b in bal_list if b['asset'] == 'USDT'), None)
                self.status['account_balance'] = round(float(usdt_asset['free']) if usdt_asset else 0.0, 2)
        except Exception:
            pass
        # 批量拉所有持仓（一次请求）
        positions = []
        try:
            positions, _ = self.trader.get_positions()
        except Exception:
            positions = []
        pos_by_symbol = {}
        for p in positions:
            pos_by_symbol[p.get('symbol')] = p
        for s in self.status['symbols']:
            sym = s['symbol']
            pos = pos_by_symbol.get(sym)
            if not pos or pos.get('contracts', 0) <= 0:
                # 外部平仓检测：引擎认为有仓但交易所已无仓
                if s.get('position') and s.get('position', 0) > 0 and s.get('side') != 'none':
                    self._alert(f"检测到持仓已被外部平仓({sym})，清理挂单并同步状态")
                    s['sell_count'] += 1
                    self.status['sell_count'] += 1
                    try:
                        self.trader.cancel_all_orders(sym)
                    except Exception:
                        pass
                self._release_share(s)  # 优先匹配：外部平仓同样回收份额
                s['position'] = 0
                s['side'] = 'none'
                s['entry_price'] = 0.0
                s['unrealized_pnl'] = 0.0
                s['liquidation_price'] = 0.0
                s['margin'] = 0.0
            else:
                s['position'] = pos['contracts']
                s['side'] = pos.get('side', 'long')
                s['entry_price'] = pos.get('entry_price', 0.0) or 0.0
                s['unrealized_pnl'] = pos.get('unrealized_pnl', 0.0) or 0.0
                s['liquidation_price'] = pos.get('liquidation_price', 0.0) or 0.0
                s['margin'] = pos.get('margin', 0.0) or 0.0

    # ---------- 信号计算 ----------
    def _compute_signal(self, s):
        """拉取该币对K线 → 计算指标 → 返回当前根信号（1=做多, -1=做空/平多, 0=无）"""
        sym, timeframe = s['symbol'], s['timeframe']
        indicator_params = strategy_params(s['strategy'])
        candles, err = self.trader.get_ohlcv(sym, timeframe, limit=200)
        if err or not candles:
            return 0, err
        df = pd.DataFrame(candles)
        df['timestamp'] = pd.to_datetime(df['ts'], unit='ms')
        df = df.set_index('timestamp')
        df = TechnicalIndicators.calculate_all_indicators(df, indicator_params)
        engine = BacktestEngine(timeframe=timeframe, signal_mode='or')
        signals = engine.calculate_signals(df, indicator_params)
        sig = int(signals.iloc[-1]) if not signals.empty else 0
        if not df.empty:
            s['last_close'] = float(df['close'].iloc[-1])
        return sig, None

    def _refresh_last_close(self, s):
        """拉取最近一根已收盘K线收盘价（市价单3%偏差校验基准）"""
        try:
            candles, err = self.trader.get_ohlcv(s['symbol'], s['timeframe'], limit=2)
            if not err and candles:
                s['last_close'] = float(candles[-1]['close'])
        except Exception:
            pass

    # ---------- 价格偏离校验 / 止盈止损 / 成交邮件 ----------
    def _check_deviation(self, s, price):
        """市价单前校验：当前价与最近一根已收盘K线收盘价价差≤3%"""
        if not price or price <= 0:
            return None
        close = s.get('last_close', 0.0) or 0.0
        if close <= 0:
            return None
        dev = abs(price - close) / close
        if dev > 0.03:
            msg = f"{s['symbol']} 当前价 {price:.6f} 距最近收盘 {close:.6f} 偏移 {dev*100:.2f}% > 3%，已拦截市价单"
            self._log(f"⚠ {msg}")
            if mailer.is_configured():
                mailer.send_async("🛡️ 币安量化拦截(综合): 价格偏移超3%", msg)
            s['last_error'] = f"价格偏移{dev*100:.2f}%>3%，拦截"
            return f"当前价偏离上次收盘价 {dev*100:.2f}% > 3%"
        return None

    def _mail_trade(self, s, side, action, qty, price, extra=''):
        """下单成功邮件通知"""
        if not mailer.is_configured():
            return
        body = (f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"品种: {s['symbol']}\n方向: {side}\n操作: {action}\n"
                f"数量: {qty}\n价格: {price}\n"
                + (f"备注: {extra}\n" if extra else ""))
        subject = f"💰 币安量化实盘成交({side}): {s['symbol']} 近{price}"
        mailer.send_async(subject, body)
        self._log(f"已发送成交邮件: {s['symbol']} {side} {action}")

    def _check_tp_sl(self, s, price):
        """止盈/止损监控：开仓均价达到设置的止盈/止损价则市价平仓"""
        tp = s.get('take_profit_pct') or 0.0
        sl = s.get('stop_loss_pct') or 0.0
        if tp <= 0 and sl <= 0:
            return
        pos, avg, side = s.get('position', 0) or 0, s.get('entry_price', 0.0) or 0.0, s.get('side', 'none')
        if pos <= 0 or avg <= 0 or side == 'none':
            return
        trigger, pct = None, 0.0
        if side == 'long':
            if tp > 0 and price >= avg * (1 + tp):
                trigger, pct = '止盈', tp
            elif sl > 0 and price <= avg * (1 - sl):
                trigger, pct = '止损', sl
        else:  # short
            if tp > 0 and price <= avg * (1 - tp):
                trigger, pct = '止盈', tp
            elif sl > 0 and price >= avg * (1 + sl):
                trigger, pct = '止损', sl
        if not trigger:
            return
        self._log(f"{trigger}触发: {s['symbol']} {side} 现价 {price:.6f} 开仓价 {avg:.6f} ({trigger}{pct*100:.1f}%)")
        self._exit_position(s, side, trigger)

    def _calc_contracts(self, symbol, usdt_amount):
        """按 USDT 金额和杠杆，结合张数精度计算可开仓的张数"""
        price = self._refresh_one_price(symbol)
        if not price or price <= 0:
            return 0, price
        notional = usdt_amount * self.leverage
        contracts = notional / price
        contracts = self.trader.round_amount(symbol, contracts)
        return contracts, price

    def _refresh_one_price(self, symbol):
        p, _ = self.trader.get_ticker(symbol)
        return p

    def _exit_position(self, s, side, reason):
        """市价平掉该币对合约持仓（止盈/止损/强制平仓共用，reduceOnly）"""
        pos = s.get('position', 0) or 0
        if pos <= 0:
            return
        contracts = self.trader.round_amount(s['symbol'], pos)
        if contracts <= 0:
            self._log(f"{reason}: {s['symbol']} 平仓数量精度不足，跳过")
            return
        side_cmd = 'sell' if side == 'long' else 'buy'
        price = (s.get('last_price') or 0.0)
        dev = self._check_deviation(s, price)
        if dev:
            self._log(f"{reason}被拦截(价格偏移): {dev}")
            return
        order, err = self.trader.place_order(s['symbol'], side_cmd, 'market', contracts, reduce_only=True)
        if err:
            self._log(f"{reason}平仓失败: {err}")
            s['last_error'] = f"{reason}平仓失败: {err}"
            return
        pnl = s.get('unrealized_pnl', 0.0) or 0.0
        s['buy_balance'] = round((s.get('buy_balance', 0.0) or 0.0) + pnl, 8)
        s['position'] = 0
        s['side'] = 'none'
        s['entry_price'] = 0.0
        s['sell_count'] += 1
        self.status['sell_count'] += 1
        self._release_share(s)  # 优先匹配：卖出后回收份额
        _, cerr = self.trader.cancel_all_orders(s['symbol'])
        if cerr:
            self._log(f"清理挂单失败(可能无挂单): {cerr}")
        s['last_trade'] = {
            'time': datetime.now().isoformat(),
            'action': 'CLOSE', 'symbol': s['symbol'],
            'price': price, 'contracts': contracts,
            'pnl': pnl, 'reason': reason,
        }
        self._log(f"{reason}平仓: {contracts} 张 {s['symbol']} @ {price} (盈亏{pnl:+.2f}U, 复利池→{s['buy_balance']:.2f}U)")
        self._mail_trade(s, '平仓', reason, contracts, price or 0, extra=f"平仓价 {price}")
        self.save_state()

    def _close_position(self, s, side):
        """平掉该币对当前持仓（reduceOnly），按张数"""
        position = s.get('position', 0) or 0
        if position <= 0:
            return
        contracts = self.trader.round_amount(s['symbol'], position)
        if contracts <= 0:
            return
        side_cmd = 'sell' if side == 'long' else 'buy'
        price = (s.get('last_price') or 0.0)
        dev = self._check_deviation(s, price)
        if dev:
            s['last_error'] = f'平仓被拦截: {dev}'
            self._log(f"平仓被拦截(价格偏移): {dev}")
            return
        order, err = self.trader.place_order(s['symbol'], side_cmd, 'market', contracts, reduce_only=True)
        if err:
            s['last_error'] = f'平仓失败: {err}'
            self._log(f"平仓失败: {err}")
            return
        pnl = s.get('unrealized_pnl', 0.0) or 0.0
        s['buy_balance'] = round((s.get('buy_balance', 0.0) or 0.0) + pnl, 8)
        s['position'] = 0
        s['side'] = 'none'
        s['entry_price'] = 0.0
        s['sell_count'] += 1
        self.status['sell_count'] += 1
        self._release_share(s)  # 优先匹配：卖出后回收份额
        _, cerr = self.trader.cancel_all_orders(s['symbol'])
        if cerr:
            self._log(f"清理挂单失败(可能无挂单): {cerr}")
        s['last_trade'] = {
            'time': datetime.now().isoformat(),
            'action': 'CLOSE', 'symbol': s['symbol'],
            'price': price, 'contracts': contracts,
            'pnl': pnl,
        }
        self._log(f"平仓: {contracts} 张 {s['symbol']} @ {price} (盈亏{pnl:+.2f}U, 复利池→{s['buy_balance']:.2f}U)")
        self._mail_trade(s, '平仓', '平仓(信号)', contracts, price or 0, extra=f"平仓价 {price}")
        self.save_state()

    def _open_long(self, s):
        """开多：用币对复利池 × 安全系数 计算下单金额，并挂止损保护单"""
        contracts, price = self._calc_contracts(s['symbol'], self._inv(s))
        if contracts <= 0:
            s['last_error'] = f"{s['symbol']} 开多张数为0，跳过"
            return 'skip'
        dev = self._check_deviation(s, price)
        if dev:
            s['last_error'] = f'开多被拦截: {dev}'
            return 'skip'
        order, err = self.trader.place_order(s['symbol'], 'buy', 'market', contracts)
        if err:
            s['last_error'] = f'开多失败: {err}'
            self._log(f"开多失败: {err}")
            return 'error'
        s['position'] = contracts
        s['side'] = 'long'
        s['entry_price'] = price
        s['buy_count'] += 1
        self.status['buy_count'] += 1
        s['last_trade'] = {
            'time': datetime.now().isoformat(), 'action': 'OPEN_LONG', 'symbol': s['symbol'],
            'price': price, 'contracts': contracts, 'notional': contracts * price,
            'invest': self._inv(s),
        }
        self._log(f"开多: {contracts} 张 {s['symbol']} @ {price} (投入{self._inv(s):.2f}U, 杠杆{self.leverage}x)")
        self._mail_trade(s, '做多', '开仓', contracts, price, extra=f"开仓价 {price}, 杠杆{self.leverage}x")
        self.save_state()
        return 'ok'

    def _open_short(self, s):
        """开空：用币对复利池 × 安全系数 计算下单金额（仅 allow_short 时使用）"""
        contracts, price = self._calc_contracts(s['symbol'], self._inv(s))
        if contracts <= 0:
            s['last_error'] = f"{s['symbol']} 开空张数为0，跳过"
            return 'skip'
        dev = self._check_deviation(s, price)
        if dev:
            s['last_error'] = f'开空被拦截: {dev}'
            return 'skip'
        order, err = self.trader.place_order(s['symbol'], 'sell', 'market', contracts)
        if err:
            s['last_error'] = f'开空失败: {err}'
            self._log(f"开空失败: {err}")
            return 'error'
        s['position'] = contracts
        s['side'] = 'short'
        s['entry_price'] = price
        s['buy_count'] += 1
        self.status['buy_count'] += 1
        s['last_trade'] = {
            'time': datetime.now().isoformat(), 'action': 'OPEN_SHORT', 'symbol': s['symbol'],
            'price': price, 'contracts': contracts, 'notional': contracts * price,
        }
        self._log(f"开空: {contracts} 张 {s['symbol']} @ {price} (投入{self._inv(s):.2f}U, 杠杆{self.leverage}x)")
        self._mail_trade(s, '做空', '开仓', contracts, price, extra=f"开仓价 {price}, 杠杆{self.leverage}x")
        self.save_state()
        return 'ok'

    def _apply_orders(self, s):
        """对单个币对应用信号：买点开多/持有，卖点平多/开空（allow_short）。
        优先匹配模式下，开仓前须先抢占份额，份额耗尽则暂停新开仓、仅监控卖出。"""
        sig = s.get('_sig')
        current_side = s.get('side', 'none')
        position = s.get('position', 0) or 0

        if sig == 1:  # 买点：开多
            if position > 0 and current_side == 'long':
                return 'idle'
            if position > 0 and current_side == 'short':
                # 持空仓遇买点：平空（不反手开多，与单币种引擎一致）
                self._close_position(s, 'short')
                return 'closed'
            return self._open_with_share(s, self._open_long)
        elif sig == -1:  # 卖点
            if current_side == 'long' and position > 0:
                self._close_position(s, 'long')
                return 'closed'
            if position > 0 and current_side == 'short':
                return 'idle'
            # 仅做多：卖点只平多不开空
            if s.get('long_only', True):
                return 'idle'
            if not s.get('allow_short', False):
                return 'idle'
            return self._open_with_share(s, self._open_short)
        return 'idle'

    def _open_with_share(self, s, open_fn):
        """优先匹配模式下，开仓前先抢占份额；下单失败/跳过则回滚份额。"""
        if not self.status.get('prioritize'):
            return open_fn(s)
        if not self._grant_share(s):
            s['signal'] = '待命(份额耗尽)'
            return 'no_share'
        result = open_fn(s)
        if result != 'ok':
            self._release_share(s)  # 未成交，回收份额
        return result

    # ---------- 主循环 ----------
    def _run_loop(self):
        self._running = True
        self.status['running'] = True
        self.status['started_at'] = datetime.now().isoformat()
        n_sym = len(self.status['symbols'])
        names = ', '.join(f"{s['symbol']}({s['strategy']},{s['fund_ratio']*100:.0f}%)" for s in self.status['symbols'])
        self._log(f"美股综合量化已启动: {self.status['name']} 共{n_sym}个币对, 总资金{self.status['total_fund']}U, "
                  f"杠杆{self.leverage}x, 间隔{self.status['interval']}s, 安全系数{self.status['buy_pct']*100:.0f}%")
        self._log(f"资金分配: {names}")
        if self.status.get('prioritize'):
            self._log(f"量化优先匹配已开启: 总资金均分为 {self.status.get('share_count', 0)} 份份额，"
                      f"每份 {self.status['total_fund']/max(self.status.get('share_count', 1), 1):.2f}U，"
                      f"先触发买点先分配，份额用完暂停买入")
        # 逐币对设置杠杆
        for s in self.status['symbols']:
            try:
                self.trader.set_leverage(self.leverage, s['symbol'])
            except Exception as e:
                self._log(f"设置杠杆失败({s['symbol']}): {e}")

        alerted = False
        while not self._stop_event.is_set():
            try:
                # 1. 批量刷新价格
                self._refresh_prices()
                # 2. 逐币对计算信号并应用买卖（单个币对失败不阻断其它币对）
                all_sig = 0
                failed = 0
                for s in self.status['symbols']:
                    sig, err = self._compute_signal(s)
                    if err:
                        s['last_error'] = err
                        s['signal'] = '观望'
                        s['_sig'] = 0
                        failed += 1
                        self._log(f"信号计算失败({s['symbol']}): {err}")
                        continue
                    s['_sig'] = sig
                    if sig == 1:
                        s['signal'] = '买入(多)'
                        all_sig = 1
                    elif sig == -1:
                        s['signal'] = '卖出(空)'
                        if all_sig == 0:
                            all_sig = -1
                    else:
                        s['signal'] = '观望'
                    # 3. 止盈/止损监控（先于开平仓信号执行）
                    self._check_tp_sl(s, s.get('last_price') or 0.0)
                    # 4. 应用买卖信号
                    self._apply_orders(s)
                self.status['signal'] = '有买点' if all_sig == 1 else ('有卖点' if all_sig == -1 else '观望')
                # 5. 刷新实时持仓与账户余额（一次批量）
                self._refresh_positions_and_balance()
                # 仅当全部币对都失败才视为连续故障（触发重连）；个别币对失败不影响整体运行
                if failed >= len(self.status['symbols']):
                    self.status['consecutive_errors'] = self.status.get('consecutive_errors', 0) + 1
                else:
                    self.status['consecutive_errors'] = 0
                if alerted and failed == 0:
                    alerted = False
                    self._log("轮询已恢复正常（邮件告警序列已重置）")
            except Exception as e:
                self.status['last_error'] = str(e)
                self.status['consecutive_errors'] = self.status.get('consecutive_errors', 0) + 1
                self._log(f"运行异常({self.status['consecutive_errors']}次): {e}")
                if self.status['consecutive_errors'] >= ALERT_THRESHOLD:
                    if not alerted:
                        self._alert(f"轮询连续失败{self.status['consecutive_errors']}次，正在自动重连")
                        alerted = True
                    self._reconnect()
            self.status['last_loop_time'] = datetime.now().isoformat()  # 心跳
            self.status['monitor_loop'] += 1
            self._stop_event.wait(self.status['interval'] or 30)

    # ---------- 网络恢复 ----------
    def _reconnect(self, max_tries=5):
        self._log("网络波动，尝试重新连接...")
        for i in range(max_tries):
            if self._stop_event.is_set():
                return False
            try:
                self.trader._create_exchange()
                self._log(f"已重新连接（第 {i + 1} 次尝试）")
                self.status['consecutive_errors'] = 0
                return True
            except Exception as e:
                self.status['last_error'] = f'重连失败: {e}'
                self._log(f"重连失败（{i + 1}/{max_tries}）: {e}")
                time.sleep(3)
        return False

    # ---------- 启停 ----------
    def start(self, name, total_fund, symbol_configs, interval=30, buy_pct=DEFAULT_BUY_PCT, task_id=None,
              prioritize=False, share_count=0):
        with self._lock:
            if self._running:
                return False, '已在运行中'
            self.reset_status()
            self.status['name'] = (name or '美股综合量化任务').strip()
            self.status['total_fund'] = float(total_fund or 0)
            self.status['buy_pct'] = float(buy_pct or DEFAULT_BUY_PCT) or DEFAULT_BUY_PCT
            self.status['interval'] = max(5, int(interval or 30))
            self.status['prioritize'] = bool(prioritize)
            self.status['symbols'] = self._build_symbols(symbol_configs, self.status['total_fund'],
                                                         self.status['buy_pct'],
                                                         prioritize=self.status['prioritize'],
                                                         share_count=share_count)
            if not self.status['symbols']:
                return False, '请至少选择一个币对'
            if sum(s['fund_ratio'] for s in self.status['symbols']) <= 0:
                return False, '请为各币对设置资金权重(>0)'
            self._stop_event.clear()
            # 恢复任务时复用旧 task_id，任务列表不新增、日志续写
            self._task_id = task_id or (datetime.now().strftime('%Y%m%d%H%M%S') + f"{int(time.time() * 1000) % 1000:03d}")
            self.log_file = os.path.join(LOG_DIR, f'composite_{self._task_id}.log')
            self._running = True
            self.status['running'] = True
            self.status['started_at'] = datetime.now().isoformat()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            self._save_task()
            self.save_state()
            return True, '已启动'

    def stop(self):
        with self._lock:
            if not self._running:
                return False, '未在运行'
            self._stop_event.set()
            self._running = False
            self.status['running'] = False
        self._log("美股综合量化已停止")
        self._update_task_status('stopped')
        self.save_state()
        return True, '已停止'

    @staticmethod
    def list_tasks():
        return CompositeTrader._read_tasks()

    @staticmethod
    def delete_task(task_id):
        tasks = CompositeTrader._read_tasks()
        remains = [t for t in tasks if t.get('id') != task_id]
        if len(remains) == len(tasks):
            return False, f"任务不存在: {task_id}"
        try:
            os.makedirs(os.path.dirname(TASKS_FILE), exist_ok=True)
            with open(TASKS_FILE, 'w', encoding='utf-8') as f:
                json.dump(remains, f, ensure_ascii=False, indent=2)
            return True, '已删除'
        except Exception as e:
            return False, str(e)

    def get_status(self):
        with self._lock:
            s = dict(self.status)
            s['symbols'] = [dict(x) for x in self.status['symbols']]
            s['log'] = list(self.status['log'])
            # 移除内部信号暂存字段
            for x in s['symbols']:
                x.pop('_sig', None)
            return s
