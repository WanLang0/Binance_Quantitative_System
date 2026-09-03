# -*- coding: utf-8 -*-
"""
自动现货交易引擎（实时信号监控 + 自动买入/卖出）
复用 BacktestEngine.calculate_signals 的 OR 逻辑与 TechnicalIndicators，保证与回测一致。
通过独立后台线程运行，周期拉取K线 → 计算指标 → 校验信号 → 调用 DemoTrader 下单。
"""
import json
import os
import threading
import time
from datetime import datetime

import pandas as pd

from indicators import TechnicalIndicators
from backtest_engine import BacktestEngine
from divergence_signals import compute_divergence_signals, find_divergence_name, any_divergence_variant
from demo_trader import DemoTrader
import mailer

# 状态持久化文件（模块级，供 /auto/api/status 在未启动时也能读取上次配置）
STATE_FILE = os.path.join('data', 'auto_trader_state.json')
# 任务历史列表（与自动合约一致：崩溃后可一键恢复）
TASKS_FILE = os.path.join('data', 'spot_tasks.json')
# 任务日志目录（每任务一个文件，供前端切换查看与导出）
LOG_DIR = os.path.join('data', 'logs')

# 邮件告警序列：达到阈值立即发第1封，之后每10分钟一封、共3封；恢复后另发一封恢复邮件（与自动合约 auto_futures 一致）
MAIL_INTERVAL, MAIL_MAX = 600, 3


class AutoTrader:
    """实时自动交易引擎（单交易对，运行于独立后台线程）"""

    def __init__(self, api_key='', api_secret='', proxy=None, trader=None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.proxy = proxy
        # 可复用外部已建好的交易器（如已缓存的 DemoTrader，避免反复 load_markets 拖慢启动）
        self.trader = trader or DemoTrader(api_key, api_secret, proxy)
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._running = False
        self._mail_count = 0      # 本轮故障周期已发邮件数
        self._mail_last_ts = 0.0  # 上封邮件时间戳
        self.log_file = None      # 任务日志文件（start 时按 task_id 生成，磁盘持久化供导出）
        self._task_id = None
        self.reset_status()

    # ---------- 状态管理 ----------
    @property
    def state_file(self):
        """状态持久化文件路径（保存配置与持仓，便于网络中断后恢复）"""
        return STATE_FILE

    def reset_status(self):
        self.status = {
            'running': False,
            'symbol': 'BTC/USDT',
            'timeframe': '15m',
            'strategies': [],           # 已选的策略名称列表
            'mode': 'standard',         # 策略模式：standard(标准多策略OR)/grid(网格)
            'qty_usdt': 1000,           # 每次买入金额（初始预算）
            'buy_balance': 0.0,          # 当前可用买入金额（本金+累计盈利，复利滚动）
            'buy_pct': 0.95,             # 可用买入金额的实际投入比例（95%）
            'position_cost': 0.0,        # 当前持仓累计成本（卖出后据此计算已实现盈利）
            'interval': 30,             # 轮询间隔（秒）
            'position': 0.0,            # 当前持仓数量（标准/网格模式：总持仓）
            'avg_price': 0.0,           # 持仓均价（模拟，按最近买入价）
            'last_price': 0.0,          # 最近价格
            'last_close': 0.0,          # 最近一根已收盘K线收盘价（市价单3%偏差校验基准）
            'take_profit_pct': 0.0,     # 止盈比例（相对均价，0=不启用）
            'stop_loss_pct': 0.0,       # 止损比例（相对均价，0=不启用）
            'signal': '等待',           # 最近信号（等待/买入/卖出/无）
            'real_position': 0.0,      # 账户真实持仓数量（总余额 total）
            'real_free_position': 0.0, # 账户真实可卖数量（free，部分被挂单占用）
            'real_avg_price': 0.0,     # 账户真实持仓均价（按最近价位估算）
            'real_value_usdt': 0.0,    # 账户真实持仓市值（USDT）
            'account_balance': 0.0,    # 账户可用 USDT 余额
            'grid': {                   # 网格交易状态
                'levels': [],           # [{buy_price, qty}] 已买入的网格持仓
                'filled': 0,            # 已买入格数
                'total_invest': 0,      # 网格累计投入资金
                'step_pct': 0.01,       # 每格百分比
                'max_levels': 12,       # 最大格数
            },
            'last_signal_time': None,
            'last_trade': None,
            'buy_count': 0,
            'sell_count': 0,
            'monitor_loop': 0,          # 已执行轮询次数
            'log': [],                  # 最近日志（最多 50 条）
            'started_at': None,
            'last_error': None,
            'consecutive_errors': 0,    # 连续失败次数（用于恢复重连）
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

    def _mail_alert(self, reason, errors):
        """网络故障邮件告警：每10分钟一封、共3封（恢复后 _mail_reset 重置）"""
        if not mailer.is_configured():
            if not getattr(self, '_mail_no_cfg_noted', False):
                self._mail_no_cfg_noted = True
                self._log("⚠ 达到告警阈值但邮箱未配置，无法发送邮件（请在「个人设置」页填写QQ邮箱+SMTP授权码）")
            return
        if self._mail_count >= MAIL_MAX:
            return
        if self._mail_count > 0 and time.time() - self._mail_last_ts < MAIL_INTERVAL:
            return
        if self._mail_count == 0:
            self._mail_fault_ts = time.time()
        sym = self.status.get('symbol') or '?'
        tf = self.status.get('timeframe') or '?'
        mode = '网格' if self.status.get('mode') == 'grid' else '标准'
        body = (f"告警时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"引擎: 自动现货[{mode}]  品种: {sym}  周期: {tf}\n"
                f"故障: {reason}\n连续失败: {errors} 次\n\n"
                f"引擎仍在自动重连；现货持仓无合约爆仓风险，但中断期间无法执行买卖信号。\n"
                f"本故障期内邮件提醒每10分钟一封、最多{MAIL_MAX}封（第 {self._mail_count + 1} 封）；恢复后另发一封恢复邮件。")
        mailer.send_async(f"⚠ 币安量化告警(现货 {self._mail_count + 1}/{MAIL_MAX}): 无法访问币安", body)
        self._log(f"已触发邮件告警(现货 {self._mail_count + 1}/{MAIL_MAX})")
        self._mail_count += 1
        self._mail_last_ts = time.time()

    def _mail_reset(self):
        """轮询恢复正常：若本轮故障发过告警邮件则补发一封恢复通知，并重置序列（下次故障重新计3封）"""
        if self._mail_count:
            if mailer.is_configured():
                dur = (time.time() - getattr(self, '_mail_fault_ts', time.time())) / 60
                sym = self.status.get('symbol') or '?'
                tf = self.status.get('timeframe') or '?'
                mode = '网格' if self.status.get('mode') == 'grid' else '标准'
                body = (f"恢复时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"引擎: 自动现货[{mode}]  品种: {sym}  周期: {tf}\n"
                        f"故障持续: 约{dur:.0f}分钟（自首封告警起算）\n本轮已发告警: {self._mail_count}封\n\n"
                        f"引擎已恢复正常轮询，无需操作。")
                mailer.send_async("✅ 币安量化告警恢复(现货): 币安访问已恢复", body)
                self._log("已发送恢复通知邮件")
            self._log("轮询已恢复正常（现货邮件告警序列已重置）")
        self._mail_count = 0
        self._mail_no_cfg_noted = False

    # ---------- 任务历史列表（崩溃后一键恢复） ----------
    def _save_task(self):
        """启动时把本次量化配置写入任务列表（保留最近20条），供崩溃后手动恢复"""
        rec = {
            'id': self._task_id,
            'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': self.status['symbol'],
            'timeframe': self.status['timeframe'],
            'strategies': self.status['strategies'],
            'mode': self.status['mode'],
            'qty_usdt': self.status['qty_usdt'],
            'interval': self.status['interval'],
            'grid_step': self.status['grid'].get('step_pct', 0.01),
            'grid_max_levels': self.status['grid'].get('max_levels', 12),
            'take_profit_pct': self.status['take_profit_pct'],
            'stop_loss_pct': self.status['stop_loss_pct'],
            'status': 'running',
            'last_active': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        self._task_id = rec['id']
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

    @staticmethod
    def list_tasks():
        """返回量化任务历史列表（最新在前）"""
        return AutoTrader._read_tasks()

    @staticmethod
    def delete_task(task_id):
        """删除指定的量化任务记录（不影响正在运行的引擎）"""
        tasks = AutoTrader._read_tasks()
        remains = [t for t in tasks if t.get('id') != task_id]
        if len(remains) == len(tasks):
            return False, f"任务不存在: {task_id}"
        try:
            with open(TASKS_FILE, 'w', encoding='utf-8') as f:
                json.dump(remains, f, ensure_ascii=False, indent=2)
            return True, '任务记录已删除'
        except Exception as e:
            return False, f'删除失败: {e}'

    # ---------- 状态持久化（网络中断后恢复） ----------
    def save_state(self):
        """将核心配置与持仓状态写入磁盘，便于断网/重启后恢复"""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            data = {
                'symbol': self.status['symbol'],
                'timeframe': self.status['timeframe'],
                'strategies': self.status['strategies'],
                'mode': self.status['mode'],
                'qty_usdt': self.status['qty_usdt'],
            'buy_balance': self.status['buy_balance'],
            'position_cost': self.status['position_cost'],
            'interval': self.status['interval'],
            'take_profit_pct': self.status['take_profit_pct'],
            'stop_loss_pct': self.status['stop_loss_pct'],
            'position': self.status['position'],
                'avg_price': self.status['avg_price'],
                'grid': self.status['grid'],
                'buy_count': self.status['buy_count'],
                'sell_count': self.status['sell_count'],
            }
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def load_state(self):
        """从磁盘恢复上次保存的配置与持仓"""
        try:
            if not os.path.exists(self.state_file):
                return
            with open(self.state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.status['symbol'] = data.get('symbol', self.status['symbol'])
            self.status['timeframe'] = data.get('timeframe', self.status['timeframe'])
            self.status['strategies'] = data.get('strategies', self.status['strategies'])
            self.status['mode'] = data.get('mode', self.status['mode'])
            self.status['qty_usdt'] = data.get('qty_usdt', self.status['qty_usdt'])
            self.status['buy_balance'] = data.get('buy_balance', self.status['buy_balance'])
            self.status['position_cost'] = data.get('position_cost', self.status['position_cost'])
            self.status['interval'] = data.get('interval', self.status['interval'])
            self.status['take_profit_pct'] = data.get('take_profit_pct', self.status['take_profit_pct'])
            self.status['stop_loss_pct'] = data.get('stop_loss_pct', self.status['stop_loss_pct'])
            self.status['position'] = data.get('position', self.status['position'])
            self.status['avg_price'] = data.get('avg_price', self.status['avg_price'])
            grid = data.get('grid', {})
            self.status['grid'].update(grid)
            self.status['buy_count'] = data.get('buy_count', self.status['buy_count'])
            self.status['sell_count'] = data.get('sell_count', self.status['sell_count'])
        except Exception:
            pass

    def _refresh_last_price(self, symbol):
        price, _ = self.trader.get_ticker(symbol)
        if price:
            self.status['last_price'] = price
        return price

    def _refresh_real_position(self, symbol):
        """从交易所账户刷新真实持仓（数量/均价/市值）与 USDT 余额，供前端展示"""
        try:
            base = symbol.split('/')[0]
            bal_list, _ = self.trader.get_balance()
            if not bal_list:
                return
            base_asset = next((b for b in bal_list if b['asset'] == base), None)
            usdt_asset = next((b for b in bal_list if b['asset'] == 'USDT'), None)
            qty = float(base_asset['total']) if base_asset else 0.0
            free_qty = float(base_asset['free']) if base_asset else 0.0
            self.status['real_position'] = round(qty, 8)
            # 可卖余额=free（部分持仓可能被挂单占用，total 含 locked，不能用于卖出上限）
            self.status['real_free_position'] = round(free_qty, 8)
            self.status['account_balance'] = round(float(usdt_asset['free']) if usdt_asset else 0.0, 2)
            price = self.status.get('last_price') or 0.0
            self.status['real_avg_price'] = self.status.get('avg_price', 0.0)
            self.status['real_value_usdt'] = round(qty * price, 2)
            # 若用户在交易所手动清仓（真实持仓为0），同步重置本地记账持仓/均价/网格，
            # 避免界面与策略仍认为持有仓位，导致后续信号误判。
            if qty <= 0:
                self.status['position'] = 0.0
                self.status['avg_price'] = 0.0
                self.status['position_cost'] = 0.0
                if 'grid' in self.status:
                    self.status['grid']['levels'] = []
                    self.status['grid']['filled'] = 0
        except Exception as e:
            self.status['last_error'] = f"刷新持仓失败: {e}"

    # ---------- 信号计算 ----------
    def _compute_signal(self, symbol, timeframe, indicator_params, strategies=None):
        """拉取最近K线 → 计算指标 → 返回当前根信号（1=买入, -1=卖出, 0=无）

        固定组合背离策略（macd+背离 / macd+背离+量能 / macd+背离+均线+量能）
        走 divergence_signals 专用信号构造，与回测口径一致；其余策略走 BacktestEngine。
        """
        candles, err = self.trader.get_ohlcv(symbol, timeframe, limit=200)
        if err or not candles:
            return 0, None, err
        df = pd.DataFrame(candles)
        df['timestamp'] = pd.to_datetime(df['ts'], unit='ms')
        df = df.set_index('timestamp')
        # 固定组合背离策略：使用专用信号构造（与回测一致）
        div_name = find_divergence_name(strategies)
        if div_name:
            dfd, signals = compute_divergence_signals(df, div_name)
            if signals is None or signals.empty:
                return 0, df, None
            sig = int(signals.iloc[-1])
        else:
            df = TechnicalIndicators.calculate_all_indicators(df, indicator_params)
            # 调用回测引擎的信号逻辑（OR：任一策略满足即触发）
            engine = BacktestEngine(timeframe=timeframe, signal_mode='or')
            signals = engine.calculate_signals(df, indicator_params)
            sig = int(signals.iloc[-1]) if not signals.empty else 0
        # 记录最近一根已收盘K线收盘价（供市价单3%偏差校验）
        if not df.empty:
            self.status['last_close'] = float(df['close'].iloc[-1])
        return sig, df, None

    # ---------- 网格交易 ----------
    def _grid_buy(self, symbol, price, qty_usdt, step_pct, max_levels):
        """
        网格买入：跌一格买一格。price 相对最近已买格每下跌 step_pct 就补一格。
        返回 (是否买入) —— 买入数量 = qty_usdt(单格资金) / price。
        """
        grid = self.status['grid']
        levels = grid.get('levels', [])
        # 若已有持仓，需要价格较上一格再下跌 step_pct 才补仓
        if levels:
            last_buy = levels[-1]['buy_price']
            if price > last_buy * (1 - step_pct):
                return False, '未达新网格位'
        else:
            # 首个网格：可直接买入，或跌破基准价跌一格再买（这里首次直接建仓）
            pass
        # 网格已满则不买入
        if len(levels) >= max_levels:
            return False, '网格已满'
        # 单格资金 = 当前复利池 buy_balance × 投入比例(95%)
        buy_balance = self.status.get('buy_balance', 0.0) or 0.0
        invest = buy_balance * self.status.get('buy_pct', 0.95)
        quantity = invest / price
        if quantity <= 0:
            return False, '买入数量为0'
        # 按币安数量精度取整，确保记录数量与实际成交一致（防止卖出时数量超出持仓）
        quantity = self.trader.round_amount(symbol, quantity)
        if quantity <= 0:
            return False, '买入数量精度不足'
        # 市价单价格保护：当前价与最近收盘价差≤3%
        dev = self._check_deviation(symbol, price)
        if dev:
            return False, f'网格买入被拦截: {dev}'
        order, err = self.trader.place_order(symbol, 'buy', 'market', quantity)
        if err:
            self._log(f"网格买入失败: {err}")
            return False, err
        # 以订单实际成交数量记录（市价单可能部分成交），避免"记录数量超出实际持仓"
        if order:
            filled = order.get('filled') or order.get('amount') or quantity
            try:
                filled = float(filled)
                if filled > 0:
                    quantity = self.trader.round_amount(symbol, filled)
            except (TypeError, ValueError):
                pass
        if quantity <= 0:
            return False, '实际成交数量为0'
        levels.append({'buy_price': price, 'qty': quantity})
        grid['levels'] = levels
        grid['filled'] = len(levels)
        grid['total_invest'] = round(grid.get('total_invest', 0) + quantity * price, 2)
        self.status['position'] = round(self.status.get('position', 0) + quantity, 8)
        self.status['position_cost'] = round((self.status.get('position_cost', 0.0) or 0.0) + quantity * price, 8)
        self.status['buy_count'] += 1
        self.status['last_trade'] = {'time': datetime.now().isoformat(), 'action': 'BUY',
                                     'symbol': symbol, 'price': price, 'quantity': quantity,
                                     'value_usdt': quantity * price, 'mode': 'grid'}
        self._log(f"网格买入(格{len(levels)}): {quantity:.6f} {symbol} @ {price}")
        self.save_state()
        return True, 'ok'

    def _grid_sell(self, symbol, price, step_pct):
        """
        网格卖出：价格较最近的买入格每上涨 step_pct，卖出该格对应数量。
        返回 (是否卖出)。
        """
        grid = self.status['grid']
        levels = grid.get('levels', [])
        if not levels:
            return False, '无网格持仓'
        # 以账户真实可卖数量(free)为卖出上限，避免市价单部分成交或挂单占用导致"记录数量超出实际持仓"
        self._refresh_real_position(symbol)
        real_qty = self.status.get('real_free_position', 0.0) or 0.0
        # 从最早的一格开始，若该格盈利达 step_pct 则卖出该格数量
        sold_any = False
        remains = []
        for lv in levels:
            if not sold_any and price >= lv['buy_price'] * (1 + step_pct):
                # 卖出本格：数量按币安精度取整，且不超过当前账户真实持仓（防超出余额）
                qty = self.trader.round_amount(symbol, lv['qty'])
                if qty <= 0:
                    remains.append(lv)
                    continue
                if qty > real_qty:
                    # 账户实际不足该格数量：按剩余可卖数量卖，若无可卖则保留该格
                    qty = self.trader.round_amount(symbol, real_qty)
                    if qty <= 0:
                        remains.append(lv)
                        continue
                # 市价单价格保护：当前价与最近收盘价差≤3%
                dev = self._check_deviation(symbol, price)
                if dev:
                    self._log(f"网格卖出被拦截: {dev}")
                    remains.append(lv)
                    continue
                order, err = self.trader.place_order(symbol, 'sell', 'market', qty)
                if err:
                    if 'insufficient balance' in str(err).lower():
                        # 仍超出持仓：保留该格，等待后续可用金额，避免误丢
                        self._log(f"网格卖出数量({qty:.8f})超出持仓，保留该格")
                        remains.append(lv)
                    else:
                        self._log(f"网格卖出失败: {err}")
                        remains.append(lv)
                    continue
                self.status['sell_count'] += 1
                profit = qty * price - qty * lv['buy_price']
                # 盈利计入复利池：可用买入金额 = 原金额 + 本次已实现盈利
                self.status['buy_balance'] = round((self.status.get('buy_balance', 0.0) or 0.0) + profit, 8)
                cost_part = qty * lv['buy_price']  # 已卖出部分的持仓成本
                self.status['position_cost'] = round((self.status.get('position_cost', 0.0) or 0.0) - cost_part, 8)
                self.status['last_trade'] = {'time': datetime.now().isoformat(), 'action': 'SELL',
                                             'symbol': symbol, 'price': price, 'quantity': qty,
                                             'value_usdt': qty * price, 'profit': round(profit, 2), 'mode': 'grid'}
                self._log(f"网格卖出: {qty:.6f} {symbol} @ {price} (盈利{profit:.2f}U, 可用买入金额→{self.status['buy_balance']:.2f}U)")
                self._mail_trade(symbol, '卖出', '网格止盈', qty, price, extra=f"盈利 {profit:.2f}U")
                sold_any = True
                # 本格卖出后不再保留
            else:
                remains.append(lv)
        grid['levels'] = remains
        grid['filled'] = len(remains)
        self.status['position'] = round(sum(l['qty'] for l in remains), 8) if remains else 0.0
        if sold_any:
            self.save_state()
            return True, 'ok'
        return False, '未达网格盈利位'

    def _run_grid_loop(self, symbol, qty_usdt, step_pct, max_levels):
        """网格模式主循环：每次拉最新价格，触发网格买卖（不依赖技术信号）"""
        self._log(f"网格模式已启动: {symbol}, 单格资金 {qty_usdt}U, 每格 {step_pct*100:.1f}%, 最多 {max_levels} 格")
        while not self._stop_event.is_set():
            try:
                price = self._refresh_last_price(symbol)
                if not price:
                    raise ValueError('无法获取价格')
                self.status['last_price'] = price
                self.status['signal'] = '网格'
                # 刷新已收盘K线收盘价（供3%偏差校验）
                self._refresh_last_close(symbol, self.status['timeframe'])
                # 先尝试卖出盈利格，再尝试买入补格
                self._grid_sell(symbol, price, step_pct)
                self._grid_buy(symbol, price, qty_usdt, step_pct, max_levels)
                # 止盈/止损监控（市价平仓）
                self._check_tp_sl(symbol, price)
                self._refresh_real_position(symbol)
                self.status['monitor_loop'] += 1
                self.status['consecutive_errors'] = 0
                self._mail_reset()
            except Exception as e:
                self.status['last_error'] = str(e)
                self.status['consecutive_errors'] = self.status.get('consecutive_errors', 0) + 1
                self._log(f"网格运行异常({self.status['consecutive_errors']}次): {e}")
                if self.status['consecutive_errors'] >= 3:
                    self._mail_alert(str(e), self.status['consecutive_errors'])
                    self._reconnect()
            self._stop_event.wait(self.status['interval'] or 30)

    # ---------- 下单 ----------
    def _refresh_last_close(self, symbol, timeframe):
        """拉取最近一根**已收盘**K线收盘价（市价单3%偏差校验基准）。
        标准模式在 _compute_signal 已更新；网格模式需在循环里单独获取。"""
        try:
            candles, err = self.trader.get_ohlcv(symbol, timeframe, limit=2)
            if not err and candles:
                self.status['last_close'] = float(candles[-1]['close'])
                return self.status['last_close']
        except Exception:
            pass
        return 0.0

    def _check_deviation(self, symbol, price):
        """市价单前校验：当前价与最近一根已收盘K线收盘价价差≤3%。
        超限则拒绝下单并返回错误提示（防价格跳变/滑点失控）。
        返回 None=允许下单；字符串=拦截原因。"""
        if not price or price <= 0:
            return None
        close = self.status.get('last_close', 0.0) or 0.0
        if close <= 0:
            return None
        dev = abs(price - close) / close
        if dev > 0.03:
            msg = f"{symbol} 当前价 {price:.6f} 距最近收盘 {close:.6f} 偏移 {dev*100:.2f}% > 3%，已拦截市价单"
            self._log(f"⚠ {msg}")
            if mailer.is_configured():
                mailer.send_async(f"🛡️ 币安量化拦截(现货): 价格偏移超3%", msg)
            self.status['last_error'] = f"价格偏移{dev*100:.2f}%>3%，拦截"
            return f"当前价偏离上次收盘价 {dev*100:.2f}% > 3%"
        return None

    def _mail_trade(self, symbol, side, action, qty, price, extra=''):
        """下单成功邮件通知（开仓/平仓/止盈/止损均提示），包含任务名称与账户信息"""
        if not mailer.is_configured():
            return
        st = self.status
        mode = '网格' if st.get('mode') == 'grid' else '标准多策略(OR)'
        tf = st.get('timeframe') or '?'
        strategies = ' + '.join(st.get('strategies') or []) or '—'
        pos = st.get('position', 0.0) or 0.0
        avg = st.get('avg_price', 0.0) or 0.0
        acct = st.get('account_balance', 0.0) or 0.0
        buy_bal = st.get('buy_balance', 0.0) or 0.0
        real_val = st.get('real_value_usdt', 0.0) or 0.0
        task_id = self._task_id or '—'
        note = (f"备注: {extra}\n") if extra else ""
        body = (
            f"📋 量化任务: 自动现货-{task_id}（{mode}）\n"
            f"⏱ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"──────────── 交易明细 ────────────\n"
            f"品种: {symbol}\n周期: {tf}\n策略: {strategies}\n"
            f"方向: {side}\n操作: {action}\n数量: {qty}\n价格: {price}\n"
            f"{note}"
            f"──────────── 账户信息 ────────────\n"
            f"账户可用余额: {acct:.2f} USDT\n复利可用买入资金: {buy_bal:.2f} USDT\n"
            f"当前持仓: {pos}（均价 {avg:.6f}）\n持仓市值: {real_val:.2f} USDT\n"
            f"累计买入/卖出: {st.get('buy_count', 0)}/{st.get('sell_count', 0)} 次"
        )
        subject = f"💰 币安量化成交(现货|{task_id}|{side}): {symbol} 近{price}"
        mailer.send_async(subject, body)
        self._log(f"已发送成交邮件: {symbol} {side} {action}")

    def _check_tp_sl(self, symbol, price):
        """止盈/止损监控：持仓均价达到设置的止盈/止损价则市价平仓获利/止损。"""
        tp = self.status.get('take_profit_pct', 0.0) or 0.0
        sl = self.status.get('stop_loss_pct', 0.0) or 0.0
        if tp <= 0 and sl <= 0:
            return
        pos = self.status.get('position', 0.0) or 0.0
        avg = self.status.get('avg_price', 0.0) or 0.0
        if pos <= 0 or avg <= 0:
            return
        trigger = None
        pct = 0.0
        if tp > 0 and price >= avg * (1 + tp):
            trigger, pct = '止盈', tp
        elif sl > 0 and price <= avg * (1 - sl):
            trigger, pct = '止损', sl
        else:
            return
        self._log(f"{trigger}触发: {symbol} 现价 {price:.6f} 均价 {avg:.6f} ({trigger}{pct*100:.1f}%)")
        self._exit_position(symbol, trigger)

    def _exit_position(self, symbol, reason):
        """市价平掉全部现货持仓（止盈/止损/强制卖出共用）"""
        qty = self.trader.round_amount(symbol, self.status.get('position', 0.0))
        if qty <= 0:
            self._log(f"{reason}: 卖出数量精度不足，跳过")
            return
        # 3% 偏差校验，超限则本轮跳过（下轮再测，避免在极端行情下盲目平仓）
        price, _ = self.trader.get_ticker(symbol)
        dev = self._check_deviation(symbol, price)
        if dev:
            self._log(f"{reason}被拦截(价格偏移): {dev}")
            return
        order, err = self.trader.place_order(symbol, 'sell', 'market', qty)
        if err:
            self._log(f"{reason}卖出失败: {err}")
            self.status['last_error'] = f"{reason}卖出失败: {err}"
            return
        # 已实现盈利计入复利池，更新可用买入金额
        sell_value = qty * (price or 0)
        cost = self.status.get('position_cost', 0.0) or 0.0
        profit = sell_value - cost
        self.status['buy_balance'] = round((self.status.get('buy_balance', 0.0) or 0.0) + profit, 8)
        self.status['position'] = 0
        self.status['avg_price'] = 0.0
        self.status['position_cost'] = 0.0
        self.status['sell_count'] += 1
        self.status['last_trade'] = {
            'time': datetime.now().isoformat(), 'action': 'SELL', 'symbol': symbol,
            'price': price, 'quantity': qty, 'value_usdt': sell_value,
            'profit': round(profit, 2), 'reason': reason,
        }
        self._log(f"{reason}平仓: {qty:.6f} {symbol} @ {price} (盈利{profit:+.2f}U, 可用买入金额→{self.status['buy_balance']:.2f}U)")
        self._mail_trade(symbol, '卖出', reason, qty, price, extra=f"平仓价 {price}")
        self.save_state()

    def _apply_orders(self, symbol, qty_usdt, buy_pct=0.95):
        """根据信号执行买/卖。买入按 qty_usdt 金额（自动换算数量），卖出全部持仓。"""
        sig = self.status['signal']
        if sig == 1 and self.status['position'] <= 0:
            price, _ = self.trader.get_ticker(symbol)
            if not price:
                self.status['last_error'] = '无法获取价格，跳过买入'
                return 'skip'
            # 市价单价格保护：当前价与最近收盘价差≤3%，超限拦截
            dev = self._check_deviation(symbol, price)
            if dev:
                self.status['last_error'] = f'买入被拦截: {dev}'
                return 'skip'
            # 可用买入金额 = 当前复利池 buy_balance，按 buy_pct(95%) 实际投入
            buy_balance = self.status.get('buy_balance', 0.0) or 0.0
            invest = buy_balance * self.status.get('buy_pct', 0.95)
            quantity = invest / price
            if quantity <= 0:
                self.status['last_error'] = '买入数量为0，跳过'
                return 'skip'
            # 按币安数量精度取整，确保持仓与实际成交一致（防止卖出超出持仓）
            quantity = self.trader.round_amount(symbol, quantity)
            if quantity <= 0:
                self.status['last_error'] = '买入数量精度不足，跳过'
                return 'skip'
            order, err = self.trader.place_order(symbol, 'buy', 'market', quantity)
            if err:
                self.status['last_error'] = f'买入失败: {err}'
                self._log(f"买入失败: {err}")
                return 'error'
            self.status['position'] = quantity
            self.status['avg_price'] = price
            self.status['position_cost'] = round(quantity * price, 8)   # 记录持仓成本
            self.status['buy_count'] += 1
            self.status['last_trade'] = {
                'time': datetime.now().isoformat(),
                'action': 'BUY', 'symbol': symbol,
                'price': price, 'quantity': quantity,
                'value_usdt': quantity * price,
            }
            self._log(f"买入: {quantity:.6f} {symbol} @ {price} (投入{invest:.2f}U/可用{buy_balance:.2f}U)")
            self._mail_trade(symbol, '买入', '开仓', quantity, price, extra=f"开仓价 {price}")
            self.save_state()
            return 'ok'
        elif sig == -1 and self.status['position'] > 0:
            quantity = self.trader.round_amount(symbol, self.status['position'])
            if quantity <= 0:
                self.status['last_error'] = '卖出数量精度不足，跳过'
                return 'skip'
            # 市价单价格保护：当前价与最近收盘价差≤3%，超限拦截
            price, _ = self.trader.get_ticker(symbol)
            dev = self._check_deviation(symbol, price)
            if dev:
                self.status['last_error'] = f'卖出被拦截: {dev}'
                return 'skip'
            order, err = self.trader.place_order(symbol, 'sell', 'market', quantity)
            if err:
                self.status['last_error'] = f'卖出失败: {err}'
                self._log(f"卖出失败: {err}")
                return 'error'
            # 已实现盈利 = 卖出金额 − 持仓成本，计入复利池
            sell_value = quantity * (price or 0)
            cost = self.status.get('position_cost', 0.0) or 0.0
            profit = sell_value - cost
            self.status['buy_balance'] = round((self.status.get('buy_balance', 0.0) or 0.0) + profit, 8)
            self.status['position'] = 0
            self.status['avg_price'] = 0
            self.status['position_cost'] = 0.0
            self.status['sell_count'] += 1
            self.status['last_trade'] = {
                'time': datetime.now().isoformat(),
                'action': 'SELL', 'symbol': symbol,
                'price': price, 'quantity': quantity,
                'value_usdt': sell_value, 'profit': round(profit, 2),
            }
            self._log(f"卖出: {quantity:.6f} {symbol} @ {price} (盈利{profit:+.2f}U, 可用买入金额→{self.status['buy_balance']:.2f}U)")
            self._mail_trade(symbol, '卖出', '平仓(信号)', quantity, price, extra=f"平仓价 {price}")
            self.save_state()
            return 'ok'
        return 'idle'

    # ---------- 网络恢复 ----------
    def _reconnect(self, max_tries=5):
        """网络中断后重建交易所连接（更新密钥重建 ccxt 实例），直到成功或达到重试上限"""
        self._log("网络波动，尝试重新连接...")
        for i in range(max_tries):
            if self._stop_event.is_set():
                return False
            try:
                # 重建连接，同时刷新汇率/时间戳
                self.trader._create_exchange()
                self._log(f"已重新连接（第 {i + 1} 次尝试）")
                self.status['consecutive_errors'] = 0
                return True
            except Exception as e:
                self.status['last_error'] = f'重连失败: {e}'
                self._log(f"重连失败（{i + 1}/{max_tries}）: {e}")
                time.sleep(3)
        return False

    # ---------- 主循环 ----------
    def _run_loop(self, symbol, timeframe, indicator_params, qty_usdt, interval):
        self._running = True
        self.status['running'] = True
        self.status['started_at'] = datetime.now().isoformat()
        mode = self.status.get('mode', 'standard')
        mode_label = '标准多策略(OR)' if mode == 'standard' else '网格交易'
        self._log(f"自动交易已启动: {symbol} {timeframe} [{mode_label}], 每次买入 {qty_usdt} USDT, 间隔 {interval}s")
        while not self._stop_event.is_set():
            try:
                sig, df, err = self._compute_signal(symbol, timeframe, indicator_params,
                                                    strategies=self.status.get('strategies'))
                if err:
                    # 网络/数据异常：记录并按次数尝试重连，期间不丢失持仓状态
                    self.status['last_error'] = err
                    self.status['consecutive_errors'] = self.status.get('consecutive_errors', 0) + 1
                    self._log(f"信号计算失败({self.status['consecutive_errors']}次): {err}")
                    if self.status['consecutive_errors'] >= 3:
                        self._mail_alert(err, self.status['consecutive_errors'])
                        self._reconnect()
                else:
                    self.status['consecutive_errors'] = 0
                    self._mail_reset()
                    price = self._refresh_last_price(symbol)
                    # 更新信号状态
                    if sig == 1:
                        self.status['signal'] = '买入'
                    elif sig == -1:
                        self.status['signal'] = '卖出'
                    else:
                        self.status['signal'] = '观望'
                    self.status['last_price'] = price or self.status['last_price']
                    # 止盈/止损监控（市价平仓，先于买卖信号执行）
                    self._check_tp_sl(symbol, price or self.status['last_price'])
                    self._apply_orders(symbol, qty_usdt)
                    self._refresh_real_position(symbol)
                self.status['monitor_loop'] += 1
            except Exception as e:
                self.status['last_error'] = str(e)
                self.status['consecutive_errors'] = self.status.get('consecutive_errors', 0) + 1
                self._log(f"运行异常({self.status['consecutive_errors']}次): {e}")
                if self.status['consecutive_errors'] >= 3:
                    self._mail_alert(str(e), self.status['consecutive_errors'])
                    self._reconnect()
            self._stop_event.wait(interval)

    # ---------- 启停 ----------
    def start(self, symbol, timeframe, indicator_params, qty_usdt=1000, interval=30, strategies=None,
              mode='standard', step_pct=0.01, max_levels=12,
              take_profit_pct=0.0, stop_loss_pct=0.0, task_id=None):
        """启动自动交易线程

        Args:
            mode: 策略模式。standard=标准多策略OR；grid=网格交易
            step_pct: 网格每格涨跌幅（如 0.01=1%）
            max_levels: 网格最大买入格数
            take_profit_pct: 止盈比例（相对持仓均价，0=不启用）
            stop_loss_pct: 止损比例（相对持仓均价，0=不启用）
        """
        with self._lock:
            if self._running:
                return False, '已在运行中'
            self.reset_status()
            self._mail_count, self._mail_last_ts = 0, 0.0  # 新一轮运行重置邮件告警序列
            self.status['symbol'] = symbol
            self.status['timeframe'] = timeframe
            self.status['qty_usdt'] = qty_usdt
            self.status['buy_balance'] = float(qty_usdt)  # 复利起点 = 初始预算（本金+累计盈利）
            self.status['position_cost'] = 0.0
            self.status['interval'] = interval
            self.status['strategies'] = strategies or []
            self.status['mode'] = mode
            self.status['grid']['step_pct'] = step_pct
            self.status['grid']['max_levels'] = max_levels
            self.status['take_profit_pct'] = float(take_profit_pct or 0)
            self.status['stop_loss_pct'] = float(stop_loss_pct or 0)
            self._stop_event.clear()
            # 先生成任务ID与日志文件（在线程启动前，确保首条日志也能落盘）
            # 恢复任务时复用旧 task_id，任务列表不新增、日志续写
            self._task_id = task_id or (datetime.now().strftime('%Y%m%d%H%M%S') + f"{int(time.time() * 1000) % 1000:03d}")
            self.log_file = os.path.join(LOG_DIR, f'spot_{self._task_id}.log')
            # 立刻标记为运行中（不依赖后台线程是否已执行到置位处，避免页面显示"未运行"）
            self._running = True
            self.status['running'] = True
            self.status['started_at'] = datetime.now().isoformat()
            if mode == 'grid':
                self._thread = threading.Thread(
                    target=self._run_grid_loop,
                    args=(symbol, qty_usdt, step_pct, max_levels),
                    daemon=True,
                )
            else:
                self._thread = threading.Thread(
                    target=self._run_loop,
                    args=(symbol, timeframe, indicator_params, qty_usdt, interval),
                    daemon=True,
                )
            self._thread.start()
            self.save_state()
            self._save_task()  # 记录任务，供服务崩溃后一键恢复
            return True, '已启动'

    def stop(self):
        """停止自动交易线程"""
        with self._lock:
            if not self._running:
                return False, '未在运行'
            self._stop_event.set()
            self._running = False
            self.status['running'] = False
        # 注意：_log/save_state 须在持有 _lock 的代码块之外调用，
        # 否则 _log/内部对同一把锁再加锁，而 _lock 是不可重入锁，会死锁导致"停止很慢"。
        self._log("自动交易已停止")
        self.save_state()
        self._update_task_status('stopped')
        return True, '已停止'

    def get_status(self):
        """获取当前状态快照"""
        with self._lock:
            s = dict(self.status)
            s['log'] = list(self.status['log'])
            return s
