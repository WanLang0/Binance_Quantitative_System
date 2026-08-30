# -*- coding: utf-8 -*-
"""
自动合约交易引擎（USDT 永续）
复用 AutoTrader 的信号计算（标准多策略OR / 网格交易），
但面向合约：按【张数】下单、支持做多/做空、跟踪持仓方向、显示未实现盈亏与强平价。
通过独立后台线程运行，周期拉K线 → 计算指标 → 校验信号 → 调用 FuturesTrader 开平仓。
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
STATE_FILE = os.path.join('data', 'auto_futures_state.json')
# 量化任务历史列表（用于崩溃后查看/手动恢复）
TASKS_FILE = os.path.join('data', 'futures_tasks.json')
# 心跳告警阈值：连续失败达到该次数触发告警
ALERT_THRESHOLD = 3


class AutoFutures:
    """实时自动合约交易引擎（单交易对，独立后台线程）"""

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
        self.reset_status()

    # ---------- 状态管理 ----------
    @property
    def state_file(self):
        return STATE_FILE

    def reset_status(self):
        self.status = {
            'running': False,
            'symbol': 'BTC/USDT',
            'timeframe': '15m',
            'strategies': [],           # 已选的策略名称列表
            'mode': 'standard',         # 策略模式：standard/grid
            'long_only': False,         # 仅做多：卖出信号只平仓不开空（验证过的安全模式）
            'qty_usdt': 1000,           # 每次开仓金额（USDT）
            'interval': 30,             # 轮询间隔（秒）
            'leverage': self.leverage,  # 杠杆倍数
            'stop_pct': 0.05,           # 断线保护止损比例（0=不挂保护单）
            'stop_order_id': None,      # 交易所侧止损保护单ID
            'stop_price': 0.0,          # 止损保护触发价
            'side': 'none',             # 持仓方向：long(做多)/short(做空)/none(空仓)
            'position': 0,              # 当前持仓张数（合约）
            'entry_price': 0.0,         # 开仓均价
            'last_price': 0.0,          # 最近价格
            'signal': '等待',           # 最近信号
            'unrealized_pnl': 0.0,      # 未实现盈亏（USDT）
            'liquidation_price': 0.0,   # 强平价
            'margin': 0.0,              # 占用保证金
            'account_balance': 0.0,     # 账户可用 USDT 余额
            'grid': {
                'levels': [],           # [{buy_price, qty}] 已买入的网格持仓
                'filled': 0,            # 已买入格数
                'total_invest': 0,      # 网格累计投入资金
                'step_pct': 0.01,       # 每格百分比
                'max_levels': 12,       # 最大格数
            },
            'last_signal_time': None,
            'last_trade': None,
            'last_loop_time': None,      # 心跳：最近一次成功轮询时间
            'alerts': [],                # 告警记录（最近20条）
            'buy_count': 0,
            'sell_count': 0,
            'monitor_loop': 0,
            'log': [],                  # 最近日志（最多 50 条）
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
            with open(os.path.join('data', 'futures_alerts.log'), 'a', encoding='utf-8') as f:
                f.write(line + "\n")
        except Exception:
            pass

    # ---------- 量化任务历史（崩溃恢复用） ----------
    def _save_task(self):
        """启动时把本次量化配置写入任务列表（保留最近20条），供崩溃后手动恢复"""
        rec = {
            'id': datetime.now().strftime('%Y%m%d%H%M%S') + f"{int(time.time() * 1000) % 1000:03d}",
            'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': self.status['symbol'],
            'timeframe': self.status['timeframe'],
            'strategies': self.status['strategies'],
            'mode': self.status['mode'],
            'long_only': self.status.get('long_only', False),
            'qty_usdt': self.status['qty_usdt'],
            'interval': self.status['interval'],
            'leverage': self.status['leverage'],
            'stop_pct': self.status.get('stop_pct', 0),
            'grid_step': self.status['grid'].get('step_pct', 0.01),
            'grid_max_levels': self.status['grid'].get('max_levels', 12),
            'status': 'running',
            'last_active': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        self._task_id = rec['id']
        tasks = self._read_tasks()
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
                'symbol': self.status['symbol'],
                'timeframe': self.status['timeframe'],
                'strategies': self.status['strategies'],
                'mode': self.status['mode'],
                'long_only': self.status.get('long_only', False),
                'qty_usdt': self.status['qty_usdt'],
                'interval': self.status['interval'],
                'leverage': self.status['leverage'],
                'stop_pct': self.status.get('stop_pct', 0),
                'side': self.status['side'],
                'position': self.status['position'],
                'entry_price': self.status['entry_price'],
                'grid': self.status['grid'],
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
            self.status['symbol'] = data.get('symbol', self.status['symbol'])
            self.status['timeframe'] = data.get('timeframe', self.status['timeframe'])
            self.status['strategies'] = data.get('strategies', self.status['strategies'])
            self.status['mode'] = data.get('mode', self.status['mode'])
            self.status['long_only'] = data.get('long_only', False)
            self.status['qty_usdt'] = data.get('qty_usdt', self.status['qty_usdt'])
            self.status['interval'] = data.get('interval', self.status['interval'])
            self.status['leverage'] = data.get('leverage', self.status['leverage'])
            self.status['stop_pct'] = data.get('stop_pct', self.status.get('stop_pct', 0))
            self.status['side'] = data.get('side', self.status['side'])
            self.status['position'] = data.get('position', self.status['position'])
            self.status['entry_price'] = data.get('entry_price', self.status['entry_price'])
            self.status['last_loop_time'] = data.get('last_loop_time')
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
        """从交易所刷新真实合约持仓（方向/张数/均价/未实现盈亏/强平价）与 USDT 余额"""
        try:
            prev_pos = self.status.get('position', 0)
            prev_side = self.status.get('side', 'none')
            # 余额
            bal_list, _ = self.trader.get_balance()
            if bal_list:
                usdt_asset = next((b for b in bal_list if b['asset'] == 'USDT'), None)
                self.status['account_balance'] = round(float(usdt_asset['free']) if usdt_asset else 0.0, 2)
            # 持仓
            positions, _ = self.trader.get_positions(symbol)
            pos = next((p for p in positions if p.get('symbol') == symbol), None)
            if not pos or pos.get('contracts', 0) <= 0:
                # 外部平仓检测：引擎认为有仓但交易所已无仓（多为止损保护单触发）
                if prev_pos and prev_pos > 0 and prev_side != 'none':
                    self._alert(f"检测到持仓已被外部平仓({symbol}，可能为止损保护单触发)，清理挂单并同步状态")
                    self.status['sell_count'] += 1
                    try:
                        self.trader.cancel_all_orders(symbol)
                    except Exception:
                        pass
                    self.status['stop_order_id'] = None
                    self.status['stop_price'] = 0.0
                    self.save_state()
                self.status['position'] = 0
                self.status['side'] = 'none'
                self.status['entry_price'] = 0.0
                self.status['unrealized_pnl'] = 0.0
                self.status['liquidation_price'] = 0.0
                self.status['margin'] = 0.0
            else:
                self.status['position'] = pos['contracts']
                self.status['side'] = pos.get('side', 'long')
                self.status['entry_price'] = pos.get('entry_price', 0.0) or 0.0
                self.status['unrealized_pnl'] = pos.get('unrealized_pnl', 0.0) or 0.0
                self.status['liquidation_price'] = pos.get('liquidation_price', 0.0) or 0.0
                self.status['margin'] = pos.get('margin', 0.0) or 0.0
                # 保护单自愈：有持仓但保护单缺失（如服务重启/挂单被撤）时自动补挂
                if (self.status.get('stop_pct', 0) > 0 and not self.status.get('stop_order_id')
                        and self.status.get('running') and self.status.get('mode') == 'standard'):
                    self._place_stop_protection(symbol, self.status['side'],
                                                self.status['entry_price'], self.status['position'])
        except Exception as e:
            self.status['last_error'] = f"刷新持仓失败: {e}"

    # ---------- 信号计算 ----------
    def _compute_signal(self, symbol, timeframe, indicator_params):
        """拉取K线 → 计算指标 → 返回当前根信号（1=做多, -1=做空/平多, 0=无）"""
        candles, err = self.trader.get_ohlcv(symbol, timeframe, limit=200)
        if err or not candles:
            return 0, None, err
        df = pd.DataFrame(candles)
        df['timestamp'] = pd.to_datetime(df['ts'], unit='ms')
        df = df.set_index('timestamp')
        df = TechnicalIndicators.calculate_all_indicators(df, indicator_params)
        engine = BacktestEngine(timeframe=timeframe, signal_mode='or')
        signals = engine.calculate_signals(df, indicator_params)
        sig = int(signals.iloc[-1]) if not signals.empty else 0
        return sig, df, None

    # ---------- 合约下单 ----------
    def _calc_contracts(self, symbol, usdt_amount):
        """按 USDT 金额和杠杆，结合张数精度计算可开仓的张数"""
        price = self._refresh_last_price(symbol)
        if not price or price <= 0:
            return 0, price
        # 名义价值 = usdt * leverage；张数 = 名义价值 / 价格
        notional = usdt_amount * self.leverage
        contracts = notional / price
        contracts = self.trader.round_amount(symbol, contracts)
        return contracts, price

    def _place_stop_protection(self, symbol, side, entry_price, contracts):
        """开仓后在交易所挂 STOP_MARKET + reduceOnly 止损保护单。
        服务宕机/断线时交易所自动触发平仓，作为最后兜底。"""
        pct = self.status.get('stop_pct', 0) or 0
        if pct <= 0 or entry_price <= 0 or contracts <= 0:
            return
        stop_price = entry_price * (1 - pct) if side == 'long' else entry_price * (1 + pct)
        close_side = 'sell' if side == 'long' else 'buy'
        order, err = self.trader.place_stop_order(symbol, close_side, stop_price, contracts)
        if err:
            self._alert(f"止损保护单挂单失败({symbol} {side} 触发价{stop_price:.4f}): {err}")
            return
        self.status['stop_order_id'] = order.get('id')
        self.status['stop_price'] = stop_price
        self._log(f"已挂止损保护单: {close_side} {contracts}张 @ {stop_price:.4f} (距开仓{pct*100:.1f}%)")
        self.save_state()

    def _apply_orders(self, symbol, qty_usdt):
        """根据信号执行合约开/平仓。开多/空按 qty_usdt 金额（换算张数），平仓清空持仓。"""
        sig = self.status['signal']
        current_side = self.status.get('side', 'none')
        position = self.status.get('position', 0)

        if sig == 1:  # 做多信号
            if position > 0 and current_side == 'long':
                # 已持多仓：延续，不动作
                return 'idle'
            if position > 0 and current_side == 'short':
                # 持空仓遇做多信号：平空（reduceOnly buy），与平多对称，仅平仓不反手
                self._close_position(symbol, 'short')
                return 'closed'
            contracts, price = self._calc_contracts(symbol, qty_usdt)
            if contracts <= 0:
                self.status['last_error'] = '开仓张数为0，跳过'
                return 'skip'
            order, err = self.trader.place_order(symbol, 'buy', 'market', contracts)
            if err:
                self.status['last_error'] = f'开多失败: {err}'
                self._log(f"开多失败: {err}")
                return 'error'
            self.status['position'] = contracts
            self.status['side'] = 'long'
            self.status['entry_price'] = price
            self.status['buy_count'] += 1
            self.status['last_trade'] = {
                'time': datetime.now().isoformat(),
                'action': 'OPEN_LONG', 'symbol': symbol,
                'price': price, 'contracts': contracts,
                'notional': contracts * price,
            }
            self._log(f"开多: {contracts} 张 {symbol} @ {price} (杠杆{self.leverage}x)")
            self._place_stop_protection(symbol, 'long', price, contracts)
            self.save_state()
            return 'ok'
        elif sig == -1:  # 做空/平多信号
            if current_side == 'long' and position > 0:
                # 平多仓（reduceOnly sell）
                self._close_position(symbol, 'long')
                return 'closed'
            if position > 0 and current_side == 'short':
                return 'idle'
            # 仅做多模式：卖出信号只平多不开空（ETH 4h KDJ 8/5 等验证配置用）
            if self.status.get('long_only'):
                return 'idle'
            # 空仓时做空
            contracts, price = self._calc_contracts(symbol, qty_usdt)
            if contracts <= 0:
                self.status['last_error'] = '开空张数为0，跳过'
                return 'skip'
            order, err = self.trader.place_order(symbol, 'sell', 'market', contracts)
            if err:
                self.status['last_error'] = f'开空失败: {err}'
                self._log(f"开空失败: {err}")
                return 'error'
            self.status['position'] = contracts
            self.status['side'] = 'short'
            self.status['entry_price'] = price
            self.status['buy_count'] += 1
            self.status['last_trade'] = {
                'time': datetime.now().isoformat(),
                'action': 'OPEN_SHORT', 'symbol': symbol,
                'price': price, 'contracts': contracts,
                'notional': contracts * price,
            }
            self._log(f"开空: {contracts} 张 {symbol} @ {price} (杠杆{self.leverage}x)")
            self._place_stop_protection(symbol, 'short', price, contracts)
            self.save_state()
            return 'ok'
        return 'idle'

    def _close_position(self, symbol, side):
        """平掉当前持仓（reduceOnly），按张数"""
        position = self.status.get('position', 0)
        if position <= 0:
            return
        contracts = self.trader.round_amount(symbol, position)
        if contracts <= 0:
            return
        # 平多：reduceOnly sell；平空：reduceOnly buy
        side_cmd = 'sell' if side == 'long' else 'buy'
        order, err = self.trader.place_order(symbol, side_cmd, 'market', contracts, reduce_only=True)
        price, _ = self.trader.get_ticker(symbol)
        if err:
            self.status['last_error'] = f'平仓失败: {err}'
            self._log(f"平仓失败: {err}")
            return
        self.status['position'] = 0
        self.status['side'] = 'none'
        self.status['entry_price'] = 0.0
        self.status['sell_count'] += 1
        # 平仓后撤销该品种全部挂单（清理止损保护单，防止残留触发反向开仓）
        _, cerr = self.trader.cancel_all_orders(symbol)
        if cerr:
            self._log(f"清理挂单失败(可能无挂单): {cerr}")
        self.status['stop_order_id'] = None
        self.status['stop_price'] = 0.0
        self.status['last_trade'] = {
            'time': datetime.now().isoformat(),
            'action': 'CLOSE', 'symbol': symbol,
            'price': price, 'contracts': contracts,
            'pnl': self.status.get('unrealized_pnl', 0.0),
        }
        self._log(f"平仓: {contracts} 张 {symbol} @ {price}")
        self.save_state()

    # ---------- 网格交易（合约版） ----------
    def _grid_buy(self, symbol, price, qty_usdt, step_pct, max_levels):
        grid = self.status['grid']
        levels = grid.get('levels', [])
        if levels:
            last_buy = levels[-1]['buy_price']
            if price > last_buy * (1 - step_pct):
                return False, '未达新网格位'
        if len(levels) >= max_levels:
            return False, '网格已满'
        # 按 USDT 金额换算张数
        contracts, _ = self._calc_contracts(symbol, qty_usdt)
        if contracts <= 0:
            return False, '开仓张数为0'
        order, err = self.trader.place_order(symbol, 'buy', 'market', contracts)
        if err:
            self._log(f"网格买入失败: {err}")
            return False, err
        if order:
            filled = order.get('filled') or order.get('amount') or contracts
            try:
                filled = float(filled)
                if filled > 0:
                    contracts = self.trader.round_amount(symbol, filled)
            except (TypeError, ValueError):
                pass
        if contracts <= 0:
            return False, '实际成交张数为0'
        levels.append({'buy_price': price, 'qty': contracts})
        grid['levels'] = levels
        grid['filled'] = len(levels)
        grid['total_invest'] = round(grid.get('total_invest', 0) + contracts * price, 2)
        self.status['position'] = round(self.status.get('position', 0) + contracts, 8)
        self.status['side'] = 'long'
        self.status['buy_count'] += 1
        self.status['last_trade'] = {
            'time': datetime.now().isoformat(), 'action': 'BUY',
            'symbol': symbol, 'price': price, 'contracts': contracts,
            'value_usdt': contracts * price, 'mode': 'grid'
        }
        self._log(f"网格买入(格{len(levels)}): {contracts} 张 {symbol} @ {price}")
        self.save_state()
        return True, 'ok'

    def _grid_sell(self, symbol, price, step_pct):
        grid = self.status['grid']
        levels = grid.get('levels', [])
        if not levels:
            return False, '无网格持仓'
        self._refresh_real_position(symbol)
        real_qty = self.status.get('position', 0.0) or 0.0
        sold_any = False
        remains = []
        for lv in levels:
            if not sold_any and price >= lv['buy_price'] * (1 + step_pct):
                qty = self.trader.round_amount(symbol, lv['qty'])
                if qty <= 0:
                    remains.append(lv)
                    continue
                if qty > real_qty:
                    qty = self.trader.round_amount(symbol, real_qty)
                    if qty <= 0:
                        remains.append(lv)
                        continue
                order, err = self.trader.place_order(symbol, 'sell', 'market', qty, reduce_only=True)
                if err:
                    self._log(f"网格卖出失败: {err}")
                    remains.append(lv)
                    continue
                self.status['sell_count'] += 1
                profit = qty * price - qty * lv['buy_price']
                self.status['last_trade'] = {
                    'time': datetime.now().isoformat(), 'action': 'SELL',
                    'symbol': symbol, 'price': price, 'contracts': qty,
                    'value_usdt': qty * price, 'profit': round(profit, 2), 'mode': 'grid'
                }
                self._log(f"网格卖出: {qty} 张 {symbol} @ {price} (盈利{profit:.2f}U)")
                sold_any = True
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
        self._log(f"网格模式已启动: {symbol}, 单格资金 {qty_usdt}U, 每格 {step_pct*100:.1f}%, 最多 {max_levels} 格")
        while not self._stop_event.is_set():
            try:
                price = self._refresh_last_price(symbol)
                if not price:
                    raise ValueError('无法获取价格')
                self.status['last_price'] = price
                self.status['signal'] = '网格'
                self._grid_sell(symbol, price, step_pct)
                self._grid_buy(symbol, price, qty_usdt, step_pct, max_levels)
                self._refresh_real_position(symbol)
                self.status['monitor_loop'] += 1
                self.status['consecutive_errors'] = 0
            except Exception as e:
                self.status['last_error'] = str(e)
                self.status['consecutive_errors'] = self.status.get('consecutive_errors', 0) + 1
                self._log(f"网格运行异常({self.status['consecutive_errors']}次): {e}")
                if self.status['consecutive_errors'] >= 3:
                    self._reconnect()
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

    # ---------- 主循环 ----------
    def _run_loop(self, symbol, timeframe, indicator_params, qty_usdt, interval):
        self._running = True
        self.status['running'] = True
        self.status['started_at'] = datetime.now().isoformat()
        mode = self.status.get('mode', 'standard')
        if mode == 'grid':
            mode_label = '网格交易'
        elif self.status.get('long_only'):
            mode_label = '标准策略(仅做多)'
        else:
            mode_label = '标准多策略(OR)'
        self._log(f"自动合约已启动: {symbol} {timeframe} [{mode_label}], 每次开仓 {qty_usdt} USDT, 杠杆{self.leverage}x, 间隔 {interval}s")
        # 设置杠杆
        self.trader.set_leverage(self.leverage, symbol)
        alerted = False  # 本轮故障周期是否已发过告警（恢复后重置）
        # 邮件告警序列：达到阈值立即发第1封，之后每10分钟一封、共3封；恢复后另发一封恢复邮件并重置
        MAIL_INTERVAL, MAIL_MAX = 600, 3
        mail_count, last_mail_ts = 0, 0.0
        fault_start_ts = 0.0
        mail_no_cfg_noted = False  # "邮箱未配置"每轮故障只提示一次，避免刷屏

        def _send_alert_mail(reason, errors):
            nonlocal mail_count, last_mail_ts, fault_start_ts, mail_no_cfg_noted
            if not mailer.is_configured():
                if not mail_no_cfg_noted:
                    mail_no_cfg_noted = True
                    self._log("⚠ 达到告警阈值但邮箱未配置，无法发送邮件（请在「个人设置」页填写QQ邮箱+SMTP授权码）")
                return
            if mail_count >= MAIL_MAX:
                return
            if mail_count > 0 and time.time() - last_mail_ts < MAIL_INTERVAL:
                return
            if mail_count == 0:
                fault_start_ts = time.time()
            body = (f"告警时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"品种: {symbol}  周期: {timeframe}\n"
                    f"故障: {reason}\n连续失败: {errors} 次\n\n"
                    f"引擎仍在自动重连，持仓已由交易所端 STOP_MARKET 保护单托底。\n"
                    f"本故障期内邮件提醒每10分钟一封、最多{MAIL_MAX}封（第 {mail_count + 1} 封）；恢复后另发一封恢复邮件。")
            mailer.send_async(f"⚠ 币安量化告警({mail_count + 1}/{MAIL_MAX}): 无法访问币安", body)
            self._log(f"已触发邮件告警({mail_count + 1}/{MAIL_MAX})")
            mail_count += 1
            last_mail_ts = time.time()

        def _send_recover_mail():
            """网络恢复：若本轮故障发过告警邮件，补发一封恢复通知并重置序列"""
            nonlocal mail_count, fault_start_ts, mail_no_cfg_noted
            if mail_count > 0 and mailer.is_configured():
                dur = (time.time() - fault_start_ts) / 60
                body = (f"恢复时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"品种: {symbol}  周期: {timeframe}\n"
                        f"故障持续: 约{dur:.0f}分钟（自首封告警起算）\n本轮已发告警: {mail_count}封\n\n"
                        f"引擎已恢复正常轮询，无需操作。")
                mailer.send_async("✅ 币安量化告警恢复: 币安访问已恢复", body)
                self._log("已发送恢复通知邮件")
            mail_count = 0
            mail_no_cfg_noted = False

        while not self._stop_event.is_set():
            try:
                sig, df, err = self._compute_signal(symbol, timeframe, indicator_params)
                if err:
                    self.status['last_error'] = err
                    self.status['consecutive_errors'] = self.status.get('consecutive_errors', 0) + 1
                    self._log(f"信号计算失败({self.status['consecutive_errors']}次): {err}")
                    if self.status['consecutive_errors'] >= ALERT_THRESHOLD:
                        if not alerted:
                            self._alert(f"轮询连续失败{self.status['consecutive_errors']}次({symbol})，正在自动重连；若长时间未恢复请检查网络/代理")
                            alerted = True
                        _send_alert_mail(err, self.status['consecutive_errors'])
                        self._reconnect()
                else:
                    self.status['consecutive_errors'] = 0
                    if alerted:
                        alerted = False
                        _send_recover_mail()
                        self._log("轮询已恢复正常（邮件告警序列已重置）")
                    self.status['last_loop_time'] = datetime.now().isoformat()  # 心跳
                    price = self._refresh_last_price(symbol)
                    if sig == 1:
                        self.status['signal'] = '买入(多)'
                    elif sig == -1:
                        self.status['signal'] = '卖出(空)'
                    else:
                        self.status['signal'] = '观望'
                    self.status['last_price'] = price or self.status['last_price']
                    self._apply_orders(symbol, qty_usdt)
                    self._refresh_real_position(symbol)
                self.status['monitor_loop'] += 1
            except Exception as e:
                self.status['last_error'] = str(e)
                self.status['consecutive_errors'] = self.status.get('consecutive_errors', 0) + 1
                self._log(f"运行异常({self.status['consecutive_errors']}次): {e}")
                if self.status['consecutive_errors'] >= 3:
                    self._reconnect()
            self._stop_event.wait(interval)

    # ---------- 启停 ----------
    def start(self, symbol, timeframe, indicator_params, qty_usdt=1000, interval=30, strategies=None,
              mode='standard', step_pct=0.01, max_levels=12, stop_pct=0.05, long_only=False):
        with self._lock:
            if self._running:
                return False, '已在运行中'
            # 若之前有未平仓，先提示但不强制平仓（保留状态）
            self.reset_status()
            self.status['symbol'] = symbol
            self.status['timeframe'] = timeframe
            self.status['qty_usdt'] = qty_usdt
            self.status['interval'] = interval
            self.status['strategies'] = strategies or []
            self.status['mode'] = mode
            self.status['long_only'] = bool(long_only) and mode == 'standard'  # 网格模式不适用
            self.status['leverage'] = self.leverage
            self.status['stop_pct'] = stop_pct or 0
            self.status['grid']['step_pct'] = step_pct
            self.status['grid']['max_levels'] = max_levels
            self._stop_event.clear()
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
            self._save_task()  # 记录本次量化任务（崩溃后可从任务列表手动恢复）
            self.save_state()
            return True, '已启动'

    def stop(self):
        with self._lock:
            if not self._running:
                return False, '未在运行'
            self._stop_event.set()
            self._running = False
            self.status['running'] = False
        # 移出锁外调用（_lock 不可重入，避免死锁）
        self._log("自动合约已停止")
        self._update_task_status('stopped')
        self.save_state()
        return True, '已停止'

    def list_tasks(self):
        """返回量化任务历史列表（最新在前）"""
        return self._read_tasks()

    def delete_task(self, task_id):
        """删除指定的量化任务记录（不影响正在运行的引擎）"""
        tasks = self._read_tasks()
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
            s['log'] = list(self.status['log'])
            return s
