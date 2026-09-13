# -*- coding: utf-8 -*-
"""横截面动量实盘量化引擎（币安 USDT 永续合约，测试网 / 主网可切换）。

与「横截面动量回测」共用同一策略口径（R14 排名，Top/Bottom 对称，dollar-neutral）：
  - R14 = close / close.shift(14) - 1
  - 流动性过滤：24h 成交额 = volume × close > liq_min（默认 100M USDT）
  - 波动率过滤：20d 年化波动率（vol20 = pct_change().rolling(20).std() × sqrt(365)）
  - 排名：按 R14 降序，Top n 做多、Bottom n 做空（n = 候选数 × top_frac）
  - 绝对动量闸门（abs_mom）：只做多 R14>0、做空 R14<0（震荡期自动空仓）
  - 资金分配：dollar-neutral，每个仓位名义 = 权益 × 杠杆 / (多头数 + 空头数)

与「虚拟币综合量化」同为量化任务：独立后台线程、启停、状态快照、日志、
资金曲线、任务历史（崩溃后可恢复）。本模块复用 futures_trader.FuturesTrader
接入币安合约（测试网 set_sandbox_mode / 主网 fapi），不复制其网络逻辑。
"""
import json
import os
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from futures_trader import FuturesTrader

# ---- 文件路径（与综合量化独立命名，避免互相污染） ----
LOG_DIR = os.path.join('data', 'logs')
TASKS_FILE = os.path.join('data', 'momentum_tasks.json')

# 候选币池：市值前30回测标的 ∩ 币安 USDT 永续（与回测 BASES 完全一致）
BASES = ['BTC', 'ETH', 'BNB', 'XRP', 'SOL', 'TRX', 'HYPE', 'ZEC', 'DOGE', 'XMR',
         'LINK', 'ADA', 'XLM', 'BCH', 'CC', 'LTC', 'UNI', 'GRAM', 'HBAR', 'AVAX',
         'SUI', 'NEAR', 'M', 'TAO', 'ASTER', 'AAVE', 'ONDO', 'MORPHO', 'DOT', 'ICP']

NAMES = {
    'BTC': '比特币', 'ETH': '以太坊', 'BNB': '币安币', 'XRP': '瑞波币', 'SOL': 'Solana',
    'TRX': '波场', 'HYPE': 'Hyperliquid', 'ZEC': 'Zcash', 'DOGE': '狗狗币', 'XMR': '门罗币',
    'LINK': 'Chainlink', 'ADA': '艾达币', 'XLM': '恒星币', 'BCH': '比特现金', 'CC': 'Canton',
    'LTC': '莱特币', 'UNI': 'Uniswap', 'GRAM': 'Gram', 'HBAR': 'Hedera', 'AVAX': '雪崩',
    'SUI': 'Sui', 'NEAR': 'NEAR协议', 'M': 'MemeCore', 'TAO': 'Bittensor',
    'ASTER': 'Aster', 'AAVE': 'Aave', 'ONDO': 'Ondo', 'MORPHO': 'Morpho',
    'DOT': '波卡', 'ICP': '互联网计算机',
}

DEFAULT_BUY_PCT = 0.95   # 名义安全系数（保留部分现金做保证金缓冲）


def tasks_file():
    return TASKS_FILE


def state_file(task_id=None):
    name = f'momentum_state_{task_id}.json' if task_id else 'momentum_state.json'
    return os.path.join('data', name)


def equity_file(task_id=None):
    if not task_id:
        return None
    return os.path.join('data', f'momentum_equity_{task_id}.json')


def log_prefix():
    return 'momentum_'


def load_equity_points(task_id=None):
    f = equity_file(task_id)
    if not f or not os.path.exists(f):
        return []
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            pts = json.load(fh)
        return [(float(t), float(v)) for t, v in pts if isinstance(t, (int, float))]
    except Exception:
        return []


class MomentumTrader:
    """横截面动量实盘引擎（单任务 = 一个候选池的定期横截面调仓）"""

    def __init__(self, api_key='', api_secret='', trader=None, leverage=1, testnet=True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.leverage = leverage
        self.testnet = testnet
        self.trader = trader or FuturesTrader(api_key, api_secret, testnet=testnet, leverage=leverage)
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._running = False
        self._task_id = None
        self.log_file = None
        self._last_rebalance_ts = 0.0
        self.reset_status()

    # ---------- 状态 ----------
    @property
    def state_file(self):
        return state_file(getattr(self, '_task_id', None))

    def reset_status(self):
        self.status = {
            'running': False,
            'name': '横截面动量量化任务',
            'total_fund': 10000.0,
            'leverage': self.leverage,
            'rebal_days': 5,             # 调仓周期（自然日）
            'top_frac': 0.20,            # Top/Bottom 比例
            'liq_min': 100_000_000.0,    # 流动性阈值（USDT 24h 成交额）
            'long_only': False,          # 只做多（砍掉做空腿）
            'abs_mom': False,            # 绝对动量闸门
            'vol_max': None,             # 波动率过滤（年化，None=关闭）
            'interval': 30,              # 轮询间隔（秒）
            'buy_pct': DEFAULT_BUY_PCT,
            'symbols': [],               # 候选币状态
            'longs': [],                 # 当前做多名单
            'shorts': [],                # 当前做空名单
            'n_eligible': 0,             # 通过流动性/波动率过滤的候选数
            'dispersion': 0.0,           # 横截面离散度 std(R14)
            'realized_pnl': 0.0,         # 累计已实现盈亏
            'account_balance': 0.0,
            'last_loop_time': None,
            'last_rebalance_time': None,
            'next_rebalance_time': None,
            'signal': '等待',
            'buy_count': 0,
            'sell_count': 0,
            'monitor_loop': 0,
            'alerts': [],
            'log': [],
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
        if self.log_file:
            try:
                os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(line + "\n")
            except Exception:
                pass

    def _alert(self, msg):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        with self._lock:
            self.status['alerts'].append(line)
            if len(self.status['alerts']) > 20:
                self.status['alerts'] = self.status['alerts'][-20:]
        self._log(f"⚠ 告警: {msg}")

    # ---------- 任务历史 ----------
    def _read_tasks(self):
        try:
            if os.path.exists(TASKS_FILE):
                with open(TASKS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    def _save_task(self):
        rec = {
            'id': self._task_id,
            'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'name': self.status['name'],
            'total_fund': self.status['total_fund'],
            'leverage': self.status['leverage'],
            'rebal_days': self.status['rebal_days'],
            'top_frac': self.status['top_frac'],
            'liq_min': self.status['liq_min'],
            'long_only': self.status['long_only'],
            'abs_mom': self.status['abs_mom'],
            'vol_max': self.status['vol_max'],
            'interval': self.status['interval'],
            'buy_pct': self.status['buy_pct'],
            'testnet': bool(self.testnet),
            'status': 'running',
            'last_active': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        tasks = self._read_tasks()
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

    # ---------- 状态持久化 ----------
    def save_state(self):
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            data = {
                'task_id': self._task_id,
                'name': self.status['name'],
                'total_fund': self.status['total_fund'],
                'leverage': self.status['leverage'],
                'rebal_days': self.status['rebal_days'],
                'top_frac': self.status['top_frac'],
                'liq_min': self.status['liq_min'],
                'long_only': self.status['long_only'],
                'abs_mom': self.status['abs_mom'],
                'vol_max': self.status['vol_max'],
                'interval': self.status['interval'],
                'buy_pct': self.status['buy_pct'],
                'longs': self.status['longs'],
                'shorts': self.status['shorts'],
                'realized_pnl': self.status['realized_pnl'],
                'buy_count': self.status['buy_count'],
                'sell_count': self.status['sell_count'],
                'last_rebalance_time': self.status['last_rebalance_time'],
                'symbols': [{
                    'symbol': s['symbol'], 'side': s['side'], 'position': s['position'],
                    'entry_price': s['entry_price'],
                } for s in self.status['symbols']],
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
            for k in ('name', 'total_fund', 'leverage', 'rebal_days', 'top_frac',
                      'liq_min', 'long_only', 'abs_mom', 'vol_max', 'interval', 'buy_pct'):
                if k in data:
                    self.status[k] = data[k]
            self.status['longs'] = data.get('longs') or []
            self.status['shorts'] = data.get('shorts') or []
            self.status['realized_pnl'] = float(data.get('realized_pnl') or 0.0)
            self.status['buy_count'] = int(data.get('buy_count') or 0)
            self.status['sell_count'] = int(data.get('sell_count') or 0)
            self.status['last_rebalance_time'] = data.get('last_rebalance_time')
            saved = {s['symbol']: s for s in (data.get('symbols') or [])}
            for s in self.status['symbols']:
                src = saved.get(s['symbol'])
                if src:
                    s['side'] = src.get('side', 'none')
                    s['position'] = src.get('position', 0)
                    s['entry_price'] = src.get('entry_price', 0.0)
        except Exception:
            pass

    # ---------- 候选池构建 ----------
    def _build_symbols(self):
        return [{
            'symbol': f'{b}/USDT',
            'name': NAMES.get(b, b),
            'r14': 0.0,
            'liq': 0.0,
            'vol20': 0.0,
            'side': 'none',
            'position': 0,
            'entry_price': 0.0,
            'last_price': 0.0,
            'unrealized_pnl': 0.0,
            'pnl_pct': 0.0,
        } for b in BASES]

    # ---------- 权益 ----------
    def _task_equity(self):
        unreal = sum(float(s.get('unrealized_pnl') or 0.0) for s in self.status['symbols'])
        return round(self.status['total_fund'] + self.status.get('realized_pnl', 0.0) + unreal, 4)

    # ---------- 资金曲线 ----------
    def _record_equity(self, force=False):
        f = equity_file(self._task_id)
        if not f:
            return
        now = time.time()
        pts = load_equity_points(self._task_id)
        eq = self._task_equity()
        if pts and not force and now - pts[-1][0] < 60:
            pts[-1] = [pts[-1][0], eq]
        else:
            pts.append([round(now, 3), eq])
        compact = []
        for ts, v in pts:
            age = now - ts
            if age <= 7 * 86400:
                compact.append([ts, v])
            elif age <= 90 * 86400:
                if ts % 900 < 90:
                    compact.append([ts, v])
            elif ts % 7200 < 120:
                compact.append([ts, v])
        try:
            os.makedirs(os.path.dirname(f), exist_ok=True)
            tmp = f + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump(compact, fh)
            os.replace(tmp, f)
        except Exception:
            pass

    # ---------- 价格 / 持仓刷新 ----------
    def _refresh_prices(self):
        syms = [s['symbol'] for s in self.status['symbols']]
        prices = {}
        try:
            prices = self.trader.get_tickers(syms)
        except Exception:
            prices = {}
        for s in self.status['symbols']:
            p = prices.get(s['symbol'])
            if p:
                s['last_price'] = p

    def _refresh_positions_and_balance(self):
        try:
            bal_list, berr = self.trader.get_balance()
            if not berr and bal_list:
                usdt = next((b for b in bal_list if b['asset'] == 'USDT'), None)
                self.status['account_balance'] = round(float(usdt['free']) if usdt else 0.0, 2)
        except Exception:
            pass
        try:
            positions, perr = self.trader.get_positions()
            if perr:
                return
        except Exception:
            return
        pos_by = {}
        for p in positions:
            pos_by[p.get('symbol')] = p
            base = p.get('symbol_base') or (p.get('symbol') or '').rsplit(':', 1)[0]
            if base and base != p.get('symbol'):
                pos_by.setdefault(base, p)
        for s in self.status['symbols']:
            pos = pos_by.get(s['symbol'])
            if not pos or pos.get('contracts', 0) <= 0:
                s['position'] = 0
                s['side'] = 'none'
                s['entry_price'] = 0.0
                s['unrealized_pnl'] = 0.0
                s['pnl_pct'] = 0.0
            else:
                s['position'] = pos.get('contracts', 0)
                s['side'] = pos.get('side') or 'long'
                s['entry_price'] = pos.get('entry_price', 0.0) or 0.0
                s['unrealized_pnl'] = pos.get('unrealized_pnl', 0.0) or 0.0
                _csize = pos.get('contract_size', 1) or 1
                _lev = pos.get('leverage', self.leverage) or self.leverage
                _cost = (s['entry_price'] or 0.0) * s['position'] * _csize
                _base = (_cost / _lev) if _lev > 0 else 0.0
                s['pnl_pct'] = round(s['unrealized_pnl'] / _base * 100, 2) if _base > 0 else 0.0

    # ---------- 横截面信号 ----------
    def _compute_target(self):
        """拉取候选池日线 → 计算 R14 → 流动性/波动率过滤 → Top/Bottom 名单。
        返回 (longs, shorts, n_eligible, dispersion)。"""
        rows = []
        for s in self.status['symbols']:
            sym = s['symbol']
            candles, err = self.trader.get_ohlcv(sym, '1d', limit=60)
            if err or not candles:
                continue
            df = pd.DataFrame(candles)
            if len(df) < 16:
                continue
            close = df['close'].astype(float)
            r14 = float(close.iloc[-1] / close.iloc[-15] - 1.0) if close.iloc[-15] else np.nan
            liq = float(df['volume'].astype(float).iloc[-1] * close.iloc[-1])
            vol20 = float(close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(365)) if len(close) >= 21 else np.nan
            s['r14'] = r14 if np.isfinite(r14) else 0.0
            s['liq'] = liq if np.isfinite(liq) else 0.0
            s['vol20'] = vol20 if np.isfinite(vol20) else 0.0
            if np.isfinite(r14) and np.isfinite(liq):
                rows.append((s['symbol'].split('/')[0], r14, liq, vol20 if np.isfinite(vol20) else np.nan))
        if not rows:
            return [], [], 0, 0.0
        # 流动性过滤
        rows = [r for r in rows if r[2] > self.status['liq_min']]
        if not rows:
            return [], [], 0, 0.0
        # 波动率过滤
        if self.status['vol_max'] is not None:
            rows = [r for r in rows if not np.isfinite(r[3]) or r[3] <= self.status['vol_max']]
        if not rows:
            return [], [], 0, 0.0
        disp = float(np.std([r[1] for r in rows]))
        rows.sort(key=lambda x: x[1], reverse=True)
        n = int(len(rows) * self.status['top_frac'])
        if n < 1:
            n = 1
        longs = [r[0] for r in rows[:n]]
        shorts = [r[0] for r in rows[-n:]] if n > 0 else []
        if self.status['abs_mom']:
            longs = [b for b in longs if next((r[1] for r in rows if r[0] == b), 0.0) > 0]
            shorts = [b for b in shorts if next((r[1] for r in rows if r[0] == b), 0.0) < 0]
        if self.status['long_only']:
            shorts = []
        return longs, shorts, len(rows), disp

    # ---------- 下单 ----------
    def _close_symbol(self, s, reason):
        pos = s.get('position', 0) or 0
        if pos <= 0:
            return
        contracts = self.trader.round_amount(s['symbol'], pos)
        if contracts <= 0:
            return
        side_cmd = 'sell' if s['side'] == 'long' else 'buy'
        order, err = self.trader.place_order(s['symbol'], side_cmd, 'market', contracts, reduce_only=True)
        if err:
            self._log(f"平仓失败({s['symbol']} {s['side']}): {err}")
            s['last_error'] = f"平仓失败: {err}"
            return
        pnl = s.get('unrealized_pnl', 0.0) or 0.0
        self.status['realized_pnl'] = round(self.status.get('realized_pnl', 0.0) + pnl, 8)
        self.status['sell_count'] += 1
        self._log(f"平仓 {s['symbol']} {s['side']} {contracts}张 盈亏{pnl:+.2f}U ({reason})")
        s['position'] = 0
        s['side'] = 'none'
        s['entry_price'] = 0.0
        s['unrealized_pnl'] = 0.0
        s['pnl_pct'] = 0.0

    def _open_symbol(self, s, side, per):
        price = s.get('last_price') or 0.0
        if price <= 0:
            self._log(f"开仓跳过({s['symbol']}): 无有效价格")
            return
        contracts = self.trader.round_amount(s['symbol'], per / price)
        if contracts <= 0:
            self._log(f"开仓跳过({s['symbol']}): 张数为0（分配 {per:.2f}U @ {price}）")
            return
        order, err = self.trader.place_order(s['symbol'], 'buy' if side == 'long' else 'sell', 'market', contracts)
        if err:
            self._log(f"开{('多' if side == 'long' else '空')}失败({s['symbol']}): {err}")
            s['last_error'] = f"开仓失败: {err}"
            return
        self.status['buy_count'] += 1
        s['side'] = side
        s['position'] = contracts
        s['entry_price'] = price
        s['unrealized_pnl'] = 0.0
        s['pnl_pct'] = 0.0
        self._log(f"开{('多' if side == 'long' else '空')} {s['symbol']} {contracts}张 @ {price:.6f} (名义≈{per:.2f}U)")

    def _rebalance(self, longs, shorts):
        target = {}
        for b in longs:
            target[b] = 'long'
        for b in shorts:
            target[b] = 'short'
        # 1) 平掉方向改变/移出名单的持仓
        for s in self.status['symbols']:
            base = s['symbol'].split('/')[0]
            if s['position'] > 0 and target.get(base) != s['side']:
                self._close_symbol(s, '调仓移出')
        # 2) 刷新价格与持仓（平仓后重新对齐）
        self._refresh_positions_and_balance()
        self._refresh_prices()
        # 3) 计算每仓名义（dollar-neutral：权益 × 杠杆 × 安全系数 / 总仓位数）
        n_total = len(longs) + len(shorts)
        equity = self._task_equity()
        per = (equity * self.leverage * self.status['buy_pct'] / n_total) if n_total > 0 and equity > 0 else 0.0
        # 4) 开新仓
        if per > 0:
            for s in self.status['symbols']:
                base = s['symbol'].split('/')[0]
                if s['position'] <= 0:
                    if base in longs:
                        self._open_symbol(s, 'long', per)
                    elif base in shorts:
                        self._open_symbol(s, 'short', per)
        # 5) 重新同步真实持仓与未实现盈亏
        self._refresh_positions_and_balance()
        self.save_state()

    # ---------- 主循环 ----------
    def _run_loop(self):
        self._running = True
        self.status['running'] = True
        self.status['started_at'] = datetime.now().isoformat()
        self._log(f"横截面动量任务已启动: {self.status['name']} · 总资金{self.status['total_fund']}U · "
                  f"杠杆{self.status['leverage']}x · 调仓周期{self.status['rebal_days']}日 · "
                  f"Top/Bottom {self.status['top_frac']*100:.0f}% · 流动性>{self.status['liq_min']/1e6:.0f}M · "
                  f"{'只做多' if self.status['long_only'] else '多空'} · 轮询{self.status['interval']}s")
        self._log(f"候选池({len(self.status['symbols'])}个): {', '.join(BASES)}")
        self._log(f"网络: {'测试网(模拟)' if self.testnet else '主网(真实资金)'}")

        # 逐币设置杠杆与逐仓保证金模式（与回测逐仓口径一致）
        for s in self.status['symbols']:
            try:
                ok, err = self.trader.set_leverage(self.leverage, s['symbol'])
                if not ok:
                    self._log(f"设置杠杆失败({s['symbol']}): {err}")
            except Exception as e:
                self._log(f"设置杠杆失败({s['symbol']}): {e}")

        try:
            self._record_equity(force=True)
        except Exception:
            pass

        while not self._stop_event.is_set():
            try:
                self.trader.wait_ip_ban()
                # 刷新真实持仓 + 价格
                self._refresh_positions_and_balance()
                self._refresh_prices()
                # 调仓判断：距上次调仓满 rebal_days 自然日
                now = time.time()
                if not self._last_rebalance_ts or (now - self._last_rebalance_ts) >= self.status['rebal_days'] * 86400:
                    self._log("到达调仓日，计算横截面动量排名...")
                    longs, shorts, n_elig, disp = self._compute_target()
                    self.status['longs'] = longs
                    self.status['shorts'] = shorts
                    self.status['n_eligible'] = n_elig
                    self.status['dispersion'] = round(disp, 4)
                    self._log(f"信号: 做多{len(longs)} {longs} / 做空{len(shorts)} {shorts} / 候选{n_elig} / 离散度{disp:.4f}")
                    self.status['signal'] = f"多头{len(longs)} · 空头{len(shorts)}"
                    self._rebalance(longs, shorts)
                    self._last_rebalance_ts = now
                    self.status['last_rebalance_time'] = datetime.now().isoformat()
                    self.status['next_rebalance_time'] = (
                        datetime.now() + timedelta(days=self.status['rebal_days'])).isoformat()
                # 记录权益快照
                try:
                    self._record_equity()
                except Exception:
                    pass
                self.status['last_loop_time'] = datetime.now().isoformat()
                self.status['monitor_loop'] += 1
            except Exception as e:
                self.status['last_error'] = str(e)
                self.status['consecutive_errors'] = self.status.get('consecutive_errors', 0) + 1
                self._log(f"运行异常({self.status['consecutive_errors']}次): {e}")
            self._stop_event.wait(self.status['interval'] or 30)

        try:
            self._record_equity(force=True)
        except Exception:
            pass

    # ---------- 启停 ----------
    def start(self, name, total_fund, rebal_days=5, top_frac=0.20, liq_min=100_000_000.0,
              long_only=False, abs_mom=False, vol_max=None, interval=30, buy_pct=DEFAULT_BUY_PCT,
              task_id=None):
        with self._lock:
            if self._running:
                return False, '已在运行中'
            self.reset_status()
            self.status['name'] = (name or '横截面动量量化任务').strip()
            self.status['total_fund'] = float(total_fund or 0)
            self.status['rebal_days'] = max(1, int(rebal_days or 5))
            self.status['top_frac'] = float(top_frac or 0.20)
            self.status['liq_min'] = float(liq_min or 100_000_000.0)
            self.status['long_only'] = bool(long_only)
            self.status['abs_mom'] = bool(abs_mom)
            self.status['vol_max'] = float(vol_max) if vol_max else None
            self.status['interval'] = max(5, int(interval or 30))
            self.status['buy_pct'] = float(buy_pct or DEFAULT_BUY_PCT) or DEFAULT_BUY_PCT
            self.status['symbols'] = self._build_symbols()
            self._task_id = task_id or (datetime.now().strftime('%Y%m%d%H%M%S') + f"{int(time.time() * 1000) % 1000:03d}")
            if task_id:
                self.load_state()
            self._last_rebalance_ts = 0.0
            self._stop_event.clear()
            self.log_file = os.path.join(LOG_DIR, f'{log_prefix()}{self._task_id}.log')
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
        self._log("横截面动量任务已停止")
        self._update_task_status('stopped')
        self.save_state()
        return True, '已停止'

    @staticmethod
    def list_tasks():
        try:
            if os.path.exists(TASKS_FILE):
                with open(TASKS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    @staticmethod
    def delete_task(task_id):
        tasks = MomentumTrader.list_tasks()
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
            init_fund = float(self.status.get('total_fund') or 0.0)
            cur_fund = self._task_equity()
            s['initial_fund'] = round(init_fund, 2)
            s['current_fund'] = round(cur_fund, 2)
            s['total_return_pct'] = round((cur_fund - init_fund) / init_fund * 100, 2) if init_fund > 0 else 0.0
            anchor = init_fund
            try:
                pts = load_equity_points(self._task_id)
                if pts:
                    midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
                    before = [v for t, v in pts if t < midnight]
                    anchor = before[-1] if before else pts[0][1]
            except Exception:
                pass
            s['today_pnl'] = round(cur_fund - anchor, 2)
            s['today_pnl_pct'] = round((cur_fund - anchor) / anchor * 100, 2) if anchor > 0 else 0.0
            return s
