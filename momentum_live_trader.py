# -*- coding: utf-8 -*-
"""横截面动量实盘量化引擎（币安 USDT 永续合约，测试网 / 主网可切换）。

与「横截面动量回测」共用同一策略口径（R14 排名，Top/Bottom 对称，dollar-neutral）：
  - R14 = close / close.shift(14) - 1
  - 流动性过滤：24h 成交额 = volume × close > liq_min（默认 100M USDT）
  - 排名：按 R14 降序，Top n 做多、Bottom n 做空（n = 候选数 × top_frac）
  - 市场趋势闸门（V2 冻结口径）：Top30 等权指数 > MA60 才交易，否则全部空仓
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

# 候选币池（universe_mode='fixed' 时的名单）：市值前30回测标的 ∩ 币安 USDT 永续
# 注意：此为旧 V2 口径（存在幸存者偏差，R5-R8 已审计）。生产推荐 universe_mode='u8'。
BASES = ['BTC', 'ETH', 'BNB', 'XRP', 'SOL', 'TRX', 'HYPE', 'ZEC', 'DOGE', 'XMR',
         'LINK', 'ADA', 'XLM', 'BCH', 'CC', 'LTC', 'UNI', 'GRAM', 'HBAR', 'AVAX',
         'SUI', 'NEAR', 'M', 'TAO', 'ASTER', 'AAVE', 'ONDO', 'MORPHO', 'DOT', 'ICP']

# U8 动态宇宙（V2@U8 生产口径，R7-R14 审计冻结）：
#   ADV30 ≥ adv_min 且 上市年龄 ≥ age_min 天 且 当日成交额 > liq_floor，
#   按 ADV30 降序取前 top_n。每个调仓日重算（PIT，无幸存者偏差）。
U8_PARAMS = {
    'adv_min': 100_000_000.0,   # 30日均成交额下限（USDT）
    'age_min': 365,             # 上市天数下限
    'liq_floor': 10_000_000.0,  # 当日成交额下限（USDT）
    'top_n': 50,                # ADV30 排名截断
    'adv_win': 30,              # ADV 均值窗口（天）
}
ONBOARD_CACHE_FILE = os.path.join('data', 'momentum_onboard.json')

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
            'ma_gate': 60,               # 市场趋势闸门（V2：等权指数>MA_N 才交易，0=关闭）
            'universe_mode': 'fixed',    # 'fixed'=BASES 名单 | 'u8'=PIT 动态宇宙（生产推荐）
            'interval': 30,              # 轮询间隔（秒）
            'buy_pct': DEFAULT_BUY_PCT,
            'symbols': [],               # 候选币状态
            'longs': [],                 # 当前做多名单
            'shorts': [],                # 当前做空名单
            'n_eligible': 0,             # 通过流动性过滤的候选数
            'dispersion': 0.0,           # 横截面离散度 std(R14)
            'gate_on': None,             # 趋势闸门状态（True=开仓允许/False=空仓/None=未判断）
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
            'ma_gate': self.status['ma_gate'],
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
                'ma_gate': self.status['ma_gate'],
                'universe_mode': self.status['universe_mode'],
                'interval': self.status['interval'],
                'buy_pct': self.status['buy_pct'],
                'longs': self.status['longs'],
                'shorts': self.status['shorts'],
                'realized_pnl': self.status['realized_pnl'],
                'buy_count': self.status['buy_count'],
                'sell_count': self.status['sell_count'],
                'last_rebalance_time': self.status['last_rebalance_time'],
                'gate_hist': self.status.get('gate_hist') or [],   # 阶梯指数历史（重启后闸门立即可用）
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
                      'liq_min', 'ma_gate', 'universe_mode', 'interval', 'buy_pct'):
                if k in data:
                    self.status[k] = data[k]
            self.status['longs'] = data.get('longs') or []
            self.status['shorts'] = data.get('shorts') or []
            self.status['realized_pnl'] = float(data.get('realized_pnl') or 0.0)
            self.status['buy_count'] = int(data.get('buy_count') or 0)
            self.status['sell_count'] = int(data.get('sell_count') or 0)
            self.status['last_rebalance_time'] = data.get('last_rebalance_time')
            if isinstance(data.get('gate_hist'), list):
                self.status['gate_hist'] = [
                    [str(d), float(m)] for d, m in data['gate_hist'] if m == m]
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
    def _build_symbols(self, bases=None):
        bases = bases if bases is not None else BASES
        return [{
            'symbol': f'{b}/USDT',
            'name': NAMES.get(b, b),
            'r14': 0.0,
            'liq': 0.0,
            'side': 'none',
            'position': 0,
            'entry_price': 0.0,
            'last_price': 0.0,
            'unrealized_pnl': 0.0,
            'pnl_pct': 0.0,
        } for b in bases]

    # ---------- U8 动态宇宙（V2@U8 生产口径） ----------
    def _load_onboard_cache(self):
        try:
            if os.path.exists(ONBOARD_CACHE_FILE):
                with open(ONBOARD_CACHE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_onboard_cache(self, cache):
        try:
            os.makedirs(os.path.dirname(ONBOARD_CACHE_FILE), exist_ok=True)
            with open(ONBOARD_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache, f)
        except Exception:
            pass

    def _u8_pool(self, now_ms=None):
        """U8 PIT 动态宇宙：全市场 USDT 永续 → ADV30/年龄/当日流动性过滤 → ADV30 Top N。
        仅在调仓日调用（全市场扫描，约几百次 klines 请求）。"""
        p = U8_PARAMS
        now_ms = now_ms or int(time.time() * 1000)
        onboard = self._load_onboard_cache()
        markets = {}
        try:
            markets = self.trader.exchange.markets or {}
        except Exception:
            markets = {}
        if not markets:
            try:
                markets = self.trader.exchange.load_markets() or {}
            except Exception:
                markets = {}
        # 候选：活跃 USDT 永续；onboardDate 补缓存
        cands = []
        for sym, m in markets.items():
            if not (m.get('swap') and m.get('quote') == 'USDT' and m.get('active', True)):
                continue
            base = m.get('base')
            if not base:
                continue
            obd = (m.get('info') or {}).get('onboardDate')
            if obd and base not in onboard:
                onboard[base] = int(obd)
            cands.append(base)
        if onboard:
            self._save_onboard_cache(onboard)
        rows = []
        n_scan = 0
        for base in sorted(set(cands)):
            sym = f'{base}/USDT'
            try:
                candles, err = self.trader.get_ohlcv(sym, '1d', limit=p['adv_win'] + 5)
            except Exception:
                continue
            if err or not candles:
                continue
            n_scan += 1
            df = pd.DataFrame(candles)
            close = df['close'].astype(float)
            qv = df['volume'].astype(float) * close
            if not len(qv):
                continue
            # R7 冻结语义：ADV30 用可用 bars 计算（新上市不足 30 根也参与排名，
            # 会占 Top-N 名额但被 age 过滤挡掉——与回测 pool_at 完全同构）
            adv = float(qv.iloc[-p['adv_win']:].mean())
            liq = float(qv.iloc[-1])
            ob_ms = onboard.get(base)
            if not ob_ms:
                continue  # 无上市日期（保守排除）
            age_days = (now_ms - int(ob_ms)) / 86400_000.0
            if liq > p['liq_floor']:
                rows.append((base, adv, age_days))
        # R7 冻结语义：先按 ADV30 降序取 Top N，再过滤 adv_min/age_min（不回填）
        rows.sort(key=lambda x: -x[1])
        pool = [b for b, a, g in rows[:p['top_n']] if a >= p['adv_min'] and g >= p['age_min']]
        self._log(f"U8 动态宇宙: 扫描{n_scan}个USDT永续 → 候选{len(rows)} → "
                  f"Top{p['top_n']}过adv/age → 池{len(pool)}: "
                  f"{', '.join(pool[:15])}{'...' if len(pool) > 15 else ''}")
        return pool

    def _refresh_universe(self, now_ms=None):
        """u8 模式：重算宇宙并重建 symbols（保留现有持仓信息）。
        跌出池但仍有持仓的币必须保留在 symbols 中，否则 _rebalance 的
        平仓循环遍历不到 → 幽灵持仓（R15 Replay 抓获的 bug）。"""
        old = {s['symbol']: s for s in self.status['symbols']}
        pool = self._u8_pool(now_ms=now_ms)
        keep = set(pool)
        bases = list(pool)
        for sym, s in old.items():
            base = sym.split('/')[0]
            if (s.get('position') or 0) > 0 and base not in keep:
                bases.append(base)
                self._log(f"{base} 跌出 U8 池但仍有持仓，保留待平仓")
        new_syms = self._build_symbols(bases)
        for s in new_syms:
            src = old.get(s['symbol'])
            if src:
                for k in ('side', 'position', 'entry_price', 'last_price',
                          'unrealized_pnl', 'pnl_pct'):
                    s[k] = src[k]
        self.status['symbols'] = new_syms
        return pool

    # ---------- 权益 ----------
    def _task_equity(self):
        """权益 = 账户可用余额 + 保证金占用 + 未实现盈亏。
        优先用交易所真实账户（现金口径，天然含手续费与 funding 扣减），
        与回测 cash + 持仓市值 的会计同构；不可用时退回本地记账。"""
        bal = float(self.status.get('account_balance') or 0.0)
        margin = 0.0
        unreal = 0.0
        for s in self.status['symbols']:
            p = float(s.get('position') or 0.0)
            if p > 0:
                unreal += float(s.get('unrealized_pnl') or 0.0)
                entry = float(s.get('entry_price') or 0.0)
                margin += entry * p / max(1.0, float(self.leverage or 1))
        if bal > 0:
            # 保证金占用已在开仓时从可用余额中划出（逐仓），权益 = 可用 + 占用 + 浮盈亏
            return round(bal + margin + unreal, 4)
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
    def _compute_target(self, now_ts=None):
        """拉取候选池日线 → 计算 R14 → 流动性过滤 → Top/Bottom 名单 → V2 趋势闸门。
        universe_mode='u8'：先全市场重算 U8 动态宇宙（PIT，年龄按 now_ts 计算），
        池内流动性阈值用 U8 口径；跌出池的幽灵持仓币不参与排名（仅保留待平仓）。
        趋势闸门（V2 冻结口径）：调仓网格点（每 rebal_days 一点）的池等权日收益
        累乘成阶梯指数，指数 > MA_N 才交易。与 R7 回测 build_gate 完全同构。
        闸门不通过（或 MA 未就绪）则全部空仓。
        返回 (longs, shorts, n_eligible, dispersion, gate_on)。"""
        u8_mode = self.status.get('universe_mode') == 'u8'
        pool = None
        if u8_mode:
            pool = set(self._refresh_universe(now_ms=(now_ts * 1000) if now_ts else None))
        liq_min = U8_PARAMS['liq_floor'] if u8_mode else self.status['liq_min']
        ma_n = int(self.status.get('ma_gate') or 0)
        limit = max(60, ma_n + 15)
        rows = []
        rets = {}   # base -> 当日收益（池等权指数成分）
        last_ts = 0
        for s in self.status['symbols']:
            base = s['symbol'].split('/')[0]
            if pool is not None and base not in pool:
                continue   # 幽灵持仓：仅保留待平仓，不参与排名/闸门
            sym = s['symbol']
            candles, err = self.trader.get_ohlcv(sym, '1d', limit=limit)
            if err or not candles:
                continue
            df = pd.DataFrame(candles)
            if len(df) < 16:
                continue
            close = df['close'].astype(float)
            r14 = float(close.iloc[-1] / close.iloc[-15] - 1.0) if close.iloc[-15] else np.nan
            liq = float(df['volume'].astype(float).iloc[-1] * close.iloc[-1])
            s['r14'] = r14 if np.isfinite(r14) else 0.0
            s['liq'] = liq if np.isfinite(liq) else 0.0
            if len(close) >= 2 and close.iloc[-2]:
                r1 = float(close.iloc[-1] / close.iloc[-2] - 1.0)
                if np.isfinite(r1):
                    rets[base] = r1
            last_ts = max(last_ts, int(df['ts'].iloc[-1]))
            if np.isfinite(r14) and np.isfinite(liq):
                rows.append((base, r14, liq))
        if not rows:
            return [], [], 0, 0.0, None
        # 流动性过滤
        rows = [r for r in rows if r[2] > liq_min]
        if not rows:
            return [], [], 0, 0.0, None
        disp = float(np.std([r[1] for r in rows]))
        rows.sort(key=lambda x: x[1], reverse=True)
        n = int(len(rows) * self.status['top_frac'])
        if n < 1:
            n = 1
        longs = [r[0] for r in rows[:n]]
        shorts = [r[0] for r in rows[-n:]] if n > 0 else []
        # V2 趋势闸门：阶梯指数（每调仓日一点）> MA_N 才交易（R7 build_gate 同构）
        gate_on = True
        if ma_n > 0:
            hist = self.status.setdefault('gate_hist', [])
            if rets:
                d_str = datetime.utcfromtimestamp(last_ts / 1000.0).strftime('%Y-%m-%d')
                mkt = float(np.mean(list(rets.values())))
                if hist and hist[-1][0] == d_str:
                    hist[-1][1] = mkt
                else:
                    hist.append([d_str, mkt])
                self.status['gate_hist'] = hist[-(ma_n + 60):]   # 有界增长
            idx_series = np.cumprod([1.0] + [1.0 + m for _, m in self.status['gate_hist']])[1:]
            if len(idx_series) >= ma_n:
                ma = float(np.mean(idx_series[-ma_n:]))
                gate_on = bool(idx_series[-1] > ma)
                self.status['gate_index'] = round(float(idx_series[-1]), 6)
                self.status['gate_ma'] = round(ma, 6)
            else:
                # MA 未就绪（需 ma_n 个调仓点）：与回测一致，空仓等待
                return [], [], len(rows), disp, False
            if not gate_on:
                return [], [], len(rows), disp, False
        return longs, shorts, len(rows), disp, gate_on

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
        # U8 池换入的新币可能未设置杠杆（start 时按旧名单设置），开仓前确保
        if not s.get('lev_set'):
            try:
                ok, err = self.trader.set_leverage(self.leverage, s['symbol'])
                if ok:
                    s['lev_set'] = True
                else:
                    self._log(f"设置杠杆失败({s['symbol']}): {err}")
            except Exception as e:
                self._log(f"设置杠杆失败({s['symbol']}): {e}")
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

    # ---------- 单次调仓判断（主循环与历史 Replay 共用） ----------
    def tick(self, now_ts=None, force_rebalance=False):
        """一轮监测：刷新持仓/价格；到达调仓日则重算信号并执行。
        now_ts 可注入（历史 Replay 用），默认真实时间。返回是否执行了调仓。"""
        now_ts = now_ts if now_ts is not None else time.time()
        self.trader.wait_ip_ban()
        self._refresh_positions_and_balance()
        self._refresh_prices()
        due = (not self._last_rebalance_ts) or \
              (now_ts - self._last_rebalance_ts) >= self.status['rebal_days'] * 86400
        if force_rebalance or due:
            self._log("到达调仓日，计算横截面动量排名...")
            longs, shorts, n_elig, disp, gate_on = self._compute_target(now_ts=now_ts)
            self.status['longs'] = longs
            self.status['shorts'] = shorts
            self.status['n_eligible'] = n_elig
            self.status['dispersion'] = round(disp, 4)
            self.status['gate_on'] = gate_on
            if gate_on is False:
                gi = self.status.get('gate_index')
                gm = self.status.get('gate_ma')
                self._log(f"趋势闸门关闭: 等权指数{gi} ≤ MA{self.status['ma_gate']}({gm})，全部空仓")
                self.status['signal'] = f"闸门空仓（指数≤MA{self.status['ma_gate']}）"
            else:
                self._log(f"信号: 做多{len(longs)} {longs} / 做空{len(shorts)} {shorts} / 候选{n_elig} / 离散度{disp:.4f}")
                self.status['signal'] = f"多头{len(longs)} · 空头{len(shorts)}"
            self._rebalance(longs, shorts)
            self._last_rebalance_ts = now_ts
            self.status['last_rebalance_time'] = datetime.utcfromtimestamp(now_ts).isoformat()
            self.status['next_rebalance_time'] = (
                datetime.utcfromtimestamp(now_ts + self.status['rebal_days'] * 86400)).isoformat()
            return True
        return False

    # ---------- 主循环 ----------
    def _run_loop(self):
        self._running = True
        self.status['running'] = True
        self.status['started_at'] = datetime.now().isoformat()
        self._log(f"横截面动量V2任务已启动: {self.status['name']} · 总资金{self.status['total_fund']}U · "
                  f"杠杆{self.status['leverage']}x · 调仓周期{self.status['rebal_days']}日 · "
                  f"Top/Bottom {self.status['top_frac']*100:.0f}% · 宇宙模式{self.status['universe_mode']}"
                  f"{'(U8 PIT动态)' if self.status['universe_mode'] == 'u8' else ''} · "
                  f"趋势闸门{'指数>MA' + str(self.status['ma_gate']) if self.status['ma_gate'] else '关闭'} · "
                  f"轮询{self.status['interval']}s")
        if self.status['universe_mode'] == 'u8':
            self._log(f"U8 动态宇宙参数: {U8_PARAMS}")
        else:
            self._log(f"候选池({len(self.status['symbols'])}个): {', '.join(BASES)}")
        self._log(f"网络: {'测试网(模拟)' if self.testnet else '主网(真实资金)'}")

        # 逐币设置杠杆与逐仓保证金模式（与回测逐仓口径一致）
        for s in self.status['symbols']:
            try:
                ok, err = self.trader.set_leverage(self.leverage, s['symbol'])
                if not ok:
                    self._log(f"设置杠杆失败({s['symbol']}): {err}")
                else:
                    s['lev_set'] = True
            except Exception as e:
                self._log(f"设置杠杆失败({s['symbol']}): {e}")

        try:
            self._record_equity(force=True)
        except Exception:
            pass

        while not self._stop_event.is_set():
            try:
                self.tick()
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
              ma_gate=60, interval=30, buy_pct=DEFAULT_BUY_PCT,
              task_id=None, universe_mode='fixed'):
        with self._lock:
            if self._running:
                return False, '已在运行中'
            self.reset_status()
            self.status['name'] = (name or '横截面动量量化任务').strip()
            self.status['total_fund'] = float(total_fund or 0)
            self.status['rebal_days'] = max(1, int(rebal_days or 5))
            self.status['top_frac'] = float(top_frac or 0.20)
            self.status['liq_min'] = float(liq_min or 100_000_000.0)
            self.status['ma_gate'] = max(0, int(ma_gate or 0))
            self.status['universe_mode'] = 'u8' if universe_mode == 'u8' else 'fixed'
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
