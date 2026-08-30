# -*- coding: utf-8 -*-
"""策略记录存储：最优策略 + 历史测试记录（SQLite 单表，is_top 区分）

表结构 strategy_records:
  id, is_top(1=最优策略 0=历史记录), sort_order(最优区内排序),
  symbol/timeframe/strategy/mode/tpsl/ret/daily/monthly/trades/winrate/mdd/stability/market/note,
  source(来源研究轮次), created_at
"""
import os
import re
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_FILE = os.path.join(DATA_DIR, 'strategy_records.db')  # 独立文件：策略记录可随仓库分发，用户库 users.db 不上传

FIELDS = ['symbol', 'timeframe', 'strategy', 'mode', 'tpsl', 'ret', 'daily', 'monthly',
          'trades', 'winrate', 'mdd', 'stability', 'market', 'note', 'source', 'sharpe', 'period']

# 可排序字段（数值语义解析：文本字段抽取数字后比较）
SORTABLE = {'timeframe', 'ret', 'daily', 'monthly', 'trades', 'winrate', 'mdd', 'sharpe'}
_TF_MIN = {'15m': 15, '1h': 60, '4h': 240, '1d': 1440}

# 美股代币（bStocks/美股永续）：按 symbol 基础币名归类，其余为虚拟货币
US_TOKENS = {'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'MU', 'MUU',
             'QQQ', 'TQQQ', 'MUB', 'MUUB', 'MUBD', 'SNDKB', 'SKHYB', 'NVDAB',
             'UNITREE', 'CXMT', 'TREE'}


def is_us_symbol(symbol):
    """symbol 如 'AAPL/USDT:USDT'、'AAPL~TSLA等7币'、'ETH/USDT' → 是否美股代币"""
    base = (symbol or '').split('/')[0].split('~')[0].upper()
    return base in US_TOKENS


def _num(v, field=''):
    """把文本字段解析为数值用于排序：'+201.8%(2026年1~8月)'→201.8；无数字排最后

    用 match 从头匹配：收益类文本均以数值开头，避免'亏损/爆仓(2023-2026四年)'
    这类文本被尾缀年份干扰。
    """
    if v is None or v == '':
        return float('-inf')
    if field == 'timeframe':
        return _TF_MIN.get(str(v), float('-inf'))
    m = re.match(r'\s*[+-]?\d+\.?\d*', str(v))
    return float(m.group()) if m else float('-inf')


def _conn():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_tables():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS strategy_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            is_top INTEGER NOT NULL DEFAULT 0,
            sort_order INTEGER NOT NULL DEFAULT 0,
            symbol TEXT, timeframe TEXT, strategy TEXT, mode TEXT, tpsl TEXT,
            ret TEXT, daily TEXT, monthly TEXT, trades TEXT, winrate TEXT, mdd TEXT,
            stability TEXT, market TEXT, note TEXT, source TEXT, sharpe TEXT, period TEXT,
            trades_detail TEXT,
            created_at TEXT NOT NULL
        )""")
        cols = [r[1] for r in c.execute("PRAGMA table_info(strategy_records)")]
        if 'sharpe' not in cols:  # 老库平滑加列
            c.execute("ALTER TABLE strategy_records ADD COLUMN sharpe TEXT")
        if 'period' not in cols:
            c.execute("ALTER TABLE strategy_records ADD COLUMN period TEXT")
        if 'trades_detail' not in cols:
            c.execute("ALTER TABLE strategy_records ADD COLUMN trades_detail TEXT")


def count_all():
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM strategy_records").fetchone()[0]


def seed_initial(rows, top_rows):
    """首次启动导入历史数据（仅当表为空）：top_rows=初始最优，rows=其余历史记录"""
    if count_all() > 0:
        return False
    now = datetime.now().isoformat(timespec='seconds')
    with _conn() as c:
        for i, r in enumerate(top_rows):
            c.execute(
                "INSERT INTO strategy_records (is_top, sort_order, created_at, " + ",".join(FIELDS) + ") "
                "VALUES (1, ?, ?, " + ",".join("?" * len(FIELDS)) + ")",
                (i, now, *[r.get(f) for f in FIELDS]))
        for r in rows:
            c.execute(
                "INSERT INTO strategy_records (is_top, sort_order, created_at, " + ",".join(FIELDS) + ") "
                "VALUES (0, 0, ?, " + ",".join("?" * len(FIELDS)) + ")",
                (now, *[r.get(f) for f in FIELDS]))
    return True


def list_top():
    """最优策略（按 sort_order 升序）"""
    with _conn() as c:
        rows = c.execute("SELECT * FROM strategy_records WHERE is_top=1 ORDER BY sort_order ASC, id ASC").fetchall()
    return [dict(r) for r in rows]


def list_history(page=1, per_page=15, q='', sort=None, order='desc', cat=''):
    """历史测试记录（分页 + 搜索 + 数值排序 + 类别筛选），返回 (记录, 总数, 总页数)

    q 在八个字段模糊匹配（含日期）；sort 为 SORTABLE 内字段时按数值语义排序（Python 侧解析）。
    cat: ''/'all'=全部, 'us'=美股代币, 'crypto'=虚拟货币。
    """
    where = "is_top=0"
    params = []
    if q:
        like = f"%{q}%"
        where += (" AND (symbol LIKE ? OR timeframe LIKE ? OR strategy LIKE ? OR mode LIKE ? "
                  "OR tpsl LIKE ? OR source LIKE ? OR note LIKE ? OR period LIKE ?)")
        params = [like] * 8
    if cat in ('us', 'crypto'):
        # SQLite 无原生按集合过滤，取回 Python 侧按 is_us_symbol 过滤（总量小，无性能问题）
        pass
    with _conn() as c:
        if cat in ('us', 'crypto'):
            all_rows = c.execute(f"SELECT * FROM strategy_records WHERE {where} ORDER BY id DESC",
                                 params).fetchall()
            all_rows = [dict(r) for r in all_rows if is_us_symbol(r['symbol']) == (cat == 'us')]
            total = len(all_rows)
            pages = max(1, (total + per_page - 1) // per_page)
            page = max(1, min(page, pages))
            if sort in SORTABLE:
                all_rows.sort(key=lambda r: _num(r.get(sort), sort), reverse=(order != 'asc'))
            return all_rows[(page - 1) * per_page: page * per_page], total, pages
        total = c.execute(f"SELECT COUNT(*) FROM strategy_records WHERE {where}", params).fetchone()[0]
        pages = max(1, (total + per_page - 1) // per_page)
        page = max(1, min(page, pages))
        if sort in SORTABLE:
            rows = c.execute(f"SELECT * FROM strategy_records WHERE {where} ORDER BY id DESC", params).fetchall()
            recs = [dict(r) for r in rows]
            recs.sort(key=lambda r: _num(r.get(sort), sort), reverse=(order != 'asc'))
            rows = recs[(page - 1) * per_page: page * per_page]
        else:
            rows = c.execute(f"SELECT * FROM strategy_records WHERE {where} "
                             "ORDER BY id DESC LIMIT ? OFFSET ?",
                             params + [per_page, (page - 1) * per_page]).fetchall()
            rows = [dict(r) for r in rows]
    return rows, total, pages


def promote(history_id):
    """历史记录 → 最优策略（排到末位）"""
    with _conn() as c:
        row = c.execute("SELECT id FROM strategy_records WHERE id=? AND is_top=0", (history_id,)).fetchone()
        if row is None:
            return False, '记录不存在'
        max_sort = c.execute("SELECT COALESCE(MAX(sort_order), -1) FROM strategy_records WHERE is_top=1").fetchone()[0]
        c.execute("UPDATE strategy_records SET is_top=1, sort_order=? WHERE id=?", (max_sort + 1, history_id))
    return True, '已加入最优策略'


def demote(top_id):
    """最优策略 → 历史记录"""
    with _conn() as c:
        row = c.execute("SELECT id FROM strategy_records WHERE id=? AND is_top=1", (top_id,)).fetchone()
        if row is None:
            return False, '记录不存在'
        c.execute("UPDATE strategy_records SET is_top=0, sort_order=0 WHERE id=?", (top_id,))
    return True, '已移回历史记录'


def move(top_id, direction):
    """最优策略排序：direction='up'上移 / 'down'下移（与相邻记录交换 sort_order）"""
    tops = list_top()
    idx = next((i for i, t in enumerate(tops) if t['id'] == top_id), None)
    if idx is None:
        return False, '记录不存在'
    if direction == 'up' and idx == 0:
        return False, '已在顶部'
    if direction == 'down' and idx == len(tops) - 1:
        return False, '已在底部'
    other = tops[idx - 1] if direction == 'up' else tops[idx + 1]
    with _conn() as c:
        # 交换两者的 sort_order（用 id 兜底保证唯一）
        c.execute("UPDATE strategy_records SET sort_order=? WHERE id=?", (-1, tops[idx]['id']))
        c.execute("UPDATE strategy_records SET sort_order=? WHERE id=?", (tops[idx]['sort_order'], other['id']))
        c.execute("UPDATE strategy_records SET sort_order=? WHERE id=?", (other['sort_order'], tops[idx]['id']))
    return True, '排序已更新'


def delete(record_id):
    """删除记录（最优或历史均可）"""
    with _conn() as c:
        cur = c.execute("DELETE FROM strategy_records WHERE id=?", (record_id,))
        if cur.rowcount == 0:
            return False, '记录不存在'
    return True, '已删除'


def add_history(rec, dedupe=True):
    """通用入口：新增一条历史测试记录（回测结果/研究数据统一走此函数入库）

    rec 需含 FIELDS 中的字段（symbol/strategy/source 至少其一非空）。
    dedupe=True 时按 symbol+strategy+mode+tpsl+source 查重，已存在则跳过。
    返回 (是否写入, 提示)。
    """
    with _conn() as c:
        if dedupe:
            dup = c.execute("SELECT 1 FROM strategy_records WHERE symbol IS ? AND strategy IS ? "
                            "AND mode IS ? AND tpsl IS ? AND source IS ?",
                            (rec.get('symbol'), rec.get('strategy'), rec.get('mode'),
                             rec.get('tpsl'), rec.get('source'))).fetchone()
            if dup:
                return False, '记录已存在（查重跳过）'
        c.execute("INSERT INTO strategy_records (is_top, sort_order, created_at, " + ",".join(FIELDS) + ") "
                  "VALUES (0, 0, ?, " + ",".join("?" * len(FIELDS)) + ")",
                  (datetime.now().isoformat(timespec='seconds'), *[rec.get(f) for f in FIELDS]))
    return True, '已入库'
