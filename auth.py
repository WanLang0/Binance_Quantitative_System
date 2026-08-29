# -*- coding: utf-8 -*-
"""用户认证模块：SQLite 存储账密，密码使用加盐哈希（werkzeug 标准：scrypt，旧环境自动回退 pbkdf2:sha256）

安全设计：
- 数据库只存哈希，不存明文密码（泄库也无法还原密码）
- 每个用户独立随机盐（同一密码哈希也不同）
- 首次初始化后禁止再注册（单用户系统，无开放注册面）
- 校验失败与用户不存在消耗相同时间（防时序侧信道探测用户名）
"""
import os
import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_FILE = os.path.join(DATA_DIR, 'users.db')

# 独立 dummy 哈希：用户名不存在时也执行一次校验，拉平响应时间
_DUMMY_HASH = generate_password_hash('dummy-password-for-timing')


def _conn():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化用户表（幂等，服务启动时调用）"""
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_login_at TEXT
        )""")


def user_exists_any():
    """是否已初始化过账号（决定登录页显示'初始化'还是'登录'）"""
    with _conn() as c:
        return c.execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None


def create_user(username, password):
    """创建账号（仅允许在系统未初始化时调用），返回 (是否成功, 提示)"""
    username = (username or '').strip()
    if not (4 <= len(username) <= 20):
        return False, '用户名长度需 4~20 个字符'
    if len(password or '') < 8:
        return False, '密码至少 8 位（建议字母+数字混合）'
    if user_exists_any():
        return False, '系统已初始化账号，禁止重复创建'
    pw_hash = generate_password_hash(password)  # 格式: pbkdf2:sha256:260000$salt$hash
    try:
        with _conn() as c:
            c.execute("INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                      (username, pw_hash, datetime.now().isoformat(timespec='seconds')))
    except sqlite3.IntegrityError:
        return False, '用户名已存在'
    return True, '账号创建成功'


def verify_user(username, password):
    """校验账密，成功则更新最后登录时间"""
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE username = ?",
                        ((username or '').strip(),)).fetchone()
    if row is None:
        check_password_hash(_DUMMY_HASH, password or '')  # 拉平响应时间
        return False
    if not check_password_hash(row['password_hash'], password or ''):
        return False
    with _conn() as c:
        c.execute("UPDATE users SET last_login_at = ? WHERE id = ?",
                  (datetime.now().isoformat(timespec='seconds'), row['id']))
    return True


def change_password(username, old_password, new_password):
    """修改密码（需验证旧密码），返回 (是否成功, 提示)"""
    if not verify_user(username, old_password):
        return False, '原密码错误'
    if len(new_password or '') < 8:
        return False, '新密码至少 8 位'
    with _conn() as c:
        c.execute("UPDATE users SET password_hash = ? WHERE username = ?",
                  (generate_password_hash(new_password), (username or '').strip()))
    return True, '密码已更新'
