import os
import json
import time
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np

from flask import Flask, render_template, request, send_file, abort, session, redirect, url_for
import io

# 导入自定义模块
from data_fetcher import BinanceDataFetcher
from indicators import TechnicalIndicators
from backtest_engine import BacktestEngine
from chart_utils import ChartUtils
from demo_trader import DemoTrader
from auto_trader import AutoTrader, STATE_FILE
from futures_trader import FuturesTrader
from auto_futures import AutoFutures, STATE_FILE as FUTURES_STATE_FILE
import auth as _auth
import strategies_store as _store

app = Flask(__name__)

from changelog import APP_VERSION, CHANGELOG

@app.context_processor
def _inject_version():
    """全模板可用 APP_VERSION（顶栏显示）"""
    return {'APP_VERSION': APP_VERSION}


def _load_secret_key():
    """SECRET_KEY 随机生成一次并持久化到 data/secret_key（重启不失效、不硬编码）"""
    key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'secret_key')
    os.makedirs(os.path.dirname(key_file), exist_ok=True)
    if os.path.exists(key_file):
        with open(key_file, 'r') as f:
            key = f.read().strip()
        if key:
            return key
    key = os.urandom(32).hex()
    with open(key_file, 'w') as f:
        f.write(key)
    return key


app.secret_key = _load_secret_key()
app.permanent_session_lifetime = timedelta(days=7)  # 登录态保持 7 天
_auth.init_db()  # 启动时确保用户表存在

# ==================== 登录认证（全站拦截） ====================
_login_fail = {}          # 客户端标识 -> {'n': 失败次数, 'until': 锁定截止时间戳}
LOGIN_MAX_FAIL = 5        # 连续失败次数上限
LOGIN_LOCK_SECONDS = 600  # 锁定时长（10分钟）


def _client_key():
    return request.remote_addr or 'unknown'


@app.before_request
def _require_login():
    """全局登录拦截：除登录页/静态资源/健康检查外，未登录一律跳转登录页"""
    # 放行：登录页、静态资源、健康检查
    if request.endpoint in ('login', 'static') or request.path == '/health':
        return None
    if session.get('user'):
        # 登录态一致性校验：数据库被重建/删除时，旧 cookie 对应的用户已不存在，强制重新登录
        if not _auth.user_exists(session['user']):
            session.clear()
            return redirect(url_for('login'))
        return None
    # AJAX 接口未登录返回 401 JSON（避免前端拿到登录页 HTML 解析报错刷屏）
    if request.path.startswith('/futures/api/') or request.path.startswith('/demo/api/') \
            or request.path.startswith('/auto/api/'):
        from flask import jsonify
        return jsonify({'error': 'unauthorized'}), 401
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    is_setup = not _auth.user_exists_any()   # 首次访问显示"初始化账号"，之后显示"登录"
    error, message = None, None
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        key = _client_key()
        rec = _login_fail.get(key, {'n': 0, 'until': 0})
        if time.time() < rec['until']:
            remain = int(rec['until'] - time.time())
            error = f'失败次数过多，已锁定，请 {remain // 60 + 1} 分钟后再试'
        elif is_setup:
            ok, msg = _auth.create_user(username, password)
            if ok:
                session['user'] = username.strip()
                session.permanent = True
                return redirect(url_for('index'))
            error = msg
        else:
            if _auth.verify_user(username, password):
                _login_fail.pop(key, None)
                session['user'] = username.strip()
                session.permanent = True
                return redirect(url_for('index'))
            rec['n'] += 1
            if rec['n'] >= LOGIN_MAX_FAIL:
                rec['until'] = time.time() + LOGIN_LOCK_SECONDS
                rec['n'] = 0
                error = f'连续失败 {LOGIN_MAX_FAIL} 次，锁定 10 分钟'
            else:
                error = f'用户名或密码错误（还剩 {LOGIN_MAX_FAIL - rec["n"]} 次机会）'
            _login_fail[key] = rec
    return render_template('login.html', is_setup=is_setup, error=error, message=message)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ==================== 环境配置（.env） ====================
def _load_env_file():
    """加载项目根目录 .env（KEY=VALUE，支持 # 注释与引号）；已存在的系统环境变量优先，不覆盖"""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if not os.path.exists(env_path):
        return
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, val = line.partition('=')
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if val and not os.environ.get(key):
                os.environ[key] = val


_load_env_file()

# 代理说明：优先级 系统环境变量 > .env 文件 > 无代理直连（境外服务器直连场景）
# 本地默认在 .env 中配置 HTTP_PROXY/HTTPS_PROXY=http://127.0.0.1:7892

# 全局数据获取器（单例，按需切换现货/合约）
_data_fetcher = None
# 全局自动交易引擎（单例）
_auto_trader = None
# DemoTrader 实例缓存（按 api_key 缓存，避免每次请求重建 ccxt 触发 load_markets 拖慢页面）
_demo_traders = {}
# 市场概览缓存（60秒过期，避免每次打开首页都走代理请求阻塞）
_overview_cache = {'key': None, 'ts': 0, 'rows': []}

def get_data_fetcher():
    global _data_fetcher
    if _data_fetcher is None:
        _data_fetcher = BinanceDataFetcher()
    return _data_fetcher

data_fetcher = get_data_fetcher()

# 预热行情目录：ccxt 首次 fetch_tickers 会触发 load_markets()（拉全市场目录，走代理很慢）。
# 在服务启动后立即用后台线程预热，避免用户在首次加载首页时被阻塞十几秒。
def _warm_market_cache():
    try:
        data_fetcher.exchange.load_markets()
        data_fetcher.get_tickers_prices(['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT'])
    except Exception as e:
        print(f"市场预热失败: {e}")

import threading
threading.Thread(target=_warm_market_cache, daemon=True).start()

# 时间周期选项
TIMEFRAME_OPTIONS = {
    "1分钟": "1m",
    "5分钟": "5m",
    "15分钟": "15m",
    "1小时": "1h",
    "4小时": "4h",
    "1天": "1d",
}

# 策略指标选项
STRATEGIES = ["RSI", "KDJ", "布林带", "EMA", "MACD"]

# 固定交易对列表（symbol, 显示名称）
SYMBOL_LIST = [
    ("BTC/USDT", "BTC"),
    ("ETH/USDT", "ETH"),
    ("XRP/USDT", "XRP"),
    ("BNB/USDT", "BNB"),
    ("MUUB/USDT", "MUUB - Direxion MU Bull 2X ETF (bStocks)"),
    ("SNDKB/USDT", "SNDKB - SanDisk (bStocks)"),
    ("SKHYB/USDT", "SKHYB - SK Hynix (bStocks)"),
    ("MUB/USDT", "MUB - Micron Technology (bStocks)"),
    ("NVDAB/USDT", "NVDAB - NVIDIA (bStocks)"),
]

# 展示指标（不参与交易，仅展示）
DISPLAY_INDICATORS = ["SMA", "随机指标", "ATR"]


def _to_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _build_indicators(form, selected_strategies):
    """根据表单与所选策略列表构建指标参数字典（支持多策略叠加）"""
    indicators = {}

    if "RSI" in selected_strategies:
        indicators['rsi'] = True
        indicators['rsi_period'] = _to_int(form.get('rsi_period'), 14)
        indicators['rsi_oversold'] = _to_int(form.get('rsi_oversold'), 30)
        indicators['rsi_overbought'] = _to_int(form.get('rsi_overbought'), 70)

    if "KDJ" in selected_strategies:
        indicators['kdj'] = True
        indicators['kdj_k_period'] = _to_int(form.get('kdj_k_period'), 9)
        indicators['kdj_d_period'] = _to_int(form.get('kdj_d_period'), 3)
        indicators['kdj_j_period'] = _to_int(form.get('kdj_d_period'), 3)
        indicators['kdj_buy_threshold'] = _to_int(form.get('kdj_buy_threshold'), 20)
        indicators['kdj_sell_threshold'] = _to_int(form.get('kdj_sell_threshold'), 80)

    if "布林带" in selected_strategies:
        indicators['boll'] = True
        indicators['bb_period'] = _to_int(form.get('bb_period'), 20)
        indicators['bb_std'] = _to_float(form.get('bb_std'), 2.0)

    if "EMA" in selected_strategies:
        indicators['ema'] = True
        indicators['ema_short'] = _to_int(form.get('ema_short'), 12)
        indicators['ema_long'] = _to_int(form.get('ema_long'), 26)
        indicators['ema_periods'] = [indicators['ema_short'], indicators['ema_long']]

    if "MACD" in selected_strategies:
        indicators['macd'] = True
        indicators['macd_fast'] = _to_int(form.get('macd_fast'), 12)
        indicators['macd_slow'] = _to_int(form.get('macd_slow'), 26)
        indicators['macd_signal'] = _to_int(form.get('macd_signal'), 9)

    if "双均线交叉" in selected_strategies:
        indicators['ma_cross'] = True
        indicators['ma_cross_short'] = _to_int(form.get('ma_cross_short'), 10)
        indicators['ma_cross_long'] = _to_int(form.get('ma_cross_long'), 30)
        indicators['ma_cross_periods'] = [indicators['ma_cross_short'], indicators['ma_cross_long']]

    # 展示指标（仅图表展示，不参与交易）
    if form.get('sma_show'):
        indicators['sma'] = True
        indicators['sma_periods'] = [_to_int(form.get('sma_short'), 20), _to_int(form.get('sma_long'), 50)]
    if form.get('stoch_show'):
        indicators['stoch'] = True
        indicators['stoch_k_period'] = _to_int(form.get('stoch_k_period'), 14)
        indicators['stoch_d_period'] = _to_int(form.get('stoch_d_period'), 3)
    if form.get('atr_show'):
        indicators['atr'] = True
        indicators['atr_period'] = _to_int(form.get('atr_period'), 14)

    return indicators


def _strategy_params_from_names(names):
    """按策略名列表构建指标参数（用于恢复历史量化任务）"""
    p = {}
    for n in names:
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


def _get_symbols(market_type):
    """返回固定交易对列表 (symbol, 显示名称)，合约模式加 :USDT 后缀"""
    if market_type == 'future':
        symbols = [(s + ":USDT" if not s.endswith(":USDT") else s, label) for s, label in SYMBOL_LIST]
    else:
        symbols = list(SYMBOL_LIST)
    default_symbol = symbols[0][0]
    return symbols, default_symbol


def _get_market_overview(market_type):
    """获取市场概览（主要币种当前价格）——单次批量请求 + 60秒缓存，避免每次打开首页都走代理阻塞"""
    suffix = ":USDT" if market_type == 'future' else ""
    main_coins = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT']
    tokens = [coin + suffix for coin in main_coins]
    now = time.time()
    cache_key = f"overview_{market_type}"
    # 命中缓存则直接用，避免长耗时网络请求阻塞/排队
    if _overview_cache.get('key') == cache_key and now - _overview_cache.get('ts', 0) < 60:
        return _overview_cache.get('rows', [])
    prices = data_fetcher.get_tickers_prices(tokens)
    rows = []
    for coin in main_coins:
        price = prices.get(coin + suffix)
        if price:
            rows.append({"coin": coin, "price": f"${price:,.2f}"})
    # 拉取失败（代理抖动/超时）时回退旧缓存，避免页面白等
    if not rows and _overview_cache.get('key') == cache_key and _overview_cache.get('rows'):
        return _overview_cache['rows']
    _overview_cache['key'] = cache_key
    _overview_cache['ts'] = now
    _overview_cache['rows'] = rows
    return rows


def _render_strategy_params(form, selected_strategies):
    """渲染所有策略的参数表单 HTML，每个策略一个可切换区块"""
    def num(name, label, value, minv, maxv, step="1"):
        return f"""
        <div class="field-inline">
            <label>{label}</label>
            <input type="number" name="{name}" value="{value}" min="{minv}" max="{maxv}" step="{step}">
        </div>"""

    def section(key, title, inner_html):
        display = "block" if key in selected_strategies else "none"
        return f'<div class="strategy-params" id="params-{key}" style="display:{display}"><div class="params-title">{title}</div>{inner_html}</div>'

    sections = []
    sections.append(section("rsi", "RSI 参数",
        num("rsi_period", "RSI周期", form.get('rsi_period') or 14, 5, 50)
        + num("rsi_oversold", "超卖线", form.get('rsi_oversold') or 30, 10, 40)
        + num("rsi_overbought", "超买线", form.get('rsi_overbought') or 70, 60, 90)
    ))
    sections.append(section("kdj", "KDJ 参数",
        num("kdj_k_period", "K周期", form.get('kdj_k_period') or 9, 5, 20)
        + num("kdj_d_period", "D周期", form.get('kdj_d_period') or 3, 2, 10)
        + num("kdj_buy_threshold", "买入阈值", form.get('kdj_buy_threshold') or 20, 10, 30)
        + num("kdj_sell_threshold", "卖出阈值", form.get('kdj_sell_threshold') or 80, 70, 90)
    ))
    sections.append(section("boll", "布林带参数",
        num("bb_period", "布林带周期", form.get('bb_period') or 20, 10, 50)
        + num("bb_std", "标准差倍数", form.get('bb_std') or 2.0, 1.0, 3.0, "0.1")
    ))
    sections.append(section("ema", "EMA 参数",
        num("ema_short", "短期EMA", form.get('ema_short') or 12, 5, 20)
        + num("ema_long", "长期EMA", form.get('ema_long') or 26, 20, 50)
    ))
    sections.append(section("macd", "MACD 参数",
        num("macd_fast", "MACD快线", form.get('macd_fast') or 12, 5, 20)
        + num("macd_slow", "MACD慢线", form.get('macd_slow') or 26, 20, 50)
        + num("macd_signal", "MACD信号线", form.get('macd_signal') or 9, 5, 20)
    ))
    sections.append(section("ma_cross", "双均线交叉参数",
        num("ma_cross_short", "短期均线", form.get('ma_cross_short') or 10, 5, 20)
        + num("ma_cross_long", "长期均线", form.get('ma_cross_long') or 30, 20, 60)
    ))
    return "\n".join(sections)


@app.route("/", methods=["GET", "POST"])
def index():
    # 当前市场类型（现货/合约）
    market_label = request.form.get("market_type", "现货")
    market_type = 'future' if market_label == '合约' else 'spot'
    data_fetcher.set_market_type(market_type)

    # 默认表单值
    default_start = datetime.now().strftime("%Y-%m-%d")
    default_end = datetime.now().strftime("%Y-%m-%d")

    symbols, default_symbol = _get_symbols(market_type)

    # 从表单读取（未提交时用默认值）
    form = request.form
    selected_symbol = form.get("symbol", default_symbol)
    selected_timeframe = form.get("timeframe", "15分钟")
    # 多选策略：从复选框读取，默认 RSI
    strategy_keys = {"strategy_rsi": "RSI", "strategy_kdj": "KDJ", "strategy_boll": "布林带", "strategy_ema": "EMA", "strategy_macd": "MACD", "strategy_ma_cross": "双均线交叉"}
    if request.method == "POST":
        selected_strategies = [v for k, v in strategy_keys.items() if form.get(k)]
    else:
        selected_strategies = ["RSI"]  # GET 默认选 RSI
    initial_capital = _to_float(form.get("initial_capital"), 10000)
    commission = _to_float(form.get("commission"), 0.1) / 100
    start_date = form.get("start_date", default_start)
    end_date = form.get("end_date", default_end)
    use_take_profit = form.get("use_take_profit")
    use_stop_loss = form.get("use_stop_loss")
    take_profit_pct = _to_float(form.get("take_profit_pct"), 3) / 100 if use_take_profit else None
    stop_loss_pct = _to_float(form.get("stop_loss_pct"), 3) / 100 if use_stop_loss else None

    # 构建展示用表单值（GET 时用默认值，POST 时用表单值）
    form_state = {k: form.get(k) for k in [
        "market_type", "initial_capital", "commission", "symbol", "timeframe",
        "strategy_rsi", "strategy_kdj", "strategy_boll", "strategy_ema", "strategy_macd", "strategy_ma_cross",
        "use_take_profit", "use_stop_loss",
        "take_profit_pct", "stop_loss_pct",
        "rsi_period", "rsi_oversold", "rsi_overbought",
        "kdj_k_period", "kdj_d_period", "kdj_buy_threshold", "kdj_sell_threshold",
        "bb_period", "bb_std", "ema_short", "ema_long",
        "macd_fast", "macd_slow", "macd_signal",
        "sma_show", "sma_short", "sma_long", "stoch_show", "stoch_k_period",
        "stoch_d_period", "atr_show", "atr_period",
    ]}
    # 日期使用已计算的默认值（GET 时 form.get 返回 None）
    form_state["start_date"] = start_date
    form_state["end_date"] = end_date
    form_state["initial_capital"] = initial_capital
    form_state["commission"] = commission * 100
    # 确保策略复选框状态
    for k in strategy_keys:
        form_state[k] = form.get(k)

    # 构建回测上下文
    context = {
        "form": form_state,
        "market_label": market_label,
        "symbols": symbols,
        "selected_symbol": selected_symbol,
        "selected_timeframe": selected_timeframe,
        "timeframe_keys": list(TIMEFRAME_OPTIONS.keys()),
        "market_choices": ["现货", "合约"],
        "strategies": STRATEGIES,
        "selected_strategies": selected_strategies,
        "strategy_params_html": _render_strategy_params(form_state, selected_strategies),
    }

    if request.method == "POST" and form.get("run_backtest") == "1":
        # 运行回测（仅在点击"运行回测"按钮时执行，策略切换不触发）
        if not selected_strategies:
            context["error"] = "请至少选择一个策略"
        else:
          try:
            indicator_params = _build_indicators(form, selected_strategies)
            print(f"[DEBUG] 策略参数: {indicator_params}")
            print(f"[DEBUG] symbol={selected_symbol}, timeframe={selected_timeframe}, dates={start_date}~{end_date}")
            df = data_fetcher.fetch_historical_data(
                selected_symbol, start_date, end_date, TIMEFRAME_OPTIONS[selected_timeframe]
            )

            if df.empty:
                context["error"] = "无法获取数据，请检查网络连接或选择其他时间范围"
            else:
                df_with_indicators = TechnicalIndicators.calculate_all_indicators(df, indicator_params)
                engine = BacktestEngine(initial_capital, commission, take_profit_pct, stop_loss_pct, TIMEFRAME_OPTIONS[selected_timeframe], signal_mode='or')
                results = engine.run_backtest(df_with_indicators, indicator_params)
                print(f"[DEBUG] 数据行数={len(df)}, 交易数={len(results.get('trades', []))}, 总收益率={results.get('total_return')}")

                if results:
                    metrics = engine.get_performance_metrics()

                    # 保存回测结果到本地
                    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
                    os.makedirs(save_dir, exist_ok=True)
                    save_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    symbol_safe = selected_symbol.replace("/", "_")

                    trades_path = None
                    if not results['trades'].empty:
                        trades_path = os.path.join(save_dir, f"trades_{symbol_safe}_{save_ts}.csv")
                        results['trades'].to_csv(trades_path, index=False, encoding="utf-8-sig")

                    summary = {
                        "时间": datetime.now().isoformat(),
                        "交易对": selected_symbol,
                        "开始日期": str(start_date),
                        "结束日期": str(end_date),
                        "时间周期": selected_timeframe,
                        "市场类型": market_label,
                        "初始资金": initial_capital,
                        "手续费率": commission,
                        "止盈": take_profit_pct,
                        "止损": stop_loss_pct,
                        "策略参数": indicator_params,
                        "总收益率": results.get("total_return"),
                        "年化收益率": results.get("annual_return"),
                        "最大回撤": results.get("max_drawdown"),
                        "夏普比率": results.get("sharpe_ratio"),
                        "胜率": results.get("win_rate"),
                        "总交易次数": results.get("total_trades"),
                        "最终资金": results.get("final_equity"),
                    }
                    summary_path = os.path.join(save_dir, f"summary_{symbol_safe}_{save_ts}.json")
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump(summary, f, ensure_ascii=False, indent=2)

                    # 构建图表 JSON（供 Plotly.js 渲染）
                    selected_indicators = [k for k in indicator_params.keys() if k not in [
                        'rsi_period', 'rsi_oversold', 'rsi_overbought',
                        'kdj_k_period', 'kdj_d_period', 'kdj_j_period',
                        'kdj_buy_threshold', 'kdj_sell_threshold',
                        'bb_period', 'bb_std', 'ema_periods', 'ema_short',
                        'ema_long', 'sma_periods', 'macd_fast', 'macd_slow',
                        'macd_signal', 'stoch_k_period', 'stoch_d_period', 'atr_period']]

                    charts = {
                        "technical": ChartUtils.create_technical_chart(
                            df_with_indicators, selected_indicators, f"{selected_symbol} 技术分析").to_json(),
                        "equity": ChartUtils.create_equity_chart(
                            results['equity_curve'], f"{selected_symbol} 权益曲线").to_json(),
                        "drawdown": ChartUtils.create_drawdown_chart(
                            results['equity_curve'], f"{selected_symbol} 回撤分析").to_json(),
                        "trade": ChartUtils.create_trade_chart(
                            df_with_indicators, results['trades'], f"{selected_symbol} 交易点位").to_json(),
                    }

                    trades_json = results['trades'].to_json(orient="records", date_format="iso")
                    trades_csv = results['trades'].to_csv(index=False) if not results['trades'].empty else ""
                    session["trades_csv"] = trades_csv  # 存入 session 供下载使用

                    context.update({
                        "data_count": len(df),
                        "metrics": list(metrics.items()),
                        "charts": charts,
                        "charts_json": json.dumps(charts) if charts else '{}',
                        "trades_json": trades_json,
                        "trades_csv": trades_csv,
                        "has_trades": not results['trades'].empty,
                        "trades_count": len(results['trades']),
                        "indicator_params": indicator_params,
                        "save_dir": save_dir,
                        "saved_file": os.path.basename(trades_path) if trades_path else "",
                    })
                else:
                    context["error"] = "回测运行失败，请检查参数设置"
          except Exception as e:
            context["error"] = f"回测出错: {e}"
    else:
        # GET：显示欢迎页与市场概览
        context["market_overview"] = _get_market_overview(market_type)

    return render_template("index.html", **context)


@app.route("/download", methods=["GET"])
def download_trades():
    csv = session.get("trades_csv", "")
    if not csv:
        abort(404)
    return send_file(
        io.BytesIO(csv.encode("utf-8-sig")),
        as_attachment=True,
        download_name="trades.csv",
        mimetype="text/csv"
    )


# ==================== 模拟现货交易 ====================

DEMO_SYMBOLS = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "BNB/USDT", "MUUB/USDT", "TQQQ/USDT", "QQQ/USDT"]
# 现货 Demo 默认 API 密钥（已移除硬编码，请在页面手动填写或配置环境变量）
DEFAULT_DEMO_API_KEY = ""
DEFAULT_DEMO_API_SECRET = ""

def _get_demo_trader():
    """从 session 中恢复 DemoTrader 实例（按密钥缓存，避免每次重建 ccxt 拖慢页面）"""
    api_key = session.get("demo_api_key", DEFAULT_DEMO_API_KEY)
    api_secret = session.get("demo_api_secret", DEFAULT_DEMO_API_SECRET)
    key = api_key
    if key and key in _demo_traders:
        t = _demo_traders[key]
        # 密钥未变化则复用，变化才重建
        if t.api_secret == api_secret:
            return t
    t = DemoTrader(api_key, api_secret)
    if key:
        _demo_traders[key] = t
    return t

def _get_demo_trader_for(api_key, api_secret):
    """按传入密钥获取（或缓存）DemoTrader，供 /demo 路由使用"""
    if api_key and api_key in _demo_traders:
        t = _demo_traders[api_key]
        if t.api_secret == api_secret:
            return t
    t = DemoTrader(api_key, api_secret) if api_key and api_secret else None
    if t and api_key:
        _demo_traders[api_key] = t
        # 后台预热该实例 market 目录，避免首次请求（自动交易信号/余额）触发 load_markets 卡十几秒
        threading.Thread(target=_prewarm_trader, args=(t,), daemon=True).start()
    return t

def _prewarm_trader(trader):
    """后台预热单个 DemoTrader 实例的 market 目录，避免阻塞请求线程"""
    try:
        trader.exchange.load_markets()
    except Exception:
        pass


# ==================== 自动合约交易 ====================

# 合约交易对列表（USDT永续；bStocks 永续需用完整符号 :USDT）
FUTURES_SYMBOLS = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT",
                   "TON/USDT", "NEAR/USDT", "AVAX/USDT", "XLM/USDT", "ICP/USDT", "LTC/USDT",
                   "MU/USDT:USDT", "QQQ/USDT:USDT", "TQQQ/USDT:USDT"]
# 合约交易器缓存与自动合约引擎
_futures_traders = {}
_auto_futures = None
# 每格默认参数
FUTURES_LEVERAGE = 5
# 合约测试网默认 API 密钥（已移除硬编码，请在页面手动填写或配置环境变量）
DEFAULT_FUTURES_API_KEY = ""
DEFAULT_FUTURES_API_SECRET = ""


def _get_futures_trader(api_key, api_secret, leverage=FUTURES_LEVERAGE, testnet=True):
    """按密钥+网络获取（或缓存）FuturesTrader 实例，避免每次重建 ccxt 触发 load_markets 拖慢"""
    cache_key = f"{api_key}|{'test' if testnet else 'main'}"
    if api_key and cache_key in _futures_traders:
        t = _futures_traders[cache_key]
        if t.api_secret == api_secret:
            return t
    t = FuturesTrader(api_key, api_secret, testnet=testnet, leverage=leverage) if api_key and api_secret else None
    if t and api_key:
        _futures_traders[cache_key] = t
        threading.Thread(target=_prewarm_trader, args=(t,), daemon=True).start()
    return t


# 合约页面展示数据缓存：页面渲染原先串行执行 5~6 个走代理的请求，代理抖动时每次都要等数秒。
# 这里聚合成一次缓存读取（TTL 8 秒），刷新页面不再重复拉取；实时性由 JS 轮询 /futures/api/status 保证。
_futures_page_cache = {'key': None, 'ts': 0, 'data': None}


def _futures_display_data(auto_futures, symbol, ttl=8):
    now = time.time()
    key = f"{symbol}|{auto_futures.api_key}|{auto_futures.trader.testnet}"
    if _futures_page_cache.get('key') == key and now - _futures_page_cache.get('ts', 0) < ttl:
        return _futures_page_cache['data']
    data = {'mark_price': None, 'positions': [], 'open_orders': [], 'balance': None}
    try:
        auto_futures._refresh_real_position(symbol)
    except Exception:
        pass
    try:
        data['mark_price'], _ = auto_futures.trader.get_ticker(symbol)
    except Exception:
        pass
    try:
        data['positions'], _ = auto_futures.trader.get_positions(symbol)
    except Exception:
        data['positions'] = []
    try:
        bal_list, _ = auto_futures.trader.get_balance()
        for b in bal_list or []:
            if b['asset'] == 'USDT':
                data['balance'] = b['free']
                break
    except Exception:
        data['balance'] = None
    try:
        data['open_orders'], _ = auto_futures.trader.get_open_orders()
    except Exception:
        data['open_orders'] = []
    _futures_page_cache['key'] = key
    _futures_page_cache['ts'] = now
    _futures_page_cache['data'] = data
    return data


@app.route("/futures", methods=["GET", "POST"])
def futures():
    """自动合约交易监控页面"""
    from flask import jsonify
    form = request.form
    # strip() 去除复制粘贴带入的空格/换行（否则币安报 -2008 Invalid Api-Key ID）
    api_key = (form.get("api_key") or session.get("futures_api_key", DEFAULT_FUTURES_API_KEY)).strip()
    api_secret = (form.get("api_secret") or session.get("futures_api_secret", DEFAULT_FUTURES_API_SECRET)).strip()
    leverage = _to_int(form.get("leverage"), session.get("futures_leverage", FUTURES_LEVERAGE))
    # 网络切换：testnet(默认)/mainnet(主网真实资金)。测试网无 TQQQ 等美股永续
    network = form.get("network") or session.get("futures_network", "testnet")
    testnet = network != "mainnet"
    error = None
    message = None

    # 保存/绑定合约 API 密钥（PRG，刷新不重放；绑定成功后刷新账户余额）
    if form.get("save_keys"):
        session["futures_api_key"] = api_key
        session["futures_api_secret"] = api_secret
        session["futures_leverage"] = leverage
        session["futures_network"] = network
        return redirect(url_for('futures', _saved='1'))

    if not api_key or not api_secret:
        return render_template("futures.html", error="请输入合约测试网 API 密钥",
                              message=None, running=False, status=None, symbols=FUTURES_SYMBOLS,
                              timeframe_options=TIMEFRAME_OPTIONS, strategies=STRATEGIES,
                              leverage=leverage, api_key=api_key, api_secret=api_secret, network=network,
                              alert_email=(_auth.get_email_config(session.get('user', '')) or {}).get('email'),
                              mark_price=None, positions=None, open_orders=None, account_balance=None)

    global _auto_futures
    shared_trader = _get_futures_trader(api_key, api_secret, leverage, testnet)
    if not _auto_futures or _auto_futures.api_key != api_key or _auto_futures.trader.testnet != testnet:
        _auto_futures = AutoFutures(api_key, api_secret, trader=shared_trader, leverage=leverage)
    # 同步杠杆变更到引擎与交易器（改杠杆无需重启，下次开仓即按新杠杆计算张数）
    _auto_futures.leverage = leverage
    if shared_trader:
        shared_trader.leverage = leverage
    if not _auto_futures.status.get('running'):
        _auto_futures.status['leverage'] = leverage

    # 若尚未启动，从磁盘恢复上次的配置与持仓
    if not _auto_futures.status.get('running'):
        _auto_futures.load_state()

    # 展示数据（持仓/余额/价格/挂单）走 8 秒缓存，避免每次刷新页面都串行等待多个代理请求
    disp = _futures_display_data(_auto_futures, _auto_futures.status.get('symbol') or 'BTC/USDT')

    # 绑定成功提示：校验密钥可用性并读取 USDT 余额
    if request.args.get('_saved') == '1':
        try:
            usdt = 0.0
            bal_list, bErr = _auto_futures.trader.get_balance()
            if bErr:
                raise RuntimeError(bErr)
            for b in bal_list:
                if b['asset'] == 'USDT':
                    usdt = float(b['free'])
                    break
            _auto_futures.status['account_balance'] = round(usdt, 2)
            message = f"API 密钥绑定成功！合约账户 USDT 余额：{usdt:.2f} USDT"
        except Exception as e:
            error = f"API 密钥绑定失败，请检查密钥/IP/合约权限：{e}"

    # 启停/平仓等 action 传入的错误信息优先展示
    arg_error = request.args.get('_error', '') or None
    if arg_error:
        error = arg_error
    if request.method == "POST" and form.get("action") == "start":
        symbol = form.get("symbol", "BTC/USDT")
        timeframe = form.get("timeframe", "15m")
        qty_usdt = _to_float(form.get("qty_usdt"), 1000)
        interval = _to_int(form.get("interval"), 30)
        mode = form.get("mode", "standard")
        session["futures_leverage"] = leverage  # 启动时持久化当前杠杆
        stop_pct = _to_float(form.get("stop_pct"), 5) / 100  # 断线保护止损比例
        # 测试网只有主流币永续，无 TQQQ 等美股合约
        if testnet and ":USDT:USDT" in symbol:
            return redirect(url_for('futures', _error=f"测试网无 {symbol} 合约，请把网络切换为「主网」或选择币种合约"))
        if mode == 'grid':
            step_pct = _to_float(form.get("grid_step"), 1) / 100
            max_levels = _to_int(form.get("grid_max_levels"), 12)
            ok, msg = _auto_futures.start(symbol, timeframe, {}, qty_usdt, interval,
                                          strategies=[], mode=mode, step_pct=step_pct, max_levels=max_levels,
                                          stop_pct=0)
        else:
            selected_strategies = [v for v in STRATEGIES if form.get(f"strategy_{v.lower()}")]
            if not selected_strategies:
                selected_strategies = ["RSI"]
            indicator_params = _build_indicators(form, selected_strategies)
            long_only = bool(form.get("long_only"))  # 仅做多：卖出信号只平仓不开空
            ok, msg = _auto_futures.start(symbol, timeframe, indicator_params, qty_usdt, interval,
                                          strategies=selected_strategies, mode='standard',
                                          stop_pct=stop_pct, long_only=long_only)
        return redirect(url_for('futures', _error='' if ok else msg))

    elif request.method == "POST" and form.get("action") == "resume_task":
        # 恢复历史量化任务（服务崩溃后从任务列表一键续跑）
        tid = form.get("task_id")
        task = next((t for t in _auto_futures.list_tasks() if t.get('id') == tid), None)
        if not task:
            return redirect(url_for('futures', _error=f"任务不存在或已丢失: {tid}"))
        if _auto_futures.status.get('running'):
            return redirect(url_for('futures', _error='已有任务在运行，请先停止再恢复'))
        if testnet and ":USDT:USDT" in task['symbol']:
            return redirect(url_for('futures', _error=f"测试网无 {task['symbol']} 合约，请把网络切换为「主网」后恢复"))
        _auto_futures.leverage = int(task.get('leverage', 5))
        if shared_trader:
            shared_trader.leverage = _auto_futures.leverage
        sp = float(task.get('stop_pct', 0) or 0)
        if task.get('mode') == 'grid':
            ok, msg = _auto_futures.start(task['symbol'], task['timeframe'], {},
                                          float(task.get('qty_usdt', 1000)), int(task.get('interval', 30)),
                                          strategies=[], mode='grid',
                                          step_pct=float(task.get('grid_step', 0.01)),
                                          max_levels=int(task.get('grid_max_levels', 12)), stop_pct=0)
        else:
            names = task.get('strategies') or ["RSI"]
            ip = _strategy_params_from_names(names)
            ok, msg = _auto_futures.start(task['symbol'], task['timeframe'], ip,
                                          float(task.get('qty_usdt', 1000)), int(task.get('interval', 30)),
                                          strategies=names, mode='standard', stop_pct=sp,
                                          long_only=bool(task.get('long_only', False)))
        return redirect(url_for('futures', _error='' if ok else msg))

    elif request.method == "POST" and form.get("action") == "delete_task":
        # 删除量化任务记录（仅删历史记录，不影响运行中的引擎）
        ok, msg = _auto_futures.delete_task(form.get("task_id"))
        return redirect(url_for('futures', _error='' if ok else msg))

    elif request.method == "POST" and form.get("action") == "save_email":
        # 旧入口转发到设置页（邮箱配置已迁移至 /settings）
        return redirect(url_for('settings'))

    elif request.method == "POST" and form.get("action") == "test_email":
        return redirect(url_for('settings'))

    elif request.method == "POST" and form.get("action") == "stop":
        _auto_futures.stop()
        return redirect(url_for('futures'))
    elif request.method == "POST" and form.get("action") == "close_position":
        # 手动平掉当前持仓
        if _auto_futures:
            symbol = _auto_futures.status.get('symbol') or 'BTC/USDT'
            side = _auto_futures.status.get('side', 'none')
            if side != 'none':
                _auto_futures._close_position(symbol, side)
                error = None
        return redirect(url_for('futures'))

    status = _auto_futures.get_status()

    # 展示数据直接取缓存结果（_futures_display_data 已聚合拉取）
    mark_price = disp['mark_price']
    positions = disp['positions']
    acc_bal = disp['balance']
    open_orders = disp['open_orders']

    return render_template("futures.html", error=error, message=message,
                          running=status['running'], status=status, symbols=FUTURES_SYMBOLS,
                          timeframe_options=TIMEFRAME_OPTIONS,
                          strategies=STRATEGIES, mark_price=mark_price,
                          positions=positions, open_orders=open_orders,
                          account_balance=acc_bal, leverage=leverage, network=network,
                          tasks=_auto_futures.list_tasks(),
                          api_key=api_key, api_secret=api_secret)


# ==================== 最优策略汇总（历次寻优回测结果） ====================
# 2026 窗口为 2026-01-01 ~ 2026-08-26；2025 组为全年；bStocks 上市较晚，窗口以数据起始为准
STRATEGY_SUMMARY = [
    {
        'name': '最终推荐（按品种分类）', 'tag': 'rec', 'window': '2023~2026 四年验证',
        'desc': '全部研究（12000+组寻优 + 1700组山寨币扫描）收敛后的最终结论：主流币用超卖回归，山寨币用趋势跟随，方向相反不可混用',
        'rows': [
            {'品种': 'ETH/USDT', '周期': '4h', '策略': 'KDJ(9,3,3) 买20/卖80', '模式': '仅做多', '止盈止损': '8%/5%',
             '收益': '+40.5%', '月均': '~1.0%', '交易次数': 102, '胜率': '57.9%', '最大回撤': '-38.9%',
             '稳定性': '四年3盈 最差年-7%', '市场': '合约/现货', '备注': '★主力：唯一无灾难年的配置，超卖回归+快进快出'},
            {'品种': 'XLM/USDT', '周期': '4h', '策略': 'EMA(12/26)', '模式': '双向', '止盈止损': '5%/5%',
             '收益': '+239.1%', '月均': '~3.6%', '交易次数': 234, '胜率': '43.8%', '最大回撤': '-33.1%',
             '稳定性': '四年3盈 最差年-11.8%', '市场': '合约', '备注': '★山寨仓：高波动币趋势跟随，熊市靠空头盈利(2025+38.9pp/2026+42.6pp)'},
            {'品种': 'BTC/USDT', '周期': '4h', '策略': 'KDJ(9,3,3) 买20/卖80', '模式': '仅做多', '止盈止损': '8%/5%',
             '收益': '+19.2%', '月均': '~0.5%', '交易次数': 94, '胜率': '59.0%', '最大回撤': '-24.0%',
             '稳定性': '四年2盈2亏', '市场': '合约/现货', '备注': '可选第二路：累计为正但年度胜负随机，仓位减半'},
            {'品种': 'ETH/BTC等蓝筹', '周期': '4h', '策略': 'KDJ/RSI/布林带', '模式': '双向做空', '止盈止损': '任意',
             '收益': '亏损/爆仓', '交易次数': '-', '胜率': '-', '最大回撤': '-100%',
             '稳定性': '10币扫描反转做空21次爆仓', '市场': '-', '备注': '❌禁止：蓝筹超卖回归策略移植到山寨币0/10全亏'},
            {'品种': 'DOGE~XLM等10山寨', '周期': '4h', '策略': 'KDJ(9,3,3)', '模式': '任意', '止盈止损': '任意',
             '收益': '0/10为正', '交易次数': '-', '胜率': '-', '最大回撤': '-84%~-92%',
             '稳定性': '平均四年-55%~-92%', '市场': '-', '备注': '❌禁止：KDJ只对ETH有效，DOGE~XLM十币全军覆没'},
        ],
    },
    {
        'name': '市值10-20山寨币四年扫描 Top', 'tag': 'aggressive', 'window': '2023~2026（约1700组）',
        'desc': 'DOGE/TRX/LINK/DOT/BCH/LTC/UNI/APT/ICP/XLM × 7策略 × 双向/仅多 × 3组止盈止损 × 4年。'
                '止盈止损规则与蓝筹币相反：趋势单不设让利润奔跑（收益最高但回撤-40%+），设5/5收益砍2/3换回撤减半',
        'rows': [
            {'品种': 'XLM/USDT', '周期': '4h', '策略': 'EMA(12/26) 双向 不设', '模式': '双向', '止盈止损': '不设',
             '收益': '+815.4%', '月均': '~12%', '交易次数': 244, '胜率': '35.5%', '最大回撤': '-43.1%',
             '稳定性': '收益高度集中', '市场': '合约', '备注': '⚠️+815%几乎全靠2024年11月单月+402%（XLM暴涨），不可外推'},
            {'品种': 'XLM/USDT', '周期': '4h', '策略': 'EMA(12/26) 双向 5/5', '模式': '双向', '止盈止损': '5%/5%',
             '收益': '+239.1%', '月均': '~3.6%', '交易次数': 234, '胜率': '42.4%', '最大回撤': '-33.1%',
             '稳定性': '最差年-11.8%', '市场': '合约', '备注': '★山寨首选（同上保护版）：2023~-11.8/2024+74.4/2025+30.4/2026+69.0'},
            {'品种': 'ICP/USDT', '周期': '4h', '策略': '双均线(10/30) 双向 不设', '模式': '双向', '止盈止损': '不设',
             '收益': '+330.9%', '月均': '~4.4%', '交易次数': 307, '胜率': '38.5%', '最大回撤': '-61.1%',
             '稳定性': '四年全正/回撤深', '市场': '合约', '备注': '+74.7/+107.8/+13.2/+4.9 四年每年都正，但回撤-42%~-61%，过山车'},
            {'品种': 'LTC/USDT', '周期': '4h', '策略': 'RSI(14) 双向 不设', '模式': '双向', '止盈止损': '不设',
             '收益': '+311.2%', '月均': '~4.2%', '交易次数': 73, '胜率': '72.6%', '最大回撤': '-90.9%',
             '稳定性': '四年全正/极波动', '市场': '合约', '备注': '⚠️高胜率双向，空头连续3年主要盈利；但同策略其余9币爆仓12币·年，10选1幸存者'},
            {'品种': 'BCH/USDT', '周期': '4h', '策略': 'EMA 仅多 不设', '模式': '仅做多', '止盈止损': '不设',
             '收益': '+206.7%', '月均': '~2.7%', '交易次数': 129, '胜率': '29.3%', '最大回撤': '-37.7%',
             '稳定性': '不跨牛熊', '市场': '合约/现货', '备注': '2023/2024牛市赚翻（+76.8/+132.3），2025/2026阴亏（-11.1/-16.0）'},
            {'品种': 'LINK/USDT', '周期': '4h', '策略': '布林带 仅多 不设', '模式': '仅做多', '止盈止损': '不设',
             '收益': '+175.9%', '月均': '~2.4%', '交易次数': 94, '胜率': '—', '最大回撤': '—',
             '稳定性': '最差年-28.2%', '市场': '合约/现货', '备注': 'LINK专属：布林带仅多在其上5币为正（LINK/TRX/LTC/UNI/BCH）'},
            {'品种': 'DOGE~XLM等10山寨', '周期': '4h', '策略': 'EMA(12/26) 仅多 不设', '模式': '仅做多', '止盈止损': '不设',
             '收益': '6/10币为正', '月均': '平均+59.9%', '交易次数': '—', '胜率': '—', '最大回撤': '平均约-30%',
             '稳定性': '唯一系统性有效', '市场': '合约/现货', '备注': '★普适性冠军：XLM+382%/BCH+207%/ICP+98%/DOGE+90%/TRX+73%/LINK+9%，双均线仅多同款6/10'},
            {'品种': 'DOGE~XLM等10山寨', '周期': '4h', '策略': '9组合AND过滤 vs EMA单策略', '模式': '双向/仅多', '止盈止损': '不设/5/5',
             '收益': '全部不如基准', '交易次数': '2712→7笔', '胜率': '—', '最大回撤': '—',
             '稳定性': '系统性证伪', '市场': '—', '备注': '❌组合策略结论：EMA+RSI/KDJ/布林40币·年零信号（趋势与超卖互斥）；过滤越狠错过大行情越多；个别亮点LINK EMA+双均线+354%属10选1，勿用'},
        ],
    },
    {
        'name': '稳健型（低回撤优先）', 'tag': 'stable', 'window': '2026年',
        'desc': '回撤小、日波动低，适合大资金长期运行',
        'rows': [
            {'品种': 'NEAR/USDT', '周期': '4h', '策略': 'EMA', '模式': '仅做多', '止盈止损': '2%/2%',
             '收益': '+32.9%', '日均': '0.13%', '交易次数': 22, '胜率': '68.2%', '最大回撤': '-4.8%',
             '稳定性': '最差日-2.8%', '市场': '合约/现货', '备注': '全场最稳'},
            {'品种': 'NEAR/USDT', '周期': '4h', '策略': 'EMA+MACD', '模式': '仅做多', '止盈止损': '3%/3%',
             '收益': '+49.6%', '日均': '0.19%', '交易次数': 62, '胜率': '51.6%', '最大回撤': '-16.2%',
             '稳定性': '最差日-4.6%', '市场': '合约/现货', '备注': '收益回撤比 3:1'},
            {'品种': 'TON/USDT', '周期': '4h', '策略': 'MACD', '模式': '双向', '止盈止损': '5%/5%',
             '收益': '+50.1%', '日均': '0.19%', '交易次数': 62, '胜率': '43.5%', '最大回撤': '-28.6%',
             '稳定性': '日波动<2%', '市场': '合约', '备注': 'MACD系在TON上稳定有效'},
            {'品种': 'ETH/USDT', '周期': '4h', '策略': 'KDJ', '模式': '仅做多', '止盈止损': '5%/5%',
             '收益': '+32.2%', '月均': '3.91%', '交易次数': 27, '胜率': '66.7%', '最大回撤': '-10.7%',
             '稳定性': '高收益中回撤最低', '市场': '合约/现货', '备注': '高收益方案中回撤最低'},
        ],
    },
    {
        'name': '均衡型（收益回撤比最优）', 'tag': 'balanced', 'window': '2026年',
        'desc': '收益与风险平衡，适合中等仓位',
        'rows': [
            {'品种': 'TQQQ/USDT', '周期': '4h', '策略': 'RSI+MACD', '模式': '双向', '止盈止损': '不设',
             '收益': '+42.3%(2026年6月底~8月)', '日均': '0.64%', '交易次数': 16, '胜率': '62.5%', '最大回撤': '-7.1%',
             '稳定性': '盈利日49%', '市场': '合约(仅主网)', '备注': '收益回撤比6:1，全场最优；3倍纳指ETF（6月29日上市）'},
            {'品种': 'TQQQ/USDT', '周期': '4h', '策略': 'MACD', '模式': '双向', '止盈止损': '5%/5%',
             '收益': '+32.3%(2026年6月底~8月)', '日均': '0.49%', '交易次数': 16, '胜率': '68.8%', '最大回撤': '-6.9%',
             '稳定性': '日波动1.9%', '市场': '合约(仅主网)', '备注': '稳定组日均最高'},
            {'品种': 'ETH/USDT', '周期': '4h', '策略': 'KDJ', '模式': '仅做多', '止盈止损': '8%/5%',
             '收益': '+19.7%', '月均': '~2.5%', '交易次数': 25, '胜率': '64.0%', '最大回撤': '-20.0%',
             '稳定性': '高胜率', '市场': '合约/现货', '备注': 'ETH月均收益冠军（2026年1~8月口径）'},
            {'品种': 'MUB/USDT', '周期': '4h', '策略': 'KDJ', '模式': '双向(模拟)', '止盈止损': '不设',
             '收益': '+201.8%(2026年6~8月)', '日均': '1.57%', '交易次数': 17, '胜率': '82.4%', '最大回撤': '-14.7%',
             '稳定性': '盈利日57%', '市场': '现货代币(主网1倍MU合约可做空)',
             '备注': '2倍美光现货代币，6月12日上市；主网无MUB/MUUB合约（测试网亦无），本结果为系统模拟多空测试'},
        ],
    },
    {
        'name': '激进型（高日均，回撤大）', 'tag': 'aggressive', 'window': '2026年',
        'desc': '波动大，务必小仓位+严格风控',
        'rows': [
            {'品种': 'TQQQ/USDT', '周期': '1h', '策略': 'KDJ', '模式': '双向', '止盈止损': '不设',
             '收益': '+54.5%(2026年6月底~8月)', '日均': '0.80%', '交易次数': 41, '胜率': '80.5%', '最大回撤': '-17.4%',
             '稳定性': '盈利日64%', '市场': '合约(仅主网)', '备注': '日均最接近1%目标'},
            {'品种': 'TON/USDT', '周期': '4h', '策略': 'MACD', '模式': '双向', '止盈止损': '不设',
             '收益': '+189.6%', '日均': '0.51%', '交易次数': 63, '胜率': '41.3%', '最大回撤': '-29.5%',
             '稳定性': '盈利日37%', '市场': '合约', '备注': '加密币2026年度冠军，靠盈亏比取胜'},
            {'品种': 'MUUB/USDT', '周期': '4h', '策略': 'KDJ', '模式': '双向(模拟)', '止盈止损': '不设',
             '收益': '+136.8%(2026年7~8月)', '日均': '2.94%', '交易次数': 8, '胜率': '87.5%', '最大回撤': '-33.4%',
             '稳定性': '盈利日53%', '市场': '现货代币(主网1倍MU合约可做空)',
             '备注': '2倍美光ETF代币，7月22日上市，仅38天8笔；主网无对应合约，本结果为系统模拟多空测试，过拟合风险极高'},
        ],
    },
    {
        'name': '2025全年回测（交叉验证）', 'tag': 'balanced', 'window': '2025年',
        'desc': '2025全年(365天)15币寻优结果，用于验证策略跨年稳健性',
        'rows': [
            {'品种': 'BCH/USDT', '周期': '4h', '策略': 'RSI+EMA', '模式': '双向', '止盈止损': '不设',
             '收益': '+337.2%', '日均': '0.47%', '交易次数': 75, '胜率': '56.0%', '最大回撤': '-43.3%',
             '稳定性': '盈利日58%', '市场': '合约', '备注': '2025年度冠军；同配置5%/5%版回撤-24.7%'},
            {'品种': 'LTC/USDT', '周期': '1h', '策略': 'RSI', '模式': '双向', '止盈止损': '不设',
             '收益': '+293.1%', '日均': '0.46%', '交易次数': 72, '胜率': '73.6%', '最大回撤': '-33.9%',
             '稳定性': '盈利日55%', '市场': '合约', '备注': '莱特币'},
            {'品种': 'XRP/USDT', '周期': '4h', '策略': 'RSI', '模式': '双向', '止盈止损': '不设',
             '收益': '+230.7%', '日均': '0.43%', '交易次数': 26, '胜率': '76.9%', '最大回撤': '-55.8%',
             '稳定性': '盈利日55%', '市场': '合约', '备注': '瑞波'},
            {'品种': 'AVAX/USDT', '周期': '4h', '策略': 'EMA+MACD', '模式': '双向', '止盈止损': '5%/5%',
             '收益': '+214.4%', '日均': '0.38%', '交易次数': 165, '胜率': '44.8%', '最大回撤': '-37.4%',
             '稳定性': '盈利日43%', '市场': '合约', '备注': '★2025/2026两年都强，跨年最稳品种'},
            {'品种': 'NEAR/USDT', '周期': '4h', '策略': 'RSI', '模式': '双向', '止盈止损': '不设',
             '收益': '+134.6%', '日均': '0.36%', '交易次数': 22, '胜率': '63.6%', '最大回撤': '-45.0%',
             '稳定性': '盈利日51%', '市场': '合约', '备注': '★2025/2026两年都强'},
            {'品种': 'BTC/USDT', '周期': '4h', '策略': '布林带', '模式': '双向', '止盈止损': '不设',
             '收益': '+65.4%', '日均': '0.16%', '交易次数': 55, '胜率': '63.6%', '最大回撤': '-36.6%',
             '稳定性': '日波动<2%', '市场': '合约', '备注': '2025稳定组冠军，盈利日53%'},
        ],
    },
]


from year_matrix import YEAR_MATRIX  # 数据单独维护于 year_matrix.py（迁移④逐年回填仍依赖）

from research_log import RESEARCH_LOG  # 数据单独维护于 research_log.py（界面暂不展示）


@app.route("/settings", methods=["GET", "POST"])
def settings():
    """个人设置：邮件告警配置 + 修改密码（需登录，before_request 已拦截未登录）"""
    error, message = None, None
    if request.method == "POST":
        action = request.form.get("action")
        if action == "save_email":
            ok, msg = _auth.save_email_config(session.get('user', ''),
                                              request.form.get("alert_email", ""),
                                              request.form.get("email_auth_code", "").strip() or None)
            error, message = (None, msg) if ok else (msg, None)
        elif action == "test_email":
            import mailer as _mailer
            ok, msg = _mailer.send_email("✅ 币安量化系统：邮件告警测试",
                                         "这是一封测试邮件。\n收到即说明邮件告警配置正确：\n"
                                         "当引擎无法访问币安时，将每10分钟提醒一次、共3封；网络恢复后另发一封恢复邮件。")
            error, message = (None, msg) if ok else (msg, None)
        elif action == "change_pwd":
            old_pwd = request.form.get("old_password", "")
            new_pwd = request.form.get("new_password", "")
            confirm = request.form.get("confirm_password", "")
            if new_pwd != confirm:
                error = '两次输入的新密码不一致'
            else:
                ok, msg = _auth.change_password(session.get('user', ''), old_pwd, new_pwd)
                error, message = (None, msg) if ok else (msg, None)
        return redirect(url_for('settings', _error=error, _message=message))

    error = request.args.get('_error') or None
    message = request.args.get('_message') or None
    alert_email = (_auth.get_email_config(session.get('user', '')) or {}).get('email')
    return render_template("settings.html", error=error, message=message, alert_email=alert_email,
                           changelog=CHANGELOG)


# ==================== 最优策略页数据种子（首次启动导入历史回测记录） ====================
# 年化夏普比率（2026-08 重跑回测，从权益曲线逐bar收益率计算，4h年化√2190；
# 来源 scripts/results/sharpe_backfill.json，由 scripts/tmp_sharpe_backfill.py 生成）
SHARPE_DATA = {
    'ETH_KDJ': '0.81', 'XLM_OPT': '1.11',
    'DOGE_EMA': '0.59', 'DOGE_KDJ': '-1.00', 'TRX_EMA': '0.57', 'TRX_KDJ': '-0.01',
    'LINK_EMA': '0.32', 'LINK_KDJ': '-0.15', 'DOT_EMA': '-0.23', 'DOT_KDJ': '-0.72',
    'BCH_EMA': '0.81', 'BCH_KDJ': '0.10', 'LTC_EMA': '-0.56', 'LTC_KDJ': '0.16',
    'UNI_EMA': '-0.12', 'UNI_KDJ': '0.19', 'APT_EMA': '-0.22', 'APT_KDJ': '-0.73',
    'ICP_EMA': '0.61', 'ICP_KDJ': '-0.99', 'XLM_EMA': '0.97', 'XLM_KDJ': '-0.01',
    'AAPL_M7': '1.42', 'MSFT_M7': '1.97', 'NVDA_M7': '2.63', 'GOOGL_M7': '1.34',
    'AMZN_M7': '1.39', 'META_M7': '2.39', 'TSLA_M7': '0.96',
    'MUB_SB': '4.54', 'MUUB_SB': '6.14',
}


def _migrate_strategy_records():
    """数据修订（幂等）：① 收益补时间周期 ② MUB/MUUB 标注模拟 ③ 补实盘口径记录 ④ yearly列+历年矩阵数据回填"""
    PERIODS = [('最终推荐', '(2023-01-01~2026-08-26)'), ('市值10-20', '(2023-01-01~2026-08-26)'),
               ('分品种最终结论', '(2023-01-01~2026-08-26)'),
               ('2025全年回测', '(2025-01-01~2025-12-31)'), ('稳健型', '(2026-01-01~2026-08-26)'),
               ('均衡型', '(2026-01-01~2026-08-26)'), ('激进型', '(2026-01-01~2026-08-26)')]
    # 美股代币精确到日：起点=上市日（首根4h K线），终点=测试数据末日 2026-08-30
    US_WINDOW = {'TSLA': '2026-01-28~2026-08-30', 'AMZN': '2026-02-09~2026-08-30',
                 'NVDA': '2026-03-26~2026-08-30', 'GOOGL': '2026-03-26~2026-08-30',
                 'META': '2026-03-26~2026-08-30', 'AAPL': '2026-04-06~2026-08-30',
                 'MSFT': '2026-04-20~2026-08-30', 'MU': '2026-04-07~2026-08-30',
                 'TQQQ': '2026-06-29~2026-08-30', 'MUB': '2026-06-12~2026-08-30',
                 'MUUB': '2026-07-22~2026-08-30'}

    # ---- ④ 老库平滑加 yearly 列，并把 YEAR_MATRIX 匹配回填为年度明细 ----
    import re as _re
    with _store._conn() as c:
        cols = [r[1] for r in c.execute("PRAGMA table_info(strategy_records)")]
        if 'yearly' not in cols:
            c.execute("ALTER TABLE strategy_records ADD COLUMN yearly TEXT")

    def _match_matrix(cfg, rec):
        """YEAR_MATRIX 配置名匹配到记录：币种前缀 + 周期 + 策略词 + 方向 + 止盈止损"""
        base = (rec['symbol'] or '').split('/')[0]
        if not cfg.startswith(base + ' '):
            return False
        for tfw in ('15m', '1h', '4h', '1d'):
            if tfw in cfg:
                if rec['timeframe'] != tfw:
                    return False
                break
        words = [w for w in ('KDJ', 'EMA', 'RSI', 'MACD', '布林', '双均线') if w in (rec['strategy'] or '')]
        if not words or not any(w in cfg for w in words):
            return False
        mode = rec['mode'] or ''
        if '双向' in cfg and '双向' not in mode and mode != '任意' and '双向做空' not in mode:
            return False
        if '仅多' in cfg and '仅做多' not in mode and mode != '任意':
            return False
        if '不设' in cfg:  # 含"不设"优先按无止盈止损匹配（EMA12/26、双均线10/30 等参数不干扰）
            if rec['tpsl'] not in (None, '', '不设'):
                return False
        else:
            ms = _re.findall(r'(\d+)/(\d+)', cfg)
            if ms and rec['tpsl'] != f"{ms[-1][0]}%/{ms[-1][1]}%":  # 取最后一个 X/X 为止盈止损
                return False
        return True

    with _store._conn() as c:
        recs = [dict(r) for r in c.execute(
            "SELECT id, symbol, strategy, mode, tpsl, yearly, timeframe FROM strategy_records").fetchall()]
        for mx in YEAR_MATRIX:
            cfg = mx.get('配置', '')
            hits = [r for r in recs if _match_matrix(cfg, r) and not r['yearly']]
            if hits:  # 同配置可能在最优区和历史区各有一条（不同来源），都写入
                detail = json.dumps({'y2023': mx['y2023'], 'y2024': mx['y2024'], 'y2025': mx['y2025'],
                                     'y2026': mx['y2026'], 'verdict': mx['评价']}, ensure_ascii=False)
                for h in hits:
                    c.execute("UPDATE strategy_records SET yearly = ? WHERE id = ?", (detail, h['id']))
                    h['yearly'] = detail

    with _store._conn() as c:
        rows = c.execute("SELECT id, ret, period, source, mode, note, symbol FROM strategy_records").fetchall()
        for r in rows:
            updates = {}
            ret = r['ret'] or ''
            per = r['period'] or ''
            # ---- ⑪ 日期独立成列：把收益尾部的 (时间标注) 拆到 period 列 ----
            m = _re.search(r'^(.*?)\(([^()]*)\)\s*$', ret)
            if m and any(k in m.group(2) for k in ('年', '月', '天', '至')):
                base, per = m.group(1), m.group(2)
                updates['ret'], updates['period'] = base, per
            if per == '四年':  # 最优两行的笼统标注 → 按来源精确化
                per = next((p[1:-1] for key, p in PERIODS if key in (r['source'] or '')), '四年')
                updates['period'] = per
            if ret and ret != '—' and not per and not updates.get('period'):
                per = next((p[1:-1] for key, p in PERIODS if key in (r['source'] or '')), '2026-01-01~2026-08-26')
                updates['period'] = per
            # ---- ⑮ 日期精确到日：旧月份/笼统标签 → 统一日期区间（美股按上市日~08-30）----
            base_coin = (r['symbol'] or '').split('/')[0].split('~')[0]
            per_now = updates.get('period') or r['period'] or ''
            if base_coin in US_WINDOW:
                if per_now != US_WINDOW[base_coin]:
                    updates['period'] = US_WINDOW[base_coin]
            elif per_now != '各股自上市至8月':
                if '四年' in per_now:
                    updates['period'] = '2023-01-01~2026-08-26'
                elif per_now == '2025全年':
                    updates['period'] = '2025-01-01~2025-12-31'
                elif per_now.startswith('2026年'):
                    updates['period'] = '2026-01-01~2026-08-26'
            if r['symbol'] in ('MUB/USDT', 'MUUB/USDT') and r['mode'] == '双向':
                updates['mode'] = '双向(模拟)'
                note = r['note'] or ''
                if '系统模拟多空测试' not in note:
                    updates['note'] = '美股代币做空需走主网合约；MUB/MUUB为2倍现货代币、暂无对应合约（主网现有1倍MU合约），测试网亦无此品种。本结果为系统模拟多空测试' + (('；' + note) if note else '')
            if updates:
                sets = ", ".join(f"{k} = ?" for k in updates)
                c.execute(f"UPDATE strategy_records SET {sets} WHERE id = ?", (*updates.values(), r['id']))
        # ---- ⑩ 最优策略两条的夏普回填（仅置顶行，空值才写，幂等）----
        for sym, sv in (('ETH/USDT', SHARPE_DATA.get('ETH_KDJ')),
                        ('XLM/USDT', SHARPE_DATA.get('XLM_OPT'))):
            if sv:
                c.execute("UPDATE strategy_records SET sharpe=? WHERE symbol=? AND is_top=1 "
                          "AND (sharpe IS NULL OR sharpe='')", (sv, sym))
        # ---- ⑫ 其余历史记录夏普回填（2026-08 复跑回测并核对累计收益≤20%偏差后才采信；
        #      MUUB双向/BTC-KDJ/ETH-2026 三条原口径无法复现、蓝筹做空汇总行非单一回测，保持'—'）----
        # (symbol, strategy, mode, tpsl, timeframe, sharpe)
        EXTRA_SHARPE = [
            ('AVAX/USDT', 'EMA+MACD', '双向', '5%/5%', '4h', '2.04'),
            ('BCH/USDT', 'RSI+EMA', '双向', '不设', '4h', '2.40'),
            ('BTC/USDT', '布林带', '双向', '不设', '4h', '1.40'),
            ('LTC/USDT', 'RSI', '双向', '不设', '1h', '2.04'),
            ('NEAR/USDT', 'RSI', '双向', '不设', '4h', '1.34'),
            ('XRP/USDT', 'RSI', '双向', '不设', '4h', '1.81'),
            ('MUB/USDT', 'KDJ', '双向(模拟)', '不设', '4h', '6.43'),
            ('TQQQ/USDT', 'RSI+MACD', '双向', '不设', '4h', '4.23'),
            ('TQQQ/USDT', 'MACD', '双向', '5%/5%', '4h', '4.47'),
            ('TQQQ/USDT', 'KDJ', '双向', '不设', '1h', '4.55'),
            ('TON/USDT', 'MACD', '双向', '不设', '4h', '2.83'),
            ('TON/USDT', 'MACD', '双向', '5%/5%', '4h', '1.75'),
            ('ICP/USDT', '双均线(10/30) 双向 不设', '双向', '不设', '4h', '0.89'),
            ('LINK/USDT', '布林带 仅多 不设', '仅做多', '不设', '4h', '0.79'),
            ('LTC/USDT', 'RSI(14) 双向 不设', '双向', '不设', '4h', '0.90'),
            ('XLM/USDT', 'EMA(12/26) 双向 不设', '双向', '不设', '4h', '1.15'),
            ('BCH/USDT', 'EMA 仅多 不设', '仅做多', '不设', '4h', '0.81'),
            ('NEAR/USDT', 'EMA', '仅做多', '2%/2%', '4h', '2.16'),
            ('NEAR/USDT', 'EMA+MACD', '仅做多', '3%/3%', '4h', '1.89'),
        ]
        for sym, strat, mode, tpsl, tf, sv in EXTRA_SHARPE:
            c.execute("UPDATE strategy_records SET sharpe=? WHERE symbol=? AND strategy=? AND mode=? "
                      "AND tpsl IS ? AND timeframe=? AND (sharpe IS NULL OR sharpe='')",
                      (sv, sym, strat, mode, tpsl, tf))
        # ---- ⑤ MUB/MUUB 标注口径修正（含上一版文案的平滑替换）----
        for r in c.execute("SELECT id, symbol, mode, note, market FROM strategy_records "
                           "WHERE symbol IN ('MUB/USDT','MUUB/USDT')").fetchall():
            note = r['note'] or ''
            updates = {}
            if '系统模拟多空测试' not in note:
                for old_prefix in ('现货无永续合约，做空为回测模拟、实盘仅能做多；',
                                   '现货做空不可用后的真实口径；'):
                    note = note.replace(old_prefix, '')
                if (r['mode'] or '') == '双向(模拟)':
                    updates['note'] = ('美股代币做空需走主网合约；MUB/MUUB为2倍现货代币、暂无对应合约'
                                       '（主网现有1倍MU合约），测试网亦无此品种。本结果为系统模拟多空测试'
                                       + (('；' + note) if note else ''))
                    updates['market'] = '现货代币(主网1倍MU合约可做空)'
                else:
                    updates['note'] = ('现货口径；主网1倍MU合约可实现双向做空（2倍MUB暂无合约）'
                                       + (('；' + note) if note else ''))
                    updates['market'] = '现货/主网合约'
                sets = ", ".join(f"{k} = ?" for k in updates)
                c.execute(f"UPDATE strategy_records SET {sets} WHERE id = ?", (*updates.values(), r['id']))
        # 插入 MUB/MUUB 实盘口径（仅做多）记录：upsert（存在则刷新标签字段，保留置顶/排序）
        now = datetime.now().isoformat(timespec='seconds')
        new_rows = [
            {'symbol': 'MUB/USDT', 'timeframe': '4h', 'strategy': 'KDJ', 'mode': '仅做多', 'tpsl': '不设',
             'ret': '+76.2%', 'period': '2026-06-12~2026-08-30', 'daily': '+0.96%', 'trades': '9', 'winrate': '—', 'mdd': '-12.7%',
             'stability': '6月12日上市', 'market': '现货/主网合约', 'source': '美股代币实盘口径复验（2026-08）',
             'note': '现货口径；主网1倍MU合约可实现双向做空（2倍MUB暂无合约）；对照双向模拟版+201.8%：约2/3利润来自做空',
             'sharpe': SHARPE_DATA.get('MUB_SB')},
            {'symbol': 'MUUB/USDT', 'timeframe': '4h', 'strategy': 'KDJ', 'mode': '仅做多', 'tpsl': '不设',
             'ret': '+76.0%', 'period': '2026-07-22~2026-08-30', 'daily': '+2.00%', 'trades': '5', 'winrate': '—', 'mdd': '-9.0%',
             'stability': '7月22日上市', 'market': '现货/主网合约', 'source': '美股代币实盘口径复验（2026-08）',
             'note': '现货口径；主网1倍MU合约可实现双向做空（2倍MUB暂无合约）；上市极短无统计意义，观察即可',
             'sharpe': SHARPE_DATA.get('MUUB_SB')},
        ]
        for rec in new_rows:
            exist = c.execute("SELECT id FROM strategy_records WHERE symbol=? AND source=?",
                              (rec['symbol'], rec['source'])).fetchone()
            if exist:
                sets = ", ".join(f"{f} = ?" for f in _store.FIELDS)
                c.execute(f"UPDATE strategy_records SET {sets} WHERE id = ?",
                          (*[rec.get(f) for f in _store.FIELDS], exist[0]))
            else:
                c.execute("INSERT INTO strategy_records (is_top, sort_order, created_at, " + ",".join(_store.FIELDS) + ") "
                          "VALUES (0, 0, ?, " + ",".join("?" * len(_store.FIELDS)) + ")",
                          (now, *[rec.get(f) for f in _store.FIELDS]))
        # 插入美股七姐妹扫描结果：upsert（存在则刷新标签字段，保留置顶/排序；范围汇总行随后被⑦删除）
        now2 = datetime.now().isoformat(timespec='seconds')
        m7_rows = [
            {'symbol': 'NVDA/USDT:USDT', 'timeframe': '4h', 'strategy': 'RSI+MACD', 'mode': '仅做多', 'tpsl': '不设',
             'ret': '+31.2%', 'period': '2026-03-26~2026-08-30', 'daily': '+0.20%', 'trades': '29', 'winrate': '—', 'mdd': '-8.9%',
             'stability': '前后半都赚', 'market': '主网合约', 'source': '美股七姐妹扫描（2026-08）',
             'note': '★七姐妹最强个股：前半+20%/后半+8%；标的自身+25.4%，策略略胜且回撤更浅',
             'sharpe': SHARPE_DATA.get('NVDA_M7')},
            {'symbol': 'META/USDT:USDT', 'timeframe': '4h', 'strategy': 'RSI', 'mode': '仅做多', 'tpsl': '5%/5%',
             'ret': '+26.4%', 'period': '2026-03-26~2026-08-30', 'daily': '+0.17%', 'trades': '8', 'winrate': '—', 'mdd': '-10.0%',
             'stability': '前后半都赚', 'market': '主网合约', 'source': '美股七姐妹扫描（2026-08）',
             'note': '横盘股策略大幅超越标的（标的仅+4.2%）；8笔小样本',
             'sharpe': SHARPE_DATA.get('META_M7')},
            {'symbol': 'AAPL~TSLA等7币', 'timeframe': '4h', 'strategy': 'RSI(14)', 'mode': '仅做多', 'tpsl': '5%/5%',
             'ret': '+9.4%', 'period': '各股自上市~2026-08-30', 'daily': '—', 'trades': '—', 'winrate': '—', 'mdd': '—',
             'stability': '7/7全胜', 'market': '主网合约', 'source': '美股七姐妹扫描（2026-08）',
             'note': '★普适性冠军：唯一7/7全胜配置，蓝筹属性与ETH同族（超卖回归+快进快出）；EMA仅多不设6/7(+10.6%)；7股上线：TSLA 1月/AMZN 2月/NVDA·GOOGL·META 3月/AAPL·MSFT 4月'},
            {'symbol': 'AAPL~TSLA等7币', 'timeframe': '4h', 'strategy': '双向做空(全部28配置)', 'mode': '双向', 'tpsl': '—',
             'ret': '全部≤4/7', 'period': '各股自上市~2026-08-30', 'daily': '—', 'trades': '—', 'winrate': '—', 'mdd': '—',
             'stability': '系统性无效', 'market': '主网合约', 'source': '美股七姐妹扫描（2026-08）',
             'note': '❌禁用做空：样本期七姐妹整体上涨，做空逆势全灭（KDJ双向5/5为0/7）；蓝筹做空逻辑同ETH禁用'},
        ]
        for rec in m7_rows:
            exist = c.execute("SELECT id FROM strategy_records WHERE symbol=? AND strategy=? AND source=?",
                              (rec['symbol'], rec['strategy'], rec['source'])).fetchone()
            if exist:
                sets = ", ".join(f"{f} = ?" for f in _store.FIELDS)
                c.execute(f"UPDATE strategy_records SET {sets} WHERE id = ?",
                          (*[rec.get(f) for f in _store.FIELDS], exist[0]))
            else:
                c.execute("INSERT INTO strategy_records (is_top, sort_order, created_at, " + ",".join(_store.FIELDS) + ") "
                          "VALUES (0, 0, ?, " + ",".join("?" * len(_store.FIELDS)) + ")",
                          (now2, *[rec.get(f) for f in _store.FIELDS]))
        # ---- ⑥ 品种名规范化：汇总/警示行的伪标题改为真实标的范围名 ----
        _RENAME = {'组合策略结论': 'DOGE~XLM等10山寨', '普适性冠军': 'DOGE~XLM等10山寨',
                   '山寨币': 'DOGE~XLM等10山寨', 'ETH等蓝筹': 'ETH/BTC等蓝筹',
                   '七姐妹普适冠军': 'AAPL~TSLA等7币', '七姐妹禁用': 'AAPL~TSLA等7币'}
        for old, new in _RENAME.items():
            c.execute("UPDATE strategy_records SET symbol = ? WHERE symbol = ?", (new, old))
        # ---- ⑧ 山寨币去汇总化：删除 DOGE~XLM 范围行，写入逐币记录 ----
        MC_SRC = '山寨10币逐币复验（2026-08）'
        c.execute("DELETE FROM strategy_records WHERE symbol = 'DOGE~XLM等10山寨'")
        # 清理早期版本 note/source 颠倒的错位记录
        c.execute("DELETE FROM strategy_records WHERE note = ? AND source != ?", (MC_SRC, MC_SRC))
        mc_ema = [  # (币, 四年EMA仅多不设收益%, 回撤, 笔数)
            ('DOGE', 90.3, -48.1, 127), ('TRX', 73.0, -40.7, 135), ('LINK', 9.6, -49.2, 145),
            ('DOT', -57.1, -58.3, 128), ('BCH', 207.3, -37.7, 129), ('LTC', -75.1, -65.4, 149),
            ('UNI', -60.1, -51.3, 147), ('APT', -67.5, -59.4, 126), ('ICP', 97.9, -46.2, 139),
            ('XLM', 382.3, -44.3, 121)]
        mc_kdj = [  # (币, 四年KDJ8/5仅多收益%, 回撤, 笔数)
            ('DOGE', -91.0, -67.0, 65), ('TRX', -10.8, -38.5, 84), ('LINK', -51.9, -45.4, 93),
            ('DOT', -84.1, -75.1, 81), ('BCH', -26.4, -66.9, 70), ('LTC', -8.8, -43.0, 76),
            ('UNI', -30.7, -61.6, 91), ('APT', -91.2, -72.6, 85), ('ICP', -95.0, -77.5, 76),
            ('XLM', -48.4, -71.4, 77)]
        now_mc = datetime.now().isoformat(timespec='seconds')
        for coin, ret, mdd, trades in mc_ema:
            sym = f"{coin}/USDT"
            dup = c.execute("SELECT id FROM strategy_records WHERE symbol=? AND strategy=? AND source=?",
                            (sym, 'EMA(12/26)', MC_SRC)).fetchone()
            good = ret > 0
            mc_vals = (sym, '4h', 'EMA(12/26)', '仅做多', '不设',
                       f"{ret:+.1f}%", None, None, str(trades), '—', f"{mdd:.1f}%",
                       '6/10为正' if good else 'EMA失效币', '合约',
                       ('★EMA普适家族正收益成员' if good else 'EMA仅多在其上为负（山寨中4/10失效）'),
                       MC_SRC, SHARPE_DATA.get(f'{coin}_EMA'), '2023-01-01~2026-08-26')
            if dup:  # 已存在则补夏普（保留其他字段与置顶状态）
                if not c.execute("SELECT sharpe FROM strategy_records WHERE id=?", (dup[0],)).fetchone()[0]:
                    c.execute("UPDATE strategy_records SET sharpe=? WHERE id=?",
                              (SHARPE_DATA.get(f'{coin}_EMA'), dup[0]))
            else:
                c.execute("INSERT INTO strategy_records (is_top, sort_order, created_at, " + ",".join(_store.FIELDS) + ") "
                          "VALUES (0, 0, ?, " + ",".join("?" * len(_store.FIELDS)) + ")",
                          (now_mc, *mc_vals))
        for coin, ret, mdd, trades in mc_kdj:
            sym = f"{coin}/USDT"
            dup = c.execute("SELECT id FROM strategy_records WHERE symbol=? AND strategy=? AND source=?",
                            (sym, 'KDJ(9,3,3)', MC_SRC)).fetchone()
            if dup:
                if not c.execute("SELECT sharpe FROM strategy_records WHERE id=?", (dup[0],)).fetchone()[0]:
                    c.execute("UPDATE strategy_records SET sharpe=? WHERE id=?",
                              (SHARPE_DATA.get(f'{coin}_KDJ'), dup[0]))
            else:
                c.execute("INSERT INTO strategy_records (is_top, sort_order, created_at, " + ",".join(_store.FIELDS) + ") "
                          "VALUES (0, 0, ?, " + ",".join("?" * len(_store.FIELDS)) + ")",
                          (now_mc, sym, '4h', 'KDJ(9,3,3)', '仅做多', '8%/5%',
                           f"{ret:+.1f}%", None, None, str(trades), '—', f"{mdd:.1f}%",
                           '0/10为正', '合约', '❌ETH专用策略移植失败（KDJ超卖回归在山寨全灭）', MC_SRC,
                           SHARPE_DATA.get(f'{coin}_KDJ'), '2023-01-01~2026-08-26'))
        # ---- ⑦ 七姐妹去汇总化：删除范围汇总行，分别写入 7 条个股记录 ----
        M7_SRC = '美股七姐妹扫描（2026-08）'
        # 删除汇总行（symbol 不含 '/USDT' 的即范围汇总名；个股均为 X/USDT:USDT）
        c.execute("DELETE FROM strategy_records WHERE source = ? AND symbol NOT LIKE '%/USDT%'", (M7_SRC,))
        # 清理早期版本的错位记录（note/source 字段颠倒的产物）
        c.execute("DELETE FROM strategy_records WHERE note = ? AND source != ?", (M7_SRC, M7_SRC))
        m7_stocks = [
            ('AAPL/USDT:USDT', '双均线', '仅做多', '不设', '+11.2%', '2026-04-06~2026-08-30', '+0.08%', '14', '-10.3%',
             '标的+23.3%：单边牛股策略跑输买入持有（止盈截断趋势）'),
            ('MSFT/USDT:USDT', 'EMA', '双向', '不设', '+24.4%', '2026-04-20~2026-08-30', '+0.19%', '25', '-12.2%',
             '前半+3%/后半+23%稳定；标的+23.4%，策略略胜且回撤更浅'),
            ('NVDA/USDT:USDT', 'RSI+MACD', '仅做多', '不设', '+31.2%', '2026-03-26~2026-08-30', '+0.20%', '29', '-8.9%',
             '★七姐妹最强：前半+20%/后半+8%；标的+25.4%，策略胜出且回撤浅'),
            ('GOOGL/USDT:USDT', '双均线', '仅做多', '不设', '+14.2%', '2026-03-26~2026-08-30', '+0.09%', '16', '-20.2%',
             '前半+22%/后半-6%（单段依赖）；标的+21.5%跑输持有'),
            ('AMZN/USDT:USDT', 'EMA', '仅做多', '不设', '+19.6%', '2026-02-09~2026-08-30', '+0.10%', '22', '-12.4%',
             '前半+17%/后半+2%；标的+27.7%跑输持有；2月9日上市，样本最长（仅TSLA更早）'),
            ('META/USDT:USDT', 'RSI', '仅做多', '5%/5%', '+26.4%', '2026-03-26~2026-08-30', '+0.17%', '8', '-10.0%',
             '横盘股策略大幅超越标的（标的仅+4.2%）；前半+7%/后半+12%'),
            ('TSLA/USDT:USDT', 'EMA', '双向', '5%/5%', '+15.0%', '2026-01-28~2026-08-30', '+0.07%', '42', '-19.7%',
             '标的-18.9%下跌市中双向策略盈利（前半+21%/后半-5%）；七姐妹唯一跌的股'),
        ]
        now_m7 = datetime.now().isoformat(timespec='seconds')
        for sym, strat, mode, tpsl, ret, period, daily, trades, mdd, note in m7_stocks:
            dup = c.execute("SELECT id FROM strategy_records WHERE symbol=? AND strategy=? AND mode=? "
                            "AND tpsl=? AND source=?", (sym, strat, mode, tpsl, M7_SRC)).fetchone()
            base = sym.split('/')[0]
            vals = (sym, '4h', strat, mode, tpsl, ret, daily, None, trades, '—', mdd,
                    '各股上市至8月底', '主网合约', note, M7_SRC, SHARPE_DATA.get(f'{base}_M7'), period)
            if dup:  # 已存在则刷新标签字段（保留置顶/排序），修正早期天数口径
                sets = ", ".join(f"{f} = ?" for f in _store.FIELDS)
                c.execute(f"UPDATE strategy_records SET {sets} WHERE id = ?", (*vals, dup[0]))
            else:  # 同连接内直接插入，避免嵌套连接写锁
                c.execute("INSERT INTO strategy_records (is_top, sort_order, created_at, " + ",".join(_store.FIELDS) + ") "
                          "VALUES (0, 0, ?, " + ",".join("?" * len(_store.FIELDS)) + ")",
                          (now_m7, *vals))
        # ---- ⑬ 交易明细回填：scripts/results/trades_detail.json（按 symbol|strategy|mode|tpsl|timeframe 匹配；
        #      由 scripts/tmp_trades_backfill.py 复跑回测生成，含每笔开/平仓时间、方向、数量、价格、单笔收益）。
        #      放在块末尾：⑦/⑧/实盘口径本轮插入的行也能立即回填 ----
        _td_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'scripts', 'results', 'trades_detail.json')
        if os.path.exists(_td_path):
            try:
                with open(_td_path, encoding='utf-8') as f:
                    _td_map = json.load(f)
            except Exception:
                _td_map = {}
            if _td_map:
                for r in c.execute("SELECT id, symbol, strategy, mode, tpsl, timeframe FROM strategy_records "
                                   "WHERE trades_detail IS NULL OR trades_detail=''").fetchall():
                    v = _td_map.get(f"{r['symbol']}|{r['strategy']}|{r['mode']}|{r['tpsl']}|{r['timeframe']}")
                    if v:
                        c.execute("UPDATE strategy_records SET trades_detail=? WHERE id=?",
                                  (json.dumps(v, ensure_ascii=False), r['id']))
        # ---- ⑭ 胜率回填：山寨/七姐妹/实盘扫描当年未统计胜率（标'—'），现从交易明细真实计算 ----
        for r in c.execute("SELECT id, trades_detail FROM strategy_records "
                           "WHERE (winrate IS NULL OR winrate='' OR winrate='—') "
                           "AND trades_detail IS NOT NULL AND trades_detail!=''").fetchall():
            try:
                trs = (json.loads(r['trades_detail']) or {}).get('trades') or []
                if trs:
                    wins = sum(1 for t in trs if (t.get('ret') or 0) > 0)
                    c.execute("UPDATE strategy_records SET winrate=? WHERE id=?",
                              (f"{wins / len(trs) * 100:.1f}%", r['id']))
            except Exception:
                pass


def _seed_strategy_records():
    """表为空时导入：两条当前最优 + STRATEGY_SUMMARY 全部历史记录（去掉与最优重复的行）"""
    _store.init_tables()
    top_rows = [
        {'symbol': 'ETH/USDT', 'timeframe': '4h', 'strategy': 'KDJ(9,3,3) K上穿D且K<20买 / K下穿D且K>80卖', 'mode': '仅做多',
         'tpsl': '8%/5%', 'ret': '+40.5%', 'period': '2023-01-01~2026-08-26', 'trades': '102', 'winrate': '55.9%', 'mdd': '-38.9%',
         'stability': '最差年仅-7%', 'market': '合约/现货',
         'note': '★主力：唯一无灾难年配置，超卖回归+快进快出；历年 +8.8/+16.1/-7.0/+19.7',
         'source': '分品种最终结论', 'sharpe': SHARPE_DATA.get('ETH_KDJ')},
        {'symbol': 'XLM/USDT', 'timeframe': '4h', 'strategy': 'EMA(12/26)', 'mode': '双向',
         'tpsl': '5%/5%', 'ret': '+239.1%', 'period': '2023-01-01~2026-08-26', 'trades': '234', 'winrate': '42.4%', 'mdd': '-33.1%',
         'stability': '最差年-11.8%', 'market': '合约',
         'note': '★山寨仓：高波动币趋势跟随，熊市靠空头盈利；历年 -11.8/+74.4/+30.4/+69.0',
         'source': '市值10-20山寨币四年扫描', 'sharpe': SHARPE_DATA.get('XLM_OPT')},
    ]
    # 与最优重复的行不重复入库
    def _dup(r):
        return ((r.get('品种') == 'ETH/USDT' and 'KDJ' in str(r.get('策略', '')) and r.get('止盈止损') == '8%/5%'
                 and r.get('模式') == '仅做多')
                or (r.get('品种') == 'XLM/USDT' and 'EMA' in str(r.get('策略', '')) and r.get('止盈止损') == '5%/5%'
                    and r.get('模式') == '双向'))
    key_map = {'品种': 'symbol', '周期': 'timeframe', '策略': 'strategy', '模式': 'mode', '止盈止损': 'tpsl',
               '收益': 'ret', '日均': 'daily', '月均': 'monthly', '交易次数': 'trades', '胜率': 'winrate',
               '最大回撤': 'mdd', '稳定性': 'stability', '市场': 'market', '备注': 'note'}
    rows = []
    for g in STRATEGY_SUMMARY:
        for r in g['rows']:
            if _dup(r):
                continue
            rec = {v: r.get(k) for k, v in key_map.items()}
            rec['source'] = f"{g['name']}（{g.get('window', '')}）"
            rows.append(rec)
    _store.seed_initial(rows, top_rows)


_seed_strategy_records()
_migrate_strategy_records()


@app.route("/strategies", methods=["GET", "POST"])
def strategies_summary():
    """最优策略页：可管理的最优列表 + 可翻页历史测试记录 + 历年矩阵 + 研究时间线"""
    if request.method == "POST":
        action = request.form.get("action")
        rid = request.form.get("rid")
        if action == "promote":
            ok, msg = _store.promote(rid)
        elif action == "demote":
            ok, msg = _store.demote(rid)
        elif action == "move":
            ok, msg = _store.move(rid, request.form.get("dir", "up"))
        elif action == "delete":
            ok, msg = _store.delete(rid)
        else:
            ok, msg = False, '未知操作'
        return redirect(url_for('strategies_summary', _error='' if ok else msg,
                                _message=msg if ok else None,
                                page=request.form.get("page", 1),
                                q=request.form.get("q", ""),
                                cat=request.form.get("cat", "")))

    try:
        page = max(1, int(request.args.get('page', 1)))
    except ValueError:
        page = 1
    q = (request.args.get('q') or '').strip()
    cat = request.args.get('cat') or ''
    if cat not in ('us', 'crypto'):
        cat = ''
    sort = request.args.get('sort') or None
    order = 'asc' if request.args.get('order') == 'asc' else 'desc'
    if sort not in _store.SORTABLE:
        sort = None
    history, total, pages = _store.list_history(page=page, per_page=15, q=q,
                                                sort=sort, order=order, cat=cat)
    for r in history:  # 历年明细 JSON → dict（详情展开行使用）
        try:
            r['yearly'] = json.loads(r['yearly']) if r.get('yearly') else None
        except Exception:
            r['yearly'] = None
        try:
            r['trades_list'] = json.loads(r['trades_detail']) if r.get('trades_detail') else None
        except Exception:
            r['trades_list'] = None
    tops = _store.list_top()
    for t in tops:
        try:
            t['yearly'] = json.loads(t['yearly']) if t.get('yearly') else None
        except Exception:
            t['yearly'] = None
        try:
            t['trades_list'] = json.loads(t['trades_detail']) if t.get('trades_detail') else None
        except Exception:
            t['trades_list'] = None
    return render_template("strategies.html",
                           tops=tops,
                           history=history, total=total, pages=pages, page=page, q=q,
                           sort=sort, order=order, cat=cat,
                           error=request.args.get('_error') or None,
                           message=request.args.get('_message') or None)


@app.route("/futures/api/status")
def futures_status():
    """JSON 接口：返回自动合约交易状态（供前端 AJAX 轮询刷新）"""
    from flask import jsonify
    if not _auto_futures:
        saved = {}
        try:
            if os.path.exists(FUTURES_STATE_FILE):
                with open(FUTURES_STATE_FILE, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
        except Exception:
            saved = {}
        return jsonify({"running": False, "log": ["未启动"], **saved})
    # 实时刷新合约持仓与余额，让前端看到最新未实现盈亏/强平价
    try:
        symbol = _auto_futures.status.get('symbol') or 'BTC/USDT'
        _auto_futures._refresh_real_position(symbol)
    except Exception:
        pass
    return jsonify(_auto_futures.get_status())


@app.route("/demo", methods=["GET", "POST"])
def demo():
    form = request.form
    action = form.get("action", "")
    api_key = form.get("api_key", session.get("demo_api_key", DEFAULT_DEMO_API_KEY))
    api_secret = form.get("api_secret", session.get("demo_api_secret", DEFAULT_DEMO_API_SECRET))
    error = None
    message = None

    # 保存 API 密钥
    if form.get("save_keys"):
        session["demo_api_key"] = api_key
        session["demo_api_secret"] = api_secret
        message = "API 密钥已保存"

    trader = _get_demo_trader_for(api_key, api_secret)
    if not trader or not trader.is_configured():
        return render_template("demo.html", error="请先输入 API 密钥",
                              api_key=api_key, api_secret=api_secret,
                              symbols=DEMO_SYMBOLS, balances=None, open_orders=None,
                              prices={}, portfolio_value=0)

    # 处理操作
    if action == "place_order":
        symbol = form.get("order_symbol", "BTC/USDT")
        side = form.get("side", "buy")
        order_type = form.get("order_type", "market")
        qty_mode = form.get("qty_mode", "amount")  # amount=按数量, usdt=按USDT金额, percent=按百分比
        try:
            qty_input = float(form.get("quantity", 0))
        except (TypeError, ValueError):
            qty_input = 0
        price = None
        if order_type == "limit":
            try:
                price = float(form.get("price", 0))
            except (TypeError, ValueError):
                price = 0

        if qty_input <= 0:
            error = "数量/金额/百分比必须大于 0"
        elif order_type == "limit" and (not price or price <= 0):
            error = "限价单必须填写价格"
        else:
            # 按USDT金额下单：根据当前价格换算成数量
            if qty_mode == "usdt":
                cur_price = trader.get_price(symbol)
                if not cur_price or cur_price <= 0:
                    error = "无法获取当前价格，无法按USDT金额下单"
                else:
                    quantity = qty_input / cur_price
                    message_prefix = f"{side.upper()} {qty_input} USDT ≈ {quantity:.8f} {symbol}"
            elif qty_mode == "percent":
                # 按余额百分比下单
                cur_price = trader.get_price(symbol)
                if not cur_price or cur_price <= 0:
                    error = "无法获取当前价格，无法按百分比下单"
                else:
                    base = symbol.split('/')[0]
                    quote = symbol.split('/')[1]
                    pct = qty_input / 100.0
                    if side == "buy":
                        # 买入：使用选择的计价币（通常是 USDT）余额
                        balances, _ = trader.get_balance()
                        quote_balance = 0.0
                        for b in balances:
                            if b['asset'] == quote:
                                quote_balance = b['free']
                                break
                        if quote_balance <= 0:
                            error = f"余额中没有足够的 {quote} 可用资金"
                        else:
                            amount_usdt = quote_balance * pct
                            quantity = amount_usdt / cur_price
                            message_prefix = f"{side.upper()} {pct*100:.0f}% of {quote} ≈ {quantity:.8f} {symbol}"
                    else:
                        # 卖出：使用持有币种余额
                        balances, _ = trader.get_balance()
                        base_balance = 0.0
                        for b in balances:
                            if b['asset'] == base:
                                base_balance = b['free']
                                break
                        if base_balance <= 0:
                            error = f"余额中没有足够的 {base} 可卖出"
                        else:
                            quantity = base_balance * pct
                            message_prefix = f"{side.upper()} {pct*100:.0f}% of {base} ≈ {quantity:.8f} {symbol}"
            else:
                quantity = qty_input
                message_prefix = f"{side.upper()} {quantity} {symbol}"

            if not error:
                result, err = trader.place_order(symbol, side, order_type, quantity, price)
                if err:
                    error = f"下单失败: {err}"
                else:
                    message = f"下单成功: {message_prefix} ({order_type})"

    elif action == "cancel_order":
        order_id = form.get("order_id", "")
        symbol = form.get("cancel_symbol", "")
        result, err = trader.cancel_order(order_id, symbol)
        if err:
            error = f"撤单失败: {err}"
        else:
            message = f"已撤销订单 {order_id}"

    # 获取余额和挂单
    balances, bal_err = trader.get_balance()
    if bal_err:
        error = error or bal_err

    open_orders, ord_err = trader.get_open_orders()
    if ord_err:
        error = error or ord_err

    # 获取当前价格（含余额中非 USDT 资产的估价）——批量单次请求，避免逐个请求拖慢刷新
    price_syms = list(DEMO_SYMBOLS)
    for b in balances:
        asset = b['asset']
        if asset != 'USDT' and f"{asset}/USDT" not in price_syms:
            price_syms.append(f"{asset}/USDT")
    all_prices = trader.get_tickers(price_syms) or {}

    prices = {sym: all_prices[sym] for sym in DEMO_SYMBOLS if sym in all_prices}

    # 为余额中的非 USDT 资产获取价格并计算 USDT 估值
    portfolio_value = 0.0
    for b in balances:
        asset = b['asset']
        if asset == 'USDT':
            portfolio_value += b['total']
        else:
            sym = f"{asset}/USDT"
            p = all_prices.get(sym)
            if p:
                b['price'] = p
                b['value_usdt'] = b['total'] * p
                portfolio_value += b['value_usdt']
            else:
                b['price'] = None
                b['value_usdt'] = None

    return render_template("demo.html", error=error, message=message,
                          api_key=api_key, api_secret=api_secret,
                          symbols=DEMO_SYMBOLS, balances=balances,
                          open_orders=open_orders, prices=prices,
                          portfolio_value=portfolio_value)


@app.route("/demo/api/prices")
def demo_prices():
    """JSON 接口：返回所有交易对当前价格（供前端 AJAX 刷新）"""
    from flask import jsonify
    api_key = session.get("demo_api_key", DEFAULT_DEMO_API_KEY)
    api_secret = session.get("demo_api_secret", DEFAULT_DEMO_API_SECRET)
    if not api_key or not api_secret:
        return jsonify({"error": "未配置 API 密钥"}), 403

    trader = _get_demo_trader_for(api_key, api_secret)
    if not trader:
        return jsonify({"error": "未配置 API 密钥"}), 403
    # 合并下拉框交易对 + 余额中的资产交易对
    symbols = list(DEMO_SYMBOLS)
    balances, _ = trader.get_balance()
    for b in balances:
        asset = b['asset']
        if asset != 'USDT':
            sym = f"{asset}/USDT"
            if sym not in symbols:
                symbols.append(sym)

    prices = trader.get_tickers(symbols)
    return jsonify({"prices": prices})


@app.route("/auto", methods=["GET", "POST"])
def auto():
    """自动现货交易监控页面"""
    from flask import jsonify
    form = request.form
    # 自动现货使用独立密钥（不与模拟现货共用）；预填充优先级：表单 > auto会话 > 合约会话 > 合约默认密钥
    api_key = form.get("api_key") or session.get("auto_api_key") or session.get("futures_api_key") or DEFAULT_FUTURES_API_KEY
    api_secret = form.get("api_secret") or session.get("auto_api_secret") or session.get("futures_api_secret") or DEFAULT_FUTURES_API_SECRET
    message = None

    # 保存自动现货 API 密钥（PRG，刷新不重放）
    if form.get("save_keys"):
        session["auto_api_key"] = api_key
        session["auto_api_secret"] = api_secret
        return redirect(url_for('auto', _saved='1'))

    if not api_key or not api_secret:
        return render_template("auto.html", error="请输入自动现货 API 密钥",
                              message=None, running=False, status=None, symbols=DEMO_SYMBOLS,
                              timeframe_options=TIMEFRAME_OPTIONS, strategies=STRATEGIES,
                              api_key=api_key, api_secret=api_secret)

    global _auto_trader
    # 复用缓存 DemoTrader，避免每次启动自动交易都重建 ccxt 触发 load_markets 拖慢
    shared_trader = _get_demo_trader_for(api_key, api_secret)
    if not _auto_trader or _auto_trader.api_key != api_key:
        _auto_trader = AutoTrader(api_key, api_secret, trader=shared_trader)

    # 若尚未启动，从磁盘恢复上次的配置与持仓（断网/重启后恢复）
    if not _auto_trader.status.get('running'):
        _auto_trader.load_state()

    # 绑定成功提示：校验密钥可用性并读取 USDT 余额
    error = None
    if request.args.get('_saved') == '1':
        try:
            bal_list, bErr = _auto_trader.trader.get_balance()
            if bErr:
                raise RuntimeError(bErr)
            usdt = 0.0
            for b in bal_list or []:
                if b['asset'] == 'USDT':
                    usdt = float(b['free'])
                    break
            _auto_trader.status['account_balance'] = round(usdt, 2)
            message = f"API 密钥绑定成功！现货账户 USDT 余额：{usdt:.2f} USDT"
        except Exception as e:
            error = f"API 密钥绑定失败，请检查密钥权限：{e}"

    # 启动自动交易（仅 POST；_saved 提示优先展示）
    if request.args.get('_saved') != '1':
        error = request.args.get('_error', '') or None
    if request.method == "POST" and form.get("action") == "start":
        symbol = form.get("symbol", "BTC/USDT")
        timeframe = form.get("timeframe", "15m")
        qty_usdt = _to_float(form.get("qty_usdt"), 1000)
        interval = _to_int(form.get("interval"), 30)
        mode = form.get("mode", "standard")
        if mode == 'grid':
            step_pct = _to_float(form.get("grid_step"), 1) / 100  # 前端为百分数(如1表示1%)，转为小数
            max_levels = _to_int(form.get("grid_max_levels"), 12)
            ok, msg = _auto_trader.start(symbol, timeframe, {}, qty_usdt, interval,
                                         strategies=[], mode=mode, step_pct=step_pct, max_levels=max_levels)
        else:
            # 标准多策略 OR
            selected_strategies = [v for v in STRATEGIES if form.get(f"strategy_{v.lower()}")]
            if not selected_strategies:
                selected_strategies = ["RSI"]
            indicator_params = _build_indicators(form, selected_strategies)
            ok, msg = _auto_trader.start(symbol, timeframe, indicator_params, qty_usdt, interval,
                                         strategies=selected_strategies, mode='standard')
        # PRG：POST 处理完重定向到 GET，避免刷新页面重放 start 导致重复下单
        return redirect(url_for('auto', _error='' if ok else msg))

    elif request.method == "POST" and form.get("action") == "resume_task":
        # 恢复历史现货量化任务（服务崩溃后从任务列表一键续跑）
        tid = form.get("task_id")
        task = next((t for t in _auto_trader.list_tasks() if t.get('id') == tid), None)
        if not task:
            return redirect(url_for('auto', _error=f"任务不存在或已丢失: {tid}"))
        if _auto_trader.status.get('running'):
            return redirect(url_for('auto', _error='已有任务在运行，请先停止再恢复'))
        if task.get('mode') == 'grid':
            ok, msg = _auto_trader.start(task['symbol'], task['timeframe'], {},
                                         float(task.get('qty_usdt', 1000)), int(task.get('interval', 30)),
                                         strategies=[], mode='grid',
                                         step_pct=float(task.get('grid_step', 0.01)),
                                         max_levels=int(task.get('grid_max_levels', 12)))
        else:
            names = task.get('strategies') or ["RSI"]
            ip = _strategy_params_from_names(names)
            ok, msg = _auto_trader.start(task['symbol'], task['timeframe'], ip,
                                         float(task.get('qty_usdt', 1000)), int(task.get('interval', 30)),
                                         strategies=names, mode='standard')
        return redirect(url_for('auto', _error='' if ok else msg))

    elif request.method == "POST" and form.get("action") == "delete_task":
        # 删除现货量化任务记录（仅删历史记录，不影响运行中的引擎）
        ok, msg = _auto_trader.delete_task(form.get("task_id"))
        return redirect(url_for('auto', _error='' if ok else msg))

    elif request.method == "POST" and form.get("action") == "stop":
        _auto_trader.stop()
        return redirect(url_for('auto'))

    status = _auto_trader.get_status()
    return render_template("auto.html", error=error, message=message, running=status['running'],
                          status=status, symbols=DEMO_SYMBOLS,
                          timeframe_options=TIMEFRAME_OPTIONS,
                          strategies=STRATEGIES,
                          tasks=_auto_trader.list_tasks(),
                          api_key=api_key, api_secret=api_secret)


@app.route("/auto/api/status")
def auto_status():
    """JSON 接口：返回自动交易状态（供前端 AJAX 轮询刷新）"""
    from flask import jsonify
    if not _auto_trader:
        # 未启动：读取上次持久化状态，便于断网/重启后恢复展示
        saved = {}
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
        except Exception:
            saved = {}
        return jsonify({"running": False, "log": ["未启动"], **saved})
    # 若已配置密钥，实时从交易所刷新真实持仓与余额，让前端看到最新持仓（即使交易已停止）
    try:
        trader = getattr(_auto_trader, 'trader', None)
        symbol = _auto_trader.status.get('symbol') or 'BTC/USDT'
        if trader and getattr(trader, 'is_configured', lambda: False)():
            _auto_trader._refresh_real_position(symbol)
    except Exception:
        pass
    return jsonify(_auto_trader.get_status())


if __name__ == "__main__":
    # threaded=True：让 Flask 并发处理请求。否则单个长耗时网络请求（如市场概览/信号计算）
    # 会占住唯一工作线程，导致其它请求（状态轮询/启动/停止）全部排队阻塞。
    app.run(debug=True, host="0.0.0.0", port=8502, threaded=True)
