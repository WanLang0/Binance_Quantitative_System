# -*- coding: utf-8 -*-
"""
币安合约交易连接器（USDT 永续测试网 testnet.binancefuture.com）
现货 Demo(demo-api.binance.com) 不支持合约，必须用合约测试网/真实合约接口。
与 DemoTrader 保持同样的方法签名，便于 AutoTrader 复用。

合约与现货关键差异：
- 交易对象是【张数 contracts】，按以结算货币(USDT)报价的多少张下单
- 需单独设置杠杆(set_leverage)，区分持仓方向 long/short
- 需查询未实现盈亏、强平价、持仓量、保证金模式
"""
import ccxt
import os
import re
import time


class FuturesTrader:
    """币安 USDT 永续合约交易引擎（默认测试网 testnet，可切换真实合约）"""

    def __init__(self, api_key='', api_secret='', proxy=None, testnet=True, leverage=1):
        self.api_key = api_key
        self.api_secret = api_secret
        self.proxy = proxy
        self.testnet = testnet
        self.leverage = leverage
        # closed-candle K线缓存：同一币对/周期在「新一根K线收盘」前不复用重复拉取，
        # 极大降低每秒 REST 请求量，规避币安测试网 -1003 IP 封禁（Way too many requests）
        self._ohlcv_cache = {}
        # IP 封禁截止时间戳（毫秒）：命中 -1003 后记录，主循环据此暂停请求等待到期，
        # 避免封禁期内持续重试把 banned until 不断后移、导致封禁永不结束
        self._ip_ban_until_ms = 0
        self._create_exchange()

    def _create_exchange(self):
        """创建指向币安合约（USDT永续）的 ccxt 实例"""
        config = {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'timeout': 15000,  # 毫秒；避免代理/VPN 慢时无限阻塞请求线程
            'options': {
                'defaultType': 'future',   # 关键：USDT永续
                'adjustForTimeDifference': True,
            },
        }
        # 代理
        if self.proxy:
            config['proxies'] = {'http': self.proxy, 'https': self.proxy}
        else:
            env_http = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
            env_https = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
            if env_http or env_https:
                config['proxies'] = {
                    'http': env_http or env_https,
                    'https': env_https or env_http,
                }

        exchange = ccxt.binanceusdm(config)
        # 测试网：用 ccxt 官方 set_sandbox_mode 切换，
        # 会把实际下单/私有接口地址 fapiPublic/fapiPrivate 正确指向 testnet.binancefuture.com。
        # （手动覆盖 urls['api']['private'] 无效，因为 binanceusdm 走的是 fapiPrivate，默认是生产 fapi.binance.com）
        if self.testnet:
            exchange.set_sandbox_mode(True)
        else:
            # 真实 USDT 永续（fapi），binanceusdm 默认即指向真实
            pass

        exchange.options['fetchBalance'] = {'type': 'future'}
        exchange.options['fetchCurrencies'] = False
        exchange.options['warnOnFetchOpenOrdersWithoutSymbol'] = False

        self.exchange = exchange

    def update_keys(self, api_key, api_secret):
        """更新 API 密钥并重建连接"""
        self.api_key = api_key
        self.api_secret = api_secret
        self._create_exchange()

    def is_configured(self):
        """是否已配置 API 密钥"""
        return bool(self.api_key and self.api_secret)

    def set_leverage(self, leverage, symbol):
        """设置合约杠杆（默认单向持仓模式）"""
        self.leverage = leverage
        try:
            self.exchange.set_leverage(leverage, symbol)
            return True, None
        except Exception as e:
            return False, str(e)

    def _is_ip_ban(self, e):
        """判断异常是否为币安 -1003 IP 封禁（Way too many requests），并记录封禁截止时间"""
        try:
            msg = str(e)
        except Exception:
            return False
        if '-1003' not in msg:
            return False
        # 解析 banned until <epoch_ms>，用于精确等待封禁到期
        m = re.search(r'banned\s+until\s+(\d+)', msg, re.IGNORECASE)
        if m:
            try:
                until = int(m.group(1))
                self._ip_ban_until_ms = max(getattr(self, '_ip_ban_until_ms', 0), until)
            except (TypeError, ValueError):
                pass
        return True

    def wait_ip_ban(self):
        """若当前已被 -1003 封禁且未到期，阻塞等待到期（带 1s 粒度），避免在封禁期内继续打请求。"""
        if getattr(self, '_ip_ban_until_ms', 0) <= time.time() * 1000:
            return
        remain = (self._ip_ban_until_ms - time.time() * 1000) / 1000.0
        # 最多等待到封禁截止；防止异常时间戳导致无限等待
        if remain <= 0 or remain > 3600:
            self._ip_ban_until_ms = 0
            return
        time.sleep(min(remain, 3600))

    def _is_banned_now(self):
        """非阻塞判断当前是否处于 IP 封禁期内。封禁期内直接短路，不再发起网络请求，
        避免同一轮循环里后续币对继续打请求把 banned until 不断后移。"""
        return getattr(self, '_ip_ban_until_ms', 0) > time.time() * 1000

    def fetch_ticker(self, symbol):
        if self._is_banned_now():
            return None, f"IP被封禁中({symbol})"
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last'], None
        except Exception as e:
            if self._is_ip_ban(e):
                return None, f"IP被限流({symbol})"
            return None, str(e)

    def get_ohlcv(self, symbol, timeframe='15m', limit=100):
        """获取最近 limit 根**已收盘**K线（合约）。丢弃未收盘的进行中K线，
        避免指标随现价跳动导致信号在K线内反复翻转（与回测收盘口径不一致）。

        已收盘K线缓存：在「下一根K线收盘」前直接复用上次结果，不再重复请求币安。
        这显著降低每秒 REST 请求量（19 币对 × 多任务从"每轮都拉"变为"每K线收盘拉一次"），
        是规避测试网 -1003 IP 封禁的关键。K线收盘后缓存自动失效并重新拉取。"""
        tf_ms = {'1m': 60_000, '3m': 180_000, '5m': 300_000, '15m': 900_000,
                 '30m': 1_800_000, '1h': 3_600_000, '2h': 7_200_000,
                 '4h': 14_400_000, '6h': 21_600_000, '12h': 43_200_000,
                 '1d': 86_400_000}.get(timeframe)
        # 封禁期内直接短路（本次仍可用缓存数据做信号，不再发起网络请求延长封禁）
        if self._is_banned_now():
            cache = getattr(self, '_ohlcv_cache', {}).get((symbol, timeframe, limit)) if tf_ms else None
            if cache:
                return cache[1], None
            return None, f"IP被封禁中({symbol})"
        now_ms = int(time.time() * 1000)
        key = (symbol, timeframe, limit)
        cache = getattr(self, '_ohlcv_cache', {}).get(key) if tf_ms else None
        if cache:
            last_close_ms, data = cache[0], cache[1]
            # 缓存 = 最近一根已收盘K线。其开盘点为 last_close_ms，在 now < last_close_ms + 2*tf_ms 前
            # 不会出现"更新的已收盘K线"，直接复用；否则说明新一根K线已收盘，刷新。timing 与回测收盘口径完全一致。
            if now_ms < last_close_ms + 2 * tf_ms:
                return data, None
        try:
            candles = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not candles:
                return None, "无K线数据"
            if tf_ms and candles[-1][0] + tf_ms > now_ms:
                candles = candles[:-1]  # 丢弃未收盘的进行中K线
            if not candles:
                return None, "无K线数据"
            data = [
                {
                    'ts': int(c[0]),
                    'open': float(c[1]),
                    'high': float(c[2]),
                    'low': float(c[3]),
                    'close': float(c[4]),
                    'volume': float(c[5]),
                }
                for c in candles
            ]
            if tf_ms:
                # 以「最新已收盘K线的开盘点」为缓存有效期依据
                cache_map = getattr(self, '_ohlcv_cache', None)
                if cache_map is None:
                    cache_map = {}
                    self._ohlcv_cache = cache_map
                cache_map[key] = (int(data[-1]['ts']), data)
            return data, None
        except Exception as e:
            if self._is_ip_ban(e):
                return None, f"IP被限流({symbol})"
            return None, str(e)

    def get_balance(self):
        """获取账户余额（USDT 可用/持仓，及保证金模式）"""
        if self._is_banned_now():
            return [], f"IP被封禁中(get_balance)"
        try:
            balance = self.exchange.fetch_balance()
            balances = []
            total_map = balance.get('total', {})
            for asset, info in total_map.items():
                try:
                    total_val = float(info) if info else 0
                except (TypeError, ValueError):
                    total_val = 0
                if total_val > 0:
                    free = balance.get('free', {}).get(asset, 0)
                    used = balance.get('used', {}).get(asset, 0)
                    balances.append({
                        'asset': asset,
                        'total': total_val,
                        'free': float(free) if free else 0,
                        'used': float(used) if used else 0,
                    })
            return balances, None
        except Exception as e:
            if self._is_ip_ban(e):
                return [], f"IP被限流(get_balance)"
            return [], str(e)

    def get_positions(self, symbol=None):
        """获取合约持仓（含方向、张数、开仓均价、未实现盈亏、强平价、保证金）

        单向持仓(one-way)模式下 ccxt 返回空单 contracts 为负数，此处统一归一化为
        正数张数并兜底推导方向——否则空单会被当"无持仓"过滤，引发引擎误判
        "已被外部平仓"进而循环开空、仓位无限累积（-2019 保证金不足事故）。
        """
        if self._is_banned_now():
            return [], f"IP被封禁中(get_positions)"
        try:
            positions = self.exchange.fetch_positions(symbol) if symbol else self.exchange.fetch_positions()
            result = []
            for p in positions:
                raw = p.get('contracts')
                if raw is None:
                    continue
                contracts = float(raw)
                if contracts == 0:
                    continue
                side = p.get('side') or ('short' if contracts < 0 else 'long')
                contracts = abs(contracts)
                unified = p.get('symbol') or ''
                # 兼容键：ccxt 统一符号为 BTC/USDT:USDT，而综合量化/自动合约的币种配置
                # 可能是不带后缀的 BTC/USDT —— 两个键都返回，供下游匹配
                result.append({
                    'symbol': unified,
                    'symbol_base': unified.rsplit(':', 1)[0] if ':' in unified else unified,
                    'side': side,
                    'contracts': contracts,
                    'contract_size': p.get('contractSize', 1),
                    'entry_price': p.get('entryPrice'),
                    'mark_price': p.get('markPrice'),
                    'notional': p.get('notional'),
                    'unrealized_pnl': p.get('unrealizedPnl'),
                    'liquidation_price': p.get('liquidationPrice'),
                    'leverage': p.get('leverage'),
                    'margin': p.get('initialMargin'),
                })
            return result, None
        except Exception as e:
            if self._is_ip_ban(e):
                return [], f"IP被限流(get_positions)"
            return [], str(e)

    def place_order(self, symbol, side, order_type, quantity, price=None, reduce_only=False):
        """
        下单（合约：quantity 为张数）

        Args:
            symbol: 交易对，如 BTC/USDT
            side: 'buy' 或 'sell'
            order_type: 'market' 或 'limit'
            quantity: 张数（contracts）
            price: 限价单价格（市价单忽略）
            reduce_only: 是否只减仓（平仓用，防止反手开新仓）
        """
        try:
            params = {}
            if reduce_only:
                params['reduceOnly'] = True
            if order_type == 'limit' and price:
                order = self.exchange.create_order(symbol, 'limit', side, quantity, price, params)
            else:
                order = self.exchange.create_order(symbol, 'market', side, quantity, None, params)
            return order, None
        except Exception as e:
            # 美股代币永续：若提示未签署 TradFi 协议(-4411)，自动补签一次并重试，避免首单失败
            msg = str(e)
            if '-4411' in msg:
                try:
                    self.sign_tradfi_agreement()
                    if order_type == 'limit' and price:
                        order = self.exchange.create_order(symbol, 'limit', side, quantity, price, params)
                    else:
                        order = self.exchange.create_order(symbol, 'market', side, quantity, None, params)
                    return order, None
                except Exception as e2:
                    return None, str(e2)
            return None, msg

    def fetch_order(self, order_id, symbol):
        """查询单笔订单的成交回报（实际成交均价/数量）"""
        try:
            od = self.exchange.fetch_order(order_id, symbol)
            return od, None
        except Exception as e:
            return None, str(e)

    def cancel_order(self, order_id, symbol):
        try:
            result = self.exchange.cancel_order(order_id, symbol)
            return result, None
        except Exception as e:
            return None, str(e)

    def place_stop_order(self, symbol, side, stop_price, quantity):
        """
        下止损保护单：STOP_MARKET + reduceOnly（只减仓）。
        挂在交易所侧，即使本服务宕机，价格触发后交易所自动平仓兜底。

        Args:
            symbol: 交易对
            side: 触发后方向（平多='sell'，平空='buy'）
            stop_price: 触发价
            quantity: 张数（与持仓一致）
        """
        try:
            params = {
                'stopPrice': self.exchange.price_to_precision(symbol, stop_price),
                'reduceOnly': True,
                'type': 'STOP_MARKET',
            }
            order = self.exchange.create_order(symbol, 'market', side, quantity, None, params)
            return order, None
        except Exception as e:
            return None, str(e)

    def cancel_all_orders(self, symbol):
        """撤销该交易对全部挂单（用于平仓后清理止损保护单）"""
        try:
            result = self.exchange.cancel_all_orders(symbol)
            return result, None
        except Exception as e:
            return None, str(e)

    def get_open_orders(self, symbol=None):
        try:
            orders = self.exchange.fetch_open_orders(symbol) if symbol else self.exchange.fetch_open_orders()
            return orders, None
        except Exception as e:
            return [], str(e)

    def get_ticker(self, symbol):
        """获取当前价格（get_ticker 别名）"""
        return self.fetch_ticker(symbol)

    def get_tickers(self, symbols):
        """批量获取多个交易对当前价格"""
        if self._is_banned_now():
            return {}
        try:
            tickers = self.exchange.fetch_tickers(symbols)
            prices = {}
            for sym in symbols:
                t = tickers.get(sym) or {}
                if t.get('last'):
                    prices[sym] = t['last']
            if prices:
                return prices
        except Exception as e:
            if self._is_ip_ban(e):
                return {}
        prices = {}
        for sym in symbols:
            try:
                ticker = self.exchange.fetch_ticker(sym)
                prices[sym] = ticker['last']
            except Exception:
                pass
        return prices

    def get_price(self, symbol):
        price, _ = self.get_ticker(symbol)
        return price

    def get_market_info(self, symbol):
        """获取交易对信息（价格精度、数量精度、最小张数）"""
        try:
            market = self.exchange.market(symbol)
            return {
                'precision': market.get('precision', {}),
                'limits': market.get('limits', {}),
            }, None
        except Exception as e:
            return None, str(e)

    def get_symbol_amount_precision(self, symbol):
        """获取数量精度（张数 precision）"""
        try:
            market = self.exchange.market(symbol)
            return market.get('precision', {}).get('amount', 0)
        except Exception:
            return 0

    def get_symbol_price_precision(self, symbol):
        try:
            market = self.exchange.market(symbol)
            return market.get('precision', {}).get('price', 8)
        except Exception:
            return 8

    def round_amount(self, symbol, amount):
        """按合约张数精度取整（合约张数为整数或0.1步进）"""
        prec = self.get_symbol_amount_precision(symbol)
        try:
            prec_int = int(prec)
        except (TypeError, ValueError):
            prec_int = 0
        return round(amount, prec_int)

    def sign_tradfi_agreement(self):
        """签署币安 TradFi-Perps（美股/商品等传统金融永续）协议，避免下单报 -4411。

        一次性操作，对普通加密货币永续无影响。
        返回 (响应, 是否可交易)。bool 表示「该校验可放行」：
          已签署/签署成功 → True；网络失败暂时无法判断 → True（不阻断，交由下单时兜底补签）；
          明确拒绝（如无权限）→ False。
        """
        import hmac
        import hashlib
        import urllib.request
        import urllib.error
        base = 'https://testnet.binancefuture.com' if self.testnet else 'https://fapi.binance.com'
        endpoint = '/fapi/v1/stock/contract'
        timestamp = int(time.time() * 1000)
        qs = f'timestamp={timestamp}'
        signature = hmac.new(self.api_secret.encode('utf-8'), qs.encode('utf-8'),
                             hashlib.sha256).hexdigest()
        url = f'{base}{endpoint}?{qs}&signature={signature}'
        req = urllib.request.Request(url, data=b'', method='POST')
        req.add_header('X-MBX-APIKEY', self.api_key)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode('utf-8'), True
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8', 'ignore')
            msg = (body or str(e)).lower()
            # 已签署过视为放行（币安对重复签署返回 -4099 / "already signed" / "user has signed" 等）
            if '-4099' in msg or 'already signed' in msg or 'user has signed' in msg or 'already_sign' in msg:
                return body, True
            # 时间戳偏移(-1021)属临时时钟问题，非权限拒绝，不阻断启动
            if '-1021' in msg:
                return body, True
            if e.code in (400, 401, 403):
                # 权限/鉴权问题：明确不可交易（如未开通合约权限、IP 未白名单）
                return body, False
            # 其余网络/服务错误：暂时无法判断，不阻断（下单时 -4411 会自动补签兜底）
            return body, True
        except Exception as e:
            return None, True
