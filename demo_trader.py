import ccxt
import os
import time

class DemoTrader:
    """币安模拟现货交易引擎（Demo Mode: demo-api.binance.com）"""

    def __init__(self, api_key='', api_secret='', proxy=None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.proxy = proxy
        self._create_exchange()

    def _create_exchange(self):
        """创建指向币安 Demo Mode 的 ccxt 实例"""
        config = {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'timeout': 15000,  # 毫秒；避免代理/VPN 慢时无限阻塞请求线程
            'options': {
                'defaultType': 'spot',
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

        exchange = ccxt.binance(config)
        # 覆盖为 Demo Mode URL（base 已含 /api/v3，不可遗漏 /v3）
        exchange.urls['api']['public'] = 'https://demo-api.binance.com/api/v3'
        exchange.urls['api']['private'] = 'https://demo-api.binance.com/api/v3'
        exchange.urls['api']['v1'] = 'https://demo-api.binance.com/api/v1'
        # Demo Mode 不支持 sapi，强制用 /api/v3/account 查余额
        exchange.options['fetchBalance'] = {'type': 'spot'}
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

    def get_balance(self):
        """获取账户余额（显示所有有余额的资产，附带 USDT 估值）"""
        try:
            balance = self.exchange.fetch_balance()
            balances = []
            # 优先用 total 字典，回退到 balances 列表
            total_map = balance.get('total', {})
            if not total_map and 'info' in balance:
                # Demo Mode 可能返回原始 balances 列表
                for item in balance['info'].get('balances', []):
                    asset = item.get('asset', '')
                    free = float(item.get('free', 0))
                    locked = float(item.get('locked', 0))
                    total = free + locked
                    if total > 0:
                        total_map[asset] = total
                        if 'free' not in balance:
                            balance['free'] = {}
                        if 'used' not in balance:
                            balance['used'] = {}
                        balance['free'][asset] = free
                        balance['used'][asset] = locked

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
            return [], str(e)

    def place_order(self, symbol, side, order_type, quantity, price=None):
        """
        下单

        Args:
            symbol: 交易对，如 BTC/USDT
            side: 'buy' 或 'sell'
            order_type: 'market' 或 'limit'
            quantity: 数量
            price: 限价单价格（市价单忽略）
        """
        try:
            params = {}
            if order_type == 'limit' and price:
                order = self.exchange.create_order(symbol, 'limit', side, quantity, price, params)
            else:
                order = self.exchange.create_order(symbol, 'market', side, quantity, None, params)
            return order, None
        except Exception as e:
            return None, str(e)

    def cancel_order(self, order_id, symbol):
        """撤单"""
        try:
            result = self.exchange.cancel_order(order_id, symbol)
            return result, None
        except Exception as e:
            return None, str(e)

    def get_open_orders(self, symbol=None):
        """获取挂单"""
        try:
            orders = self.exchange.fetch_open_orders(symbol) if symbol else self.exchange.fetch_open_orders()
            return orders, None
        except Exception as e:
            return [], str(e)

    def get_ticker(self, symbol):
        """获取当前价格"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last'], None
        except Exception as e:
            return None, str(e)

    def get_tickers(self, symbols):
        """批量获取多个交易对的当前价格（单次批量请求，避免逐个请求拖慢刷新）"""
        try:
            # 一次请求获取全部（binance 返回所有交易对，这里只取需要的）
            tickers = self.exchange.fetch_tickers(symbols)
            prices = {}
            for sym in symbols:
                t = tickers.get(sym) or {}
                if t.get('last'):
                    prices[sym] = t['last']
            if prices:
                return prices
        except Exception:
            pass
        # 回退：逐个请求
        prices = {}
        for sym in symbols:
            try:
                ticker = self.exchange.fetch_ticker(sym)
                prices[sym] = ticker['last']
            except Exception:
                pass
        return prices

    def get_price(self, symbol):
        """获取当前价格（get_ticker 的别名，方便快速调用）"""
        price, _ = self.get_ticker(symbol)
        return price

    def get_ohlcv(self, symbol, timeframe='15m', limit=100):
        """获取最近 limit 根**已收盘**K线（用于计算技术指标生成实时信号）

        币安默认返回包含当前进行中的未收盘K线；未收盘K线的指标值会随现价跳动，
        导致信号在同一根K线内反复翻转（与回测收盘口径不一致），必须丢弃。
        """
        try:
            candles = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not candles:
                return None, "无K线数据"
            tf_ms = {'1m': 60_000, '3m': 180_000, '5m': 300_000, '15m': 900_000,
                     '30m': 1_800_000, '1h': 3_600_000, '2h': 7_200_000,
                     '4h': 14_400_000, '6h': 21_600_000, '12h': 43_200_000,
                     '1d': 86_400_000}.get(timeframe, 0)
            if tf_ms and candles[-1][0] + tf_ms > int(time.time() * 1000):
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
            return data, None
        except Exception as e:
            return None, str(e)

    def get_market_info(self, symbol):
        """获取交易对信息（价格精度、最小下单量）"""
        try:
            market = self.exchange.market(symbol)
            return {
                'precision': market.get('precision', {}),
                'limits': market.get('limits', {}),
            }, None
        except Exception as e:
            return None, str(e)

    def get_symbol_price_precision(self, symbol):
        """获取价格精度（用于下单价格取整）"""
        info, _ = self.get_market_info(symbol)
        if info and info.get('precision'):
            return info['precision'].get('price', 8)
        return 8

    def get_symbol_amount_precision(self, symbol):
        """获取数量精度（用于下单数量取整，防止卖出时数量超出实际持仓）"""
        try:
            market = self.exchange.market(symbol)
            return market.get('precision', {}).get('amount', 8)
        except Exception:
            return 8

    def round_amount(self, symbol, amount):
        """按交易对数量精度向下取整到有效位数（避免超出交易所允许的最小步长/实际持仓）"""
        prec = self.get_symbol_amount_precision(symbol)
        if prec is None:
            return amount
        return round(amount, int(prec))
