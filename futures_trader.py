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


class FuturesTrader:
    """币安 USDT 永续合约交易引擎（默认测试网 testnet，可切换真实合约）"""

    def __init__(self, api_key='', api_secret='', proxy=None, testnet=True, leverage=5):
        self.api_key = api_key
        self.api_secret = api_secret
        self.proxy = proxy
        self.testnet = testnet
        self.leverage = leverage
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

    def fetch_ticker(self, symbol):
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last'], None
        except Exception as e:
            return None, str(e)

    def get_ohlcv(self, symbol, timeframe='15m', limit=100):
        """获取最近 limit 根K线（合约）"""
        try:
            candles = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
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

    def get_balance(self):
        """获取账户余额（USDT 可用/持仓，及保证金模式）"""
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
            return [], str(e)

    def get_positions(self, symbol=None):
        """获取合约持仓（含方向、张数、开仓均价、未实现盈亏、强平价、保证金）"""
        try:
            positions = self.exchange.fetch_positions(symbol) if symbol else self.exchange.fetch_positions()
            result = []
            for p in positions:
                contracts = float(p.get('contracts') or p.get('contractSize') or 0)
                if contracts <= 0:
                    continue
                side = p.get('side')  # long / short
                result.append({
                    'symbol': p.get('symbol'),
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
        try:
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
