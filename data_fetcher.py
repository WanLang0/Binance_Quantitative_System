import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import time
import os

class BinanceDataFetcher:
    """币安数据获取器"""
    
    def __init__(self, proxy=None, market_type='spot'):
        """
        初始化数据获取器
        
        Args:
            proxy: 代理地址，例如 'http://127.0.0.1:7890' 
                   如果为None，则从环境变量读取 HTTP_PROXY/HTTPS_PROXY
            market_type: 市场类型，'spot' 表示现货，'future' 表示合约（USDT永续）
        """
        self.market_type = market_type
        self.proxy = proxy
        self._create_exchange(proxy)
    
    def _create_exchange(self, proxy=None):
        """创建（或重建）ccxt 交易所实例，支持现货/合约"""
        config = {
            'enableRateLimit': True,
            'timeout': 15000,  # 毫秒；避免代理/VPN 慢时无限阻塞页面请求线程
            'options': {
                'defaultType': self.market_type,
                'adjustForTimeDifference': True,
            }
        }
        
        if proxy:
            config['proxies'] = {
                'http': proxy,
                'https': proxy
            }
        else:
            env_http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
            env_https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
            if env_http_proxy or env_https_proxy:
                config['proxies'] = {
                    'http': env_http_proxy or env_https_proxy,
                    'https': env_https_proxy or env_http_proxy
                }
                print(f"使用环境变量代理: HTTP={env_http_proxy}, HTTPS={env_https_proxy}")
        
        self.exchange = ccxt.binance(config)
    
    def set_market_type(self, market_type):
        """切换市场类型（'spot' 现货 / 'future' 合约），只在变化时重建连接"""
        if market_type not in ('spot', 'future'):
            market_type = 'spot'
        if market_type != self.market_type:
            self.market_type = market_type
            self._create_exchange(self.proxy)
            print(f"已切换到 {'合约' if market_type == 'future' else '现货'} 市场")
    
    def get_available_symbols(self):
        """获取可用的USDT永续交易对（现货为 BTC/USDT 格式；合约只保留USDT永续，剔除带到期日的合约）"""
        try:
            markets = self.exchange.load_markets()
            if self.market_type == 'future':
                # 只保留 USDT 永续（形如 BTC/USDT:USDT），剔除带到期日的合约
                usdt_pairs = [symbol for symbol in markets.keys() if symbol.endswith('/USDT:USDT')]
            else:
                usdt_pairs = [symbol for symbol in markets.keys() if symbol.endswith('/USDT')]
            return sorted(usdt_pairs)
        except Exception as e:
            print(f"获取交易对失败: {e}")
            return []
    
    def fetch_ohlcv(self, symbol, timeframe='1d', limit=1000, since=None):
        """
        获取K线数据
        
        Args:
            symbol: 交易对，如 'BTC/USDT'
            timeframe: 时间周期，如 '1m', '5m', '1h', '1d'
            limit: 获取数量
            since: 开始时间戳
        """
        try:
            # 转换时间格式
            if since:
                if isinstance(since, str):
                    since = int(datetime.strptime(since, '%Y-%m-%d').timestamp() * 1000)
                elif isinstance(since, datetime):
                    since = int(since.timestamp() * 1000)
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            
            # 转换为DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"获取数据失败: {e}")
            return pd.DataFrame()
    
    def fetch_historical_data(self, symbol, start_date, end_date, timeframe='1d'):
        """
        获取历史数据
        
        Args:
            symbol: 交易对
            start_date: 开始日期
            end_date: 结束日期
            timeframe: 时间周期
        """
        try:
            # 转换日期格式
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            elif isinstance(start_date, date):
                start_date = datetime.combine(start_date, datetime.min.time())
            elif not isinstance(start_date, datetime):
                start_date = datetime.strptime(str(start_date), '%Y-%m-%d')
                
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
            elif isinstance(end_date, date):
                end_date = datetime.combine(end_date, datetime.min.time())
            elif not isinstance(end_date, datetime):
                end_date = datetime.strptime(str(end_date), '%Y-%m-%d')

            # 结束日期加一天，确保包含完整最后一天的数据
            end_date = end_date + timedelta(days=1)
            
            # 计算时间戳
            since = int(start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000)
            
            all_data = []
            current_since = since
            
            while current_since < end_timestamp:
                # 获取数据
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, current_since, 1000)
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                
                # 更新时间戳
                current_since = ohlcv[-1][0] + 1
                
                # 避免请求过于频繁
                time.sleep(0.1)
            
            if not all_data:
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 过滤日期范围
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            return df
            
        except Exception as e:
            print(f"获取历史数据失败: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol):
        """获取当前价格"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            print(f"获取当前价格失败: {e}")
            return None

    def get_tickers_prices(self, symbols):
        """一次性批量获取多个交易对价格（单次请求，避免逐个请求拖慢页面）。

        Args:
            symbols: 交易对列表，如 ['BTC/USDT', 'ETH/USDT']

        Returns:
            dict: {symbol: 价格}，仅包含成功获取的部分
        """
        try:
            tickers = self.exchange.fetch_tickers(symbols)
            result = {}
            for sym in symbols:
                t = tickers.get(sym) or {}
                if t.get('last'):
                    result[sym] = t['last']
            if result:
                return result
        except Exception as e:
            print(f"批量获取价格失败: {e}")
        return {}
