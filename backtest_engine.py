import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, initial_capital=10000, commission=0.001, take_profit=None, stop_loss=None, timeframe='1d', weekly_close=False, signal_mode='and'):
        """
        初始化回测引擎

        Args:
            initial_capital: 初始资金
            commission: 手续费率
            take_profit: 止盈百分比，如 0.1 表示10%
            stop_loss: 止损百分比，如 0.05 表示5%
            timeframe: 时间周期，用于计算夏普比率年化系数
            weekly_close: 是否每周周末（周日24点）强制清仓
            signal_mode: 多策略信号合成方式，'and'=所有策略都满足才触发，'or'=任一策略满足即触发
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.timeframe = timeframe
        self.weekly_close = weekly_close
        self.signal_mode = signal_mode
        self.reset()
    
    def reset(self):
        """重置回测状态"""
        self.capital = self.initial_capital
        self.position = 0
        self.avg_buy_price = 0  # 平均买入价格
        self.trades = []
        self.equity_curve = []
        self.current_price = 0
    
    def calculate_signals(self, df, strategy_params):
        """
        计算交易信号（AND逻辑：所有启用策略都发出买入信号才买入，都发出卖出信号才卖出）

        Args:
            df: 包含技术指标的DataFrame
            strategy_params: 策略参数字典
        """
        # 收集每个策略的买卖信号
        strategy_buys = []
        strategy_sells = []
        active_count = 0

        # RSI策略
        if 'rsi' in strategy_params and 'RSI' in df.columns:
            active_count += 1
            rsi_oversold = strategy_params.get('rsi_oversold', 30)
            rsi_overbought = strategy_params.get('rsi_overbought', 70)
            rsi_buy = (df['RSI'] < rsi_oversold) & (df['RSI'].shift(1) >= rsi_oversold)
            rsi_sell = (df['RSI'] > rsi_overbought) & (df['RSI'].shift(1) <= rsi_overbought)
            strategy_buys.append(rsi_buy)
            strategy_sells.append(rsi_sell)

        # KDJ策略
        if 'kdj' in strategy_params and all(col in df.columns for col in ['K', 'D', 'J']):
            active_count += 1
            kdj_buy_threshold = strategy_params.get('kdj_buy_threshold', 20)
            kdj_sell_threshold = strategy_params.get('kdj_sell_threshold', 80)
            kdj_buy = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1)) & (df['K'] < kdj_buy_threshold)
            kdj_sell = (df['K'] < df['D']) & (df['K'].shift(1) >= df['D'].shift(1)) & (df['K'] > kdj_sell_threshold)
            strategy_buys.append(kdj_buy)
            strategy_sells.append(kdj_sell)

        # 布林带策略
        if 'boll' in strategy_params and all(col in df.columns for col in ['BB_upper', 'BB_middle', 'BB_lower']):
            active_count += 1
            boll_buy = df['close'] <= df['BB_lower']
            boll_sell = df['close'] >= df['BB_upper']
            strategy_buys.append(boll_buy)
            strategy_sells.append(boll_sell)

        # EMA策略
        if 'ema' in strategy_params:
            ema_short = strategy_params.get('ema_short', 12)
            ema_long = strategy_params.get('ema_long', 26)
            if f'EMA_{ema_short}' in df.columns and f'EMA_{ema_long}' in df.columns:
                active_count += 1
                ema_buy = (df[f'EMA_{ema_short}'] > df[f'EMA_{ema_long}']) & \
                         (df[f'EMA_{ema_short}'].shift(1) <= df[f'EMA_{ema_long}'].shift(1))
                ema_sell = (df[f'EMA_{ema_short}'] < df[f'EMA_{ema_long}']) & \
                          (df[f'EMA_{ema_short}'].shift(1) >= df[f'EMA_{ema_long}'].shift(1))
                strategy_buys.append(ema_buy)
                strategy_sells.append(ema_sell)

        # MACD策略
        if 'macd' in strategy_params and all(col in df.columns for col in ['MACD', 'MACD_signal']):
            active_count += 1
            macd_buy = (df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))
            macd_sell = (df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1))
            strategy_buys.append(macd_buy)
            strategy_sells.append(macd_sell)

        # 双均线(MA 金叉)策略：短期均线上穿长期均线买入，下穿卖出
        if 'ma_cross' in strategy_params:
            ma_short = strategy_params.get('ma_cross_short', 10)
            ma_long = strategy_params.get('ma_cross_long', 30)
            if f'SMA_{ma_short}' in df.columns and f'SMA_{ma_long}' in df.columns:
                active_count += 1
                s, l = df[f'SMA_{ma_short}'], df[f'SMA_{ma_long}']
                ma_buy = (s > l) & (s.shift(1) <= l.shift(1))  # 金叉
                ma_sell = (s < l) & (s.shift(1) >= l.shift(1))  # 死叉
                strategy_buys.append(ma_buy)
                strategy_sells.append(ma_sell)

        # 策略信号合成：'and'=所有策略都满足才触发，'or'=任一策略满足即触发
        if active_count == 0:
            return pd.Series(0, index=df.index)

        if self.signal_mode == 'or':
            buy_signals = strategy_buys[0]
            for s in strategy_buys[1:]:
                buy_signals = buy_signals | s
            sell_signals = strategy_sells[0]
            for s in strategy_sells[1:]:
                sell_signals = sell_signals | s
        else:
            buy_signals = strategy_buys[0]
            for s in strategy_buys[1:]:
                buy_signals = buy_signals & s
            sell_signals = strategy_sells[0]
            for s in strategy_sells[1:]:
                sell_signals = sell_signals & s

        signals = pd.Series(0, index=df.index)
        signals[buy_signals] = 1
        signals[sell_signals] = -1

        return signals
    
    def run_backtest(self, df, strategy_params):
        """
        运行回测
        
        Args:
            df: 包含技术指标的DataFrame
            strategy_params: 策略参数字典
        """
        self.reset()
        
        # 计算交易信号
        signals = self.calculate_signals(df, strategy_params)
        
        # 执行回测
        for i, (timestamp, row) in enumerate(df.iterrows()):
            self.current_price = row['close']
            
            # 检查止盈止损
            if self.position > 0 and self.avg_buy_price > 0:
                current_return = (self.current_price - self.avg_buy_price) / self.avg_buy_price

                # 止盈检查
                if self.take_profit and current_return >= self.take_profit:
                    # 触发止盈
                    revenue = self.position * self.current_price * (1 - self.commission)
                    self.capital += revenue

                    self.trades.append({
                        'timestamp': timestamp,
                        'action': 'TAKE_PROFIT',
                        'price': self.current_price,
                        'shares': self.position,
                        'revenue': revenue,
                        'capital': self.capital,
                        'position': 0,
                        'return_pct': current_return * 100
                    })

                    self.position = 0
                    self.avg_buy_price = 0
                    # 更新权益曲线（修复：止盈当天也要记录权益数据点）
                    self.equity_curve.append({
                        'timestamp': timestamp,
                        'equity': self.capital,
                        'capital': self.capital,
                        'position': 0,
                        'price': self.current_price
                    })
                    continue

                # 止损检查
                if self.stop_loss and current_return <= -self.stop_loss:
                    # 触发止损
                    revenue = self.position * self.current_price * (1 - self.commission)
                    self.capital += revenue

                    self.trades.append({
                        'timestamp': timestamp,
                        'action': 'STOP_LOSS',
                        'price': self.current_price,
                        'shares': self.position,
                        'revenue': revenue,
                        'capital': self.capital,
                        'position': 0,
                        'return_pct': current_return * 100
                    })

                    self.position = 0
                    self.avg_buy_price = 0
                    # 更新权益曲线（修复：止损当天也要记录权益数据点）
                    self.equity_curve.append({
                        'timestamp': timestamp,
                        'equity': self.capital,
                        'capital': self.capital,
                        'position': 0,
                        'price': self.current_price
                    })
                    continue
            
            # 更新权益曲线
            current_equity = self.capital + self.position * self.current_price
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': current_equity,
                'capital': self.capital,
                'position': self.position,
                'price': self.current_price
            })
            
            # 处理交易信号
            if i > 0:  # 跳过第一个数据点
                signal = signals.iloc[i]
                
                # 周末强制清仓：在自然周（周一~周日）的最后一根K线（周日23:45，即下一根进入周一或数据结束）平掉所有仓位
                is_week_last_bar = timestamp.weekday() == 6 and (
                    i + 1 >= len(df) or df.index[i + 1].weekday() != 6
                )
                if self.weekly_close and self.position > 0 and is_week_last_bar:
                    revenue = self.position * self.current_price * (1 - self.commission)
                    self.capital += revenue
                    
                    self.trades.append({
                        'timestamp': timestamp,
                        'action': 'WEEKLY_CLOSE',
                        'price': self.current_price,
                        'shares': self.position,
                        'revenue': revenue,
                        'capital': self.capital,
                        'position': 0,
                        'return_pct': (self.current_price - self.avg_buy_price) / self.avg_buy_price * 100 if self.avg_buy_price > 0 else 0
                    })
                    
                    self.position = 0
                    self.avg_buy_price = 0
                    continue
                
                if signal == 1 and self.position == 0:  # 买入信号
                    # 计算可买入数量（支持小数份额，避免高单价币种因向下取整为0而无法买入）
                    available_capital = self.capital * 0.95  # 保留5%现金
                    shares = available_capital / self.current_price
                    
                    if shares > 0:
                        cost = shares * self.current_price * (1 + self.commission)
                        if cost <= self.capital:
                            self.position = shares
                            self.capital -= cost
                            self.avg_buy_price = self.current_price  # 记录买入价格
                            
                            self.trades.append({
                                'timestamp': timestamp,
                                'action': 'BUY',
                                'price': self.current_price,
                                'shares': shares,
                                'cost': cost,
                                'capital': self.capital,
                                'position': self.position
                            })
                
                elif signal == -1 and self.position > 0:  # 卖出信号
                    # 卖出所有持仓
                    revenue = self.position * self.current_price * (1 - self.commission)
                    self.capital += revenue
                    
                    self.trades.append({
                        'timestamp': timestamp,
                        'action': 'SELL',
                        'price': self.current_price,
                        'shares': self.position,
                        'revenue': revenue,
                        'capital': self.capital,
                        'position': 0
                    })
                    
                    self.position = 0
                    self.avg_buy_price = 0
        
        # 最后一天强制平仓
        if self.position > 0:
            revenue = self.position * self.current_price * (1 - self.commission)
            self.capital += revenue
            
            self.trades.append({
                'timestamp': df.index[-1],
                'action': 'SELL',
                'price': self.current_price,
                'shares': self.position,
                'revenue': revenue,
                'capital': self.capital,
                'position': 0
            })
        
        return self.get_results()
    
    def get_results(self):
        """获取回测结果"""
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        # 计算收益率
        initial_equity = self.initial_capital
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - initial_equity) / initial_equity * 100
        
        # 计算年化收益率
        days = (equity_df['timestamp'].iloc[-1] - equity_df['timestamp'].iloc[0]).days
        annual_return = (final_equity / initial_equity) ** (365 / days) - 1 if days > 0 else 0
        
        # 计算最大回撤
        equity_df['peak'] = equity_df['equity'].expanding().max()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        max_drawdown = equity_df['drawdown'].min()
        
        # 计算夏普比率（根据时间周期动态年化）
        equity_df['bar_return'] = equity_df['equity'].pct_change()
        # 每根K线对应的年化系数：crypto 7x24 全年无休
        bars_per_year_map = {'1m': 525600, '5m': 105120, '15m': 35040, '30m': 17520,
                             '1h': 8760, '4h': 2190, '1d': 365}
        bars_per_year = bars_per_year_map.get(self.timeframe, 365)
        sharpe_ratio = equity_df['bar_return'].mean() / equity_df['bar_return'].std() * np.sqrt(bars_per_year) if equity_df['bar_return'].std() > 0 else 0
        
        # 计算胜率和止盈止损统计
        if not trades_df.empty:
            # 计算止盈止损次数
            take_profit_count = len(trades_df[trades_df['action'] == 'TAKE_PROFIT'])
            stop_loss_count = len(trades_df[trades_df['action'] == 'STOP_LOSS'])
            normal_sell_count = len(trades_df[trades_df['action'].isin(['SELL', 'WEEKLY_CLOSE'])])
            
            # 计算每笔交易的收益
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            sell_trades = trades_df[trades_df['action'].isin(['SELL', 'TAKE_PROFIT', 'STOP_LOSS', 'WEEKLY_CLOSE'])]
            
            if len(buy_trades) > 0 and len(sell_trades) > 0:
                trade_returns = []
                for i in range(min(len(buy_trades), len(sell_trades))):
                    buy_price = buy_trades.iloc[i]['price']
                    sell_price = sell_trades.iloc[i]['price']
                    # 扣除买卖手续费后的实际收益率
                    trade_return = (sell_price * (1 - self.commission) - buy_price * (1 + self.commission)) / (buy_price * (1 + self.commission))
                    trade_returns.append(trade_return)
                
                win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) * 100 if trade_returns else 0
            else:
                win_rate = 0
        else:
            win_rate = 0
            take_profit_count = 0
            stop_loss_count = 0
            normal_sell_count = 0
        
        return {
            'initial_capital': initial_equity,
            'final_equity': final_equity,
            'total_return': total_return,
            'annual_return': annual_return * 100,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'total_trades': len(trades_df) // 2 if trades_df is not None else 0,
            'take_profit_count': take_profit_count,
            'stop_loss_count': stop_loss_count,
            'normal_sell_count': normal_sell_count,
            'equity_curve': equity_df,
            'trades': trades_df
        }
    
    def get_performance_metrics(self):
        """获取性能指标"""
        results = self.get_results()
        
        metrics = {
            '总收益率': f"{results.get('total_return', 0):.2f}%",
            '年化收益率': f"{results.get('annual_return', 0):.2f}%",
            '最大回撤': f"{results.get('max_drawdown', 0):.2f}%",
            '夏普比率': f"{results.get('sharpe_ratio', 0):.2f}",
            '胜率': f"{results.get('win_rate', 0):.2f}%",
            '总交易次数': results.get('total_trades', 0),
            '止盈次数': results.get('take_profit_count', 0),
            '止损次数': results.get('stop_loss_count', 0),
            '正常卖出次数': results.get('normal_sell_count', 0),
            '初始资金': f"${results.get('initial_capital', 0):,.2f}",
            '最终资金': f"${results.get('final_equity', 0):,.2f}"
        }
        
        return metrics
