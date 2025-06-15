import pandas as pd
import numpy as np
from typing import Dict, Any, Callable
from ..evaluation.metrics import sharpe_ratio, sortino_ratio, max_drawdown
from .risk_management import RiskManager

class Backtester:
    """
    Backtesting engine for trading strategies with risk management, advanced order types, and custom execution logic.

    Attributes:
        data (pd.DataFrame): Historical price data.
        strategy_func (Callable): Function implementing the trading strategy.
        stop_loss_pct (float): Stop-loss as a fraction (e.g., 0.02 for 2%).
        take_profit_pct (float): Take-profit as a fraction (e.g., 0.04 for 4%).
        max_risk (float): Max risk per trade as a fraction of balance.
        order_type (str): Order type ('standard', 'trailing_stop', 'oco').
        execution_logic (Callable): Optional custom execution function. If provided, overrides built-in logic.
    """
    def __init__(self, data: pd.DataFrame, strategy_func: Callable, stop_loss_pct=0.02, take_profit_pct=0.04, max_risk=0.01, order_type='standard', execution_logic=None):
        """
        Initialize the Backtester.

        Args:
            data (pd.DataFrame): Historical price data.
            strategy_func (Callable): Function that takes a DataFrame and returns a trading signal.
            stop_loss_pct (float): Stop-loss as a fraction (default 0.02).
            take_profit_pct (float): Take-profit as a fraction (default 0.04).
            max_risk (float): Max risk per trade as a fraction of balance (default 0.01).
            order_type (str): Order type ('standard', 'trailing_stop', 'oco').
            execution_logic (Callable): Optional custom execution function. Should accept (data, entry_idx, entry_price, stop_loss, take_profit, size, order_type) and return exit_price.
        """
        self.data = data
        self.strategy_func = strategy_func
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_risk = max_risk
        self.order_type = order_type
        self.execution_logic = execution_logic

    def run(self) -> Dict[str, Any]:
        """
        Run the backtest and compute performance metrics, including risk management, advanced order types, and custom execution logic.

        Returns:
            dict: Dictionary containing:
                - trades (list): List of trade PnLs.
                - win_rate (float): Proportion of profitable trades.
                - sharpe (float): Sharpe Ratio of returns.
                - sortino (float): Sortino Ratio of returns.
                - max_drawdown (float): Maximum drawdown of returns.
                - final_balance (float): Final account balance after backtest.
        """
        trades = []
        balance = 10000
        equity_curve = [balance]
        risk_manager = RiskManager(balance, max_risk=self.max_risk)
        for i in range(1, len(self.data)):
            signal = self.strategy_func(self.data.iloc[:i])
            if signal == 'buy':
                entry = self.data['close'].iloc[i]
                stop_loss = entry * (1 - self.stop_loss_pct)
                take_profit = entry * (1 + self.take_profit_pct)
                size = risk_manager.position_size(self.stop_loss_pct)
                exit_price = None
                if self.execution_logic is not None:
                    # Custom execution logic API: (data, entry_idx, entry_price, stop_loss, take_profit, size, order_type) -> exit_price
                    exit_price = self.execution_logic(self.data, i, entry, stop_loss, take_profit, size, self.order_type)
                else:
                    if self.order_type == 'trailing_stop':
                        trailing_stop = stop_loss
                        for j in range(i+1, min(i+11, len(self.data))):
                            price = self.data['close'].iloc[j]
                            if price > entry and price * (1 - self.stop_loss_pct) > trailing_stop:
                                trailing_stop = price * (1 - self.stop_loss_pct)
                            if price <= trailing_stop:
                                exit_price = trailing_stop
                                break
                            if price >= take_profit:
                                exit_price = take_profit
                                break
                    elif self.order_type == 'oco':
                        for j in range(i+1, min(i+11, len(self.data))):
                            price = self.data['close'].iloc[j]
                            if price <= stop_loss:
                                exit_price = stop_loss
                                break
                            if price >= take_profit:
                                exit_price = take_profit
                                break
                    else:  # standard
                        for j in range(i+1, min(i+11, len(self.data))):
                            price = self.data['close'].iloc[j]
                            if price <= stop_loss:
                                exit_price = stop_loss
                                break
                            if price >= take_profit:
                                exit_price = take_profit
                                break
                if exit_price is None:
                    exit_price = self.data['close'].iloc[min(i+10, len(self.data)-1)]
                pnl = (exit_price - entry) / entry * size
                balance += pnl
                trades.append(pnl)
                equity_curve.append(balance)
                risk_manager.update_equity(balance)
        win_rate = np.mean([1 if t > 0 else 0 for t in trades]) if trades else 0
        sharpe = sharpe_ratio(np.array(trades)) if trades else 0
        sortino = sortino_ratio(np.array(trades)) if trades else 0
        mdd = max_drawdown(np.array(trades)) if trades else 0
        return {
            'trades': trades,  # List of trade PnLs
            'win_rate': win_rate,  # Proportion of profitable trades
            'sharpe': sharpe,  # Sharpe Ratio
            'sortino': sortino,  # Sortino Ratio
            'max_drawdown': mdd,  # Maximum Drawdown
            'final_balance': balance  # Final account balance
        }

def grid_search_optimizer(
    data,
    strategy_func,
    stop_loss_range,
    take_profit_range,
    max_risk_range,
    objective_metric='sharpe',
):
    """
    Perform grid search over Backtester parameters to maximize the objective metric.

    Args:
        data (pd.DataFrame): Historical price data.
        strategy_func (Callable): Trading strategy function.
        stop_loss_range (list): List of stop-loss percentages to try.
        take_profit_range (list): List of take-profit percentages to try.
        max_risk_range (list): List of max risk values to try.
        objective_metric (str): Metric to maximize ('sharpe', 'win_rate', etc.).

    Returns:
        dict: {'best_params': ..., 'best_result': ..., 'all_results': ...}
    """
    best_score = -float('inf')
    best_params = None
    best_result = None
    all_results = []
    for sl in stop_loss_range:
        for tp in take_profit_range:
            for mr in max_risk_range:
                bt = Backtester(data, strategy_func, stop_loss_pct=sl, take_profit_pct=tp, max_risk=mr)
                result = bt.run()
                score = result.get(objective_metric, 0)
                all_results.append({'params': {'stop_loss_pct': sl, 'take_profit_pct': tp, 'max_risk': mr}, 'result': result})
                if score > best_score:
                    best_score = score
                    best_params = {'stop_loss_pct': sl, 'take_profit_pct': tp, 'max_risk': mr}
                    best_result = result
    return {'best_params': best_params, 'best_result': best_result, 'all_results': all_results}

# Example usage:
# data = pd.read_csv('BTCUSD_15m.csv')
# def my_strategy(df): ...
# backtester = Backtester(data, my_strategy)
# report = backtester.run()
# print(report) 