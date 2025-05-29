"""
Performance Metrics Module

This module provides utilities for calculating trading performance metrics
from backtest results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_performance_metrics(
    equity_curve: List[Dict[str, Any]],
    trades: List[Dict[str, Any]],
    initial_capital: float
) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics from backtest results.
    
    Args:
        equity_curve: List of equity points throughout the backtest
        trades: List of trade dictionaries with entry/exit information
        initial_capital: Initial trading capital
        
    Returns:
        Dictionary of performance metrics
    """
    # Convert equity curve to DataFrame
    equity_df = pd.DataFrame(equity_curve)
    if len(equity_df) == 0:
        return {"error": "No equity data to analyze"}
    
    # Convert timestamps to datetime if they are strings
    if 'timestamp' in equity_df.columns and equity_df['timestamp'].dtype == 'object':
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
    
    # Filter out the first row with None timestamp
    if 'timestamp' in equity_df.columns:
        equity_df = equity_df[equity_df['timestamp'].notna()]
    
    # Calculate metrics
    metrics = {}
    
    # Get final equity and return
    initial_equity = float(initial_capital)
    if len(equity_df) > 0:
        final_equity = float(equity_df['equity'].iloc[-1])
    else:
        final_equity = initial_equity
    
    total_return = (final_equity / initial_equity) - 1
    metrics['initial_capital'] = initial_equity
    metrics['final_equity'] = final_equity
    metrics['absolute_return'] = final_equity - initial_equity
    metrics['percentage_return'] = total_return * 100
    
    # Calculate trade metrics
    if trades:
        # Extract trade data
        trade_returns = [trade['pnl_pct'] / 100 for trade in trades]  # Convert to decimal
        trade_pnls = [trade['net_pnl'] for trade in trades]
        win_trades = [t for t in trade_pnls if t > 0]
        loss_trades = [t for t in trade_pnls if t <= 0]
        
        # Win rate
        metrics['total_trades'] = len(trades)
        metrics['winning_trades'] = len(win_trades)
        metrics['losing_trades'] = len(loss_trades)
        metrics['win_rate'] = len(win_trades) / len(trades) if len(trades) > 0 else 0
        
        # Average trade metrics
        metrics['avg_trade_pnl'] = np.mean(trade_pnls) if trade_pnls else 0
        metrics['avg_winning_trade'] = np.mean(win_trades) if win_trades else 0
        metrics['avg_losing_trade'] = np.mean(loss_trades) if loss_trades else 0
        
        # Profit factor
        metrics['profit_factor'] = (
            sum(win_trades) / abs(sum(loss_trades))
            if loss_trades and sum(loss_trades) != 0
            else float('inf') if win_trades
            else 0
        )
        
        # Expectancy
        win_expectancy = metrics['win_rate'] * (np.mean(win_trades) if win_trades else 0)
        loss_expectancy = (1 - metrics['win_rate']) * (np.mean(loss_trades) if loss_trades else 0)
        metrics['expectancy'] = win_expectancy + loss_expectancy
        
        # Risk-reward ratio
        metrics['risk_reward_ratio'] = (
            abs(np.mean(win_trades) / np.mean(loss_trades))
            if loss_trades and np.mean(loss_trades) != 0
            else float('inf')
        )
        
        # Get trade length data
        if 'holding_time' in trades[0]:
            holding_times = [trade['holding_time'] for trade in trades]
            metrics['avg_holding_time'] = np.mean(holding_times)
            metrics['max_holding_time'] = np.max(holding_times)
            metrics['min_holding_time'] = np.min(holding_times)
    else:
        # No trades
        metrics['total_trades'] = 0
        metrics['winning_trades'] = 0
        metrics['losing_trades'] = 0
        metrics['win_rate'] = 0
        metrics['profit_factor'] = 0
        metrics['expectancy'] = 0
        metrics['risk_reward_ratio'] = 0
        metrics['avg_trade_pnl'] = 0
    
    # Calculate time-based metrics if we have timestamps
    if 'timestamp' in equity_df.columns and len(equity_df) > 1:
        # Convert timestamps to datetime if necessary
        if not pd.api.types.is_datetime64_any_dtype(equity_df['timestamp']):
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        
        # Calculate returns
        equity_df['return'] = equity_df['equity'].pct_change()
        
        # Trading period
        start_date = equity_df['timestamp'].iloc[0]
        end_date = equity_df['timestamp'].iloc[-1]
        trading_days = (end_date - start_date).days or 1  # Avoid division by zero
        
        metrics['trading_period_days'] = trading_days
        
        # Annualized return
        if trading_days > 0:
            annual_factor = 365 / trading_days
            metrics['annual_return'] = ((1 + total_return) ** annual_factor) - 1
        else:
            metrics['annual_return'] = 0
        
        # Calculate drawdown metrics
        dd_metrics = _calculate_drawdown_metrics(equity_df['equity'].values)
        metrics.update(dd_metrics)
        
        # Calculate risk-adjusted return metrics if we have at least 2 data points
        if len(equity_df) > 2:
            risk_metrics = _calculate_risk_metrics(equity_df)
            metrics.update(risk_metrics)
    
    # Round metrics for readability
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if 'time' in key or 'period' in key:
                # Don't round time periods
                continue
            metrics[key] = round(value, 4)
    
    return metrics


def _calculate_drawdown_metrics(equity_values: np.ndarray) -> Dict[str, float]:
    """
    Calculate drawdown metrics from an equity curve.
    
    Args:
        equity_values: Array of equity values
        
    Returns:
        Dictionary with drawdown metrics
    """
    metrics = {}
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_values)
    
    # Calculate drawdown in percentage terms
    drawdown = (running_max - equity_values) / running_max
    
    # Calculate various drawdown metrics
    metrics['max_drawdown'] = np.max(drawdown) * 100  # as percentage
    metrics['current_drawdown'] = drawdown[-1] * 100  # as percentage
    
    # Find the max drawdown period
    if len(equity_values) > 1:
        # Find the highest peak before the biggest valley
        max_dd_idx = np.argmax(drawdown)
        # Find the highest peak before this valley
        peak_idx = np.argmax(equity_values[:max_dd_idx+1])
        
        # Calculate drawdown duration
        dd_duration = max_dd_idx - peak_idx
        metrics['max_drawdown_duration'] = int(dd_duration)
        
        # Calculate time to recovery if recovery happened
        if max_dd_idx < len(equity_values) - 1:
            # Find the point where equity exceeds the previous peak
            for i in range(max_dd_idx + 1, len(equity_values)):
                if equity_values[i] >= equity_values[peak_idx]:
                    metrics['recovery_duration'] = i - max_dd_idx
                    break
            else:
                # No recovery yet
                metrics['recovery_duration'] = None
        else:
            # Drawdown is at the end of the data
            metrics['recovery_duration'] = None
    
    return metrics


def _calculate_risk_metrics(equity_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate risk-adjusted return metrics.
    
    Args:
        equity_df: DataFrame with equity and return columns
        
    Returns:
        Dictionary with risk metrics
    """
    metrics = {}
    
    # Calculate volatility (annualized)
    returns = equity_df['return'].dropna().values
    
    if len(returns) > 1:
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        metrics['volatility'] = volatility * 100  # as percentage
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        avg_return = np.mean(returns)
        metrics['sharpe_ratio'] = (avg_return / volatility) * np.sqrt(252) if volatility > 0 else 0
        
        # Calculate Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 1 else volatility
        metrics['sortino_ratio'] = (avg_return / downside_deviation) * np.sqrt(252) if downside_deviation > 0 else 0
        
        # Calculate Calmar ratio (annualized return / max drawdown)
        if 'max_drawdown' in metrics and metrics['max_drawdown'] > 0:
            annual_return = ((equity_df['equity'].iloc[-1] / equity_df['equity'].iloc[0]) ** (252 / len(returns)) - 1)
            metrics['calmar_ratio'] = annual_return / (metrics['max_drawdown'] / 100)
        else:
            metrics['calmar_ratio'] = 0
    
    return metrics


def calculate_daily_returns(equity_curve: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Calculate daily returns from an equity curve.
    
    Args:
        equity_curve: List of equity points
        
    Returns:
        DataFrame with daily returns
    """
    # Convert equity curve to DataFrame
    equity_df = pd.DataFrame(equity_curve)
    
    # Convert timestamps to datetime if they are strings
    if 'timestamp' in equity_df.columns and equity_df['timestamp'].dtype == 'object':
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
    
    # Filter out the first row with None timestamp
    if 'timestamp' in equity_df.columns:
        equity_df = equity_df[equity_df['timestamp'].notna()]
    
    # Calculate returns
    equity_df['return'] = equity_df['equity'].pct_change()
    
    # Resample to daily returns if we have more frequent data
    if 'timestamp' in equity_df.columns and len(equity_df) > 1:
        equity_df.set_index('timestamp', inplace=True)
        daily_returns = equity_df['equity'].resample('D').last().pct_change().dropna()
        return pd.DataFrame({'date': daily_returns.index, 'return': daily_returns.values})
    
    return equity_df[['timestamp', 'return']].rename(columns={'timestamp': 'date'})


def calculate_drawdowns(equity_curve: List[Dict[str, Any]], threshold: float = 0.05) -> List[Dict[str, Any]]:
    """
    Calculate significant drawdown periods.
    
    Args:
        equity_curve: List of equity points
        threshold: Minimum drawdown size to be considered significant (as decimal)
        
    Returns:
        List of drawdown periods
    """
    # Convert equity curve to DataFrame
    equity_df = pd.DataFrame(equity_curve)
    
    # Convert timestamps to datetime if they are strings
    if 'timestamp' in equity_df.columns and equity_df['timestamp'].dtype == 'object':
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
    
    # Filter out the first row with None timestamp
    if 'timestamp' in equity_df.columns:
        equity_df = equity_df[equity_df['timestamp'].notna()]
    
    # Check if we have enough data
    if len(equity_df) <= 1:
        return []
    
    # Get equity values
    equity_values = equity_df['equity'].values
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_values)
    
    # Calculate drawdown in percentage terms
    drawdown = (running_max - equity_values) / running_max
    
    # Find drawdown periods
    is_drawdown = False
    drawdown_periods = []
    current_period = {}
    
    for i in range(len(drawdown)):
        # Get timestamp if available
        timestamp = equity_df['timestamp'].iloc[i] if 'timestamp' in equity_df.columns else None
        
        if not is_drawdown and drawdown[i] >= threshold:
            # Start of a new drawdown period
            is_drawdown = True
            current_period = {
                'start_idx': i,
                'start_timestamp': timestamp,
                'start_equity': equity_values[i],
                'peak_equity': running_max[i],
                'peak_idx': np.argmax(equity_values[:i+1]),
            }
            current_period['peak_timestamp'] = equity_df['timestamp'].iloc[current_period['peak_idx']] if 'timestamp' in equity_df.columns else None
        
        elif is_drawdown:
            if drawdown[i] < threshold:
                # End of drawdown period
                is_drawdown = False
                current_period['end_idx'] = i
                current_period['end_timestamp'] = timestamp
                current_period['end_equity'] = equity_values[i]
                current_period['drawdown'] = (current_period['peak_equity'] - min(equity_values[current_period['start_idx']:i+1])) / current_period['peak_equity']
                current_period['duration'] = i - current_period['start_idx']
                
                # Save period
                if current_period['drawdown'] >= threshold:
                    drawdown_periods.append(current_period)
            elif i == len(drawdown) - 1:
                # End of data while still in drawdown
                current_period['end_idx'] = i
                current_period['end_timestamp'] = timestamp
                current_period['end_equity'] = equity_values[i]
                current_period['drawdown'] = (current_period['peak_equity'] - min(equity_values[current_period['start_idx']:i+1])) / current_period['peak_equity']
                current_period['duration'] = i - current_period['start_idx']
                
                # Save period
                if current_period['drawdown'] >= threshold:
                    drawdown_periods.append(current_period)
    
    return drawdown_periods 