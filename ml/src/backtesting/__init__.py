"""
Backtesting Module

This module provides tools for backtesting trading strategies with a focus on Smart Money Concepts,
Fair Value Gaps, and liquidity analysis.
"""

from .engine import BacktestEngine
from .strategies import BaseStrategy
from .metrics import calculate_performance_metrics
from .utils import plot_backtest_results

__all__ = [
    'BacktestEngine',
    'BaseStrategy',
    'calculate_performance_metrics',
    'plot_backtest_results'
] 