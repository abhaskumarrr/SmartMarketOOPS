"""
Backtesting Engine

This module implements the core backtesting engine for simulating trading strategies
with a focus on Smart Money Concepts.
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
import json
import os

from .strategies import BaseStrategy
from .metrics import calculate_performance_metrics
from .utils import plot_backtest_results

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Engine for backtesting trading strategies using historical data.
    
    Features:
    - Support for Smart Money Concepts strategies
    - Realistic order execution with slippage and fees
    - Comprehensive performance metrics
    - Historical trade analysis
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.001,  # 0.1% trading fee
        slippage_factor: float = 0.0005,  # 0.05% slippage
        enable_fractional: bool = True,
        logging_level: str = "INFO"
    ):
        """
        Initialize the backtesting engine.
        
        Args:
            strategy: The trading strategy to backtest
            initial_capital: Starting capital for the backtest
            fee_rate: Trading fee rate (e.g., 0.001 for 0.1%)
            slippage_factor: Slippage factor for order execution (e.g., 0.0005 for 0.05%)
            enable_fractional: Whether to allow fractional positions
            logging_level: Level of logging detail
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage_factor = slippage_factor
        self.enable_fractional = enable_fractional
        
        # Set logging level
        self._set_logging_level(logging_level)
        
        # Internal state
        self.positions = {}  # Current positions {symbol: {size, entry_price, entry_time}}
        self.closed_positions = []  # History of closed positions
        self.orders = []  # Order history
        self.equity_curve = []  # Historical equity values
        self.metrics = {}  # Performance metrics
        
        # Execution settings
        self.position_sizing = 1.0  # Default position sizing (fraction of capital)
        
        # Results storage
        self.results = None
        
        logger.info(f"Backtesting engine initialized with {initial_capital:.2f} initial capital")
    
    def _set_logging_level(self, level: str):
        """Set the logging level"""
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            logging.warning(f"Invalid log level: {level}, using INFO")
            numeric_level = logging.INFO
        logging.getLogger().setLevel(numeric_level)
        logger.setLevel(numeric_level)
    
    def run(
        self,
        data: pd.DataFrame,
        symbol: str = "BTC/USDT",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the backtest on the provided data.
        
        Args:
            data: OHLCV dataframe with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            symbol: Trading symbol identifier
            start_date: Optional start date for the backtest (ISO format)
            end_date: Optional end date for the backtest (ISO format)
            
        Returns:
            Dictionary of backtest results
        """
        logger.info(f"Starting backtest for {symbol}")
        
        # Ensure timestamp column exists and reset index
        if 'timestamp' not in data.columns:
            if isinstance(data.index, pd.DatetimeIndex):
                data = data.reset_index()
                data.rename(columns={'index': 'timestamp'}, inplace=True)
            else:
                raise ValueError("DataFrame must have a 'timestamp' column or DatetimeIndex")
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Filter data by date range if specified
        if start_date:
            start_dt = pd.to_datetime(start_date)
            data = data[data['timestamp'] >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            data = data[data['timestamp'] <= end_dt]
        
        # Reset internal state
        self._reset_state()
        
        # Prepare the strategy
        self.strategy.initialize(data, symbol)
        
        # Loop through data rows for backtesting
        logger.info(f"Running backtest on {len(data)} candles from {data['timestamp'].min()} to {data['timestamp'].max()}")
        
        for i, row in data.iterrows():
            # Skip last row to avoid forward-looking trading
            if i >= len(data) - 1:
                break
            
            # Get current and next candle
            current_candle = row.to_dict()
            next_candle = data.iloc[i + 1].to_dict()
            
            # Update current index position for strategy access
            self.strategy.update_current_index(i)
            
            # Process orders and analyze market
            signals = self.strategy.generate_signals(current_candle, i)
            
            # Execute signals
            for signal in signals:
                self._execute_signal(signal, current_candle, next_candle)
            
            # Update portfolio value
            self._update_equity(current_candle['timestamp'], next_candle['close'])
        
        # Close any remaining positions at the end of the backtest
        self._close_all_positions(data.iloc[-1].to_dict())
        
        # Calculate performance metrics
        self.metrics = calculate_performance_metrics(
            self.equity_curve,
            self.closed_positions,
            self.initial_capital
        )
        
        # Prepare results
        self.results = {
            'symbol': symbol,
            'start_date': data['timestamp'].min(),
            'end_date': data['timestamp'].max(),
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'metrics': self.metrics,
            'equity_curve': self.equity_curve,
            'trades': self.closed_positions,
            'orders': self.orders
        }
        
        logger.info(f"Backtest completed. Final capital: {self.current_capital:.2f}, "
                   f"Return: {(self.current_capital/self.initial_capital - 1)*100:.2f}%, "
                   f"# Trades: {len(self.closed_positions)}")
        
        return self.results
    
    def _reset_state(self):
        """Reset internal state for a new backtest run"""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.closed_positions = []
        self.orders = []
        self.equity_curve = [{'timestamp': None, 'equity': self.initial_capital}]
        self.metrics = {}
    
    def _execute_signal(self, signal: Dict[str, Any], current_candle: Dict[str, Any], next_candle: Dict[str, Any]):
        """
        Execute a trading signal.
        
        Args:
            signal: Signal dictionary with action, direction, size, etc.
            current_candle: Current market data
            next_candle: Next candle for simulating execution
        """
        action = signal.get('action', '').lower()
        symbol = signal.get('symbol', '')
        direction = signal.get('direction', '').lower()
        size = signal.get('size', self.position_sizing)
        reason = signal.get('reason', 'strategy signal')
        
        # Check if we have a valid symbol
        if not symbol:
            logger.warning("Signal missing symbol, ignoring")
            return
        
        timestamp = current_candle['timestamp']
        current_price = current_candle['close']
        
        # Normalize position sizing
        if isinstance(size, float) and 0 < size <= 1.0:
            # Size as a fraction of capital
            size = self.position_sizing = size
        else:
            # Use default position sizing as a fraction
            size = self.position_sizing
        
        # Execute action based on signal
        if action == 'enter':
            # Calculate position size in asset units
            position_value = self.current_capital * size
            
            # Calculate execution price with slippage
            if direction == 'long':
                execution_price = next_candle['open'] * (1 + self.slippage_factor)
            elif direction == 'short':
                execution_price = next_candle['open'] * (1 - self.slippage_factor)
            else:
                logger.warning(f"Invalid direction '{direction}', ignoring signal")
                return
            
            # Calculate fees
            fees = position_value * self.fee_rate
            
            # Calculate actual position size in units
            position_size = position_value / execution_price
            
            # Check if we can open the position
            if position_value + fees > self.current_capital:
                logger.warning(f"Insufficient capital to open position: {position_value + fees:.2f} > {self.current_capital:.2f}")
                return
            
            # Add position
            self.positions[symbol] = {
                'direction': direction,
                'size': position_size,
                'value': position_value,
                'entry_price': execution_price,
                'entry_time': timestamp,
                'entry_reason': reason,
                'fees_paid': fees,
            }
            
            # Update capital
            self.current_capital -= (position_value + fees)
            
            # Record order
            self.orders.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'action': 'enter',
                'direction': direction,
                'size': position_size,
                'price': execution_price,
                'value': position_value,
                'fees': fees,
                'reason': reason
            })
            
            logger.debug(f"Opened {direction} position of {position_size:.6f} {symbol} at {execution_price:.2f} "
                        f"(value: {position_value:.2f}, fees: {fees:.2f})")
        
        elif action == 'exit':
            # Check if we have an open position
            if symbol not in self.positions:
                logger.warning(f"No open position for {symbol}, ignoring exit signal")
                return
            
            position = self.positions[symbol]
            position_size = position['size']
            entry_price = position['entry_price']
            entry_time = position['entry_time']
            position_direction = position['direction']
            entry_reason = position['reason'] if 'reason' in position else position['entry_reason']
            
            # Calculate execution price with slippage
            if position_direction == 'long':
                execution_price = next_candle['open'] * (1 - self.slippage_factor)
            else:  # short
                execution_price = next_candle['open'] * (1 + self.slippage_factor)
            
            # Calculate position value and P&L
            exit_value = position_size * execution_price
            entry_value = position['value']
            
            # Calculate P&L including fees
            if position_direction == 'long':
                pnl = exit_value - entry_value
            else:  # short
                pnl = entry_value - exit_value
            
            # Calculate exit fees
            exit_fees = exit_value * self.fee_rate
            total_fees = position['fees_paid'] + exit_fees
            
            # Update capital
            self.current_capital += (exit_value - exit_fees)
            
            # Record closed position
            self.closed_positions.append({
                'symbol': symbol,
                'direction': position_direction,
                'entry_time': entry_time,
                'exit_time': timestamp,
                'entry_price': entry_price,
                'exit_price': execution_price,
                'size': position_size,
                'pnl': pnl,
                'pnl_pct': (pnl / entry_value) * 100,
                'fees': total_fees,
                'net_pnl': pnl - total_fees,
                'entry_reason': entry_reason,
                'exit_reason': reason,
                'holding_time': (timestamp - entry_time).total_seconds() / 86400  # in days
            })
            
            # Record order
            self.orders.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'action': 'exit',
                'direction': position_direction,
                'size': position_size,
                'price': execution_price,
                'value': exit_value,
                'fees': exit_fees,
                'reason': reason
            })
            
            # Close position
            del self.positions[symbol]
            
            logger.debug(f"Closed {position_direction} position of {position_size:.6f} {symbol} at {execution_price:.2f} "
                        f"(P&L: {pnl:.2f}, fees: {total_fees:.2f}, net: {pnl-total_fees:.2f})")
    
    def _update_equity(self, timestamp, current_price):
        """
        Update the equity curve with current portfolio value.
        
        Args:
            timestamp: Current timestamp
            current_price: Current price for valuing open positions
        """
        # Calculate open positions value
        positions_value = sum(
            pos['size'] * current_price if pos['direction'] == 'long' else
            pos['value'] - (pos['size'] * current_price - pos['value'])
            for pos in self.positions.values()
        )
        
        # Total equity = cash + positions value
        total_equity = self.current_capital + positions_value
        
        # Add to equity curve
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'cash': self.current_capital,
            'positions_value': positions_value
        })
    
    def _close_all_positions(self, last_candle):
        """
        Close all open positions at the end of the backtest.
        
        Args:
            last_candle: Last candle data for execution
        """
        if not self.positions:
            return
        
        logger.info(f"Closing {len(self.positions)} open positions at the end of the backtest")
        
        for symbol, position in list(self.positions.items()):
            # Create exit signal
            exit_signal = {
                'action': 'exit',
                'symbol': symbol,
                'direction': position['direction'],
                'reason': 'end of backtest'
            }
            
            # Execute exit signal
            self._execute_signal(exit_signal, last_candle, last_candle)
    
    def set_position_sizing(self, sizing: float):
        """
        Set position sizing as a fraction of capital.
        
        Args:
            sizing: Position size as a fraction (0.0-1.0)
        """
        if not 0 < sizing <= 1.0:
            logger.warning(f"Invalid position sizing {sizing}, must be between 0 and 1")
            return
        
        self.position_sizing = sizing
        logger.info(f"Position sizing set to {sizing:.2%} of capital")
    
    def plot_results(
        self,
        output_dir: Optional[str] = None,
        plot_trades: bool = True,
        plot_equity: bool = True,
        plot_drawdown: bool = True
    ):
        """
        Plot backtest results.
        
        Args:
            output_dir: Directory to save plots (if None, plots are displayed)
            plot_trades: Whether to plot individual trades
            plot_equity: Whether to plot equity curve
            plot_drawdown: Whether to plot drawdown chart
        """
        if not self.results:
            logger.warning("No backtest results to plot, run backtest first")
            return
        
        # Ensure output directory exists if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Plot using utility function
        plot_backtest_results(
            self.results,
            output_dir=output_dir,
            plot_trades=plot_trades,
            plot_equity=plot_equity,
            plot_drawdown=plot_drawdown
        )
    
    def save_results(self, output_path: str):
        """
        Save backtest results to file.
        
        Args:
            output_path: Path to save results JSON file
        """
        if not self.results:
            logger.warning("No backtest results to save, run backtest first")
            return
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = self._prepare_results_for_serialization()
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Backtest results saved to {output_path}")
    
    def _prepare_results_for_serialization(self):
        """
        Prepare results for JSON serialization.
        
        Returns:
            JSON-serializable results dictionary
        """
        import copy
        results = copy.deepcopy(self.results)
        
        # Convert datetime objects to ISO format
        if isinstance(results['start_date'], datetime):
            results['start_date'] = results['start_date'].isoformat()
        if isinstance(results['end_date'], datetime):
            results['end_date'] = results['end_date'].isoformat()
        
        # Convert equity curve timestamps
        for item in results['equity_curve']:
            if isinstance(item['timestamp'], datetime):
                item['timestamp'] = item['timestamp'].isoformat()
        
        # Convert trade timestamps
        for trade in results['trades']:
            if isinstance(trade['entry_time'], datetime):
                trade['entry_time'] = trade['entry_time'].isoformat()
            if isinstance(trade['exit_time'], datetime):
                trade['exit_time'] = trade['exit_time'].isoformat()
        
        # Convert order timestamps
        for order in results['orders']:
            if isinstance(order['timestamp'], datetime):
                order['timestamp'] = order['timestamp'].isoformat()
        
        return results
    
    @classmethod
    def load_results(cls, input_path: str) -> Dict[str, Any]:
        """
        Load saved backtest results from file.
        
        Args:
            input_path: Path to results JSON file
            
        Returns:
            Dictionary of backtest results
        """
        with open(input_path, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Loaded backtest results from {input_path}")
        return results 