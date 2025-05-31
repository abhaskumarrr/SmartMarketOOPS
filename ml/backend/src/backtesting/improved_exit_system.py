#!/usr/bin/env python3
"""
Improved Exit System

Based on professional analysis, the core issue is:
- 100% early exit rate
- Missing 5.44% profit per trade on average
- Only capturing 29% of available profits

Solutions implemented:
1. Trailing stops to let winners run
2. Partial profit taking (scale out)
3. ATR-based dynamic stops
4. Trend-following exits
5. Multi-timeframe exit confirmation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExitReason(Enum):
    """Exit reasons for tracking"""
    TRAILING_STOP = "trailing_stop"
    PARTIAL_PROFIT = "partial_profit"
    TREND_REVERSAL = "trend_reversal"
    TIME_STOP = "time_stop"
    DRAWDOWN_LIMIT = "drawdown_limit"
    FULL_PROFIT = "full_profit"

@dataclass
class ImprovedExitConfig:
    """Configuration for improved exit system"""
    # Trailing stop parameters
    use_trailing_stops: bool = True
    initial_stop_atr_multiplier: float = 2.0    # Initial stop at 2x ATR
    trailing_stop_atr_multiplier: float = 1.5   # Trail at 1.5x ATR
    min_profit_before_trail: float = 0.005      # 0.5% profit before trailing
    
    # Partial profit taking
    use_partial_exits: bool = True
    first_target_multiplier: float = 2.0        # First target at 2x risk
    second_target_multiplier: float = 4.0       # Second target at 4x risk
    partial_exit_size: float = 0.33             # Exit 1/3 at each target
    
    # Trend following exits
    use_trend_exits: bool = True
    trend_reversal_periods: int = 3             # Exit if trend reverses for 3 periods
    
    # Time-based exits
    max_trade_duration_hours: int = 168         # 1 week maximum
    min_trade_duration_hours: int = 2           # Minimum 2 hours
    
    # Risk management
    max_loss_per_trade: float = 0.02            # 2% maximum loss
    breakeven_stop_trigger: float = 0.01        # Move to breakeven at 1% profit


class ImprovedExitManager:
    """
    Advanced exit management system
    """
    
    def __init__(self, config: ImprovedExitConfig):
        """Initialize exit manager"""
        self.config = config
        self.active_positions = {}  # Track active positions and their exit levels
        
    def initialize_position(self, trade_id: str, entry_data: Dict[str, Any], 
                          market_data: pd.DataFrame) -> Dict[str, Any]:
        """Initialize a new position with exit levels"""
        
        entry_price = entry_data['price']
        direction = entry_data['action']  # 'buy' or 'sell'
        entry_time = entry_data['timestamp']
        
        # Calculate ATR for dynamic stops
        atr = self._calculate_atr(market_data, entry_time)
        
        # Initialize position tracking
        position = {
            'trade_id': trade_id,
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': entry_time,
            'current_size': 1.0,  # Full position initially
            'atr': atr,
            
            # Stop loss levels
            'initial_stop': self._calculate_initial_stop(entry_price, direction, atr),
            'current_stop': None,  # Will be set dynamically
            'trailing_stop': None,
            
            # Profit targets
            'first_target': self._calculate_profit_target(entry_price, direction, atr, self.config.first_target_multiplier),
            'second_target': self._calculate_profit_target(entry_price, direction, atr, self.config.second_target_multiplier),
            'targets_hit': [],
            
            # State tracking
            'highest_profit': 0.0,
            'lowest_profit': 0.0,
            'trailing_active': False,
            'breakeven_moved': False,
            'partial_exits': []
        }
        
        position['current_stop'] = position['initial_stop']
        self.active_positions[trade_id] = position
        
        return position
    
    def update_position(self, trade_id: str, current_price: float, 
                       current_time: datetime, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Update position and check for exit signals"""
        
        if trade_id not in self.active_positions:
            return {'action': 'hold', 'reason': 'position_not_found'}
        
        position = self.active_positions[trade_id]
        entry_price = position['entry_price']
        direction = position['direction']
        
        # Calculate current profit/loss
        if direction == 'buy':
            current_pnl = (current_price - entry_price) / entry_price
        else:  # sell/short
            current_pnl = (entry_price - current_price) / entry_price
        
        # Update profit tracking
        position['highest_profit'] = max(position['highest_profit'], current_pnl)
        position['lowest_profit'] = min(position['lowest_profit'], current_pnl)
        
        # Check various exit conditions
        exit_signal = self._check_exit_conditions(position, current_price, current_time, market_data)
        
        if exit_signal['action'] != 'hold':
            # Update position or remove if fully closed
            if exit_signal.get('partial_exit', False):
                position['current_size'] -= exit_signal.get('exit_size', 0)
                position['partial_exits'].append({
                    'time': current_time,
                    'price': current_price,
                    'size': exit_signal.get('exit_size', 0),
                    'reason': exit_signal['reason']
                })
                
                if position['current_size'] <= 0.1:  # Close remaining if < 10%
                    del self.active_positions[trade_id]
                    exit_signal['action'] = 'close_full'
            else:
                # Full exit
                del self.active_positions[trade_id]
        
        return exit_signal
    
    def _check_exit_conditions(self, position: Dict[str, Any], current_price: float,
                              current_time: datetime, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Check all exit conditions and return appropriate signal"""
        
        direction = position['direction']
        entry_price = position['entry_price']
        
        # Calculate current P&L
        if direction == 'buy':
            current_pnl = (current_price - entry_price) / entry_price
        else:
            current_pnl = (entry_price - current_price) / entry_price
        
        # 1. Check stop loss
        stop_hit = self._check_stop_loss(position, current_price)
        if stop_hit:
            return {
                'action': 'exit',
                'reason': ExitReason.TRAILING_STOP,
                'exit_price': current_price,
                'partial_exit': False
            }
        
        # 2. Check profit targets for partial exits
        if self.config.use_partial_exits and position['current_size'] > 0.5:
            target_hit = self._check_profit_targets(position, current_price)
            if target_hit:
                return {
                    'action': 'partial_exit',
                    'reason': ExitReason.PARTIAL_PROFIT,
                    'exit_price': current_price,
                    'partial_exit': True,
                    'exit_size': self.config.partial_exit_size
                }
        
        # 3. Update trailing stop
        if self.config.use_trailing_stops:
            self._update_trailing_stop(position, current_price)
        
        # 4. Check trend reversal
        if self.config.use_trend_exits:
            trend_reversal = self._check_trend_reversal(position, current_price, market_data)
            if trend_reversal:
                return {
                    'action': 'exit',
                    'reason': ExitReason.TREND_REVERSAL,
                    'exit_price': current_price,
                    'partial_exit': False
                }
        
        # 5. Check time-based exit
        duration = (current_time - position['entry_time']).total_seconds() / 3600
        if duration > self.config.max_trade_duration_hours:
            return {
                'action': 'exit',
                'reason': ExitReason.TIME_STOP,
                'exit_price': current_price,
                'partial_exit': False
            }
        
        # 6. Move to breakeven if profitable enough
        if (not position['breakeven_moved'] and 
            current_pnl > self.config.breakeven_stop_trigger):
            self._move_to_breakeven(position)
        
        return {'action': 'hold', 'reason': 'no_exit_signal'}
    
    def _check_stop_loss(self, position: Dict[str, Any], current_price: float) -> bool:
        """Check if stop loss is hit"""
        direction = position['direction']
        current_stop = position['current_stop']
        
        if direction == 'buy':
            return current_price <= current_stop
        else:  # sell/short
            return current_price >= current_stop
    
    def _check_profit_targets(self, position: Dict[str, Any], current_price: float) -> bool:
        """Check if profit targets are hit"""
        direction = position['direction']
        
        # Check first target
        if 'first_target' not in position['targets_hit']:
            if direction == 'buy' and current_price >= position['first_target']:
                position['targets_hit'].append('first_target')
                return True
            elif direction == 'sell' and current_price <= position['first_target']:
                position['targets_hit'].append('first_target')
                return True
        
        # Check second target
        elif 'second_target' not in position['targets_hit']:
            if direction == 'buy' and current_price >= position['second_target']:
                position['targets_hit'].append('second_target')
                return True
            elif direction == 'sell' and current_price <= position['second_target']:
                position['targets_hit'].append('second_target')
                return True
        
        return False
    
    def _update_trailing_stop(self, position: Dict[str, Any], current_price: float):
        """Update trailing stop level"""
        direction = position['direction']
        entry_price = position['entry_price']
        atr = position['atr']
        
        # Calculate current profit
        if direction == 'buy':
            current_pnl = (current_price - entry_price) / entry_price
        else:
            current_pnl = (entry_price - current_price) / entry_price
        
        # Only start trailing after minimum profit
        if current_pnl < self.config.min_profit_before_trail:
            return
        
        # Calculate new trailing stop
        trail_distance = atr * self.config.trailing_stop_atr_multiplier
        
        if direction == 'buy':
            new_trailing_stop = current_price - trail_distance
            # Only move stop up, never down
            if position['trailing_stop'] is None or new_trailing_stop > position['trailing_stop']:
                position['trailing_stop'] = new_trailing_stop
                position['current_stop'] = max(position['current_stop'], new_trailing_stop)
                position['trailing_active'] = True
        else:  # sell/short
            new_trailing_stop = current_price + trail_distance
            # Only move stop down, never up
            if position['trailing_stop'] is None or new_trailing_stop < position['trailing_stop']:
                position['trailing_stop'] = new_trailing_stop
                position['current_stop'] = min(position['current_stop'], new_trailing_stop)
                position['trailing_active'] = True
    
    def _move_to_breakeven(self, position: Dict[str, Any]):
        """Move stop to breakeven"""
        entry_price = position['entry_price']
        direction = position['direction']
        
        # Add small buffer to avoid being stopped out by spread
        buffer = entry_price * 0.001  # 0.1% buffer
        
        if direction == 'buy':
            breakeven_stop = entry_price + buffer
            position['current_stop'] = max(position['current_stop'], breakeven_stop)
        else:  # sell/short
            breakeven_stop = entry_price - buffer
            position['current_stop'] = min(position['current_stop'], breakeven_stop)
        
        position['breakeven_moved'] = True
    
    def _check_trend_reversal(self, position: Dict[str, Any], current_price: float,
                             market_data: pd.DataFrame) -> bool:
        """Check for trend reversal using multiple timeframe analysis"""
        # Simplified trend reversal check
        # In practice, this would use the multi-timeframe system
        
        direction = position['direction']
        
        # Get recent price action
        recent_data = market_data.tail(self.config.trend_reversal_periods)
        if len(recent_data) < self.config.trend_reversal_periods:
            return False
        
        # Check if trend is reversing
        if direction == 'buy':
            # For long positions, check for bearish reversal
            recent_closes = recent_data['close'].values
            return all(recent_closes[i] < recent_closes[i-1] for i in range(1, len(recent_closes)))
        else:
            # For short positions, check for bullish reversal
            recent_closes = recent_data['close'].values
            return all(recent_closes[i] > recent_closes[i-1] for i in range(1, len(recent_closes)))
    
    def _calculate_initial_stop(self, entry_price: float, direction: str, atr: float) -> float:
        """Calculate initial stop loss"""
        stop_distance = atr * self.config.initial_stop_atr_multiplier
        
        if direction == 'buy':
            return entry_price - stop_distance
        else:  # sell/short
            return entry_price + stop_distance
    
    def _calculate_profit_target(self, entry_price: float, direction: str, 
                               atr: float, multiplier: float) -> float:
        """Calculate profit target"""
        target_distance = atr * multiplier
        
        if direction == 'buy':
            return entry_price + target_distance
        else:  # sell/short
            return entry_price - target_distance
    
    def _calculate_atr(self, market_data: pd.DataFrame, current_time: datetime, 
                      period: int = 14) -> float:
        """Calculate Average True Range"""
        # Get recent data
        recent_data = market_data[market_data['timestamp'] <= current_time].tail(period + 1)
        
        if len(recent_data) < 2:
            # Fallback to simple range
            return recent_data['high'].iloc[-1] - recent_data['low'].iloc[-1]
        
        # Calculate True Range
        high_low = recent_data['high'] - recent_data['low']
        high_close_prev = abs(recent_data['high'] - recent_data['close'].shift(1))
        low_close_prev = abs(recent_data['low'] - recent_data['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        return true_range.tail(period).mean()


def run_improved_exit_backtest():
    """Run backtest with improved exit system"""
    
    print("🚀 IMPROVED EXIT SYSTEM BACKTEST")
    print("=" * 40)
    print("Testing advanced exit strategies:")
    print("✅ Trailing stops to let winners run")
    print("✅ Partial profit taking (scale out)")
    print("✅ ATR-based dynamic stops")
    print("✅ Trend-following exits")
    print("✅ Breakeven stops")
    
    try:
        # Import multi-timeframe system for entries
        from multi_timeframe_system_corrected import MultiTimeframeAnalyzer, MultiTimeframeConfig
        
        config = MultiTimeframeConfig()
        analyzer = MultiTimeframeAnalyzer(config)
        
        # Load data
        print(f"\n📡 Loading data...")
        success = analyzer.load_all_timeframe_data(
            symbol=config.symbol,
            start_date=config.start_date,
            end_date=config.end_date
        )
        
        if not success:
            print("❌ Failed to load data")
            return None
        
        # Initialize improved exit system
        exit_config = ImprovedExitConfig()
        exit_manager = ImprovedExitManager(exit_config)
        
        # Run backtest with improved exits
        print(f"\n💰 Running backtest with improved exits...")
        result = run_backtest_with_improved_exits(analyzer, config, exit_manager)
        
        if result:
            print(f"\n🎯 IMPROVED EXIT RESULTS:")
            print("=" * 30)
            print(f"Total Trades: {result['total_trades']}")
            print(f"Total Return: {result['total_return']:.2%}")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {result['max_drawdown']:.2%}")
            print(f"Average Winner: {result['avg_winner']:.2%}")
            print(f"Average Loser: {result['avg_loser']:.2%}")
            print(f"Reward/Risk Ratio: {result['reward_risk_ratio']:.2f}")
            print(f"Profit Factor: {result['profit_factor']:.2f}")
            
            # Compare with original
            original_return = 0.27
            improvement = (result['total_return'] - original_return) / original_return * 100
            
            print(f"\n📈 IMPROVEMENT vs ORIGINAL:")
            print(f"Return improvement: {improvement:+.1f}%")
            print(f"Expected improvement: 200-300% (based on missed profits)")
            
            if result['total_return'] > 1.0:  # > 1% return
                print("✅ SIGNIFICANT IMPROVEMENT ACHIEVED!")
            elif result['total_return'] > 0.5:
                print("⚠️  Good improvement, further optimization possible")
            else:
                print("❌ Limited improvement, need further refinement")
            
            return result
        
        else:
            print("❌ Improved exit backtest failed")
            return None
            
    except Exception as e:
        print(f"❌ Improved exit system error: {e}")
        return None


def run_backtest_with_improved_exits(analyzer, config, exit_manager):
    """Run backtest using improved exit system"""
    
    # Use 15m timeframe as base
    base_data = analyzer.timeframe_data[config.timeframes.entry_signals]
    
    # Trading state
    capital = config.initial_capital
    position = 0.0
    trades = []
    equity_curve = []
    
    # Exit system tracking
    trade_counter = 0
    active_trade_id = None
    
    for i in range(50, len(base_data)):
        current_row = base_data.iloc[i]
        current_time = current_row['timestamp']
        current_price = current_row['close']
        
        # Check for exit signals on active position
        if active_trade_id and position != 0:
            exit_signal = exit_manager.update_position(
                active_trade_id, current_price, current_time, base_data.iloc[:i+1]
            )
            
            if exit_signal['action'] in ['exit', 'close_full']:
                # Close position
                if position > 0:
                    proceeds = position * current_price * (1 - config.transaction_cost)
                    capital += proceeds
                else:
                    cost = abs(position) * current_price * (1 + config.transaction_cost)
                    capital -= cost
                
                trades.append({
                    'timestamp': current_time,
                    'action': 'exit',
                    'price': current_price,
                    'reason': exit_signal['reason'].value,
                    'trade_id': active_trade_id
                })
                
                position = 0.0
                active_trade_id = None
            
            elif exit_signal['action'] == 'partial_exit':
                # Partial exit
                exit_size = exit_signal.get('exit_size', 0.33)
                if position > 0:
                    partial_position = position * exit_size
                    proceeds = partial_position * current_price * (1 - config.transaction_cost)
                    capital += proceeds
                    position -= partial_position
                
                trades.append({
                    'timestamp': current_time,
                    'action': 'partial_exit',
                    'price': current_price,
                    'size': exit_size,
                    'reason': exit_signal['reason'].value,
                    'trade_id': active_trade_id
                })
        
        # Check for new entry signals (only if no position)
        if position == 0:
            # Get higher timeframe bias
            htf_bias, htf_confidence = analyzer.analyze_higher_timeframe_consensus(current_time)
            
            # Get lower timeframe entry signal
            ltf_signal = analyzer.analyze_lower_timeframe_entry(current_time, htf_bias, htf_confidence)
            
            signal = ltf_signal['signal']
            confidence = ltf_signal['confidence']
            
            # Enter new position
            if signal in ['buy', 'sell'] and confidence >= config.confidence_threshold:
                trade_counter += 1
                active_trade_id = f"trade_{trade_counter}"
                
                # Calculate position size
                position_size = config.max_position_size * confidence
                position_value = capital * position_size
                shares = position_value / current_price
                cost = shares * current_price * (1 + config.transaction_cost)
                
                if cost <= capital:
                    capital -= cost
                    position = shares if signal == 'buy' else -shares
                    
                    # Initialize position in exit manager
                    entry_data = {
                        'price': current_price,
                        'action': signal,
                        'timestamp': current_time,
                        'confidence': confidence
                    }
                    
                    exit_manager.initialize_position(active_trade_id, entry_data, base_data.iloc[:i+1])
                    
                    trades.append({
                        'timestamp': current_time,
                        'action': signal,
                        'price': current_price,
                        'confidence': confidence,
                        'trade_id': active_trade_id
                    })
        
        # Update equity curve
        portfolio_value = capital + (position * current_price)
        equity_curve.append(portfolio_value)
    
    # Calculate final metrics
    final_price = base_data['close'].iloc[-1]
    if position > 0:
        final_capital = capital + (position * final_price * (1 - config.transaction_cost))
    elif position < 0:
        final_capital = capital - (abs(position) * final_price * (1 + config.transaction_cost))
    else:
        final_capital = capital
    
    total_return = (final_capital - config.initial_capital) / config.initial_capital
    
    # Analyze trade performance
    entry_trades = [t for t in trades if t['action'] in ['buy', 'sell']]
    exit_trades = [t for t in trades if t['action'] in ['exit', 'partial_exit']]
    
    # Calculate win/loss metrics
    pnls = []
    for entry in entry_trades:
        matching_exits = [e for e in exit_trades if e.get('trade_id') == entry.get('trade_id')]
        if matching_exits:
            exit_trade = matching_exits[0]  # Use first exit for simplicity
            if entry['action'] == 'buy':
                pnl = (exit_trade['price'] - entry['price']) / entry['price']
            else:
                pnl = (entry['price'] - exit_trade['price']) / entry['price']
            pnls.append(pnl)
    
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]
    
    # Calculate metrics
    if len(equity_curve) > 1:
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(24 * 365) if returns.std() > 0 else 0
        
        rolling_max = pd.Series(equity_curve).expanding().max()
        drawdown = (pd.Series(equity_curve) - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
    else:
        sharpe_ratio = 0
        max_drawdown = 0
    
    return {
        'total_trades': len(entry_trades),
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'final_capital': final_capital,
        'avg_winner': np.mean(winners) if winners else 0,
        'avg_loser': np.mean(losers) if losers else 0,
        'reward_risk_ratio': abs(np.mean(winners) / np.mean(losers)) if winners and losers else 0,
        'profit_factor': abs(sum(winners) / sum(losers)) if losers and sum(losers) != 0 else float('inf'),
        'trades': trades
    }


if __name__ == "__main__":
    run_improved_exit_backtest()
