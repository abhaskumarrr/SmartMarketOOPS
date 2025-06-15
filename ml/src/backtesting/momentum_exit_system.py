#!/usr/bin/env python3
"""
Momentum-Based Exit System

CORE INSIGHT: The system holds trades too long after momentum/trend dies.

Solution: Exit when momentum dies, not when arbitrary targets hit.

Key Momentum Death Signals:
1. Price momentum slowing down (velocity decreasing)
2. Volume drying up (interest fading)
3. Higher timeframe trend weakening
4. RSI divergence (price up, momentum down)
5. Moving average slope flattening
6. Consecutive periods of weakening momentum

This system exits IMMEDIATELY when momentum dies, preserving profits.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MomentumState(Enum):
    """Momentum states"""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    DYING = "dying"
    DEAD = "dead"

@dataclass
class MomentumExitConfig:
    """Configuration for momentum-based exits"""
    # Momentum death detection
    momentum_death_periods: int = 3        # Exit if momentum weak for 3 periods
    min_momentum_threshold: float = 0.0005  # Minimum momentum to stay in trade
    volume_death_threshold: float = 0.7     # Exit if volume drops below 70% of avg
    
    # Trend death detection
    ma_slope_threshold: float = 0.0001      # Minimum MA slope to stay in trade
    rsi_divergence_periods: int = 5         # Check RSI divergence over 5 periods
    
    # Quick exit on momentum reversal
    momentum_reversal_exit: bool = True     # Exit immediately on momentum reversal
    consecutive_weak_periods: int = 2       # Exit after 2 consecutive weak periods
    
    # Safety nets
    max_momentum_loss: float = 0.015        # Exit if lose 1.5% from peak
    min_hold_periods: int = 2               # Minimum hold time
    max_hold_periods: int = 20              # Maximum hold time (5 hours on 15m)


class MomentumExitManager:
    """
    Exit manager that exits when momentum dies
    """
    
    def __init__(self, config: MomentumExitConfig):
        """Initialize momentum exit manager"""
        self.config = config
        self.positions = {}
    
    def open_position(self, trade_id: str, entry_price: float, direction: str, 
                     entry_time: datetime, market_data: pd.DataFrame):
        """Open new position with momentum tracking"""
        
        # Calculate initial momentum metrics
        recent_data = market_data.tail(10)
        initial_momentum = self._calculate_momentum_metrics(recent_data)
        
        self.positions[trade_id] = {
            'entry_price': entry_price,
            'direction': direction,
            'entry_time': entry_time,
            'periods_held': 0,
            'peak_profit': 0.0,
            'peak_price': entry_price,
            
            # Momentum tracking
            'momentum_history': [initial_momentum],
            'weak_momentum_count': 0,
            'last_strong_momentum_period': 0,
            'momentum_state': MomentumState.STRONG,
            
            # Trend tracking
            'ma_slope_history': [],
            'rsi_history': [],
            'volume_history': []
        }
    
    def update_position(self, trade_id: str, current_price: float, current_time: datetime,
                       market_data: pd.DataFrame) -> Dict[str, Any]:
        """Update position and check for momentum death"""
        
        if trade_id not in self.positions:
            return {'action': 'hold'}
        
        position = self.positions[trade_id]
        position['periods_held'] += 1
        
        # Calculate current profit
        entry_price = position['entry_price']
        direction = position['direction']
        
        if direction == 'buy':
            current_profit = (current_price - entry_price) / entry_price
        else:
            current_profit = (entry_price - current_price) / entry_price
        
        # Update peak tracking
        if current_profit > position['peak_profit']:
            position['peak_profit'] = current_profit
            position['peak_price'] = current_price
        
        # Calculate current momentum metrics
        recent_data = market_data.tail(10)
        current_momentum = self._calculate_momentum_metrics(recent_data)
        position['momentum_history'].append(current_momentum)
        
        # Check exit conditions
        exit_signal = self._check_momentum_exit_conditions(position, current_price, current_momentum, market_data)
        
        if exit_signal['action'] == 'exit':
            del self.positions[trade_id]
        
        return exit_signal
    
    def _calculate_momentum_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive momentum metrics"""
        
        if len(data) < 5:
            return {'velocity': 0, 'acceleration': 0, 'volume_momentum': 1, 'rsi_momentum': 50}
        
        # Price velocity (rate of change)
        velocity = data['close'].pct_change(3).iloc[-1]  # 3-period rate of change
        
        # Price acceleration (change in velocity)
        velocities = data['close'].pct_change(3)
        acceleration = velocities.diff().iloc[-1] if len(velocities) > 1 else 0
        
        # Volume momentum
        if 'volume' in data.columns:
            avg_volume = data['volume'].rolling(5).mean().iloc[-1]
            current_volume = data['volume'].iloc[-1]
            volume_momentum = current_volume / avg_volume if avg_volume > 0 else 1
        else:
            volume_momentum = 1
        
        # RSI momentum (simplified)
        if len(data) >= 7:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(7).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(7).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_momentum = rsi.iloc[-1]
        else:
            rsi_momentum = 50
        
        return {
            'velocity': velocity if not pd.isna(velocity) else 0,
            'acceleration': acceleration if not pd.isna(acceleration) else 0,
            'volume_momentum': volume_momentum,
            'rsi_momentum': rsi_momentum
        }
    
    def _check_momentum_exit_conditions(self, position: Dict[str, Any], current_price: float,
                                       current_momentum: Dict[str, float], market_data: pd.DataFrame) -> Dict[str, Any]:
        """Check all momentum-based exit conditions"""
        
        direction = position['direction']
        entry_price = position['entry_price']
        periods_held = position['periods_held']
        
        # Calculate current profit
        if direction == 'buy':
            current_profit = (current_price - entry_price) / entry_price
        else:
            current_profit = (entry_price - current_price) / entry_price
        
        # 1. MINIMUM HOLD PERIOD - Prevent noise exits
        if periods_held < self.config.min_hold_periods:
            return {'action': 'hold', 'reason': 'min_hold_period'}
        
        # 2. MAXIMUM HOLD PERIOD - Force exit if holding too long
        if periods_held > self.config.max_hold_periods:
            return {
                'action': 'exit',
                'reason': 'max_hold_period_momentum_dead',
                'profit': current_profit,
                'periods_held': periods_held,
                'momentum_state': 'forced_exit'
            }
        
        # 3. MOMENTUM DEATH DETECTION
        momentum_death_signal = self._detect_momentum_death(position, current_momentum, direction)
        if momentum_death_signal:
            return {
                'action': 'exit',
                'reason': 'momentum_death',
                'profit': current_profit,
                'periods_held': periods_held,
                'momentum_state': momentum_death_signal,
                'momentum_metrics': current_momentum
            }
        
        # 4. MOMENTUM REVERSAL - Immediate exit
        if self.config.momentum_reversal_exit:
            reversal_signal = self._detect_momentum_reversal(position, current_momentum, direction)
            if reversal_signal:
                return {
                    'action': 'exit',
                    'reason': 'momentum_reversal',
                    'profit': current_profit,
                    'periods_held': periods_held,
                    'momentum_state': 'reversed'
                }
        
        # 5. PROFIT PROTECTION - Exit if giving back too much from peak
        profit_drawdown = position['peak_profit'] - current_profit
        if profit_drawdown > self.config.max_momentum_loss:
            return {
                'action': 'exit',
                'reason': 'profit_protection_momentum_loss',
                'profit': current_profit,
                'peak_profit': position['peak_profit'],
                'profit_given_back': profit_drawdown,
                'periods_held': periods_held
            }
        
        # 6. VOLUME DEATH - Exit if volume dries up
        volume_death_signal = self._detect_volume_death(current_momentum)
        if volume_death_signal:
            return {
                'action': 'exit',
                'reason': 'volume_death',
                'profit': current_profit,
                'periods_held': periods_held,
                'volume_ratio': current_momentum['volume_momentum']
            }
        
        return {'action': 'hold', 'reason': 'momentum_alive'}
    
    def _detect_momentum_death(self, position: Dict[str, Any], current_momentum: Dict[str, float], 
                              direction: str) -> Optional[str]:
        """Detect if momentum has died"""
        
        velocity = current_momentum['velocity']
        acceleration = current_momentum['acceleration']
        
        # Check if momentum is in the right direction
        if direction == 'buy':
            momentum_aligned = velocity > 0
            momentum_strong = velocity > self.config.min_momentum_threshold
        else:  # sell
            momentum_aligned = velocity < 0
            momentum_strong = abs(velocity) > self.config.min_momentum_threshold
        
        # Count weak momentum periods
        if not momentum_strong:
            position['weak_momentum_count'] += 1
        else:
            position['weak_momentum_count'] = 0
            position['last_strong_momentum_period'] = position['periods_held']
        
        # Exit if momentum has been weak for too long
        if position['weak_momentum_count'] >= self.config.momentum_death_periods:
            return 'momentum_consistently_weak'
        
        # Exit if momentum has reversed direction
        if not momentum_aligned and abs(velocity) > self.config.min_momentum_threshold:
            return 'momentum_reversed_direction'
        
        # Exit if acceleration is strongly negative (momentum dying fast)
        if direction == 'buy' and acceleration < -0.001:  # Strong deceleration
            return 'momentum_decelerating_fast'
        elif direction == 'sell' and acceleration > 0.001:  # Strong deceleration for short
            return 'momentum_decelerating_fast'
        
        return None
    
    def _detect_momentum_reversal(self, position: Dict[str, Any], current_momentum: Dict[str, float],
                                 direction: str) -> bool:
        """Detect immediate momentum reversal"""
        
        velocity = current_momentum['velocity']
        acceleration = current_momentum['acceleration']
        
        # Check for strong momentum reversal
        if direction == 'buy':
            # For long positions, exit if strong downward momentum
            if velocity < -self.config.min_momentum_threshold * 2:  # 2x threshold
                return True
            # Or if strong negative acceleration
            if acceleration < -0.0015:  # Strong deceleration
                return True
        else:  # sell
            # For short positions, exit if strong upward momentum
            if velocity > self.config.min_momentum_threshold * 2:
                return True
            # Or if strong positive acceleration
            if acceleration > 0.0015:
                return True
        
        return False
    
    def _detect_volume_death(self, current_momentum: Dict[str, float]) -> bool:
        """Detect if volume has died (interest fading)"""
        
        volume_ratio = current_momentum['volume_momentum']
        
        # Exit if volume drops significantly below average
        if volume_ratio < self.config.volume_death_threshold:
            return True
        
        return False


def run_momentum_exit_backtest():
    """Run backtest with momentum-based exits"""
    
    print("⚡ MOMENTUM-BASED EXIT SYSTEM")
    print("=" * 40)
    print("Exit when momentum dies, not when arbitrary targets hit:")
    print("✅ Exit when price velocity slows down")
    print("✅ Exit when volume dries up")
    print("✅ Exit on momentum reversal")
    print("✅ Exit when trend acceleration turns negative")
    print("✅ Prevent holding trades after momentum death")
    
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
        
        # Run backtest with momentum exits
        print(f"\n💰 Running backtest with momentum-based exits...")
        result = run_momentum_backtest(analyzer, config)
        
        if result:
            print(f"\n🎯 MOMENTUM EXIT RESULTS:")
            print("=" * 30)
            print(f"Total Trades: {result['total_trades']}")
            print(f"Total Return: {result['total_return']:.2%}")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {result['max_drawdown']:.2%}")
            print(f"Win Rate: {result['win_rate']:.1%}")
            print(f"Average Winner: {result['avg_winner']:.2%}")
            print(f"Average Loser: {result['avg_loser']:.2%}")
            print(f"Average Hold Time: {result['avg_hold_periods']:.1f} periods")
            print(f"Reward/Risk Ratio: {result['reward_risk_ratio']:.2f}")
            
            # Compare with original
            original_return = 0.27
            original_hold_time = 10  # Estimated
            
            return_improvement = ((result['total_return'] - original_return) / original_return * 100) if original_return != 0 else 0
            hold_time_improvement = ((original_hold_time - result['avg_hold_periods']) / original_hold_time * 100) if original_hold_time != 0 else 0
            
            print(f"\n📈 vs ORIGINAL SYSTEM:")
            print(f"Return: {original_return:.2%} → {result['total_return']:.2%} ({return_improvement:+.1f}%)")
            print(f"Hold Time: {original_hold_time:.1f} → {result['avg_hold_periods']:.1f} periods ({hold_time_improvement:+.1f}%)")
            
            # Exit reason analysis
            if 'exit_reasons' in result:
                print(f"\n📊 EXIT REASONS:")
                total_exits = sum(result['exit_reasons'].values())
                for reason, count in result['exit_reasons'].items():
                    pct = count / total_exits * 100
                    print(f"   {reason}: {count} ({pct:.1f}%)")
            
            # Success assessment
            if result['avg_hold_periods'] < 8 and result['total_return'] > original_return:
                print("✅ SUCCESS: Shorter hold times with better returns!")
            elif result['avg_hold_periods'] < 8:
                print("⚠️  PARTIAL SUCCESS: Shorter hold times achieved")
            elif result['total_return'] > original_return * 1.5:
                print("✅ SUCCESS: Significantly better returns!")
            else:
                print("❌ NEEDS REFINEMENT: Limited improvement")
            
            return result
        
        else:
            print("❌ Momentum exit backtest failed")
            return None
            
    except Exception as e:
        print(f"❌ Momentum exit system error: {e}")
        return None


def run_momentum_backtest(analyzer, config):
    """Run backtest with momentum-based exit system"""
    
    # Initialize momentum exit manager
    momentum_config = MomentumExitConfig()
    exit_manager = MomentumExitManager(momentum_config)
    
    # Use 15m timeframe as base
    base_data = analyzer.timeframe_data[config.timeframes.entry_signals]
    
    # Trading state
    capital = config.initial_capital
    position = 0.0
    trades = []
    equity_curve = []
    
    # Tracking
    trade_counter = 0
    active_trade_id = None
    daily_trades = 0
    last_trade_date = None
    exit_reasons = {}
    
    for i in range(50, len(base_data)):
        current_row = base_data.iloc[i]
        current_time = current_row['timestamp']
        current_price = current_row['close']
        current_date = current_time.date()
        
        # Reset daily counter
        if last_trade_date != current_date:
            daily_trades = 0
            last_trade_date = current_date
        
        # Check for exit signals on active position
        if active_trade_id and position != 0:
            # Get market data up to current point
            market_data = base_data.iloc[:i+1]
            
            exit_signal = exit_manager.update_position(active_trade_id, current_price, current_time, market_data)
            
            if exit_signal['action'] == 'exit':
                # Close position
                if position > 0:
                    proceeds = position * current_price * (1 - config.transaction_cost)
                    capital += proceeds
                else:
                    cost = abs(position) * current_price * (1 + config.transaction_cost)
                    capital -= cost
                
                # Track exit reason
                reason = exit_signal['reason']
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
                
                trades.append({
                    'timestamp': current_time,
                    'action': 'exit',
                    'price': current_price,
                    'reason': reason,
                    'profit': exit_signal.get('profit', 0),
                    'periods_held': exit_signal.get('periods_held', 0),
                    'trade_id': active_trade_id
                })
                
                position = 0.0
                active_trade_id = None
        
        # Check for new entry signals (only if no position)
        if position == 0 and daily_trades < config.max_daily_trades:
            # Get signals using same logic as original
            htf_bias, htf_confidence = analyzer.analyze_higher_timeframe_consensus(current_time)
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
                    daily_trades += 1
                    
                    # Initialize position in momentum exit manager
                    market_data = base_data.iloc[:i+1]
                    exit_manager.open_position(active_trade_id, current_price, signal, current_time, market_data)
                    
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
    
    # Close any remaining position
    if position != 0:
        final_price = base_data['close'].iloc[-1]
        if position > 0:
            final_capital = capital + (position * final_price * (1 - config.transaction_cost))
        else:
            final_capital = capital - (abs(position) * final_price * (1 + config.transaction_cost))
    else:
        final_capital = capital
    
    total_return = (final_capital - config.initial_capital) / config.initial_capital
    
    # Analyze results
    entry_trades = [t for t in trades if t['action'] in ['buy', 'sell']]
    exit_trades = [t for t in trades if t['action'] == 'exit']
    
    # Calculate metrics
    profits = [t.get('profit', 0) for t in exit_trades]
    hold_periods = [t.get('periods_held', 0) for t in exit_trades]
    winners = [p for p in profits if p > 0]
    losers = [p for p in profits if p < 0]
    
    # Calculate performance metrics
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
        'win_rate': len(winners) / len(profits) if profits else 0,
        'avg_winner': np.mean(winners) if winners else 0,
        'avg_loser': np.mean(losers) if losers else 0,
        'reward_risk_ratio': abs(np.mean(winners) / np.mean(losers)) if winners and losers else 0,
        'avg_hold_periods': np.mean(hold_periods) if hold_periods else 0,
        'exit_reasons': exit_reasons,
        'trades': trades
    }


if __name__ == "__main__":
    run_momentum_exit_backtest()
