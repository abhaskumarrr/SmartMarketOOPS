#!/usr/bin/env python3
"""
Refined Exit System - Conservative Approach

Based on analysis showing we're missing 5.44% profit per trade, this system focuses on:
1. Simple trailing stops (not complex partial exits)
2. Conservative parameters to avoid over-trading
3. Focus on the core issue: letting winners run longer
4. Maintain the same entry logic but improve exits only

Key insight: We had good entries (60% win rate) but terrible exits (100% early exit rate)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RefinedExitConfig:
    """Refined exit configuration - conservative approach"""
    # Simple trailing stop
    use_trailing_stop: bool = True
    trailing_stop_pct: float = 0.015      # 1.5% trailing stop
    min_profit_to_trail: float = 0.01     # Start trailing after 1% profit
    
    # Breakeven protection
    move_to_breakeven_at: float = 0.008   # Move to breakeven at 0.8% profit
    
    # Maximum loss protection
    max_loss_pct: float = 0.02            # 2% maximum loss
    
    # Time-based exit (prevent holding too long)
    max_hold_hours: int = 72              # 3 days maximum
    
    # Minimum hold time (prevent noise exits)
    min_hold_hours: int = 1               # Minimum 1 hour


class SimpleTrailingStopManager:
    """
    Simple trailing stop manager focused on letting winners run
    """
    
    def __init__(self, config: RefinedExitConfig):
        """Initialize trailing stop manager"""
        self.config = config
        self.positions = {}
    
    def open_position(self, trade_id: str, entry_price: float, direction: str, entry_time: datetime):
        """Open a new position"""
        initial_stop = self._calculate_initial_stop(entry_price, direction)
        
        self.positions[trade_id] = {
            'entry_price': entry_price,
            'direction': direction,
            'entry_time': entry_time,
            'current_stop': initial_stop,
            'highest_price': entry_price if direction == 'buy' else entry_price,
            'lowest_price': entry_price if direction == 'sell' else entry_price,
            'trailing_active': False,
            'moved_to_breakeven': False,
            'max_profit_seen': 0.0
        }
    
    def update_position(self, trade_id: str, current_price: float, current_time: datetime) -> Dict[str, Any]:
        """Update position and check for exit"""
        if trade_id not in self.positions:
            return {'action': 'hold', 'reason': 'position_not_found'}
        
        position = self.positions[trade_id]
        entry_price = position['entry_price']
        direction = position['direction']
        
        # Calculate current profit
        if direction == 'buy':
            current_profit = (current_price - entry_price) / entry_price
            position['highest_price'] = max(position['highest_price'], current_price)
        else:  # sell
            current_profit = (entry_price - current_price) / entry_price
            position['lowest_price'] = min(position['lowest_price'], current_price)
        
        position['max_profit_seen'] = max(position['max_profit_seen'], current_profit)
        
        # Check exit conditions
        exit_signal = self._check_exit_conditions(position, current_price, current_time)
        
        if exit_signal['action'] == 'exit':
            # Remove position
            del self.positions[trade_id]
        
        return exit_signal
    
    def _check_exit_conditions(self, position: Dict[str, Any], current_price: float, current_time: datetime) -> Dict[str, Any]:
        """Check all exit conditions"""
        direction = position['direction']
        entry_price = position['entry_price']
        entry_time = position['entry_time']
        
        # Calculate current profit
        if direction == 'buy':
            current_profit = (current_price - entry_price) / entry_price
        else:
            current_profit = (entry_price - current_price) / entry_price
        
        # 1. Check maximum loss
        if current_profit <= -self.config.max_loss_pct:
            return {
                'action': 'exit',
                'reason': 'max_loss_hit',
                'exit_price': current_price,
                'profit': current_profit
            }
        
        # 2. Check minimum hold time
        hold_time_hours = (current_time - entry_time).total_seconds() / 3600
        if hold_time_hours < self.config.min_hold_hours:
            return {'action': 'hold', 'reason': 'min_hold_time'}
        
        # 3. Check maximum hold time
        if hold_time_hours > self.config.max_hold_hours:
            return {
                'action': 'exit',
                'reason': 'max_hold_time',
                'exit_price': current_price,
                'profit': current_profit
            }
        
        # 4. Move to breakeven if profitable enough
        if (not position['moved_to_breakeven'] and 
            current_profit >= self.config.move_to_breakeven_at):
            self._move_to_breakeven(position)
        
        # 5. Activate trailing stop if profitable enough
        if (not position['trailing_active'] and 
            current_profit >= self.config.min_profit_to_trail):
            position['trailing_active'] = True
            self._update_trailing_stop(position, current_price)
        
        # 6. Update trailing stop if active
        if position['trailing_active']:
            self._update_trailing_stop(position, current_price)
        
        # 7. Check if trailing stop is hit
        if self._is_stop_hit(position, current_price):
            return {
                'action': 'exit',
                'reason': 'trailing_stop_hit',
                'exit_price': current_price,
                'profit': current_profit,
                'max_profit_seen': position['max_profit_seen']
            }
        
        return {'action': 'hold', 'reason': 'no_exit_condition'}
    
    def _calculate_initial_stop(self, entry_price: float, direction: str) -> float:
        """Calculate initial stop loss"""
        stop_distance = entry_price * self.config.max_loss_pct
        
        if direction == 'buy':
            return entry_price - stop_distance
        else:  # sell
            return entry_price + stop_distance
    
    def _move_to_breakeven(self, position: Dict[str, Any]):
        """Move stop to breakeven plus small buffer"""
        entry_price = position['entry_price']
        direction = position['direction']
        buffer = entry_price * 0.002  # 0.2% buffer
        
        if direction == 'buy':
            new_stop = entry_price + buffer
            position['current_stop'] = max(position['current_stop'], new_stop)
        else:  # sell
            new_stop = entry_price - buffer
            position['current_stop'] = min(position['current_stop'], new_stop)
        
        position['moved_to_breakeven'] = True
    
    def _update_trailing_stop(self, position: Dict[str, Any], current_price: float):
        """Update trailing stop"""
        direction = position['direction']
        trail_distance = current_price * self.config.trailing_stop_pct
        
        if direction == 'buy':
            new_stop = current_price - trail_distance
            # Only move stop up, never down
            if new_stop > position['current_stop']:
                position['current_stop'] = new_stop
        else:  # sell
            new_stop = current_price + trail_distance
            # Only move stop down, never up
            if new_stop < position['current_stop']:
                position['current_stop'] = new_stop
    
    def _is_stop_hit(self, position: Dict[str, Any], current_price: float) -> bool:
        """Check if stop is hit"""
        direction = position['direction']
        current_stop = position['current_stop']
        
        if direction == 'buy':
            return current_price <= current_stop
        else:  # sell
            return current_price >= current_stop


def run_refined_exit_backtest():
    """Run backtest with refined exit system"""
    
    print("🎯 REFINED EXIT SYSTEM BACKTEST")
    print("=" * 40)
    print("Conservative approach to fix core issue:")
    print("✅ Simple trailing stops (1.5%)")
    print("✅ Let winners run (start trailing at 1% profit)")
    print("✅ Breakeven protection (at 0.8% profit)")
    print("✅ Same entry logic as original")
    print("✅ Focus on capturing missed 5.44% per trade")
    
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
        
        # Run backtest with refined exits
        print(f"\n💰 Running backtest with refined exits...")
        result = run_backtest_with_refined_exits(analyzer, config)
        
        if result:
            print(f"\n🎯 REFINED EXIT RESULTS:")
            print("=" * 30)
            print(f"Total Trades: {result['total_trades']}")
            print(f"Total Return: {result['total_return']:.2%}")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {result['max_drawdown']:.2%}")
            print(f"Win Rate: {result['win_rate']:.1%}")
            print(f"Average Winner: {result['avg_winner']:.2%}")
            print(f"Average Loser: {result['avg_loser']:.2%}")
            print(f"Reward/Risk Ratio: {result['reward_risk_ratio']:.2f}")
            print(f"Max Profit Captured: {result['max_profit_captured']:.2%}")
            
            # Compare with original
            original_return = 0.27
            original_avg_winner = 2.18
            
            return_improvement = (result['total_return'] - original_return) / original_return * 100
            winner_improvement = (result['avg_winner'] - original_avg_winner) / original_avg_winner * 100
            
            print(f"\n📈 IMPROVEMENT vs ORIGINAL:")
            print(f"Return improvement: {return_improvement:+.1f}%")
            print(f"Average winner improvement: {winner_improvement:+.1f}%")
            print(f"Expected improvement: 200-300% (based on 5.44% missed profits)")
            
            if result['total_return'] > 1.0:  # > 1% return
                print("✅ SIGNIFICANT IMPROVEMENT ACHIEVED!")
            elif result['total_return'] > 0.5:
                print("⚠️  Good improvement, further optimization possible")
            elif result['total_return'] > original_return:
                print("✅ Positive improvement in right direction")
            else:
                print("❌ Need further refinement")
            
            # Analysis of exit reasons
            if 'exit_reasons' in result:
                print(f"\n📊 EXIT ANALYSIS:")
                for reason, count in result['exit_reasons'].items():
                    pct = count / result['total_trades'] * 100
                    print(f"   {reason}: {count} ({pct:.1f}%)")
            
            return result
        
        else:
            print("❌ Refined exit backtest failed")
            return None
            
    except Exception as e:
        print(f"❌ Refined exit system error: {e}")
        return None


def run_backtest_with_refined_exits(analyzer, config):
    """Run backtest using refined exit system"""
    
    # Initialize refined exit system
    exit_config = RefinedExitConfig()
    exit_manager = SimpleTrailingStopManager(exit_config)
    
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
            exit_signal = exit_manager.update_position(active_trade_id, current_price, current_time)
            
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
                    'max_profit_seen': exit_signal.get('max_profit_seen', 0),
                    'trade_id': active_trade_id
                })
                
                position = 0.0
                active_trade_id = None
        
        # Check for new entry signals (only if no position)
        if position == 0 and daily_trades < config.max_daily_trades:
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
                    daily_trades += 1
                    
                    # Initialize position in exit manager
                    exit_manager.open_position(active_trade_id, current_price, signal, current_time)
                    
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
    
    # Analyze trade performance
    entry_trades = [t for t in trades if t['action'] in ['buy', 'sell']]
    exit_trades = [t for t in trades if t['action'] == 'exit']
    
    # Calculate win/loss metrics
    pnls = []
    max_profits_seen = []
    
    for entry in entry_trades:
        matching_exits = [e for e in exit_trades if e.get('trade_id') == entry.get('trade_id')]
        if matching_exits:
            exit_trade = matching_exits[0]
            pnl = exit_trade.get('profit', 0)
            max_profit = exit_trade.get('max_profit_seen', 0)
            pnls.append(pnl)
            max_profits_seen.append(max_profit)
    
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
        'win_rate': len(winners) / len(pnls) if pnls else 0,
        'avg_winner': np.mean(winners) if winners else 0,
        'avg_loser': np.mean(losers) if losers else 0,
        'reward_risk_ratio': abs(np.mean(winners) / np.mean(losers)) if winners and losers else 0,
        'profit_factor': abs(sum(winners) / sum(losers)) if losers and sum(losers) != 0 else float('inf'),
        'max_profit_captured': np.mean(max_profits_seen) if max_profits_seen else 0,
        'exit_reasons': exit_reasons,
        'trades': trades
    }


if __name__ == "__main__":
    run_refined_exit_backtest()
