#!/usr/bin/env python3
"""
Final Optimized Trading System

Based on professional analysis, the core issue is clear:
- Original system: 60% win rate, 2.18% avg winner, 0.27% total return
- Missing 5.44% profit per trade due to early exits
- Need to let winners run 2-3x longer while maintaining entry quality

Final approach:
1. Keep EXACT same entry logic (it's working - 60% win rate)
2. Only modify exits to capture more of the trend
3. Use very conservative trailing stops
4. Focus on the 3 best trades that could give us the missing profits
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
class FinalOptimizedConfig:
    """Final optimized configuration"""
    # Keep original entry parameters (they work!)
    symbol: str = "BTCUSDT"
    start_date: str = "2024-01-01"
    end_date: str = "2024-02-01"
    initial_capital: float = 10000.0
    confidence_threshold: float = 0.6  # Same as original
    max_position_size: float = 0.1     # Same as original
    max_daily_trades: int = 20          # Same as original
    transaction_cost: float = 0.001     # Same as original
    
    # ONLY modify exit strategy
    use_trend_following_exits: bool = True
    min_hold_periods: int = 8           # Hold minimum 8 periods (2 hours on 15m)
    profit_target_1: float = 0.02       # Take 25% profit at 2%
    profit_target_2: float = 0.04       # Take 25% profit at 4%
    trailing_stop_distance: float = 0.025  # 2.5% trailing stop (wider)
    max_loss: float = 0.015             # 1.5% max loss (tighter than trailing)


class TrendFollowingExitManager:
    """
    Simple trend-following exit manager
    Focus: Let winners run, cut losses short
    """
    
    def __init__(self, config: FinalOptimizedConfig):
        """Initialize exit manager"""
        self.config = config
        self.positions = {}
    
    def open_position(self, trade_id: str, entry_price: float, direction: str, 
                     entry_time: datetime, confidence: float):
        """Open new position"""
        self.positions[trade_id] = {
            'entry_price': entry_price,
            'direction': direction,
            'entry_time': entry_time,
            'confidence': confidence,
            'size_remaining': 1.0,
            'highest_price': entry_price,
            'lowest_price': entry_price,
            'periods_held': 0,
            'profit_targets_hit': 0,
            'max_profit_seen': 0.0
        }
    
    def update_position(self, trade_id: str, current_price: float, 
                       current_time: datetime) -> Dict[str, Any]:
        """Update position and check exits"""
        if trade_id not in self.positions:
            return {'action': 'hold'}
        
        position = self.positions[trade_id]
        position['periods_held'] += 1
        
        # Update price tracking
        position['highest_price'] = max(position['highest_price'], current_price)
        position['lowest_price'] = min(position['lowest_price'], current_price)
        
        # Calculate current profit
        entry_price = position['entry_price']
        direction = position['direction']
        
        if direction == 'buy':
            current_profit = (current_price - entry_price) / entry_price
        else:
            current_profit = (entry_price - current_price) / entry_price
        
        position['max_profit_seen'] = max(position['max_profit_seen'], current_profit)
        
        # Check exit conditions
        exit_signal = self._check_exit_conditions(position, current_price, current_time)
        
        if exit_signal['action'] == 'exit':
            del self.positions[trade_id]
        elif exit_signal['action'] == 'partial_exit':
            position['size_remaining'] -= exit_signal.get('size_reduction', 0.25)
            position['profit_targets_hit'] += 1
            
            if position['size_remaining'] <= 0.25:  # Close remaining if < 25%
                del self.positions[trade_id]
                exit_signal['action'] = 'exit'
                exit_signal['reason'] = 'final_partial_close'
        
        return exit_signal
    
    def _check_exit_conditions(self, position: Dict[str, Any], current_price: float, 
                              current_time: datetime) -> Dict[str, Any]:
        """Check all exit conditions"""
        entry_price = position['entry_price']
        direction = position['direction']
        
        # Calculate current profit
        if direction == 'buy':
            current_profit = (current_price - entry_price) / entry_price
        else:
            current_profit = (entry_price - current_price) / entry_price
        
        # 1. STOP LOSS - Cut losses short
        if current_profit <= -self.config.max_loss:
            return {
                'action': 'exit',
                'reason': 'stop_loss',
                'profit': current_profit,
                'periods_held': position['periods_held']
            }
        
        # 2. MINIMUM HOLD PERIOD - Prevent noise exits
        if position['periods_held'] < self.config.min_hold_periods:
            return {'action': 'hold', 'reason': 'min_hold_period'}
        
        # 3. PARTIAL PROFIT TAKING - Lock in some profits
        if (self.config.use_trend_following_exits and 
            position['profit_targets_hit'] == 0 and 
            current_profit >= self.config.profit_target_1):
            return {
                'action': 'partial_exit',
                'reason': 'profit_target_1',
                'profit': current_profit,
                'size_reduction': 0.25,  # Take 25% profit
                'periods_held': position['periods_held']
            }
        
        elif (position['profit_targets_hit'] == 1 and 
              current_profit >= self.config.profit_target_2):
            return {
                'action': 'partial_exit',
                'reason': 'profit_target_2',
                'profit': current_profit,
                'size_reduction': 0.25,  # Take another 25% profit
                'periods_held': position['periods_held']
            }
        
        # 4. TRAILING STOP - Let winners run but protect profits
        if current_profit > 0.01:  # Only trail if profitable
            max_profit = position['max_profit_seen']
            profit_drawdown = max_profit - current_profit
            
            # Exit if we've given back too much profit
            if profit_drawdown > self.config.trailing_stop_distance:
                return {
                    'action': 'exit',
                    'reason': 'trailing_stop',
                    'profit': current_profit,
                    'max_profit_seen': max_profit,
                    'profit_given_back': profit_drawdown,
                    'periods_held': position['periods_held']
                }
        
        # 5. MAXIMUM HOLD TIME - Prevent holding forever
        max_periods = 48  # 12 hours on 15m timeframe
        if position['periods_held'] > max_periods:
            return {
                'action': 'exit',
                'reason': 'max_hold_time',
                'profit': current_profit,
                'periods_held': position['periods_held']
            }
        
        return {'action': 'hold', 'reason': 'no_exit_condition'}


def run_final_optimized_backtest():
    """Run final optimized backtest"""
    
    print("🏆 FINAL OPTIMIZED TRADING SYSTEM")
    print("=" * 45)
    print("Conservative approach to capture missed profits:")
    print("✅ Keep SAME entry logic (60% win rate proven)")
    print("✅ Let winners run 2-3x longer")
    print("✅ Partial profit taking at 2% and 4%")
    print("✅ Wider trailing stops (2.5%)")
    print("✅ Minimum hold period (2 hours)")
    print("✅ Target: Capture the missing 5.44% per trade")
    
    try:
        # Import multi-timeframe system for entries
        from multi_timeframe_system_corrected import MultiTimeframeAnalyzer, MultiTimeframeConfig
        
        # Use same config as original (proven to work)
        config = FinalOptimizedConfig()
        analyzer = MultiTimeframeAnalyzer(MultiTimeframeConfig())
        
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
        
        # Run backtest
        print(f"\n💰 Running final optimized backtest...")
        result = run_final_backtest(analyzer, config)
        
        if result:
            print(f"\n🎯 FINAL OPTIMIZED RESULTS:")
            print("=" * 35)
            print(f"Total Trades: {result['total_trades']}")
            print(f"Total Return: {result['total_return']:.2%}")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {result['max_drawdown']:.2%}")
            print(f"Win Rate: {result['win_rate']:.1%}")
            print(f"Average Winner: {result['avg_winner']:.2%}")
            print(f"Average Loser: {result['avg_loser']:.2%}")
            print(f"Reward/Risk Ratio: {result['reward_risk_ratio']:.2f}")
            print(f"Profit Factor: {result['profit_factor']:.2f}")
            print(f"Average Hold Time: {result['avg_hold_periods']:.1f} periods")
            
            # Compare with original
            original_return = 0.27
            original_avg_winner = 2.18
            original_trades = 10
            
            return_improvement = ((result['total_return'] - original_return) / original_return * 100) if original_return != 0 else 0
            winner_improvement = ((result['avg_winner'] - original_avg_winner) / original_avg_winner * 100) if original_avg_winner != 0 else 0
            
            print(f"\n📈 vs ORIGINAL SYSTEM:")
            print(f"Return: {original_return:.2%} → {result['total_return']:.2%} ({return_improvement:+.1f}%)")
            print(f"Avg Winner: {original_avg_winner:.2%} → {result['avg_winner']:.2%} ({winner_improvement:+.1f}%)")
            print(f"Trades: {original_trades} → {result['total_trades']}")
            
            # Success criteria
            target_return = 1.0  # 1% target (original 0.27% + missed 5.44%/trade * efficiency)
            
            if result['total_return'] >= target_return:
                print("🎉 SUCCESS: Target return achieved!")
            elif result['total_return'] > original_return * 2:
                print("✅ GOOD: Significant improvement achieved")
            elif result['total_return'] > original_return:
                print("⚠️  PARTIAL: Some improvement, needs refinement")
            else:
                print("❌ FAILED: No improvement")
            
            # Exit analysis
            if 'exit_analysis' in result:
                print(f"\n📊 EXIT ANALYSIS:")
                for reason, data in result['exit_analysis'].items():
                    count = data['count']
                    avg_profit = data['avg_profit']
                    pct = count / result['total_trades'] * 100
                    print(f"   {reason}: {count} ({pct:.1f}%) - Avg: {avg_profit:.2%}")
            
            return result
        
        else:
            print("❌ Final optimized backtest failed")
            return None
            
    except Exception as e:
        print(f"❌ Final optimization error: {e}")
        return None


def run_final_backtest(analyzer, config):
    """Run backtest with final optimized system"""
    
    # Initialize exit manager
    exit_manager = TrendFollowingExitManager(config)
    
    # Use 15m timeframe as base
    base_data = analyzer.timeframe_data['15m']
    
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
    exit_analysis = {}
    
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
            
            if exit_signal['action'] in ['exit', 'partial_exit']:
                # Handle exit
                exit_size = 1.0 if exit_signal['action'] == 'exit' else exit_signal.get('size_reduction', 0.25)
                
                if position > 0:
                    exit_position = position * exit_size
                    proceeds = exit_position * current_price * (1 - config.transaction_cost)
                    capital += proceeds
                    position -= exit_position
                else:
                    exit_position = abs(position) * exit_size
                    cost = exit_position * current_price * (1 + config.transaction_cost)
                    capital -= cost
                    position += exit_position
                
                # Track exit
                reason = exit_signal['reason']
                if reason not in exit_analysis:
                    exit_analysis[reason] = {'count': 0, 'profits': []}
                
                exit_analysis[reason]['count'] += 1
                exit_analysis[reason]['profits'].append(exit_signal.get('profit', 0))
                
                trades.append({
                    'timestamp': current_time,
                    'action': 'exit',
                    'price': current_price,
                    'reason': reason,
                    'profit': exit_signal.get('profit', 0),
                    'periods_held': exit_signal.get('periods_held', 0),
                    'exit_size': exit_size,
                    'trade_id': active_trade_id
                })
                
                if exit_signal['action'] == 'exit' or abs(position) < 0.01:
                    position = 0.0
                    active_trade_id = None
        
        # Check for new entry signals (only if no position)
        if position == 0 and daily_trades < config.max_daily_trades:
            # Get signals using SAME logic as original
            htf_bias, htf_confidence = analyzer.analyze_higher_timeframe_consensus(current_time)
            ltf_signal = analyzer.analyze_lower_timeframe_entry(current_time, htf_bias, htf_confidence)
            
            signal = ltf_signal['signal']
            confidence = ltf_signal['confidence']
            
            # Enter new position with SAME criteria as original
            if signal in ['buy', 'sell'] and confidence >= config.confidence_threshold:
                trade_counter += 1
                active_trade_id = f"trade_{trade_counter}"
                
                # Calculate position size (same as original)
                position_size = config.max_position_size * confidence
                position_value = capital * position_size
                shares = position_value / current_price
                cost = shares * current_price * (1 + config.transaction_cost)
                
                if cost <= capital:
                    capital -= cost
                    position = shares if signal == 'buy' else -shares
                    daily_trades += 1
                    
                    # Initialize position in exit manager
                    exit_manager.open_position(active_trade_id, current_price, signal, current_time, confidence)
                    
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
    winners = [p for p in profits if p > 0]
    losers = [p for p in profits if p < 0]
    hold_periods = [t.get('periods_held', 0) for t in exit_trades]
    
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
    
    # Process exit analysis
    for reason in exit_analysis:
        exit_analysis[reason]['avg_profit'] = np.mean(exit_analysis[reason]['profits'])
    
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
        'profit_factor': abs(sum(winners) / sum(losers)) if losers and sum(losers) != 0 else float('inf'),
        'avg_hold_periods': np.mean(hold_periods) if hold_periods else 0,
        'exit_analysis': exit_analysis,
        'trades': trades
    }


if __name__ == "__main__":
    run_final_optimized_backtest()
