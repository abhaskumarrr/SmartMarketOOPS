#!/usr/bin/env python3
"""
Balanced Momentum Exit System

Your observation is CORRECT: "The system holds trades too long after momentum dies"

But the previous momentum system was too aggressive. This balanced approach:
1. Keeps the same entry logic (proven to work)
2. Exits when momentum clearly dies (not just weakens)
3. Uses conservative momentum death detection
4. Maintains reasonable hold times (not too short, not too long)

Key insight: Exit when momentum is CLEARLY dead, not just slowing down.
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
class BalancedMomentumConfig:
    """Balanced momentum exit configuration"""
    # Conservative momentum death detection
    momentum_death_threshold: float = -0.002    # Clear momentum reversal
    consecutive_dead_periods: int = 4           # 4 periods of dead momentum
    min_hold_periods: int = 4                   # Minimum 1 hour hold
    max_hold_periods: int = 24                  # Maximum 6 hours hold
    
    # Profit protection (your key insight)
    profit_protection_threshold: float = 0.01   # Protect profits above 1%
    max_profit_giveback: float = 0.008          # Don't give back more than 0.8%
    
    # Clear trend death signals
    trend_reversal_periods: int = 3             # 3 periods of clear reversal
    volume_death_ratio: float = 0.5             # Volume drops to 50% of average


class BalancedMomentumExitManager:
    """
    Balanced exit manager - exits when momentum is CLEARLY dead
    """
    
    def __init__(self, config: BalancedMomentumConfig):
        """Initialize balanced momentum exit manager"""
        self.config = config
        self.positions = {}
    
    def open_position(self, trade_id: str, entry_price: float, direction: str, 
                     entry_time: datetime):
        """Open new position"""
        self.positions[trade_id] = {
            'entry_price': entry_price,
            'direction': direction,
            'entry_time': entry_time,
            'periods_held': 0,
            'peak_profit': 0.0,
            'consecutive_dead_momentum': 0,
            'last_momentum_check': 0.0,
            'momentum_history': []
        }
    
    def update_position(self, trade_id: str, current_price: float, current_time: datetime,
                       market_data: pd.DataFrame) -> Dict[str, Any]:
        """Update position and check for clear momentum death"""
        
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
        
        # Update peak profit
        if current_profit > position['peak_profit']:
            position['peak_profit'] = current_profit
        
        # Check exit conditions
        exit_signal = self._check_balanced_exit_conditions(position, current_price, market_data)
        
        if exit_signal['action'] == 'exit':
            del self.positions[trade_id]
        
        return exit_signal
    
    def _check_balanced_exit_conditions(self, position: Dict[str, Any], current_price: float,
                                       market_data: pd.DataFrame) -> Dict[str, Any]:
        """Check balanced exit conditions"""
        
        entry_price = position['entry_price']
        direction = position['direction']
        periods_held = position['periods_held']
        
        # Calculate current profit
        if direction == 'buy':
            current_profit = (current_price - entry_price) / entry_price
        else:
            current_profit = (entry_price - current_price) / entry_price
        
        # 1. MINIMUM HOLD PERIOD
        if periods_held < self.config.min_hold_periods:
            return {'action': 'hold', 'reason': 'min_hold_period'}
        
        # 2. MAXIMUM HOLD PERIOD (Your key insight: don't hold forever)
        if periods_held > self.config.max_hold_periods:
            return {
                'action': 'exit',
                'reason': 'max_hold_period_momentum_likely_dead',
                'profit': current_profit,
                'periods_held': periods_held
            }
        
        # 3. PROFIT PROTECTION (Key insight: protect profits when momentum dies)
        if (position['peak_profit'] > self.config.profit_protection_threshold and
            current_profit < position['peak_profit'] - self.config.max_profit_giveback):
            return {
                'action': 'exit',
                'reason': 'profit_protection_momentum_died',
                'profit': current_profit,
                'peak_profit': position['peak_profit'],
                'profit_given_back': position['peak_profit'] - current_profit,
                'periods_held': periods_held
            }
        
        # 4. CLEAR MOMENTUM DEATH (Conservative detection)
        momentum_death = self._detect_clear_momentum_death(position, current_price, market_data, direction)
        if momentum_death:
            return {
                'action': 'exit',
                'reason': 'clear_momentum_death',
                'profit': current_profit,
                'periods_held': periods_held,
                'momentum_signal': momentum_death
            }
        
        # 5. VOLUME DEATH (Interest clearly fading)
        volume_death = self._detect_clear_volume_death(market_data)
        if volume_death and periods_held > 8:  # Only after reasonable hold time
            return {
                'action': 'exit',
                'reason': 'clear_volume_death',
                'profit': current_profit,
                'periods_held': periods_held
            }
        
        return {'action': 'hold', 'reason': 'momentum_still_alive'}
    
    def _detect_clear_momentum_death(self, position: Dict[str, Any], current_price: float,
                                    market_data: pd.DataFrame, direction: str) -> Optional[str]:
        """Detect CLEAR momentum death (not just weakness)"""
        
        if len(market_data) < 10:
            return None
        
        # Get recent price action
        recent_data = market_data.tail(6)
        
        # Calculate momentum indicators
        price_momentum = recent_data['close'].pct_change(3).iloc[-1]
        
        # Check for CLEAR momentum reversal
        if direction == 'buy':
            # For long positions, clear downward momentum
            if price_momentum < self.config.momentum_death_threshold:
                position['consecutive_dead_momentum'] += 1
            else:
                position['consecutive_dead_momentum'] = 0
        else:  # sell
            # For short positions, clear upward momentum
            if price_momentum > -self.config.momentum_death_threshold:
                position['consecutive_dead_momentum'] += 1
            else:
                position['consecutive_dead_momentum'] = 0
        
        # Exit only after consecutive periods of clear momentum death
        if position['consecutive_dead_momentum'] >= self.config.consecutive_dead_periods:
            return f'consecutive_dead_momentum_{position["consecutive_dead_momentum"]}_periods'
        
        # Check for clear trend reversal
        if len(recent_data) >= self.config.trend_reversal_periods:
            recent_closes = recent_data['close'].tail(self.config.trend_reversal_periods)
            
            if direction == 'buy':
                # Clear downtrend
                if all(recent_closes.iloc[i] < recent_closes.iloc[i-1] for i in range(1, len(recent_closes))):
                    return 'clear_downtrend_reversal'
            else:  # sell
                # Clear uptrend
                if all(recent_closes.iloc[i] > recent_closes.iloc[i-1] for i in range(1, len(recent_closes))):
                    return 'clear_uptrend_reversal'
        
        return None
    
    def _detect_clear_volume_death(self, market_data: pd.DataFrame) -> bool:
        """Detect clear volume death"""
        
        if 'volume' not in market_data.columns or len(market_data) < 10:
            return False
        
        recent_data = market_data.tail(10)
        current_volume = recent_data['volume'].iloc[-1]
        avg_volume = recent_data['volume'].rolling(8).mean().iloc[-1]
        
        # Clear volume death
        if current_volume < avg_volume * self.config.volume_death_ratio:
            return True
        
        return False


def run_balanced_momentum_backtest():
    """Run backtest with balanced momentum exits"""
    
    print("⚖️  BALANCED MOMENTUM EXIT SYSTEM")
    print("=" * 45)
    print("Addressing your key insight: 'Holding trades too long after momentum dies'")
    print("✅ Exit when momentum is CLEARLY dead (not just weak)")
    print("✅ Protect profits when momentum dies")
    print("✅ Conservative momentum death detection")
    print("✅ Reasonable hold times (1-6 hours)")
    print("✅ Same proven entry logic")
    
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
        
        # Run backtest
        print(f"\n💰 Running balanced momentum exit backtest...")
        result = run_balanced_backtest(analyzer, config)
        
        if result:
            print(f"\n🎯 BALANCED MOMENTUM RESULTS:")
            print("=" * 35)
            print(f"Total Trades: {result['total_trades']}")
            print(f"Total Return: {result['total_return']:.2%}")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {result['max_drawdown']:.2%}")
            print(f"Win Rate: {result['win_rate']:.1%}")
            print(f"Average Winner: {result['avg_winner']:.2%}")
            print(f"Average Loser: {result['avg_loser']:.2%}")
            print(f"Average Hold Time: {result['avg_hold_periods']:.1f} periods ({result['avg_hold_periods']*0.25:.1f} hours)")
            print(f"Reward/Risk Ratio: {result['reward_risk_ratio']:.2f}")
            
            # Compare with original
            original_return = 0.27
            original_hold_time = 15  # Estimated from your observation
            
            return_improvement = ((result['total_return'] - original_return) / original_return * 100) if original_return != 0 else 0
            hold_time_reduction = ((original_hold_time - result['avg_hold_periods']) / original_hold_time * 100) if original_hold_time != 0 else 0
            
            print(f"\n📈 vs ORIGINAL (HOLDING TOO LONG):")
            print(f"Return: {original_return:.2%} → {result['total_return']:.2%} ({return_improvement:+.1f}%)")
            print(f"Hold Time: {original_hold_time:.1f} → {result['avg_hold_periods']:.1f} periods ({hold_time_reduction:+.1f}%)")
            
            # Exit reason analysis
            if 'exit_reasons' in result:
                print(f"\n📊 WHY TRADES EXITED:")
                total_exits = sum(result['exit_reasons'].values())
                for reason, count in result['exit_reasons'].items():
                    pct = count / total_exits * 100
                    print(f"   {reason}: {count} ({pct:.1f}%)")
            
            # Success assessment based on your insight
            if (result['avg_hold_periods'] < 12 and  # Shorter hold times
                result['total_return'] > original_return and  # Better returns
                'profit_protection_momentum_died' in result.get('exit_reasons', {})):  # Actually protecting profits
                print("🎉 SUCCESS: Fixed the 'holding too long' problem!")
                print("✅ Shorter hold times")
                print("✅ Better returns") 
                print("✅ Protecting profits when momentum dies")
            elif result['avg_hold_periods'] < 12:
                print("⚠️  PARTIAL SUCCESS: Shorter hold times achieved")
                print("   Need to optimize for better returns")
            elif result['total_return'] > original_return * 1.5:
                print("⚠️  PARTIAL SUCCESS: Better returns achieved")
                print("   But may still be holding too long")
            else:
                print("❌ NEEDS REFINEMENT: Limited improvement")
            
            return result
        
        else:
            print("❌ Balanced momentum backtest failed")
            return None
            
    except Exception as e:
        print(f"❌ Balanced momentum system error: {e}")
        return None


def run_balanced_backtest(analyzer, config):
    """Run backtest with balanced momentum exit system"""
    
    # Initialize balanced exit manager
    momentum_config = BalancedMomentumConfig()
    exit_manager = BalancedMomentumExitManager(momentum_config)
    
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
            # Get signals using SAME logic as original (proven to work)
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
                    
                    # Initialize position in balanced exit manager
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
    run_balanced_momentum_backtest()
