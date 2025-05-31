#!/usr/bin/env python3
"""
Optimized Dynamic System

BASED ON DYNAMIC RESEARCH FINDINGS:
- Dynamic momentum works (20.2% bullish, 14.5% bearish)
- Candle patterns detected (25.1% pin bars, 8.1% doji)
- Price zones working (119 zones, 89% zone-based exits)
- Hold times perfect (2.8 periods average)

OPTIMIZATION NEEDED:
- Relax entry criteria (too strict - only 9 trades)
- Improve confluence logic
- Better risk management
- Focus on quality setups

PINE SCRIPT APPROACH:
- Use dynamic filters (not hard timeframes)
- Multiple confirmation levels
- Adaptive position sizing
- Zone-based profit taking
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizedDynamicConfig:
    """Optimized configuration based on research findings"""
    # Relaxed entry parameters (from research)
    momentum_threshold: float = 0.25           # Lower from 0.3 (more opportunities)
    rsi_range_min: float = 25                  # Wider RSI range
    rsi_range_max: float = 75                  # Wider RSI range
    pattern_confirmation: bool = False         # Don't require patterns (optional)
    zone_confirmation: bool = False            # Don't require zones (optional)
    
    # Dynamic momentum parameters
    rsi_length: int = 14
    stoch_length: int = 14
    momentum_confirmation_periods: int = 2
    
    # Zone parameters
    pivot_left_bars: int = 5
    pivot_right_bars: int = 5
    zone_proximity_percent: float = 0.8        # Wider zone proximity
    
    # Trading parameters
    symbol: str = "BTCUSDT"
    start_date: str = "2024-01-01"
    end_date: str = "2024-02-01"
    initial_capital: float = 10000.0
    base_position_size: float = 0.05           # Smaller base size
    max_position_size: float = 0.12            # Higher max for strong signals
    max_daily_trades: int = 8                  # More opportunities
    transaction_cost: float = 0.001
    
    # Risk management
    stop_loss_percent: float = 0.02            # 2% stop loss
    take_profit_percent: float = 0.04          # 4% take profit
    max_hold_periods: int = 16                 # 4 hours max


class OptimizedDynamicSystem:
    """Optimized dynamic system based on research findings"""
    
    def __init__(self, config: OptimizedDynamicConfig):
        """Initialize optimized dynamic system"""
        self.config = config
    
    def run_optimized_backtest(self) -> Dict[str, Any]:
        """Run optimized dynamic backtest"""
        
        print("🎯 OPTIMIZED DYNAMIC SYSTEM")
        print("=" * 45)
        print("BASED ON RESEARCH FINDINGS:")
        print("✅ Relaxed entry criteria (more opportunities)")
        print("✅ Optional pattern/zone confirmation")
        print("✅ Dynamic momentum filters")
        print("✅ Adaptive position sizing")
        print("✅ Zone-based profit taking")
        
        try:
            # Load data
            from production_real_data_backtester import RealDataFetcher
            
            data_fetcher = RealDataFetcher()
            
            print(f"\n📡 Loading data...")
            data = data_fetcher.fetch_real_data(
                self.config.symbol, self.config.start_date, self.config.end_date, "15m"
            )
            
            if data is None or len(data) < 100:
                print("❌ Insufficient data")
                return None
            
            print(f"✅ Loaded {len(data)} data points")
            
            # Prepare data with optimized indicators
            enhanced_data = self._prepare_optimized_data(data)
            
            # Execute optimized strategy
            result = self._execute_optimized_strategy(enhanced_data)
            
            if result:
                self._display_optimized_results(result)
                return result
            else:
                print("❌ Optimized backtest failed")
                return None
                
        except Exception as e:
            print(f"❌ Optimized system error: {e}")
            return None
    
    def _prepare_optimized_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with optimized indicators"""
        
        print(f"🔄 Calculating optimized indicators...")
        
        # Calculate RSI
        data['rsi'] = self._calculate_rsi(data)
        
        # Calculate Stochastic
        data['stoch_k'], data['stoch_d'] = self._calculate_stochastic(data)
        
        # Calculate momentum direction and strength
        data['momentum_direction'] = self._calculate_momentum_direction(data)
        data['momentum_strength'] = self._calculate_momentum_strength(data)
        
        # Calculate simple candle patterns
        data['bullish_pattern'] = self._detect_bullish_patterns(data)
        data['bearish_pattern'] = self._detect_bearish_patterns(data)
        
        # Calculate dynamic zones
        data['near_support'] = self._calculate_support_proximity(data)
        data['near_resistance'] = self._calculate_resistance_proximity(data)
        
        # Calculate trend strength
        data['trend_strength'] = self._calculate_trend_strength(data)
        
        print(f"✅ Optimized indicators calculated")
        
        return data
    
    def _calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI"""
        close = data['close']
        delta = close.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_length).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)
    
    def _calculate_stochastic(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic"""
        length = self.config.stoch_length
        
        lowest_low = data['low'].rolling(window=length).min()
        highest_high = data['high'].rolling(window=length).max()
        
        k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()
        
        return k_percent.fillna(50), d_percent.fillna(50)
    
    def _calculate_momentum_direction(self, data: pd.DataFrame) -> pd.Series:
        """Calculate momentum direction"""
        rsi = data['rsi']
        stoch_k = data['stoch_k']
        
        # RSI momentum
        rsi_rising = rsi > rsi.shift(self.config.momentum_confirmation_periods)
        rsi_falling = rsi < rsi.shift(self.config.momentum_confirmation_periods)
        
        # Stochastic momentum
        stoch_rising = stoch_k > stoch_k.shift(self.config.momentum_confirmation_periods)
        stoch_falling = stoch_k < stoch_k.shift(self.config.momentum_confirmation_periods)
        
        # Combined momentum
        momentum_direction = pd.Series(0, index=rsi.index)
        
        # Bullish momentum
        bullish = (rsi_rising & stoch_rising) & (rsi > 45) & (stoch_k > 45)
        momentum_direction[bullish] = 1
        
        # Bearish momentum
        bearish = (rsi_falling & stoch_falling) & (rsi < 55) & (stoch_k < 55)
        momentum_direction[bearish] = -1
        
        return momentum_direction
    
    def _calculate_momentum_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate momentum strength"""
        rsi = data['rsi']
        stoch_k = data['stoch_k']
        
        # RSI strength
        rsi_strength = abs(rsi - 50) / 50
        
        # Stochastic strength
        stoch_strength = np.minimum(stoch_k / 100, (100 - stoch_k) / 100) * 2
        
        # Combined strength
        momentum_strength = (rsi_strength + stoch_strength) / 2
        
        return momentum_strength.fillna(0)
    
    def _detect_bullish_patterns(self, data: pd.DataFrame) -> pd.Series:
        """Detect bullish candle patterns"""
        
        # Simple bullish patterns
        green_candle = data['close'] > data['open']
        large_body = abs(data['close'] - data['open']) > (data['high'] - data['low']) * 0.6
        
        # Hammer-like pattern
        lower_wick = data['open'].where(data['open'] < data['close'], data['close']) - data['low']
        body_size = abs(data['close'] - data['open'])
        hammer_like = (lower_wick > body_size * 1.5) & green_candle
        
        # Bullish engulfing
        prev_red = data['close'].shift(1) < data['open'].shift(1)
        current_green = data['close'] > data['open']
        engulfs = (data['close'] > data['open'].shift(1)) & (data['open'] < data['close'].shift(1))
        bullish_engulfing = prev_red & current_green & engulfs
        
        # Combined bullish pattern
        bullish_pattern = (green_candle & large_body) | hammer_like | bullish_engulfing
        
        return bullish_pattern.astype(int)
    
    def _detect_bearish_patterns(self, data: pd.DataFrame) -> pd.Series:
        """Detect bearish candle patterns"""
        
        # Simple bearish patterns
        red_candle = data['close'] < data['open']
        large_body = abs(data['close'] - data['open']) > (data['high'] - data['low']) * 0.6
        
        # Shooting star-like pattern
        upper_wick = data['high'] - data['open'].where(data['open'] > data['close'], data['close'])
        body_size = abs(data['close'] - data['open'])
        shooting_star_like = (upper_wick > body_size * 1.5) & red_candle
        
        # Bearish engulfing
        prev_green = data['close'].shift(1) > data['open'].shift(1)
        current_red = data['close'] < data['open']
        engulfs = (data['close'] < data['open'].shift(1)) & (data['open'] > data['close'].shift(1))
        bearish_engulfing = prev_green & current_red & engulfs
        
        # Combined bearish pattern
        bearish_pattern = (red_candle & large_body) | shooting_star_like | bearish_engulfing
        
        return bearish_pattern.astype(int)
    
    def _calculate_support_proximity(self, data: pd.DataFrame) -> pd.Series:
        """Calculate proximity to support levels"""
        
        # Find recent lows
        low_window = 20
        recent_lows = data['low'].rolling(window=low_window).min()
        
        # Check proximity to recent lows
        proximity_threshold = self.config.zone_proximity_percent / 100
        near_support = abs(data['close'] - recent_lows) / data['close'] <= proximity_threshold
        
        return near_support.astype(int)
    
    def _calculate_resistance_proximity(self, data: pd.DataFrame) -> pd.Series:
        """Calculate proximity to resistance levels"""
        
        # Find recent highs
        high_window = 20
        recent_highs = data['high'].rolling(window=high_window).max()
        
        # Check proximity to recent highs
        proximity_threshold = self.config.zone_proximity_percent / 100
        near_resistance = abs(data['close'] - recent_highs) / data['close'] <= proximity_threshold
        
        return near_resistance.astype(int)
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate overall trend strength"""
        
        # Simple trend strength based on moving averages
        ma_short = data['close'].rolling(10).mean()
        ma_long = data['close'].rolling(20).mean()
        
        # Trend strength
        trend_strength = abs(ma_short - ma_long) / ma_long
        
        return trend_strength.fillna(0)
    
    def _execute_optimized_strategy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute optimized trading strategy"""
        
        # Trading state
        capital = self.config.initial_capital
        position = 0.0
        trades = []
        equity_curve = []
        
        # Tracking
        trade_counter = 0
        active_trade_id = None
        entry_price = 0
        periods_in_trade = 0
        daily_trades = 0
        last_date = None
        
        print(f"💰 Executing optimized strategy...")
        
        for i in range(20, len(data)):  # Start after indicators stabilize
            current_row = data.iloc[i]
            current_time = current_row['timestamp']
            current_price = current_row['close']
            current_date = current_time.date()
            
            # Reset daily counter
            if last_date != current_date:
                daily_trades = 0
                last_date = current_date
            
            # Check for exit signals
            if active_trade_id and position != 0:
                periods_in_trade += 1
                
                should_exit, exit_reason = self._check_optimized_exit(current_row, periods_in_trade, entry_price, position)
                
                if should_exit:
                    # Execute exit
                    if position > 0:
                        proceeds = position * current_price * (1 - self.config.transaction_cost)
                        capital += proceeds
                    else:
                        cost = abs(position) * current_price * (1 + self.config.transaction_cost)
                        capital -= cost
                    
                    # Calculate profit
                    if position > 0:
                        profit = (current_price - entry_price) / entry_price
                    else:
                        profit = (entry_price - current_price) / entry_price
                    
                    trades.append({
                        'timestamp': current_time,
                        'action': 'exit',
                        'price': current_price,
                        'reason': exit_reason,
                        'profit': profit,
                        'periods_held': periods_in_trade,
                        'trade_id': active_trade_id
                    })
                    
                    position = 0.0
                    active_trade_id = None
                    periods_in_trade = 0
                    entry_price = 0
            
            # Check for entry signals
            if position == 0 and daily_trades < self.config.max_daily_trades:
                entry_signal, signal_strength = self._check_optimized_entry(current_row)
                
                if entry_signal != 'none' and signal_strength > 0.3:
                    trade_counter += 1
                    active_trade_id = f"trade_{trade_counter}"
                    
                    # Adaptive position sizing
                    position_size = self.config.base_position_size + (signal_strength * (self.config.max_position_size - self.config.base_position_size))
                    position_value = capital * position_size
                    shares = position_value / current_price
                    cost = shares * current_price * (1 + self.config.transaction_cost)
                    
                    if cost <= capital:
                        capital -= cost
                        position = shares if entry_signal == 'buy' else -shares
                        entry_price = current_price
                        periods_in_trade = 0
                        daily_trades += 1
                        
                        trades.append({
                            'timestamp': current_time,
                            'action': entry_signal,
                            'price': current_price,
                            'signal_strength': signal_strength,
                            'position_size': position_size,
                            'momentum_direction': current_row['momentum_direction'],
                            'momentum_strength': current_row['momentum_strength'],
                            'trade_id': active_trade_id
                        })
            
            # Update equity curve
            portfolio_value = capital + (position * current_price)
            equity_curve.append(portfolio_value)
        
        # Final calculations
        final_price = data['close'].iloc[-1]
        if position != 0:
            if position > 0:
                final_capital = capital + (position * final_price * (1 - self.config.transaction_cost))
            else:
                final_capital = capital - (abs(position) * final_price * (1 + self.config.transaction_cost))
        else:
            final_capital = capital
        
        total_return = (final_capital - self.config.initial_capital) / self.config.initial_capital
        
        # Analyze results
        entry_trades = [t for t in trades if t['action'] in ['buy', 'sell']]
        exit_trades = [t for t in trades if t['action'] == 'exit']
        
        # Calculate metrics
        profits = [t.get('profit', 0) for t in exit_trades]
        hold_periods = [t.get('periods_held', 0) for t in exit_trades]
        winners = [p for p in profits if p > 0]
        losers = [p for p in profits if p < 0]
        
        # Performance metrics
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(24 * 365) if returns.std() > 0 else 0
            
            rolling_max = pd.Series(equity_curve).expanding().max()
            drawdown = (pd.Series(equity_curve) - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Exit reasons
        exit_reasons = {}
        for trade in exit_trades:
            reason = trade.get('reason', 'unknown')
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
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
            'exit_reasons': exit_reasons,
            'trades': trades
        }
    
    def _check_optimized_entry(self, current_row: pd.Series) -> Tuple[str, float]:
        """Check for optimized entry signals"""
        
        # Get current values
        momentum_direction = current_row['momentum_direction']
        momentum_strength = current_row['momentum_strength']
        rsi = current_row['rsi']
        stoch_k = current_row['stoch_k']
        bullish_pattern = current_row['bullish_pattern']
        bearish_pattern = current_row['bearish_pattern']
        near_support = current_row['near_support']
        near_resistance = current_row['near_resistance']
        trend_strength = current_row['trend_strength']
        
        # BULLISH ENTRY CONDITIONS (relaxed)
        bullish_momentum = momentum_direction == 1 and momentum_strength >= self.config.momentum_threshold
        bullish_rsi = self.config.rsi_range_min < rsi < self.config.rsi_range_max
        
        # BEARISH ENTRY CONDITIONS (relaxed)
        bearish_momentum = momentum_direction == -1 and momentum_strength >= self.config.momentum_threshold
        bearish_rsi = self.config.rsi_range_min < rsi < self.config.rsi_range_max
        
        # Calculate signal strength with multiple confirmation levels
        if bullish_momentum and bullish_rsi:
            signal_strength = momentum_strength
            
            # Add bonuses for additional confirmations
            if bullish_pattern:
                signal_strength += 0.2
            if near_support:
                signal_strength += 0.2
            if trend_strength > 0.01:
                signal_strength += 0.1
            if stoch_k < 30:  # Oversold
                signal_strength += 0.1
            
            return 'buy', min(signal_strength, 1.0)
        
        elif bearish_momentum and bearish_rsi:
            signal_strength = momentum_strength
            
            # Add bonuses for additional confirmations
            if bearish_pattern:
                signal_strength += 0.2
            if near_resistance:
                signal_strength += 0.2
            if trend_strength > 0.01:
                signal_strength += 0.1
            if stoch_k > 70:  # Overbought
                signal_strength += 0.1
            
            return 'sell', min(signal_strength, 1.0)
        
        return 'none', 0.0
    
    def _check_optimized_exit(self, current_row: pd.Series, periods_in_trade: int, 
                             entry_price: float, position: float) -> Tuple[bool, str]:
        """Check for optimized exit conditions"""
        
        current_price = current_row['close']
        momentum_direction = current_row['momentum_direction']
        momentum_strength = current_row['momentum_strength']
        rsi = current_row['rsi']
        near_support = current_row['near_support']
        near_resistance = current_row['near_resistance']
        
        # Calculate current profit
        if position > 0:
            current_profit = (current_price - entry_price) / entry_price
        else:
            current_profit = (entry_price - current_price) / entry_price
        
        # 1. STOP LOSS
        if current_profit <= -self.config.stop_loss_percent:
            return True, "stop_loss"
        
        # 2. TAKE PROFIT
        if current_profit >= self.config.take_profit_percent:
            return True, "take_profit"
        
        # 3. MAXIMUM HOLD PERIOD
        if periods_in_trade >= self.config.max_hold_periods:
            return True, "max_hold_period"
        
        # 4. MOMENTUM REVERSAL
        if position > 0 and momentum_direction == -1 and momentum_strength > 0.4:
            return True, "momentum_reversal_bearish"
        elif position < 0 and momentum_direction == 1 and momentum_strength > 0.4:
            return True, "momentum_reversal_bullish"
        
        # 5. RSI EXTREME CONDITIONS
        if position > 0 and rsi > 80:
            return True, "rsi_overbought"
        elif position < 0 and rsi < 20:
            return True, "rsi_oversold"
        
        # 6. ZONE-BASED EXITS
        if position > 0 and near_resistance:
            return True, "resistance_zone_reached"
        elif position < 0 and near_support:
            return True, "support_zone_reached"
        
        return False, "hold"
    
    def _display_optimized_results(self, result: Dict[str, Any]):
        """Display optimized results"""
        
        print(f"\n🎯 OPTIMIZED DYNAMIC RESULTS:")
        print("=" * 40)
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
        
        # Exit analysis
        if 'exit_reasons' in result:
            print(f"\n📊 OPTIMIZED EXIT ANALYSIS:")
            total_exits = sum(result['exit_reasons'].values())
            for reason, count in result['exit_reasons'].items():
                pct = count / total_exits * 100
                print(f"   {reason}: {count} ({pct:.1f}%)")
        
        # Compare with original dynamic system
        print(f"\n📈 vs ORIGINAL DYNAMIC SYSTEM:")
        print(f"Trades: 9 → {result['total_trades']} ({result['total_trades']/9:.1f}x)")
        print(f"Return: -69.78% → {result['total_return']:.2%}")
        print(f"Win Rate: 33.3% → {result['win_rate']:.1%}")
        
        # Performance assessment
        if (result['total_return'] > 0.01 and 
            result['profit_factor'] >= 1.5 and
            result['win_rate'] >= 0.55):
            print(f"\n🎉 EXCELLENT: Optimized system working perfectly!")
        elif result['total_return'] > 0 and result['profit_factor'] >= 1.2:
            print(f"\n✅ GOOD: Significant improvement achieved")
        elif result['total_return'] > -0.1:
            print(f"\n⚠️  MODERATE: Better but needs more optimization")
        else:
            print(f"\n❌ POOR: Still needs major refinement")


def run_optimized_dynamic_system():
    """Run the optimized dynamic system"""
    
    config = OptimizedDynamicConfig()
    system = OptimizedDynamicSystem(config)
    
    result = system.run_optimized_backtest()
    
    return result


if __name__ == "__main__":
    print("🎯 Starting Optimized Dynamic System")
    print("Based on research findings with relaxed entry criteria")
    
    result = run_optimized_dynamic_system()
    
    if result:
        print(f"\n🎉 OPTIMIZED DYNAMIC SYSTEM COMPLETED!")
        print("Pine Script best practices with improved entry logic!")
    else:
        print(f"\n❌ Optimization failed")
