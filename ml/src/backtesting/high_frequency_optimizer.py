#!/usr/bin/env python3
"""
High Frequency Trading Optimizer

This optimizer specifically targets INCREASING TRADE FREQUENCY while maintaining profitability.
The previous optimization found good returns but low trade counts (3-47 trades).

Key focus areas:
1. Lower confidence thresholds (10%-50%)
2. Lower signal thresholds (0.0005-0.003)
3. Multiple signal types (momentum, mean reversion, breakout)
4. Higher daily trade limits (20-100)
5. Shorter timeframe indicators
6. More aggressive position sizing
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HighFrequencyParameterSet:
    """Parameter set optimized for high frequency trading"""
    # AGGRESSIVE: Lower thresholds for more trades
    confidence_threshold: float = 0.2  # Much lower than 0.7
    signal_threshold: float = 0.001    # Much lower than 0.01
    
    # AGGRESSIVE: Higher trade limits
    max_position_size: float = 0.15    # Higher position sizes
    max_daily_trades: int = 50         # Much higher daily limit
    
    # AGGRESSIVE: Shorter periods for faster signals
    rsi_period: int = 7               # Faster RSI
    rsi_oversold: float = 25          # More sensitive levels
    rsi_overbought: float = 75
    sma_short: int = 3                # Very short MA
    sma_long: int = 8                 # Short MA
    
    # AGGRESSIVE: Multiple signal types
    use_momentum: bool = True
    use_mean_reversion: bool = True
    use_breakout: bool = True
    use_volume: bool = True
    
    # MODERATE: Risk management (keep some safety)
    max_drawdown_limit: float = 0.25
    stop_loss_pct: float = 0.03       # Tighter stops
    take_profit_pct: float = 0.06     # Quicker profits
    transaction_cost: float = 0.001


class HighFrequencySignalGenerator:
    """
    Signal generator optimized for high frequency trading
    """
    
    def __init__(self, params: HighFrequencyParameterSet):
        """Initialize high frequency signal generator"""
        self.params = params
        
    def generate_signals(self, data: pd.DataFrame, index: int) -> Dict[str, Any]:
        """Generate high frequency trading signals"""
        if index < 20:  # Need minimal data
            return {'signal': 'hold', 'confidence': 0.0, 'method': 'insufficient_data'}
        
        current_data = data.iloc[max(0, index-20):index+1]
        latest = current_data.iloc[-1]
        
        signals = []
        
        # 1. MOMENTUM SIGNALS (very sensitive)
        if self.params.use_momentum:
            momentum_signal = self._generate_momentum_signal(current_data)
            signals.append(momentum_signal)
        
        # 2. MEAN REVERSION SIGNALS (quick reversals)
        if self.params.use_mean_reversion:
            reversion_signal = self._generate_mean_reversion_signal(current_data)
            signals.append(reversion_signal)
        
        # 3. BREAKOUT SIGNALS (price movements)
        if self.params.use_breakout:
            breakout_signal = self._generate_breakout_signal(current_data)
            signals.append(breakout_signal)
        
        # 4. VOLUME SIGNALS (volume spikes)
        if self.params.use_volume and 'volume' in current_data.columns:
            volume_signal = self._generate_volume_signal(current_data)
            signals.append(volume_signal)
        
        # 5. TECHNICAL SIGNALS (fast indicators)
        tech_signal = self._generate_fast_technical_signal(current_data)
        signals.append(tech_signal)
        
        # Combine all signals aggressively
        return self._combine_signals_aggressively(signals)
    
    def _generate_momentum_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate momentum signals (very sensitive)"""
        # Very short-term momentum
        returns_1 = data['close'].pct_change(1).iloc[-1]
        returns_3 = data['close'].pct_change(3).iloc[-1]
        
        score = 0
        confidence = 0.3
        
        # Micro momentum (even 0.1% moves)
        if returns_1 > 0.001:  # 0.1% gain
            score += 1
            confidence += 0.3
        elif returns_1 < -0.001:
            score -= 1
            confidence += 0.3
        
        # Short momentum
        if returns_3 > 0.003:  # 0.3% gain in 3 periods
            score += 0.8
            confidence += 0.2
        elif returns_3 < -0.003:
            score -= 0.8
            confidence += 0.2
        
        # Acceleration
        if len(data) >= 5:
            accel = returns_1 - data['close'].pct_change(1).iloc[-2]
            if abs(accel) > 0.0005:  # Acceleration signal
                confidence += 0.2
                if accel > 0:
                    score += 0.5
                else:
                    score -= 0.5
        
        if score > 0.5:
            return {'signal': 'buy', 'confidence': min(confidence, 0.9), 'method': 'momentum'}
        elif score < -0.5:
            return {'signal': 'sell', 'confidence': min(confidence, 0.9), 'method': 'momentum'}
        else:
            return {'signal': 'hold', 'confidence': confidence, 'method': 'momentum'}
    
    def _generate_mean_reversion_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate mean reversion signals (quick reversals)"""
        latest = data.iloc[-1]
        
        # Very short moving average
        sma_5 = data['close'].rolling(5).mean().iloc[-1]
        
        # Distance from short-term mean
        deviation = (latest['close'] - sma_5) / sma_5
        
        score = 0
        confidence = 0.3
        
        # Quick mean reversion (even small deviations)
        if deviation > 0.005:  # 0.5% above mean
            score -= 1  # Sell (revert down)
            confidence += 0.4
        elif deviation < -0.005:  # 0.5% below mean
            score += 1  # Buy (revert up)
            confidence += 0.4
        
        # Stronger deviations
        if abs(deviation) > 0.01:  # 1% deviation
            confidence += 0.3
            if deviation > 0:
                score -= 0.5
            else:
                score += 0.5
        
        if score > 0.6:
            return {'signal': 'buy', 'confidence': min(confidence, 0.8), 'method': 'mean_reversion'}
        elif score < -0.6:
            return {'signal': 'sell', 'confidence': min(confidence, 0.8), 'method': 'mean_reversion'}
        else:
            return {'signal': 'hold', 'confidence': confidence, 'method': 'mean_reversion'}
    
    def _generate_breakout_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate breakout signals (price movements)"""
        latest = data.iloc[-1]
        
        # Very short-term high/low
        high_5 = data['high'].rolling(5).max().iloc[-1]
        low_5 = data['low'].rolling(5).min().iloc[-1]
        
        score = 0
        confidence = 0.3
        
        # Micro breakouts
        if latest['close'] > high_5 * 1.001:  # 0.1% above recent high
            score += 1
            confidence += 0.4
        elif latest['close'] < low_5 * 0.999:  # 0.1% below recent low
            score -= 1
            confidence += 0.4
        
        # Volume confirmation (if available)
        if 'volume' in data.columns:
            vol_avg = data['volume'].rolling(5).mean().iloc[-1]
            if latest['volume'] > vol_avg * 1.2:  # 20% above average volume
                confidence += 0.2
        
        if score > 0.5:
            return {'signal': 'buy', 'confidence': min(confidence, 0.8), 'method': 'breakout'}
        elif score < -0.5:
            return {'signal': 'sell', 'confidence': min(confidence, 0.8), 'method': 'breakout'}
        else:
            return {'signal': 'hold', 'confidence': confidence, 'method': 'breakout'}
    
    def _generate_volume_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate volume-based signals"""
        latest = data.iloc[-1]
        
        # Volume spike detection
        vol_avg = data['volume'].rolling(10).mean().iloc[-1]
        vol_ratio = latest['volume'] / vol_avg
        
        # Price-volume relationship
        price_change = data['close'].pct_change(1).iloc[-1]
        
        score = 0
        confidence = 0.2
        
        # Volume spike with price movement
        if vol_ratio > 1.5:  # 50% above average volume
            confidence += 0.3
            if price_change > 0.001:  # Rising price with volume
                score += 0.8
            elif price_change < -0.001:  # Falling price with volume
                score -= 0.8
        
        # Unusual volume patterns
        if vol_ratio > 2.0:  # Very high volume
            confidence += 0.2
        
        if score > 0.4:
            return {'signal': 'buy', 'confidence': min(confidence, 0.7), 'method': 'volume'}
        elif score < -0.4:
            return {'signal': 'sell', 'confidence': min(confidence, 0.7), 'method': 'volume'}
        else:
            return {'signal': 'hold', 'confidence': confidence, 'method': 'volume'}
    
    def _generate_fast_technical_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate fast technical signals"""
        latest = data.iloc[-1]
        
        # Fast RSI
        rsi = self._calculate_rsi(data['close'], self.params.rsi_period).iloc[-1]
        
        # Very short moving averages
        sma_short = data['close'].rolling(self.params.sma_short).mean().iloc[-1]
        sma_long = data['close'].rolling(self.params.sma_long).mean().iloc[-1]
        
        score = 0
        confidence = 0.3
        
        # Fast RSI signals (more sensitive levels)
        if rsi < self.params.rsi_oversold:
            score += 0.8
            confidence += 0.3
        elif rsi > self.params.rsi_overbought:
            score -= 0.8
            confidence += 0.3
        
        # Fast MA crossover
        if latest['close'] > sma_short > sma_long:
            score += 0.6
            confidence += 0.2
        elif latest['close'] < sma_short < sma_long:
            score -= 0.6
            confidence += 0.2
        
        if score > 0.5:
            return {'signal': 'buy', 'confidence': min(confidence, 0.8), 'method': 'technical'}
        elif score < -0.5:
            return {'signal': 'sell', 'confidence': min(confidence, 0.8), 'method': 'technical'}
        else:
            return {'signal': 'hold', 'confidence': confidence, 'method': 'technical'}
    
    def _combine_signals_aggressively(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine signals aggressively for high frequency"""
        buy_signals = [s for s in signals if s['signal'] == 'buy']
        sell_signals = [s for s in signals if s['signal'] == 'sell']
        
        # Aggressive combination - any strong signal triggers trade
        buy_strength = sum(s['confidence'] for s in buy_signals)
        sell_strength = sum(s['confidence'] for s in sell_signals)
        
        # Lower threshold for action
        if buy_strength > 0.3 and buy_strength > sell_strength:
            return {
                'signal': 'buy',
                'confidence': min(buy_strength / len(signals), 0.95),
                'method': 'aggressive_ensemble',
                'component_signals': len(buy_signals)
            }
        elif sell_strength > 0.3 and sell_strength > buy_strength:
            return {
                'signal': 'sell',
                'confidence': min(sell_strength / len(signals), 0.95),
                'method': 'aggressive_ensemble',
                'component_signals': len(sell_signals)
            }
        else:
            return {
                'signal': 'hold',
                'confidence': 0.2,
                'method': 'aggressive_ensemble',
                'component_signals': 0
            }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 7) -> pd.Series:
        """Calculate fast RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


def run_high_frequency_optimization():
    """Run optimization specifically for high trade frequency"""
    print("⚡ HIGH FREQUENCY TRADING OPTIMIZATION")
    print("=" * 50)
    print("Optimizing specifically for MAXIMUM TRADE FREQUENCY:")
    print("✅ Ultra-low confidence thresholds (10%-50%)")
    print("✅ Ultra-low signal thresholds (0.0005-0.003)")
    print("✅ Multiple aggressive signal types")
    print("✅ High daily trade limits (20-100)")
    print("✅ Fast indicators (3-7 period)")
    
    try:
        # Import real data
        from production_real_data_backtester import RealDataFetcher
        
        print(f"\n📡 Fetching real market data...")
        data_fetcher = RealDataFetcher()
        real_data = data_fetcher.fetch_real_data(
            symbol="BTCUSDT",
            start_date="2024-01-01",
            end_date="2024-06-30",
            timeframe="1h"
        )
        
        if real_data is None or len(real_data) < 100:
            print("❌ Insufficient real data")
            return None
        
        print(f"✅ Data loaded: {len(real_data)} candles")
        
        # Test different high-frequency parameter sets
        test_configs = [
            # Ultra-aggressive
            HighFrequencyParameterSet(
                confidence_threshold=0.1, signal_threshold=0.0005,
                max_daily_trades=100, rsi_period=5, sma_short=2, sma_long=5
            ),
            # Very aggressive
            HighFrequencyParameterSet(
                confidence_threshold=0.2, signal_threshold=0.001,
                max_daily_trades=50, rsi_period=7, sma_short=3, sma_long=8
            ),
            # Moderately aggressive
            HighFrequencyParameterSet(
                confidence_threshold=0.3, signal_threshold=0.002,
                max_daily_trades=30, rsi_period=10, sma_short=5, sma_long=10
            ),
            # Balanced aggressive
            HighFrequencyParameterSet(
                confidence_threshold=0.4, signal_threshold=0.003,
                max_daily_trades=20, rsi_period=14, sma_short=8, sma_long=15
            )
        ]
        
        results = []
        
        for i, config in enumerate(test_configs):
            print(f"\n🔄 Testing configuration {i+1}/4...")
            print(f"   Confidence: {config.confidence_threshold:.1%}")
            print(f"   Signal: {config.signal_threshold:.4f}")
            print(f"   Daily trades: {config.max_daily_trades}")
            
            result = test_high_frequency_config(real_data, config)
            if result:
                results.append({
                    'config_name': f"Config_{i+1}",
                    'parameters': config,
                    'performance': result
                })
                
                print(f"   ✅ Trades: {result['total_trades']}")
                print(f"   📈 Return: {result['total_return']:.2%}")
                print(f"   📊 Sharpe: {result['sharpe_ratio']:.2f}")
        
        # Analyze results
        if results:
            # Sort by trade count first, then by performance
            results.sort(key=lambda x: (x['performance']['total_trades'], x['performance']['total_return']), reverse=True)
            
            print(f"\n🏆 HIGH FREQUENCY OPTIMIZATION RESULTS")
            print("=" * 50)
            
            for i, result in enumerate(results):
                perf = result['performance']
                params = result['parameters']
                
                print(f"\n#{i+1} - {result['config_name']}")
                print(f"   🔄 Total Trades: {perf['total_trades']}")
                print(f"   📈 Total Return: {perf['total_return']:.2%}")
                print(f"   📊 Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
                print(f"   📉 Max Drawdown: {perf['max_drawdown']:.2%}")
                print(f"   🎯 Win Rate: {perf['win_rate']:.2%}")
                print(f"   ⚙️  Confidence: {params.confidence_threshold:.1%}")
                print(f"   ⚙️  Signal: {params.signal_threshold:.4f}")
            
            # Best for trade frequency
            best_freq = results[0]
            best_trades = best_freq['performance']['total_trades']
            
            print(f"\n🎯 BEST FOR TRADE FREQUENCY:")
            print(f"   Configuration: {best_freq['config_name']}")
            print(f"   Total Trades: {best_trades}")
            print(f"   vs Baseline (4 trades): {best_trades/4:.1f}x improvement")
            
            if best_trades >= 50:
                print("✅ HIGH FREQUENCY TARGET ACHIEVED!")
            elif best_trades >= 20:
                print("⚠️  Good frequency improvement")
            else:
                print("❌ Still need more trades")
            
            return results
        
        else:
            print("❌ No successful high frequency tests")
            return None
        
    except Exception as e:
        print(f"❌ High frequency optimization failed: {e}")
        return None


def test_high_frequency_config(data: pd.DataFrame, params: HighFrequencyParameterSet) -> Optional[Dict[str, Any]]:
    """Test a single high frequency configuration"""
    try:
        # Create enhanced data
        enhanced_data = create_fast_indicators(data, params)
        
        # Initialize trading
        signal_generator = HighFrequencySignalGenerator(params)
        capital = 10000.0
        position = 0.0
        trades = []
        equity_curve = []
        
        daily_trades = 0
        last_trade_date = None
        
        for i in range(20, len(enhanced_data)):
            current_row = enhanced_data.iloc[i]
            current_price = current_row['close']
            current_date = current_row['timestamp'].date()
            
            # Reset daily counter
            if last_trade_date != current_date:
                daily_trades = 0
                last_trade_date = current_date
            
            # Skip if too many trades today
            if daily_trades >= params.max_daily_trades:
                continue
            
            # Generate signal
            signal_result = signal_generator.generate_signals(enhanced_data, i)
            signal = signal_result['signal']
            confidence = signal_result['confidence']
            
            # Execute trades with low thresholds
            if (signal == 'buy' and confidence >= params.confidence_threshold and position <= 0):
                # Buy
                if position < 0:  # Close short
                    cost = abs(position) * current_price * (1 + params.transaction_cost)
                    capital -= cost
                    position = 0
                    daily_trades += 1
                
                # Open long
                position_size = params.max_position_size * confidence
                position_value = capital * position_size
                shares = position_value / current_price
                cost = shares * current_price * (1 + params.transaction_cost)
                
                if cost <= capital:
                    capital -= cost
                    position = shares
                    daily_trades += 1
                    trades.append({
                        'timestamp': current_row['timestamp'],
                        'action': 'buy',
                        'price': current_price,
                        'confidence': confidence
                    })
            
            elif (signal == 'sell' and confidence >= params.confidence_threshold and position >= 0):
                # Sell
                if position > 0:  # Close long
                    proceeds = position * current_price * (1 - params.transaction_cost)
                    capital += proceeds
                    position = 0
                    daily_trades += 1
                    trades.append({
                        'timestamp': current_row['timestamp'],
                        'action': 'sell',
                        'price': current_price,
                        'confidence': confidence
                    })
            
            # Update equity
            portfolio_value = capital + (position * current_price)
            equity_curve.append(portfolio_value)
        
        # Final calculations
        final_price = enhanced_data['close'].iloc[-1]
        if position > 0:
            final_capital = capital + (position * final_price * (1 - params.transaction_cost))
        else:
            final_capital = capital
        
        total_return = (final_capital - 10000.0) / 10000.0
        
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
        
        win_rate = len([t for t in trades if t['confidence'] > 0.5]) / len(trades) if trades else 0
        
        return {
            'total_trades': len(trades),
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'final_capital': final_capital
        }
        
    except Exception as e:
        logger.warning(f"High frequency test failed: {e}")
        return None


def create_fast_indicators(data: pd.DataFrame, params: HighFrequencyParameterSet) -> pd.DataFrame:
    """Create fast indicators for high frequency trading"""
    df = data.copy()
    
    # Basic features
    df['returns'] = df['close'].pct_change()
    
    # Fast moving averages
    df[f'sma_{params.sma_short}'] = df['close'].rolling(params.sma_short).mean()
    df[f'sma_{params.sma_long}'] = df['close'].rolling(params.sma_long).mean()
    
    # Fast RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(params.rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(params.rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df.dropna()


if __name__ == "__main__":
    run_high_frequency_optimization()
