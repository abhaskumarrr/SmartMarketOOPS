#!/usr/bin/env python3
"""
High-Profit Trending Markets System

PERFECT PARAMETERS DISCOVERED:
- Confidence Threshold: 65% (high-quality trades only)
- Signal Threshold: 0.3% (significant moves only)
- Position Size: 10% (optimal risk/reward)
- Max Hold Periods: 16 (4 hours maximum)

MARKET FILTERING (from research):
- ADX > 25 (strong trending markets only)
- Choppiness Index < 38.2 (avoid ranging markets)
- ATR Percentile > 70% (high volatility periods only)

STRATEGY:
1. Only trade in STRONG trending conditions
2. Use perfect parameters for high-profit trades
3. Completely avoid range-bound markets
4. Exit when trend dies or reaches zones
5. Focus on quality over quantity
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
class HighProfitConfig:
    """Configuration for high-profit trending system"""
    # Perfect parameters (from optimization)
    confidence_threshold: float = 0.65
    signal_threshold: float = 0.003
    max_position_size: float = 0.10
    max_hold_periods: int = 16
    
    # Market filtering (from research)
    min_adx_threshold: float = 25.0        # Strong trend only
    max_choppiness_index: float = 38.2     # Avoid ranging markets
    min_atr_percentile: float = 70.0       # High volatility only
    
    # Trading parameters
    symbol: str = "BTCUSDT"
    start_date: str = "2024-01-01"
    end_date: str = "2024-02-01"
    initial_capital: float = 10000.0
    max_daily_trades: int = 3              # Quality over quantity
    transaction_cost: float = 0.001
    
    # Risk management
    max_drawdown_limit: float = 0.10
    min_profit_factor: float = 2.0
    target_win_rate: float = 0.65


class TrendingMarketFilter:
    """Advanced market condition filter for trending markets only"""
    
    def __init__(self, config: HighProfitConfig):
        """Initialize trending market filter"""
        self.config = config
    
    def add_market_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add all market condition indicators"""
        
        enhanced_data = data.copy()
        
        # Calculate ADX (Average Directional Index)
        enhanced_data['adx'] = self._calculate_adx(data)
        
        # Calculate Choppiness Index
        enhanced_data['choppiness'] = self._calculate_choppiness_index(data)
        
        # Calculate ATR Percentile
        enhanced_data['atr_percentile'] = self._calculate_atr_percentile(data)
        
        # Determine if market is trending
        enhanced_data['is_trending'] = self._is_trending_market(enhanced_data)
        
        # Calculate trend strength
        enhanced_data['trend_strength'] = self._calculate_trend_strength(enhanced_data)
        
        return enhanced_data
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        
        # True Range
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm = data['high'].diff()
        minus_dm = data['low'].diff() * -1
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Smooth values
        tr_smooth = tr.rolling(period).mean()
        plus_dm_smooth = plus_dm.rolling(period).mean()
        minus_dm_smooth = minus_dm.rolling(period).mean()
        
        # Directional Indicators
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx.fillna(0)
    
    def _calculate_choppiness_index(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Choppiness Index"""
        
        # True Range
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Choppiness Index
        atr_sum = tr.rolling(period).sum()
        high_max = data['high'].rolling(period).max()
        low_min = data['low'].rolling(period).min()
        
        choppiness = 100 * np.log10(atr_sum / (high_max - low_min)) / np.log10(period)
        
        return choppiness.fillna(50)
    
    def _calculate_atr_percentile(self, data: pd.DataFrame, period: int = 14, lookback: int = 100) -> pd.Series:
        """Calculate ATR percentile"""
        
        # True Range and ATR
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Rolling percentile
        atr_percentile = atr.rolling(lookback).rank(pct=True) * 100
        
        return atr_percentile.fillna(50)
    
    def _is_trending_market(self, data: pd.DataFrame) -> pd.Series:
        """Determine if market is in trending condition"""
        
        trending_conditions = (
            (data['adx'] >= self.config.min_adx_threshold) &
            (data['choppiness'] <= self.config.max_choppiness_index) &
            (data['atr_percentile'] >= self.config.min_atr_percentile)
        )
        
        return trending_conditions
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate overall trend strength score"""
        
        # Normalize indicators to 0-1 scale
        adx_norm = np.clip((data['adx'] - 20) / 30, 0, 1)  # ADX 20-50 -> 0-1
        chop_norm = np.clip((60 - data['choppiness']) / 30, 0, 1)  # Chop 60-30 -> 0-1
        atr_norm = data['atr_percentile'] / 100  # Already 0-100 -> 0-1
        
        # Weighted trend strength
        trend_strength = (adx_norm * 0.4) + (chop_norm * 0.3) + (atr_norm * 0.3)
        
        return trend_strength


class HighProfitTradingSystem:
    """High-profit trading system for trending markets only"""
    
    def __init__(self, config: HighProfitConfig):
        """Initialize high-profit trading system"""
        self.config = config
        self.market_filter = TrendingMarketFilter(config)
        self.positions = {}
    
    def run_high_profit_backtest(self) -> Dict[str, Any]:
        """Run high-profit backtest on trending markets only"""
        
        print("🚀 HIGH-PROFIT TRENDING MARKETS SYSTEM")
        print("=" * 50)
        print("PERFECT PARAMETERS IMPLEMENTATION:")
        print(f"✅ Confidence Threshold: {self.config.confidence_threshold:.0%}")
        print(f"✅ Signal Threshold: {self.config.signal_threshold:.1%}")
        print(f"✅ Position Size: {self.config.max_position_size:.0%}")
        print(f"✅ Max Hold: {self.config.max_hold_periods} periods")
        print("MARKET FILTERING:")
        print(f"✅ ADX > {self.config.min_adx_threshold} (strong trend)")
        print(f"✅ Choppiness < {self.config.max_choppiness_index} (not ranging)")
        print(f"✅ ATR Percentile > {self.config.min_atr_percentile}% (high volatility)")
        
        try:
            # Load data and analyzer
            from multi_timeframe_system_corrected import MultiTimeframeAnalyzer, MultiTimeframeConfig
            
            base_config = MultiTimeframeConfig()
            base_config.confidence_threshold = self.config.confidence_threshold
            base_config.max_position_size = self.config.max_position_size
            base_config.max_daily_trades = self.config.max_daily_trades
            
            analyzer = MultiTimeframeAnalyzer(base_config)
            
            # Load data
            print(f"\n📡 Loading multi-timeframe data...")
            success = analyzer.load_all_timeframe_data(
                symbol=self.config.symbol,
                start_date=self.config.start_date,
                end_date=self.config.end_date
            )
            
            if not success:
                print("❌ Failed to load data")
                return None
            
            # Run high-profit backtest
            print(f"\n💰 Running high-profit backtest...")
            result = self._execute_high_profit_strategy(analyzer)
            
            if result:
                self._display_results(result)
                return result
            else:
                print("❌ High-profit backtest failed")
                return None
                
        except Exception as e:
            print(f"❌ High-profit system error: {e}")
            return None
    
    def _execute_high_profit_strategy(self, analyzer) -> Dict[str, Any]:
        """Execute the high-profit trading strategy"""
        
        # Get base data and add market conditions
        base_data = analyzer.timeframe_data['15m'].copy()
        enhanced_data = self.market_filter.add_market_indicators(base_data)
        
        # Filter for trending periods only
        trending_periods = enhanced_data[enhanced_data['is_trending'] == True]
        
        print(f"📊 Market Analysis:")
        print(f"   Total periods: {len(enhanced_data)}")
        print(f"   Trending periods: {len(trending_periods)} ({len(trending_periods)/len(enhanced_data):.1%})")
        print(f"   Range-bound periods filtered out: {len(enhanced_data) - len(trending_periods)}")
        
        if len(trending_periods) < 50:
            print("❌ Insufficient trending periods for trading")
            return None
        
        # Trading state
        capital = self.config.initial_capital
        position = 0.0
        trades = []
        equity_curve = []
        
        # Tracking
        trade_counter = 0
        active_trade_id = None
        daily_trades = 0
        last_trade_date = None
        periods_in_trade = 0
        
        # Market condition tracking
        trending_trades = 0
        ranging_periods_skipped = 0
        
        for i in range(50, len(enhanced_data)):
            current_row = enhanced_data.iloc[i]
            current_time = current_row['timestamp']
            current_price = current_row['close']
            current_date = current_time.date()
            
            # Reset daily counter
            if last_trade_date != current_date:
                daily_trades = 0
                last_trade_date = current_date
            
            # Skip if not trending market
            if not current_row['is_trending']:
                ranging_periods_skipped += 1
                if position != 0:
                    periods_in_trade += 1
                continue
            
            # Check for exit signals on active position
            if active_trade_id and position != 0:
                periods_in_trade += 1
                
                should_exit = False
                exit_reason = ""
                
                # 1. Maximum hold period (perfect parameter)
                if periods_in_trade >= self.config.max_hold_periods:
                    should_exit = True
                    exit_reason = "max_hold_period_reached"
                
                # 2. Trend strength weakening
                elif current_row['trend_strength'] < 0.3:
                    should_exit = True
                    exit_reason = "trend_strength_weakening"
                
                # 3. Market becoming choppy
                elif current_row['choppiness'] > self.config.max_choppiness_index:
                    should_exit = True
                    exit_reason = "market_becoming_choppy"
                
                # 4. ADX falling below threshold
                elif current_row['adx'] < self.config.min_adx_threshold:
                    should_exit = True
                    exit_reason = "adx_below_threshold"
                
                # Execute exit
                if should_exit:
                    if position > 0:
                        proceeds = position * current_price * (1 - self.config.transaction_cost)
                        capital += proceeds
                    else:
                        cost = abs(position) * current_price * (1 + self.config.transaction_cost)
                        capital -= cost
                    
                    # Calculate profit
                    entry_trade = next((t for t in trades if t.get('trade_id') == active_trade_id and t['action'] in ['buy', 'sell']), None)
                    if entry_trade:
                        entry_price = entry_trade['price']
                        if position > 0:
                            profit = (current_price - entry_price) / entry_price
                        else:
                            profit = (entry_price - current_price) / entry_price
                    else:
                        profit = 0
                    
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
            
            # Check for new entry signals (only in trending markets)
            if (position == 0 and 
                daily_trades < self.config.max_daily_trades and
                current_row['is_trending'] and
                current_row['trend_strength'] > 0.6):  # High trend strength only
                
                # Get multi-timeframe signals
                htf_bias, htf_confidence = analyzer.analyze_higher_timeframe_consensus(current_time)
                ltf_signal = analyzer.analyze_lower_timeframe_entry(current_time, htf_bias, htf_confidence)
                
                signal = ltf_signal['signal']
                confidence = ltf_signal['confidence']
                
                # Enter only with perfect parameters
                if (signal in ['buy', 'sell'] and 
                    confidence >= self.config.confidence_threshold and
                    htf_confidence > 0.5):
                    
                    trade_counter += 1
                    active_trade_id = f"trade_{trade_counter}"
                    
                    # Calculate position size
                    position_size = self.config.max_position_size * confidence
                    position_value = capital * position_size
                    shares = position_value / current_price
                    cost = shares * current_price * (1 + self.config.transaction_cost)
                    
                    if cost <= capital:
                        capital -= cost
                        position = shares if signal == 'buy' else -shares
                        daily_trades += 1
                        trending_trades += 1
                        periods_in_trade = 0
                        
                        trades.append({
                            'timestamp': current_time,
                            'action': signal,
                            'price': current_price,
                            'confidence': confidence,
                            'htf_confidence': htf_confidence,
                            'trend_strength': current_row['trend_strength'],
                            'adx': current_row['adx'],
                            'choppiness': current_row['choppiness'],
                            'trade_id': active_trade_id
                        })
            
            # Update equity curve
            portfolio_value = capital + (position * current_price)
            equity_curve.append(portfolio_value)
        
        # Final calculations
        final_price = enhanced_data['close'].iloc[-1]
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
            'trending_trades': trending_trades,
            'ranging_periods_skipped': ranging_periods_skipped,
            'exit_reasons': exit_reasons,
            'trades': trades
        }
    
    def _display_results(self, result: Dict[str, Any]):
        """Display high-profit system results"""
        
        print(f"\n🎯 HIGH-PROFIT SYSTEM RESULTS:")
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
        
        print(f"\n📊 MARKET FILTERING EFFECTIVENESS:")
        print(f"Trending market trades: {result['trending_trades']}")
        print(f"Range-bound periods skipped: {result['ranging_periods_skipped']}")
        print(f"Market filter efficiency: {result['ranging_periods_skipped']/(result['ranging_periods_skipped']+result['total_trades']*10):.1%}")
        
        # Exit analysis
        if 'exit_reasons' in result:
            print(f"\n📊 EXIT REASONS:")
            total_exits = sum(result['exit_reasons'].values())
            for reason, count in result['exit_reasons'].items():
                pct = count / total_exits * 100
                print(f"   {reason}: {count} ({pct:.1f}%)")
        
        # Performance assessment
        if (result['total_return'] > 0.02 and 
            result['profit_factor'] >= 2.0 and
            result['win_rate'] >= 0.6):
            print(f"\n🎉 EXCELLENT: High-profit system working perfectly!")
        elif result['total_return'] > 0.01 and result['profit_factor'] >= 1.5:
            print(f"\n✅ GOOD: System performing well")
        elif result['total_return'] > 0:
            print(f"\n⚠️  MODERATE: Positive but needs optimization")
        else:
            print(f"\n❌ POOR: System needs refinement")


def run_high_profit_trending_system():
    """Run the high-profit trending markets system"""
    
    config = HighProfitConfig()
    system = HighProfitTradingSystem(config)
    
    result = system.run_high_profit_backtest()
    
    return result


if __name__ == "__main__":
    print("🚀 Starting High-Profit Trending Markets System")
    print("Using perfect parameters and market filtering")
    
    result = run_high_profit_trending_system()
    
    if result:
        print(f"\n🎉 HIGH-PROFIT SYSTEM COMPLETED!")
        print("Perfect parameters implemented with market filtering")
    else:
        print(f"\n❌ System failed - check configuration")
