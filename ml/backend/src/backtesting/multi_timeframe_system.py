#!/usr/bin/env python3
"""
Multi-Timeframe Trading System

This implements proper multi-timeframe analysis following the principle:
- HIGHER TIMEFRAMES (4H, 1D) = Trend Direction & Market Bias
- LOWER TIMEFRAMES (5m, 15m, 1h) = Precise Entry/Exit Execution

Key Concepts:
1. Top-Down Analysis: Start from higher timeframes for bias
2. Fractal Market Structure: Each timeframe reflects the larger structure
3. Higher TF Influence: Longer timeframes have more influence on price moves
4. Lower TF Precision: Shorter timeframes provide precise entry/exit points
5. Noise Filtering: Higher TF filters out lower TF noise

Strategy Flow:
1. Analyze 4H/1D for overall trend and bias
2. Wait for higher TF signal confirmation
3. Drop to 1H for intermediate structure
4. Use 15m/5m for precise entry timing
5. Execute trades aligned with higher TF bias
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

class MarketBias(Enum):
    """Market bias from higher timeframes"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    RANGING = "ranging"

class TrendStrength(Enum):
    """Trend strength classification"""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"

@dataclass
class TimeframeHierarchy:
    """Timeframe hierarchy for multi-timeframe analysis"""
    # Higher timeframes for bias and trend
    trend_timeframe: str = "1d"      # Primary trend (Daily)
    bias_timeframe: str = "4h"       # Market bias (4-hour)
    structure_timeframe: str = "1h"   # Market structure (1-hour)
    
    # Lower timeframes for execution
    entry_timeframe: str = "15m"     # Entry signals (15-minute)
    execution_timeframe: str = "5m"   # Precise execution (5-minute)

@dataclass
class MultiTimeframeConfig:
    """Configuration for multi-timeframe trading"""
    symbol: str = "BTCUSDT"
    start_date: str = "2024-01-01"
    end_date: str = "2024-02-01"
    initial_capital: float = 10000.0
    
    # Timeframe hierarchy
    timeframes: TimeframeHierarchy = TimeframeHierarchy()
    
    # Trading parameters
    max_position_size: float = 0.15
    risk_per_trade: float = 0.02  # 2% risk per trade
    
    # Multi-timeframe rules
    require_htf_alignment: bool = True  # Require higher TF alignment
    min_trend_strength: TrendStrength = TrendStrength.MODERATE
    allow_counter_trend: bool = False   # Allow counter-trend trades
    
    # Entry/Exit parameters
    entry_confidence_threshold: float = 0.6
    exit_confidence_threshold: float = 0.4
    
    # Risk management
    max_drawdown_limit: float = 0.15
    transaction_cost: float = 0.001


class MultiTimeframeAnalyzer:
    """
    Multi-timeframe market analysis engine
    """
    
    def __init__(self, config: MultiTimeframeConfig):
        """Initialize multi-timeframe analyzer"""
        self.config = config
        self.timeframe_data = {}  # Store data for each timeframe
        self.current_bias = MarketBias.NEUTRAL
        self.trend_strength = TrendStrength.WEAK
        
    def load_timeframe_data(self, symbol: str, start_date: str, end_date: str) -> bool:
        """Load data for all required timeframes"""
        try:
            from production_real_data_backtester import RealDataFetcher
            
            data_fetcher = RealDataFetcher()
            timeframes = [
                self.config.timeframes.trend_timeframe,
                self.config.timeframes.bias_timeframe,
                self.config.timeframes.structure_timeframe,
                self.config.timeframes.entry_timeframe,
                self.config.timeframes.execution_timeframe
            ]
            
            for tf in timeframes:
                logger.info(f"Loading {tf} data...")
                
                # Adjust date range for higher timeframes (need more history)
                if tf in ['1d', '4h']:
                    # Need more history for higher timeframes
                    extended_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
                    data = data_fetcher.fetch_real_data(symbol, extended_start, end_date, tf)
                else:
                    data = data_fetcher.fetch_real_data(symbol, start_date, end_date, tf)
                
                if data is not None and len(data) > 50:
                    # Create enhanced indicators for this timeframe
                    enhanced_data = self._create_timeframe_indicators(data, tf)
                    self.timeframe_data[tf] = enhanced_data
                    logger.info(f"✅ Loaded {len(enhanced_data)} {tf} candles")
                else:
                    logger.warning(f"❌ Failed to load {tf} data")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load timeframe data: {e}")
            return False
    
    def _create_timeframe_indicators(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Create indicators appropriate for each timeframe"""
        df = data.copy()
        
        # Basic price action
        df['returns'] = df['close'].pct_change()
        df['hl2'] = (df['high'] + df['low']) / 2
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Timeframe-specific parameters
        if timeframe in ['1d', '4h']:
            # Higher timeframes - trend and bias indicators
            fast_ma = 20
            slow_ma = 50
            rsi_period = 14
            atr_period = 14
        elif timeframe in ['1h']:
            # Structure timeframe
            fast_ma = 15
            slow_ma = 30
            rsi_period = 14
            atr_period = 14
        else:
            # Lower timeframes - execution indicators
            fast_ma = 10
            slow_ma = 20
            rsi_period = 10
            atr_period = 10
        
        # Moving averages for trend
        df['sma_fast'] = df['close'].rolling(fast_ma).mean()
        df['sma_slow'] = df['close'].rolling(slow_ma).mean()
        df['ema_fast'] = df['close'].ewm(span=fast_ma).mean()
        df['ema_slow'] = df['close'].ewm(span=slow_ma).mean()
        
        # Trend strength indicators
        df['ma_distance'] = (df['sma_fast'] - df['sma_slow']) / df['sma_slow']
        df['price_above_ma'] = (df['close'] > df['sma_fast']).astype(int)
        
        # RSI for momentum
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR for volatility
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(atr_period).mean()
        
        # Market structure levels
        df['swing_high'] = df['high'].rolling(5, center=True).max() == df['high']
        df['swing_low'] = df['low'].rolling(5, center=True).min() == df['low']
        
        # Higher timeframe specific indicators
        if timeframe in ['1d', '4h']:
            # Long-term trend indicators
            df['sma_200'] = df['close'].rolling(200).mean()
            df['long_term_trend'] = (df['close'] > df['sma_200']).astype(int)
            
            # Trend strength
            df['trend_strength'] = abs(df['ma_distance'])
            
        # Lower timeframe specific indicators
        elif timeframe in ['15m', '5m']:
            # Short-term momentum
            df['momentum_3'] = df['close'].pct_change(3)
            df['momentum_5'] = df['close'].pct_change(5)
            
            # Bollinger Bands for entry signals
            bb_period = 20
            df['bb_sma'] = df['close'].rolling(bb_period).mean()
            df['bb_std'] = df['close'].rolling(bb_period).std()
            df['bb_upper'] = df['bb_sma'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_sma'] - (df['bb_std'] * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df.dropna()
    
    def analyze_higher_timeframe_bias(self, current_time: datetime) -> Tuple[MarketBias, TrendStrength]:
        """Analyze higher timeframes for market bias and trend strength"""
        try:
            # Get current data for bias timeframe (4H)
            bias_data = self.timeframe_data[self.config.timeframes.bias_timeframe]
            trend_data = self.timeframe_data[self.config.timeframes.trend_timeframe]
            
            # Find current candle
            bias_current = self._get_current_candle(bias_data, current_time)
            trend_current = self._get_current_candle(trend_data, current_time)
            
            if bias_current is None or trend_current is None:
                return MarketBias.NEUTRAL, TrendStrength.WEAK
            
            # Analyze trend direction
            bias_signals = []
            trend_signals = []
            
            # 1. Moving Average Analysis
            if bias_current['close'] > bias_current['sma_fast'] > bias_current['sma_slow']:
                bias_signals.append(1)  # Bullish
            elif bias_current['close'] < bias_current['sma_fast'] < bias_current['sma_slow']:
                bias_signals.append(-1)  # Bearish
            else:
                bias_signals.append(0)  # Neutral
            
            # 2. Daily trend confirmation
            if 'long_term_trend' in trend_current:
                if trend_current['long_term_trend'] == 1:
                    trend_signals.append(1)
                else:
                    trend_signals.append(-1)
            
            # 3. Price momentum
            if bias_current['rsi'] > 60:
                bias_signals.append(1)
            elif bias_current['rsi'] < 40:
                bias_signals.append(-1)
            else:
                bias_signals.append(0)
            
            # 4. Trend strength analysis
            ma_distance = abs(bias_current['ma_distance'])
            if ma_distance > 0.02:  # 2% separation
                strength = TrendStrength.STRONG
            elif ma_distance > 0.01:  # 1% separation
                strength = TrendStrength.MODERATE
            else:
                strength = TrendStrength.WEAK
            
            # Determine overall bias
            total_bias = sum(bias_signals) + sum(trend_signals)
            
            if total_bias >= 2:
                bias = MarketBias.BULLISH
            elif total_bias <= -2:
                bias = MarketBias.BEARISH
            elif abs(total_bias) <= 1 and ma_distance < 0.005:
                bias = MarketBias.RANGING
            else:
                bias = MarketBias.NEUTRAL
            
            return bias, strength
            
        except Exception as e:
            logger.error(f"Higher timeframe analysis error: {e}")
            return MarketBias.NEUTRAL, TrendStrength.WEAK
    
    def analyze_lower_timeframe_entry(self, current_time: datetime, htf_bias: MarketBias) -> Dict[str, Any]:
        """Analyze lower timeframes for precise entry signals"""
        try:
            # Get entry timeframe data (15m)
            entry_data = self.timeframe_data[self.config.timeframes.entry_timeframe]
            exec_data = self.timeframe_data[self.config.timeframes.execution_timeframe]
            
            entry_current = self._get_current_candle(entry_data, current_time)
            exec_current = self._get_current_candle(exec_data, current_time)
            
            if entry_current is None or exec_current is None:
                return {'signal': 'hold', 'confidence': 0.0, 'method': 'no_data'}
            
            # Only trade in direction of higher timeframe bias
            if htf_bias == MarketBias.NEUTRAL or htf_bias == MarketBias.RANGING:
                return {'signal': 'hold', 'confidence': 0.0, 'method': 'no_htf_bias'}
            
            signals = []
            confidence = 0.3
            
            # 1. Entry timeframe momentum alignment
            if htf_bias == MarketBias.BULLISH:
                # Look for bullish entry signals
                if (entry_current['close'] > entry_current['ema_fast'] and 
                    entry_current['rsi'] > 45 and entry_current['rsi'] < 70):
                    signals.append('buy')
                    confidence += 0.3
                
                # Execution timeframe confirmation
                if 'momentum_3' in exec_current:
                    if exec_current['momentum_3'] > 0.001:  # Positive momentum
                        signals.append('buy')
                        confidence += 0.2
                
                # Bollinger Band entry
                if 'bb_position' in exec_current:
                    if exec_current['bb_position'] < 0.3:  # Near lower band (buy dip)
                        signals.append('buy')
                        confidence += 0.2
            
            elif htf_bias == MarketBias.BEARISH:
                # Look for bearish entry signals
                if (entry_current['close'] < entry_current['ema_fast'] and 
                    entry_current['rsi'] < 55 and entry_current['rsi'] > 30):
                    signals.append('sell')
                    confidence += 0.3
                
                # Execution timeframe confirmation
                if 'momentum_3' in exec_current:
                    if exec_current['momentum_3'] < -0.001:  # Negative momentum
                        signals.append('sell')
                        confidence += 0.2
                
                # Bollinger Band entry
                if 'bb_position' in exec_current:
                    if exec_current['bb_position'] > 0.7:  # Near upper band (sell rally)
                        signals.append('sell')
                        confidence += 0.2
            
            # 2. Structure alignment
            struct_data = self.timeframe_data[self.config.timeframes.structure_timeframe]
            struct_current = self._get_current_candle(struct_data, current_time)
            
            if struct_current is not None:
                # Structure should align with bias
                if htf_bias == MarketBias.BULLISH and struct_current['close'] > struct_current['sma_fast']:
                    confidence += 0.2
                elif htf_bias == MarketBias.BEARISH and struct_current['close'] < struct_current['sma_fast']:
                    confidence += 0.2
            
            # Determine final signal
            buy_signals = signals.count('buy')
            sell_signals = signals.count('sell')
            
            if buy_signals > sell_signals and buy_signals >= 2:
                return {
                    'signal': 'buy',
                    'confidence': min(confidence, 0.95),
                    'method': 'multi_timeframe_bullish',
                    'htf_bias': htf_bias.value,
                    'entry_signals': buy_signals
                }
            elif sell_signals > buy_signals and sell_signals >= 2:
                return {
                    'signal': 'sell',
                    'confidence': min(confidence, 0.95),
                    'method': 'multi_timeframe_bearish',
                    'htf_bias': htf_bias.value,
                    'entry_signals': sell_signals
                }
            else:
                return {
                    'signal': 'hold',
                    'confidence': confidence,
                    'method': 'insufficient_alignment',
                    'htf_bias': htf_bias.value,
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals
                }
                
        except Exception as e:
            logger.error(f"Lower timeframe entry analysis error: {e}")
            return {'signal': 'hold', 'confidence': 0.0, 'method': 'error'}
    
    def _get_current_candle(self, data: pd.DataFrame, current_time: datetime) -> Optional[pd.Series]:
        """Get the current candle for a given timeframe"""
        try:
            # Find the most recent candle before or at current_time
            mask = data['timestamp'] <= current_time
            if mask.any():
                return data[mask].iloc[-1]
            return None
        except Exception:
            return None
