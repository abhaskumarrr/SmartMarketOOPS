"""
Trading Strategies Module

This module provides the base class and implementations for trading strategies.
Strategies focus on Smart Money Concepts (SMC), Fair Value Gaps (FVGs), and liquidity analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
import logging
import torch
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    
    This class defines the interface that all strategies must implement.
    Derived classes should override the abstract methods to implement
    specific trading logic.
    """
    
    def __init__(self, name: str = "BaseStrategy", params: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name for identification
            params: Strategy-specific parameters
        """
        self.name = name
        self.params = params or {}
        self.data = None
        self.symbol = None
        self.current_index = 0
        
        # Initialize indicators
        self.indicators = {}
        
        logger.info(f"Strategy '{name}' initialized with parameters: {self.params}")
    
    def initialize(self, data: pd.DataFrame, symbol: str):
        """
        Initialize the strategy with data.
        
        Args:
            data: Historical price data
            symbol: Trading symbol
        """
        self.data = data
        self.symbol = symbol
        self.current_index = 0
        
        # Calculate indicators
        self._calculate_indicators()
        
        logger.info(f"Strategy '{self.name}' initialized with {len(data)} candles for {symbol}")
    
    def update_current_index(self, index: int):
        """
        Update the current data index.
        
        Args:
            index: Current data index for backtesting
        """
        self.current_index = index
    
    @abstractmethod
    def generate_signals(self, candle: Dict[str, Any], index: int) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on current market data.
        
        Args:
            candle: Current candle data
            index: Current position in the data
            
        Returns:
            List of signal dictionaries. Each signal should be a dictionary with:
            - action: 'enter' or 'exit'
            - direction: 'long' or 'short'
            - symbol: Trading symbol
            - size: Position size (optional)
            - reason: Reason for the signal (optional)
        """
        pass
    
    def _calculate_indicators(self):
        """
        Calculate technical indicators for the strategy.
        
        This method should be overridden by derived classes to
        calculate strategy-specific indicators.
        """
        pass
    
    def _get_indicator_value(self, indicator_name: str, index: Optional[int] = None) -> Any:
        """
        Get the value of an indicator at a specific index.
        
        Args:
            indicator_name: Name of the indicator
            index: Index to get the value for (default: current_index)
            
        Returns:
            Indicator value
        """
        if index is None:
            index = self.current_index
        
        if indicator_name not in self.indicators:
            logger.warning(f"Indicator '{indicator_name}' not found")
            return None
        
        indicator = self.indicators[indicator_name]
        if isinstance(indicator, pd.Series):
            return indicator.iloc[index] if index < len(indicator) else None
        elif isinstance(indicator, np.ndarray):
            return indicator[index] if index < len(indicator) else None
        else:
            return indicator
    
    def _get_price_data(self, field: str = 'close', offset: int = 0) -> float:
        """
        Get price data at a specific offset from current index.
        
        Args:
            field: Price field ('open', 'high', 'low', 'close', 'volume')
            offset: Offset from current index (0 = current, -1 = previous, etc.)
            
        Returns:
            Price value
        """
        index = self.current_index + offset
        if index < 0 or index >= len(self.data):
            return None
        
        return self.data.iloc[index][field]
    
    def _crossover(self, series1: Union[pd.Series, np.ndarray], series2: Union[pd.Series, np.ndarray], index: Optional[int] = None) -> bool:
        """
        Check if series1 crosses above series2 at the specified index.
        
        Args:
            series1: First series
            series2: Second series
            index: Index to check (default: current_index)
            
        Returns:
            True if a crossover occurred, False otherwise
        """
        if index is None:
            index = self.current_index
        
        # Need at least 2 values to check for crossover
        if index < 1:
            return False
        
        # Check if series1 was below series2 and is now above
        return (series1.iloc[index-1] < series2.iloc[index-1]) and (series1.iloc[index] > series2.iloc[index])
    
    def _crossunder(self, series1: Union[pd.Series, np.ndarray], series2: Union[pd.Series, np.ndarray], index: Optional[int] = None) -> bool:
        """
        Check if series1 crosses below series2 at the specified index.
        
        Args:
            series1: First series
            series2: Second series
            index: Index to check (default: current_index)
            
        Returns:
            True if a crossunder occurred, False otherwise
        """
        if index is None:
            index = self.current_index
        
        # Need at least 2 values to check for crossunder
        if index < 1:
            return False
        
        # Check if series1 was above series2 and is now below
        return (series1.iloc[index-1] > series2.iloc[index-1]) and (series1.iloc[index] < series2.iloc[index])
    
    def _identify_swing_high(self, lookback: int = 5, lookforward: int = 5) -> Optional[float]:
        """
        Identify a swing high at the current index.
        
        Args:
            lookback: Number of candles to look back
            lookforward: Number of candles to look forward
            
        Returns:
            Price of swing high if found, None otherwise
        """
        # We need enough data points both before and after the current index
        if self.current_index < lookback or self.current_index + lookforward >= len(self.data):
            return None
        
        high_price = self._get_price_data('high')
        
        # Check if current high is higher than all lookback and lookforward highs
        for i in range(1, lookback + 1):
            if high_price <= self._get_price_data('high', -i):
                return None
        
        for i in range(1, lookforward + 1):
            if high_price <= self._get_price_data('high', i):
                return None
        
        return high_price
    
    def _identify_swing_low(self, lookback: int = 5, lookforward: int = 5) -> Optional[float]:
        """
        Identify a swing low at the current index.
        
        Args:
            lookback: Number of candles to look back
            lookforward: Number of candles to look forward
            
        Returns:
            Price of swing low if found, None otherwise
        """
        # We need enough data points both before and after the current index
        if self.current_index < lookback or self.current_index + lookforward >= len(self.data):
            return None
        
        low_price = self._get_price_data('low')
        
        # Check if current low is lower than all lookback and lookforward lows
        for i in range(1, lookback + 1):
            if low_price >= self._get_price_data('low', -i):
                return None
        
        for i in range(1, lookforward + 1):
            if low_price >= self._get_price_data('low', i):
                return None
        
        return low_price
    
    # ==========================================================================
    # Smart Money Concepts Methods
    # ==========================================================================
    
    def _detect_bos(self, direction: str = 'bullish', lookback: int = 20) -> bool:
        """
        Detect a Break of Structure (BOS).
        
        Args:
            direction: 'bullish' or 'bearish'
            lookback: Number of candles to look back for structure
            
        Returns:
            True if BOS detected, False otherwise
        """
        if self.current_index < lookback:
            return False
        
        if direction == 'bullish':
            # Find the most recent swing high in the lookback period
            swing_highs = []
            for i in range(1, lookback):
                idx = self.current_index - i
                high = self._identify_swing_high(lookback=3, lookforward=3)
                if high is not None:
                    swing_highs.append((idx, high))
            
            if not swing_highs:
                return False
            
            # Sort by price (highest first)
            swing_highs.sort(key=lambda x: x[1], reverse=True)
            
            # Check if current high breaks the highest swing high
            current_high = self._get_price_data('high')
            return current_high > swing_highs[0][1]
        
        elif direction == 'bearish':
            # Find the most recent swing low in the lookback period
            swing_lows = []
            for i in range(1, lookback):
                idx = self.current_index - i
                low = self._identify_swing_low(lookback=3, lookforward=3)
                if low is not None:
                    swing_lows.append((idx, low))
            
            if not swing_lows:
                return False
            
            # Sort by price (lowest first)
            swing_lows.sort(key=lambda x: x[1])
            
            # Check if current low breaks the lowest swing low
            current_low = self._get_price_data('low')
            return current_low < swing_lows[0][1]
        
        return False
    
    def _detect_choch(self, direction: str = 'bullish', lookback: int = 20) -> bool:
        """
        Detect a Change of Character (CHoCH).
        
        Args:
            direction: 'bullish' or 'bearish'
            lookback: Number of candles to look back for structure
            
        Returns:
            True if CHoCH detected, False otherwise
        """
        if self.current_index < lookback:
            return False
        
        if direction == 'bullish':
            # In an uptrend, a CHoCH occurs when we break below a significant higher low
            
            # Find the higher lows in the lookback period
            higher_lows = []
            prev_low = float('inf')
            for i in range(lookback, 0, -1):
                idx = self.current_index - i
                if idx < 0:
                    continue
                
                # Check if this is a swing low
                low = self._identify_swing_low(lookback=3, lookforward=3)
                if low is not None and low > prev_low:
                    higher_lows.append((idx, low))
                    prev_low = low
            
            if not higher_lows:
                return False
            
            # Check if current low breaks the most recent higher low
            current_low = self._get_price_data('low')
            return current_low < higher_lows[-1][1]
        
        elif direction == 'bearish':
            # In a downtrend, a CHoCH occurs when we break above a significant lower high
            
            # Find the lower highs in the lookback period
            lower_highs = []
            prev_high = float('-inf')
            for i in range(lookback, 0, -1):
                idx = self.current_index - i
                if idx < 0:
                    continue
                
                # Check if this is a swing high
                high = self._identify_swing_high(lookback=3, lookforward=3)
                if high is not None and high < prev_high:
                    lower_highs.append((idx, high))
                    prev_high = high
            
            if not lower_highs:
                return False
            
            # Check if current high breaks the most recent lower high
            current_high = self._get_price_data('high')
            return current_high > lower_highs[-1][1]
        
        return False
    
    def _detect_fair_value_gap(self, direction: str = 'bullish', gap_threshold: float = 0.001) -> Dict[str, Any]:
        """
        Detect a Fair Value Gap (FVG).
        
        Args:
            direction: 'bullish' or 'bearish'
            gap_threshold: Minimum gap size as a percentage
            
        Returns:
            Dictionary with FVG details or None if not found
        """
        if self.current_index < 2:
            return None
        
        # Get the three candles needed to identify FVG
        candle_1 = {
            'open': self._get_price_data('open', -2),
            'high': self._get_price_data('high', -2),
            'low': self._get_price_data('low', -2),
            'close': self._get_price_data('close', -2)
        }
        
        candle_2 = {
            'open': self._get_price_data('open', -1),
            'high': self._get_price_data('high', -1),
            'low': self._get_price_data('low', -1),
            'close': self._get_price_data('close', -1)
        }
        
        candle_3 = {
            'open': self._get_price_data('open', 0),
            'high': self._get_price_data('high', 0),
            'low': self._get_price_data('low', 0),
            'close': self._get_price_data('close', 0)
        }
        
        if direction == 'bullish':
            # A bullish FVG occurs when the low of candle 3 is greater than the high of candle 1
            # creating an empty space (imbalance) in the price chart
            if candle_3['low'] > candle_1['high']:
                gap_size = candle_3['low'] - candle_1['high']
                gap_percentage = gap_size / candle_1['high']
                
                if gap_percentage >= gap_threshold:
                    return {
                        'type': 'bullish',
                        'top': candle_3['low'],
                        'bottom': candle_1['high'],
                        'size': gap_size,
                        'percentage': gap_percentage,
                        'candle_1_idx': self.current_index - 2,
                        'candle_3_idx': self.current_index,
                        'filled': False
                    }
        
        elif direction == 'bearish':
            # A bearish FVG occurs when the high of candle 3 is less than the low of candle 1
            # creating an empty space (imbalance) in the price chart
            if candle_3['high'] < candle_1['low']:
                gap_size = candle_1['low'] - candle_3['high']
                gap_percentage = gap_size / candle_1['low']
                
                if gap_percentage >= gap_threshold:
                    return {
                        'type': 'bearish',
                        'top': candle_1['low'],
                        'bottom': candle_3['high'],
                        'size': gap_size,
                        'percentage': gap_percentage,
                        'candle_1_idx': self.current_index - 2,
                        'candle_3_idx': self.current_index,
                        'filled': False
                    }
        
        return None
    
    def _detect_order_block(self, direction: str = 'bullish', lookback: int = 10) -> Dict[str, Any]:
        """
        Detect an Order Block (OB).
        
        Args:
            direction: 'bullish' or 'bearish'
            lookback: Number of candles to look back
            
        Returns:
            Dictionary with Order Block details or None if not found
        """
        if self.current_index < lookback:
            return None
        
        if direction == 'bullish':
            # For a bullish Order Block, we look for a bearish candle
            # before a significant move up (BOS)
            
            # First, identify a bullish Break of Structure
            bos_detected = False
            bos_index = None
            
            for i in range(1, lookback):
                idx = self.current_index - i
                if idx < 0:
                    continue
                
                # Update current index temporarily
                prev_index = self.current_index
                self.update_current_index(idx)
                bos = self._detect_bos('bullish')
                self.update_current_index(prev_index)
                
                if bos:
                    bos_detected = True
                    bos_index = idx
                    break
            
            if not bos_detected or bos_index is None or bos_index <= 0:
                return None
            
            # Find the last bearish candle before the BOS
            for i in range(1, min(5, bos_index + 1)):
                idx = bos_index - i
                if idx < 0:
                    break
                
                candle_open = self.data.iloc[idx]['open']
                candle_close = self.data.iloc[idx]['close']
                
                # Check if bearish candle
                if candle_close < candle_open:
                    return {
                        'type': 'bullish',
                        'top': candle_open,
                        'bottom': candle_close,
                        'high': self.data.iloc[idx]['high'],
                        'low': self.data.iloc[idx]['low'],
                        'index': idx,
                        'bos_index': bos_index
                    }
        
        elif direction == 'bearish':
            # For a bearish Order Block, we look for a bullish candle
            # before a significant move down (BOS)
            
            # First, identify a bearish Break of Structure
            bos_detected = False
            bos_index = None
            
            for i in range(1, lookback):
                idx = self.current_index - i
                if idx < 0:
                    continue
                
                # Update current index temporarily
                prev_index = self.current_index
                self.update_current_index(idx)
                bos = self._detect_bos('bearish')
                self.update_current_index(prev_index)
                
                if bos:
                    bos_detected = True
                    bos_index = idx
                    break
            
            if not bos_detected or bos_index is None or bos_index <= 0:
                return None
            
            # Find the last bullish candle before the BOS
            for i in range(1, min(5, bos_index + 1)):
                idx = bos_index - i
                if idx < 0:
                    break
                
                candle_open = self.data.iloc[idx]['open']
                candle_close = self.data.iloc[idx]['close']
                
                # Check if bullish candle
                if candle_close > candle_open:
                    return {
                        'type': 'bearish',
                        'top': candle_close,
                        'bottom': candle_open,
                        'high': self.data.iloc[idx]['high'],
                        'low': self.data.iloc[idx]['low'],
                        'index': idx,
                        'bos_index': bos_index
                    }
        
        return None
    
    def _detect_liquidity(self, direction: str = 'buy', lookback: int = 20, threshold: float = 0.0001) -> Dict[str, Any]:
        """
        Detect liquidity areas (equal highs/lows).
        
        Args:
            direction: 'buy' or 'sell'
            lookback: Number of candles to look back
            threshold: Price difference threshold for equality
            
        Returns:
            Dictionary with liquidity details or None if not found
        """
        if self.current_index < lookback:
            return None
        
        if direction == 'buy':
            # Buy-side liquidity is located above equal highs
            # Identify equal highs within the lookback period
            highs = [self.data.iloc[self.current_index - i]['high'] for i in range(lookback) if self.current_index - i >= 0]
            
            # Find clusters of equal highs
            clusters = []
            for i in range(len(highs)):
                cluster = []
                for j in range(len(highs)):
                    if i != j and abs(highs[i] - highs[j]) / highs[i] < threshold:
                        cluster.append(j)
                
                if len(cluster) >= 2:  # Need at least 3 equal highs (including i)
                    clusters.append((i, highs[i], cluster))
            
            if not clusters:
                return None
            
            # Sort clusters by size (largest first)
            clusters.sort(key=lambda x: len(x[2]), reverse=True)
            
            # Get the largest cluster
            _, price, _ = clusters[0]
            
            return {
                'type': 'buy',
                'price': price,
                'liquidity_level': price * (1 + 0.001)  # Slightly above equal highs
            }
        
        elif direction == 'sell':
            # Sell-side liquidity is located below equal lows
            # Identify equal lows within the lookback period
            lows = [self.data.iloc[self.current_index - i]['low'] for i in range(lookback) if self.current_index - i >= 0]
            
            # Find clusters of equal lows
            clusters = []
            for i in range(len(lows)):
                cluster = []
                for j in range(len(lows)):
                    if i != j and abs(lows[i] - lows[j]) / lows[i] < threshold:
                        cluster.append(j)
                
                if len(cluster) >= 2:  # Need at least 3 equal lows (including i)
                    clusters.append((i, lows[i], cluster))
            
            if not clusters:
                return None
            
            # Sort clusters by size (largest first)
            clusters.sort(key=lambda x: len(x[2]), reverse=True)
            
            # Get the largest cluster
            _, price, _ = clusters[0]
            
            return {
                'type': 'sell',
                'price': price,
                'liquidity_level': price * (1 - 0.001)  # Slightly below equal lows
            }
        
        return None


# Example strategy implementation using SMC concepts
class SmcBasedStrategy(BaseStrategy):
    """
    A strategy implementing Smart Money Concepts.
    
    This strategy looks for:
    1. Break of Structure (BOS)
    2. Fair Value Gaps (FVG)
    3. Order Blocks (OB)
    4. Liquidity areas
    
    It generates signals when price retraces to an FVG
    after a validated BOS/CHoCH event.
    """
    
    def __init__(self, name: str = "SMC Strategy", params: Optional[Dict[str, Any]] = None):
        """
        Initialize the SMC strategy.
        
        Args:
            name: Strategy name
            params: Strategy parameters
        """
        super().__init__(name, params)
        
        # Set default parameters if not provided
        self.params.setdefault('fvg_threshold', 0.001)  # Minimum FVG size
        self.params.setdefault('ob_lookback', 15)  # Lookback for Order Blocks
        self.params.setdefault('liquidity_lookback', 30)  # Lookback for liquidity
        self.params.setdefault('liquidity_threshold', 0.0002)  # Threshold for equal highs/lows
        
        # Track identified SMC components
        self.active_fvgs = []  # List of active (unfilled) FVGs
        self.active_order_blocks = []  # List of active Order Blocks
        self.liquidity_areas = []  # List of identified liquidity areas
    
    def _calculate_indicators(self):
        """Calculate indicators for the SMC strategy"""
        # Calculate moving averages for trend identification
        self.indicators['sma20'] = self.data['close'].rolling(window=20).mean()
        self.indicators['sma50'] = self.data['close'].rolling(window=50).mean()
        
        # Calculate Bollinger Bands for volatility and trend strength
        sma20 = self.indicators['sma20']
        std20 = self.data['close'].rolling(window=20).std()
        self.indicators['bb_upper'] = sma20 + (std20 * 2)
        self.indicators['bb_lower'] = sma20 - (std20 * 2)
        
        # Calculate RSI for momentum
        delta = self.data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        self.indicators['rsi'] = 100 - (100 / (1 + rs))
    
    def generate_signals(self, candle: Dict[str, Any], index: int) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on SMC concepts.
        
        Args:
            candle: Current candle data
            index: Current position in the data
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        # Update FVGs status (check if they've been filled)
        self._update_fvgs_status()
        
        # Detect new FVGs
        bullish_fvg = self._detect_fair_value_gap('bullish', self.params['fvg_threshold'])
        if bullish_fvg:
            self.active_fvgs.append(bullish_fvg)
            logger.debug(f"Detected bullish FVG at index {index}: {bullish_fvg}")
        
        bearish_fvg = self._detect_fair_value_gap('bearish', self.params['fvg_threshold'])
        if bearish_fvg:
            self.active_fvgs.append(bearish_fvg)
            logger.debug(f"Detected bearish FVG at index {index}: {bearish_fvg}")
        
        # Detect Order Blocks
        bullish_ob = self._detect_order_block('bullish', self.params['ob_lookback'])
        if bullish_ob:
            self.active_order_blocks.append(bullish_ob)
            logger.debug(f"Detected bullish Order Block at index {index}: {bullish_ob}")
        
        bearish_ob = self._detect_order_block('bearish', self.params['ob_lookback'])
        if bearish_ob:
            self.active_order_blocks.append(bearish_ob)
            logger.debug(f"Detected bearish Order Block at index {index}: {bearish_ob}")
        
        # Detect liquidity areas
        buy_liquidity = self._detect_liquidity('buy', self.params['liquidity_lookback'], self.params['liquidity_threshold'])
        if buy_liquidity:
            self.liquidity_areas.append(buy_liquidity)
            logger.debug(f"Detected buy-side liquidity at index {index}: {buy_liquidity}")
        
        sell_liquidity = self._detect_liquidity('sell', self.params['liquidity_lookback'], self.params['liquidity_threshold'])
        if sell_liquidity:
            self.liquidity_areas.append(sell_liquidity)
            logger.debug(f"Detected sell-side liquidity at index {index}: {sell_liquidity}")
        
        # Generate entry signals when price retraces to an FVG
        entry_signal = self._check_entry_conditions()
        if entry_signal:
            signals.append(entry_signal)
        
        # Generate exit signals based on hitting liquidity or contrary signals
        exit_signal = self._check_exit_conditions()
        if exit_signal:
            signals.append(exit_signal)
        
        return signals
    
    def _update_fvgs_status(self):
        """Update the status of active FVGs (check if filled)"""
        current_high = self._get_price_data('high')
        current_low = self._get_price_data('low')
        
        for fvg in self.active_fvgs:
            if fvg['type'] == 'bullish' and current_low <= fvg['bottom']:
                fvg['filled'] = True
            elif fvg['type'] == 'bearish' and current_high >= fvg['top']:
                fvg['filled'] = True
    
    def _check_entry_conditions(self) -> Optional[Dict[str, Any]]:
        """
        Check for entry conditions based on SMC concepts.
        
        Returns:
            Signal dictionary or None
        """
        # Get current price and indicators
        close = self._get_price_data('close')
        high = self._get_price_data('high')
        low = self._get_price_data('low')
        
        # Check for trend using moving averages
        sma20 = self._get_indicator_value('sma20')
        sma50 = self._get_indicator_value('sma50')
        
        # Determine overall trend
        uptrend = sma20 > sma50 if sma20 is not None and sma50 is not None else None
        
        # Entry logic for bullish setup
        if uptrend:
            # Look for active (unfilled) bullish FVGs that price is currently in
            for fvg in self.active_fvgs:
                if fvg['type'] == 'bullish' and not fvg['filled']:
                    if low <= fvg['top'] and high >= fvg['bottom']:
                        # Price is currently in the FVG
                        # Check if we also have a bullish Order Block nearby
                        for ob in self.active_order_blocks:
                            if ob['type'] == 'bullish' and abs(low - ob['bottom']) / ob['bottom'] < 0.01:
                                return {
                                    'action': 'enter',
                                    'direction': 'long',
                                    'symbol': self.symbol,
                                    'reason': f"Bullish FVG + Order Block at {self.current_index}"
                                }
        
        # Entry logic for bearish setup
        if not uptrend:
            # Look for active (unfilled) bearish FVGs that price is currently in
            for fvg in self.active_fvgs:
                if fvg['type'] == 'bearish' and not fvg['filled']:
                    if high >= fvg['bottom'] and low <= fvg['top']:
                        # Price is currently in the FVG
                        # Check if we also have a bearish Order Block nearby
                        for ob in self.active_order_blocks:
                            if ob['type'] == 'bearish' and abs(high - ob['top']) / ob['top'] < 0.01:
                                return {
                                    'action': 'enter',
                                    'direction': 'short',
                                    'symbol': self.symbol,
                                    'reason': f"Bearish FVG + Order Block at {self.current_index}"
                                }
        
        return None
    
    def _check_exit_conditions(self) -> Optional[Dict[str, Any]]:
        """
        Check for exit conditions based on SMC concepts.
        
        Returns:
            Signal dictionary or None
        """
        # Get current price
        close = self._get_price_data('close')
        high = self._get_price_data('high')
        low = self._get_price_data('low')
        
        # Exit when price approaches liquidity areas
        for liquidity in self.liquidity_areas:
            if liquidity['type'] == 'buy' and high >= liquidity['price'] * 0.998:
                return {
                    'action': 'exit',
                    'symbol': self.symbol,
                    'reason': f"Approaching buy-side liquidity at {liquidity['price']}"
                }
            elif liquidity['type'] == 'sell' and low <= liquidity['price'] * 1.002:
                return {
                    'action': 'exit',
                    'symbol': self.symbol,
                    'reason': f"Approaching sell-side liquidity at {liquidity['price']}"
                }
        
        # Exit when a contrary Order Block is detected
        for ob in self.active_order_blocks:
            if ob['type'] == 'bearish' and high >= ob['bottom']:
                return {
                    'action': 'exit',
                    'symbol': self.symbol,
                    'reason': f"Hit bearish Order Block at {ob['bottom']}"
                }
            elif ob['type'] == 'bullish' and low <= ob['top']:
                return {
                    'action': 'exit',
                    'symbol': self.symbol,
                    'reason': f"Hit bullish Order Block at {ob['top']}"
                }
        
        return None 


class MLModelStrategy(BaseStrategy):
    """
    MLModelStrategy integrates any trained ML model (PyTorch) as a backtest strategy.
    It loads a model checkpoint and optional preprocessor, and generates signals from model predictions.
    """
    def __init__(self, model_class, model_kwargs, model_checkpoint, preprocessor_path=None, device='cpu', threshold=0.5, name="MLModelStrategy", params=None):
        super().__init__(name=name, params=params)
        self.model = model_class(**model_kwargs)
        checkpoint = torch.load(model_checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.threshold = threshold
        self.preprocessor = joblib.load(preprocessor_path) if preprocessor_path else None

    def initialize(self, data: pd.DataFrame, symbol: str):
        self.data = data
        self.symbol = symbol
        self.current_index = 0
        # No indicators needed for ML model

    def generate_signals(self, candle: Dict[str, Any], index: int) -> List[Dict[str, Any]]:
        # Prepare features for the model
        X = pd.DataFrame([candle])
        # Drop 'timestamp' column if present
        if 'timestamp' in X.columns:
            X = X.drop(columns=['timestamp'])
        # Align columns to preprocessor's expected order if possible
        if self.preprocessor:
            expected_cols = None
            # Try to get feature names from preprocessor (sklearn >=1.0)
            if hasattr(self.preprocessor, 'feature_names_in_'):
                expected_cols = list(self.preprocessor.feature_names_in_)
            elif isinstance(self.preprocessor, dict) and 'feature_scaler' in self.preprocessor and hasattr(self.preprocessor['feature_scaler'], 'feature_names_in_'):
                expected_cols = list(self.preprocessor['feature_scaler'].feature_names_in_)
            if expected_cols is not None:
                # Check for missing or extra columns
                missing = [col for col in expected_cols if col not in X.columns]
                extra = [col for col in X.columns if col not in expected_cols]
                if missing or extra:
                    logger.error(f"Feature column mismatch in MLModelStrategy.\nExpected: {expected_cols}\nActual: {list(X.columns)}\nMissing: {missing}\nExtra: {extra}")
                    raise ValueError("Feature column mismatch. See logs for details.")
                # Reorder columns
                X = X[expected_cols]
            X = self.preprocessor.transform(X)
        else:
            raise RuntimeError("MLModelStrategy requires a preprocessor to ensure feature engineering matches training. Please provide the preprocessor used during training (e.g., --preprocessor path/to/preprocessor.joblib). This prevents input_dim mismatches. See https://discuss.pytorch.org/t/time-series-lstm-size-mismatch-beginner-question/4704 for details.")
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy()
        signals = []
        # Binary classification: output shape (batch, 1)
        if preds.ndim == 2 and preds.shape[1] == 1:
            if preds[0, 0] > self.threshold:
                signals.append({'action': 'enter', 'direction': 'long', 'symbol': self.symbol, 'reason': f"MLModel > threshold ({preds[0,0]:.3f})"})
            elif preds[0, 0] < 1 - self.threshold:
                signals.append({'action': 'enter', 'direction': 'short', 'symbol': self.symbol, 'reason': f"MLModel < 1-threshold ({preds[0,0]:.3f})"})
        # Multi-class: output shape (batch, n_classes)
        elif preds.ndim == 2 and preds.shape[1] > 1:
            direction = np.argmax(preds[0])
            if direction == 1:
                signals.append({'action': 'enter', 'direction': 'long', 'symbol': self.symbol, 'reason': "MLModel class=long"})
            elif direction == 2:
                signals.append({'action': 'enter', 'direction': 'short', 'symbol': self.symbol, 'reason': "MLModel class=short"})
        # Regression: output shape (batch,)
        elif preds.ndim == 1:
            if preds[0] > self.threshold:
                signals.append({'action': 'enter', 'direction': 'long', 'symbol': self.symbol, 'reason': f"MLModel regression > threshold ({preds[0]:.3f})"})
            elif preds[0] < -self.threshold:
                signals.append({'action': 'enter', 'direction': 'short', 'symbol': self.symbol, 'reason': f"MLModel regression < -threshold ({preds[0]:.3f})"})
        return signals 