#!/usr/bin/env python3
"""
Liquidity Level Mapping System for Smart Money Concepts

This module implements a sophisticated liquidity mapping system that identifies
and tracks institutional liquidity levels including Buy-Side Liquidity (BSL),
Sell-Side Liquidity (SSL), equal highs/lows, and liquidity grab/stop hunt detection.

Key Features:
- Equal highs/lows detection with precision tolerance
- Buy-Side and Sell-Side Liquidity zone identification
- Liquidity sweep and grab detection algorithms
- Stop hunt pattern recognition
- Institutional order flow analysis
- Multi-timeframe liquidity confluence
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiquidityType(Enum):
    """Enumeration for liquidity types"""
    BUY_SIDE = "buy_side"  # BSL - Above resistance levels
    SELL_SIDE = "sell_side"  # SSL - Below support levels
    EQUAL_HIGHS = "equal_highs"
    EQUAL_LOWS = "equal_lows"


class LiquidityStatus(Enum):
    """Enumeration for liquidity status"""
    ACTIVE = "active"
    SWEPT = "swept"
    PARTIALLY_SWEPT = "partially_swept"
    EXPIRED = "expired"


class SweepType(Enum):
    """Enumeration for liquidity sweep types"""
    STOP_HUNT = "stop_hunt"  # Quick sweep and reversal
    LIQUIDITY_GRAB = "liquidity_grab"  # Sweep for institutional entry
    BREAKOUT = "breakout"  # Genuine breakout continuation
    FALSE_BREAKOUT = "false_breakout"  # Failed breakout


@dataclass
class LiquidityLevel:
    """Data class representing a liquidity level"""
    type: LiquidityType
    price: float
    strength: float
    formation_timestamp: datetime
    formation_index: int
    touches: int = 0
    last_touch_timestamp: Optional[datetime] = None
    last_touch_index: Optional[int] = None
    status: LiquidityStatus = LiquidityStatus.ACTIVE
    sweep_timestamp: Optional[datetime] = None
    sweep_index: Optional[int] = None
    sweep_type: Optional[SweepType] = None
    volume_context: Dict[str, float] = field(default_factory=dict)
    equal_level_count: int = 1  # Number of equal highs/lows at this level
    confluence_score: float = 0.0
    institutional_interest: float = 0.0

    def is_price_near_level(self, price: float, tolerance_pct: float = 0.001) -> bool:
        """Check if price is near the liquidity level within tolerance"""
        tolerance = self.price * tolerance_pct
        return abs(price - self.price) <= tolerance

    def calculate_strength(self) -> float:
        """Calculate the strength of the liquidity level"""
        base_strength = min(self.touches * 0.2, 1.0)  # Touch-based strength
        equal_strength = min(self.equal_level_count * 0.1, 0.5)  # Equal levels strength

        # Handle potential NaN values in volume context
        avg_volume_ratio = self.volume_context.get('avg_volume_ratio', 1.0)
        if pd.isna(avg_volume_ratio) or avg_volume_ratio <= 0:
            avg_volume_ratio = 1.0
        volume_strength = min(avg_volume_ratio * 0.3, 0.3)

        confluence_strength = self.confluence_score * 0.2

        total_strength = base_strength + equal_strength + volume_strength + confluence_strength
        return min(max(total_strength, 0.1), 1.0)  # Ensure minimum strength of 0.1


@dataclass
class LiquiditySweep:
    """Data class representing a liquidity sweep event"""
    level: LiquidityLevel
    sweep_timestamp: datetime
    sweep_index: int
    sweep_price: float
    sweep_type: SweepType
    reversal_strength: float
    volume_spike: float
    follow_through: bool
    institutional_signature: float

    def is_stop_hunt(self) -> bool:
        """Determine if this is a stop hunt pattern"""
        return (self.sweep_type == SweepType.STOP_HUNT and
                self.reversal_strength > 0.5 and
                not self.follow_through)


class LiquidityMapper:
    """
    Advanced Liquidity Level Mapping System

    Identifies and tracks institutional liquidity levels including:
    - Equal highs and lows with precision tolerance
    - Buy-Side Liquidity (BSL) above resistance
    - Sell-Side Liquidity (SSL) below support
    - Liquidity sweeps and stop hunt detection
    - Institutional order flow signatures
    """

    def __init__(self, ohlcv: pd.DataFrame, equal_tolerance_pct: float = 0.002,
                 min_touches: int = 2, lookback_period: int = 50):
        """
        Initialize the Liquidity Mapper

        Args:
            ohlcv: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            equal_tolerance_pct: Tolerance for equal highs/lows detection (default 0.2%)
            min_touches: Minimum touches required for liquidity level (default 2)
            lookback_period: Lookback period for level detection (default 50)
        """
        self.ohlcv = ohlcv.copy()
        self.equal_tolerance_pct = equal_tolerance_pct
        self.min_touches = min_touches
        self.lookback_period = lookback_period

        # Ensure timestamp column is datetime
        if 'timestamp' in self.ohlcv.columns:
            self.ohlcv['timestamp'] = pd.to_datetime(self.ohlcv['timestamp'])
        else:
            # Create timestamp column if not present
            self.ohlcv['timestamp'] = pd.date_range(
                start=datetime.now() - timedelta(minutes=len(self.ohlcv) * 15),
                periods=len(self.ohlcv),
                freq='15T'
            )

        # Calculate technical indicators
        self._calculate_indicators()

        # Storage for detected liquidity levels and sweeps
        self.liquidity_levels: List[LiquidityLevel] = []
        self.liquidity_sweeps: List[LiquiditySweep] = []

        logger.info(f"LiquidityMapper initialized with {len(self.ohlcv)} candles")

    def _calculate_indicators(self):
        """Calculate technical indicators needed for liquidity mapping"""
        # Calculate volume metrics
        self.ohlcv['volume_ma'] = self.ohlcv['volume'].rolling(window=20).mean()
        self.ohlcv['volume_ratio'] = self.ohlcv['volume'] / self.ohlcv['volume_ma']

        # Calculate price movement metrics
        self.ohlcv['price_change_pct'] = self.ohlcv['close'].pct_change()
        self.ohlcv['high_change_pct'] = self.ohlcv['high'].pct_change()
        self.ohlcv['low_change_pct'] = self.ohlcv['low'].pct_change()

        # Calculate Average True Range for volatility context
        self._calculate_atr()

        # Calculate swing highs and lows
        self.ohlcv['swing_high'] = self._detect_swing_highs()
        self.ohlcv['swing_low'] = self._detect_swing_lows()

        # Calculate candle characteristics
        self.ohlcv['body_size'] = abs(self.ohlcv['close'] - self.ohlcv['open'])
        self.ohlcv['upper_wick'] = self.ohlcv['high'] - np.maximum(self.ohlcv['open'], self.ohlcv['close'])
        self.ohlcv['lower_wick'] = np.minimum(self.ohlcv['open'], self.ohlcv['close']) - self.ohlcv['low']

        logger.info("Technical indicators calculated successfully")

    def _calculate_atr(self, period: int = 14):
        """Calculate Average True Range"""
        high_low = self.ohlcv['high'] - self.ohlcv['low']
        high_close_prev = abs(self.ohlcv['high'] - self.ohlcv['close'].shift(1))
        low_close_prev = abs(self.ohlcv['low'] - self.ohlcv['close'].shift(1))

        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        self.ohlcv['atr'] = true_range.rolling(window=period).mean()

    def _detect_swing_highs(self, lookback: int = 5) -> pd.Series:
        """Detect swing highs using rolling window"""
        swing_highs = pd.Series(False, index=self.ohlcv.index)

        for i in range(lookback, len(self.ohlcv) - lookback):
            current_high = self.ohlcv.iloc[i]['high']
            left_highs = self.ohlcv.iloc[i-lookback:i]['high']
            right_highs = self.ohlcv.iloc[i+1:i+lookback+1]['high']

            if current_high > left_highs.max() and current_high > right_highs.max():
                swing_highs.iloc[i] = True

        return swing_highs

    def _detect_swing_lows(self, lookback: int = 5) -> pd.Series:
        """Detect swing lows using rolling window"""
        swing_lows = pd.Series(False, index=self.ohlcv.index)

        for i in range(lookback, len(self.ohlcv) - lookback):
            current_low = self.ohlcv.iloc[i]['low']
            left_lows = self.ohlcv.iloc[i-lookback:i]['low']
            right_lows = self.ohlcv.iloc[i+1:i+lookback+1]['low']

            if current_low < left_lows.min() and current_low < right_lows.min():
                swing_lows.iloc[i] = True

        return swing_lows

    def identify_equal_levels(self) -> List[LiquidityLevel]:
        """
        Identify equal highs and lows that represent liquidity levels

        Returns:
            List of detected equal high/low liquidity levels
        """
        equal_levels = []

        # Get swing highs and lows
        swing_highs = self.ohlcv[self.ohlcv['swing_high']].copy()
        swing_lows = self.ohlcv[self.ohlcv['swing_low']].copy()

        # Detect equal highs
        equal_highs = self._find_equal_levels(swing_highs, 'high', LiquidityType.EQUAL_HIGHS)
        equal_levels.extend(equal_highs)

        # Detect equal lows
        equal_lows = self._find_equal_levels(swing_lows, 'low', LiquidityType.EQUAL_LOWS)
        equal_levels.extend(equal_lows)

        logger.info(f"Identified {len(equal_levels)} equal levels ({len(equal_highs)} highs, {len(equal_lows)} lows)")
        return equal_levels

    def _find_equal_levels(self, swing_data: pd.DataFrame, price_col: str,
                          level_type: LiquidityType) -> List[LiquidityLevel]:
        """Find equal levels in swing data"""
        equal_levels = []

        if len(swing_data) < 2:
            return equal_levels

        # Group swing points by similar price levels
        price_groups = {}

        for idx, row in swing_data.iterrows():
            price = row[price_col]
            timestamp = row['timestamp']

            # Find if this price belongs to an existing group
            group_found = False
            for group_price, group_data in price_groups.items():
                if abs(price - group_price) / group_price <= self.equal_tolerance_pct:
                    group_data['prices'].append(price)
                    group_data['timestamps'].append(timestamp)
                    group_data['indices'].append(idx)
                    group_found = True
                    break

            if not group_found:
                price_groups[price] = {
                    'prices': [price],
                    'timestamps': [timestamp],
                    'indices': [idx]
                }

        # Create liquidity levels for groups with multiple touches
        for group_price, group_data in price_groups.items():
            if len(group_data['prices']) >= self.min_touches:
                avg_price = np.mean(group_data['prices'])
                formation_timestamp = min(group_data['timestamps'])
                formation_index = min(group_data['indices'])

                # Calculate volume context
                volume_context = self._get_volume_context_for_indices(group_data['indices'])

                level = LiquidityLevel(
                    type=level_type,
                    price=avg_price,
                    strength=0.0,  # Will be calculated later
                    formation_timestamp=formation_timestamp,
                    formation_index=formation_index,
                    touches=len(group_data['prices']),
                    equal_level_count=len(group_data['prices']),
                    volume_context=volume_context
                )

                # Calculate strength
                level.strength = level.calculate_strength()
                equal_levels.append(level)

        return equal_levels

    def _get_volume_context_for_indices(self, indices: List[int]) -> Dict[str, float]:
        """Get volume context for given indices"""
        volumes = [self.ohlcv.iloc[idx]['volume'] for idx in indices if idx < len(self.ohlcv)]
        volume_ratios = [self.ohlcv.iloc[idx]['volume_ratio'] for idx in indices if idx < len(self.ohlcv)]

        return {
            'avg_volume': np.mean(volumes) if volumes else 0,
            'max_volume': max(volumes) if volumes else 0,
            'avg_volume_ratio': np.mean(volume_ratios) if volume_ratios else 1.0,
            'max_volume_ratio': max(volume_ratios) if volume_ratios else 1.0
        }

    def identify_bsl_ssl_levels(self) -> List[LiquidityLevel]:
        """
        Identify Buy-Side Liquidity (BSL) and Sell-Side Liquidity (SSL) levels

        BSL: Liquidity above resistance levels where buy stops are placed
        SSL: Liquidity below support levels where sell stops are placed

        Returns:
            List of BSL and SSL liquidity levels
        """
        bsl_ssl_levels = []

        # Identify resistance levels (potential BSL)
        resistance_levels = self._identify_resistance_levels()
        for level in resistance_levels:
            bsl_level = LiquidityLevel(
                type=LiquidityType.BUY_SIDE,
                price=level['price'],
                strength=level['strength'],
                formation_timestamp=level['timestamp'],
                formation_index=level['index'],
                touches=level['touches'],
                volume_context=level['volume_context']
            )
            bsl_level.strength = bsl_level.calculate_strength()
            bsl_ssl_levels.append(bsl_level)

        # Identify support levels (potential SSL)
        support_levels = self._identify_support_levels()
        for level in support_levels:
            ssl_level = LiquidityLevel(
                type=LiquidityType.SELL_SIDE,
                price=level['price'],
                strength=level['strength'],
                formation_timestamp=level['timestamp'],
                formation_index=level['index'],
                touches=level['touches'],
                volume_context=level['volume_context']
            )
            ssl_level.strength = ssl_level.calculate_strength()
            bsl_ssl_levels.append(ssl_level)

        logger.info(f"Identified {len(bsl_ssl_levels)} BSL/SSL levels")
        return bsl_ssl_levels

    def _identify_resistance_levels(self) -> List[Dict]:
        """Identify resistance levels from swing highs"""
        resistance_levels = []
        swing_highs = self.ohlcv[self.ohlcv['swing_high']].copy()

        for idx, row in swing_highs.iterrows():
            # Look for subsequent tests of this level
            test_count = self._count_level_tests(idx, row['high'], 'high', direction='above')

            if test_count >= self.min_touches:
                volume_context = self._get_volume_context_around_index(idx)

                resistance_levels.append({
                    'price': row['high'],
                    'strength': min(test_count * 0.2, 1.0),
                    'timestamp': row['timestamp'],
                    'index': idx,
                    'touches': test_count,
                    'volume_context': volume_context
                })

        return resistance_levels

    def _identify_support_levels(self) -> List[Dict]:
        """Identify support levels from swing lows"""
        support_levels = []
        swing_lows = self.ohlcv[self.ohlcv['swing_low']].copy()

        for idx, row in swing_lows.iterrows():
            # Look for subsequent tests of this level
            test_count = self._count_level_tests(idx, row['low'], 'low', direction='below')

            if test_count >= self.min_touches:
                volume_context = self._get_volume_context_around_index(idx)

                support_levels.append({
                    'price': row['low'],
                    'strength': min(test_count * 0.2, 1.0),
                    'timestamp': row['timestamp'],
                    'index': idx,
                    'touches': test_count,
                    'volume_context': volume_context
                })

        return support_levels

    def _count_level_tests(self, start_idx: int, level_price: float,
                          price_col: str, direction: str) -> int:
        """Count how many times a level has been tested"""
        test_count = 1  # Include the initial formation
        tolerance = level_price * self.equal_tolerance_pct

        # Look forward from the formation point
        for i in range(start_idx + 1, len(self.ohlcv)):
            candle = self.ohlcv.iloc[i]

            if direction == 'above':
                # For resistance, look for highs that approach the level
                if abs(candle[price_col] - level_price) <= tolerance:
                    test_count += 1
            else:  # direction == 'below'
                # For support, look for lows that approach the level
                if abs(candle[price_col] - level_price) <= tolerance:
                    test_count += 1

        return test_count

    def _get_volume_context_around_index(self, idx: int, window: int = 5) -> Dict[str, float]:
        """Get volume context around a specific index"""
        start_idx = max(0, idx - window)
        end_idx = min(len(self.ohlcv), idx + window + 1)

        window_data = self.ohlcv.iloc[start_idx:end_idx]

        # Handle NaN values in volume calculations
        avg_volume = window_data['volume'].mean()
        if pd.isna(avg_volume):
            avg_volume = 1000000.0  # Default volume

        max_volume = window_data['volume'].max()
        if pd.isna(max_volume):
            max_volume = avg_volume

        avg_volume_ratio = window_data['volume_ratio'].mean()
        if pd.isna(avg_volume_ratio):
            avg_volume_ratio = 1.0

        max_volume_ratio = window_data['volume_ratio'].max()
        if pd.isna(max_volume_ratio):
            max_volume_ratio = avg_volume_ratio

        return {
            'avg_volume': avg_volume,
            'max_volume': max_volume,
            'avg_volume_ratio': avg_volume_ratio,
            'max_volume_ratio': max_volume_ratio
        }

    def detect_liquidity_sweeps(self, levels: List[LiquidityLevel]) -> List[LiquiditySweep]:
        """
        Detect liquidity sweeps and classify them as stop hunts or genuine breakouts

        Args:
            levels: List of liquidity levels to monitor for sweeps

        Returns:
            List of detected liquidity sweeps
        """
        sweeps = []

        for level in levels:
            sweep = self._analyze_level_for_sweeps(level)
            if sweep:
                sweeps.append(sweep)
                # Update the level status
                level.status = LiquidityStatus.SWEPT
                level.sweep_timestamp = sweep.sweep_timestamp
                level.sweep_index = sweep.sweep_index
                level.sweep_type = sweep.sweep_type

        logger.info(f"Detected {len(sweeps)} liquidity sweeps")
        return sweeps

    def _analyze_level_for_sweeps(self, level: LiquidityLevel) -> Optional[LiquiditySweep]:
        """Analyze a specific level for sweep patterns"""
        # Look for price action that sweeps through the level
        for i in range(level.formation_index + 1, len(self.ohlcv)):
            candle = self.ohlcv.iloc[i]

            # Check if this candle sweeps the level
            sweep_occurred = False
            sweep_price = 0.0

            if level.type in [LiquidityType.BUY_SIDE, LiquidityType.EQUAL_HIGHS]:
                # For BSL and equal highs, look for upward sweeps
                if candle['high'] > level.price:
                    sweep_occurred = True
                    sweep_price = candle['high']
            else:  # SSL and equal lows
                # For SSL and equal lows, look for downward sweeps
                if candle['low'] < level.price:
                    sweep_occurred = True
                    sweep_price = candle['low']

            if sweep_occurred:
                # Analyze the sweep characteristics
                sweep_type = self._classify_sweep_type(level, i, sweep_price)
                reversal_strength = self._calculate_reversal_strength(i)
                volume_spike = self._calculate_volume_spike(i)
                follow_through = self._analyze_follow_through(i, sweep_type)
                institutional_signature = self._calculate_institutional_signature(i, level)

                return LiquiditySweep(
                    level=level,
                    sweep_timestamp=candle['timestamp'],
                    sweep_index=i,
                    sweep_price=sweep_price,
                    sweep_type=sweep_type,
                    reversal_strength=reversal_strength,
                    volume_spike=volume_spike,
                    follow_through=follow_through,
                    institutional_signature=institutional_signature
                )

        return None

    def _classify_sweep_type(self, level: LiquidityLevel, sweep_index: int, sweep_price: float) -> SweepType:
        """Classify the type of liquidity sweep"""
        # Analyze price action after the sweep
        lookforward = min(10, len(self.ohlcv) - sweep_index - 1)

        if lookforward < 3:
            return SweepType.BREAKOUT  # Not enough data to classify

        sweep_candle = self.ohlcv.iloc[sweep_index]

        # Calculate reversal metrics
        reversal_strength = self._calculate_reversal_strength(sweep_index)
        volume_spike = self._calculate_volume_spike(sweep_index)

        # Analyze follow-through
        follow_through_strength = 0.0
        for i in range(sweep_index + 1, sweep_index + lookforward + 1):
            if i < len(self.ohlcv):
                candle = self.ohlcv.iloc[i]
                if level.type in [LiquidityType.BUY_SIDE, LiquidityType.EQUAL_HIGHS]:
                    # For upward sweeps, measure continued upward movement
                    if candle['close'] > sweep_price:
                        follow_through_strength += (candle['close'] - sweep_price) / sweep_price
                else:
                    # For downward sweeps, measure continued downward movement
                    if candle['close'] < sweep_price:
                        follow_through_strength += (sweep_price - candle['close']) / sweep_price

        # Classification logic
        if reversal_strength > 0.5 and volume_spike > 1.5 and follow_through_strength < 0.01:
            return SweepType.STOP_HUNT
        elif volume_spike > 2.0 and follow_through_strength > 0.02:
            return SweepType.LIQUIDITY_GRAB
        elif follow_through_strength > 0.03:
            return SweepType.BREAKOUT
        else:
            return SweepType.FALSE_BREAKOUT

    def _calculate_reversal_strength(self, sweep_index: int, lookforward: int = 5) -> float:
        """Calculate the strength of price reversal after a sweep"""
        if sweep_index + lookforward >= len(self.ohlcv):
            lookforward = len(self.ohlcv) - sweep_index - 1

        if lookforward < 2:
            return 0.0

        sweep_candle = self.ohlcv.iloc[sweep_index]
        sweep_price = sweep_candle['close']

        max_reversal = 0.0

        for i in range(sweep_index + 1, sweep_index + lookforward + 1):
            if i < len(self.ohlcv):
                candle = self.ohlcv.iloc[i]

                # Calculate reversal from sweep price
                high_reversal = abs(candle['high'] - sweep_price) / sweep_price
                low_reversal = abs(candle['low'] - sweep_price) / sweep_price

                max_reversal = max(max_reversal, high_reversal, low_reversal)

        return max_reversal

    def _calculate_volume_spike(self, sweep_index: int, lookback: int = 10) -> float:
        """Calculate volume spike ratio during sweep"""
        if sweep_index < lookback:
            lookback = sweep_index

        if lookback == 0:
            return 1.0

        # Get average volume before sweep
        avg_volume = self.ohlcv.iloc[sweep_index - lookback:sweep_index]['volume'].mean()

        # Get sweep candle volume
        sweep_volume = self.ohlcv.iloc[sweep_index]['volume']

        return sweep_volume / avg_volume if avg_volume > 0 else 1.0

    def _analyze_follow_through(self, sweep_index: int, sweep_type: SweepType, lookforward: int = 10) -> bool:
        """Analyze if there's follow-through after the sweep"""
        if sweep_index + lookforward >= len(self.ohlcv):
            lookforward = len(self.ohlcv) - sweep_index - 1

        if lookforward < 3:
            return False

        sweep_candle = self.ohlcv.iloc[sweep_index]

        # Count candles that continue in the sweep direction
        follow_through_count = 0
        total_candles = 0

        for i in range(sweep_index + 1, sweep_index + lookforward + 1):
            if i < len(self.ohlcv):
                candle = self.ohlcv.iloc[i]
                total_candles += 1

                # Check if candle continues the sweep direction
                if sweep_type in [SweepType.BREAKOUT, SweepType.LIQUIDITY_GRAB]:
                    if candle['close'] > candle['open']:  # Bullish continuation
                        follow_through_count += 1
                elif candle['close'] < candle['open']:  # Bearish continuation
                    follow_through_count += 1

        # Return True if more than 60% of candles show follow-through
        return (follow_through_count / total_candles) > 0.6 if total_candles > 0 else False

    def _calculate_institutional_signature(self, sweep_index: int, level: LiquidityLevel) -> float:
        """Calculate institutional signature score for the sweep"""
        score = 0.0

        # Volume signature (0-0.3)
        volume_spike = self._calculate_volume_spike(sweep_index)
        volume_score = min(0.3, (volume_spike - 1.0) * 0.15)
        score += volume_score

        # Reversal signature (0-0.3)
        reversal_strength = self._calculate_reversal_strength(sweep_index)
        reversal_score = min(0.3, reversal_strength * 1.5)
        score += reversal_score

        # Level strength signature (0-0.2)
        level_score = level.strength * 0.2
        score += level_score

        # Timing signature (0-0.2) - sweeps during low liquidity periods
        timing_score = self._calculate_timing_signature(sweep_index)
        score += timing_score

        return min(1.0, score)

    def _calculate_timing_signature(self, sweep_index: int) -> float:
        """Calculate timing signature based on market conditions"""
        # This is a simplified version - in practice, you'd consider:
        # - Time of day (Asian session, London open, NY open, etc.)
        # - Day of week
        # - Economic calendar events
        # - Market volatility

        candle = self.ohlcv.iloc[sweep_index]

        # For now, use volume ratio as a proxy for liquidity conditions
        volume_ratio = candle.get('volume_ratio', 1.0)

        # Lower volume ratios suggest lower liquidity (better for institutional moves)
        if volume_ratio < 0.8:
            return 0.2
        elif volume_ratio < 1.0:
            return 0.1
        else:
            return 0.0

    def map_all_liquidity_levels(self) -> Dict[str, List[LiquidityLevel]]:
        """
        Comprehensive liquidity mapping - identify all types of liquidity levels

        Returns:
            Dictionary containing all detected liquidity levels by type
        """
        logger.info("Starting comprehensive liquidity mapping...")

        # Identify equal levels
        equal_levels = self.identify_equal_levels()

        # Identify BSL/SSL levels
        bsl_ssl_levels = self.identify_bsl_ssl_levels()

        # Combine all levels
        all_levels = equal_levels + bsl_ssl_levels

        # Store in instance variable
        self.liquidity_levels = all_levels

        # Detect sweeps
        sweeps = self.detect_liquidity_sweeps(all_levels)
        self.liquidity_sweeps = sweeps

        # Organize by type
        levels_by_type = {
            'equal_highs': [l for l in all_levels if l.type == LiquidityType.EQUAL_HIGHS],
            'equal_lows': [l for l in all_levels if l.type == LiquidityType.EQUAL_LOWS],
            'buy_side': [l for l in all_levels if l.type == LiquidityType.BUY_SIDE],
            'sell_side': [l for l in all_levels if l.type == LiquidityType.SELL_SIDE]
        }

        logger.info(f"Liquidity mapping complete: {len(all_levels)} levels, {len(sweeps)} sweeps")
        return levels_by_type

    def get_liquidity_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about detected liquidity levels and sweeps"""
        if not self.liquidity_levels:
            return {
                'total_levels': 0,
                'levels_by_type': {},
                'total_sweeps': 0,
                'sweeps_by_type': {},
                'average_level_strength': 0,
                'stop_hunt_rate': 0,
                'institutional_signature_avg': 0
            }

        # Basic level statistics
        levels_by_type = {}
        for level_type in LiquidityType:
            count = len([l for l in self.liquidity_levels if l.type == level_type])
            levels_by_type[level_type.value] = count

        # Sweep statistics
        sweeps_by_type = {}
        for sweep_type in SweepType:
            count = len([s for s in self.liquidity_sweeps if s.sweep_type == sweep_type])
            sweeps_by_type[sweep_type.value] = count

        # Calculate averages with NaN handling
        strengths = [l.strength for l in self.liquidity_levels if not pd.isna(l.strength)]
        avg_strength = np.mean(strengths) if strengths else 0

        # Stop hunt rate
        stop_hunts = len([s for s in self.liquidity_sweeps if s.sweep_type == SweepType.STOP_HUNT])
        stop_hunt_rate = stop_hunts / len(self.liquidity_sweeps) if self.liquidity_sweeps else 0

        # Institutional signature average with NaN handling
        inst_sigs = [s.institutional_signature for s in self.liquidity_sweeps if not pd.isna(s.institutional_signature)]
        inst_sig_avg = np.mean(inst_sigs) if inst_sigs else 0

        return {
            'total_levels': len(self.liquidity_levels),
            'levels_by_type': levels_by_type,
            'total_sweeps': len(self.liquidity_sweeps),
            'sweeps_by_type': sweeps_by_type,
            'average_level_strength': avg_strength,
            'stop_hunt_rate': stop_hunt_rate,
            'institutional_signature_avg': inst_sig_avg,
            'active_levels': len([l for l in self.liquidity_levels if l.status == LiquidityStatus.ACTIVE]),
            'swept_levels': len([l for l in self.liquidity_levels if l.status == LiquidityStatus.SWEPT])
        }

    def get_active_levels_near_price(self, current_price: float, proximity_pct: float = 0.05) -> List[LiquidityLevel]:
        """
        Get active liquidity levels near the current price

        Args:
            current_price: Current market price
            proximity_pct: Proximity percentage (default 5%)

        Returns:
            List of active levels near the current price
        """
        proximity_range = current_price * proximity_pct

        active_levels = []
        for level in self.liquidity_levels:
            if (level.status == LiquidityStatus.ACTIVE and
                abs(level.price - current_price) <= proximity_range):
                active_levels.append(level)

        # Sort by proximity to current price
        active_levels.sort(key=lambda l: abs(l.price - current_price))

        return active_levels

    def analyze_stop_hunt_patterns(self) -> Dict[str, Any]:
        """Analyze stop hunt patterns and institutional behavior"""
        stop_hunts = [s for s in self.liquidity_sweeps if s.sweep_type == SweepType.STOP_HUNT]

        if not stop_hunts:
            return {
                'total_stop_hunts': 0,
                'average_reversal_strength': 0,
                'average_volume_spike': 0,
                'success_rate': 0,
                'institutional_signature_avg': 0,
                'by_level_type': {
                    'equal_highs': 0,
                    'equal_lows': 0,
                    'buy_side': 0,
                    'sell_side': 0
                }
            }

        # Calculate metrics with NaN handling
        reversals = [sh.reversal_strength for sh in stop_hunts if not pd.isna(sh.reversal_strength)]
        avg_reversal = np.mean(reversals) if reversals else 0

        volume_spikes = [sh.volume_spike for sh in stop_hunts if not pd.isna(sh.volume_spike)]
        avg_volume_spike = np.mean(volume_spikes) if volume_spikes else 0

        success_rate = len([sh for sh in stop_hunts if not sh.follow_through]) / len(stop_hunts)

        inst_sigs = [sh.institutional_signature for sh in stop_hunts if not pd.isna(sh.institutional_signature)]
        avg_inst_sig = np.mean(inst_sigs) if inst_sigs else 0

        return {
            'total_stop_hunts': len(stop_hunts),
            'average_reversal_strength': avg_reversal,
            'average_volume_spike': avg_volume_spike,
            'success_rate': success_rate,
            'institutional_signature_avg': avg_inst_sig,
            'by_level_type': {
                'equal_highs': len([sh for sh in stop_hunts if sh.level.type == LiquidityType.EQUAL_HIGHS]),
                'equal_lows': len([sh for sh in stop_hunts if sh.level.type == LiquidityType.EQUAL_LOWS]),
                'buy_side': len([sh for sh in stop_hunts if sh.level.type == LiquidityType.BUY_SIDE]),
                'sell_side': len([sh for sh in stop_hunts if sh.level.type == LiquidityType.SELL_SIDE])
            }
        }

    def validate_level_quality(self, level: LiquidityLevel) -> Dict[str, Any]:
        """
        Validate the quality of a specific liquidity level

        Args:
            level: LiquidityLevel to validate

        Returns:
            Dictionary with validation details
        """
        validation = {
            'overall_strength': level.strength,
            'criteria': {
                'sufficient_touches': level.touches >= self.min_touches,
                'high_volume_context': level.volume_context.get('avg_volume_ratio', 1.0) > 1.2,
                'institutional_interest': level.institutional_interest > 0.5,
                'confluence_score': level.confluence_score > 0.3
            },
            'metrics': {
                'touches': level.touches,
                'equal_level_count': level.equal_level_count,
                'volume_ratio': level.volume_context.get('avg_volume_ratio', 1.0),
                'institutional_interest': level.institutional_interest,
                'confluence_score': level.confluence_score
            },
            'status_info': {
                'current_status': level.status.value,
                'formation_age_candles': len(self.ohlcv) - level.formation_index - 1,
                'last_touch_age': (len(self.ohlcv) - level.last_touch_index - 1) if level.last_touch_index else None
            }
        }

        # Overall quality assessment
        criteria_met = sum(validation['criteria'].values())
        validation['quality_level'] = 'high' if criteria_met >= 3 else 'medium' if criteria_met >= 2 else 'low'

        return validation


# Utility functions for integration with existing SMC system
def create_sample_liquidity_data(num_candles: int = 500) -> pd.DataFrame:
    """
    Create sample OHLCV data with intentional liquidity levels for testing

    Args:
        num_candles: Number of candles to generate

    Returns:
        DataFrame with OHLCV data containing liquidity levels
    """
    np.random.seed(42)  # For reproducible results

    base_price = 50000
    data = []

    # Create levels every 100 candles
    level_prices = []

    for i in range(num_candles):
        # Create equal highs/lows every 100 candles
        if i % 100 == 0 and i > 50:
            # Create equal highs
            if np.random.random() > 0.5:
                level_price = base_price * (1 + np.random.uniform(0.02, 0.05))
                level_prices.append(('high', level_price, i))
            else:
                # Create equal lows
                level_price = base_price * (1 - np.random.uniform(0.02, 0.05))
                level_prices.append(('low', level_price, i))

        # Normal price movement
        volatility = base_price * 0.01
        open_price = base_price + np.random.normal(0, volatility * 0.5)

        # Check if we should create a level test
        level_test = False
        for level_type, level_price, level_start in level_prices:
            if i > level_start + 10 and i < level_start + 50:
                if level_type == 'high' and np.random.random() > 0.9:
                    # Test the high level
                    open_price = level_price * (1 - np.random.uniform(0.001, 0.005))
                    level_test = True
                elif level_type == 'low' and np.random.random() > 0.9:
                    # Test the low level
                    open_price = level_price * (1 + np.random.uniform(0.001, 0.005))
                    level_test = True

        close_price = open_price * (1 + np.random.normal(0, 0.01))

        if level_test:
            # Create more precise level test
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.002)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.002)))
        else:
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))

        # Higher volume during level tests
        base_volume = 1000000
        if level_test:
            volume = base_volume * np.random.uniform(1.5, 3.0)
        else:
            volume = base_volume * np.random.uniform(0.5, 1.5)

        timestamp = pd.Timestamp.now() - pd.Timedelta(minutes=(num_candles - i) * 15)

        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })

        base_price = close_price

    return pd.DataFrame(data)


def get_enhanced_liquidity_levels(ohlcv: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
    """
    Convenience function to get enhanced liquidity levels in dictionary format

    Args:
        ohlcv: OHLCV DataFrame
        **kwargs: Additional parameters for LiquidityMapper

    Returns:
        List of detected liquidity levels in dictionary format
    """
    mapper = LiquidityMapper(ohlcv, **kwargs)
    levels_by_type = mapper.map_all_liquidity_levels()

    # Flatten all levels into a single list
    all_levels = []
    for level_type, levels in levels_by_type.items():
        all_levels.extend(levels)

    # Convert to dictionary format
    result = []
    for level in all_levels:
        result.append({
            'type': level.type.value,
            'price': level.price,
            'strength': level.strength,
            'formation_timestamp': level.formation_timestamp,
            'formation_index': level.formation_index,
            'touches': level.touches,
            'status': level.status.value,
            'equal_level_count': level.equal_level_count,
            'confluence_score': level.confluence_score,
            'institutional_interest': level.institutional_interest,
            'volume_context': level.volume_context,
            'last_touch_timestamp': level.last_touch_timestamp,
            'sweep_timestamp': level.sweep_timestamp,
            'sweep_type': level.sweep_type.value if level.sweep_type else None
        })

    return result


def get_liquidity_sweeps(ohlcv: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
    """
    Convenience function to get liquidity sweeps in dictionary format

    Args:
        ohlcv: OHLCV DataFrame
        **kwargs: Additional parameters for LiquidityMapper

    Returns:
        List of detected liquidity sweeps in dictionary format
    """
    mapper = LiquidityMapper(ohlcv, **kwargs)
    mapper.map_all_liquidity_levels()  # This also detects sweeps

    # Convert sweeps to dictionary format
    result = []
    for sweep in mapper.liquidity_sweeps:
        result.append({
            'level_type': sweep.level.type.value,
            'level_price': sweep.level.price,
            'sweep_timestamp': sweep.sweep_timestamp,
            'sweep_index': sweep.sweep_index,
            'sweep_price': sweep.sweep_price,
            'sweep_type': sweep.sweep_type.value,
            'reversal_strength': sweep.reversal_strength,
            'volume_spike': sweep.volume_spike,
            'follow_through': sweep.follow_through,
            'institutional_signature': sweep.institutional_signature,
            'is_stop_hunt': sweep.is_stop_hunt()
        })

    return result


# Example usage and testing
if __name__ == "__main__":
    print("Liquidity Level Mapping System - Testing")
    print("=" * 50)

    # Create sample data with intentional liquidity levels
    print("Creating sample OHLCV data with liquidity levels...")
    ohlcv_data = create_sample_liquidity_data(400)

    # Initialize Liquidity Mapper
    print("Initializing Liquidity Mapper...")
    mapper = LiquidityMapper(
        ohlcv_data,
        equal_tolerance_pct=0.002,  # 0.2% tolerance for equal levels
        min_touches=2,              # Minimum 2 touches for level validation
        lookback_period=50          # 50-candle lookback for analysis
    )

    # Map all liquidity levels
    print("Mapping liquidity levels...")
    levels_by_type = mapper.map_all_liquidity_levels()

    # Get statistics
    stats = mapper.get_liquidity_statistics()

    # Print results
    print(f"\nLiquidity Mapping Results:")
    print(f"Total levels detected: {stats['total_levels']}")
    print(f"Equal highs: {stats['levels_by_type']['equal_highs']}")
    print(f"Equal lows: {stats['levels_by_type']['equal_lows']}")
    print(f"Buy-side liquidity: {stats['levels_by_type']['buy_side']}")
    print(f"Sell-side liquidity: {stats['levels_by_type']['sell_side']}")
    print(f"Average level strength: {stats['average_level_strength']:.3f}")

    print(f"\nSweep Analysis:")
    print(f"Total sweeps: {stats['total_sweeps']}")
    print(f"Stop hunt rate: {stats['stop_hunt_rate']:.1%}")
    print(f"Institutional signature avg: {stats['institutional_signature_avg']:.3f}")
    print(f"Active levels: {stats['active_levels']}")
    print(f"Swept levels: {stats['swept_levels']}")

    # Analyze stop hunt patterns
    stop_hunt_analysis = mapper.analyze_stop_hunt_patterns()
    print(f"\nStop Hunt Analysis:")
    print(f"Total stop hunts: {stop_hunt_analysis['total_stop_hunts']}")
    print(f"Average reversal strength: {stop_hunt_analysis['average_reversal_strength']:.3f}")
    print(f"Average volume spike: {stop_hunt_analysis['average_volume_spike']:.2f}x")
    print(f"Success rate: {stop_hunt_analysis['success_rate']:.1%}")

    # Show some example levels
    if mapper.liquidity_levels:
        print(f"\nExample Liquidity Levels (first 3):")
        for i, level in enumerate(mapper.liquidity_levels[:3]):
            print(f"Level {i+1}: {level.type.value} at {level.price:.2f}")
            print(f"  Strength: {level.strength:.3f}, Touches: {level.touches}")
            print(f"  Status: {level.status.value}")
            if level.sweep_type:
                print(f"  Sweep type: {level.sweep_type.value}")
            print()

    # Show some example sweeps
    if mapper.liquidity_sweeps:
        print(f"Example Liquidity Sweeps (first 3):")
        for i, sweep in enumerate(mapper.liquidity_sweeps[:3]):
            print(f"Sweep {i+1}: {sweep.sweep_type.value} of {sweep.level.type.value} level")
            print(f"  Level price: {sweep.level.price:.2f}, Sweep price: {sweep.sweep_price:.2f}")
            print(f"  Reversal strength: {sweep.reversal_strength:.3f}")
            print(f"  Volume spike: {sweep.volume_spike:.2f}x")
            print(f"  Institutional signature: {sweep.institutional_signature:.3f}")
            print(f"  Is stop hunt: {sweep.is_stop_hunt()}")
            print()

    # Test convenience functions
    print("Testing convenience functions...")
    enhanced_levels = get_enhanced_liquidity_levels(ohlcv_data)
    liquidity_sweeps = get_liquidity_sweeps(ohlcv_data)
    print(f"Convenience function detected {len(enhanced_levels)} levels and {len(liquidity_sweeps)} sweeps")

    print("\n✅ Liquidity Level Mapping System testing complete!")
    print("The system successfully detected and analyzed institutional liquidity levels.")


# Integration function for existing SMC system
def integrate_liquidity_mapper_with_smc(smc_detector_class):
    """
    Decorator to integrate liquidity mapping with existing SMC detector

    Args:
        smc_detector_class: Existing SMC detector class to enhance

    Returns:
        Enhanced SMC detector class with liquidity mapping capabilities
    """
    class EnhancedSMCDetector(smc_detector_class):
        def __init__(self, ohlcv: pd.DataFrame, **kwargs):
            super().__init__(ohlcv)
            # Initialize enhanced liquidity mapper
            self.liquidity_mapper = LiquidityMapper(ohlcv, **kwargs)

        def detect_liquidity_levels(self, **kwargs) -> Dict[str, List[LiquidityLevel]]:
            """Enhanced liquidity level detection"""
            return self.liquidity_mapper.map_all_liquidity_levels()

        def get_liquidity_statistics(self) -> Dict[str, Any]:
            """Get liquidity statistics"""
            # Ensure levels are mapped
            if not self.liquidity_mapper.liquidity_levels:
                self.liquidity_mapper.map_all_liquidity_levels()

            return self.liquidity_mapper.get_liquidity_statistics()

        def get_active_liquidity_near_price(self, price: float) -> List[LiquidityLevel]:
            """Get active liquidity levels near a specific price"""
            # Ensure levels are mapped
            if not self.liquidity_mapper.liquidity_levels:
                self.liquidity_mapper.map_all_liquidity_levels()

            return self.liquidity_mapper.get_active_levels_near_price(price)

        def analyze_stop_hunts(self) -> Dict[str, Any]:
            """Analyze stop hunt patterns"""
            # Ensure levels are mapped
            if not self.liquidity_mapper.liquidity_levels:
                self.liquidity_mapper.map_all_liquidity_levels()

            return self.liquidity_mapper.analyze_stop_hunt_patterns()

        def detect_all_enhanced(self) -> Dict[str, Any]:
            """Enhanced detect_all with liquidity mapping"""
            base_results = super().detect_all()

            # Add liquidity analysis
            liquidity_levels = self.detect_liquidity_levels()
            liquidity_stats = self.get_liquidity_statistics()
            stop_hunt_analysis = self.analyze_stop_hunts()

            base_results['liquidity_levels'] = liquidity_levels
            base_results['liquidity_statistics'] = liquidity_stats
            base_results['stop_hunt_analysis'] = stop_hunt_analysis

            return base_results

    return EnhancedSMCDetector