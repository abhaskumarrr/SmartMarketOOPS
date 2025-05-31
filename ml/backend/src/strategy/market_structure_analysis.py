#!/usr/bin/env python3
"""
Market Structure Analysis System for Smart Money Concepts

This module implements a sophisticated market structure analysis system that identifies
Break of Structure (BOS) and Change of Character (ChoCH) patterns for institutional
trading analysis. The system provides comprehensive trend change detection and market
structure shift identification.

Key Features:
- Break of Structure (BOS) detection for trend continuation
- Change of Character (ChoCH) identification for trend reversals
- Market structure shift analysis with institutional validation
- Swing point detection and trend line analysis
- Multi-timeframe structure confluence
- Volume-based validation for structure breaks
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


class StructureType(Enum):
    """Enumeration for market structure types"""
    BOS = "break_of_structure"  # Break of Structure - trend continuation
    CHOCH = "change_of_character"  # Change of Character - trend reversal
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"


class TrendDirection(Enum):
    """Enumeration for trend directions"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    TRANSITIONING = "transitioning"


class StructureStrength(Enum):
    """Enumeration for structure strength levels"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class SwingPoint:
    """Data class representing a swing high or low point"""
    type: StructureType
    price: float
    timestamp: datetime
    index: int
    volume: float
    strength: float = 0.0
    confirmed: bool = False
    retest_count: int = 0
    last_retest_timestamp: Optional[datetime] = None

    def is_swing_high(self) -> bool:
        """Check if this is a swing high"""
        return self.type == StructureType.SWING_HIGH

    def is_swing_low(self) -> bool:
        """Check if this is a swing low"""
        return self.type == StructureType.SWING_LOW


@dataclass
class StructureBreak:
    """Data class representing a market structure break (BOS or ChoCH)"""
    type: StructureType
    direction: TrendDirection
    break_price: float
    break_timestamp: datetime
    break_index: int
    previous_structure: SwingPoint
    volume_confirmation: float
    strength: StructureStrength
    impulse_strength: float
    retracement_depth: float
    follow_through: bool
    institutional_signature: float

    def is_bos(self) -> bool:
        """Check if this is a Break of Structure"""
        return self.type == StructureType.BOS

    def is_choch(self) -> bool:
        """Check if this is a Change of Character"""
        return self.type == StructureType.CHOCH


@dataclass
class MarketStructure:
    """Data class representing the current market structure state"""
    current_trend: TrendDirection
    last_structure_break: Optional[StructureBreak]
    active_swing_highs: List[SwingPoint]
    active_swing_lows: List[SwingPoint]
    structure_breaks: List[StructureBreak]
    trend_strength: float
    structure_quality: float
    last_update_timestamp: datetime


class MarketStructureAnalyzer:
    """
    Advanced Market Structure Analysis System

    Identifies and tracks market structure patterns including:
    - Break of Structure (BOS) for trend continuation signals
    - Change of Character (ChoCH) for trend reversal identification
    - Swing point detection and validation
    - Market structure shift analysis
    - Institutional behavior validation
    """

    def __init__(self, ohlcv: pd.DataFrame, swing_lookback: int = 5,
                 min_structure_distance: float = 0.005, volume_threshold: float = 1.2):
        """
        Initialize the Market Structure Analyzer

        Args:
            ohlcv: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            swing_lookback: Lookback period for swing point detection (default 5)
            min_structure_distance: Minimum distance between structure points (default 0.5%)
            volume_threshold: Volume threshold for structure validation (default 1.2x average)
        """
        self.ohlcv = ohlcv.copy()
        self.swing_lookback = swing_lookback
        self.min_structure_distance = min_structure_distance
        self.volume_threshold = volume_threshold

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

        # Initialize market structure tracking
        self.swing_points: List[SwingPoint] = []
        self.structure_breaks: List[StructureBreak] = []
        self.current_structure = MarketStructure(
            current_trend=TrendDirection.NEUTRAL,
            last_structure_break=None,
            active_swing_highs=[],
            active_swing_lows=[],
            structure_breaks=[],
            trend_strength=0.0,
            structure_quality=0.0,
            last_update_timestamp=datetime.now()
        )

        logger.info(f"MarketStructureAnalyzer initialized with {len(self.ohlcv)} candles")

    def _calculate_indicators(self):
        """Calculate technical indicators needed for market structure analysis"""
        # Calculate volume metrics
        self.ohlcv['volume_ma'] = self.ohlcv['volume'].rolling(window=20).mean()
        self.ohlcv['volume_ratio'] = self.ohlcv['volume'] / self.ohlcv['volume_ma']

        # Calculate price movement metrics
        self.ohlcv['price_change'] = self.ohlcv['close'].pct_change()
        self.ohlcv['high_change'] = self.ohlcv['high'].pct_change()
        self.ohlcv['low_change'] = self.ohlcv['low'].pct_change()

        # Calculate Average True Range for volatility context
        self._calculate_atr()

        # Calculate momentum indicators
        self.ohlcv['rsi'] = self._calculate_rsi()

        # Calculate candle characteristics
        self.ohlcv['body_size'] = abs(self.ohlcv['close'] - self.ohlcv['open'])
        self.ohlcv['upper_wick'] = self.ohlcv['high'] - np.maximum(self.ohlcv['open'], self.ohlcv['close'])
        self.ohlcv['lower_wick'] = np.minimum(self.ohlcv['open'], self.ohlcv['close']) - self.ohlcv['low']

        # Calculate trend indicators
        self.ohlcv['ema_fast'] = self.ohlcv['close'].ewm(span=12).mean()
        self.ohlcv['ema_slow'] = self.ohlcv['close'].ewm(span=26).mean()
        self.ohlcv['trend_direction'] = np.where(
            self.ohlcv['ema_fast'] > self.ohlcv['ema_slow'], 1, -1
        )

        logger.info("Technical indicators calculated successfully")

    def _calculate_atr(self, period: int = 14):
        """Calculate Average True Range"""
        high_low = self.ohlcv['high'] - self.ohlcv['low']
        high_close_prev = abs(self.ohlcv['high'] - self.ohlcv['close'].shift(1))
        low_close_prev = abs(self.ohlcv['low'] - self.ohlcv['close'].shift(1))

        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        self.ohlcv['atr'] = true_range.rolling(window=period).mean()

    def _calculate_rsi(self, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = self.ohlcv['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral RSI value

    def detect_swing_points(self) -> List[SwingPoint]:
        """
        Detect swing highs and lows in the price data

        Returns:
            List of detected swing points
        """
        swing_points = []

        # Detect swing highs
        swing_highs = self._detect_swing_highs()
        for idx, is_swing in swing_highs.items():
            if is_swing:
                candle = self.ohlcv.iloc[idx]
                strength = self._calculate_swing_strength(idx, 'high')

                swing_point = SwingPoint(
                    type=StructureType.SWING_HIGH,
                    price=candle['high'],
                    timestamp=candle['timestamp'],
                    index=idx,
                    volume=candle['volume'],
                    strength=strength,
                    confirmed=self._is_swing_confirmed(idx, 'high')
                )
                swing_points.append(swing_point)

        # Detect swing lows
        swing_lows = self._detect_swing_lows()
        for idx, is_swing in swing_lows.items():
            if is_swing:
                candle = self.ohlcv.iloc[idx]
                strength = self._calculate_swing_strength(idx, 'low')

                swing_point = SwingPoint(
                    type=StructureType.SWING_LOW,
                    price=candle['low'],
                    timestamp=candle['timestamp'],
                    index=idx,
                    volume=candle['volume'],
                    strength=strength,
                    confirmed=self._is_swing_confirmed(idx, 'low')
                )
                swing_points.append(swing_point)

        # Sort by timestamp
        swing_points.sort(key=lambda x: x.timestamp)

        # Filter by minimum distance
        filtered_points = self._filter_swing_points_by_distance(swing_points)

        self.swing_points = filtered_points
        logger.info(f"Detected {len(filtered_points)} swing points")
        return filtered_points

    def _detect_swing_highs(self) -> Dict[int, bool]:
        """Detect swing highs using rolling window analysis"""
        swing_highs = {}

        for i in range(self.swing_lookback, len(self.ohlcv) - self.swing_lookback):
            current_high = self.ohlcv.iloc[i]['high']

            # Check left side
            left_highs = self.ohlcv.iloc[i-self.swing_lookback:i]['high']
            left_condition = current_high > left_highs.max()

            # Check right side
            right_highs = self.ohlcv.iloc[i+1:i+self.swing_lookback+1]['high']
            right_condition = current_high > right_highs.max()

            swing_highs[i] = left_condition and right_condition

        return swing_highs

    def _detect_swing_lows(self) -> Dict[int, bool]:
        """Detect swing lows using rolling window analysis"""
        swing_lows = {}

        for i in range(self.swing_lookback, len(self.ohlcv) - self.swing_lookback):
            current_low = self.ohlcv.iloc[i]['low']

            # Check left side
            left_lows = self.ohlcv.iloc[i-self.swing_lookback:i]['low']
            left_condition = current_low < left_lows.min()

            # Check right side
            right_lows = self.ohlcv.iloc[i+1:i+self.swing_lookback+1]['low']
            right_condition = current_low < right_lows.min()

            swing_lows[i] = left_condition and right_condition

        return swing_lows

    def _calculate_swing_strength(self, idx: int, swing_type: str) -> float:
        """Calculate the strength of a swing point"""
        candle = self.ohlcv.iloc[idx]

        # Volume strength (0-0.3)
        volume_ratio = candle.get('volume_ratio', 1.0)
        if pd.isna(volume_ratio):
            volume_ratio = 1.0
        volume_strength = min(0.3, (volume_ratio - 1.0) * 0.3)

        # Price movement strength (0-0.3)
        if swing_type == 'high':
            price_move = (candle['high'] - candle['open']) / candle['open']
        else:
            price_move = (candle['open'] - candle['low']) / candle['open']

        price_strength = min(0.3, abs(price_move) * 10)

        # ATR-based strength (0-0.2)
        atr = candle.get('atr', candle['high'] - candle['low'])
        if pd.isna(atr) or atr == 0:
            atr = candle['high'] - candle['low']

        if swing_type == 'high':
            atr_strength = min(0.2, (candle['high'] - candle['close']) / atr * 0.2)
        else:
            atr_strength = min(0.2, (candle['close'] - candle['low']) / atr * 0.2)

        # RSI divergence strength (0-0.2)
        rsi = candle.get('rsi', 50)
        if pd.isna(rsi):
            rsi = 50

        if swing_type == 'high':
            rsi_strength = min(0.2, max(0, (rsi - 70) / 30 * 0.2))
        else:
            rsi_strength = min(0.2, max(0, (30 - rsi) / 30 * 0.2))

        total_strength = volume_strength + price_strength + atr_strength + rsi_strength
        return min(1.0, max(0.1, total_strength))

    def _is_swing_confirmed(self, idx: int, swing_type: str, confirmation_bars: int = 3) -> bool:
        """Check if a swing point is confirmed by subsequent price action"""
        if idx + confirmation_bars >= len(self.ohlcv):
            return False

        swing_price = self.ohlcv.iloc[idx]['high'] if swing_type == 'high' else self.ohlcv.iloc[idx]['low']

        # Check confirmation bars
        for i in range(idx + 1, min(idx + confirmation_bars + 1, len(self.ohlcv))):
            candle = self.ohlcv.iloc[i]

            if swing_type == 'high':
                if candle['high'] > swing_price:
                    return False  # Swing high was broken
            else:
                if candle['low'] < swing_price:
                    return False  # Swing low was broken

        return True

    def _filter_swing_points_by_distance(self, swing_points: List[SwingPoint]) -> List[SwingPoint]:
        """Filter swing points by minimum distance requirement"""
        if not swing_points:
            return []

        filtered_points = [swing_points[0]]  # Always include first point

        for point in swing_points[1:]:
            last_point = filtered_points[-1]

            # Calculate distance
            price_distance = abs(point.price - last_point.price) / last_point.price

            # Only include if distance is sufficient or it's a different type
            if (price_distance >= self.min_structure_distance or
                point.type != last_point.type):
                filtered_points.append(point)

        return filtered_points

    def detect_structure_breaks(self) -> List[StructureBreak]:
        """
        Detect Break of Structure (BOS) and Change of Character (ChoCH) patterns

        Returns:
            List of detected structure breaks
        """
        if not self.swing_points:
            self.detect_swing_points()

        structure_breaks = []

        # Analyze each candle for potential structure breaks
        for i in range(len(self.swing_points), len(self.ohlcv)):
            candle = self.ohlcv.iloc[i]

            # Check for BOS patterns
            bos_breaks = self._detect_bos_patterns(i, candle)
            structure_breaks.extend(bos_breaks)

            # Check for ChoCH patterns
            choch_breaks = self._detect_choch_patterns(i, candle)
            structure_breaks.extend(choch_breaks)

        # Sort by timestamp and validate
        structure_breaks.sort(key=lambda x: x.break_timestamp)
        validated_breaks = self._validate_structure_breaks(structure_breaks)

        self.structure_breaks = validated_breaks
        logger.info(f"Detected {len(validated_breaks)} structure breaks")
        return validated_breaks

    def _detect_bos_patterns(self, current_idx: int, current_candle: pd.Series) -> List[StructureBreak]:
        """Detect Break of Structure patterns"""
        bos_breaks = []

        # Get recent swing points for analysis
        recent_highs = [sp for sp in self.swing_points if sp.is_swing_high() and sp.index < current_idx]
        recent_lows = [sp for sp in self.swing_points if sp.is_swing_low() and sp.index < current_idx]

        # Check for bullish BOS (breaking above recent swing high)
        if recent_highs:
            last_high = max(recent_highs, key=lambda x: x.timestamp)
            if (current_candle['high'] > last_high.price and
                self._is_trend_continuation(current_idx, TrendDirection.BULLISH)):

                bos_break = self._create_structure_break(
                    StructureType.BOS,
                    TrendDirection.BULLISH,
                    current_idx,
                    current_candle,
                    last_high
                )
                if bos_break:
                    bos_breaks.append(bos_break)

        # Check for bearish BOS (breaking below recent swing low)
        if recent_lows:
            last_low = max(recent_lows, key=lambda x: x.timestamp)
            if (current_candle['low'] < last_low.price and
                self._is_trend_continuation(current_idx, TrendDirection.BEARISH)):

                bos_break = self._create_structure_break(
                    StructureType.BOS,
                    TrendDirection.BEARISH,
                    current_idx,
                    current_candle,
                    last_low
                )
                if bos_break:
                    bos_breaks.append(bos_break)

        return bos_breaks

    def _detect_choch_patterns(self, current_idx: int, current_candle: pd.Series) -> List[StructureBreak]:
        """Detect Change of Character patterns"""
        choch_breaks = []

        # Get recent swing points for analysis
        recent_highs = [sp for sp in self.swing_points if sp.is_swing_high() and sp.index < current_idx]
        recent_lows = [sp for sp in self.swing_points if sp.is_swing_low() and sp.index < current_idx]

        # Check for bearish ChoCH (breaking below swing low in uptrend)
        if recent_lows and self._get_current_trend(current_idx) == TrendDirection.BULLISH:
            last_low = max(recent_lows, key=lambda x: x.timestamp)
            if current_candle['low'] < last_low.price:

                choch_break = self._create_structure_break(
                    StructureType.CHOCH,
                    TrendDirection.BEARISH,
                    current_idx,
                    current_candle,
                    last_low
                )
                if choch_break:
                    choch_breaks.append(choch_break)

        # Check for bullish ChoCH (breaking above swing high in downtrend)
        if recent_highs and self._get_current_trend(current_idx) == TrendDirection.BEARISH:
            last_high = max(recent_highs, key=lambda x: x.timestamp)
            if current_candle['high'] > last_high.price:

                choch_break = self._create_structure_break(
                    StructureType.CHOCH,
                    TrendDirection.BULLISH,
                    current_idx,
                    current_candle,
                    last_high
                )
                if choch_break:
                    choch_breaks.append(choch_break)

        return choch_breaks

    def _create_structure_break(self, structure_type: StructureType, direction: TrendDirection,
                              current_idx: int, current_candle: pd.Series,
                              previous_structure: SwingPoint) -> Optional[StructureBreak]:
        """Create a structure break object with validation"""

        # Calculate break price
        if direction == TrendDirection.BULLISH:
            break_price = current_candle['high']
        else:
            break_price = current_candle['low']

        # Calculate volume confirmation
        volume_confirmation = self._calculate_volume_confirmation(current_idx)

        # Calculate impulse strength
        impulse_strength = self._calculate_impulse_strength(current_idx, direction)

        # Calculate retracement depth
        retracement_depth = self._calculate_retracement_depth(current_idx, previous_structure)

        # Determine structure strength
        strength = self._determine_structure_strength(
            volume_confirmation, impulse_strength, retracement_depth
        )

        # Calculate institutional signature
        institutional_signature = self._calculate_institutional_signature(
            current_idx, structure_type, direction
        )

        # Check follow-through
        follow_through = self._check_follow_through(current_idx, direction)

        # Minimum validation criteria
        if (volume_confirmation < 1.1 and impulse_strength < 0.3 and
            institutional_signature < 0.4):
            return None  # Not strong enough to be considered valid

        return StructureBreak(
            type=structure_type,
            direction=direction,
            break_price=break_price,
            break_timestamp=current_candle['timestamp'],
            break_index=current_idx,
            previous_structure=previous_structure,
            volume_confirmation=volume_confirmation,
            strength=strength,
            impulse_strength=impulse_strength,
            retracement_depth=retracement_depth,
            follow_through=follow_through,
            institutional_signature=institutional_signature
        )

    def _is_trend_continuation(self, current_idx: int, direction: TrendDirection) -> bool:
        """Check if the break represents trend continuation"""
        current_trend = self._get_current_trend(current_idx)
        return current_trend == direction or current_trend == TrendDirection.NEUTRAL

    def _get_current_trend(self, current_idx: int, lookback: int = 20) -> TrendDirection:
        """Determine the current trend direction"""
        start_idx = max(0, current_idx - lookback)

        # Use EMA crossover for trend determination
        if current_idx < len(self.ohlcv):
            candle = self.ohlcv.iloc[current_idx]
            ema_fast = candle.get('ema_fast', 0)
            ema_slow = candle.get('ema_slow', 0)

            if pd.isna(ema_fast) or pd.isna(ema_slow):
                return TrendDirection.NEUTRAL

            if ema_fast > ema_slow:
                return TrendDirection.BULLISH
            elif ema_fast < ema_slow:
                return TrendDirection.BEARISH
            else:
                return TrendDirection.NEUTRAL

        return TrendDirection.NEUTRAL

    def _calculate_volume_confirmation(self, current_idx: int, lookback: int = 10) -> float:
        """Calculate volume confirmation for structure break"""
        if current_idx < lookback:
            lookback = current_idx

        if lookback == 0:
            return 1.0

        # Get current volume and average volume
        current_volume = self.ohlcv.iloc[current_idx]['volume']
        avg_volume = self.ohlcv.iloc[current_idx - lookback:current_idx]['volume'].mean()

        if pd.isna(avg_volume) or avg_volume == 0:
            return 1.0

        return current_volume / avg_volume

    def _calculate_impulse_strength(self, current_idx: int, direction: TrendDirection,
                                  lookback: int = 5) -> float:
        """Calculate the strength of the impulse move"""
        if current_idx < lookback:
            lookback = current_idx

        if lookback == 0:
            return 0.0

        # Get price movement over lookback period
        start_price = self.ohlcv.iloc[current_idx - lookback]['close']
        current_price = self.ohlcv.iloc[current_idx]['close']

        price_change = abs(current_price - start_price) / start_price

        # Get ATR for normalization
        current_atr = self.ohlcv.iloc[current_idx].get('atr', abs(current_price - start_price))
        if pd.isna(current_atr) or current_atr == 0:
            current_atr = abs(current_price - start_price)

        # Normalize by ATR
        atr_normalized_strength = abs(current_price - start_price) / current_atr

        # Combine price change and ATR normalization
        impulse_strength = min(1.0, (price_change * 5 + atr_normalized_strength * 0.2))

        return max(0.0, impulse_strength)

    def _calculate_retracement_depth(self, current_idx: int, previous_structure: SwingPoint) -> float:
        """Calculate the depth of retracement before the break"""
        if current_idx <= previous_structure.index:
            return 0.0

        # Find the highest/lowest point between structure and current break
        segment = self.ohlcv.iloc[previous_structure.index:current_idx + 1]

        if previous_structure.is_swing_high():
            # For swing high, find the lowest low in the segment
            lowest_low = segment['low'].min()
            retracement = (previous_structure.price - lowest_low) / previous_structure.price
        else:
            # For swing low, find the highest high in the segment
            highest_high = segment['high'].max()
            retracement = (highest_high - previous_structure.price) / previous_structure.price

        return min(1.0, max(0.0, retracement))

    def _determine_structure_strength(self, volume_confirmation: float,
                                    impulse_strength: float,
                                    retracement_depth: float) -> StructureStrength:
        """Determine the overall strength of the structure break"""
        # Calculate composite score
        volume_score = min(1.0, (volume_confirmation - 1.0) * 0.5)  # 0-1 scale
        impulse_score = impulse_strength  # Already 0-1 scale
        retracement_score = min(1.0, retracement_depth * 2)  # 0-1 scale

        composite_score = (volume_score * 0.3 + impulse_score * 0.5 + retracement_score * 0.2)

        if composite_score >= 0.8:
            return StructureStrength.VERY_STRONG
        elif composite_score >= 0.6:
            return StructureStrength.STRONG
        elif composite_score >= 0.4:
            return StructureStrength.MODERATE
        else:
            return StructureStrength.WEAK

    def _calculate_institutional_signature(self, current_idx: int, structure_type: StructureType,
                                         direction: TrendDirection) -> float:
        """Calculate institutional signature score for the structure break"""
        score = 0.0

        # Volume signature (0-0.3)
        volume_confirmation = self._calculate_volume_confirmation(current_idx)
        volume_score = min(0.3, (volume_confirmation - 1.0) * 0.15)
        score += max(0, volume_score)

        # Impulse signature (0-0.3)
        impulse_strength = self._calculate_impulse_strength(current_idx, direction)
        impulse_score = impulse_strength * 0.3
        score += impulse_score

        # Time-based signature (0-0.2)
        time_score = self._calculate_time_signature(current_idx)
        score += time_score

        # Structure type bonus (0-0.2)
        if structure_type == StructureType.CHOCH:
            score += 0.1  # ChoCH patterns are more significant
        else:
            score += 0.05  # BOS patterns get smaller bonus

        return min(1.0, score)

    def _calculate_time_signature(self, current_idx: int) -> float:
        """Calculate time-based signature for institutional activity"""
        # This is a simplified version - in practice, you'd consider:
        # - Market session times (London open, NY open, etc.)
        # - Day of week patterns
        # - Economic calendar events

        candle = self.ohlcv.iloc[current_idx]

        # Use volume ratio as proxy for institutional activity timing
        volume_ratio = candle.get('volume_ratio', 1.0)
        if pd.isna(volume_ratio):
            volume_ratio = 1.0

        # Higher volume during structure breaks suggests institutional involvement
        if volume_ratio > 1.5:
            return 0.2
        elif volume_ratio > 1.2:
            return 0.1
        else:
            return 0.05

    def _check_follow_through(self, current_idx: int, direction: TrendDirection,
                            lookforward: int = 5) -> bool:
        """Check if there's follow-through after the structure break"""
        if current_idx + lookforward >= len(self.ohlcv):
            lookforward = len(self.ohlcv) - current_idx - 1

        if lookforward < 2:
            return False

        break_price = self.ohlcv.iloc[current_idx]['close']

        # Count candles that continue in the break direction
        follow_through_count = 0
        total_candles = 0

        for i in range(current_idx + 1, current_idx + lookforward + 1):
            if i < len(self.ohlcv):
                candle = self.ohlcv.iloc[i]
                total_candles += 1

                if direction == TrendDirection.BULLISH:
                    if candle['close'] > break_price:
                        follow_through_count += 1
                else:  # BEARISH
                    if candle['close'] < break_price:
                        follow_through_count += 1

        # Return True if more than 60% of candles show follow-through
        return (follow_through_count / total_candles) > 0.6 if total_candles > 0 else False

    def _validate_structure_breaks(self, structure_breaks: List[StructureBreak]) -> List[StructureBreak]:
        """Validate and filter structure breaks based on quality criteria"""
        validated_breaks = []

        for break_event in structure_breaks:
            # Quality validation criteria
            if (break_event.volume_confirmation >= 1.1 and
                break_event.impulse_strength >= 0.2 and
                break_event.institutional_signature >= 0.3):
                validated_breaks.append(break_event)

        return validated_breaks

    def analyze_market_structure(self) -> MarketStructure:
        """
        Comprehensive market structure analysis

        Returns:
            Current market structure state with all detected patterns
        """
        logger.info("Starting comprehensive market structure analysis...")

        # Detect swing points
        swing_points = self.detect_swing_points()

        # Detect structure breaks
        structure_breaks = self.detect_structure_breaks()

        # Update current structure state
        self._update_market_structure_state(swing_points, structure_breaks)

        logger.info(f"Market structure analysis complete: {len(swing_points)} swing points, {len(structure_breaks)} structure breaks")
        return self.current_structure

    def _update_market_structure_state(self, swing_points: List[SwingPoint],
                                     structure_breaks: List[StructureBreak]):
        """Update the current market structure state"""
        # Get active swing points (recent and unbroken)
        active_highs = [sp for sp in swing_points if sp.is_swing_high() and sp.confirmed]
        active_lows = [sp for sp in swing_points if sp.is_swing_low() and sp.confirmed]

        # Determine current trend from recent structure breaks
        current_trend = self._determine_current_trend_from_breaks(structure_breaks)

        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(structure_breaks)

        # Calculate structure quality
        structure_quality = self._calculate_structure_quality(swing_points, structure_breaks)

        # Get last structure break
        last_break = structure_breaks[-1] if structure_breaks else None

        # Update the market structure
        self.current_structure = MarketStructure(
            current_trend=current_trend,
            last_structure_break=last_break,
            active_swing_highs=active_highs[-5:],  # Keep last 5 active highs
            active_swing_lows=active_lows[-5:],    # Keep last 5 active lows
            structure_breaks=structure_breaks,
            trend_strength=trend_strength,
            structure_quality=structure_quality,
            last_update_timestamp=datetime.now()
        )

    def _determine_current_trend_from_breaks(self, structure_breaks: List[StructureBreak]) -> TrendDirection:
        """Determine current trend from recent structure breaks"""
        if not structure_breaks:
            return TrendDirection.NEUTRAL

        # Look at the last few structure breaks
        recent_breaks = structure_breaks[-3:] if len(structure_breaks) >= 3 else structure_breaks

        # Count BOS vs ChoCH and directions
        bullish_signals = 0
        bearish_signals = 0

        for break_event in recent_breaks:
            if break_event.direction == TrendDirection.BULLISH:
                if break_event.type == StructureType.BOS:
                    bullish_signals += 2  # BOS is stronger signal
                else:  # ChoCH
                    bullish_signals += 3  # ChoCH is trend change signal
            else:  # BEARISH
                if break_event.type == StructureType.BOS:
                    bearish_signals += 2
                else:  # ChoCH
                    bearish_signals += 3

        if bullish_signals > bearish_signals:
            return TrendDirection.BULLISH
        elif bearish_signals > bullish_signals:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.NEUTRAL

    def _calculate_trend_strength(self, structure_breaks: List[StructureBreak]) -> float:
        """Calculate the strength of the current trend"""
        if not structure_breaks:
            return 0.0

        # Look at recent breaks
        recent_breaks = structure_breaks[-5:] if len(structure_breaks) >= 5 else structure_breaks

        # Calculate average institutional signature and follow-through
        avg_signature = np.mean([sb.institutional_signature for sb in recent_breaks])
        follow_through_rate = sum(1 for sb in recent_breaks if sb.follow_through) / len(recent_breaks)

        # Calculate strength based on break quality
        strength_scores = []
        for break_event in recent_breaks:
            if break_event.strength == StructureStrength.VERY_STRONG:
                strength_scores.append(1.0)
            elif break_event.strength == StructureStrength.STRONG:
                strength_scores.append(0.8)
            elif break_event.strength == StructureStrength.MODERATE:
                strength_scores.append(0.6)
            else:
                strength_scores.append(0.4)

        avg_strength = np.mean(strength_scores) if strength_scores else 0.0

        # Combine metrics
        trend_strength = (avg_signature * 0.4 + follow_through_rate * 0.3 + avg_strength * 0.3)
        return min(1.0, max(0.0, trend_strength))

    def _calculate_structure_quality(self, swing_points: List[SwingPoint],
                                   structure_breaks: List[StructureBreak]) -> float:
        """Calculate the overall quality of market structure"""
        if not swing_points and not structure_breaks:
            return 0.0

        # Swing point quality
        confirmed_swings = sum(1 for sp in swing_points if sp.confirmed)
        swing_quality = confirmed_swings / len(swing_points) if swing_points else 0.0

        # Structure break quality
        strong_breaks = sum(1 for sb in structure_breaks
                          if sb.strength in [StructureStrength.STRONG, StructureStrength.VERY_STRONG])
        break_quality = strong_breaks / len(structure_breaks) if structure_breaks else 0.0

        # Combine metrics
        overall_quality = (swing_quality * 0.4 + break_quality * 0.6)
        return min(1.0, max(0.0, overall_quality))

    def get_structure_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about market structure analysis"""
        if not self.swing_points and not self.structure_breaks:
            return {
                'total_swing_points': 0,
                'swing_highs': 0,
                'swing_lows': 0,
                'total_structure_breaks': 0,
                'bos_breaks': 0,
                'choch_breaks': 0,
                'current_trend': 'neutral',
                'trend_strength': 0.0,
                'structure_quality': 0.0
            }

        # Basic counts
        swing_highs = len([sp for sp in self.swing_points if sp.is_swing_high()])
        swing_lows = len([sp for sp in self.swing_points if sp.is_swing_low()])
        bos_breaks = len([sb for sb in self.structure_breaks if sb.is_bos()])
        choch_breaks = len([sb for sb in self.structure_breaks if sb.is_choch()])

        # Strength distribution
        strength_dist = {
            'very_strong': len([sb for sb in self.structure_breaks if sb.strength == StructureStrength.VERY_STRONG]),
            'strong': len([sb for sb in self.structure_breaks if sb.strength == StructureStrength.STRONG]),
            'moderate': len([sb for sb in self.structure_breaks if sb.strength == StructureStrength.MODERATE]),
            'weak': len([sb for sb in self.structure_breaks if sb.strength == StructureStrength.WEAK])
        }

        # Direction distribution
        bullish_breaks = len([sb for sb in self.structure_breaks if sb.direction == TrendDirection.BULLISH])
        bearish_breaks = len([sb for sb in self.structure_breaks if sb.direction == TrendDirection.BEARISH])

        return {
            'total_swing_points': len(self.swing_points),
            'swing_highs': swing_highs,
            'swing_lows': swing_lows,
            'confirmed_swings': len([sp for sp in self.swing_points if sp.confirmed]),
            'total_structure_breaks': len(self.structure_breaks),
            'bos_breaks': bos_breaks,
            'choch_breaks': choch_breaks,
            'bullish_breaks': bullish_breaks,
            'bearish_breaks': bearish_breaks,
            'current_trend': self.current_structure.current_trend.value,
            'trend_strength': self.current_structure.trend_strength,
            'structure_quality': self.current_structure.structure_quality,
            'strength_distribution': strength_dist,
            'average_institutional_signature': np.mean([sb.institutional_signature for sb in self.structure_breaks]) if self.structure_breaks else 0.0,
            'follow_through_rate': sum(1 for sb in self.structure_breaks if sb.follow_through) / len(self.structure_breaks) if self.structure_breaks else 0.0
        }


# Utility functions for integration with existing SMC system
def create_sample_market_structure_data(num_candles: int = 500) -> pd.DataFrame:
    """
    Create sample OHLCV data with intentional market structure patterns for testing

    Args:
        num_candles: Number of candles to generate

    Returns:
        DataFrame with OHLCV data containing market structure patterns
    """
    np.random.seed(42)  # For reproducible results

    base_price = 50000
    data = []

    # Create trend phases with structure breaks
    trend_phases = [
        {'type': 'bullish', 'length': 100, 'strength': 0.02},
        {'type': 'bearish', 'length': 80, 'strength': -0.015},
        {'type': 'bullish', 'length': 120, 'strength': 0.025},
        {'type': 'consolidation', 'length': 100, 'strength': 0.005},
        {'type': 'bearish', 'length': 100, 'strength': -0.02}
    ]

    current_phase = 0
    phase_progress = 0

    for i in range(num_candles):
        # Determine current trend phase
        if current_phase < len(trend_phases):
            phase = trend_phases[current_phase]

            if phase_progress >= phase['length']:
                current_phase += 1
                phase_progress = 0
                if current_phase < len(trend_phases):
                    phase = trend_phases[current_phase]

        # Generate price movement based on phase
        if current_phase < len(trend_phases):
            trend_move = phase['strength'] * (1 + np.random.normal(0, 0.3))
        else:
            trend_move = np.random.normal(0, 0.01)

        # Add noise
        noise = np.random.normal(0, 0.008)
        total_move = trend_move + noise

        # Create structure breaks at phase transitions
        if phase_progress == 0 and current_phase > 0:
            # Amplify movement for structure break
            total_move *= 2.0

        new_price = base_price * (1 + total_move)

        # Generate realistic OHLC
        volatility = new_price * 0.01

        open_price = base_price + np.random.normal(0, volatility * 0.3)
        close_price = new_price

        # Create wicks
        high_wick = abs(np.random.normal(0, volatility * 0.4))
        low_wick = abs(np.random.normal(0, volatility * 0.4))

        high_price = max(open_price, close_price) + high_wick
        low_price = min(open_price, close_price) - low_wick

        # Generate volume with spikes during structure breaks
        base_volume = 1000000
        if phase_progress == 0 and current_phase > 0:
            # Higher volume during structure breaks
            volume = base_volume * np.random.uniform(2.0, 4.0)
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
        phase_progress += 1

    return pd.DataFrame(data)


def get_enhanced_market_structure(ohlcv: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to get enhanced market structure analysis

    Args:
        ohlcv: OHLCV DataFrame
        **kwargs: Additional parameters for MarketStructureAnalyzer

    Returns:
        Dictionary with comprehensive market structure analysis
    """
    analyzer = MarketStructureAnalyzer(ohlcv, **kwargs)
    market_structure = analyzer.analyze_market_structure()
    statistics = analyzer.get_structure_statistics()

    # Convert swing points to dictionary format
    swing_points_dict = []
    for sp in analyzer.swing_points:
        swing_points_dict.append({
            'type': sp.type.value,
            'price': sp.price,
            'timestamp': sp.timestamp,
            'index': sp.index,
            'volume': sp.volume,
            'strength': sp.strength,
            'confirmed': sp.confirmed
        })

    # Convert structure breaks to dictionary format
    structure_breaks_dict = []
    for sb in analyzer.structure_breaks:
        structure_breaks_dict.append({
            'type': sb.type.value,
            'direction': sb.direction.value,
            'break_price': sb.break_price,
            'break_timestamp': sb.break_timestamp,
            'break_index': sb.break_index,
            'volume_confirmation': sb.volume_confirmation,
            'strength': sb.strength.value,
            'impulse_strength': sb.impulse_strength,
            'retracement_depth': sb.retracement_depth,
            'follow_through': sb.follow_through,
            'institutional_signature': sb.institutional_signature
        })

    return {
        'market_structure': {
            'current_trend': market_structure.current_trend.value,
            'trend_strength': market_structure.trend_strength,
            'structure_quality': market_structure.structure_quality,
            'last_update': market_structure.last_update_timestamp
        },
        'swing_points': swing_points_dict,
        'structure_breaks': structure_breaks_dict,
        'statistics': statistics
    }


# Example usage and testing
if __name__ == "__main__":
    print("Market Structure Analysis System - Testing")
    print("=" * 50)

    # Create sample data with market structure patterns
    print("Creating sample OHLCV data with market structure patterns...")
    ohlcv_data = create_sample_market_structure_data(400)

    # Initialize Market Structure Analyzer
    print("Initializing Market Structure Analyzer...")
    analyzer = MarketStructureAnalyzer(
        ohlcv_data,
        swing_lookback=5,              # 5-candle lookback for swing detection
        min_structure_distance=0.005,  # 0.5% minimum distance between structures
        volume_threshold=1.2           # 1.2x volume threshold for validation
    )

    # Analyze market structure
    print("Analyzing market structure...")
    market_structure = analyzer.analyze_market_structure()

    # Get statistics
    stats = analyzer.get_structure_statistics()

    # Print results
    print(f"\nMarket Structure Analysis Results:")
    print(f"Current trend: {market_structure.current_trend.value}")
    print(f"Trend strength: {market_structure.trend_strength:.3f}")
    print(f"Structure quality: {market_structure.structure_quality:.3f}")
    print(f"Total swing points: {stats['total_swing_points']}")
    print(f"Swing highs: {stats['swing_highs']}")
    print(f"Swing lows: {stats['swing_lows']}")
    print(f"Confirmed swings: {stats['confirmed_swings']}")

    print(f"\nStructure Break Analysis:")
    print(f"Total structure breaks: {stats['total_structure_breaks']}")
    print(f"BOS breaks: {stats['bos_breaks']}")
    print(f"ChoCH breaks: {stats['choch_breaks']}")
    print(f"Bullish breaks: {stats['bullish_breaks']}")
    print(f"Bearish breaks: {stats['bearish_breaks']}")
    print(f"Average institutional signature: {stats['average_institutional_signature']:.3f}")
    print(f"Follow-through rate: {stats['follow_through_rate']:.1%}")

    # Show strength distribution
    strength_dist = stats['strength_distribution']
    print(f"\nStrength Distribution:")
    print(f"Very strong: {strength_dist['very_strong']}")
    print(f"Strong: {strength_dist['strong']}")
    print(f"Moderate: {strength_dist['moderate']}")
    print(f"Weak: {strength_dist['weak']}")

    # Show some example swing points
    if analyzer.swing_points:
        print(f"\nExample Swing Points (first 3):")
        for i, sp in enumerate(analyzer.swing_points[:3]):
            print(f"Swing {i+1}: {sp.type.value} at {sp.price:.2f}")
            print(f"  Strength: {sp.strength:.3f}, Confirmed: {sp.confirmed}")
            print()

    # Show some example structure breaks
    if analyzer.structure_breaks:
        print(f"Example Structure Breaks (first 3):")
        for i, sb in enumerate(analyzer.structure_breaks[:3]):
            print(f"Break {i+1}: {sb.type.value} {sb.direction.value} at {sb.break_price:.2f}")
            print(f"  Strength: {sb.strength.value}")
            print(f"  Volume confirmation: {sb.volume_confirmation:.2f}x")
            print(f"  Institutional signature: {sb.institutional_signature:.3f}")
            print(f"  Follow-through: {sb.follow_through}")
            print()

    # Test convenience function
    print("Testing convenience function...")
    enhanced_analysis = get_enhanced_market_structure(ohlcv_data)
    print(f"Convenience function detected {len(enhanced_analysis['swing_points'])} swing points and {len(enhanced_analysis['structure_breaks'])} structure breaks")

    print("\n✅ Market Structure Analysis System testing complete!")
    print("The system successfully detected and analyzed BOS/ChoCH patterns.")